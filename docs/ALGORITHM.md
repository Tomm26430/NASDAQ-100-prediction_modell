# Prediction Algorithm (Deep Dive)

This document explains how the NASDAQ Predictor generates forecasts, **from first principles**, using the **same logic and parameters implemented in the codebase**.

Relevant code:
- `backend/services/ensemble.py`
- `backend/services/lstm_model.py`
- `backend/services/arima_model.py`
- `backend/services/indicators.py`
- `backend/config.py`

---

## 1) Why two models instead of one?

Financial time series contain at least two very different kinds of patterns:

- **Regular, “physics-like” patterns**: momentum, mean-reversion, volatility clustering, and repeating shapes that can sometimes be inferred from recent history.
- **Irregular, “event-like” jumps**: earnings surprises, macro news, geopolitical events, product launches, etc. These often look like discontinuities to a model because the cause is not in the price/volume history.

This project uses **two complementary model families** because they make different trade-offs:

- **LSTM (neural network)** is good at learning **nonlinear relationships** across multiple inputs (returns, volume changes, indicators). It can learn “if-this-then-that” patterns that are hard to capture with simple linear models.
- **ARIMA (statistical time series)** is good at modeling **smooth, short-term autocorrelation** in a single series. It tends to behave sensibly when the recent trend continues, but it is not designed to incorporate multiple engineered features.

The goal of the ensemble is not to claim “truth,” but to **reduce model-specific failure modes**:
- If the LSTM overfits or becomes unstable, ARIMA can keep predictions grounded.
- If ARIMA lags during regime change, LSTM can react using feature context (e.g., volatility spikes reflected in indicators/returns).

In code, the ensemble happens in `backend/services/ensemble.py` via a simple weighted average.

---

## 2) How the LSTM works in this project (specifically)

### 2.1 Input window length (60 trading days)

The LSTM uses a rolling window of length `SEQUENCE_LENGTH` (default **60**) from `backend/config.py`.

Each training example is a **60-day sequence** of engineered features computed from OHLCV + indicators.

### 2.2 Features used (and why)

The LSTM does **not** ingest raw “Close” and “Volume” directly. Instead it uses **scale-free**, return-based features so the network does not need to learn absolute price levels.

The exact feature columns fed to the LSTM are defined in `lstm_model.lstm_feature_columns()`:

- **`ret_close`**: daily percentage change of close (close-to-close return).
  - Why: captures short-term momentum/reversal without encoding the absolute price level.
- **`ret_volume`**: daily percentage change of volume (clipped to avoid extreme outliers).
  - Why: volume often signals conviction; spikes can precede breakouts or reversals.
- **`rsi_14`**: RSI scaled to 0–1.
  - Why: compresses “overbought/oversold” into a bounded feature.
- **`macd_rel`** and **`macd_signal_rel`**: MACD and its signal line divided by the current close.
  - Why: MACD is in “price units”; dividing by close makes it comparable across tickers.
- **`bb_upper_rel`** and **`bb_lower_rel`**: (upper band − close)/close and (lower band − close)/close.
  - Why: expresses “distance to the band” as a percentage, which is more stable than raw price distances.

Indicators are computed by `backend/services/indicators.py`:
- RSI(14)
- MACD(12, 26, 9)
- Bollinger Bands(20, 2σ)

### 2.3 Model architecture (2× LSTM + Dropout + Dense)

The architecture is defined in `train_lstm_for_ticker()` in `backend/services/lstm_model.py`:

- Input shape: `(60, n_features)` where `n_features = 7`
- Layer 1: `LSTM(LSTM_UNITS=100, return_sequences=True)`
- Dropout: `Dropout(LSTM_DROPOUT=0.25)`
- Layer 2: `LSTM(LSTM_UNITS=100)`
- Dropout: `Dropout(LSTM_DROPOUT=0.25)`
- Output: `Dense(4)` (four horizons)

The output vector is ordered as **[+1, +7, +30, +90]** (trading days) and is interpreted as **returns**, not prices.

### 2.4 What it learns to predict: percentage returns at multiple horizons

This is the central design choice that avoids “upward drift” when forecasting multiple steps.

For each training window ending at time \(t\), the training targets are:

\[
  r_h = \left(\frac{\text{close}[t+h]}{\text{close}[t]}\right) - 1
\]

for \(h \in \{1, 7, 30, 90\}\).

This is implemented in `_build_xy()`:
- `ref = close[t]` where \(t = i + seq\_len - 1\)
- `targets = [(close[t+h] / ref - 1.0) for h in (1,7,30,90)]`

At inference time, the model returns predicted returns, and the project converts them back into prices:

\[
  \widehat{\text{price}}_{t+h} = \text{anchor\_close} \cdot (1 + \widehat{r}_h)
\]

That conversion is applied in:
- `predict_lstm_horizons()` (for +7/+30/+90 price heads)
- `predict_lstm_one_step_with_model()` (for +1 used inside rollouts)
- `predict_lstm_head_prices_with_model()` (teacher-forced “heads”)

**What “anchor_close” means**
- It is **the last close in the input window** (the last known close before the forecast horizon begins).
- In honest multi-step rollouts (backtests), future windows contain **synthetic** closes produced by earlier predictions, so the anchor close shifts forward along the synthetic path.

### 2.5 Feature scaling (“normalization”) and why it matters

Neural networks train more reliably when inputs are in a comparable numeric range.

This project uses **`sklearn.preprocessing.MinMaxScaler`**:
- It is fit on **training windows only** (to avoid leaking the validation set into the scaling statistics).
- Then it is used to transform both training and validation windows.
- The fitted scaler is saved into model metadata (`fx`) so inference uses **the same scale** as training.

Implementation details:
- Fit happens in `train_lstm_for_ticker()`:
  - Windows are flattened into 2D: `(samples * seq_len, n_features)`
  - `fx.fit_transform(...)` for training, `fx.transform(...)` for validation
- Inference scaling happens in `_scaled_window_tensor()` using `fx.transform(...)`.

**Note on “rolling normalization”**
- The current pipeline uses MinMax scaling (train-split only) rather than a per-window rolling z-score.
- The setting `LSTM_ROLLING_NORM_WINDOW` remains in `config.py` for environment compatibility but is not used by the current LSTM feature pipeline.

### 2.6 Training process (epochs, early stopping, validation split)

Training steps in `train_lstm_for_ticker()`:

1. Load OHLCV from SQLite: `services.data_fetcher.get_ohlcv_dataframe()`
2. Add indicators: `services.indicators.add_indicators(df).ffill().fillna(0.0)`
3. Build windows and targets: `_build_xy(...)`
4. Split into train/validation (default 85% / 15%, with fallback to 80/20 if too small)
5. Fit scaler on training only; transform both splits
6. Train using:
   - Optimizer: Adam with `LSTM_LEARNING_RATE` (default **0.0005**)
   - Loss: Mean Squared Error (`"mse"`)
   - Early stopping:
     - monitor `val_loss`
     - patience `LSTM_EARLY_STOPPING_PATIENCE` (default **10**)
     - restore best weights
   - Maximum epochs: `LSTM_EPOCHS` (default **75**)

### 2.7 What “learning” means at the weight level (simple explanation)

An LSTM layer contains many numbers called **weights**. You can think of them as knobs that control:
- how strongly the network reacts to each input feature,
- how much it “remembers” from earlier days in the 60-day window,
- and how it converts that internal memory into predicted returns.

During training:
- The model makes a prediction for each training window.
- It compares that prediction to the true return targets and computes an error (loss).
- A method called **gradient descent** (via Adam optimizer) slightly adjusts the weights to reduce that error.
- Over many batches, these tiny adjustments accumulate into a set of weights that tend to produce lower error on average.

The validation split is a safeguard: if validation loss stops improving, **early stopping** halts training to limit overfitting.

---

## 3) How ARIMA works in this project

### 3.1 What `auto_arima` does automatically

ARIMA models are defined by orders \((p, d, q)\):
- \(p\): how many past values the model uses (“autoregressive” terms),
- \(d\): how many times the series is differenced to remove trends (“integration”),
- \(q\): how many past forecast errors are used (“moving average” terms).

Choosing \((p, d, q)\) manually is tricky, so the project uses `pmdarima.auto_arima` which:
- tries different configurations within bounds (`max_p`, `max_q`, `max_d`),
- selects a good candidate using information criteria and heuristics,
- and returns a fitted ARIMA-like estimator.

This is done in:
- `backend/services/arima_model.py` (`train_arima_for_ticker`, `forecast_arima_with_intervals`, `arima_walk_one_step`)
- and a small local helper in `backend/services/backtester.py` (`_arima_price_h_ahead`) used for multi-step backtests without changing `arima_model.py`.

### 3.2 Why log prices are used instead of raw prices

The ARIMA fit uses:

\[
  y_t = \log(\max(\text{close}_t, 10^{-6}))
\]

Reasons:
- Prices are positive; log space avoids negative predictions.
- Many multiplicative effects become additive in log space (a 1% move is similar regardless of price level).
- Variance tends to be more stable after a log transform.

After forecasting in log space, predictions are converted back with `exp(...)` to produce prices.

### 3.3 Walk-forward updating (refitting) explained simply

The backtester uses walk-forward logic so the model is always trained only on “past” data:

- Train on a prefix (the training segment).
- Predict the next point.
- Reveal the true next value (from the holdout).
- Update the model with that true value and move forward.

In code, this is `arima_walk_one_step()` in `backend/services/arima_model.py`:
- It fits once on the training segment (log prices),
- then iterates across test points:
  - predict one step,
  - record prediction,
  - update with the realized value.

### 3.4 Why ARIMA is good at short-term trends but not surprises

ARIMA is essentially a structured, linear model of the past:
- It can extrapolate smooth trends and autocorrelation.
- It cannot “know” about outside events unless they already affected past prices.

So ARIMA often behaves well when:
- the market regime is stable,
- volatility is consistent,
- and recent patterns continue.

It struggles when:
- there is a sudden jump or crash,
- volatility changes abruptly,
- or the series changes regime (new trend, new mean, new variance).

---

## 4) How the ensemble combines LSTM + ARIMA

The ensemble calculation is implemented in `backend/services/ensemble.py`.

### 4.1 Weighting (60% LSTM + 40% ARIMA)

For each horizon \(h\) in {7, 30, 90}:

\[
  \widehat{P}^{ens}_h = w_{lstm}\cdot \widehat{P}^{lstm}_h + w_{arima}\cdot \widehat{P}^{arima}_h
\]

By default:
- `ENSEMBLE_WEIGHT_LSTM = 0.6`
- `ENSEMBLE_WEIGHT_ARIMA = 0.4`

The code normalizes these weights so they sum to 1, even if you override them.

**Why these weights?**
- The LSTM uses multiple features and can capture nonlinear patterns.
- ARIMA provides a strong “statistical baseline” that can reduce wild LSTM outputs.
- A mild LSTM preference (60/40) reflects the idea that feature-rich signals should matter, but not dominate completely.

### 4.2 Confidence bands (what they are and how they’re computed)

The app returns two kinds of intervals:

1. **ARIMA’s own 95% interval** (from `forecast_arima_with_intervals`), returned as:
   - `arima_ci_low`, `arima_ci_high`
2. A **heuristic ensemble band** returned as:
   - `ci_low`, `ci_high`

The heuristic band is:

- `spread = abs(lstm_price - arima_price) + 0.015 * max(last_close, 1.0)`
- `ci_low = max(0, ensemble - spread)`
- `ci_high = ensemble + spread`

Interpretation:
- If LSTM and ARIMA disagree strongly, the band widens.
- A small additional term based on last close adds a baseline uncertainty.

This band is not a statistically calibrated probability interval; it is a practical “uncertainty hint.”

### 4.3 Why an ensemble can beat single models (statistical intuition)

If two models make different errors, averaging them can reduce error variance:
- One model might overshoot while the other undershoots.
- Weighted averaging can partially cancel those errors.

This helps most when:
- models are somewhat independent (different assumptions, different failure modes),
- and neither model is consistently worse across all regimes.

---

## 5) Technical indicators (features) explained

All indicator calculations are in `backend/services/indicators.py`.

### 5.1 RSI (Relative Strength Index, 14)

**What it measures**
- The balance of recent gains vs recent losses over a window (14 days).
- Often described as “overbought” (high RSI) vs “oversold” (low RSI), but it’s more accurate to think of it as **recent momentum strength**.

**Formula (plain English)**
1. Compute daily price changes.
2. Separate them into gains (positive) and losses (negative).
3. Compute average gain and average loss over 14 days.
4. Compute \(RS = \frac{\text{avg gain}}{\text{avg loss}}\).
5. Convert to RSI: \(RSI = 100 - \frac{100}{1 + RS}\).

**Why it helps the LSTM**
- It summarizes momentum in a bounded range, which is easy for a neural network to use.
- It can highlight regimes where the price has been trending strongly.

### 5.2 MACD (12, 26, 9)

**What it measures**
- The difference between a fast and slow exponential moving average (EMA) of price.
- A signal line is another EMA of that difference.

**How it’s calculated**
- `ema12 = EMA(close, span=12)`
- `ema26 = EMA(close, span=26)`
- `macd = ema12 - ema26`
- `macd_signal = EMA(macd, span=9)`
- `macd_hist = macd - macd_signal`

**What “crossover” signals mean (intuition)**
- If MACD rises above the signal line, it suggests strengthening upward momentum.
- If MACD falls below the signal line, it suggests weakening momentum.

In this project, MACD features are used in relative form (`macd_rel`, `macd_signal_rel`) to avoid dependence on absolute price units.

### 5.3 Bollinger Bands (20, 2σ)

**What they represent**
- A moving average (“middle band”) plus/minus a multiple of recent volatility.

**How they’re calculated**
- `bb_middle = SMA(close, 20)`
- `std = rolling_std(close, 20)`
- `bb_upper = bb_middle + 2*std`
- `bb_lower = bb_middle - 2*std`

**What “breakouts” mean (intuition)**
- When price touches/exceeds the upper band, it can indicate strong upward momentum or high volatility.
- When it touches/exceeds the lower band, it can indicate strong downward momentum or high volatility.

In this project, the LSTM uses distance-to-band features (`bb_upper_rel`, `bb_lower_rel`) as percentages.

---

## 6) Known limitations (honest and precise)

### 6.1 News-driven moves are not predictable from price history alone

This system uses OHLCV and technical indicators derived from price/volume. It does not ingest:
- news text,
- earnings calendars,
- macro announcements,
- or order-book data.

So sudden event-driven moves will look like “unexplained shocks.”

### 6.2 Multi-step rollout error compounds over longer horizons

In honest rollouts (used in Scenario 2/3/5 backtests), the model feeds **its own previous predictions** back into the next step.

If it makes a small error early:
- the next input window includes that error,
- which can slightly bias the next prediction,
- and errors can accumulate.

Predicting returns instead of absolute prices reduces long-horizon drift, but it does not remove compounding uncertainty.

### 6.3 Direction accuracy baselines and expectations

Direction accuracy asks “did we predict up vs down correctly?”
- A naive baseline is **50%** (random guessing).
- Realistic improvements over 50% can be small and regime-dependent.

If a direction metric is around 50–55%, it may be slightly informative but not necessarily tradable once costs/slippage are considered.

### 6.4 Why ARIMA can look artificially good in Scenario 1

Scenario 1 backtests use a one-step walk-forward setup where the system predicts the next day and then immediately updates using the realized next day.

This setting is inherently easier than a forward-looking multi-step forecast because:
- each step only requires predicting one day ahead,
- and the model gets fresh “real” data at every step.

That is why Scenario 2 (honest multi-step rollout) is emphasized for forward-looking realism.

