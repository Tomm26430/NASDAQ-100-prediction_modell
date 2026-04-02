# Backtesting Guide (Scenarios 1–5)

This guide explains what backtesting is, then documents the **exact five scenarios** implemented in `backend/services/backtester.py`, including how to interpret metrics and how the holdout `years` parameter works.

Relevant code:
- `backend/services/backtester.py`
- `backend/services/lstm_model.py`
- `backend/services/arima_model.py`
- `backend/services/ensemble.py`
- `backend/services/indicators.py` (`merge_macro_features`)
- `backend/services/data_fetcher.py` (`get_macro_dataframe`)
- `backend/config.py`
- API route: `backend/routers/stocks.py` (`GET /api/stocks/{ticker}/backtest`)
- Bulk route: `backend/routers/admin.py` (`POST /api/admin/backtest-all`, `GET /api/admin/backtest-status`)

---

## 1) What is backtesting?

**Backtesting** is a way to measure how a strategy/model would have performed on **past data** under a set of rules that try to mimic “what you would have known at the time.”

In this project, the core assumption is:
- We have a historical series of daily closes.
- We pretend we are standing at some point in the past (“an anchor date”).
- We forecast forward from that point using only data available up to the anchor.
- Then we compare forecasts to the actual prices that happened after.

Why it’s useful:
- It provides a consistent, repeatable way to compare model changes.
- It reveals failure modes (drift, instability, regime sensitivity).

Why it’s not a guarantee:
- Market regimes change.
- Transaction costs and execution constraints are ignored here.
- Backtests can be optimistic if they accidentally use information from the future (data leakage). This project tries to avoid that by training LSTM only on pre-holdout rows and using honest multi-step rollouts in the key scenarios.

---

## 2) The 5 scenarios (what they measure and how they work)

All scenarios share the same high-level skeleton in `run_backtest()`:

1. Load cached OHLCV via `get_ohlcv_dataframe(...)`.
2. When **`USE_MACRO_FEATURES`** is true, load **`macro_daily`** via `get_macro_dataframe(...)` once and pass it into scenario helpers. Macro values are merged onto equity dates with **no lookahead**: rows after the last bar of the current frame (including synthetic rollout bars) are not used; missing macro dates are forward-filled from the last known value, then rolling-scaled (see `merge_macro_features`). **ARIMA** paths are unchanged (log closes only).
3. Slice a holdout window using `_holdout_slice(...)`.
4. Compute ARIMA one-step walk-forward predictions once using `arima_walk_one_step(...)` (used directly in scenarios 1 & 4; referenced in 2/3).
5. For each scenario:
   - Train a **temporary** LSTM on the pre-holdout segment (`train_lstm_for_ticker(..., model_root=tempdir)`).
   - Evaluate using that LSTM + ARIMA according to the scenario’s rules.
6. Return a JSON response with scenario label, holdout notes, metrics, and (usually) a chart series.

Set **`USE_MACRO_FEATURES=false`** in config / `.env` to reproduce the older **7-feature** LSTM-only price+indicator behavior (still requires compatible checkpoints under `saved_models/`).

### Scenario 1 — Daily prediction accuracy (optimistic)

**What it measures**
- One-day-ahead performance when the model is allowed to consume **real daily history** at each step.

**How it works technically**
- Implemented in `_scenario_1_daily(...)`.
- LSTM:
  - Trained once on rows before the holdout.
  - For each day in the holdout, it predicts **next-day close** using `predict_lstm_one_step_with_model(...)` on the indicator-enriched frame up to that day.
- ARIMA:
  - Uses `arima_walk_one_step(...)` which predicts one day ahead and then updates with the realized value each day.
- Ensemble series:
  - Dates where both LSTM and ARIMA have predictions are aligned and blended:
    - `blend = wl * lstm_pred + wa * arima_pred`

**When to use it**
- Quick sanity checks that “the model runs” and produces plausible short-term outputs.

**How to interpret results**
- Because this uses real inputs daily, it is **not a strict forward-looking simulation**. Think of it as “best-case daily tracking.”

**Good / bad numbers**
- MAPE on 1-day horizon can be small (often a few percent) even for weak models because next-day moves are usually small.
- Use Scenario 2/5 for realistic long-horizon evaluation.

---

### Scenario 2 — Forward-looking prediction (no real data) (honest multi-step rollout)

**What it measures**
- The model’s ability to forecast forward when it must **use its own previous predictions** as inputs (no peeking at realized future closes).

**How it works technically**
- Implemented in `_scenario_2_multistep(...)` via `_scenario_2_price_metrics_and_series(...)`.
- For anchors inside the holdout window, spaced by `BACKTEST_SCENARIO2_ANCHOR_STRIDE` (default 10):
  1. Take the prefix up to the anchor day.
  2. Run a chained LSTM rollout through the maximum horizon using `_lstm_rollout_closes_at_steps(...)`:
     - each new synthetic day is appended by `_append_synthetic_close_row(...)`
     - indicators are recomputed on the growing synthetic frame by `_indicator_frame_for_rollout(...)`
     - next-day predictions use `predict_lstm_one_step_with_model(...)`
  3. For each horizon (7/30/90):
     - compute ARIMA h-step forecast from the anchor via `_arima_price_h_ahead(...)`
     - blend LSTM and ARIMA
     - compare to the realized close at anchor+h
- A chart series is produced for one “primary” horizon, `BACKTEST_MULTI_STEP_CHART_HORIZON` (default 30):
  - Each chart point is at (anchor date + chart_horizon) with `actual` and blended `predicted`.

**When to use it**
- When you care about true forward-looking behavior, especially for 30/90 days.

**How to interpret results**
- This is the most important “price accuracy” scenario because it reveals drift and compounding errors.

**Good / bad numbers**
- See the interpretation ranges in section 3.

---

### Scenario 3 — Performance during high volatility (stress test)

**What it measures**
- Same honest multi-step rollout as Scenario 2, but evaluated only during stress periods.

**How it works technically**
- Scenario 3 is Scenario 2 with filtering:
  - `stress_dates = _stress_trading_date_set(...)` includes:
    - Feb–May 2020 (COVID crash window)
    - all of 2022 (selloff window)
    - and any rolling 30-trading-day window where absolute move \(|close[e]/close[s] - 1|\) exceeds 15%
- `_scenario_2_price_metrics_and_series(...)` only keeps chart points whose end-date is in `stress_dates`.

**When to use it**
- To see whether the model becomes unreliable exactly when volatility is high (often the hardest regime).

**How to interpret results**
- Expect worse metrics than Scenario 2.
- If it returns no points, the response adds a note suggesting more history or date coverage.

**Good / bad numbers**
- “Good” here is relative; the important signal is whether changes improve stress behavior without breaking normal behavior.

---

### Scenario 4 — Did the model predict up/down correctly? (direction + daily series)

**What it measures**
- Daily blended price series accuracy (like Scenario 1) **plus** a direction hit-rate for 1/7/30-day horizons.

**How it works technically**
- Implemented in `_scenario_4_direction(...)`.
- It first computes the daily one-step LSTM series across the holdout (same idea as Scenario 1).
- It then computes direction accuracy using strided anchors to limit ARIMA workload:
  - Every `BACKTEST_SCENARIO4_DIRECTION_STRIDE` days (default 10), it:
    - Uses LSTM “head” prices from `predict_lstm_head_prices_with_model(...)` for +1/+7/+30.
    - Uses ARIMA `_arima_price_h_ahead(...)` for +1/+7/+30.
    - Blends them to get an ensemble forecast at each horizon.
    - Compares the sign of the move vs the anchor close:
      - `sign(actual - ref) == sign(pred - ref)`
    - This is done by `_direction_accuracy(...)`.

**When to use it**
- When you care about “directional skill” rather than exact price.

**How to interpret results**
- Direction is hard. Small improvements above 50% can still be meaningful statistically, but can also be noise.
- The response includes:
  - `direction_accuracy_1d`, `direction_accuracy_7d`, `direction_accuracy_30d`
  - `baseline = 0.5`
  - `direction_note` describing the definition

**Good / bad numbers**
- See the interpretation ranges in section 3.

---

### Scenario 5 — Combined Honest Assessment (single LSTM training pass)

**What it measures**
- A single combined view of:
  - Scenario-2-style honest price accuracy (MAPE/MAE/RMSE by horizon), and
  - Scenario-4-style direction hit rates,
  - plus an overall “verdict” label.

**How it works technically**
- Implemented in `_scenario_5_combined(...)`.
- Trains the LSTM once into a temporary directory.
- Using the same loaded model:
  - Computes price metrics/series using `_scenario_2_price_metrics_and_series(...)`.
  - Computes direction stats using `_scenario_4_direction_samples_only(...)` (direction sampling only, no daily series).
- It then computes:
  - `mape_30` from the ensemble `h30` metric block
  - `dir_7` from `direction_accuracy_7d`
- Verdict logic is exactly in `_combined_verdict_label_and_explanation(...)`:
  - **Strong model** if `mape_30 < 8.0` **and** `dir_7 > 0.57`
  - **Decent model** if `mape_30 < 15.0` **and** `dir_7 > 0.53`
  - Otherwise **Needs improvement**
- Returns:
  - `price_accuracy` block (scenario-2-style metrics and series)
  - `direction_accuracy` block
  - `combined_verdict` and `verdict_explanation`

**When to use it**
- As the default “honest summary” for a ticker.
- As the scenario used by bulk testing.

**How to interpret results**
- Treat the verdict as a quick triage label, not a promise.
- Read the price and direction numbers to understand why the verdict was assigned.

---

## 3) Interpretation guide (metrics and thresholds)

### 3.1 MAPE (Mean Absolute Percentage Error)

MAPE is reported as a **percent**:

\[
  \text{MAPE} = \text{mean}\left(\left|\frac{pred - actual}{actual}\right|\right)\times 100
\]

In code, `_metrics(...)` uses a small denominator floor to avoid dividing by zero.

Practical interpretation ranges (rule of thumb):
- **< 5%**: very tight tracking (rare in honest multi-step settings unless the series is extremely stable)
- **5% – 15%**: reasonable for rough forecasting (often “usable for learning/monitoring”)
- **> 15%**: poor price accuracy (typical when the model drifts or the series is volatile)

Remember:
- Scenario 1 will usually look better than Scenario 2/5.
- Scenario 3 will usually look worse than Scenario 2.

### 3.2 Direction accuracy

Direction accuracy is a fraction \(0\) to \(1\), where \(0.5\) is a random baseline:

- **< 50%**: worse than guessing; suggests systematic bias or unstable predictions
- **50% – 55%**: slight edge or noisy signal; often requires a large sample to trust
- **> 55%**: strong directional skill (hard to achieve consistently)

In this project, direction comparisons are made against the anchor close (“ref”).

### 3.3 Combined verdict logic (exact)

Scenario 5 verdict is computed in `_combined_verdict_label_and_explanation(...)` using:

- `mape_30`: ensemble horizon `h30` MAPE (percent)
- `dir_7`: `direction_accuracy_7d` (fraction)

Rules:
- **Strong model**: `mape_30 < 8.0` and `dir_7 > 0.57`
- **Decent model**: `mape_30 < 15.0` and `dir_7 > 0.53`
- Else: **Needs improvement**

If either metric is missing/non-finite, the verdict falls back to **Needs improvement**.

---

## 4) The `years` parameter (holdout depth)

### 4.1 API usage

Endpoint:
- `GET /api/stocks/{ticker}/backtest?scenario=5&years=5`
- `GET /api/stocks/{ticker}/backtest?scenario=5&years=3`

Rules implemented in `_holdout_slice(...)`:
- If `years` is provided:
  - It is clamped to **minimum 1.0**.
  - It is converted to trading days by `int(years * 252)`.
- If `years` is omitted:
  - It uses `BACKTEST_YEARS` from `config.py`.
- If `max_holdout=true`:
  - It ignores `years` and uses the largest holdout that still leaves enough pre-holdout rows for LSTM training and enough rows for windowing.
- In all cases, it is **silently capped** by available data:
  - the response includes `holdout_years_requested` and `holdout_years_actual`
  - and sometimes a `holdout_note` explaining reductions

### 4.2 UI usage

The Backtesting page exposes presets:
- `1y`, `3y`, `5y`, `10y`, `Max`

These map to:
- `years=<preset>` or `max_holdout=true`

### 4.3 Why shorter vs longer holdouts give different results

- **Short holdout** (1–3y):
  - Faster runs
  - Results are dominated by recent regime
  - More variance (fewer samples)
- **Long holdout** (5–10y):
  - Slower
  - Tests multiple regimes (bull, bear, sideways)
  - More stable average metrics

### 4.4 Recommended settings (practical)

- **Quick iteration**: 1y or 3y, Scenario 5
- **Reasonable evaluation**: 5y, Scenario 5
- **Deep robustness check**: 10y or Max, Scenario 5 (expect longer runtime)

---

## 5) Bulk test (Scenario 5 across active tickers)

### 5.1 What it tests

Bulk test runs Scenario 5 sequentially for every “active” ticker (respects `LIGHT_MODE`):
- `LIGHT_MODE=true`: runs on `["AAPL","MSFT","GOOGL","AMZN","^NDX"]`
- `LIGHT_MODE=false`: runs on all tracked NASDAQ-100 tickers + `^NDX`

### 5.2 How to trigger it

API:
- `POST /api/admin/backtest-all?scenario=5&years=5` — returns **immediately** with `status: accepted` (HTTP **409** if a job is already running).
- `POST /api/admin/backtest-all?scenario=5&max_holdout=true`
- Poll **`GET /api/admin/backtest-status`** until `state` is `completed` or `error`; the full bulk JSON (same shape as before) is in **`result`** when completed.

Web UI:
- The Backtesting page starts the job, polls automatically, and shows a **progress bar** (percent + current ticker).

Implementation:
- `backend/services/backtester.py::run_backtest_all(...)` (invoked from a FastAPI **BackgroundTasks** worker).
- `backend/services/backtest_status.py` holds progress for the status endpoint.
- Runs tickers **sequentially**.
- If one ticker fails, it records `status = "error: <message>"` and continues.
- Optional persistence to SQLite (`persist=true` by default) for history/compare pages (`saved_run_id` in `result` when saving succeeds).

### 5.3 How to read the results

Response contains:
- `results[ticker]` with:
  - `combined_verdict`
  - `mape_30d`
  - `direction_accuracy_7d`
  - `holdout_days`
  - `status` ("ok" or "error: …")
- `aggregate` summary:
  - `avg_mape_30d`, `avg_direction_accuracy_7d`
  - best/worst by direction
  - lists of tickers above 50% direction
  - lists categorized as strong/decent/needs improvement (same thresholds as Scenario 5 verdicts)

### 5.4 Why multiple tickers is more reliable than one

A single ticker can look “great” or “terrible” due to:
- idiosyncratic news,
- sector rotation,
- one unusual period in the holdout.

Testing across many tickers gives a better signal of whether a model change improves the system in general, not just for one symbol.

