# Code Reference (Developer-Facing)

This document maps the codebase structure and explains the key files in depth so you can safely extend the project.

Scope note:
- This documentation only describes code that exists in this repository.
- Paths are relative to the `nasdaq-predictor/` folder.

---

## 1) Folder structure (every file, one sentence each)

### `backend/` — FastAPI + ML + SQLite

- `backend/main.py`: FastAPI app entry point (CORS, router mounting, scheduler, `/health`, `/api/index/ndx`).
- `backend/config.py`: Central settings (env-overridable) for model hyperparameters, backtest settings, and server behavior.

#### `backend/models/`
- `backend/models/__init__.py`: Package marker.
- `backend/models/database.py`: SQLAlchemy models (`PriceBar`, `AppMeta`, `MacroDaily`, `BacktestRun`, `BacktestTickerResult`) and session/engine helpers.

#### `backend/routers/`
- `backend/routers/__init__.py`: Router package marker.
- `backend/routers/stocks.py`: Public stock routes (list, history, indicators, prediction, backtest).
- `backend/routers/admin.py`: Admin routes (refresh + refresh status, training + training status, bulk backtest + backtest status).
- `backend/routers/backtest_runs.py`: Saved run list/detail/compare/delete (`/api/backtest-runs/...`).

#### `backend/services/`
- `backend/services/__init__.py`: Services package marker.
- `backend/services/data_fetcher.py`: Yahoo Finance downloads + SQLite caching + “load as DataFrame” helpers + **`fetch_macro_features` / `get_macro_dataframe`** for `macro_daily`.
- `backend/services/indicators.py`: RSI/MACD/Bollinger computations + **`merge_macro_features`** (rolling macro normalization) and **`macro_feature_columns`**.
- `backend/services/lstm_model.py`: LSTM feature engineering, training, saving/loading artifacts, inference helpers.
- `backend/services/arima_model.py`: Auto-ARIMA training and refit-at-prediction helpers, plus one-step walk-forward.
- `backend/services/ensemble.py`: Blends LSTM and ARIMA outputs and creates confidence bands.
- `backend/services/backtester.py`: Backtest engine implementing scenarios 1–5 plus bulk runner (macro-aware feature frames and rollouts).
- `backend/services/train_jobs.py`: Batch training loop over active tickers (LSTM then ARIMA).
- `backend/services/training_status.py`: Thread-safe in-memory training progress for UI polling.
- `backend/services/refresh_status.py`: In-memory price refresh job progress (`GET /api/admin/refresh-status`).
- `backend/services/backtest_status.py`: In-memory bulk backtest job progress (`GET /api/admin/backtest-status`).
- `backend/services/backtest_storage.py`: Persist/load/compare saved backtest runs in SQLite.

#### `backend/utils/`
- `backend/utils/__init__.py`: Utils package marker.
- `backend/utils/nasdaq100_tickers.py`: Tracked universe list and “active tickers” selection (LIGHT_MODE-aware).
- `backend/utils/ticker_validate.py`: Validates a ticker is in the tracked universe (404 otherwise).
- `backend/utils/ml_paths.py`: Per-ticker model directory names (`artifact_stem`, e.g. `^NDX` → `NDX`) and one-time `migrate_flat_to_subfolders`.

### `frontend/` — React + Vite UI

- `frontend/vite.config.ts`: Dev server config and `/api` proxy to FastAPI.
- `frontend/src/main.tsx`: React bootstrap.
- `frontend/src/App.tsx`: Routes and layout shell.
- `frontend/src/vite-env.d.ts`: Vite typings.

#### `frontend/src/api/`
- `frontend/src/api/client.ts`: Axios client and typed wrappers for API endpoints.

#### `frontend/src/pages/`
- `frontend/src/pages/Dashboard.tsx`: Stock list + refresh/train actions + refresh + training progress bars.
- `frontend/src/pages/StockDetail.tsx`: History/indicators charts and ensemble forecast cards for one ticker.
- `frontend/src/pages/Backtesting.tsx`: Scenario runner, holdout selector, bulk backtest (poll + progress), links to saved runs.
- `frontend/src/pages/BacktestHistory.tsx`, `BacktestCompare.tsx`, `BacktestRunDetail.tsx`, `BacktestTickerDetail.tsx`: Saved-run UX.

#### `frontend/src/components/`
- `frontend/src/components/Navbar.tsx`: Simple navigation links.
- `frontend/src/components/StockChart.tsx`: Close-price line chart (Recharts).
- `frontend/src/components/IndicatorChart.tsx`: RSI/Bollinger and MACD charts (Recharts).
- `frontend/src/components/PredictionCard.tsx`: Horizon card showing ensemble price and CI.
- `frontend/src/components/TrainingProgress.tsx`: Training status progress bar (Dashboard).
- `frontend/src/components/RefreshProgress.tsx`: Price refresh progress bar (Dashboard).
- `frontend/src/components/BacktestProgress.tsx`: Bulk backtest progress bar (Backtesting page).

---

## 2) Key files in depth

### 2.1 `backend/services/lstm_model.py`

This module defines the LSTM feature pipeline, training loop, artifact format, and inference helpers.

#### Public / frequently used functions

- `lstm_feature_columns() -> list[str]`
  - **Returns** the exact ordered feature names used as LSTM input (**7** base columns, plus **5** macro names when `USE_MACRO_FEATURES` is true → **12** total).
  - **Used by** metadata saving and feature extraction.

- `log_lstm_feature_summary() -> None`
  - Logs `LSTM input features (N): [...]` once at app startup (`main.py` lifespan).

- `min_ohlcv_rows_for_lstm_window(seq_len: int) -> int`
  - **Returns** minimum required rows to build one LSTM input window.
  - In this codebase it is simply `seq_len` (no rolling normalization warmup).

- `train_lstm_for_ticker(session, ticker, train_end_exclusive=None, model_root=None) -> dict`
  - **Input**: a DB session, ticker, optional cutoff index for walk-forward backtests, optional output folder.
  - **Does**:
    - loads OHLCV from SQLite (`get_ohlcv_dataframe`)
    - adds indicators (`add_indicators`)
    - optionally merges macro features (`get_macro_dataframe` + `merge_macro_features`) when `USE_MACRO_FEATURES` is enabled
    - builds (X, y) windows where y is cumulative simple returns for horizons (1,7,30,90)
    - splits windows into train/validation
    - fits `MinMaxScaler` on training windows only
    - trains 2-layer LSTM model and saves:
      - `lstm.keras` and `lstm_meta.joblib` under `MODEL_DIR/<ticker_dir>/` (Keras model + scaler / meta)
  - **Output**: a small status dict including sample count.

- `load_trained_lstm_bundle(ticker, model_root=None)`
  - **Loads** the Keras model and joblib meta. Calls `_load_keras_and_meta`.
  - **Guards** against incompatible artifacts with `meta_version` (current code expects **≥ 4** for macro-capable checkpoints) and missing `fx`.

- `predict_lstm_horizons(session, ticker, model_root=None) -> dict[str, float]`
  - **Returns** price forecasts for `'7'`, `'30'`, `'90'` (not `'1'`) for live API usage.
  - **Mechanics**:
    - builds last window
    - model predicts returns
    - converts to prices using `anchor_close * (1 + return)`

#### Backtester-specific inference helpers

- `predict_lstm_one_step_with_model(model, meta, feat_df) -> float`
  - **Input**: already-loaded model/meta and an indicator-enriched DataFrame up to the current day.
  - **Output**: next-day predicted close (price), computed from the predicted +1 return and anchor close.
  - **Used by** honest multi-step rollouts in `backtester.py`.

- `predict_lstm_head_prices_with_model(model, meta, feat_df) -> tuple[float,float,float,float]`
  - **Output**: (+1, +7, +30, +90) price heads from a single forward pass (teacher-forced inputs).
  - **Used by** direction accuracy sampling (Scenario 4/5) in `backtester.py`.

#### Internal helpers (important for extensions)

- `_lstm_raw_features(feat: pd.DataFrame) -> pd.DataFrame`
  - Builds scale-free per-row features:
    - `ret_close`, `ret_volume`, `rsi_14/100`, and relative MACD/Bollinger distances.
    - When macros are enabled, appends the five macro columns from `feat` (already rolling-scaled in `merge_macro_features`).

- `_lstm_feature_matrix(feat: pd.DataFrame) -> pd.DataFrame`
  - Applies forward-fill and selects the final ordered feature columns.

- `_build_xy(feat_df, seq_len) -> (X, y)`
  - Slides 60-day windows and builds return targets `(close[t+h]/close[t]) - 1`.

- `_scaled_window_tensor(feat_df, meta) -> (window_tensor, anchor_close) | None`
  - Applies the saved `MinMaxScaler` to the latest window and returns anchor close.

---

### 2.2 `backend/services/backtester.py`

This module runs walk-forward evaluations (“backtests”) on a holdout segment.

#### Key public functions

- `run_backtest(session, ticker, scenario=1, years=None, max_holdout=False, light=False) -> dict`
  - Central entry point for scenarios 1–5.
  - Loads optional **`macro_df`** via `get_macro_dataframe` when `USE_MACRO_FEATURES` is true and threads it into scenario helpers (`_build_feat_full`, `_indicator_frame_for_rollout`, rollouts) so macro data is **never taken from dates after** the current equity bar.
  - Slices holdout via `_holdout_slice`.
  - Precomputes ARIMA one-step series via `arima_walk_one_step`.
  - Trains a temporary LSTM inside each scenario (Scenario 5 trains once and reuses).

- `run_backtest_all(session, scenario=5, years=None, max_holdout=False, persist=True) -> dict`
  - Sequentially runs Scenario 5 across active tickers (LIGHT_MODE-aware).
  - Records per-ticker `status` and aggregate summaries; may persist full payloads via `backtest_storage` (`saved_run_id` in response when successful).

#### Holdout selection

- `_holdout_slice(n, years=None, max_holdout=False)`
  - Converts years → trading days using 252 days/year.
  - Silently caps holdout so there are enough rows left for LSTM training and enough room for windowing.
  - Returns both requested and actual holdout years.

- `_holdout_note(...)`
  - Generates a human-readable explanation when holdout is reduced (unless `max_holdout`).

#### Scenario functions (internal)

- `_scenario_1_daily(...)`
  - One-step walk-forward style with real daily inputs (optimistic).

- `_scenario_2_multistep(...)`
  - Honest multi-step rollout (no realized future closes used as inputs).
  - Delegates the heavy lifting to `_scenario_2_price_metrics_and_series(...)`.

- `_scenario_4_direction(...)`
  - Daily ensemble series (like scenario 1) plus direction hit rates sampled every `BACKTEST_SCENARIO4_DIRECTION_STRIDE`.

- `_scenario_5_combined(...)`
  - Trains LSTM once, then computes:
    - scenario-2-style price accuracy (`price_accuracy`)
    - scenario-4-style direction sampling (`direction_accuracy`)
    - verdict label and explanation
  - When `light=True`, drops large series arrays to reduce memory in bulk runs.

#### Honest rollout mechanics (Scenario 2/3/5)

- `_lstm_rollout_closes_at_steps(model, meta, df_upto_anchor, milestones, macro_df=None)`
  - Appends synthetic rows one business day at a time using `_append_synthetic_close_row`.
  - Recomputes indicators on the synthetic frame via `_indicator_frame_for_rollout` (passes **`macro_df`** so macro columns forward-fill only from known history up to the rollout’s last date).
  - Predicts next-day close with `predict_lstm_one_step_with_model`.

#### ARIMA helper used for multi-step backtests

- `_arima_price_h_ahead(train_close, h)`
  - Fits an ARIMA on the anchor prefix (log space) and predicts h steps ahead.
  - Exists here to avoid changing `services/arima_model.py` while enabling multi-step comparisons.

#### Metrics and verdict mapping

- `_metrics(y_true, y_pred)`
  - Computes MAE, RMSE, MAPE, n (skipping non-finite pairs).

- `_direction_accuracy(ref, actual, pred)`
  - Compares sign(actual − ref) vs sign(pred − ref).

- `_combined_verdict_label_and_explanation(ticker, mape_30, dir_7)`
  - Implements the exact “Strong/Decent/Needs improvement” thresholds used by Scenario 5.

---

### 2.3 `backend/services/ensemble.py`

This module creates the live forecast JSON returned by `GET /api/stocks/{ticker}/prediction`.

- `ensemble_forecast(session, ticker, model_root=None) -> dict`
  - Calls:
    - `forecast_arima_with_intervals(...)` (refits ARIMA from SQLite each request)
    - `predict_lstm_horizons(...)` (loads saved LSTM and predicts +7/+30/+90)
  - Blends each horizon as `wl * lstm + wa * arima`.
  - Builds a heuristic ensemble CI band:
    - `spread = abs(lstm - arima) + 0.015 * max(last_close, 1.0)`

---

### 2.4 `backend/services/indicators.py`

This module computes the indicators used for charts and as ML inputs:

- `add_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame`
  - Adds RSI(14), MACD(12,26,9), and Bollinger Bands(20, 2σ).
  - Early rows are NaN until rolling windows warm up.
  - Volume is forward/back-filled inside this function.

- `macro_feature_columns() -> list[str]`
  - Names of macro inputs (empty when `USE_MACRO_FEATURES` is false).

- `merge_macro_features(feat_df, macro_df) -> pd.DataFrame`
  - Left-aligns macro to equity dates, drops macro rows after the last equity date, reindex+ffill, applies **rolling** min–max over `LSTM_ROLLING_NORM_WINDOW` per macro column, clips to [0,1].

- `feature_columns() -> list[str]`
  - Chart/legacy column list: OHLCV-related indicators plus macro names when enabled.

---

### 2.5 `backend/services/data_fetcher.py`

This module owns the Yahoo Finance download flow and SQLite persistence.

#### Download + cache

- `download_daily_history(symbol) -> pd.DataFrame`
  - Uses `yfinance.Ticker(symbol).history(...)` with:
    - `period=f"{HISTORY_YEARS}y"`
    - `interval="1d"`
    - `auto_adjust=True`
  - Raises `ValueError` if no data is returned.

- `replace_price_history(session, symbol) -> int`
  - Deletes all existing rows for that ticker, then inserts the fresh full history window.

- `refresh_many(session, symbols) -> dict[str, str]`
  - Refreshes tickers sequentially, continues on per-ticker error, returns “ok” or error message.

#### Load for modeling / charts

- `get_ohlcv_dataframe(session, ticker) -> pd.DataFrame`
  - Loads all cached rows for ticker, returns a DataFrame indexed by date with lowercase columns.

- `get_latest_bar(...)`, `get_latest_bar_map(...)`
  - Used by the stock list endpoint and ensemble for last close.

#### Scheduler/admin runner

- `fetch_macro_features(session) -> dict`
  - Downloads `settings.MACRO_TICKERS`, aligns columns to `MACRO_FEATURE_COLUMNS`, forward-fills, replaces **`macro_daily`** rows. Skipped when `USE_MACRO_FEATURES` is false.

- `get_macro_dataframe(session) -> pd.DataFrame`
  - Reads `macro_daily` into a time-indexed frame (forward-filled).

- `run_refresh_for_active_tickers() -> dict`
  - Opens a short-lived DB session and refreshes every active ticker.
  - Updates app meta `last_price_refresh_utc` when any refresh succeeds.
  - Calls **`fetch_macro_features`** after equities when `USE_MACRO_FEATURES` is true (macro failures are logged; equity refresh still stands).

---

## 3) Data flow (Yahoo → API prediction)

Step-by-step flow for **live predictions** (not backtests):

1. Download → 2. Cache → 3. Load DF → 4. Indicators → 4b. Macro merge (if enabled) → 5. Feature scaling →
6. LSTM predict → 7. ARIMA refit+predict → 8. Ensemble blend → 9. API JSON

ASCII diagram:

```
Yahoo Finance (yfinance)
   |
   |  POST /api/admin/refresh
   v
SQLite (PriceBar + optional macro_daily)  <-------------+
   |                                                    |
   |  get_ohlcv_dataframe()                             |  scheduled_refresh_job()
   v                                                    |
pandas DataFrame (OHLCV)                               |
   |                                                    |
   |  add_indicators()                                  |
   v                                                    |
indicator-enriched DataFrame                           |
   |                                                    |
   |  merge_macro_features() if USE_MACRO_FEATURES      |
   v                                                    |
feature frame for LSTM                                 |
   |                                                    |
   |  last 60-row window + MinMaxScaler (meta fx)       |
   v                                                    |
LSTM returns (+7/+30/+90)  ---> price heads (anchor_close*(1+r))
   |
   |  ARIMA: auto_arima(refit on log close)
   v
ARIMA price forecasts (+7/+30/+90) + ARIMA CI
   |
   |  ensemble_forecast(): weighted blend + heuristic CI
   v
GET /api/stocks/{ticker}/prediction JSON
```

Notes:
- Backtesting routes follow a similar pipeline but train temporary LSTMs and use scenario-specific evaluation logic.

---

## 4) API reference (all endpoints)

Base URL (backend): `http://127.0.0.1:8000`

### 4.1 Health

- **GET** `/health`
  - **Does**: liveness probe
  - **Response**:

```json
{ "status": "ok" }
```

### 4.2 Stock list

- **GET** `/api/stocks`
  - **Does**: returns all tracked tickers and their latest cached close/date (null if not cached).
  - **Response shape** (`routers/stocks.py::StockListResponse`):

```json
{
  "stocks": [
    { "ticker": "AAPL", "last_close": 215.12, "last_trade_date": "2026-03-30", "currency": "USD" },
    { "ticker": "^NDX", "last_close": 18950.34, "last_trade_date": "2026-03-30", "currency": "USD" }
  ],
  "count": 102,
  "light_mode": true,
  "last_price_refresh_utc": "2026-03-31T10:12:01+00:00"
}
```

### 4.3 OHLCV history

- **GET** `/api/stocks/{ticker}/history?limit=2000`
  - **Params**:
    - `limit` (int, default 2000; min 50; max 5000): number of most recent bars
  - **Response**:

```json
{
  "ticker": "AAPL",
  "bars": [
    { "date": "2026-03-28", "open": 214.0, "high": 217.1, "low": 213.8, "close": 216.5, "volume": 61234567 }
  ]
}
```

### 4.4 Indicators (for charts)

- **GET** `/api/stocks/{ticker}/indicators?limit=800`
  - **Params**:
    - `limit` (int, default 800; min 50; max 3000)
  - **Response**:

```json
{
  "ticker": "AAPL",
  "rows": [
    {
      "date": "2026-03-28",
      "close": 216.5,
      "rsi_14": 57.2,
      "macd": 1.23,
      "macd_signal": 1.05,
      "macd_hist": 0.18,
      "bb_middle": 210.4,
      "bb_upper": 218.0,
      "bb_lower": 202.8
    }
  ]
}
```

### 4.5 Ensemble prediction

- **GET** `/api/stocks/{ticker}/prediction`
  - **Does**: returns ensemble forecasts for 7/30/90 days plus CIs.
  - **Notes**:
    - requires an LSTM checkpoint under `MODEL_DIR/<ticker_dir>/` (`lstm.keras` + `lstm_meta.joblib`)
    - ARIMA is refit from SQLite on every request
  - **Response** (from `services/ensemble.py`):

```json
{
  "ticker": "AAPL",
  "last_close": 216.5,
  "weights": { "lstm": 0.6, "arima": 0.4 },
  "horizons": {
    "7": {
      "ensemble": 219.1,
      "lstm": 220.0,
      "arima": 217.7,
      "ci_low": 214.8,
      "ci_high": 223.4,
      "arima_ci_low": 213.9,
      "arima_ci_high": 221.5
    },
    "30": { "ensemble": 225.0, "lstm": 228.2, "arima": 220.3, "ci_low": 215.0, "ci_high": 235.0, "arima_ci_low": 210.0, "arima_ci_high": 230.0 },
    "90": { "ensemble": 240.0, "lstm": 250.0, "arima": 225.0, "ci_low": 220.0, "ci_high": 260.0, "arima_ci_low": 205.0, "arima_ci_high": 245.0 }
  }
}
```

### 4.6 Backtest (scenarios 1–5)

- **GET** `/api/stocks/{ticker}/backtest?scenario=5&years=5`
  - **Params**:
    - `scenario` (1–5, default 1)
    - `years` (float, optional; min 1; silently capped by available data)
    - `max_holdout` (bool, optional; if true, ignores `years`)
  - **Response**:
    - Common fields always include:
      - `ticker`, `scenario`, `scenario_label`
      - `holdout_trading_days`, `holdout_trading_days_target`
      - `holdout_years_requested`, `holdout_years_actual`
      - `holdout_note` (optional)
      - `metrics` (scenario-specific object)
      - `series` (chart data; scenario 5 also includes `price_accuracy` and `direction_accuracy`)

### 4.6b Saved backtest runs (SQLite)

- **GET** `/api/backtest-runs` — list runs (metadata + ticker counts).
- **GET** `/api/backtest-runs/{id}` — one run aggregate + per-ticker summary rows.
- **GET** `/api/backtest-runs/{id}/ticker/{ticker}` — full saved JSON payload for one symbol.
- **GET** `/api/backtest-runs/compare/summary?a={id}&b={id}` — side-by-side comparison.
- **DELETE** `/api/backtest-runs/{id}` — remove a run and its ticker rows.

### 4.7 Admin: refresh prices

- **POST** `/api/admin/refresh`
  - **Does**: runs refresh for active tickers in a background task (returns immediately). Also refreshes **`macro_daily`** when `USE_MACRO_FEATURES` is true.
  - **Errors**: **409** if a refresh job is already running.
  - **Response** (shape):

```json
{
  "status": "accepted",
  "detail": "Refreshing 5 ticker(s) in the background. Poll GET /api/admin/refresh-status for progress.",
  "tickers": ["AAPL", "MSFT"]
}
```

### 4.7b Admin: refresh status

- **GET** `/api/admin/refresh-status`
  - **Does**: snapshot for polling (`state`, `percent`, `current_ticker`, `completed_tickers` / `total_tickers`, `message`, `finished_at`, `error_detail`, optional `result` summary).

### 4.8 Admin: train models

- **POST** `/api/admin/train-models`
  - **Does**: trains LSTM + ARIMA for active tickers in a background task.
  - **Errors**:
    - returns HTTP 409 if training already running.

### 4.9 Admin: training status

- **GET** `/api/admin/training-status`
  - **Does**: progress snapshot for the UI.
  - **Response** (from `services/training_status.py`):

```json
{
  "state": "running",
  "total_tickers": 5,
  "completed_tickers": 2,
  "steps_total": 10,
  "steps_done": 5,
  "percent": 50.0,
  "current_ticker": "GOOGL",
  "current_step": "lstm",
  "message": "Training GOOGL — LSTM",
  "started_at": "2026-03-31T10:12:01+00:00",
  "finished_at": null,
  "tickers_queue": ["AAPL", "MSFT", "GOOGL", "AMZN", "^NDX"],
  "last_results": []
}
```

### 4.9b Admin: bulk backtest status

- **GET** `/api/admin/backtest-status`
  - **Does**: snapshot while a bulk job runs or after completion (`state`, `percent`, `current_ticker`, `result` when `completed`, `error_detail` when `error`).

### 4.10 Admin: bulk backtest (queued)

- **POST** `/api/admin/backtest-all?scenario=5&years=5&persist=true`
  - **Does**: **queues** scenario 5 in a background task (returns immediately). The worker runs `run_backtest_all` sequentially per active ticker; continues on failures. Poll **`GET /api/admin/backtest-status`** for progress and the final payload in `result`.
  - **Errors**: **409** if a bulk backtest is already running. Only **scenario=5** is supported.
  - **Response** (immediate accept shape): `{ "status": "accepted", "detail": "...", "tickers": [...] }`
  - **Final metrics** (inside `result` when poll shows `completed`) — same aggregate shape as the synchronous runner used to return:

```json
{
  "scenario": 5,
  "years": 5.0,
  "max_holdout": false,
  "tickers_tested": ["AAPL", "MSFT", "GOOGL", "AMZN", "^NDX"],
  "results": {
    "AAPL": { "status": "ok", "combined_verdict": "Needs improvement", "mape_30d": 22.1, "direction_accuracy_7d": 0.51, "holdout_days": 1260 },
    "MSFT": { "status": "error: <message>", "combined_verdict": null, "mape_30d": 0.0, "direction_accuracy_7d": 0.0, "holdout_days": 0 }
  },
  "aggregate": {
    "avg_mape_30d": 18.4,
    "avg_direction_accuracy_7d": 0.53,
    "best_ticker": "GOOGL",
    "worst_ticker": "AAPL",
    "tickers_above_50pct_direction": ["MSFT", "GOOGL"],
    "tickers_strong_model": [],
    "tickers_decent_model": ["GOOGL"],
    "tickers_needs_improvement": ["AAPL", "MSFT", "AMZN", "^NDX"]
  }
}
```

Notes on numeric edge cases:
- The backtester uses floating-point `NaN` internally when a metric is unavailable or a model fails. Depending on JSON serialization settings, you may see `NaN` values or a serialization error if non-finite floats are not permitted. If you need strict JSON, consider sanitizing non-finite floats in the API layer.

---

## 5) Config reference (`backend/config.py`)

All settings are defined on `config.Settings` and may be overridden by environment variables (same name, uppercase).

### Core app behavior

- `LIGHT_MODE: bool = True`
  - Controls active tickers: small subset vs full universe.
- `USE_MACRO_FEATURES: bool = True`
  - When true, refresh fills `macro_daily` and LSTM uses 12 input features; when false, macro fetch is skipped and LSTM uses 7 features.
- `MACRO_TICKERS: list[str]`
  - Yahoo symbols aligned with module constant `MACRO_FEATURE_COLUMNS` in `config.py` (same length enforced by validator).
- `MACRO_FEATURE_COLUMNS` (module constant in `config.py`)
  - DB/LSTM macro column names: `vix`, `treasury_10y`, `dollar_index`, `oil_wti`, `sp500_close`.
- `DATABASE_URL: str = "sqlite:///.../nasdaq_predictor.db"`
  - SQLite path for cached prices.
- `HISTORY_YEARS: int = 12`
  - How much history is requested from Yahoo during refresh (equities and macro downloads).
- `CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"`
  - Browser origins allowed to call the API.
- `PRICE_REFRESH_INTERVAL_HOURS: int = 24`
  - Scheduler interval for automatic refresh.
- `MODEL_DIR: Path = backend/saved_models`
  - Artifact folder for LSTM/ARIMA files.

### LSTM training hyperparameters

- `LSTM_EPOCHS: int = 75`
- `LSTM_EARLY_STOPPING_PATIENCE: int = 10`
- `LSTM_BATCH_SIZE: int = 32`
- `LSTM_UNITS: int = 100`
- `LSTM_DROPOUT: float = 0.25`
- `LSTM_LEARNING_RATE: float = 0.0005`
- `SEQUENCE_LENGTH: int = 60`

- `LSTM_ROLLING_NORM_WINDOW: int = 252`
  - Rolling window for **macro** column scaling in `merge_macro_features` (min–max over trailing window, causal along the date index).

### Backtest settings

- `BACKTEST_YEARS: int = 10`
  - Default holdout depth when `years` is not provided.
- `BACKTEST_MIN_PREHOLDOUT_ROWS: int = 650`
  - Minimum pre-holdout rows reserved so the temporary backtest LSTM can train.
- `BACKTEST_SCENARIO2_ANCHOR_STRIDE: int = 10`
  - Anchor spacing for scenario 2/3 rollouts (runtime control).
- `BACKTEST_SCENARIO4_DIRECTION_STRIDE: int = 10`
  - Direction sampling stride for scenario 4/5 direction stats (runtime control).
- `BACKTEST_MULTI_STEP_CHART_HORIZON: int = 30`
  - Which horizon is plotted as the `series` in scenario 2/3/5 price charts.

### Ensemble weights

- `ENSEMBLE_WEIGHT_LSTM: float = 0.6`
- `ENSEMBLE_WEIGHT_ARIMA: float = 0.4`

### Biggest impact on model quality (practical)

- `HISTORY_YEARS`: if too small, you won’t have enough clean windows or robust holdouts.
- `LSTM_EPOCHS` / `LSTM_EARLY_STOPPING_PATIENCE` / `LSTM_LEARNING_RATE`: training convergence.
- `LSTM_UNITS` / `LSTM_DROPOUT`: capacity vs overfitting.
- `SEQUENCE_LENGTH`: how much recent context the model sees.
- `BACKTEST_*`: evaluation realism and stability (not the trained model, but what you measure).

Recommended for fast local testing:
- Keep `LIGHT_MODE=true`
- Use backtests with `years=1` or `years=3` while iterating

