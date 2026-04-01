# NASDAQ Predictor — User Guide

Complete instructions from first install through daily use, API usage, and troubleshooting.

---

## Table of contents

| Section | Description |
|--------|-------------|
| [1. What this application does](#toc-1) | Capabilities and what “analysis” means here |
| [2. Architecture at a glance](#toc-2) | Backend, frontend, database, saved models |
| [3. Prerequisites](#toc-3) | Software and access you need |
| [4. First-time installation](#toc-4) | Python venv, `pip`, `npm install` |
| [5. Configuration](#toc-5) | `.env`, `LIGHT_MODE`, and important settings |
| [6. Starting the backend](#toc-6) | Uvicorn, default URL, what happens on startup |
| [7. Verifying the backend](#toc-7) | Health check vs real data; OpenAPI docs |
| [8. Loading and updating price data](#toc-8) | Yahoo Finance cache, refresh, scheduler |
| [9. Training machine learning models](#toc-9) | LSTM + ARIMA, artifacts, time expectations |
| [10. Starting the frontend](#toc-10) | Vite dev server and API proxy |
| [11. Using the web application](#toc-11) | Dashboard, stock analysis, backtesting |
| [12. API reference](#toc-12) | All HTTP routes and typical responses |
| [13. Predictions and ensemble logic](#toc-13) | Horizons, weights, when predictions fail |
| [14. Backtesting](#toc-14) | What it measures and runtime |
| [15. Nasdaq-100 index shortcut](#toc-15) | `/api/index/ndx` |
| [16. Shutting everything down](#toc-16) | Clean stop |
| [17. Important file locations](#toc-17) | Paths on disk |
| [18. Troubleshooting](#toc-18) | Common errors and fixes |
| [Quick start checklist](#toc-quick) | End-to-end ordered steps |

---

<a id="toc-1"></a>

## 1. What this application does

The **NASDAQ Predictor** is a local full-stack tool that:

1. **Downloads and caches** daily OHLCV (open, high, low, close, volume) for Nasdaq-100 constituents and the **Nasdaq-100 index** (`^NDX`) via Yahoo Finance (`yfinance`).
2. **Computes technical indicators**: RSI(14), MACD(12/26/9), Bollinger Bands (20, 2σ).
3. **Trains per-symbol models**:
   - **LSTM** (TensorFlow/Keras) for multi-horizon closes.
   - **Auto-ARIMA** (`pmdarima`) on log closes; forecasts are **refit from cached data** when you request a prediction so they stay aligned with the latest bars.
4. **Blends** LSTM and ARIMA into an **ensemble** (default **60% LSTM / 40% ARIMA**) with simple confidence-style bands.
5. **Backtests** on a recent holdout window (walk-forward style evaluation; can take around a minute per run).

You interact through:

- A **React + Vite** web UI (dashboard, per-stock charts, backtest page), and/or  
- The **FastAPI** REST API (including interactive docs at `/docs`).

---

<a id="toc-2"></a>

## 2. Architecture at a glance

| Piece | Technology | Default location / URL |
|--------|------------|-------------------------|
| **API server** | FastAPI + Uvicorn | `http://127.0.0.1:8000` |
| **Web UI** | React, TypeScript, Vite, Recharts | `http://localhost:5173` |
| **Price database** | SQLite + SQLAlchemy | `backend/nasdaq_predictor.db` |
| **Saved ML files** | `.keras`, `.joblib`, `.pkl` | `backend/saved_models/` |
| **API prefix** | Routers mounted under `/api` | e.g. `GET /api/stocks` |

The Vite dev server **proxies** requests starting with `/api` to the backend (`vite.config.ts`), so the browser can call `/api/...` on port **5173** and traffic is forwarded to **8000**.

---

<a id="toc-3"></a>

## 3. Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python 3.11+** | 3.13 is fine if your installed TensorFlow wheel supports it. |
| **Node.js + npm** | For the frontend (`npm install`, `npm run dev`). |
| **Internet** | Required for Yahoo Finance downloads and (on first install) `pip`/`npm` packages. |
| **Disk space** | SQLite + years of daily data + TensorFlow; allow a few GB for a comfortable dev setup. |
| **Time** | First `pip install` (especially TensorFlow) can take several minutes. |

---

<a id="toc-4"></a>

## 4. First-time installation

### 4.1 Backend (Python)

From your machine, in a terminal:

```bash
cd "/path/to/nasdaq-predictor/backend"
python3 -m venv .venv
```

Activate the virtual environment:

- **macOS / Linux:** `source .venv/bin/activate`
- **Windows (cmd):** `.venv\Scripts\activate.bat`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`

Install dependencies:

```bash
pip install -r requirements.txt
```

### 4.2 Frontend (Node)

```bash
cd "/path/to/nasdaq-predictor/frontend"
npm install
```

You only need to repeat `npm install` if `package.json` or the lockfile changes.

---

<a id="toc-5"></a>

## 5. Configuration

Settings are defined in `backend/config.py` and can be overridden with **environment variables** (same names, **UPPERCASE**) or a **`.env`** file in the **`backend/`** directory (same folder as `config.py`).

### 5.1 Especially important: `LIGHT_MODE`

| Value | Behavior |
|--------|----------|
| **`true` (default)** | **Active tickers** are a small set (e.g. AAPL, MSFT, GOOGL, AMZN, `^NDX`). Price refresh, scheduled refresh, and **Train models** only process that subset. Fast for learning and testing. |
| **`false`** | Active set is the **full** tracked universe (all listed Nasdaq-100 symbols plus `^NDX`). First refresh and training can take **a very long time**. |

Example (macOS/Linux):

```bash
export LIGHT_MODE=false
```

Or in `backend/.env`:

```env
LIGHT_MODE=false
```

### 5.2 Other settings (reference)

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | SQLite URL (default: file `nasdaq_predictor.db` under `backend/`). |
| `HISTORY_YEARS` | Years of daily history requested from Yahoo when refreshing a symbol (default **12** so 10y backtests have enough pre-holdout data). |
| `CORS_ORIGINS` | Comma-separated browser origins allowed to call the API (includes Vite URLs by default). |
| `PRICE_REFRESH_INTERVAL_HOURS` | Background scheduler interval for automatic price refresh of **active** tickers. |
| `MODEL_DIR` | Directory for saved LSTM/ARIMA artifacts (default `backend/saved_models/`). |
| `LSTM_EPOCHS`, `LSTM_BATCH_SIZE`, `LSTM_UNITS`, `LSTM_DROPOUT`, `SEQUENCE_LENGTH` | LSTM training hyperparameters. |
| `LSTM_ROLLING_NORM_WINDOW` | Trailing trading days for **rolling z-score** on LSTM inputs (default **252**); replaces global MinMax. |
| `BACKTEST_YEARS` | Backtest walk-forward holdout length in **trading years** (default **10**, ≈2520 trading days; capped by cached data). |
| `BACKTEST_MIN_PREHOLDOUT_ROWS` | Minimum daily bars **before** the holdout reserved for training the temporary LSTM (default **650**). If total history is short, the holdout is shortened automatically so LSTM training still has enough clean rows. |
| `ENSEMBLE_WEIGHT_LSTM`, `ENSEMBLE_WEIGHT_ARIMA` | Ensemble blend weights (normalized when combined). |

---

<a id="toc-6"></a>

## 6. Starting the backend

Always run Uvicorn **from the `backend` directory** so imports like `config` and `models` resolve correctly.

With venv activated:

```bash
cd "/path/to/nasdaq-predictor/backend"
source .venv/bin/activate   # if needed
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

| Flag | Meaning |
|------|---------|
| `--reload` | Restart on code changes (development only). |
| `--host 127.0.0.1` | Listen only on localhost. |
| `--port 8000` | Matches the Vite proxy target in `frontend/vite.config.ts`. |

### What happens when the backend starts

1. **Database tables** are created if missing (`init_db()`).
2. If **active** tickers have **no** cached bars yet, a **startup backfill** may run in a **background thread** (so the server can accept requests while Yahoo responds).
3. **APScheduler** starts a **recurring price refresh** for active tickers; the first run is **delayed** so it does not immediately duplicate the startup backfill.

Keep this terminal open while you use the app.

---

<a id="toc-7"></a>

## 7. Verifying the backend

### 7.1 Health check (server only)

```text
GET http://127.0.0.1:8000/health
```

Returns `{"status":"ok"}`. This does **not** prove you have market data or models—only that the process is running.

### 7.2 Stock list (cached data)

```text
GET http://127.0.0.1:8000/api/stocks
```

Returns every **tracked** symbol with `last_close` / `last_trade_date` when that symbol has rows in SQLite. Symbols not yet refreshed show `null` prices.

### 7.3 Interactive API documentation

Open in a browser:

```text
http://127.0.0.1:8000/docs
```

You can execute **GET** and **POST** requests from there (refresh, train, history, prediction, etc.).

---

<a id="toc-8"></a>

## 8. Loading and updating price data

### 8.1 What “refresh” does

**POST `/api/admin/refresh`** enqueues a **background job** that, for each **active** ticker (see `LIGHT_MODE`):

1. Downloads recent daily history from Yahoo Finance.
2. **Replaces** that symbol’s rows in SQLite for a clean, consistent window.

The HTTP response returns immediately with `status: accepted`; watch the **backend terminal** for completion logs.

### 8.2 How to trigger refresh

| Method | Action |
|--------|--------|
| **Web UI** | Dashboard → **Refresh prices** |
| **curl** | `curl -X POST http://127.0.0.1:8000/api/admin/refresh` |
| **Swagger** | `POST /api/admin/refresh` at `/docs` |

### 8.3 Automatic updates

The backend also runs a **scheduled** refresh on an interval (`PRICE_REFRESH_INTERVAL_HOURS`). It uses the same **active** ticker list as manual refresh.

### 8.4 If prices stay empty

- Confirm **`LIGHT_MODE`** matches what you expect (only active tickers are refreshed by default).
- Call **POST `/api/admin/refresh`** and wait; then **GET `/api/stocks`** again.
- Check the terminal for Yahoo/network errors.

---

<a id="toc-9"></a>

## 9. Training machine learning models

### 9.1 Why training is required

- **GET `/api/stocks/{ticker}/prediction`** and **GET `/api/index/ndx`** require a **saved LSTM** for that symbol under `backend/saved_models/`.
- The **LSTM** learns **cumulative simple returns** (vs the last bar in each window) and uses **rolling z-score** inputs (`LSTM_ROLLING_NORM_WINDOW`, default 252 trading days), not raw prices or a single global MinMax fit.
- **ARIMA** training saves a pickle for bookkeeping; **live predictions** refit ARIMA from the **current** SQLite series when you call the prediction endpoint.

### 9.2 How to train

**POST `/api/admin/train-models`** queues a background job that, for each **active** ticker:

1. Trains (or retrains) the **LSTM** and writes `*_lstm.keras` + `*_lstm_meta.joblib`.
2. Fits **auto_arima** and writes `*_arima.pkl`.

| Method | Action |
|--------|--------|
| **Web UI** | Dashboard → **Train models (active tickers)** |
| **curl** | `curl -X POST http://127.0.0.1:8000/api/admin/train-models` |
| **Swagger** | `POST /api/admin/train-models` |

### 9.3 How long it takes

- **Light mode:** often **several minutes** (TensorFlow is heavy).
- **Full universe:** can take **hours**; not recommended until you are confident in the pipeline.

### 9.4 After training

- Reload the **Dashboard** or open **Analysis** for a symbol.
- If prediction still fails, check that this ticker’s LSTM files exist in `backend/saved_models/` and read the error message in the API response.

---

<a id="toc-10"></a>

## 10. Starting the frontend

In a **second** terminal:

```bash
cd "/path/to/nasdaq-predictor/frontend"
npm run dev
```

By default Vite serves at:

```text
http://localhost:5173
```

Ensure the **backend** is still running on **port 8000** so the **proxy** for `/api` works.

---

<a id="toc-11"></a>

## 11. Using the web application

### 11.1 Dashboard (`/`)

- Table of **all tracked** symbols with **last close** and date when cached.
- For a subset of symbols that have prices, the UI may attempt to load **7-day ensemble** estimates (requires trained LSTM for those symbols).
- **Refresh prices** → `POST /api/admin/refresh`
- **Train models** → `POST /api/admin/train-models`
- **Analysis** → opens the stock detail route for that ticker.

### 11.2 Stock analysis (`/stock/{ticker}`)

- **Price** chart from **`GET /api/stocks/{ticker}/history`**
- **Indicators** from **`GET /api/stocks/{ticker}/indicators`**
- **Ensemble forecast cards** (7 / 30 / 90 days) from **`GET /api/stocks/{ticker}/prediction`** when the LSTM exists

**Index symbol:** `^NDX` must be **URL-encoded** in links (the app uses `encodeURIComponent`). Example path: `/stock/%5ENDX`.

### 11.3 Backtesting (`/backtest`)

- Enter a **ticker** and run **Run backtest** → **`GET /api/stocks/{ticker}/backtest`**
- Shows **metrics** (MAE, RMSE, MAPE, sample counts) and a chart of **ensemble vs actual** where dates align.
- Expect **roughly up to a minute** per run; do not spam-click.

---

<a id="toc-12"></a>

## 12. API reference

Base URL (direct to backend): `http://127.0.0.1:8000`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness probe (`status: ok`). |
| GET | `/docs` | Swagger UI. |
| GET | `/api/stocks` | List tracked tickers + latest cached close. |
| GET | `/api/stocks/{ticker}/history` | OHLCV bars (optional `limit` query). |
| GET | `/api/stocks/{ticker}/indicators` | Indicator series for charts (optional `limit`). |
| GET | `/api/stocks/{ticker}/prediction` | Ensemble 7/30/90d forecast (needs LSTM). |
| GET | `/api/stocks/{ticker}/backtest` | Backtest metrics + series. |
| GET | `/api/index/ndx` | Same ensemble JSON as `^NDX` prediction route. |
| POST | `/api/admin/refresh` | Background Yahoo refresh for **active** tickers. |
| POST | `/api/admin/train-models` | Background LSTM + ARIMA train for **active** tickers. |

Unknown tickers (not in the configured universe) return **404** from the stock routes.

---

<a id="toc-13"></a>

## 13. Predictions and ensemble logic

- **Horizons:** **7**, **30**, and **90** trading days ahead (as implemented by the LSTM heads and ARIMA multi-step forecast).
- **Blend:** `ENSEMBLE_WEIGHT_LSTM` and `ENSEMBLE_WEIGHT_ARIMA` (default **0.6 / 0.4**).
- **Bands:** Heuristic interval around the ensemble (model disagreement + small percentage of last close); ARIMA’s own interval is also exposed in the JSON for transparency.

**Not financial advice:** outputs are **experimental** forecasts for learning and prototyping, not a trading signal service.

---

<a id="toc-14"></a>

## 14. Backtesting

- Uses a **holdout** of roughly **`BACKTEST_YEARS`** (default **10** trading years, ≈2520 sessions) at the end of the cached series, capped if you have fewer bars or if the series is too short to keep both a long holdout and at least **`BACKTEST_MIN_PREHOLDOUT_ROWS`** bars for LSTM training (see env table). **Re-fetch prices** after raising `HISTORY_YEARS` so SQLite has enough history for a full 10y-style holdout.
- **ARIMA:** one-step walk-forward updates on the test segment.
- **LSTM:** trained **only** on data **before** the holdout into a **temporary** folder so your main `saved_models/` checkpoints are not overwritten; then one-step-style evaluation over the holdout with **actual** indicator history (teacher forcing).
- **Ensemble series** aligns dates where both model streams exist.

If you see TensorFlow **retracing** warnings in the log during backtests, they are a known performance warning and do not necessarily mean wrong numbers.

---

<a id="toc-15"></a>

## 15. Nasdaq-100 index shortcut

```text
GET http://127.0.0.1:8000/api/index/ndx
```

Equivalent to a successful **`GET /api/stocks/^NDX/prediction`** (with proper URL encoding for `^` if you call the stock route directly). Requires a trained **LSTM for `^NDX`**.

---

<a id="toc-16"></a>

## 16. Shutting everything down

1. Focus the **frontend** terminal → **Ctrl+C** to stop Vite.  
2. Focus the **backend** terminal → **Ctrl+C** to stop Uvicorn.

SQLite and `saved_models/` persist on disk for the next run.

---

<a id="toc-17"></a>

## 17. Important file locations

| Path | Contents |
|------|----------|
| `nasdaq-predictor/backend/main.py` | FastAPI app, CORS, scheduler, `/api/index/ndx`. |
| `nasdaq-predictor/backend/config.py` | Settings and defaults. |
| `nasdaq-predictor/backend/nasdaq_predictor.db` | Cached OHLCV. |
| `nasdaq-predictor/backend/saved_models/` | LSTM `.keras`, `.joblib`, ARIMA `.pkl`. |
| `nasdaq-predictor/frontend/vite.config.ts` | Dev server port **5173** and `/api` proxy. |
| `nasdaq-predictor/frontend/src/pages/` | Dashboard, stock detail, backtest pages. |
| `nasdaq-predictor/frontend/src/api/client.ts` | Axios paths used by the UI. |

---

<a id="toc-18"></a>

## 18. Troubleshooting

| Symptom | Things to check |
|---------|------------------|
| UI loads but **no data** / errors on fetch | Backend running on **8000**? Vite proxy unchanged? Browser console for failed `/api` calls. |
| **`/health` works** but `/api/stocks` has all **null** prices | Run **POST `/api/admin/refresh`**; confirm `LIGHT_MODE` and **active** tickers; wait for background job. |
| **Prediction** returns **400** / train first | Run **POST `/api/admin/train-models`**; wait; confirm `*_lstm.keras` exists for that ticker. |
| **404** on a ticker | Symbol must be in the app’s **tracked universe** (see `backend/utils/nasdaq100_tickers.py`). |
| **Backtest** slow or timeout | Expected; reduce load by testing **light-mode** tickers first. |
| **TensorFlow / pip** errors on install | Python version vs. available wheels; try another Python (e.g. 3.11) in a fresh venv. |
| **CORS** errors (if you change ports) | Add your new origin to `CORS_ORIGINS` in config / `.env`. |

---

<a id="toc-quick"></a>

## Quick start checklist

1. [ ] `pip install -r backend/requirements.txt` (venv recommended)  
2. [ ] `npm install` in `frontend/`  
3. [ ] Terminal A: `uvicorn main:app --reload --host 127.0.0.1 --port 8000` from `backend/`  
4. [ ] `POST /api/admin/refresh` (or Dashboard button)  
5. [ ] `POST /api/admin/train-models` (or Dashboard button); wait for logs  
6. [ ] Terminal B: `npm run dev` in `frontend/`  
7. [ ] Open `http://localhost:5173` → Dashboard → **Analysis** / **Backtest**  

---

*This guide matches the repository layout under `nasdaq-predictor/`. If you move folders, update paths accordingly.*
