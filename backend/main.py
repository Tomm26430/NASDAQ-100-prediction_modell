"""
FastAPI entry point: configures CORS, registers routers, and starts background jobs.

Run locally (from this `backend` directory):
    uvicorn main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from models.database import get_db, get_session_factory, init_db
from routers import admin, backtest_runs, stocks
from services.ensemble import ensemble_forecast
from services.lstm_model import lstm_model_exists
from services.data_fetcher import get_latest_bar, run_refresh_for_active_tickers
from utils.nasdaq100_tickers import get_active_tickers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nasdaq_predictor")

scheduler = BackgroundScheduler()


def scheduled_refresh_job() -> None:
    """APScheduler entry point — must not raise or the job will stop firing."""
    try:
        summary = run_refresh_for_active_tickers()
        logger.info("Scheduled refresh: %s", summary)
    except Exception:
        logger.exception("Scheduled refresh raised unexpectedly")


def _startup_backfill() -> None:
    """
    If SQLite has no bars yet for the active universe, download once on boot.

    Runs in a thread pool so `uvicorn` can bind immediately while Yahoo responds.
    """
    tickers = get_active_tickers()
    if not tickers:
        return
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        warm = all(get_latest_bar(db, t) is not None for t in tickers)
    finally:
        db.close()
    if warm:
        logger.info("Startup backfill skipped; cache already has active tickers.")
        return
    logger.info("Startup backfill starting for %s", tickers)
    summary = run_refresh_for_active_tickers()
    logger.info("Startup backfill finished: %s", summary)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create SQLite file + tables before accepting traffic
    init_db()
    # Warm cache without blocking the event loop
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _startup_backfill)
    # First run is delayed so we do not duplicate the startup backfill immediately
    scheduler.add_job(
        scheduled_refresh_job,
        trigger="interval",
        hours=settings.PRICE_REFRESH_INTERVAL_HOURS,
        id="price_refresh",
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(timezone.utc) + timedelta(hours=settings.PRICE_REFRESH_INTERVAL_HOURS),
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "Scheduler running: price refresh every %sh (first run after delay)",
        settings.PRICE_REFRESH_INTERVAL_HOURS,
    )
    yield
    scheduler.shutdown(wait=False)


app = FastAPI(title="NASDAQ Predictor API", version="0.1.0", lifespan=lifespan)

# Browser security: only listed origins may call the API from JavaScript
_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stocks.router, prefix="/api")
app.include_router(admin.router, prefix="/api")
app.include_router(backtest_runs.router, prefix="/api")


@app.get("/api/index/ndx")
def ndx_index_prediction(db: Session = Depends(get_db)) -> dict:
    """Ensemble forecast for the Nasdaq-100 price index (^NDX), same JSON as /api/stocks/^NDX/prediction."""
    sym = "^NDX"
    if not lstm_model_exists(sym):
        raise HTTPException(
            status_code=400,
            detail="Train models first so ^NDX has an LSTM checkpoint (POST /api/admin/train-models).",
        )
    return ensemble_forecast(db, sym)


@app.get("/health")
def health() -> dict[str, str]:
    """Minimal probe for Docker or uptime checks."""
    return {"status": "ok"}
