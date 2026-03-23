"""
Operational endpoints (manual cache refresh). Kept separate from public stock routes.
"""

import logging

from fastapi import APIRouter, BackgroundTasks

from config import settings
from services.data_fetcher import run_refresh_for_active_tickers
from services.train_jobs import train_all_active_tickers
from utils.nasdaq100_tickers import get_active_tickers

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/refresh")
def trigger_price_refresh(background_tasks: BackgroundTasks) -> dict[str, str | int]:
    """
    Kick off a Yahoo Finance download for all *active* tickers (respects LIGHT_MODE).

    The job runs after the response is sent so the browser does not time out on large universes.
    """
    if not settings.LIGHT_MODE:
        logger.info("Full-universe refresh requested; expect a long run.")

    def _run() -> None:
        summary = run_refresh_for_active_tickers()
        logger.info("Refresh finished: %s", summary)

    background_tasks.add_task(_run)

    n = len(get_active_tickers())
    return {
        "status": "accepted",
        "detail": f"Refreshing {n} ticker(s) in the background. Poll GET /api/stocks for new prices.",
    }


@router.post("/train-models")
def trigger_model_training(background_tasks: BackgroundTasks) -> dict[str, str | int]:
    """
    Train LSTM + ARIMA checkpoints for every active ticker (respects LIGHT_MODE).

    This can take several minutes even in light mode; runs after the HTTP response returns.
    """

    def _run() -> None:
        summary = train_all_active_tickers()
        logger.info("Training finished for %s tickers", len(summary))

    background_tasks.add_task(_run)
    n = len(get_active_tickers())
    return {
        "status": "accepted",
        "detail": f"Queued training for {n} ticker(s). Watch logs; then call GET /api/stocks/{{ticker}}/prediction.",
    }
