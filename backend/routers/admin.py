"""
Operational endpoints (manual cache refresh). Kept separate from public stock routes.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from config import settings
from models.database import get_db
from services.backtester import run_backtest_all
from services.data_fetcher import run_refresh_for_active_tickers
from services.train_jobs import train_all_active_tickers
from services.training_status import get_progress, try_begin_training
from utils.nasdaq100_tickers import get_active_tickers

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/training-status")
def training_status_endpoint() -> dict:
    """
    Poll this while training runs: percent, current ticker/step, and completion state.

    States: `idle` | `running` | `completed` | `error`
    """
    return get_progress()


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

    Returns 409 if a training job is already running. Poll GET /api/admin/training-status for progress.
    """
    tickers = get_active_tickers()
    if not try_begin_training(tickers):
        raise HTTPException(
            status_code=409,
            detail="Training is already in progress. Wait for it to finish or check GET /api/admin/training-status.",
        )

    def _run() -> None:
        summary = train_all_active_tickers()
        logger.info("Training finished for %s tickers", len(summary))

    background_tasks.add_task(_run)
    n = len(tickers)
    return {
        "status": "accepted",
        "detail": f"Queued training for {n} ticker(s). Poll GET /api/admin/training-status for progress.",
    }


@router.post("/backtest-all")
def admin_backtest_all(
    scenario: int = Query(5, ge=1, le=5, description="Bulk run currently optimized for scenario 5."),
    years: float | None = Query(
        None,
        ge=1,
        le=80,
        description="Holdout years per ticker; default BACKTEST_YEARS when omitted.",
    ),
    max_holdout: bool = Query(False, description="Maximize holdout per ticker given cached history."),
    db: Session = Depends(get_db),
) -> dict:
    """
    Run Scenario 5 (default) backtests sequentially for every active ticker (respects LIGHT_MODE).
    One ticker failing does not stop the rest; each row includes status ok or error message.
    """
    if scenario != 5:
        raise HTTPException(
            status_code=400,
            detail="backtest-all currently supports scenario=5 only.",
        )
    return run_backtest_all(db, scenario=scenario, years=years, max_holdout=max_holdout)
