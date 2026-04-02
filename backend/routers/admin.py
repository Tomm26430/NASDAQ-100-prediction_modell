"""
Operational endpoints (manual cache refresh). Kept separate from public stock routes.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from config import settings
from models.database import get_session_factory
from services.backtest_status import (
    finish_backtest_error,
    finish_backtest_ok,
    get_backtest_progress,
    try_begin_backtest,
)
from services.backtester import run_backtest_all
from services.data_fetcher import run_refresh_for_active_tickers
from services.refresh_status import (
    finish_refresh_error,
    finish_refresh_ok,
    get_refresh_progress,
    try_begin_refresh,
)
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


@router.get("/refresh-status")
def refresh_status_endpoint() -> dict:
    """Poll while a price refresh runs: percent, current ticker, and completion state."""
    return get_refresh_progress()


@router.post("/refresh")
def trigger_price_refresh(background_tasks: BackgroundTasks) -> dict[str, str | int | list[str]]:
    """
    Kick off a Yahoo Finance download for all *active* tickers (respects LIGHT_MODE).

    Returns 409 if a refresh is already running. Poll GET /api/admin/refresh-status for progress.
    """
    if not settings.LIGHT_MODE:
        logger.info("Full-universe refresh requested; expect a long run.")

    tickers = get_active_tickers()
    if not try_begin_refresh(tickers):
        raise HTTPException(
            status_code=409,
            detail="A price refresh is already in progress. Poll GET /api/admin/refresh-status.",
        )

    def _run() -> None:
        summary = run_refresh_for_active_tickers()
        if summary.get("error"):
            finish_refresh_error(str(summary["error"]))
        else:
            finish_refresh_ok(summary)
        logger.info("Refresh finished: %s", summary)

    background_tasks.add_task(_run)

    n = len(tickers)
    return {
        "status": "accepted",
        "detail": f"Refreshing {n} ticker(s) in the background. Poll GET /api/admin/refresh-status for progress.",
        "tickers": tickers,
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


@router.get("/backtest-status")
def backtest_status_endpoint() -> dict:
    """Poll while a bulk backtest runs: percent, current ticker, and completion state."""
    return get_backtest_progress()


@router.post("/backtest-all")
def admin_backtest_all(
    background_tasks: BackgroundTasks,
    scenario: int = Query(5, ge=1, le=5, description="Bulk run currently optimized for scenario 5."),
    years: float | None = Query(
        None,
        ge=1,
        le=80,
        description="Holdout years per ticker; default BACKTEST_YEARS when omitted.",
    ),
    max_holdout: bool = Query(False, description="Maximize holdout per ticker given cached history."),
    persist: bool = Query(True, description="Save full per-ticker results to SQLite for history and charts."),
) -> dict[str, str | int | list[str]]:
    """
    Queue Scenario 5 (default) backtests for every active ticker (respects LIGHT_MODE).
    Returns immediately; poll GET /api/admin/backtest-status for progress and the final `result` payload.

    One ticker failing does not stop the rest; each row includes status ok or error message.
    """
    if scenario != 5:
        raise HTTPException(
            status_code=400,
            detail="backtest-all currently supports scenario=5 only.",
        )
    tickers = get_active_tickers()
    if not try_begin_backtest(tickers):
        raise HTTPException(
            status_code=409,
            detail="A bulk backtest is already in progress. Poll GET /api/admin/backtest-status.",
        )

    def _run() -> None:
        SessionLocal = get_session_factory()
        db = SessionLocal()
        try:
            out = run_backtest_all(db, scenario=scenario, years=years, max_holdout=max_holdout, persist=persist)
            finish_backtest_ok(out)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Bulk backtest failed")
            finish_backtest_error(str(exc))
        finally:
            db.close()

    background_tasks.add_task(_run)
    n = len(tickers)
    return {
        "status": "accepted",
        "detail": f"Bulk backtest queued for {n} ticker(s). Poll GET /api/admin/backtest-status for progress.",
        "tickers": tickers,
    }
