"""
Batch training entry points used by the admin API and background tasks.
"""

from __future__ import annotations

import logging

from models.database import get_session_factory
from services.arima_model import train_arima_for_ticker
from services.lstm_model import train_lstm_for_ticker
from services import training_status
from utils.nasdaq100_tickers import get_active_tickers

logger = logging.getLogger(__name__)


def train_all_active_tickers() -> list[dict]:
    """Train every ticker returned by `get_active_tickers()` (respects LIGHT_MODE). Updates training_status."""
    tickers = get_active_tickers()
    SessionLocal = get_session_factory()
    db = SessionLocal()
    results: list[dict] = []
    try:
        for t in tickers:
            out: dict[str, str | dict] = {"ticker": t}

            training_status.set_current_step(t, "lstm")
            try:
                out["lstm"] = train_lstm_for_ticker(db, t)
            except Exception as exc:  # noqa: BLE001
                logger.exception("LSTM train failed for %s", t)
                out["lstm"] = {"status": "error", "detail": str(exc)}
            training_status.record_step_finished()

            training_status.set_current_step(t, "arima")
            try:
                out["arima"] = train_arima_for_ticker(db, t)
            except Exception as exc:  # noqa: BLE001
                logger.exception("ARIMA train failed for %s", t)
                out["arima"] = {"status": "error", "detail": str(exc)}
            training_status.record_step_finished()

            results.append(out)
            training_status.record_ticker_finished(out)

        training_status.finish_training_ok()
    except Exception as exc:  # noqa: BLE001
        training_status.finish_training_error(str(exc))
        logger.exception("Training batch crashed")
    finally:
        db.close()
    return results
