"""
Batch training entry points used by the admin API and background tasks.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from models.database import get_session_factory
from services.arima_model import train_arima_for_ticker
from services.lstm_model import train_lstm_for_ticker
from utils.nasdaq100_tickers import get_active_tickers

logger = logging.getLogger(__name__)


def train_single_ticker(session: Session, ticker: str) -> dict[str, str | dict]:
    """Fit LSTM + ARIMA for one symbol; errors are captured per model."""
    out: dict[str, str | dict] = {"ticker": ticker}
    try:
        out["lstm"] = train_lstm_for_ticker(session, ticker)
    except Exception as exc:  # noqa: BLE001
        logger.exception("LSTM train failed for %s", ticker)
        out["lstm"] = {"status": "error", "detail": str(exc)}
    try:
        out["arima"] = train_arima_for_ticker(session, ticker)
    except Exception as exc:  # noqa: BLE001
        logger.exception("ARIMA train failed for %s", ticker)
        out["arima"] = {"status": "error", "detail": str(exc)}
    return out


def train_all_active_tickers() -> list[dict]:
    """Train every ticker returned by `get_active_tickers()` (respects LIGHT_MODE)."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    results: list[dict] = []
    try:
        for t in get_active_tickers():
            results.append(train_single_ticker(db, t))
    finally:
        db.close()
    return results
