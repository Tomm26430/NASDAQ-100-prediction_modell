"""
Download historical OHLCV from Yahoo Finance (via yfinance) and persist to SQLite.

The flow is intentionally simple: wipe existing rows for a symbol, then insert the latest
full history window. That avoids tricky merge logic while keeping the table small.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from config import settings
from models.database import AppMeta, PriceBar, get_session_factory
from utils.nasdaq100_tickers import get_active_tickers

logger = logging.getLogger(__name__)


def _refresh_progress_tick(sym: str, index_1based: int, total: int) -> None:
    try:
        from services.refresh_status import is_refresh_running, set_current_ticker

        if is_refresh_running():
            set_current_ticker(sym, index_1based, total)
    except Exception:  # noqa: BLE001
        pass


def _refresh_progress_done_one() -> None:
    try:
        from services.refresh_status import is_refresh_running, record_ticker_finished

        if is_refresh_running():
            record_ticker_finished()
    except Exception:  # noqa: BLE001
        pass


def _yahoo_ticker(symbol: str) -> str:
    """Normalize symbols for yfinance (already correct for US equities and ^NDX)."""
    return symbol.strip()


def download_daily_history(symbol: str) -> pd.DataFrame:
    """
    Pull daily bars for HISTORY_YEARS from Yahoo Finance.

    Returns a DataFrame indexed by date with columns: Open, High, Low, Close, Volume.
    Raises ValueError with a clear message if nothing comes back (delisted symbol, typo).
    """
    sym = _yahoo_ticker(symbol)
    t = yf.Ticker(sym)
    # repair=True can import SciPy inside yfinance; keep False unless you add scipy and need fixes
    df = t.history(period=f"{settings.HISTORY_YEARS}y", interval="1d", auto_adjust=True, repair=False)
    if df is None or df.empty:
        raise ValueError(f"No price history returned for '{sym}'. Check the ticker or try again later.")
    return df


def replace_price_history(session: Session, symbol: str) -> int:
    """
    Replace all cached rows for `symbol` with a fresh Yahoo download.

    Returns how many daily rows were stored.
    """
    sym = _yahoo_ticker(symbol)
    df = download_daily_history(sym)

    # Remove old bars so we never keep stale overlapping dates
    session.execute(delete(PriceBar).where(PriceBar.ticker == sym))

    rows: list[PriceBar] = []
    for ts, row in df.iterrows():
        # Yahoo gives a timezone-aware timestamp; we only persist the calendar date
        trade_date = pd.Timestamp(ts).date()
        rows.append(
            PriceBar(
                ticker=sym,
                trade_date=trade_date,
                open=float(row["Open"]) if pd.notna(row["Open"]) else None,
                high=float(row["High"]) if pd.notna(row["High"]) else None,
                low=float(row["Low"]) if pd.notna(row["Low"]) else None,
                close=float(row["Close"]) if pd.notna(row["Close"]) else None,
                volume=float(row["Volume"]) if pd.notna(row["Volume"]) else None,
            )
        )

    session.add_all(rows)
    session.commit()
    return len(rows)


def refresh_many(session: Session, symbols: list[str]) -> dict[str, str]:
    """
    Refresh several tickers in one go. Continues on per-symbol errors.

    Returns a map ticker -> "ok" or a short error string (for API reporting).
    """
    results: dict[str, str] = {}
    total = len(symbols)
    for i, sym in enumerate(symbols):
        _refresh_progress_tick(sym, i + 1, total)
        try:
            n = replace_price_history(session, sym)
            results[sym] = f"ok ({n} rows)"
        except Exception as exc:  # noqa: BLE001 — we want to capture Yahoo/network issues
            logger.exception("Failed to refresh %s", sym)
            session.rollback()
            results[sym] = str(exc)
        finally:
            _refresh_progress_done_one()
    return results


def get_ohlcv_dataframe(session: Session, ticker: str) -> pd.DataFrame:
    """
    Load all cached daily bars for one symbol as a time-indexed DataFrame.

    Columns: open, high, low, close, volume (lowercase, NaNs possible on bad rows).
    """
    sym = _yahoo_ticker(ticker)
    stmt = select(PriceBar).where(PriceBar.ticker == sym).order_by(PriceBar.trade_date)
    rows = list(session.execute(stmt).scalars().all())
    if len(rows) < 10:
        raise ValueError(
            f"Not enough cached rows for '{sym}' ({len(rows)}). Run POST /api/admin/refresh first."
        )
    df = pd.DataFrame(
        {
            "date": [r.trade_date for r in rows],
            "open": [r.open for r in rows],
            "high": [r.high for r in rows],
            "low": [r.low for r in rows],
            "close": [r.close for r in rows],
            "volume": [r.volume if r.volume is not None else float("nan") for r in rows],
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["volume"] = df["volume"].ffill().bfill().fillna(0.0)
    return df


def get_latest_bar(session: Session, symbol: str) -> PriceBar | None:
    """Most recent stored bar for a symbol (by trade_date), or None if missing."""
    sym = _yahoo_ticker(symbol)
    stmt = select(PriceBar).where(PriceBar.ticker == sym).order_by(PriceBar.trade_date.desc()).limit(1)
    return session.execute(stmt).scalar_one_or_none()


def get_latest_bar_map(session: Session) -> dict[str, PriceBar]:
    """
    One efficient query: latest trade_date per ticker joined back to full rows.

    Used by the dashboard list endpoint so we do not issue one query per symbol.
    """
    latest = (
        select(PriceBar.ticker.label("ticker"), func.max(PriceBar.trade_date).label("mx"))
        .group_by(PriceBar.ticker)
        .subquery()
    )
    stmt = select(PriceBar).join(
        latest,
        (PriceBar.ticker == latest.c.ticker) & (PriceBar.trade_date == latest.c.mx),
    )
    rows = session.execute(stmt).scalars().all()
    return {bar.ticker: bar for bar in rows}


def set_meta(session: Session, key: str, value: str) -> None:
    """Upsert a key in app_meta."""
    row = session.get(AppMeta, key)
    if row is None:
        session.add(AppMeta(key=key, value=value))
    else:
        row.value = value
    session.commit()


def get_meta(session: Session, key: str) -> str | None:
    """Read meta string or None."""
    row = session.get(AppMeta, key)
    return row.value if row else None


def touch_last_refresh_time(session: Session) -> None:
    """Record UTC timestamp of the last bulk refresh (for debugging/monitoring)."""
    now = datetime.now(timezone.utc).isoformat()
    set_meta(session, "last_price_refresh_utc", now)


def run_refresh_for_active_tickers() -> dict[str, Any]:
    """
    Open a short-lived session and refresh every ticker returned by get_active_tickers().

    Used by the admin POST endpoint and the APScheduler job so logic lives in one place.
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        tickers = get_active_tickers()
        results = refresh_many(db, tickers)
        if any(str(v).startswith("ok") for v in results.values()):
            touch_last_refresh_time(db)
        failures = {k: v for k, v in results.items() if not str(v).startswith("ok")}
        return {"tickers_requested": len(tickers), "results": results, "failures": failures}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Scheduled/manual refresh crashed")
        db.rollback()
        return {"error": str(exc)}
    finally:
        db.close()
