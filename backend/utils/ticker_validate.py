"""Validate path/query tickers against the tracked universe."""

from fastapi import HTTPException

from utils.nasdaq100_tickers import all_tracked_symbols

_TRACKED = frozenset(all_tracked_symbols())


def require_tracked_ticker(ticker: str) -> str:
    """Return normalized ticker or raise 404 if it is not part of the dashboard universe."""
    t = ticker.strip()
    if t not in _TRACKED:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{t}' is not in the configured NASDAQ-100 + ^NDX universe.",
        )
    return t
