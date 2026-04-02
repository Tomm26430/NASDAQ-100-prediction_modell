"""
In-memory bulk backtest job progress for the UI (poll GET /api/admin/backtest-status).

Thread-safe updates from FastAPI background tasks. Only one bulk job at a time.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

StateKind = Literal["idle", "running", "completed", "error"]


@dataclass
class BulkBacktestProgress:
    state: StateKind = "idle"
    total_tickers: int = 0
    completed_tickers: int = 0
    current_ticker: str | None = None
    current_index: int = 0  # 1-based while running a ticker
    message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error_detail: str | None = None

    def to_api_dict(self) -> dict[str, Any]:
        pct = 0.0
        if self.state == "running" and self.total_tickers > 0:
            pct = min(100.0, (self.completed_tickers / self.total_tickers) * 100.0)
        elif self.state == "completed":
            pct = 100.0
        return {
            "state": self.state,
            "total_tickers": self.total_tickers,
            "completed_tickers": self.completed_tickers,
            "percent": round(pct, 1),
            "current_ticker": self.current_ticker,
            "current_index": self.current_index,
            "message": self.message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error_detail": self.error_detail,
        }


_lock = threading.Lock()
_progress = BulkBacktestProgress()


def get_backtest_progress() -> dict[str, Any]:
    """Snapshot for GET /api/admin/backtest-status."""
    with _lock:
        return _progress.to_api_dict()


def is_backtest_running() -> bool:
    with _lock:
        return _progress.state == "running"


def try_begin_backtest(tickers: list[str]) -> bool:
    """Return False if a bulk backtest is already running."""
    with _lock:
        if _progress.state == "running":
            return False
        _progress.state = "running"
        _progress.total_tickers = len(tickers)
        _progress.completed_tickers = 0
        _progress.current_ticker = None
        _progress.current_index = 0
        _progress.message = "Starting bulk backtest…"
        _progress.started_at = datetime.now(timezone.utc).isoformat()
        _progress.finished_at = None
        _progress.result = None
        _progress.error_detail = None
    return True


def set_current_ticker(ticker: str, index_1based: int, total: int) -> None:
    """Call before processing each ticker (index_1based is 1..total)."""
    with _lock:
        if _progress.state != "running":
            return
        _progress.current_ticker = ticker
        _progress.current_index = index_1based
        _progress.message = f"Backtesting {ticker} ({index_1based}/{total})"


def record_ticker_finished() -> None:
    """Call after each ticker completes (success or error row)."""
    with _lock:
        if _progress.state != "running":
            return
        _progress.completed_tickers = min(_progress.total_tickers, _progress.completed_tickers + 1)


def finish_backtest_ok(result: dict[str, Any]) -> None:
    with _lock:
        _progress.state = "completed"
        _progress.current_ticker = None
        _progress.message = "Bulk backtest finished."
        _progress.completed_tickers = _progress.total_tickers
        _progress.result = result
        _progress.finished_at = datetime.now(timezone.utc).isoformat()


def finish_backtest_error(exc: str) -> None:
    with _lock:
        _progress.state = "error"
        _progress.error_detail = exc[:2000]
        _progress.message = exc[:500]
        _progress.current_ticker = None
        _progress.finished_at = datetime.now(timezone.utc).isoformat()


def reset_idle_if_done() -> None:
    """Optional: after UI reads result, allow next poll to show idle (not used by default)."""
    with _lock:
        if _progress.state in ("completed", "error"):
            _progress.state = "idle"
            _progress.message = None
            _progress.result = None
            _progress.error_detail = None
