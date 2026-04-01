"""
In-memory training job progress for the UI (poll GET /api/admin/training-status).

Thread-safe updates from FastAPI background tasks. Resets when a job finishes or errors.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

StateKind = Literal["idle", "running", "completed", "error"]


@dataclass
class TrainingProgress:
    state: StateKind = "idle"
    total_tickers: int = 0
    tickers: list[str] = field(default_factory=list)
    completed_tickers: int = 0
    # Two steps per ticker: LSTM + ARIMA
    steps_total: int = 0
    steps_done: int = 0
    current_ticker: str | None = None
    current_step: str | None = None  # "lstm" | "arima"
    message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    last_results: list[dict[str, Any]] = field(default_factory=list)

    def to_api_dict(self) -> dict[str, Any]:
        pct = 0.0
        if self.steps_total > 0:
            pct = min(100.0, (self.steps_done / self.steps_total) * 100.0)
        return {
            "state": self.state,
            "total_tickers": self.total_tickers,
            "completed_tickers": self.completed_tickers,
            "steps_total": self.steps_total,
            "steps_done": self.steps_done,
            "percent": round(pct, 1),
            "current_ticker": self.current_ticker,
            "current_step": self.current_step,
            "message": self.message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "tickers_queue": self.tickers,
            "last_results": self.last_results[-20:],  # cap payload size
        }


_lock = threading.Lock()
_progress = TrainingProgress()


def get_progress() -> dict[str, Any]:
    """Snapshot for API responses."""
    with _lock:
        return _progress.to_api_dict()


def try_begin_training(tickers: list[str]) -> bool:
    """Return False if a training job is already running."""
    with _lock:
        if _progress.state == "running":
            return False
        _progress.state = "running"
        _progress.tickers = list(tickers)
        _progress.total_tickers = len(tickers)
        _progress.steps_total = max(1, len(tickers) * 2)
        _progress.steps_done = 0
        _progress.completed_tickers = 0
        _progress.current_ticker = None
        _progress.current_step = None
        _progress.message = "Starting…"
        _progress.started_at = datetime.now(timezone.utc).isoformat()
        _progress.finished_at = None
        _progress.last_results = []
    return True


def set_current_step(ticker: str, step: str) -> None:
    """step is 'lstm' or 'arima'."""
    with _lock:
        if _progress.state != "running":
            return
        _progress.current_ticker = ticker
        _progress.current_step = step
        _progress.message = f"Training {ticker} — {step.upper()}"


def record_step_finished() -> None:
    """Call after each LSTM or ARIMA attempt completes."""
    with _lock:
        if _progress.state != "running":
            return
        _progress.steps_done = min(_progress.steps_total, _progress.steps_done + 1)


def record_ticker_finished(result: dict[str, Any]) -> None:
    with _lock:
        if _progress.state != "running":
            return
        _progress.completed_tickers += 1
        _progress.last_results.append(result)


def finish_training_ok() -> None:
    with _lock:
        _progress.state = "completed"
        _progress.current_ticker = None
        _progress.current_step = None
        _progress.message = "Training finished."
        _progress.steps_done = _progress.steps_total
        _progress.finished_at = datetime.now(timezone.utc).isoformat()


def finish_training_error(exc: str) -> None:
    with _lock:
        _progress.state = "error"
        _progress.message = exc[:500]
        _progress.current_ticker = None
        _progress.current_step = None
        _progress.finished_at = datetime.now(timezone.utc).isoformat()


def reset_idle() -> None:
    """Optional: clear completed state so next poll shows idle until new job."""
    with _lock:
        if _progress.state in ("completed", "error"):
            _progress.state = "idle"
            _progress.message = None
