"""
Auto-ARIMA on log-adjusted closes. Forecasts refit from SQLite each call so they stay current.

Training still pickles the last fit for reproducibility / inspection.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from pmdarima import auto_arima
from sqlalchemy.orm import Session

from config import settings
from services.data_fetcher import get_ohlcv_dataframe
from utils.ml_paths import artifact_stem

logger = logging.getLogger(__name__)


def _arima_path(ticker: str, model_root: Path | None = None) -> Path:
    """Path MODEL_DIR/<ticker_dir>/arima.pkl."""
    base = model_root if model_root is not None else settings.MODEL_DIR
    return base / artifact_stem(ticker) / "arima.pkl"


def arima_model_exists(ticker: str, model_root: Path | None = None) -> bool:
    return _arima_path(ticker, model_root).is_file()


def train_arima_for_ticker(
    session: Session,
    ticker: str,
    *,
    train_end_exclusive: int | None = None,
    model_root: Path | None = None,
) -> dict[str, str]:
    """Fit `auto_arima` on log-close and pickle the estimator."""
    df = get_ohlcv_dataframe(session, ticker)
    if train_end_exclusive is not None:
        df = df.iloc[:train_end_exclusive]
    close = df["close"].astype(float).values
    if len(close) < 120:
        raise ValueError("Need more history before ARIMA can fit reliably.")

    log_close = np.log(np.maximum(close, 1e-6))
    model = auto_arima(
        log_close,
        seasonal=False,
        suppress_warnings=True,
        error_action="ignore",
        stepwise=True,
        max_p=4,
        max_q=4,
        max_d=2,
        information_criterion="aic",
        n_fits=25,
    )

    path = _arima_path(ticker, model_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump({"model": model, "log_space": True}, fh)
    logger.info("Saved ARIMA for %s -> %s", ticker, path)
    return {"status": "ok", "order": str(model.order), "path": str(path)}


def forecast_arima_with_intervals(
    session: Session,
    ticker: str,
    *,
    train_end_exclusive: int | None = None,
) -> dict[str, dict[str, float]]:
    """
    Refit on cached OHLCV (optionally truncated) and return point + 95% interval per horizon.

    Keys: '7', '30', '90' -> point, low, high in **price** space.
    """
    df = get_ohlcv_dataframe(session, ticker)
    if train_end_exclusive is not None:
        df = df.iloc[:train_end_exclusive]
    close = df["close"].astype(float).values
    y = np.log(np.maximum(close, 1e-6))
    model = auto_arima(
        y,
        seasonal=False,
        suppress_warnings=True,
        error_action="ignore",
        stepwise=True,
        max_p=4,
        max_q=4,
        max_d=2,
        n_fits=20,
    )
    out: dict[str, dict[str, float]] = {}
    for h in (7, 30, 90):
        fc, conf = model.predict(n_periods=h, return_conf_int=True)
        lo_log, hi_log = float(conf[-1, 0]), float(conf[-1, 1])
        out[str(h)] = {
            "point": float(np.exp(float(fc[-1]))),
            "low": float(np.exp(lo_log)),
            "high": float(np.exp(hi_log)),
        }
    return out


def arima_walk_one_step(
    train_close: np.ndarray,
    test_close: np.ndarray,
) -> tuple[list[float], list[float]]:
    """
    One-step walk-forward on the test segment (log space fit, price-space outputs).

    Returns (predictions, actuals) including the first test point as "actual" target.
    """
    log_train = np.log(np.maximum(train_close.astype(float), 1e-6))
    model = auto_arima(
        log_train,
        seasonal=False,
        suppress_warnings=True,
        error_action="ignore",
        stepwise=True,
        max_p=3,
        max_q=3,
        max_d=2,
        n_fits=15,
    )
    preds_log: list[float] = []
    actuals: list[float] = []
    m = model
    for y in test_close.astype(float):
        p = m.predict(n_periods=1)[0]
        preds_log.append(float(p))
        actuals.append(float(y))
        try:
            m = m.update([np.log(max(y, 1e-6))])
        except Exception:  # noqa: BLE001
            break
    preds = [float(np.exp(x)) for x in preds_log]
    return preds, actuals
