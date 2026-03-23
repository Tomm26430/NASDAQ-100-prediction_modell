"""
Combine LSTM and ARIMA point forecasts with simple confidence bands.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from config import settings
from services.arima_model import forecast_arima_with_intervals
from services.data_fetcher import get_latest_bar
from services.lstm_model import predict_lstm_horizons


def _norm_weights() -> tuple[float, float]:
    wl = settings.ENSEMBLE_WEIGHT_LSTM
    wa = settings.ENSEMBLE_WEIGHT_ARIMA
    s = wl + wa
    if s <= 0:
        return 0.5, 0.5
    return wl / s, wa / s


def ensemble_forecast(
    session: Session,
    ticker: str,
    *,
    model_root: Path | None = None,
) -> dict:
    """
    Return blended 7/30/90d prices plus per-model breakdown and heuristic CI.

    LSTM artifacts must exist under model_root (default: saved_models/). ARIMA refits from SQLite.
    """
    wl, wa = _norm_weights()
    arima = forecast_arima_with_intervals(session, ticker)
    lstm = predict_lstm_horizons(session, ticker, model_root=model_root)

    bar = get_latest_bar(session, ticker)
    lc = float(bar.close) if bar and bar.close is not None else 0.0

    horizons: dict[str, dict] = {}
    for key in ("7", "30", "90"):
        la = lstm[key]
        ap = arima[key]["point"]
        alo, ahi = arima[key]["low"], arima[key]["high"]
        ens = wl * la + wa * ap
        spread = abs(la - ap) + 0.015 * max(lc, 1.0)
        horizons[key] = {
            "ensemble": ens,
            "lstm": la,
            "arima": ap,
            "ci_low": max(0.0, ens - spread),
            "ci_high": ens + spread,
            "arima_ci_low": alo,
            "arima_ci_high": ahi,
        }

    return {
        "ticker": ticker,
        "last_close": lc,
        "weights": {"lstm": wl, "arima": wa},
        "horizons": horizons,
    }
