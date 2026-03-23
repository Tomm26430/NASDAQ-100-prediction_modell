"""
Walk-forward style evaluation on the most recent BACKTEST_TRADING_DAYS (~1y).

ARIMA: true one-step updates. LSTM: trained only on the pre-test window, then next-day forecasts
using actual indicator history (teacher forcing).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from config import settings
from services.arima_model import arima_walk_one_step
from services.data_fetcher import get_ohlcv_dataframe
from services.indicators import add_indicators
from services.lstm_model import predict_lstm_one_step, train_lstm_for_ticker


def _norm_w() -> tuple[float, float]:
    wl = settings.ENSEMBLE_WEIGHT_LSTM
    wa = settings.ENSEMBLE_WEIGHT_ARIMA
    s = wl + wa
    if s <= 0:
        return 0.5, 0.5
    return wl / s, wa / s


def _metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    if len(y_true) < 5 or len(y_pred) < 5:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n": 0.0}
    a = np.array(y_true, dtype=float)
    p = np.array(y_pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    if mask.sum() < 5:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n": float(mask.sum())}
    a, p = a[mask], p[mask]
    err = p - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err / np.maximum(np.abs(a), 1e-6))) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape": mape, "n": float(len(a))}


def run_backtest(session: Session, ticker: str) -> dict:
    """
    Produce metrics + daily series for charts on the last `BACKTEST_TRADING_DAYS` bars.
    """
    df = get_ohlcv_dataframe(session, ticker)
    n = len(df)
    hold = min(settings.BACKTEST_TRADING_DAYS, n - settings.SEQUENCE_LENGTH - 100)
    if hold < 30:
        raise ValueError("Not enough history for a meaningful backtest window.")

    test_start = n - hold
    train_close = df["close"].iloc[:test_start].to_numpy()
    test_close = df["close"].iloc[test_start:].to_numpy()

    arima_pred, arima_actual = arima_walk_one_step(train_close, test_close)
    dates = df.index[test_start : test_start + len(arima_actual)]
    arima_series = [
        {"date": d.isoformat()[:10], "actual": float(a), "predicted": float(p)}
        for d, a, p in zip(dates, arima_actual, arima_pred, strict=False)
    ]

    lstm_series: list[dict] = []
    lstm_metrics: dict = {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n": 0.0}
    lstm_err: str | None = None
    wl, wa = _norm_w()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        try:
            train_lstm_for_ticker(
                session,
                ticker,
                train_end_exclusive=test_start,
                model_root=root,
            )
            feat_full = add_indicators(df).ffill().bfill()
            for i in range(test_start, n - 1):
                dt = df.index[i]
                if dt not in feat_full.index:
                    continue
                sub = feat_full.loc[:dt]
                if len(sub) < settings.SEQUENCE_LENGTH:
                    continue
                pred = predict_lstm_one_step(ticker, sub, model_root=root)
                act = float(df["close"].iloc[i + 1])
                if not np.isfinite(pred):
                    continue
                lstm_series.append(
                    {
                        "date": df.index[i + 1].isoformat()[:10],
                        "actual": act,
                        "predicted": pred,
                    }
                )
            if lstm_series:
                lstm_metrics = _metrics(
                    [x["actual"] for x in lstm_series],
                    [x["predicted"] for x in lstm_series],
                )
        except Exception as exc:  # noqa: BLE001
            lstm_err = str(exc)

    if lstm_err:
        lstm_metrics = {**lstm_metrics, "error": lstm_err}

    arima_m = _metrics(arima_actual, arima_pred)

    by_a = {x["date"]: x for x in arima_series}
    by_l = {x["date"]: x for x in lstm_series}
    ens_series: list[dict] = []
    for d in sorted(set(by_a) & set(by_l)):
        a, l = by_a[d], by_l[d]
        blend = wl * l["predicted"] + wa * a["predicted"]
        ens_series.append(
            {
                "date": d,
                "actual": a["actual"],
                "predicted_ensemble": blend,
                "predicted_lstm": l["predicted"],
                "predicted_arima": a["predicted"],
            }
        )

    ens_m = _metrics([r["actual"] for r in ens_series], [r["predicted_ensemble"] for r in ens_series])

    return {
        "ticker": ticker,
        "holdout_trading_days": hold,
        "metrics": {
            "arima_one_step": arima_m,
            "lstm_one_step": lstm_metrics,
            "ensemble": ens_m,
        },
        "series": {
            "arima": arima_series,
            "lstm": lstm_series,
            "ensemble": ens_series,
        },
    }
