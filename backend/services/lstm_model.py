"""
LSTM forecaster: sequence input (60 days × features), four regression heads (+1, +7, +30, +90 closes).

Weights and sklearn scalers live under `saved_models/` (or a temporary folder during backtests).
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy.orm import Session

from config import settings
from services.data_fetcher import get_ohlcv_dataframe
from services.indicators import add_indicators, feature_columns
from utils.ml_paths import artifact_stem

logger = logging.getLogger(__name__)

# Output order matches [t+1, t+7, t+30, t+90] closes (trading days ahead).
_HORIZON_DAYS = (1, 7, 30, 90)


def _paths(ticker: str, model_root: Path | None = None) -> tuple[Path, Path]:
    base = model_root if model_root is not None else settings.MODEL_DIR
    base.mkdir(parents=True, exist_ok=True)
    stem = artifact_stem(ticker)
    return base / f"{stem}_lstm.keras", base / f"{stem}_lstm_meta.joblib"


def lstm_model_exists(ticker: str, model_root: Path | None = None) -> bool:
    mp, jp = _paths(ticker, model_root)
    return mp.is_file() and jp.is_file()


def _build_xy(
    feat_df: pd.DataFrame,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over rows; each X is (seq_len × n_features), y is four future closes."""
    cols = feature_columns()
    missing = [c for c in cols if c not in feat_df.columns]
    if missing:
        raise ValueError(f"Feature frame missing columns: {missing}")

    close = feat_df["close"].to_numpy(dtype=np.float64)
    mat = feat_df[cols].to_numpy(dtype=np.float64)
    n = len(feat_df)
    X_list: list[np.ndarray] = []
    y_list: list[list[float]] = []

    for i in range(0, n - seq_len - 90):
        window = mat[i : i + seq_len]
        if np.isnan(window).any():
            continue
        t = i + seq_len - 1
        targets = [close[t + h] for h in _HORIZON_DAYS]
        if any(np.isnan(targets)):
            continue
        X_list.append(window)
        y_list.append(targets)

    if len(X_list) < 64:
        raise ValueError(
            f"Not enough clean LSTM training rows ({len(X_list)}). "
            "Fetch more history or check for excessive NaNs in indicators."
        )

    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


def train_lstm_for_ticker(
    session: Session,
    ticker: str,
    *,
    train_end_exclusive: int | None = None,
    model_root: Path | None = None,
) -> dict[str, str]:
    """
    Train (or retrain) the per-ticker LSTM from SQLite OHLCV and save artifacts.

    train_end_exclusive: if set, only bars strictly before this integer row index are used
    (walk-forward backtests pass a cutoff so the test window is never seen during training).

    model_root: override output directory (temporary path for isolated backtest training).
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    df = get_ohlcv_dataframe(session, ticker)
    if train_end_exclusive is not None:
        df = df.iloc[:train_end_exclusive]
    feat = add_indicators(df).ffill().bfill()
    seq_len = settings.SEQUENCE_LENGTH
    X_raw, y_raw = _build_xy(feat, seq_len)

    fx = MinMaxScaler()
    n_s, s_len, n_f = X_raw.shape
    X_flat = X_raw.reshape(-1, n_f)
    X_s = fx.fit_transform(X_flat).reshape(n_s, s_len, n_f)

    ys = MinMaxScaler()
    y_s = ys.fit_transform(y_raw.reshape(-1, 1)).reshape(n_s, 4)

    split = int(n_s * 0.85)
    if split < 32 or n_s - split < 8:
        split = int(n_s * 0.8)
    X_train, X_val = X_s[:split], X_s[split:]
    y_train, y_val = y_s[:split], y_s[split:]

    keras.utils.set_random_seed(42)
    inp = layers.Input(shape=(seq_len, n_f))
    x = layers.LSTM(settings.LSTM_UNITS, return_sequences=True)(inp)
    x = layers.Dropout(settings.LSTM_DROPOUT)(x)
    x = layers.LSTM(settings.LSTM_UNITS)(x)
    x = layers.Dropout(settings.LSTM_DROPOUT)(x)
    out = layers.Dense(4)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=settings.LSTM_EPOCHS,
        batch_size=settings.LSTM_BATCH_SIZE,
        verbose=0,
        callbacks=[es],
    )

    model_path, meta_path = _paths(ticker, model_root)
    model.save(model_path)
    joblib.dump({"fx": fx, "ys": ys, "feature_cols": feature_columns(), "seq_len": seq_len}, meta_path)
    logger.info("Saved LSTM for %s -> %s", ticker, model_path)
    return {"status": "ok", "model_path": str(model_path), "samples": str(n_s)}


def _load_keras_and_meta(ticker: str, model_root: Path | None = None):
    from tensorflow import keras

    model_path, meta_path = _paths(ticker, model_root)
    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"LSTM bundle missing for {ticker}; train first.")
    model = keras.models.load_model(model_path)
    meta = joblib.load(meta_path)
    return model, meta


def predict_lstm_horizons(
    session: Session,
    ticker: str,
    *,
    model_root: Path | None = None,
) -> dict[str, float]:
    """
    Return LSTM point estimates for +7, +30, +90 closes (keys '7','30','90').
    """
    df = get_ohlcv_dataframe(session, ticker)
    feat = add_indicators(df).ffill().bfill()
    model, meta = _load_keras_and_meta(ticker, model_root)
    seq_len = int(meta["seq_len"])
    cols: list[str] = meta["feature_cols"]
    if len(feat) < seq_len:
        raise ValueError("Not enough rows after indicators to form one LSTM window.")

    window = feat.iloc[-seq_len:][cols].to_numpy(dtype=np.float32)
    if np.isnan(window).any():
        raise ValueError("Latest window still contains NaNs; wait for more history.")

    fx: MinMaxScaler = meta["fx"]
    ys: MinMaxScaler = meta["ys"]
    w_flat = fx.transform(window.reshape(-1, window.shape[1])).reshape(1, seq_len, window.shape[1])
    pred_s = model.predict(w_flat, verbose=0)[0]
    pred = ys.inverse_transform(pred_s.reshape(-1, 1)).flatten()
    return {"7": float(pred[1]), "30": float(pred[2]), "90": float(pred[3])}


def predict_lstm_one_step(
    ticker: str,
    feat_df: pd.DataFrame,
    *,
    model_root: Path | None = None,
) -> float:
    """
    Next-day (+1) close from the last seq_len rows of `feat_df` (indicator-enriched).
    """
    model, meta = _load_keras_and_meta(ticker, model_root)
    seq_len = int(meta["seq_len"])
    cols: list[str] = meta["feature_cols"]
    tail = feat_df.iloc[-seq_len:][cols].to_numpy(dtype=np.float32)
    if tail.shape[0] < seq_len or np.isnan(tail).any():
        return float("nan")
    fx: MinMaxScaler = meta["fx"]
    ys: MinMaxScaler = meta["ys"]
    w = fx.transform(tail.reshape(-1, tail.shape[1])).reshape(1, seq_len, tail.shape[1])
    pred_s = model.predict(w, verbose=0)[0]
    pred = ys.inverse_transform(pred_s.reshape(-1, 1)).flatten()
    return float(pred[0])
