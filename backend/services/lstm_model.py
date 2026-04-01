"""
LSTM forecaster: sequence input (60 days × features), four regression heads (+1, +7, +30, +90).

Targets are cumulative simple returns from the reference bar; inputs use return-based features
and rolling z-score normalization (no global MinMax). Artifacts live under `saved_models/`
(or a temporary folder during backtests).
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from config import settings
from services.data_fetcher import get_ohlcv_dataframe
from services.indicators import add_indicators
from utils.ml_paths import artifact_stem

logger = logging.getLogger(__name__)

# Output order matches [t+1, t+7, t+30, t+90] — cumulative simple returns vs reference close.
_HORIZON_DAYS = (1, 7, 30, 90)


def lstm_feature_columns() -> list[str]:
    """Column order fed into the LSTM (after rolling normalization)."""
    return [
        "ret_close",
        "ret_volume",
        "rsi_14",
        "macd_rel",
        "macd_signal_rel",
        "bb_upper_rel",
        "bb_lower_rel",
    ]


def _paths(ticker: str, model_root: Path | None = None) -> tuple[Path, Path]:
    base = model_root if model_root is not None else settings.MODEL_DIR
    base.mkdir(parents=True, exist_ok=True)
    stem = artifact_stem(ticker)
    return base / f"{stem}_lstm.keras", base / f"{stem}_lstm_meta.joblib"


def lstm_model_exists(ticker: str, model_root: Path | None = None) -> bool:
    mp, jp = _paths(ticker, model_root)
    return mp.is_file() and jp.is_file()


def _min_bars_for_lstm(seq_len: int) -> int:
    """Rolling norm needs seq_len + (window-1) rows so the last window is fully normalized."""
    return seq_len + settings.LSTM_ROLLING_NORM_WINDOW - 1


def _lstm_raw_features(feat: pd.DataFrame) -> pd.DataFrame:
    """Scale-free raw features before rolling z-score (no global price level)."""
    close = feat["close"].astype(float)
    vol = feat["volume"].astype(float)
    safe = close.replace(0.0, np.nan)
    out = pd.DataFrame(index=feat.index)
    out["ret_close"] = close.pct_change().fillna(0.0)
    rv = vol.pct_change()
    out["ret_volume"] = rv.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
    out["rsi_14"] = feat["rsi_14"].astype(float) / 100.0
    out["macd_rel"] = feat["macd"].astype(float) / safe
    out["macd_signal_rel"] = feat["macd_signal"].astype(float) / safe
    out["bb_upper_rel"] = (feat["bb_upper"].astype(float) - close) / safe
    out["bb_lower_rel"] = (feat["bb_lower"].astype(float) - close) / safe
    return out.replace([np.inf, -np.inf], np.nan)


def _rolling_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-row z-score using a trailing window (causal; first window-1 rows are NaN)."""
    m = df.rolling(window=window, min_periods=window).mean()
    s = df.rolling(window=window, min_periods=window).std()
    s = s.replace(0.0, np.nan)
    z = (df - m) / (s + 1e-8)
    return z.clip(-8.0, 8.0)


def _prepare_lstm_input(
    feat: pd.DataFrame,
    *,
    causal: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Rolling z-score on LSTM raw features.

    When causal=True (default for inference and training), we only ffill then fill leading
    NaNs with 0 — no bfill, so future rows never influence past normalization inputs.
    """
    raw = _lstm_raw_features(feat)
    if causal:
        raw = raw.ffill().fillna(0.0)
    else:
        # Legacy path: avoid for walk-forward evaluation (leaks future into past NaNs).
        raw = raw.ffill().bfill()
    w = settings.LSTM_ROLLING_NORM_WINDOW
    norm = _rolling_zscore(raw, w)
    close = feat["close"].astype(float)
    return norm, close


def _build_xy(
    feat_df: pd.DataFrame,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide windows over rolling-normalized features; y = cumulative simple returns vs reference close."""
    cols = lstm_feature_columns()
    # Causal normalization so training matches honest backtest / live prediction (no bfill leakage).
    norm, close = _prepare_lstm_input(feat_df, causal=True)
    missing = [c for c in cols if c not in norm.columns]
    if missing:
        raise ValueError(f"Feature frame missing columns: {missing}")

    close_arr = close.to_numpy(dtype=np.float64)
    mat = norm[cols].to_numpy(dtype=np.float64)
    n = len(feat_df)
    w = settings.LSTM_ROLLING_NORM_WINDOW
    min_i = max(0, w - 1)

    X_list: list[np.ndarray] = []
    y_list: list[list[float]] = []

    for i in range(min_i, n - seq_len - 90):
        window = mat[i : i + seq_len]
        if np.isnan(window).any():
            continue
        t = i + seq_len - 1
        ref = close_arr[t]
        if not np.isfinite(ref) or ref == 0.0:
            continue
        targets = [(close_arr[t + h] / ref - 1.0) for h in _HORIZON_DAYS]
        if any(not np.isfinite(x) for x in targets):
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
    # Match inference: no backward-fill on indicators (avoids peeking at future for early NaNs).
    feat = add_indicators(df).ffill().fillna(0.0)
    seq_len = settings.SEQUENCE_LENGTH
    X_raw, y_raw = _build_xy(feat, seq_len)
    n_s, s_len, n_f = X_raw.shape

    split = int(n_s * 0.85)
    if split < 32 or n_s - split < 8:
        split = int(n_s * 0.8)
    X_train, X_val = X_raw[:split], X_raw[split:]
    y_train, y_val = y_raw[:split], y_raw[split:]

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
    joblib.dump(
        {
            "feature_cols": lstm_feature_columns(),
            "seq_len": seq_len,
            "rolling_window": settings.LSTM_ROLLING_NORM_WINDOW,
            "meta_version": 2,
        },
        meta_path,
    )
    logger.info("Saved LSTM for %s -> %s", ticker, model_path)
    return {"status": "ok", "model_path": str(model_path), "samples": str(n_s)}


def load_trained_lstm_bundle(ticker: str, model_root: Path | None = None):
    """Load Keras model + meta joblib after training (used by the backtester)."""
    return _load_keras_and_meta(ticker, model_root)


def _load_keras_and_meta(ticker: str, model_root: Path | None = None):
    from tensorflow import keras

    model_path, meta_path = _paths(ticker, model_root)
    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"LSTM bundle missing for {ticker}; train first.")
    model = keras.models.load_model(model_path)
    meta = joblib.load(meta_path)
    if meta.get("meta_version", 1) < 2 or "fx" in meta:
        raise ValueError(
            f"LSTM for {ticker} uses an old scaler format; retrain to use returns + rolling normalization."
        )
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
    feat = add_indicators(df).ffill().fillna(0.0)
    model, meta = _load_keras_and_meta(ticker, model_root)
    seq_len = int(meta["seq_len"])
    cols: list[str] = meta["feature_cols"]
    need = _min_bars_for_lstm(seq_len)
    if len(feat) < need:
        raise ValueError(
            f"Not enough rows ({len(feat)}) for LSTM; need at least {need} "
            "(sequence length + rolling normalization window − 1)."
        )

    norm, _close_ser = _prepare_lstm_input(feat, causal=True)
    window = norm.iloc[-seq_len:][cols].to_numpy(dtype=np.float32)
    if np.isnan(window).any():
        raise ValueError("Latest window still contains NaNs; wait for more history.")

    ref_close = float(feat["close"].iloc[-1])
    w = window.reshape(1, seq_len, window.shape[1])
    pred_ret = model.predict(w, verbose=0)[0]
    return {
        "7": ref_close * (1.0 + float(pred_ret[1])),
        "30": ref_close * (1.0 + float(pred_ret[2])),
        "90": ref_close * (1.0 + float(pred_ret[3])),
    }


def predict_lstm_one_step_with_model(
    model,
    meta: dict,
    feat_df: pd.DataFrame,
) -> float:
    """
    Next-day (+1) close using an already-loaded Keras model (backtest rollouts call this many times).
    """
    seq_len = int(meta["seq_len"])
    cols: list[str] = meta["feature_cols"]
    norm, _ = _prepare_lstm_input(feat_df, causal=True)
    if len(feat_df) < _min_bars_for_lstm(seq_len):
        return float("nan")
    tail = norm.iloc[-seq_len:][cols].to_numpy(dtype=np.float32)
    if tail.shape[0] < seq_len or np.isnan(tail).any():
        return float("nan")
    ref_close = float(feat_df["close"].astype(float).iloc[-1])
    w = tail.reshape(1, seq_len, tail.shape[1])
    pred_ret = model.predict(w, verbose=0)[0]
    return ref_close * (1.0 + float(pred_ret[0]))


def predict_lstm_head_prices_with_model(
    model,
    meta: dict,
    feat_df: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """
    Point estimates for +1, +7, +30, +90 closes (teacher-forced inputs) from the last window.
    Used for Scenario 4 direction accuracy vs multi-horizon actual moves.
    """
    seq_len = int(meta["seq_len"])
    cols: list[str] = meta["feature_cols"]
    norm, _ = _prepare_lstm_input(feat_df, causal=True)
    if len(feat_df) < _min_bars_for_lstm(seq_len):
        return (float("nan"),) * 4
    tail = norm.iloc[-seq_len:][cols].to_numpy(dtype=np.float32)
    if tail.shape[0] < seq_len or np.isnan(tail).any():
        return (float("nan"),) * 4
    ref_close = float(feat_df["close"].astype(float).iloc[-1])
    w = tail.reshape(1, seq_len, tail.shape[1])
    pred_ret = model.predict(w, verbose=0)[0]
    return tuple(ref_close * (1.0 + float(pred_ret[j])) for j in range(4))


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
    return predict_lstm_one_step_with_model(model, meta, feat_df)
