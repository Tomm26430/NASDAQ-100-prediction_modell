"""
LSTM forecaster: sequence input (60 days × N features; N=7 without macro, N=12 with USE_MACRO_FEATURES),
four heads for cumulative simple returns at +1, +7, +30, +90 trading days vs the reference bar.

Training targets are percentage returns: (close[t+h] / close[t]) - 1. At inference,
predicted_price = anchor_close * (1 + predicted_return), where anchor_close is the last row's
close in the window (real or synthetic during rollouts).

Input features are return-based columns scaled with sklearn MinMaxScaler (fit on the training
split only, then saved in the meta joblib). Artifacts live under `saved_models/<ticker>/lstm.keras`
(and `lstm_meta.joblib`), or under `<temp>/<ticker>/` during backtests.
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
from services.data_fetcher import get_macro_dataframe, get_ohlcv_dataframe
from services.indicators import add_indicators, macro_feature_columns, merge_macro_features
from utils.ml_paths import artifact_stem

logger = logging.getLogger(__name__)

# Model output order: [t+1, t+7, t+30, t+90] as cumulative simple returns vs reference close.
_HORIZON_DAYS = (1, 7, 30, 90)

# Saved checkpoints with meta_version below this must be retrained (input scaling / heads changed).
# v4: optional macro columns (12 features when USE_MACRO_FEATURES).
_META_VERSION = 4


def lstm_feature_columns() -> list[str]:
    """Column order fed into the LSTM after MinMax scaling (7 base + 5 macro when enabled)."""
    base = [
        "ret_close",
        "ret_volume",
        "rsi_14",
        "macd_rel",
        "macd_signal_rel",
        "bb_upper_rel",
        "bb_lower_rel",
    ]
    return base + macro_feature_columns()


def log_lstm_feature_summary() -> None:
    """Call once at app startup to record the active LSTM input width and names."""
    cols = lstm_feature_columns()
    logger.info("LSTM input features (%d): %s", len(cols), cols)


def min_ohlcv_rows_for_lstm_window(seq_len: int) -> int:
    """Minimum rows in an indicator-enriched frame needed to build one LSTM window (no rolling z warmup)."""
    return int(seq_len)


def _paths(ticker: str, model_root: Path | None = None) -> tuple[Path, Path]:
    """Paths under MODEL_DIR/<ticker_dir>/lstm.keras and lstm_meta.joblib."""
    base = model_root if model_root is not None else settings.MODEL_DIR
    sub = base / artifact_stem(ticker)
    return sub / "lstm.keras", sub / "lstm_meta.joblib"


def lstm_model_exists(ticker: str, model_root: Path | None = None) -> bool:
    mp, jp = _paths(ticker, model_root)
    return mp.is_file() and jp.is_file()


def _lstm_raw_features(feat: pd.DataFrame) -> pd.DataFrame:
    """Scale-free raw features before MinMax (no global price level in columns)."""
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
    if settings.USE_MACRO_FEATURES:
        for col in macro_feature_columns():
            out[col] = feat[col].astype(float) if col in feat.columns else 0.0
    return out.replace([np.inf, -np.inf], np.nan)


def _lstm_feature_matrix(feat: pd.DataFrame) -> pd.DataFrame:
    """Per-row feature matrix for windows (causal ffill; no backward fill)."""
    raw = _lstm_raw_features(feat)
    raw = raw.ffill().fillna(0.0)
    cols = lstm_feature_columns()
    return raw[cols]


def _build_xy(
    feat_df: pd.DataFrame,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide windows over raw scaled features in training; y = cumulative simple returns vs reference close."""
    cols = lstm_feature_columns()
    mat = _lstm_feature_matrix(feat_df).to_numpy(dtype=np.float64)
    close = feat_df["close"].astype(float).to_numpy(dtype=np.float64)
    n = len(feat_df)
    min_i = 0

    X_list: list[np.ndarray] = []
    y_list: list[list[float]] = []

    for i in range(min_i, n - seq_len - 90):
        window = mat[i : i + seq_len]
        if np.isnan(window).any():
            continue
        t = i + seq_len - 1
        ref = close[t]
        if not np.isfinite(ref) or ref == 0.0:
            continue
        targets = [(close[t + h] / ref - 1.0) for h in _HORIZON_DAYS]
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
    feat = add_indicators(df).ffill().fillna(0.0)
    if settings.USE_MACRO_FEATURES:
        macro = get_macro_dataframe(session)
        if macro is not None and not macro.empty:
            cutoff = df.index.max()
            macro = macro.loc[macro.index <= cutoff]
            feat = merge_macro_features(feat, macro)
    feat = feat.fillna(0.0)
    seq_len = settings.SEQUENCE_LENGTH
    X_raw, y_raw = _build_xy(feat, seq_len)
    n_s, s_len, n_f = X_raw.shape

    split = int(n_s * 0.85)
    if split < 32 or n_s - split < 8:
        split = int(n_s * 0.8)
    X_train_raw, X_val_raw = X_raw[:split], X_raw[split:]
    y_train, y_val = y_raw[:split], y_raw[split:]

    # Fit MinMaxScaler on training windows only (avoids validation-set leakage into input scale).
    fx = MinMaxScaler()
    X_train_flat = X_train_raw.reshape(-1, n_f)
    X_val_flat = X_val_raw.reshape(-1, n_f)
    X_train = fx.fit_transform(X_train_flat).reshape(-1, s_len, n_f)
    X_val = fx.transform(X_val_flat).reshape(-1, s_len, n_f)

    keras.utils.set_random_seed(42)
    inp = layers.Input(shape=(seq_len, n_f))
    x = layers.LSTM(settings.LSTM_UNITS, return_sequences=True)(inp)
    x = layers.Dropout(settings.LSTM_DROPOUT)(x)
    x = layers.LSTM(settings.LSTM_UNITS)(x)
    x = layers.Dropout(settings.LSTM_DROPOUT)(x)
    out = layers.Dense(4)(x)
    model = keras.Model(inp, out)
    lr = float(settings.LSTM_LEARNING_RATE)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")

    es = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=int(settings.LSTM_EARLY_STOPPING_PATIENCE),
        restore_best_weights=True,
    )
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
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    joblib.dump(
        {
            "fx": fx,
            "feature_cols": lstm_feature_columns(),
            "seq_len": seq_len,
            "meta_version": _META_VERSION,
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
    if int(meta.get("meta_version", 0)) < _META_VERSION or "fx" not in meta:
        raise ValueError(
            f"LSTM for {ticker} is an old checkpoint (meta_version {meta.get('meta_version')}); "
            "retrain after return targets + MinMax input pipeline change."
        )
    return model, meta


def _scaled_window_tensor(feat_df: pd.DataFrame, meta: dict) -> tuple[np.ndarray, float] | None:
    """Last seq_len rows, MinMax-transformed; returns (batch, seq, n_f) and anchor close."""
    seq_len = int(meta["seq_len"])
    cols: list[str] = meta["feature_cols"]
    mat = _lstm_feature_matrix(feat_df)
    if len(mat) < seq_len:
        return None
    tail = mat.iloc[-seq_len:][cols].to_numpy(dtype=np.float64)
    if np.isnan(tail).any():
        return None
    fx: MinMaxScaler = meta["fx"]
    w = fx.transform(tail.reshape(-1, tail.shape[1])).reshape(1, seq_len, tail.shape[1]).astype(np.float32)
    ref_close = float(feat_df["close"].astype(float).iloc[-1])
    return w, ref_close


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
    if settings.USE_MACRO_FEATURES:
        macro = get_macro_dataframe(session)
        if macro is not None and not macro.empty:
            feat = merge_macro_features(feat, macro)
    feat = feat.fillna(0.0)
    model, meta = _load_keras_and_meta(ticker, model_root)
    seq_len = int(meta["seq_len"])
    need = min_ohlcv_rows_for_lstm_window(seq_len)
    if len(feat) < need:
        raise ValueError(f"Not enough rows ({len(feat)}) for LSTM; need at least {need}.")

    sw = _scaled_window_tensor(feat, meta)
    if sw is None:
        raise ValueError("Latest window still contains NaNs; wait for more history.")
    w, ref_close = sw
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
    anchor_close = last row close; predicted_close = anchor_close * (1 + predicted_return).
    """
    seq_len = int(meta["seq_len"])
    if len(feat_df) < min_ohlcv_rows_for_lstm_window(seq_len):
        return float("nan")
    sw = _scaled_window_tensor(feat_df, meta)
    if sw is None:
        return float("nan")
    w, ref_close = sw
    pred_ret = model.predict(w, verbose=0)[0]
    return ref_close * (1.0 + float(pred_ret[0]))


def predict_lstm_head_prices_with_model(
    model,
    meta: dict,
    feat_df: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """
    Point estimates for +1, +7, +30, +90 closes (teacher-forced inputs) from the last window.
    """
    seq_len = int(meta["seq_len"])
    if len(feat_df) < min_ohlcv_rows_for_lstm_window(seq_len):
        return (float("nan"),) * 4
    sw = _scaled_window_tensor(feat_df, meta)
    if sw is None:
        return (float("nan"),) * 4
    w, ref_close = sw
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
