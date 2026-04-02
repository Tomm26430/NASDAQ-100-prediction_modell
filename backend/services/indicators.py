"""
Technical indicators used for charts and as ML features.

All outputs align 1:1 with the input index; early rows contain NaNs until windows warm up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import MACRO_FEATURE_COLUMNS, settings


def add_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI(14), MACD(12,26,9), and Bollinger Bands(20, 2σ) to a copy of the frame.

    Expects columns: open, high, low, close, volume (we normalize names to lowercase).
    """
    df = ohlcv.copy()
    df.columns = [str(c).lower() for c in df.columns]
    close = df["close"].astype(float)
    vol = df["volume"].astype(float).ffill().bfill().fillna(0.0)

    # --- RSI(14): relative strength from average gains / losses ---
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window=14, min_periods=14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, float("nan"))
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # --- MACD: fast EMA minus slow EMA; signal EMA of MACD line ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = macd_line - signal_line

    # --- Bollinger Bands(20, 2 std dev) ---
    mid = close.rolling(window=20, min_periods=20).mean()
    std = close.rolling(window=20, min_periods=20).std()
    df["bb_middle"] = mid
    df["bb_upper"] = mid + 2.0 * std
    df["bb_lower"] = mid - 2.0 * std

    df["volume"] = vol
    return df


def macro_feature_columns() -> list[str]:
    """Macro column names appended to LSTM inputs when USE_MACRO_FEATURES is True."""
    return list(MACRO_FEATURE_COLUMNS) if settings.USE_MACRO_FEATURES else []


def merge_macro_features(feat_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-align macro series onto feat_df's DatetimeIndex (no lookahead: macro rows after the last
    equity date are dropped before reindex).

    Applies rolling MinMax normalization (window LSTM_ROLLING_NORM_WINDOW) per macro column, then
    forward-fills remaining NaNs with 0 after scaling.
    """
    if not settings.USE_MACRO_FEATURES or macro_df is None or macro_df.empty:
        return feat_df

    cols = list(MACRO_FEATURE_COLUMNS)
    out = feat_df.copy()
    out.index = pd.to_datetime(out.index).normalize()
    m = macro_df.copy()
    m.index = pd.to_datetime(m.index).normalize()
    end = out.index.max()
    m = m.loc[m.index <= end]
    for c in cols:
        if c not in m.columns:
            m[c] = np.nan
    macro_aligned = m.reindex(out.index).ffill()
    w = max(1, int(settings.LSTM_ROLLING_NORM_WINDOW))
    norm = pd.DataFrame(index=out.index)
    for c in cols:
        s = macro_aligned[c].astype(float)
        rmin = s.rolling(window=w, min_periods=1).min()
        rmax = s.rolling(window=w, min_periods=1).max()
        rng = (rmax - rmin).replace(0.0, np.nan)
        scaled = (s - rmin) / rng
        norm[c] = scaled.clip(0.0, 1.0).fillna(0.0)
    for c in cols:
        if c in out.columns:
            out = out.drop(columns=[c])
    return pd.concat([out, norm], axis=1)


def feature_columns() -> list[str]:
    """OHLCV + indicators (+ macro names when enabled). LSTM uses `lstm_feature_columns` in `lstm_model`."""
    base = [
        "close",
        "volume",
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_upper",
        "bb_lower",
    ]
    return base + macro_feature_columns()
