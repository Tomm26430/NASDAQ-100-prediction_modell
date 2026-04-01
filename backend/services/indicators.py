"""
Technical indicators used for charts and as ML features.

All outputs align 1:1 with the input index; early rows contain NaNs until windows warm up.
"""

from __future__ import annotations

import pandas as pd


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


def feature_columns() -> list[str]:
    """Legacy OHLCV + indicator names (charts); LSTM uses `lstm_feature_columns` in `lstm_model`."""
    return [
        "close",
        "volume",
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_upper",
        "bb_lower",
    ]
