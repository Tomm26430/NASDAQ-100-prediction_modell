"""
Central configuration for the NASDAQ predictor backend.

Values can be overridden with environment variables (same names, uppercase),
for example: LIGHT_MODE=false
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Directory where this file lives (backend/)
_BACKEND_DIR = Path(__file__).resolve().parent
# Default SQLite file next to the app — easy to find and back up
_DEFAULT_DB = _BACKEND_DIR / "nasdaq_predictor.db"
_DEFAULT_MODEL_DIR = _BACKEND_DIR / "saved_models"


class Settings(BaseSettings):
    """Application settings loaded from the environment or defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # When True, only a handful of symbols are fetched/trained (fast local testing).
    LIGHT_MODE: bool = True

    # Full path to the SQLite database file.
    DATABASE_URL: str = f"sqlite:///{_DEFAULT_DB}"

    # How many years of daily bars to request from Yahoo Finance.
    # Use at least BACKTEST_YEARS + ~2 so the backtest has enough pre-holdout history for training.
    HISTORY_YEARS: int = 12

    # CORS: allow the Vite dev server and typical local frontends.
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # How often the scheduler refreshes prices (hours).
    PRICE_REFRESH_INTERVAL_HOURS: int = 24

    # Disk folder for .keras LSTMs, ARIMA pickles, and sklearn scalers
    MODEL_DIR: Path = _DEFAULT_MODEL_DIR

    # LSTM training (lower for faster dev; raise for production quality)
    LSTM_EPOCHS: int = 75
    LSTM_EARLY_STOPPING_PATIENCE: int = 10
    LSTM_BATCH_SIZE: int = 32
    LSTM_UNITS: int = 100
    LSTM_DROPOUT: float = 0.25
    LSTM_LEARNING_RATE: float = 0.0005
    SEQUENCE_LENGTH: int = 60

    # Deprecated: inputs now use MinMaxScaler fit on the training split (kept for env compatibility only).
    LSTM_ROLLING_NORM_WINDOW: int = 252

    # Walk-forward holdout length in **trading years** (~252 sessions/year).
    # 10 years ≈ 2520 trading days in the test window (capped by available data).
    BACKTEST_YEARS: int = 10

    # Minimum daily rows *before* the holdout so the temporary backtest LSTM can train
    # (needs hundreds of sliding windows; must be < total bars minus holdout).
    BACKTEST_MIN_PREHOLDOUT_ROWS: int = 650

    # Scenario 2/3: spacing between rollout anchors (trading days) to limit runtime (90 LSTM steps each).
    BACKTEST_SCENARIO2_ANCHOR_STRIDE: int = 10

    # Scenario 4: only compute direction (3× ARIMA refits + LSTM heads) every N anchor days — full daily series still runs each day.
    BACKTEST_SCENARIO4_DIRECTION_STRIDE: int = 10

    # Primary horizon (trading days) for Scenario 2/3 chart `predicted` vs actual (7/30/90 metrics still computed).
    # Must be one of 7, 30, 90 so it matches a rollout head.
    BACKTEST_MULTI_STEP_CHART_HORIZON: int = 30

    # Ensemble weights (LSTM, ARIMA)
    ENSEMBLE_WEIGHT_LSTM: float = 0.6
    ENSEMBLE_WEIGHT_ARIMA: float = 0.4


settings = Settings()
