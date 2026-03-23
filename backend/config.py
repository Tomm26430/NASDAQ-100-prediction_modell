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
    HISTORY_YEARS: int = 5

    # CORS: allow the Vite dev server and typical local frontends.
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # How often the scheduler refreshes prices (hours).
    PRICE_REFRESH_INTERVAL_HOURS: int = 24

    # Disk folder for .keras LSTMs, ARIMA pickles, and sklearn scalers
    MODEL_DIR: Path = _DEFAULT_MODEL_DIR

    # LSTM training (lower for faster dev; raise for production quality)
    LSTM_EPOCHS: int = 30
    LSTM_BATCH_SIZE: int = 32
    LSTM_UNITS: int = 50
    LSTM_DROPOUT: float = 0.2
    SEQUENCE_LENGTH: int = 60

    # Walk-forward / holdout length (~1 trading year)
    BACKTEST_TRADING_DAYS: int = 252

    # Ensemble weights (LSTM, ARIMA)
    ENSEMBLE_WEIGHT_LSTM: float = 0.6
    ENSEMBLE_WEIGHT_ARIMA: float = 0.4


settings = Settings()
