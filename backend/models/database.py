"""
SQLite persistence layer using SQLAlchemy.

We store one row per ticker per trading day (OHLCV). The same database will later
hold model outputs; keeping prices in normalized rows makes charts and ML prep easy.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Generator

from sqlalchemy import Date, DateTime, Float, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, Session

from config import settings


class Base(DeclarativeBase):
    """Base class for all ORM tables."""

    pass


class PriceBar(Base):
    """
    Daily OHLCV bar for a ticker or the index (^NDX).

    Yahoo Finance may omit rows on non-trading days; we only store dates Yahoo returns.
    """

    __tablename__ = "price_bars"
    __table_args__ = (UniqueConstraint("ticker", "trade_date", name="uq_price_ticker_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    # When this row was last written (useful for debugging refresh issues)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class AppMeta(Base):
    """Simple key/value store for app-wide flags (e.g. last bulk refresh time)."""

    __tablename__ = "app_meta"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(256), nullable=False)


_engine = None
_SessionLocal = None


def get_engine():
    """Return a singleton SQLAlchemy engine (SQLite)."""
    global _engine
    if _engine is None:
        # SQLite + FastAPI: allow connections from scheduler / other threads
        connect_args = {}
        if settings.DATABASE_URL.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
    return _engine


def get_session_factory():
    """Return a session factory bound to our engine."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, expire_on_commit=False)
    return _SessionLocal


def init_db() -> None:
    """Create database tables if they do not exist yet."""
    Base.metadata.create_all(bind=get_engine())


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency: yields one request-scoped session and always closes it.

    Usage:
        def route(db: Session = Depends(get_db)):
            ...
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
