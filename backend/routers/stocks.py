"""
Stock list, OHLCV history, indicators, ensemble predictions, and backtests.
"""

from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import settings
from models.database import get_db
from services.backtester import run_backtest
from services.backtest_storage import save_single_run
from services.data_fetcher import get_latest_bar_map, get_meta, get_ohlcv_dataframe
from services.ensemble import ensemble_forecast
from services.indicators import add_indicators
from services.lstm_model import lstm_model_exists
from utils.nasdaq100_tickers import all_tracked_symbols
from utils.ticker_validate import require_tracked_ticker

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stocks"])


class StockListItem(BaseModel):
    ticker: str
    last_close: float | None = None
    last_trade_date: str | None = None
    currency: str = "USD"


class StockListResponse(BaseModel):
    stocks: list[StockListItem]
    count: int
    light_mode: bool
    last_price_refresh_utc: str | None = None


class OhlcvRow(BaseModel):
    date: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None


class HistoryResponse(BaseModel):
    ticker: str
    bars: list[OhlcvRow]


class IndicatorRow(BaseModel):
    date: str
    close: float | None
    rsi_14: float | None
    macd: float | None
    macd_signal: float | None
    macd_hist: float | None
    bb_middle: float | None
    bb_upper: float | None
    bb_lower: float | None


class IndicatorsResponse(BaseModel):
    ticker: str
    rows: list[IndicatorRow]


@router.get("/stocks", response_model=StockListResponse)
def list_stocks(db: Session = Depends(get_db)) -> StockListResponse:
    latest = get_latest_bar_map(db)
    items: list[StockListItem] = []
    for ticker in all_tracked_symbols():
        bar = latest.get(ticker)
        items.append(
            StockListItem(
                ticker=ticker,
                last_close=float(bar.close) if bar and bar.close is not None else None,
                last_trade_date=bar.trade_date.isoformat() if bar else None,
            )
        )
    return StockListResponse(
        stocks=items,
        count=len(items),
        light_mode=settings.LIGHT_MODE,
        last_price_refresh_utc=get_meta(db, "last_price_refresh_utc"),
    )


@router.get("/stocks/{ticker}/history", response_model=HistoryResponse)
def stock_history(
    ticker: str,
    db: Session = Depends(get_db),
    limit: int = Query(2000, ge=50, le=5000, description="Most recent bars to return."),
) -> HistoryResponse:
    t = require_tracked_ticker(ticker)
    df = get_ohlcv_dataframe(db, t)
    if limit < len(df):
        df = df.iloc[-limit:]
    bars = [
        OhlcvRow(
            date=idx.isoformat()[:10],
            open=float(r["open"]) if r["open"] is not None else None,
            high=float(r["high"]) if r["high"] is not None else None,
            low=float(r["low"]) if r["low"] is not None else None,
            close=float(r["close"]) if r["close"] is not None else None,
            volume=float(r["volume"]) if r["volume"] is not None else None,
        )
        for idx, r in df.iterrows()
    ]
    return HistoryResponse(ticker=t, bars=bars)


@router.get("/stocks/{ticker}/indicators", response_model=IndicatorsResponse)
def stock_indicators(
    ticker: str,
    db: Session = Depends(get_db),
    limit: int = Query(800, ge=50, le=3000),
) -> IndicatorsResponse:
    t = require_tracked_ticker(ticker)
    df = get_ohlcv_dataframe(db, t)
    feat = add_indicators(df).ffill().bfill()
    if limit < len(feat):
        feat = feat.iloc[-limit:]
    rows = [
        IndicatorRow(
            date=idx.isoformat()[:10],
            close=float(r["close"]) if pd.notna(r.get("close")) else None,
            rsi_14=float(r["rsi_14"]) if pd.notna(r.get("rsi_14")) else None,
            macd=float(r["macd"]) if pd.notna(r.get("macd")) else None,
            macd_signal=float(r["macd_signal"]) if pd.notna(r.get("macd_signal")) else None,
            macd_hist=float(r["macd_hist"]) if pd.notna(r.get("macd_hist")) else None,
            bb_middle=float(r["bb_middle"]) if pd.notna(r.get("bb_middle")) else None,
            bb_upper=float(r["bb_upper"]) if pd.notna(r.get("bb_upper")) else None,
            bb_lower=float(r["bb_lower"]) if pd.notna(r.get("bb_lower")) else None,
        )
        for idx, r in feat.iterrows()
    ]
    return IndicatorsResponse(ticker=t, rows=rows)


@router.get("/stocks/{ticker}/prediction")
def stock_prediction(ticker: str, db: Session = Depends(get_db)) -> dict:
    t = require_tracked_ticker(ticker)
    if not lstm_model_exists(t):
        raise HTTPException(
            status_code=400,
            detail="No LSTM checkpoint for this ticker yet. Run POST /api/admin/train-models first.",
        )
    try:
        return ensemble_forecast(db, t)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@router.get("/stocks/{ticker}/backtest")
def stock_backtest(
    ticker: str,
    scenario: int = Query(
        1,
        ge=1,
        le=5,
        description="1=daily, 2=multi-step honest, 3=stress, 4=direction, 5=combined (one LSTM train)",
    ),
    years: float | None = Query(
        None,
        ge=1,
        le=80,
        description="Holdout length in years (min 1 trading-year); omit for BACKTEST_YEARS default. Silently capped by data.",
    ),
    max_holdout: bool = Query(
        False,
        description="If true, use the longest holdout allowed by cached bars (ignores years).",
    ),
    persist: bool = Query(
        True,
        description="If true, save this backtest result to SQLite for history and charts.",
    ),
    db: Session = Depends(get_db),
) -> dict:
    t = require_tracked_ticker(ticker)
    try:
        out = run_backtest(
            db,
            t,
            scenario=scenario,
            years=years,
            max_holdout=max_holdout,
        )
        saved_run_id: int | None = None
        if persist:
            try:
                yreq = out.get("holdout_years_requested")
                yact = out.get("holdout_years_actual")
                years_for_save = (
                    float(years)
                    if years is not None
                    else (float(yreq) if yreq is not None else None)
                )
                saved_run_id = save_single_run(
                    db,
                    scenario=scenario,
                    years=years_for_save,
                    max_holdout=max_holdout,
                    holdout_years_requested=float(yreq) if yreq is not None else None,
                    holdout_years_actual=float(yact) if yact is not None else None,
                    ticker=t,
                    response=out,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to persist single backtest: %s", exc)
        out["saved_run_id"] = saved_run_id
        return out
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}") from exc
