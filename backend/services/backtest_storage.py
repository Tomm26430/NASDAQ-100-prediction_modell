"""
Persist completed backtests to SQLite for history, drill-down charts, and comparisons.

Full per-ticker JSON (scenario 5) includes price_accuracy.series and direction_accuracy for graphs.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from sqlalchemy import delete, desc, func, select
from sqlalchemy.orm import Session

from models.database import BacktestRun, BacktestTickerResult


def sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf with None so JSON is valid and SQLite-friendly."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def dumps_payload(d: dict[str, Any]) -> str:
    return json.dumps(sanitize_for_json(d), ensure_ascii=False)


def save_bulk_run(
    session: Session,
    *,
    scenario: int,
    years: float | None,
    max_holdout: bool,
    holdout_years_requested: float | None,
    holdout_years_actual: float | None,
    aggregate: dict[str, Any],
    summary_by_ticker: dict[str, dict[str, Any]],
    full_payload_by_ticker: dict[str, dict[str, Any] | None],
) -> int:
    """
    Store one bulk run and one row per ticker.

    summary_by_ticker matches API `results` (status, verdict, mape, dir, holdout_days).
    full_payload_by_ticker holds the complete scenario-5 backtest JSON for charts (or None on error).
    """
    run = BacktestRun(
        run_type="bulk",
        scenario=scenario,
        years_requested=years,
        max_holdout=max_holdout,
        holdout_years_requested=holdout_years_requested,
        holdout_years_actual=holdout_years_actual,
        aggregate_json=dumps_payload(aggregate),
        source_ticker=None,
    )
    session.add(run)
    session.flush()

    for sym, row_summary in summary_by_ticker.items():
        status = str(row_summary.get("status", "ok"))
        verdict = row_summary.get("combined_verdict")
        mape = row_summary.get("mape_30d")
        d7 = row_summary.get("direction_accuracy_7d")
        hod = int(row_summary.get("holdout_days", 0))
        full = full_payload_by_ticker.get(sym)
        payload: dict[str, Any] | None = full if status == "ok" and isinstance(full, dict) else None
        if payload is None and status != "ok":
            payload = {"ticker": sym, "status": status}
        row = BacktestTickerResult(
            run_id=run.id,
            ticker=sym,
            status=status,
            combined_verdict=verdict if isinstance(verdict, str) else None,
            mape_30d=float(mape) if mape is not None and isinstance(mape, (int, float)) and np.isfinite(mape) else None,
            direction_accuracy_7d=float(d7) if d7 is not None and isinstance(d7, (int, float)) and np.isfinite(d7) else None,
            holdout_days=hod,
            payload_json=dumps_payload(payload) if payload else None,
        )
        session.add(row)
    session.commit()
    return run.id


def save_single_run(
    session: Session,
    *,
    scenario: int,
    years: float | None,
    max_holdout: bool,
    holdout_years_requested: float | None,
    holdout_years_actual: float | None,
    ticker: str,
    response: dict[str, Any],
) -> int:
    """Persist a single-ticker backtest; one run row + one ticker row with full payload."""
    run = BacktestRun(
        run_type="single",
        scenario=scenario,
        years_requested=years,
        max_holdout=max_holdout,
        holdout_years_requested=holdout_years_requested,
        holdout_years_actual=holdout_years_actual,
        aggregate_json=None,
        source_ticker=ticker,
    )
    session.add(run)
    session.flush()

    verdict = response.get("combined_verdict") if response.get("scenario") == 5 else None
    ens = None
    if response.get("scenario") == 5:
        pa = response.get("price_accuracy") or {}
        pm = pa.get("metrics") or {}
        ens = pm.get("ensemble") or {}
    h30 = ens.get("h30") if isinstance(ens, dict) else {}
    mape = float(h30["mape"]) if isinstance(h30, dict) and isinstance(h30.get("mape"), (int, float)) and np.isfinite(h30["mape"]) else None
    da = response.get("direction_accuracy") or {}
    d7 = float(da["direction_accuracy_7d"]) if isinstance(da, dict) and isinstance(da.get("direction_accuracy_7d"), (int, float)) else None

    row = BacktestTickerResult(
        run_id=run.id,
        ticker=ticker,
        status="ok",
        combined_verdict=verdict if isinstance(verdict, str) else None,
        mape_30d=mape,
        direction_accuracy_7d=d7,
        holdout_days=int(response.get("holdout_trading_days", 0)),
        payload_json=dumps_payload(response),
    )
    session.add(row)
    session.commit()
    return run.id


def list_runs(session: Session, limit: int = 100) -> list[dict[str, Any]]:
    stmt = select(BacktestRun).order_by(desc(BacktestRun.created_at)).limit(limit)
    rows = list(session.execute(stmt).scalars().all())
    out: list[dict[str, Any]] = []
    for r in rows:
        n = session.scalar(
            select(func.count()).select_from(BacktestTickerResult).where(BacktestTickerResult.run_id == r.id)
        )
        out.append(
            {
                "id": r.id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "run_type": r.run_type,
                "scenario": r.scenario,
                "years_requested": r.years_requested,
                "max_holdout": r.max_holdout,
                "holdout_years_requested": r.holdout_years_requested,
                "holdout_years_actual": r.holdout_years_actual,
                "source_ticker": r.source_ticker,
                "ticker_count": int(n or 0),
            }
        )
    return out


def get_run_summary(session: Session, run_id: int) -> dict[str, Any] | None:
    r = session.get(BacktestRun, run_id)
    if r is None:
        return None
    tickers = session.execute(select(BacktestTickerResult).where(BacktestTickerResult.run_id == run_id)).scalars().all()
    agg = None
    if r.aggregate_json:
        try:
            agg = json.loads(r.aggregate_json)
        except json.JSONDecodeError:
            agg = None
    return {
        "id": r.id,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "run_type": r.run_type,
        "scenario": r.scenario,
        "years_requested": r.years_requested,
        "max_holdout": r.max_holdout,
        "holdout_years_requested": r.holdout_years_requested,
        "holdout_years_actual": r.holdout_years_actual,
        "source_ticker": r.source_ticker,
        "aggregate": agg,
        "tickers": [
            {
                "ticker": t.ticker,
                "status": t.status,
                "combined_verdict": t.combined_verdict,
                "mape_30d": t.mape_30d,
                "direction_accuracy_7d": t.direction_accuracy_7d,
                "holdout_days": t.holdout_days,
                "has_detail": bool(t.payload_json),
            }
            for t in tickers
        ],
    }


def get_ticker_payload(session: Session, run_id: int, ticker: str) -> dict[str, Any] | None:
    stmt = select(BacktestTickerResult).where(
        BacktestTickerResult.run_id == run_id,
        BacktestTickerResult.ticker == ticker,
    )
    row = session.execute(stmt).scalar_one_or_none()
    if row is None or not row.payload_json:
        return None
    try:
        return json.loads(row.payload_json)
    except json.JSONDecodeError:
        return None


def compare_runs(session: Session, run_id_a: int, run_id_b: int) -> dict[str, Any] | None:
    a = get_run_summary(session, run_id_a)
    b = get_run_summary(session, run_id_b)
    if a is None or b is None:
        return None
    by_a = {x["ticker"]: x for x in a["tickers"]}
    by_b = {x["ticker"]: x for x in b["tickers"]}
    common = sorted(set(by_a) & set(by_b))
    rows: list[dict[str, Any]] = []
    for sym in common:
        ra, rb = by_a[sym], by_b[sym]
        ma = ra.get("mape_30d")
        mb = rb.get("mape_30d")
        da = ra.get("direction_accuracy_7d")
        db = rb.get("direction_accuracy_7d")
        rows.append(
            {
                "ticker": sym,
                "run_a": {"verdict": ra.get("combined_verdict"), "mape_30d": ma, "direction_accuracy_7d": da, "status": ra.get("status")},
                "run_b": {"verdict": rb.get("combined_verdict"), "mape_30d": mb, "direction_accuracy_7d": db, "status": rb.get("status")},
                "delta_mape_30d": (float(ma) - float(mb)) if ma is not None and mb is not None else None,
                "delta_direction_7d": (float(da) - float(db)) if da is not None and db is not None else None,
            }
        )
    return {
        "run_a": {k: v for k, v in a.items() if k != "tickers"},
        "run_b": {k: v for k, v in b.items() if k != "tickers"},
        "common_tickers": common,
        "comparison_rows": rows,
    }


def delete_run(session: Session, run_id: int) -> bool:
    r = session.get(BacktestRun, run_id)
    if r is None:
        return False
    session.execute(delete(BacktestTickerResult).where(BacktestTickerResult.run_id == run_id))
    session.delete(r)
    session.commit()
    return True
