"""
Read saved backtest runs from SQLite (history, drill-down, compare).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from models.database import get_db
from services.backtest_storage import compare_runs, delete_run, get_run_summary, get_ticker_payload, list_runs

router = APIRouter(prefix="/backtest-runs", tags=["backtest-runs"])


@router.get("")
def list_saved_runs(
    db: Session = Depends(get_db),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    """Return recent saved backtest runs (metadata + ticker counts)."""
    return {"runs": list_runs(db, limit=limit)}


@router.get("/compare/summary")
def compare_two_runs(
    a: int = Query(..., description="First run id"),
    b: int = Query(..., description="Second run id"),
    db: Session = Depends(get_db),
) -> dict:
    """Side-by-side metadata and per-ticker deltas for common symbols."""
    out = compare_runs(db, a, b)
    if out is None:
        raise HTTPException(status_code=404, detail="One or both runs not found.")
    return out


@router.get("/{run_id}")
def get_saved_run(run_id: int, db: Session = Depends(get_db)) -> dict:
    """Return one run with aggregate summary and per-ticker summary rows."""
    row = get_run_summary(db, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return row


@router.get("/{run_id}/ticker/{ticker:path}")
def get_saved_ticker_detail(run_id: int, ticker: str, db: Session = Depends(get_db)) -> dict:
    """
    Full stored JSON for one symbol in a run (scenario-5 shape includes price_accuracy + direction_accuracy).
    `ticker` may be URL-encoded (e.g. %5ENDX for ^NDX).
    """
    payload = get_ticker_payload(db, run_id, ticker)
    if payload is None:
        raise HTTPException(status_code=404, detail="Ticker result not found or no stored detail.")
    return payload


@router.delete("/{run_id}")
def remove_run(run_id: int, db: Session = Depends(get_db)) -> dict:
    """Delete a saved run and its ticker rows."""
    if not delete_run(db, run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    return {"ok": True, "deleted_id": run_id}
