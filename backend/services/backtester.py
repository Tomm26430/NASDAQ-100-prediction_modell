"""
Walk-forward backtests with selectable scenario modes (honest daily vs multi-step rollout,
stress windows, direction accuracy). ARIMA walk-forward helpers stay aligned with arima_model patterns;
core ARIMA fitting logic is not modified in arima_model.py — only new local forecast helpers here.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sqlalchemy.orm import Session

from config import settings
from services.arima_model import arima_walk_one_step
from services.data_fetcher import get_ohlcv_dataframe
from services.indicators import add_indicators
from services.lstm_model import (
    load_trained_lstm_bundle,
    min_ohlcv_rows_for_lstm_window,
    predict_lstm_head_prices_with_model,
    predict_lstm_one_step_with_model,
    train_lstm_for_ticker,
)
from utils.nasdaq100_tickers import get_active_tickers

logger = logging.getLogger(__name__)

# Optional UI progress for bulk runs (see services.backtest_status).
def _bulk_progress_tick(sym: str, index_1based: int, total: int) -> None:
    try:
        from services.backtest_status import is_backtest_running, set_current_ticker

        if is_backtest_running():
            set_current_ticker(sym, index_1based, total)
    except Exception:  # noqa: BLE001
        pass


def _bulk_progress_done_one() -> None:
    try:
        from services.backtest_status import is_backtest_running, record_ticker_finished

        if is_backtest_running():
            record_ticker_finished()
    except Exception:  # noqa: BLE001
        pass

# Human-readable labels returned in JSON for each scenario id.
SCENARIO_LABELS: dict[int, str] = {
    1: "Daily prediction accuracy",
    2: "Forward-looking prediction (no real data)",
    3: "Performance during high volatility",
    4: "Did the model predict up/down correctly?",
    5: "Combined Honest Assessment",
}


def _lstm_min_input_rows(seq_len: int) -> int:
    """Minimum history length for one LSTM window (MinMax inputs; no rolling z warmup)."""
    return min_ohlcv_rows_for_lstm_window(seq_len)


def _norm_w() -> tuple[float, float]:
    wl = settings.ENSEMBLE_WEIGHT_LSTM
    wa = settings.ENSEMBLE_WEIGHT_ARIMA
    s = wl + wa
    if s <= 0:
        return 0.5, 0.5
    return wl / s, wa / s


def _metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    if len(y_true) < 5 or len(y_pred) < 5:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n": 0.0}
    a = np.array(y_true, dtype=float)
    p = np.array(y_pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    if mask.sum() < 5:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n": float(mask.sum())}
    a, p = a[mask], p[mask]
    err = p - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err / np.maximum(np.abs(a), 1e-6))) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape": mape, "n": float(len(a))}


def _direction_accuracy(ref: list[float], actual: list[float], pred: list[float]) -> float:
    """Share of times sign(actual - ref) == sign(pred - ref), ignoring non-finite triples."""
    ok = 0
    tot = 0
    for r, a, p in zip(ref, actual, pred, strict=False):
        if not (np.isfinite(r) and np.isfinite(a) and np.isfinite(p)):
            continue
        if abs(a - r) < 1e-12 and abs(p - r) < 1e-12:
            ok += 1
            tot += 1
            continue
        if abs(a - r) < 1e-12 or abs(p - r) < 1e-12:
            tot += 1
            if abs(a - r) < 1e-12 and abs(p - r) < 1e-12:
                ok += 1
            continue
        tot += 1
        if np.sign(a - r) == np.sign(p - r):
            ok += 1
    return float(ok / tot) if tot > 0 else float("nan")


def _holdout_slice(
    n: int,
    *,
    years: float | None = None,
    max_holdout: bool = False,
) -> tuple[int, int, int, float | None, float, bool]:
    """
    Walk-forward holdout length in trading days; silently capped by available history.

    Returns test_start, hold, requested_trading_days (before cap), holdout_years_requested,
    holdout_years_actual, max_holdout.
    """
    min_pre = settings.BACKTEST_MIN_PREHOLDOUT_ROWS
    max_hold_for_lstm = max(0, n - min_pre)
    max_hold_for_data = max(0, n - settings.SEQUENCE_LENGTH - 100)

    if max_holdout:
        requested_trading_days = min(max_hold_for_lstm, max_hold_for_data)
        holdout_years_requested: float | None = None
        hold = requested_trading_days
    elif years is not None:
        y = max(1.0, float(years))
        requested_trading_days = int(y * 252)
        holdout_years_requested = y
        hold = min(requested_trading_days, max_hold_for_lstm, max_hold_for_data)
    else:
        y = float(settings.BACKTEST_YEARS)
        requested_trading_days = int(y * 252)
        holdout_years_requested = y
        hold = min(requested_trading_days, max_hold_for_lstm, max_hold_for_data)

    if hold < 30:
        raise ValueError(
            f"Not enough history ({n} rows). Need at least ~{min_pre + 30} daily bars "
            "(refresh prices with enough HISTORY_YEARS) or lower BACKTEST_MIN_PREHOLDOUT_ROWS."
        )
    test_start = n - hold
    holdout_years_actual = hold / 252.0
    return test_start, hold, requested_trading_days, holdout_years_requested, holdout_years_actual, max_holdout


def _holdout_note(
    hold: int,
    requested_trading_days: int,
    min_pre: int,
    *,
    max_holdout: bool,
) -> str | None:
    if max_holdout:
        return None
    if hold < requested_trading_days:
        return (
            f"Holdout reduced from {requested_trading_days} to {hold} trading days so at least "
            f"{min_pre} rows remain before the test window for LSTM training. "
            "Fetch more history (HISTORY_YEARS) or shorten the years parameter."
        )
    return None


def _indicator_frame_for_rollout(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Indicators on a growing OHLCV frame; forward-fill only (no bfill) before LSTM MinMax-scaled features."""
    feat = add_indicators(ohlcv).ffill()
    return feat.fillna(0.0)


def _append_synthetic_close_row(df: pd.DataFrame, pred_close: float) -> pd.DataFrame:
    """Append one business day with OHLC pinned to the predicted close (volume carried forward)."""
    last_vol = float(df["volume"].iloc[-1])
    next_idx = df.index[-1] + pd.offsets.BDay(1)
    new_row = pd.DataFrame(
        {
            "open": [pred_close],
            "high": [pred_close],
            "low": [pred_close],
            "close": [pred_close],
            "volume": [last_vol],
        },
        index=[next_idx],
    )
    return pd.concat([df, new_row])


def _lstm_rollout_close(
    model,
    meta: dict,
    df_upto_anchor: pd.DataFrame,
    steps: int,
) -> float:
    """
    Multi-step honest path: each new bar uses the LSTM's previous predicted close as input,
    not realized prices, for `steps` forward trading days from the last row of df_upto_anchor.
    """
    out = _lstm_rollout_closes_at_steps(model, meta, df_upto_anchor, (steps,))
    return float(out.get(steps, float("nan")))


def _lstm_rollout_closes_at_steps(
    model,
    meta: dict,
    df_upto_anchor: pd.DataFrame,
    milestones: tuple[int, ...],
) -> dict[int, float]:
    """
    One chained rollout through max(milestones) days; records synthetic close after each milestone step.
    Avoids repeating the same prefix rollout for 7/30/90 separately.
    """
    if not milestones:
        return {}
    max_h = max(milestones)
    want = set(milestones)
    work = df_upto_anchor.copy()
    recorded: dict[int, float] = {}
    for step in range(1, max_h + 1):
        feat = _indicator_frame_for_rollout(work)
        pred = predict_lstm_one_step_with_model(model, meta, feat)
        if not np.isfinite(pred):
            return {h: float("nan") for h in milestones}
        work = _append_synthetic_close_row(work, pred)
        if step in want:
            recorded[step] = float(pred)
    return recorded


def _arima_price_h_ahead(train_close: np.ndarray, h: int) -> float:
    """
    One-shot ARIMA forecast h trading days ahead from prices ending at the anchor (log space).
    Separate from arima_model.py to avoid changing that module; parameters mirror walk-forward fits.
    """
    if len(train_close) < 120 or h < 1:
        return float("nan")
    log_train = np.log(np.maximum(train_close.astype(float), 1e-6))
    m = auto_arima(
        log_train,
        seasonal=False,
        suppress_warnings=True,
        error_action="ignore",
        stepwise=True,
        max_p=3,
        max_q=3,
        max_d=2,
        n_fits=15,
    )
    fc = m.predict(n_periods=h)
    return float(np.exp(float(fc[-1])))


def _stress_trading_date_set(index: pd.DatetimeIndex, close: np.ndarray) -> set[str]:
    """
    Trading dates inside COVID crash, 2022 selloff, or any 30-trading-day window with |return| > 15%.
    """
    out: set[str] = set()
    covid_lo = pd.Timestamp("2020-02-01")
    covid_hi = pd.Timestamp("2020-05-31")
    y22_lo = pd.Timestamp("2022-01-01")
    y22_hi = pd.Timestamp("2022-12-31")
    for i, ts in enumerate(index):
        tsn = pd.Timestamp(ts).normalize()
        if covid_lo <= tsn <= covid_hi or y22_lo <= tsn <= y22_hi:
            out.add(tsn.strftime("%Y-%m-%d"))
    for e in range(30, len(close)):
        s = e - 30
        den = close[s]
        if den is None or not np.isfinite(den) or den <= 0:
            continue
        mv = abs(float(close[e]) / float(den) - 1.0)
        if mv > 0.15:
            for j in range(s, e + 1):
                out.add(pd.Timestamp(index[j]).strftime("%Y-%m-%d"))
    return out


def _build_unified_response(
    *,
    ticker: str,
    scenario: int,
    hold: int,
    target_hold: int,
    holdout_note: str | None,
    metrics: dict,
    series: list[dict],
    holdout_years_requested: float | None,
    holdout_years_actual: float,
) -> dict:
    out = {
        "ticker": ticker,
        "scenario": scenario,
        "scenario_label": SCENARIO_LABELS[scenario],
        "holdout_trading_days": hold,
        "holdout_trading_days_target": target_hold,
        "holdout_note": holdout_note,
        "holdout_years_requested": holdout_years_requested,
        "holdout_years_actual": holdout_years_actual,
        "metrics": metrics,
        "series": series,
    }
    return out


def _attach_holdout_years(base: dict, years_req: float | None, years_actual: float) -> dict:
    """Merge standard holdout year fields into a custom response dict (e.g. Scenario 5)."""
    base["holdout_years_requested"] = years_req
    base["holdout_years_actual"] = years_actual
    return base


def _scenario_2_price_metrics_and_series(
    model,
    meta: dict,
    df: pd.DataFrame,
    n: int,
    test_start: int,
    close_all: np.ndarray,
    wl: float,
    wa: float,
    arima_one_m: dict,
    stress_dates: set[str] | None,
) -> tuple[dict, list[dict], str | None]:
    """
    Scenario 2/3 price path using an already-trained LSTM (honest multi-step rollout).
    Returns (metrics dict, chart series, lstm_error_message).
    """
    chart_h = int(settings.BACKTEST_MULTI_STEP_CHART_HORIZON)
    if chart_h not in (7, 30, 90):
        chart_h = 30
    stride = max(1, int(settings.BACKTEST_SCENARIO2_ANCHOR_STRIDE))
    horizons = (7, 30, 90)
    lstm_err: str | None = None
    series: list[dict] = []
    buckets: dict[int, dict[str, list[float]]] = {
        h: {"act_l": [], "lstm": [], "act_a": [], "arima": [], "act_e": [], "ens": []} for h in horizons
    }

    try:
        max_h = max(horizons)
        for anchor in range(test_start, n - max_h, stride):
            prefix = df.iloc[: anchor + 1]
            train_c = close_all[: anchor + 1]
            need_roll = tuple(sorted(set(horizons) | {chart_h}))
            lstm_by_h = _lstm_rollout_closes_at_steps(model, meta, prefix, need_roll)
            for h in horizons:
                lstm_p = float(lstm_by_h.get(h, float("nan")))
                arima_p = _arima_price_h_ahead(train_c, h)
                act = float(close_all[anchor + h])
                ens = wl * lstm_p + wa * arima_p if np.isfinite(lstm_p) and np.isfinite(arima_p) else float("nan")
                if np.isfinite(lstm_p) and np.isfinite(act):
                    buckets[h]["act_l"].append(act)
                    buckets[h]["lstm"].append(lstm_p)
                if np.isfinite(arima_p) and np.isfinite(act):
                    buckets[h]["act_a"].append(act)
                    buckets[h]["arima"].append(arima_p)
                if np.isfinite(ens) and np.isfinite(act):
                    buckets[h]["act_e"].append(act)
                    buckets[h]["ens"].append(ens)

            end_date = df.index[anchor + chart_h]
            ds = end_date.strftime("%Y-%m-%d")
            lstm_chart = float(lstm_by_h.get(chart_h, float("nan")))
            arima_chart = _arima_price_h_ahead(train_c, chart_h)
            act_chart = float(close_all[anchor + chart_h])
            pred_chart = (
                wl * lstm_chart + wa * arima_chart
                if np.isfinite(lstm_chart) and np.isfinite(arima_chart)
                else float("nan")
            )
            if stress_dates is not None and ds not in stress_dates:
                continue
            if np.isfinite(pred_chart) and np.isfinite(act_chart):
                series.append({"date": ds, "actual": act_chart, "predicted": pred_chart})
    except Exception as exc:  # noqa: BLE001
        lstm_err = str(exc)

    lstm_metrics_by_h = {f"h{h}": _metrics(buckets[h]["act_l"], buckets[h]["lstm"]) for h in horizons}
    arima_metrics_by_h = {f"h{h}": _metrics(buckets[h]["act_a"], buckets[h]["arima"]) for h in horizons}
    ens_metrics_by_h = {f"h{h}": _metrics(buckets[h]["act_e"], buckets[h]["ens"]) for h in horizons}

    lstm_block = {**lstm_metrics_by_h, "n_anchors_stride": float(stride)}
    arima_block = {**arima_metrics_by_h, "one_step_reference": arima_one_m}
    ens_block = ens_metrics_by_h.copy()
    if lstm_err:
        lstm_block["error"] = lstm_err

    chart_m = _metrics([r["actual"] for r in series], [r["predicted"] for r in series])
    ens_block["chart_horizon_days"] = float(chart_h)
    ens_block.update(chart_m)

    metrics = {
        "arima": arima_block,
        "lstm": lstm_block,
        "ensemble": ens_block,
    }
    return metrics, series, lstm_err


def _scenario_4_direction_samples_only(
    model,
    meta: dict,
    df: pd.DataFrame,
    n: int,
    test_start: int,
    wl: float,
    wa: float,
) -> tuple[dict, str | None]:
    """
    Same directional samples as Scenario 4 (strided anchors, ensemble multi-horizon vs ref close),
    without the daily one-step series — used by Scenario 5 after a single shared LSTM train.
    """
    lstm_err: str | None = None
    ref1, act1, pr_e1 = [], [], []
    ref7, act7, pr_e7 = [], [], []
    ref30, act30, pr_e30 = [], [], []
    closes = df["close"].to_numpy(dtype=float)
    dir_stride = max(1, int(settings.BACKTEST_SCENARIO4_DIRECTION_STRIDE))

    try:
        seq_need = _lstm_min_input_rows(int(meta["seq_len"]))
        feat_full = add_indicators(df).ffill().fillna(0.0)
        for i in range(test_start, n - 30):
            if (i - test_start) % dir_stride != 0:
                continue
            dt = df.index[i]
            if dt not in feat_full.index:
                continue
            sub = feat_full.loc[:dt]
            if len(sub) < seq_need:
                continue
            r = float(closes[i])
            train_slice = closes[: i + 1]
            p1l, p7l, p30l, _ = predict_lstm_head_prices_with_model(model, meta, sub)
            p1a = _arima_price_h_ahead(train_slice, 1)
            p7a = _arima_price_h_ahead(train_slice, 7)
            p30a = _arima_price_h_ahead(train_slice, 30)
            if not all(np.isfinite(x) for x in (p1l, p7l, p30l, p1a, p7a, p30a)):
                continue
            e1 = wl * p1l + wa * p1a
            e7 = wl * p7l + wa * p7a
            e30 = wl * p30l + wa * p30a
            a1, a7, a30 = float(closes[i + 1]), float(closes[i + 7]), float(closes[i + 30])
            ref1.append(r)
            act1.append(a1)
            pr_e1.append(e1)
            ref7.append(r)
            act7.append(a7)
            pr_e7.append(e7)
            ref30.append(r)
            act30.append(a30)
            pr_e30.append(e30)
    except Exception as exc:  # noqa: BLE001
        lstm_err = str(exc)

    d1 = _direction_accuracy(ref1, act1, pr_e1)
    d7 = _direction_accuracy(ref7, act7, pr_e7)
    d30 = _direction_accuracy(ref30, act30, pr_e30)
    out = {
        "direction_accuracy_1d": d1,
        "direction_accuracy_7d": d7,
        "direction_accuracy_30d": d30,
        "baseline": 0.5,
        "direction_anchor_stride_days": float(dir_stride),
        "direction_note": (
            "Ensemble direction: sign(actual - ref) vs sign(pred - ref); ref is close on the anchor day. "
            "Sampled every direction_anchor_stride_days to limit ARIMA refits."
        ),
        "headline_7d_percent": float(d7 * 100.0) if np.isfinite(d7) else float("nan"),
    }
    if lstm_err:
        out["error"] = lstm_err
    return out, lstm_err


def _combined_verdict_label_and_explanation(
    ticker: str,
    mape_30: float,
    dir_7: float,
) -> tuple[str, str]:
    """
    Map ensemble 30d MAPE (percent) and 7d direction hit-rate (0–1) to a verdict and plain-English blurb.
    """
    m_ok = np.isfinite(mape_30)
    d_ok = np.isfinite(dir_7)
    pct = dir_7 * 100.0 if d_ok else float("nan")
    mape_s = f"{mape_30:.1f}%" if m_ok else "n/a"
    pct_s = f"{pct:.1f}%" if d_ok else "n/a"

    if m_ok and d_ok:
        if mape_30 < 8.0 and dir_7 > 0.57:
            verdict = "Strong model"
        elif mape_30 < 15.0 and dir_7 > 0.53:
            verdict = "Decent model"
        else:
            verdict = "Needs improvement"
    else:
        verdict = "Needs improvement"

    parts = [
        f"This summary is for {ticker} on the backtest holdout.",
        (
            f"Forward-looking price error (ensemble, about 30 trading days, honest rollout) averages "
            f"{mape_s} MAPE versus realized closes."
            if m_ok
            else "Multi-step MAPE was not available (too few points or model error)."
        ),
        (
            f"The blended model predicted the correct up-down direction {pct_s} of the time at a 7-day horizon, "
            f"compared with a 50 percent random baseline."
            if d_ok
            else "7-day direction accuracy was not available."
        ),
        "These are statistical backtest measures, not trading advice.",
    ]
    explanation = " ".join(parts)
    return verdict, explanation


def run_backtest(
    session: Session,
    ticker: str,
    scenario: int = 1,
    *,
    years: float | None = None,
    max_holdout: bool = False,
    light: bool = False,
) -> dict:
    """
    Run backtest for scenario 1–5. Optional `years` / `max_holdout` control holdout depth (trading years).

    `light=True` on Scenario 5 drops heavy series payloads (for bulk admin runs).
    """
    if scenario not in (1, 2, 3, 4, 5):
        raise ValueError("scenario must be 1, 2, 3, 4, or 5")

    df = get_ohlcv_dataframe(session, ticker)
    n = len(df)
    test_start, hold, requested_days, years_req, years_act, max_h = _holdout_slice(
        n,
        years=years,
        max_holdout=max_holdout,
    )
    min_pre = settings.BACKTEST_MIN_PREHOLDOUT_ROWS
    note = _holdout_note(hold, requested_days, min_pre, max_holdout=max_h)
    train_close = df["close"].iloc[:test_start].to_numpy()
    test_close = df["close"].iloc[test_start:].to_numpy()
    close_all = df["close"].to_numpy(dtype=float)
    wl, wa = _norm_w()

    # --- ARIMA one-step walk (used in scenarios 1 & 4; reference metrics in 2/3) ---
    arima_pred, arima_actual = arima_walk_one_step(train_close, test_close)
    dates = df.index[test_start : test_start + len(arima_actual)]
    arima_one_rows = [
        {"date": d.isoformat()[:10], "actual": float(a), "predicted": float(p)}
        for d, a, p in zip(dates, arima_actual, arima_pred, strict=False)
    ]
    arima_one_m = _metrics(arima_actual, arima_pred)

    ykw = {"holdout_years_requested": years_req, "holdout_years_actual": years_act}

    if scenario == 1:
        return _scenario_1_daily(
            session,
            ticker,
            df,
            n,
            test_start,
            hold,
            requested_days,
            note,
            arima_one_rows,
            arima_one_m,
            wl,
            wa,
            **ykw,
        )
    if scenario == 2:
        return _scenario_2_multistep(
            session,
            ticker,
            df,
            n,
            test_start,
            hold,
            requested_days,
            note,
            arima_one_m,
            close_all,
            wl,
            wa,
            stress_dates=None,
            **ykw,
        )
    if scenario == 3:
        stress = _stress_trading_date_set(df.index, close_all)
        return _scenario_2_multistep(
            session,
            ticker,
            df,
            n,
            test_start,
            hold,
            requested_days,
            note,
            arima_one_m,
            close_all,
            wl,
            wa,
            stress_dates=stress,
            **ykw,
        )
    if scenario == 4:
        return _scenario_4_direction(
            session,
            ticker,
            df,
            n,
            test_start,
            hold,
            requested_days,
            note,
            arima_one_rows,
            arima_one_m,
            wl,
            wa,
            **ykw,
        )
    return _scenario_5_combined(
        session,
        ticker,
        df,
        n,
        test_start,
        hold,
        requested_days,
        note,
        arima_one_m,
        close_all,
        wl,
        wa,
        light=light,
        **ykw,
    )


def _scenario_1_daily(
    session: Session,
    ticker: str,
    df: pd.DataFrame,
    n: int,
    test_start: int,
    hold: int,
    target_hold: int,
    note: str | None,
    arima_one_rows: list[dict],
    arima_one_m: dict,
    wl: float,
    wa: float,
    *,
    holdout_years_requested: float | None,
    holdout_years_actual: float,
) -> dict:
    """Scenario 1: one-step walk-forward; LSTM predicts returns with MinMax-scaled inputs."""
    lstm_series: list[dict] = []
    lstm_m: dict = {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n": 0.0}
    lstm_err: str | None = None

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        try:
            train_lstm_for_ticker(
                session,
                ticker,
                train_end_exclusive=test_start,
                model_root=root,
            )
            model, meta = load_trained_lstm_bundle(ticker, model_root=root)
            seq_need = _lstm_min_input_rows(int(meta["seq_len"]))
            feat_full = add_indicators(df).ffill().fillna(0.0)
            for i in range(test_start, n - 1):
                dt = df.index[i]
                if dt not in feat_full.index:
                    continue
                sub = feat_full.loc[:dt]
                if len(sub) < seq_need:
                    continue
                pred = predict_lstm_one_step_with_model(model, meta, sub)
                act = float(df["close"].iloc[i + 1])
                if not np.isfinite(pred):
                    continue
                lstm_series.append(
                    {
                        "date": df.index[i + 1].isoformat()[:10],
                        "actual": act,
                        "predicted": pred,
                    }
                )
            if lstm_series:
                lstm_m = _metrics(
                    [x["actual"] for x in lstm_series],
                    [x["predicted"] for x in lstm_series],
                )
        except Exception as exc:  # noqa: BLE001
            lstm_err = str(exc)

    if lstm_err:
        lstm_m = {**lstm_m, "error": lstm_err}

    by_a = {x["date"]: x for x in arima_one_rows}
    by_l = {x["date"]: x for x in lstm_series}
    series: list[dict] = []
    for d in sorted(set(by_a) & set(by_l)):
        a, l = by_a[d], by_l[d]
        blend = wl * l["predicted"] + wa * a["predicted"]
        series.append({"date": d, "actual": a["actual"], "predicted": blend})

    ens_m = _metrics([r["actual"] for r in series], [r["predicted"] for r in series])
    metrics = {
        "arima": arima_one_m,
        "lstm": lstm_m,
        "ensemble": ens_m,
    }
    return _build_unified_response(
        ticker=ticker,
        scenario=1,
        hold=hold,
        target_hold=target_hold,
        holdout_note=note,
        metrics=metrics,
        series=series,
        holdout_years_requested=holdout_years_requested,
        holdout_years_actual=holdout_years_actual,
    )


def _scenario_2_multistep(
    session: Session,
    ticker: str,
    df: pd.DataFrame,
    n: int,
    test_start: int,
    hold: int,
    target_hold: int,
    note: str | None,
    arima_one_m: dict,
    close_all: np.ndarray,
    wl: float,
    wa: float,
    *,
    stress_dates: set[str] | None,
    holdout_years_requested: float | None,
    holdout_years_actual: float,
) -> dict:
    """
    Scenario 2/3: honest h-day rollout from each anchor (synthetic path). Chart uses BACKTEST_MULTI_STEP_CHART_HORIZON.
    Scenario 3 filters rows to stress_dates.
    """
    metrics: dict = {"arima": {}, "lstm": {}, "ensemble": {}}
    series: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        try:
            train_lstm_for_ticker(
                session,
                ticker,
                train_end_exclusive=test_start,
                model_root=root,
            )
            model, meta = load_trained_lstm_bundle(ticker, model_root=root)
            metrics, series, lstm_err = _scenario_2_price_metrics_and_series(
                model,
                meta,
                df,
                n,
                test_start,
                close_all,
                wl,
                wa,
                arima_one_m,
                stress_dates,
            )
            if lstm_err:
                metrics["lstm"] = {**metrics.get("lstm", {}), "error": lstm_err}
        except Exception as exc:  # noqa: BLE001
            metrics = {
                "arima": {},
                "lstm": {"error": str(exc)},
                "ensemble": {},
            }
            series = []

    scen = 3 if stress_dates is not None else 2
    if stress_dates is not None and not series:
        note = (note + " " if note else "") + "No backtest points fell in stress windows; widen history or check dates."

    return _build_unified_response(
        ticker=ticker,
        scenario=scen,
        hold=hold,
        target_hold=target_hold,
        holdout_note=note,
        metrics=metrics,
        series=series,
        holdout_years_requested=holdout_years_requested,
        holdout_years_actual=holdout_years_actual,
    )


def _scenario_5_combined(
    session: Session,
    ticker: str,
    df: pd.DataFrame,
    n: int,
    test_start: int,
    hold: int,
    target_hold: int,
    note: str | None,
    arima_one_m: dict,
    close_all: np.ndarray,
    wl: float,
    wa: float,
    *,
    light: bool = False,
    holdout_years_requested: float | None,
    holdout_years_actual: float,
) -> dict:
    """
    Scenario 5: one LSTM train, then Scenario 2 price metrics/series plus Scenario 4-style direction
    (strided samples only — same logic as Scenario 4, without re-running daily one-step series).
    """
    price_metrics: dict = {"arima": {}, "lstm": {}, "ensemble": {}}
    price_series: list[dict] = []
    direction_block: dict = {
        "direction_accuracy_1d": float("nan"),
        "direction_accuracy_7d": float("nan"),
        "direction_accuracy_30d": float("nan"),
        "baseline": 0.5,
        "headline_7d_percent": float("nan"),
    }
    combined_err: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        try:
            train_lstm_for_ticker(
                session,
                ticker,
                train_end_exclusive=test_start,
                model_root=root,
            )
            model, meta = load_trained_lstm_bundle(ticker, model_root=root)
            # Shared model: honest multi-step price track (same as Scenario 2, full holdout).
            price_metrics, price_series, perr = _scenario_2_price_metrics_and_series(
                model,
                meta,
                df,
                n,
                test_start,
                close_all,
                wl,
                wa,
                arima_one_m,
                stress_dates=None,
            )
            if perr:
                combined_err.append(perr)
            # Same model: directional stats aligned with Scenario 4 sampling rules.
            direction_block, derr = _scenario_4_direction_samples_only(
                model, meta, df, n, test_start, wl, wa
            )
            if derr:
                combined_err.append(derr)
        except Exception as exc:  # noqa: BLE001
            combined_err.append(str(exc))

    ens = price_metrics.get("ensemble") or {}
    h30 = ens.get("h30") if isinstance(ens, dict) else None
    mape_30 = float(h30["mape"]) if isinstance(h30, dict) and "mape" in h30 else float("nan")
    dir_7 = float(direction_block.get("direction_accuracy_7d", float("nan")))

    verdict, explanation = _combined_verdict_label_and_explanation(ticker, mape_30, dir_7)
    if combined_err:
        note = (note + " " if note else "") + "Warnings: " + "; ".join(combined_err)

    out: dict = {
        "ticker": ticker,
        "scenario": 5,
        "scenario_label": SCENARIO_LABELS[5],
        "holdout_trading_days": hold,
        "holdout_trading_days_target": target_hold,
        "holdout_note": note,
        "holdout_years_requested": holdout_years_requested,
        "holdout_years_actual": holdout_years_actual,
        "series": price_series,
        "metrics": {
            "summary": "Use price_accuracy for Scenario 2-style metrics and direction_accuracy for Scenario 4-style stats.",
        },
        "price_accuracy": {
            "metrics": price_metrics,
            "series": price_series,
        },
        "direction_accuracy": direction_block,
        "combined_verdict": verdict,
        "verdict_explanation": explanation,
    }
    # Bulk admin runs: drop duplicate long series to limit memory.
    if light:
        out.pop("series", None)
        out["price_accuracy"] = {"metrics": price_metrics}
    return out


def _scenario_4_direction(
    session: Session,
    ticker: str,
    df: pd.DataFrame,
    n: int,
    test_start: int,
    hold: int,
    target_hold: int,
    note: str | None,
    arima_one_rows: list[dict],
    arima_one_m: dict,
    wl: float,
    wa: float,
    *,
    holdout_years_requested: float | None,
    holdout_years_actual: float,
) -> dict:
    """Scenario 4: same daily ensemble series as scenario 1, plus directional accuracy (one LSTM train)."""
    lstm_series: list[dict] = []
    lstm_m: dict = {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n": 0.0}
    lstm_err: str | None = None
    ref1, act1, pr_e1 = [], [], []
    ref7, act7, pr_e7 = [], [], []
    ref30, act30, pr_e30 = [], [], []
    closes = df["close"].to_numpy(dtype=float)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        try:
            train_lstm_for_ticker(
                session,
                ticker,
                train_end_exclusive=test_start,
                model_root=root,
            )
            model, meta = load_trained_lstm_bundle(ticker, model_root=root)
            seq_need = _lstm_min_input_rows(int(meta["seq_len"]))
            feat_full = add_indicators(df).ffill().fillna(0.0)
            # Sparse ARIMA refits for direction stats (full daily LSTM series still above).
            dir_stride = max(1, int(settings.BACKTEST_SCENARIO4_DIRECTION_STRIDE))
            for i in range(test_start, n - 1):
                dt = df.index[i]
                if dt not in feat_full.index:
                    continue
                sub = feat_full.loc[:dt]
                if len(sub) < seq_need:
                    continue
                pred = predict_lstm_one_step_with_model(model, meta, sub)
                act = float(closes[i + 1])
                if np.isfinite(pred):
                    lstm_series.append(
                        {
                            "date": df.index[i + 1].isoformat()[:10],
                            "actual": act,
                            "predicted": pred,
                        }
                    )
                if i + 30 < n and (i - test_start) % dir_stride == 0:
                    r = float(closes[i])
                    train_slice = closes[: i + 1]
                    p1l, p7l, p30l, _ = predict_lstm_head_prices_with_model(model, meta, sub)
                    p1a = _arima_price_h_ahead(train_slice, 1)
                    p7a = _arima_price_h_ahead(train_slice, 7)
                    p30a = _arima_price_h_ahead(train_slice, 30)
                    if not all(np.isfinite(x) for x in (p1l, p7l, p30l, p1a, p7a, p30a)):
                        continue
                    e1 = wl * p1l + wa * p1a
                    e7 = wl * p7l + wa * p7a
                    e30 = wl * p30l + wa * p30a
                    a1, a7, a30 = float(closes[i + 1]), float(closes[i + 7]), float(closes[i + 30])
                    ref1.append(r)
                    act1.append(a1)
                    pr_e1.append(e1)
                    ref7.append(r)
                    act7.append(a7)
                    pr_e7.append(e7)
                    ref30.append(r)
                    act30.append(a30)
                    pr_e30.append(e30)
        except Exception as exc:  # noqa: BLE001
            lstm_err = str(exc)

    if lstm_err:
        lstm_m = {**lstm_m, "error": lstm_err}
    elif lstm_series:
        lstm_m = _metrics(
            [x["actual"] for x in lstm_series],
            [x["predicted"] for x in lstm_series],
        )

    by_a = {x["date"]: x for x in arima_one_rows}
    by_l = {x["date"]: x for x in lstm_series}
    series: list[dict] = []
    for d in sorted(set(by_a) & set(by_l)):
        a, l = by_a[d], by_l[d]
        blend = wl * l["predicted"] + wa * a["predicted"]
        series.append({"date": d, "actual": a["actual"], "predicted": blend})

    ens_m = _metrics([r["actual"] for r in series], [r["predicted"] for r in series])
    metrics: dict = {
        "arima": arima_one_m,
        "lstm": lstm_m,
        "ensemble": ens_m,
        "direction_accuracy_1d": _direction_accuracy(ref1, act1, pr_e1),
        "direction_accuracy_7d": _direction_accuracy(ref7, act7, pr_e7),
        "direction_accuracy_30d": _direction_accuracy(ref30, act30, pr_e30),
        "baseline": 0.5,
        "direction_anchor_stride_days": float(max(1, int(settings.BACKTEST_SCENARIO4_DIRECTION_STRIDE))),
        "direction_note": (
            "Ensemble direction: sign(actual - ref) vs sign(pred - ref); ref is close on the anchor day. "
            "Sampled every direction_anchor_stride_days to limit ARIMA refits."
        ),
    }

    return _build_unified_response(
        ticker=ticker,
        scenario=4,
        hold=hold,
        target_hold=target_hold,
        holdout_note=note,
        metrics=metrics,
        series=series,
        holdout_years_requested=holdout_years_requested,
        holdout_years_actual=holdout_years_actual,
    )


def run_backtest_all(
    session: Session,
    *,
    scenario: int = 5,
    years: float | None = None,
    max_holdout: bool = False,
    persist: bool = True,
) -> dict:
    """
    Run `run_backtest` sequentially for every active ticker (LIGHT_MODE-aware).
    Used by POST /api/admin/backtest-all. Continues on per-ticker failure.

    Uses light=False so each ticker returns full scenario-5 payloads (series/metrics) for SQLite storage.
    """
    tickers = get_active_tickers()
    results: dict[str, dict] = {}
    full_payload_by_ticker: dict[str, dict | None] = {}
    holdout_years_requested: float | None = None
    holdout_years_actual: float | None = None
    for i, sym in enumerate(tickers):
        _bulk_progress_tick(sym, i + 1, len(tickers))
        try:
            r = run_backtest(
                session,
                sym,
                scenario=scenario,
                years=years,
                max_holdout=max_holdout,
                light=False,
            )
            if holdout_years_actual is None and r.get("holdout_years_actual") is not None:
                holdout_years_requested = r.get("holdout_years_requested")  # type: ignore[assignment]
                holdout_years_actual = float(r["holdout_years_actual"])
            pa = r.get("price_accuracy")
            pm = pa.get("metrics", {}) if isinstance(pa, dict) else {}
            ens = pm.get("ensemble", {}) if isinstance(pm, dict) else {}
            h30 = ens.get("h30", {}) if isinstance(ens, dict) else {}
            mape_30 = (
                float(h30["mape"])
                if isinstance(h30, dict) and isinstance(h30.get("mape"), (int, float)) and np.isfinite(h30["mape"])
                else float("nan")
            )
            da = r.get("direction_accuracy", {}) if isinstance(r.get("direction_accuracy"), dict) else {}
            d7 = float(da.get("direction_accuracy_7d", float("nan")))
            results[sym] = {
                "status": "ok",
                "combined_verdict": r.get("combined_verdict"),
                "mape_30d": mape_30,
                "direction_accuracy_7d": d7,
                "holdout_days": int(r.get("holdout_trading_days", 0)),
            }
            full_payload_by_ticker[sym] = r
        except Exception as exc:  # noqa: BLE001
            logger.exception("Backtest-all failed for %s", sym)
            results[sym] = {
                "status": f"error: {exc}",
                "combined_verdict": None,
                "mape_30d": float("nan"),
                "direction_accuracy_7d": float("nan"),
                "holdout_days": 0,
            }
            full_payload_by_ticker[sym] = None
        finally:
            _bulk_progress_done_one()

    ok_rows = {k: v for k, v in results.items() if v["status"] == "ok"}
    mapes = [v["mape_30d"] for v in ok_rows.values() if np.isfinite(v["mape_30d"])]
    dirs = [v["direction_accuracy_7d"] for v in ok_rows.values() if np.isfinite(v["direction_accuracy_7d"])]

    def _best_by_dir() -> str | None:
        if not dirs:
            return None
        mx = max(dirs)
        for k, v in ok_rows.items():
            if np.isfinite(v["direction_accuracy_7d"]) and v["direction_accuracy_7d"] == mx:
                return k
        return None

    def _worst_by_dir() -> str | None:
        if not dirs:
            return None
        mn = min(dirs)
        for k, v in ok_rows.items():
            if np.isfinite(v["direction_accuracy_7d"]) and v["direction_accuracy_7d"] == mn:
                return k
        return None

    avg_mape = float(np.mean(mapes)) if mapes else float("nan")
    avg_dir = float(np.mean(dirs)) if dirs else float("nan")

    strong: list[str] = []
    decent: list[str] = []
    for k, v in ok_rows.items():
        m = v["mape_30d"]
        d = v["direction_accuracy_7d"]
        if np.isfinite(m) and np.isfinite(d):
            if m < 8.0 and d > 0.57:
                strong.append(k)
            elif m < 15.0 and d > 0.53:
                decent.append(k)

    # Everyone not classified as strong or decent (includes errors and weak ok rows).
    needs = [k for k in results if k not in strong and k not in decent]

    above_50 = [k for k, v in ok_rows.items() if np.isfinite(v["direction_accuracy_7d"]) and v["direction_accuracy_7d"] > 0.5]

    years_reported: float | None
    if max_holdout:
        years_reported = None
    elif years is not None:
        years_reported = float(years)
    else:
        years_reported = float(settings.BACKTEST_YEARS)

    aggregate = {
        "avg_mape_30d": avg_mape,
        "avg_direction_accuracy_7d": avg_dir,
        "best_ticker": _best_by_dir(),
        "worst_ticker": _worst_by_dir(),
        "tickers_above_50pct_direction": above_50,
        "tickers_strong_model": strong,
        "tickers_decent_model": decent,
        "tickers_needs_improvement": needs,
    }

    saved_run_id: int | None = None
    if persist and scenario == 5:
        from services.backtest_storage import save_bulk_run

        try:
            saved_run_id = save_bulk_run(
                session,
                scenario=scenario,
                years=years_reported if not max_holdout else None,
                max_holdout=max_holdout,
                holdout_years_requested=holdout_years_requested,
                holdout_years_actual=holdout_years_actual,
                aggregate=aggregate,
                summary_by_ticker=results,
                full_payload_by_ticker=full_payload_by_ticker,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to persist bulk backtest: %s", exc)

    return {
        "scenario": scenario,
        "years": years_reported,
        "max_holdout": max_holdout,
        "tickers_tested": tickers,
        "results": results,
        "aggregate": aggregate,
        "saved_run_id": saved_run_id,
        "holdout_years_requested": holdout_years_requested,
        "holdout_years_actual": holdout_years_actual,
    }
