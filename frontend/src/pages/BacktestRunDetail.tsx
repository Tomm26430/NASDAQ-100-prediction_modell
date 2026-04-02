import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { fetchSavedRun, type SavedRunDetail } from "../api/client";
import { verdictStyle } from "../components/Scenario5Panels";

export function BacktestRunDetail() {
  const { runId } = useParams();
  const id = runId ? parseInt(runId, 10) : NaN;
  const [data, setData] = useState<SavedRunDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!Number.isFinite(id)) return;
    let c = false;
    (async () => {
      try {
        const d = await fetchSavedRun(id);
        if (!c) setData(d);
      } catch (e: unknown) {
        if (!c) setErr(e instanceof Error ? e.message : "Load failed");
      }
    })();
    return () => {
      c = true;
    };
  }, [id]);

  if (!Number.isFinite(id)) return <main>Invalid run id</main>;

  return (
    <main>
      <p>
        <Link to="/backtest/history">← Saved backtests</Link>
      </p>
      <h1>Backtest run #{id}</h1>
      {err && <p className="err">{err}</p>}
      {data && (
        <>
          <div className="card" style={{ fontSize: "0.9rem", color: "#334155" }}>
            <p>
              <strong>Type:</strong> {data.run_type} · <strong>Scenario:</strong> {data.scenario} ·{" "}
              <strong>Holdout:</strong>{" "}
              {data.max_holdout
                ? "max"
                : data.holdout_years_requested != null
                  ? `${data.holdout_years_requested}y requested`
                  : "—"}
              {data.holdout_years_actual != null ? ` · ${data.holdout_years_actual.toFixed(2)}y actual` : ""}
            </p>
            <p style={{ marginBottom: 0 }}>
              <strong>Started:</strong> {data.created_at ?? "—"}
              {data.source_ticker ? ` · single ticker: ${data.source_ticker}` : ""}
            </p>
          </div>
          {data.aggregate && (
            <div className="card">
              <h2 style={{ marginTop: 0, fontSize: "1rem" }}>Aggregate</h2>
              <pre style={{ fontSize: "0.75rem", overflow: "auto" }}>{JSON.stringify(data.aggregate, null, 2)}</pre>
            </div>
          )}
          <div className="card">
            <h2 style={{ marginTop: 0, fontSize: "1rem" }}>Symbols</h2>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", fontSize: "0.85rem", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ textAlign: "left", borderBottom: "1px solid #e2e8f0" }}>
                    <th style={{ padding: 6 }}>Ticker</th>
                    <th style={{ padding: 6 }}>Verdict</th>
                    <th style={{ padding: 6 }}>MAPE 30d</th>
                    <th style={{ padding: 6 }}>Dir 7d</th>
                    <th style={{ padding: 6 }}>Status</th>
                    <th style={{ padding: 6 }}>Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {data.tickers.map((t) => {
                    const st = verdictStyle(t.combined_verdict, t.status);
                    const to = `/backtest/runs/${id}/ticker/${encodeURIComponent(t.ticker)}`;
                    return (
                      <tr key={t.ticker} style={{ borderBottom: "1px solid #f1f5f9" }}>
                        <td style={{ padding: 6 }}>{t.ticker}</td>
                        <td style={{ padding: 6, ...st }}>{t.combined_verdict ?? "—"}</td>
                        <td style={{ padding: 6 }}>
                          {t.mape_30d != null && Number.isFinite(t.mape_30d) ? `${t.mape_30d.toFixed(2)}%` : "—"}
                        </td>
                        <td style={{ padding: 6 }}>
                          {t.direction_accuracy_7d != null && Number.isFinite(t.direction_accuracy_7d)
                            ? `${(t.direction_accuracy_7d * 100).toFixed(1)}%`
                            : "—"}
                        </td>
                        <td style={{ padding: 6, fontSize: "0.75rem" }}>{t.status}</td>
                        <td style={{ padding: 6 }}>
                          {t.has_detail ? (
                            <Link to={to}>Charts &amp; analysis</Link>
                          ) : (
                            <span style={{ color: "#94a3b8" }}>—</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </main>
  );
}
