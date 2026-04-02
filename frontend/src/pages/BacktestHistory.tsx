import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchSavedRuns, type SavedRunListItem } from "../api/client";

export function BacktestHistory() {
  const [runs, setRuns] = useState<SavedRunListItem[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let c = false;
    (async () => {
      try {
        const d = await fetchSavedRuns();
        if (!c) setRuns(d.runs);
      } catch (e: unknown) {
        if (!c) setErr(e instanceof Error ? e.message : "Failed to load");
      }
    })();
    return () => {
      c = true;
    };
  }, []);

  return (
    <main>
      <p>
        <Link to="/backtest">← Backtesting</Link>
      </p>
      <h1>Saved backtests</h1>
      <p style={{ color: "#475569" }}>
        Every completed bulk test (scenario 5) and single backtest is stored in SQLite with full per-symbol metrics and
        charts. Open a run to drill into each ticker or index.
      </p>
      <p>
        <Link to="/backtest/compare">Compare two runs →</Link>
      </p>
      {err && <p className="err">{err}</p>}
      <div className="card" style={{ overflowX: "auto" }}>
        <table className="data" style={{ width: "100%", fontSize: "0.9rem" }}>
          <thead>
            <tr>
              <th>ID</th>
              <th>When</th>
              <th>Type</th>
              <th>Scenario</th>
              <th>Holdout</th>
              <th>Symbols</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id}>
                <td>{r.id}</td>
                <td style={{ whiteSpace: "nowrap" }}>{r.created_at ?? "—"}</td>
                <td>{r.run_type}</td>
                <td>{r.scenario}</td>
                <td>
                  {r.max_holdout
                    ? "max"
                    : r.years_requested != null
                      ? `${r.years_requested}y req`
                      : "—"}
                  {r.holdout_years_actual != null ? ` · ${r.holdout_years_actual.toFixed(2)}y actual` : ""}
                </td>
                <td>{r.ticker_count}</td>
                <td>
                  <Link to={`/backtest/runs/${r.id}`}>Open</Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {runs.length === 0 && !err && <p style={{ color: "#64748b" }}>No saved runs yet. Run a backtest or bulk test.</p>}
      </div>
    </main>
  );
}
