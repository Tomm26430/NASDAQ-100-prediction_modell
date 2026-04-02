import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchCompareRuns, fetchSavedRuns, type CompareRunsResponse, type SavedRunListItem } from "../api/client";

export function BacktestCompare() {
  const [runs, setRuns] = useState<SavedRunListItem[]>([]);
  const [a, setA] = useState("1");
  const [b, setB] = useState("2");
  const [cmp, setCmp] = useState<CompareRunsResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    void fetchSavedRuns().then((d) => {
      setRuns(d.runs);
      if (d.runs.length >= 2) {
        setA(String(d.runs[0].id));
        setB(String(d.runs[1].id));
      }
    });
  }, []);

  async function loadCompare() {
    setErr(null);
    setCmp(null);
    const ia = parseInt(a, 10);
    const ib = parseInt(b, 10);
    if (!Number.isFinite(ia) || !Number.isFinite(ib)) {
      setErr("Enter two valid run IDs.");
      return;
    }
    try {
      const d = await fetchCompareRuns(ia, ib);
      setCmp(d);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "Compare failed");
    }
  }

  return (
    <main>
      <p>
        <Link to="/backtest/history">← Saved backtests</Link>
      </p>
      <h1>Compare backtest runs</h1>
      <p style={{ color: "#475569" }}>
        Pick two saved runs by ID. The table shows metrics for symbols that appear in both runs (typical for bulk tests
        with the same active universe).
      </p>
      <div className="card" style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem", alignItems: "flex-end" }}>
        <label>
          Run A
          <select value={a} onChange={(e) => setA(e.target.value)} style={{ marginLeft: 8 }}>
            {runs.map((r) => (
              <option key={r.id} value={String(r.id)}>
                #{r.id} · {r.created_at?.slice(0, 19) ?? "?"} · {r.run_type}
              </option>
            ))}
          </select>
        </label>
        <label>
          Run B
          <select value={b} onChange={(e) => setB(e.target.value)} style={{ marginLeft: 8 }}>
            {runs.map((r) => (
              <option key={r.id} value={String(r.id)}>
                #{r.id} · {r.created_at?.slice(0, 19) ?? "?"} · {r.run_type}
              </option>
            ))}
          </select>
        </label>
        <button type="button" onClick={() => void loadCompare()}>
          Compare
        </button>
      </div>
      {err && <p className="err">{err}</p>}
      {cmp && (
        <>
          <div className="card" style={{ marginTop: "1rem" }}>
            <h2 style={{ marginTop: 0, fontSize: "1rem" }}>Run metadata</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", fontSize: "0.85rem" }}>
              <div>
                <strong>Run A</strong>
                <pre style={{ fontSize: "0.75rem", overflow: "auto" }}>{JSON.stringify(cmp.run_a, null, 2)}</pre>
              </div>
              <div>
                <strong>Run B</strong>
                <pre style={{ fontSize: "0.75rem", overflow: "auto" }}>{JSON.stringify(cmp.run_b, null, 2)}</pre>
              </div>
            </div>
          </div>
          <div className="card">
            <h2 style={{ marginTop: 0, fontSize: "1rem" }}>Per-ticker deltas</h2>
            <p style={{ fontSize: "0.85rem", color: "#64748b" }}>
              Δ MAPE = run A − run B (negative means B is better). Δ Dir = same for 7d direction hit-rate.
            </p>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", fontSize: "0.8rem", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ textAlign: "left", borderBottom: "1px solid #e2e8f0" }}>
                    <th style={{ padding: 6 }}>Ticker</th>
                    <th style={{ padding: 6 }}>A verdict</th>
                    <th style={{ padding: 6 }}>B verdict</th>
                    <th style={{ padding: 6 }}>Δ MAPE</th>
                    <th style={{ padding: 6 }}>Δ Dir 7d</th>
                  </tr>
                </thead>
                <tbody>
                  {cmp.comparison_rows.map((row) => (
                    <tr key={row.ticker} style={{ borderBottom: "1px solid #f1f5f9" }}>
                      <td style={{ padding: 6 }}>{row.ticker}</td>
                      <td style={{ padding: 6 }}>{row.run_a.verdict ?? "—"}</td>
                      <td style={{ padding: 6 }}>{row.run_b.verdict ?? "—"}</td>
                      <td style={{ padding: 6 }}>
                        {row.delta_mape_30d != null && Number.isFinite(row.delta_mape_30d)
                          ? row.delta_mape_30d.toFixed(2)
                          : "—"}
                      </td>
                      <td style={{ padding: 6 }}>
                        {row.delta_direction_7d != null && Number.isFinite(row.delta_direction_7d)
                          ? row.delta_direction_7d.toFixed(4)
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </main>
  );
}
