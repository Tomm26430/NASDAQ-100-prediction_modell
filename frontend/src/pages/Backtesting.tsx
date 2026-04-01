import { useState } from "react";
import { Link } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchBacktest, type BacktestResponse, type BacktestSeriesRow } from "../api/client";

const SCENARIOS = [
  { id: 1, short: "1 — Daily (real inputs)" },
  { id: 2, short: "2 — Multi-step honest" },
  { id: 3, short: "3 — Stress / volatility" },
  { id: 4, short: "4 — Direction accuracy" },
  { id: 5, short: "5 — Combined honest assessment" },
] as const;

/** Bar color: green above 55%, yellow 52–55%, red below 52% (vs 50% random baseline). */
function directionBarColor(percent: number): string {
  if (!Number.isFinite(percent)) return "#94a3b8";
  if (percent > 55) return "#16a34a";
  if (percent >= 52) return "#ca8a04";
  return "#dc2626";
}

function toPercentFraction(v: unknown): number {
  if (typeof v === "number" && Number.isFinite(v)) return v * 100;
  return NaN;
}

function lineDataFromSeries(rows: BacktestSeriesRow[] | undefined) {
  return (
    rows?.map((r) => ({
      date: r.date,
      actual: r.actual,
      predicted: r.predicted,
    })) ?? []
  );
}

type HorizonKey = "h7" | "h30" | "h90";

function PriceHorizonTable({ metrics }: { metrics: Record<string, unknown> | undefined }) {
  const ens = metrics?.ensemble as Record<string, { mae?: number; rmse?: number; mape?: number; n?: number }> | undefined;
  if (!ens) return <p style={{ color: "#64748b", fontSize: "0.85rem" }}>No ensemble horizon metrics.</p>;
  const keys: HorizonKey[] = ["h7", "h30", "h90"];
  return (
    <table style={{ width: "100%", fontSize: "0.8rem", borderCollapse: "collapse" }}>
      <thead>
        <tr style={{ textAlign: "left", borderBottom: "1px solid #e2e8f0" }}>
          <th style={{ padding: "4px 8px" }}>Horizon</th>
          <th style={{ padding: "4px 8px" }}>MAE</th>
          <th style={{ padding: "4px 8px" }}>RMSE</th>
          <th style={{ padding: "4px 8px" }}>MAPE %</th>
          <th style={{ padding: "4px 8px" }}>n</th>
        </tr>
      </thead>
      <tbody>
        {keys.map((k) => {
          const row = ens[k];
          if (!row) return null;
          return (
            <tr key={k} style={{ borderBottom: "1px solid #f1f5f9" }}>
              <td style={{ padding: "4px 8px" }}>{k}</td>
              <td style={{ padding: "4px 8px" }}>{row.mae != null ? row.mae.toFixed(4) : "—"}</td>
              <td style={{ padding: "4px 8px" }}>{row.rmse != null ? row.rmse.toFixed(4) : "—"}</td>
              <td style={{ padding: "4px 8px" }}>{row.mape != null ? row.mape.toFixed(2) : "—"}</td>
              <td style={{ padding: "4px 8px" }}>{row.n != null ? String(row.n) : "—"}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function Scenario5Panels({ data }: { data: BacktestResponse }) {
  const pa = data.price_accuracy;
  const da = data.direction_accuracy;
  const lineData = lineDataFromSeries(pa?.series);
  const d1 = toPercentFraction(da?.direction_accuracy_1d);
  const d7 = toPercentFraction(da?.direction_accuracy_7d);
  const d30 = toPercentFraction(da?.direction_accuracy_30d);
  const headline =
    typeof da?.headline_7d_percent === "number" && Number.isFinite(da.headline_7d_percent)
      ? da.headline_7d_percent
      : d7;
  const barData = [
    { name: "1d", accuracy: d1, baseline: 50 },
    { name: "7d", accuracy: d7, baseline: 50 },
    { name: "30d", accuracy: d30, baseline: 50 },
  ];

  return (
    <>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          gap: "1rem",
          marginTop: "0.75rem",
        }}
      >
        <div className="card">
          <h2 style={{ marginTop: 0, fontSize: "1rem" }}>Price Accuracy (Scenario 2)</h2>
          <p style={{ margin: "0 0 0.5rem", fontSize: "0.8rem", color: "#64748b" }}>
            Ensemble vs actual (honest multi-step rollout); metrics are by horizon.
          </p>
          {lineData.length > 0 ? (
            <div style={{ width: "100%", height: 280 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lineData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="date" tick={{ fontSize: 9 }} minTickGap={24} />
                  <YAxis domain={["auto", "auto"]} tick={{ fontSize: 9 }} width={48} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="actual" stroke="#0f172a" dot={false} name="Actual" />
                  <Line type="monotone" dataKey="predicted" stroke="#2563eb" dot={false} name="Predicted" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p style={{ color: "#64748b" }}>No price series returned.</p>
          )}
          <h3 style={{ fontSize: "0.9rem", marginTop: "1rem" }}>MAE / RMSE / MAPE (ensemble)</h3>
          <PriceHorizonTable metrics={pa?.metrics} />
        </div>

        <div className="card">
          <h2 style={{ marginTop: 0, fontSize: "1rem" }}>Direction Accuracy (Scenario 4)</h2>
          <p style={{ margin: "0 0 0.25rem", fontSize: "2rem", fontWeight: 700, color: "#0f172a" }}>
            {Number.isFinite(headline) ? `${headline.toFixed(1)}%` : "—"}
            <span style={{ fontSize: "0.9rem", fontWeight: 400, color: "#64748b", marginLeft: 8 }}>
              7-day direction hit rate
            </span>
          </p>
          <p style={{ margin: "0 0 0.75rem", fontSize: "0.8rem", color: "#64748b" }}>
            Bars: model accuracy (%). Gray line: 50% random baseline. Green &gt;55%, yellow 52–55%, red &lt;52%.
          </p>
          <div style={{ width: "100%", height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, "auto"]} tick={{ fontSize: 10 }} width={36} unit="%" />
                <Tooltip formatter={(v: number) => [`${v.toFixed(1)}%`, "Accuracy"]} />
                <ReferenceLine y={50} stroke="#64748b" strokeDasharray="4 4" label={{ value: "50% baseline", fill: "#64748b", fontSize: 10 }} />
                <Bar dataKey="accuracy" name="Direction accuracy %" radius={[4, 4, 0, 0]}>
                  {barData.map((e, i) => (
                    <Cell key={`c-${i}`} fill={directionBarColor(e.accuracy)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div
        className="card"
        style={{
          marginTop: "1rem",
          background: "#f8fafc",
          border: "1px solid #cbd5e1",
        }}
      >
        <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>{data.combined_verdict ?? "Verdict"}</h2>
        <p style={{ margin: 0, fontSize: "0.95rem", color: "#334155", lineHeight: 1.5 }}>
          {data.verdict_explanation ?? ""}
        </p>
      </div>
    </>
  );
}

export function Backtesting() {
  const [ticker, setTicker] = useState("AAPL");
  const [scenario, setScenario] = useState(1);
  const [data, setData] = useState<BacktestResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true);
    setErr(null);
    setData(null);
    try {
      const d = await fetchBacktest(ticker.trim(), scenario);
      setData(d);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "Backtest failed");
    } finally {
      setLoading(false);
    }
  }

  const chartData =
    data?.series.map((r) => ({
      date: r.date,
      actual: r.actual,
      predicted: r.predicted,
    })) ?? [];

  return (
    <main>
      <p>
        <Link to="/">← Dashboard</Link>
      </p>
      <h1>Backtesting</h1>
      <p style={{ color: "#475569" }}>
        Walk-forward evaluation on the holdout window (default ~10 trading years). Heavy: may take many minutes,
        especially scenarios 2–3 and 5 (Scenario 5 runs one LSTM train but both rollout and direction work).
      </p>
      <div className="card" style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem", alignItems: "center" }}>
        <label>
          Ticker{" "}
          <input value={ticker} onChange={(e) => setTicker(e.target.value)} style={{ marginLeft: 6 }} />
        </label>
        <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={{ fontSize: "0.75rem", color: "#64748b" }}>Scenario</span>
          <select
            value={scenario}
            onChange={(e) => setScenario(Number(e.target.value))}
            style={{ minWidth: 240 }}
          >
            {SCENARIOS.map((s) => (
              <option key={s.id} value={s.id}>
                {s.short}
              </option>
            ))}
          </select>
        </label>
        <button type="button" onClick={run} disabled={loading}>
          {loading ? "Running…" : "Run backtest"}
        </button>
      </div>
      {scenario === 1 && (
        <div
          className="card"
          style={{
            background: "#eff6ff",
            border: "1px solid #93c5fd",
            marginTop: "0.75rem",
          }}
        >
          <p style={{ margin: 0, fontSize: "0.9rem", color: "#1e3a8a" }}>
            <strong>Note:</strong> this scenario uses real daily prices as input and is optimistic. Use Scenario 2 for
            realistic forward-looking accuracy.
          </p>
        </div>
      )}
      {scenario === 5 && (
        <div
          className="card"
          style={{
            background: "#f0fdf4",
            border: "1px solid #86efac",
            marginTop: "0.75rem",
          }}
        >
          <p style={{ margin: 0, fontSize: "0.9rem", color: "#14532d" }}>
            <strong>Scenario 5</strong> trains the LSTM once, then runs the same honest price rollout as Scenario 2 and
            the same strided direction sampling as Scenario 4 — no duplicate training.
          </p>
        </div>
      )}
      {err && <p className="err">{err}</p>}
      {data && (
        <>
          <p style={{ marginTop: "0.75rem", color: "#334155", fontSize: "0.95rem" }}>
            <strong>{data.scenario_label}</strong>
            {data.scenario === 2 || data.scenario === 3 ? (
              <span style={{ color: "#64748b" }}> — chart uses multi-step rollout (see metrics for 7d / 30d / 90d).</span>
            ) : null}
          </p>
          {data.holdout_note && (
            <div className="card" style={{ background: "#fffbeb", border: "1px solid #fcd34d" }}>
              <p style={{ margin: 0, fontSize: "0.9rem", color: "#92400e" }}>{data.holdout_note}</p>
            </div>
          )}

          {data.scenario === 5 ? (
            <Scenario5Panels data={data} />
          ) : (
            <>
              <div className="card">
                <h2 style={{ marginTop: 0, fontSize: "1rem" }}>
                  Metrics ({data.holdout_trading_days} days
                  {data.holdout_trading_days_target != null
                    ? ` · target ${data.holdout_trading_days_target} d`
                    : ""}
                  )
                </h2>
                <pre style={{ fontSize: "0.8rem", overflow: "auto" }}>{JSON.stringify(data.metrics, null, 2)}</pre>
              </div>
              {chartData.length > 0 && (
                <div className="card">
                  <h2 style={{ marginTop: 0, fontSize: "1rem" }}>Predicted vs actual (aligned dates)</h2>
                  <div style={{ width: "100%", height: 360 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="date" tick={{ fontSize: 9 }} minTickGap={20} />
                        <YAxis domain={["auto", "auto"]} tick={{ fontSize: 9 }} width={52} />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="actual" stroke="#0f172a" dot={false} name="Actual" />
                        <Line type="monotone" dataKey="predicted" stroke="#2563eb" dot={false} name="Predicted" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </>
          )}
        </>
      )}
    </main>
  );
}
