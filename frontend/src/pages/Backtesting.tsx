import { useMemo, useState } from "react";
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
import {
  fetchBacktest,
  postBacktestAll,
  type BacktestResponse,
  type BacktestSeriesRow,
  type BulkBacktestResponse,
} from "../api/client";

type HoldYearsPreset = "1" | "3" | "5" | "10" | "max";

function backtestOptsFromPreset(preset: HoldYearsPreset): { years?: number; maxHoldout?: boolean } {
  if (preset === "max") return { maxHoldout: true };
  return { years: Number(preset) };
}

function verdictStyle(verdict: string | null | undefined, status: string): { color: string; fontWeight: number } {
  if (!status.startsWith("ok")) return { color: "#b91c1c", fontWeight: 600 };
  if (verdict === "Strong model") return { color: "#15803d", fontWeight: 600 };
  if (verdict === "Decent model") return { color: "#1d4ed8", fontWeight: 600 };
  return { color: "#b45309", fontWeight: 500 };
}

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
  const [holdPreset, setHoldPreset] = useState<HoldYearsPreset>("5");
  const [data, setData] = useState<BacktestResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [bulk, setBulk] = useState<BulkBacktestResponse | null>(null);
  const [bulkErr, setBulkErr] = useState<string | null>(null);
  const [bulkLoading, setBulkLoading] = useState(false);
  const [sortKey, setSortKey] = useState<"ticker" | "verdict" | "mape" | "dir" | "status">("ticker");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  async function run() {
    setLoading(true);
    setErr(null);
    setData(null);
    try {
      const d = await fetchBacktest(ticker.trim(), scenario, backtestOptsFromPreset(holdPreset));
      setData(d);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "Backtest failed");
    } finally {
      setLoading(false);
    }
  }

  async function runBulk() {
    setBulkLoading(true);
    setBulkErr(null);
    setBulk(null);
    try {
      const b = await postBacktestAll(5, backtestOptsFromPreset(holdPreset));
      setBulk(b);
    } catch (e: unknown) {
      setBulkErr(e instanceof Error ? e.message : "Bulk backtest failed");
    } finally {
      setBulkLoading(false);
    }
  }

  function toggleSort(k: typeof sortKey) {
    if (sortKey === k) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(k);
      setSortDir("asc");
    }
  }

  const bulkRows = useMemo(() => {
    if (!bulk?.results) return [];
    return Object.entries(bulk.results).map(([t, r]) => ({ ticker: t, ...r }));
  }, [bulk]);

  const sortedBulkRows = useMemo(() => {
    const mul = sortDir === "asc" ? 1 : -1;
    return [...bulkRows].sort((a, b) => {
      switch (sortKey) {
        case "ticker":
          return mul * a.ticker.localeCompare(b.ticker);
        case "verdict":
          return mul * String(a.combined_verdict ?? "").localeCompare(String(b.combined_verdict ?? ""));
        case "mape":
          return mul * ((a.mape_30d ?? NaN) - (b.mape_30d ?? NaN));
        case "dir":
          return mul * ((a.direction_accuracy_7d ?? NaN) - (b.direction_accuracy_7d ?? NaN));
        case "status":
          return mul * a.status.localeCompare(b.status);
        default:
          return 0;
      }
    });
  }, [bulkRows, sortKey, sortDir]);

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
        Walk-forward evaluation on a configurable holdout window (years below). Heavy: may take many minutes,
        especially scenarios 2–3 and 5. Retrain LSTMs after pipeline changes (POST /api/admin/train-models).
      </p>
      <div className="card" style={{ marginBottom: "0.75rem" }}>
        <span style={{ fontSize: "0.75rem", color: "#64748b", display: "block", marginBottom: 6 }}>Holdout depth</span>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
          {(["1", "3", "5", "10", "max"] as const).map((y) => (
            <button
              key={y}
              type="button"
              onClick={() => setHoldPreset(y)}
              style={{
                padding: "0.35rem 0.65rem",
                borderRadius: 6,
                border: holdPreset === y ? "2px solid #2563eb" : "1px solid #cbd5e1",
                background: holdPreset === y ? "#eff6ff" : "#fff",
                cursor: "pointer",
                fontSize: "0.85rem",
              }}
            >
              {y === "max" ? "Max" : `${y}y`}
            </button>
          ))}
        </div>
      </div>
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
        <button type="button" onClick={runBulk} disabled={bulkLoading} style={{ marginLeft: "0.25rem" }}>
          {bulkLoading ? "Bulk running…" : "Run bulk test (scenario 5)"}
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
      {bulkErr && <p className="err">{bulkErr}</p>}
      {data && (
        <>
          <p style={{ marginTop: "0.75rem", color: "#334155", fontSize: "0.95rem" }}>
            <strong>{data.scenario_label}</strong>
            {data.scenario === 2 || data.scenario === 3 ? (
              <span style={{ color: "#64748b" }}> — chart uses multi-step rollout (see metrics for 7d / 30d / 90d).</span>
            ) : null}
          </p>
          {data.holdout_years_actual != null && (
            <p style={{ margin: "0.25rem 0 0", fontSize: "0.85rem", color: "#64748b" }}>
              Holdout years requested:{" "}
              {data.holdout_years_requested != null ? data.holdout_years_requested : "max (full cache)"} · actual:{" "}
              {typeof data.holdout_years_actual === "number" ? data.holdout_years_actual.toFixed(2) : "—"} (
              {data.holdout_trading_days} trading days)
            </p>
          )}
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
      {bulk && (
        <div className="card" style={{ marginTop: "1.25rem" }}>
          <h2 style={{ marginTop: 0, fontSize: "1rem" }}>Bulk backtest (active tickers)</h2>
          <p style={{ fontSize: "0.85rem", color: "#64748b" }}>
            Scenario {bulk.scenario}, years param: {bulk.years == null ? "max holdout" : bulk.years} · tested{" "}
            {bulk.tickers_tested.length} symbols.
          </p>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", fontSize: "0.8rem", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ textAlign: "left", borderBottom: "1px solid #e2e8f0" }}>
                  <th style={{ padding: 6, cursor: "pointer" }} onClick={() => toggleSort("ticker")}>
                    Ticker {sortKey === "ticker" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                  </th>
                  <th style={{ padding: 6, cursor: "pointer" }} onClick={() => toggleSort("verdict")}>
                    Verdict {sortKey === "verdict" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                  </th>
                  <th style={{ padding: 6, cursor: "pointer" }} onClick={() => toggleSort("mape")}>
                    MAPE 30d {sortKey === "mape" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                  </th>
                  <th style={{ padding: 6, cursor: "pointer" }} onClick={() => toggleSort("dir")}>
                    Dir 7d {sortKey === "dir" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                  </th>
                  <th style={{ padding: 6, cursor: "pointer" }} onClick={() => toggleSort("status")}>
                    Status {sortKey === "status" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedBulkRows.map((row) => {
                  const st = verdictStyle(row.combined_verdict, row.status);
                  return (
                    <tr key={row.ticker} style={{ borderBottom: "1px solid #f1f5f9" }}>
                      <td style={{ padding: 6 }}>{row.ticker}</td>
                      <td style={{ padding: 6, ...st }}>{row.combined_verdict ?? "—"}</td>
                      <td style={{ padding: 6 }}>
                        {Number.isFinite(row.mape_30d) ? `${row.mape_30d.toFixed(2)}%` : "—"}
                      </td>
                      <td style={{ padding: 6 }}>
                        {Number.isFinite(row.direction_accuracy_7d)
                          ? `${(row.direction_accuracy_7d * 100).toFixed(1)}%`
                          : "—"}
                      </td>
                      <td style={{ padding: 6, fontSize: "0.75rem", color: row.status === "ok" ? "#334155" : "#b91c1c" }}>
                        {row.status}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <pre style={{ fontSize: "0.7rem", marginTop: "0.75rem", overflow: "auto", maxHeight: 200 }}>
            {JSON.stringify(bulk.aggregate, null, 2)}
          </pre>
        </div>
      )}
    </main>
  );
}
