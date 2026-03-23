import { useState } from "react";
import { Link } from "react-router-dom";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchBacktest, type BacktestResponse } from "../api/client";

export function Backtesting() {
  const [ticker, setTicker] = useState("AAPL");
  const [data, setData] = useState<BacktestResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true);
    setErr(null);
    setData(null);
    try {
      const d = await fetchBacktest(ticker.trim());
      setData(d);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "Backtest failed");
    } finally {
      setLoading(false);
    }
  }

  const chartData =
    data?.series.ensemble.map((r) => ({
      date: r.date,
      actual: r.actual,
      ensemble: r.predicted_ensemble,
    })) ?? [];

  return (
    <main>
      <p>
        <Link to="/">← Dashboard</Link>
      </p>
      <h1>Backtesting</h1>
      <p style={{ color: "#475569" }}>
        One-step walk-forward on the recent holdout window (see API for metrics). Heavy: may take a minute.
      </p>
      <div className="card" style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
        <label>
          Ticker{" "}
          <input value={ticker} onChange={(e) => setTicker(e.target.value)} style={{ marginLeft: 6 }} />
        </label>
        <button type="button" onClick={run} disabled={loading}>
          {loading ? "Running…" : "Run backtest"}
        </button>
      </div>
      {err && <p className="err">{err}</p>}
      {data && (
        <>
          <div className="card">
            <h2 style={{ marginTop: 0, fontSize: "1rem" }}>Metrics ({data.holdout_trading_days} days)</h2>
            <pre style={{ fontSize: "0.8rem", overflow: "auto" }}>{JSON.stringify(data.metrics, null, 2)}</pre>
          </div>
          {chartData.length > 0 && (
            <div className="card">
              <h2 style={{ marginTop: 0, fontSize: "1rem" }}>Ensemble vs actual (aligned dates)</h2>
              <div style={{ width: "100%", height: 360 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="date" tick={{ fontSize: 9 }} minTickGap={20} />
                    <YAxis domain={["auto", "auto"]} tick={{ fontSize: 9 }} width={52} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="actual" stroke="#0f172a" dot={false} name="Actual" />
                    <Line type="monotone" dataKey="ensemble" stroke="#2563eb" dot={false} name="Ensemble" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </>
      )}
    </main>
  );
}
