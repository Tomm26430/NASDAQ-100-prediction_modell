import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { fetchSavedTickerDetail, type BacktestResponse } from "../api/client";
import { Scenario5Panels } from "../components/Scenario5Panels";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export function BacktestTickerDetail() {
  const { runId, ticker: tickerEnc } = useParams();
  const id = runId ? parseInt(runId, 10) : NaN;
  const ticker = tickerEnc ? decodeURIComponent(tickerEnc) : "";
  const [data, setData] = useState<BacktestResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!Number.isFinite(id) || !ticker) return;
    let c = false;
    (async () => {
      try {
        const d = await fetchSavedTickerDetail(id, ticker);
        if (!c) setData(d);
      } catch (e: unknown) {
        if (!c) setErr(e instanceof Error ? e.message : "Load failed");
      }
    })();
    return () => {
      c = true;
    };
  }, [id, ticker]);

  return (
    <main>
      <p>
        <Link to={`/backtest/runs/${runId}`}>← Run #{runId}</Link>
      </p>
      <h1>
        {ticker} <span style={{ fontSize: "0.85rem", color: "#64748b" }}>(saved)</span>
      </h1>
      {err && <p className="err">{err}</p>}
      {data && (
        <>
          <p style={{ color: "#64748b", fontSize: "0.9rem" }}>
            {data.scenario_label} · {data.holdout_trading_days} trading days holdout
            {data.holdout_years_actual != null ? ` · ${data.holdout_years_actual.toFixed(2)}y actual` : ""}
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
                <h2 style={{ marginTop: 0, fontSize: "1rem" }}>Metrics</h2>
                <pre style={{ fontSize: "0.8rem", overflow: "auto" }}>{JSON.stringify(data.metrics, null, 2)}</pre>
              </div>
              {data.series?.length ? (
                <div className="card">
                  <h2 style={{ marginTop: 0, fontSize: "1rem" }}>Predicted vs actual</h2>
                  <div style={{ width: "100%", height: 360 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={data.series.map((r) => ({ date: r.date, actual: r.actual, predicted: r.predicted }))}
                        margin={{ top: 8, right: 16, left: 0, bottom: 0 }}
                      >
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
              ) : null}
            </>
          )}
        </>
      )}
    </main>
  );
}
