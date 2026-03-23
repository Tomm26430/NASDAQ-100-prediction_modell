import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchPrediction, fetchStockList, postRefresh, postTrainModels, type StockRow } from "../api/client";

type RowExtra = StockRow & { pred7?: number | null; predErr?: string };

export function Dashboard() {
  const [rows, setRows] = useState<RowExtra[]>([]);
  const [light, setLight] = useState(true);
  const [refreshedAt, setRefreshedAt] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setLoading(true);
        const data = await fetchStockList();
        if (cancelled) return;
        setRows(data.stocks);
        setLight(data.light_mode);
        setRefreshedAt(data.last_price_refresh_utc);
        const withPrice = data.stocks.filter((s) => s.last_close != null).slice(0, 12);
        const next: RowExtra[] = [...data.stocks];
        await Promise.all(
          withPrice.map(async (s) => {
            try {
              const p = await fetchPrediction(s.ticker);
              const i = next.findIndex((r) => r.ticker === s.ticker);
              if (i >= 0) next[i] = { ...next[i], pred7: p.horizons["7"].ensemble };
            } catch {
              const i = next.findIndex((r) => r.ticker === s.ticker);
              if (i >= 0) next[i] = { ...next[i], predErr: "no model" };
            }
          }),
        );
        if (!cancelled) setRows(next);
      } catch (e: unknown) {
        if (!cancelled) setErr(e instanceof Error ? e.message : "Failed to load stocks");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function onRefresh() {
    try {
      setBusy("Refreshing prices…");
      await postRefresh();
      setBusy("Refresh queued. Reload in a few seconds.");
    } catch (e: unknown) {
      setBusy(e instanceof Error ? e.message : "Refresh failed");
    }
  }

  async function onTrain() {
    try {
      setBusy("Training queued (check terminal logs)…");
      await postTrainModels();
      setBusy("Training started. Wait several minutes, then reload.");
    } catch (e: unknown) {
      setBusy(e instanceof Error ? e.message : "Train request failed");
    }
  }

  return (
    <main>
      <h1>Dashboard</h1>
      <p style={{ color: "#475569", marginTop: "-0.5rem" }}>
        Cached NASDAQ-100 constituents + index. Light mode:{" "}
        <strong>{light ? "on (subset only)" : "off"}</strong>
        {refreshedAt ? ` · last refresh ${refreshedAt}` : ""}
      </p>
      <div style={{ display: "flex", gap: "0.75rem", marginBottom: "1rem", flexWrap: "wrap" }}>
        <button type="button" onClick={onRefresh}>
          Refresh prices
        </button>
        <button type="button" onClick={onTrain}>
          Train models (active tickers)
        </button>
      </div>
      {busy && <p className="err">{busy}</p>}
      {err && <p className="err">{err}</p>}
      {loading && <p>Loading…</p>}
      {!loading && !err && (
        <div className="card" style={{ padding: 0, overflow: "auto" }}>
          <table className="data">
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Last close</th>
                <th>As of</th>
                <th>7d ensemble (est.)</th>
                <th>Trend</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {rows.map((s) => {
                const pct =
                  s.last_close && s.pred7 != null
                    ? ((s.pred7 - s.last_close) / s.last_close) * 100
                    : null;
                const up = pct != null && pct >= 0;
                return (
                  <tr key={s.ticker}>
                    <td>
                      <strong>{s.ticker}</strong>
                    </td>
                    <td>{s.last_close?.toFixed(2) ?? "—"}</td>
                    <td>{s.last_trade_date ?? "—"}</td>
                    <td>
                      {s.pred7 != null ? s.pred7.toFixed(2) : s.predErr === "no model" ? "—" : "—"}
                    </td>
                    <td>
                      {pct == null ? "—" : (
                        <span style={{ color: up ? "#16a34a" : "#dc2626" }}>
                          {up ? "▲" : "▼"} {pct.toFixed(2)}%
                        </span>
                      )}
                    </td>
                    <td>
                      <Link to={`/stock/${encodeURIComponent(s.ticker)}`}>Analysis</Link>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}
