import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  fetchHistory,
  fetchIndicators,
  fetchPrediction,
  type IndicatorRow,
  type OhlcvBar,
  type PredictionResponse,
} from "../api/client";
import { IndicatorChart, MacdChart } from "../components/IndicatorChart";
import { PredictionCard } from "../components/PredictionCard";
import { StockChart } from "../components/StockChart";

export function StockDetail() {
  const { ticker: raw } = useParams();
  const ticker = raw ? decodeURIComponent(raw) : "";
  const [bars, setBars] = useState<OhlcvBar[]>([]);
  const [inds, setInds] = useState<IndicatorRow[]>([]);
  const [pred, setPred] = useState<PredictionResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!ticker) return;
    let c = false;
    (async () => {
      try {
        setErr(null);
        const [h, i] = await Promise.all([fetchHistory(ticker), fetchIndicators(ticker)]);
        if (c) return;
        setBars(h.bars);
        setInds(i.rows);
        try {
          const p = await fetchPrediction(ticker);
          if (!c) setPred(p);
        } catch {
          if (!c) setPred(null);
        }
      } catch (e: unknown) {
        if (!c) setErr(e instanceof Error ? e.message : "Load failed");
      }
    })();
    return () => {
      c = true;
    };
  }, [ticker]);

  if (!ticker) return <main>Missing ticker</main>;

  return (
    <main>
      <p>
        <Link to="/">← Dashboard</Link>
      </p>
      <h1>{ticker}</h1>
      {err && <p className="err">{err}</p>}
      {!err && bars.length > 0 && (
        <div className="card">
          <StockChart bars={bars} />
        </div>
      )}
      {!err && inds.length > 0 && (
        <>
          <div className="card">
            <IndicatorChart rows={inds} />
          </div>
          <div className="card">
            <MacdChart rows={inds} />
          </div>
        </>
      )}
      <div className="card">
        <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Ensemble forecast</h2>
        {!pred && (
          <p style={{ color: "#64748b" }}>
            No trained LSTM yet for this symbol. Use <strong>Train models</strong> on the dashboard, wait for
            completion, then reload.
          </p>
        )}
        {pred && (
          <>
            <p style={{ color: "#475569", fontSize: "0.9rem" }}>
              Last close {pred.last_close.toFixed(2)} · blend {pred.weights.lstm.toFixed(2)} LSTM /{" "}
              {pred.weights.arima.toFixed(2)} ARIMA
            </p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem" }}>
              <PredictionCard days={7} lastClose={pred.last_close} block={pred.horizons["7"]} />
              <PredictionCard days={30} lastClose={pred.last_close} block={pred.horizons["30"]} />
              <PredictionCard days={90} lastClose={pred.last_close} block={pred.horizons["90"]} />
            </div>
          </>
        )}
      </div>
    </main>
  );
}
