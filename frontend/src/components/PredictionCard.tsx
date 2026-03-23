import type { HorizonBlock } from "../api/client";

type Props = {
  days: 7 | 30 | 90;
  lastClose: number;
  block: HorizonBlock;
};

export function PredictionCard({ days, lastClose, block }: Props) {
  const pct = lastClose > 0 ? ((block.ensemble - lastClose) / lastClose) * 100 : 0;
  const up = pct >= 0;
  return (
    <div
      className="card"
      style={{
        flex: "1 1 180px",
        minWidth: 160,
        borderLeft: `4px solid ${up ? "#16a34a" : "#dc2626"}`,
      }}
    >
      <div style={{ fontSize: "0.8rem", color: "#64748b" }}>{days}-day horizon</div>
      <div style={{ fontSize: "1.25rem", fontWeight: 700 }}>{block.ensemble.toFixed(2)}</div>
      <div style={{ fontSize: "0.85rem", marginTop: 4 }}>
        Δ vs last close:{" "}
        <span style={{ color: up ? "#16a34a" : "#dc2626" }}>
          {up ? "▲" : "▼"} {pct.toFixed(2)}%
        </span>
      </div>
      <div style={{ fontSize: "0.75rem", color: "#64748b", marginTop: 8 }}>
        CI [{block.ci_low.toFixed(2)} – {block.ci_high.toFixed(2)}]
      </div>
      <div style={{ fontSize: "0.7rem", color: "#94a3b8", marginTop: 4 }}>
        LSTM {block.lstm.toFixed(2)} · ARIMA {block.arima.toFixed(2)}
      </div>
    </div>
  );
}
