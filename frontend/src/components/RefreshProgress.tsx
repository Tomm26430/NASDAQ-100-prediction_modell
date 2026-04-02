import type { RefreshJobStatus } from "../api/client";

type Props = {
  status: RefreshJobStatus | null;
};

export function RefreshProgress({ status }: Props) {
  if (!status || status.state === "idle") {
    return null;
  }

  const { state, percent, message, current_ticker, current_index, total_tickers, completed_tickers, started_at, finished_at } =
    status;

  const barColor = state === "error" ? "#dc2626" : state === "completed" ? "#16a34a" : "#0ea5e9";

  return (
    <div className="card" style={{ marginBottom: "1rem" }}>
      <h2 style={{ margin: "0 0 0.75rem", fontSize: "1.05rem" }}>Price refresh progress</h2>
      <div
        style={{
          height: 12,
          background: "#e2e8f0",
          borderRadius: 6,
          overflow: "hidden",
          marginBottom: "0.75rem",
        }}
      >
        <div
          style={{
            width: `${Math.min(100, Math.max(0, percent))}%`,
            height: "100%",
            background: barColor,
            transition: "width 0.35s ease",
          }}
        />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "0.5rem" }}>
        <strong style={{ fontSize: "1.1rem" }}>{percent.toFixed(1)}%</strong>
        <span style={{ color: "#64748b", fontSize: "0.9rem" }}>
          {completed_tickers} / {total_tickers} symbols
          {state === "running" && current_index > 0 && total_tickers > 0
            ? ` · now ${current_index}/${total_tickers}`
            : ""}
        </span>
      </div>
      {message && (
        <p style={{ margin: "0.5rem 0 0", color: "#334155", fontSize: "0.95rem" }}>
          {message}
        </p>
      )}
      {current_ticker && state === "running" && (
        <p style={{ margin: "0.35rem 0 0", fontSize: "0.85rem", color: "#475569" }}>
          Current: <code>{current_ticker}</code>
        </p>
      )}
      <p style={{ margin: "0.5rem 0 0", fontSize: "0.75rem", color: "#94a3b8" }}>
        {started_at && `Started ${started_at}`}
        {finished_at && ` · Finished ${finished_at}`}
      </p>
      {state === "error" && status.error_detail && (
        <p style={{ margin: "0.5rem 0 0", color: "#b91c1c", fontSize: "0.9rem", whiteSpace: "pre-wrap" }}>
          {status.error_detail}
        </p>
      )}
    </div>
  );
}
