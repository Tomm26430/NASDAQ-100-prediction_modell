import type { TrainingStatusResponse } from "../api/client";

type Props = {
  status: TrainingStatusResponse | null;
};

export function TrainingProgress({ status }: Props) {
  if (!status || status.state === "idle") {
    return null;
  }

  const { state, percent, message, current_ticker, current_step, completed_tickers, total_tickers, started_at, finished_at } =
    status;

  const barColor =
    state === "error" ? "#dc2626" : state === "completed" ? "#16a34a" : "#2563eb";

  return (
    <div className="card" style={{ marginBottom: "1rem" }}>
      <h2 style={{ margin: "0 0 0.75rem", fontSize: "1.05rem" }}>Model training</h2>
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
          {completed_tickers} / {total_tickers} tickers · steps {status.steps_done} / {status.steps_total}
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
          {current_step ? ` · ${current_step.toUpperCase()}` : ""}
        </p>
      )}
      <p style={{ margin: "0.5rem 0 0", fontSize: "0.75rem", color: "#94a3b8" }}>
        {started_at && `Started ${started_at}`}
        {finished_at && ` · Finished ${finished_at}`}
      </p>
      {state === "error" && (
        <p style={{ margin: "0.5rem 0 0", color: "#b91c1c", fontSize: "0.9rem" }}>
          Check the API response or server logs for details.
        </p>
      )}
    </div>
  );
}
