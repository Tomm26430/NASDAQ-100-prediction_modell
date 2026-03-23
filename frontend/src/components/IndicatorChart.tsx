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
import type { IndicatorRow } from "../api/client";

type Props = { rows: IndicatorRow[] };

export function IndicatorChart({ rows }: Props) {
  const data = rows.map((r) => ({
    date: r.date,
    rsi: r.rsi_14 ?? undefined,
    macd: r.macd ?? undefined,
    signal: r.macd_signal ?? undefined,
    bbUpper: r.bb_upper ?? undefined,
    bbLower: r.bb_lower ?? undefined,
    close: r.close ?? undefined,
  }));
  return (
    <div style={{ width: "100%", height: 280 }}>
      <h3 style={{ margin: "0 0 0.5rem", fontSize: "1rem" }}>RSI &amp; Bollinger context (close)</h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="date" tick={{ fontSize: 9 }} minTickGap={28} />
          <YAxis yAxisId="left" tick={{ fontSize: 9 }} width={42} />
          <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 9 }} width={42} />
          <Tooltip />
          <Legend />
          <Line yAxisId="left" type="monotone" dataKey="rsi" stroke="#7c3aed" dot={false} name="RSI" />
          <Line yAxisId="right" type="monotone" dataKey="close" stroke="#0d9488" dot={false} name="Close" />
          <Line yAxisId="right" type="monotone" dataKey="bbUpper" stroke="#94a3b8" dot={false} name="BB upper" />
          <Line yAxisId="right" type="monotone" dataKey="bbLower" stroke="#94a3b8" dot={false} name="BB lower" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function MacdChart({ rows }: Props) {
  const data = rows.map((r) => ({
    date: r.date,
    macd: r.macd ?? undefined,
    signal: r.macd_signal ?? undefined,
    hist: r.macd_hist ?? undefined,
  }));
  return (
    <div style={{ width: "100%", height: 220 }}>
      <h3 style={{ margin: "0 0 0.5rem", fontSize: "1rem" }}>MACD</h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="date" tick={{ fontSize: 9 }} minTickGap={28} />
          <YAxis tick={{ fontSize: 9 }} width={48} />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="macd" stroke="#2563eb" dot={false} name="MACD" />
          <Line type="monotone" dataKey="signal" stroke="#f97316" dot={false} name="Signal" />
          <Line type="monotone" dataKey="hist" stroke="#64748b" dot={false} name="Hist" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
