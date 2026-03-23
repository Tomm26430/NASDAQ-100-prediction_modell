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
import type { OhlcvBar } from "../api/client";

type Props = {
  bars: OhlcvBar[];
  title?: string;
};

export function StockChart({ bars, title = "Close price" }: Props) {
  const data = bars.map((b) => ({
    date: b.date,
    close: b.close ?? undefined,
  }));
  return (
    <div style={{ width: "100%", height: 320 }}>
      <h3 style={{ margin: "0 0 0.5rem", fontSize: "1rem" }}>{title}</h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={24} />
          <YAxis domain={["auto", "auto"]} tick={{ fontSize: 10 }} width={56} />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="close" stroke="#2563eb" dot={false} name="Close" strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
