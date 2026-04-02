import { Link } from "react-router-dom";

export function Navbar() {
  return (
    <header
      style={{
        background: "#0f172a",
        color: "#f8fafc",
        padding: "0.75rem 1.5rem",
        display: "flex",
        alignItems: "center",
        gap: "1.25rem",
      }}
    >
      <strong>NASDAQ Predictor</strong>
      <nav style={{ display: "flex", gap: "1rem" }}>
        <Link to="/" style={{ color: "#93c5fd" }}>
          Dashboard
        </Link>
        <Link to="/backtest" style={{ color: "#93c5fd" }}>
          Backtesting
        </Link>
        <Link to="/backtest/history" style={{ color: "#93c5fd" }}>
          Saved tests
        </Link>
        <Link to="/backtest/compare" style={{ color: "#93c5fd" }}>
          Compare
        </Link>
      </nav>
    </header>
  );
}
