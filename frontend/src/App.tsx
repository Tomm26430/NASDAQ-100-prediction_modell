import { Route, Routes } from "react-router-dom";
import { Navbar } from "./components/Navbar";
import { Backtesting } from "./pages/Backtesting";
import { BacktestCompare } from "./pages/BacktestCompare";
import { BacktestHistory } from "./pages/BacktestHistory";
import { BacktestRunDetail } from "./pages/BacktestRunDetail";
import { BacktestTickerDetail } from "./pages/BacktestTickerDetail";
import { Dashboard } from "./pages/Dashboard";
import { StockDetail } from "./pages/StockDetail";

export default function App() {
  return (
    <>
      <Navbar />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/stock/:ticker" element={<StockDetail />} />
        <Route path="/backtest" element={<Backtesting />} />
        <Route path="/backtest/history" element={<BacktestHistory />} />
        <Route path="/backtest/compare" element={<BacktestCompare />} />
        <Route path="/backtest/runs/:runId" element={<BacktestRunDetail />} />
        <Route path="/backtest/runs/:runId/ticker/:ticker" element={<BacktestTickerDetail />} />
      </Routes>
    </>
  );
}
