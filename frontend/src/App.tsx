import { Route, Routes } from "react-router-dom";
import { Navbar } from "./components/Navbar";
import { Backtesting } from "./pages/Backtesting";
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
      </Routes>
    </>
  );
}
