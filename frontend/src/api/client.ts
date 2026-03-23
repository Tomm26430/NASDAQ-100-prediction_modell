import axios from "axios";

/** Empty baseURL: Vite dev server proxies `/api` to FastAPI (see vite.config.ts). */
export const api = axios.create({ baseURL: "" });

export type StockRow = {
  ticker: string;
  last_close: number | null;
  last_trade_date: string | null;
  currency: string;
};

export type StockListResponse = {
  stocks: StockRow[];
  count: number;
  light_mode: boolean;
  last_price_refresh_utc: string | null;
};

export async function fetchStockList(): Promise<StockListResponse> {
  const { data } = await api.get<StockListResponse>("/api/stocks");
  return data;
}

export type OhlcvBar = {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
};

export async function fetchHistory(ticker: string): Promise<{ ticker: string; bars: OhlcvBar[] }> {
  const enc = encodeURIComponent(ticker);
  const { data } = await api.get(`/api/stocks/${enc}/history`);
  return data;
}

export type IndicatorRow = {
  date: string;
  close: number | null;
  rsi_14: number | null;
  macd: number | null;
  macd_signal: number | null;
  macd_hist: number | null;
  bb_middle: number | null;
  bb_upper: number | null;
  bb_lower: number | null;
};

export async function fetchIndicators(ticker: string): Promise<{ ticker: string; rows: IndicatorRow[] }> {
  const enc = encodeURIComponent(ticker);
  const { data } = await api.get(`/api/stocks/${enc}/indicators`);
  return data;
}

export type HorizonBlock = {
  ensemble: number;
  lstm: number;
  arima: number;
  ci_low: number;
  ci_high: number;
  arima_ci_low: number;
  arima_ci_high: number;
};

export type PredictionResponse = {
  ticker: string;
  last_close: number;
  weights: { lstm: number; arima: number };
  horizons: Record<"7" | "30" | "90", HorizonBlock>;
};

export async function fetchPrediction(ticker: string): Promise<PredictionResponse> {
  const enc = encodeURIComponent(ticker);
  const { data } = await api.get<PredictionResponse>(`/api/stocks/${enc}/prediction`);
  return data;
}

export type BacktestResponse = {
  ticker: string;
  holdout_trading_days: number;
  metrics: {
    arima_one_step: Record<string, number>;
    lstm_one_step: Record<string, number>;
    ensemble: Record<string, number>;
  };
  series: {
    arima: { date: string; actual: number; predicted: number }[];
    lstm: { date: string; actual: number; predicted: number }[];
    ensemble: {
      date: string;
      actual: number;
      predicted_ensemble: number;
      predicted_lstm: number;
      predicted_arima: number;
    }[];
  };
};

export async function fetchBacktest(ticker: string): Promise<BacktestResponse> {
  const enc = encodeURIComponent(ticker);
  const { data } = await api.get<BacktestResponse>(`/api/stocks/${enc}/backtest`);
  return data;
}

export async function postRefresh(): Promise<void> {
  await api.post("/api/admin/refresh");
}

export async function postTrainModels(): Promise<void> {
  await api.post("/api/admin/train-models");
}
