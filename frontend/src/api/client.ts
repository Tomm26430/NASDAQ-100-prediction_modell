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

/** One row per chart point; `predicted` is the ensemble (or scenario-specific) forecast vs `actual`. */
export type BacktestSeriesRow = {
  date: string;
  actual: number;
  predicted: number;
};

export type BacktestResponse = {
  ticker: string;
  scenario: number;
  scenario_label: string;
  holdout_trading_days: number;
  holdout_trading_days_target?: number;
  holdout_note?: string | null;
  holdout_years_requested?: number | null;
  holdout_years_actual?: number;
  metrics: Record<string, unknown>;
  series: BacktestSeriesRow[];
  /** Scenario 5 only: Scenario 2–style price metrics and series (same single model pass). */
  price_accuracy?: {
    metrics: Record<string, unknown>;
    series: BacktestSeriesRow[];
  };
  /** Scenario 5 only: direction hit rates (fractions 0–1) plus headline_7d_percent (0–100). */
  direction_accuracy?: Record<string, unknown>;
  combined_verdict?: string;
  verdict_explanation?: string;
};

export type BacktestQueryOpts = {
  /** Holdout depth in years (min 1). Omit to use server BACKTEST_YEARS default (unless maxHoldout). */
  years?: number;
  /** Longest holdout allowed by cached history (ignores years). */
  maxHoldout?: boolean;
};

export async function fetchBacktest(
  ticker: string,
  scenario: number = 1,
  opts?: BacktestQueryOpts,
): Promise<BacktestResponse> {
  const enc = encodeURIComponent(ticker);
  const params: Record<string, number | boolean> = { scenario };
  if (opts?.maxHoldout) params.max_holdout = true;
  else if (opts?.years != null) params.years = opts.years;
  const { data } = await api.get<BacktestResponse>(`/api/stocks/${enc}/backtest`, { params });
  return data;
}

export type BulkBacktestRow = {
  status: string;
  combined_verdict: string | null;
  mape_30d: number;
  direction_accuracy_7d: number;
  holdout_days: number;
};

export type BulkBacktestResponse = {
  scenario: number;
  years: number | null;
  max_holdout: boolean;
  tickers_tested: string[];
  results: Record<string, BulkBacktestRow>;
  aggregate: Record<string, unknown>;
};

export async function postBacktestAll(
  scenario: number = 5,
  opts?: { years?: number; maxHoldout?: boolean },
): Promise<BulkBacktestResponse> {
  const params: Record<string, number | boolean> = { scenario };
  if (opts?.maxHoldout) params.max_holdout = true;
  else if (opts?.years != null) params.years = opts.years;
  const { data } = await api.post<BulkBacktestResponse>("/api/admin/backtest-all", {}, { params });
  return data;
}

export async function postRefresh(): Promise<void> {
  await api.post("/api/admin/refresh");
}

export type TrainingStatusResponse = {
  state: "idle" | "running" | "completed" | "error";
  total_tickers: number;
  completed_tickers: number;
  steps_total: number;
  steps_done: number;
  percent: number;
  current_ticker: string | null;
  current_step: string | null;
  message: string | null;
  started_at: string | null;
  finished_at: string | null;
  tickers_queue: string[];
  last_results: unknown[];
};

export async function fetchTrainingStatus(): Promise<TrainingStatusResponse> {
  const { data } = await api.get<TrainingStatusResponse>("/api/admin/training-status");
  return data;
}

export async function postTrainModels(): Promise<{ status: string; detail: string }> {
  const { data } = await api.post<{ status: string; detail: string }>("/api/admin/train-models");
  return data;
}
