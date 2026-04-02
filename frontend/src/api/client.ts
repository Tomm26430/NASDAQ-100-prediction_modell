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
  /** Save result to SQLite (default true). */
  persist?: boolean;
};

export async function fetchBacktest(
  ticker: string,
  scenario: number = 1,
  opts?: BacktestQueryOpts,
): Promise<BacktestResponse & { saved_run_id?: number | null }> {
  const enc = encodeURIComponent(ticker);
  const params: Record<string, number | boolean> = { scenario };
  if (opts?.maxHoldout) params.max_holdout = true;
  else if (opts?.years != null) params.years = opts.years;
  if (opts?.persist === false) params.persist = false;
  const { data } = await api.get<BacktestResponse & { saved_run_id?: number | null }>(`/api/stocks/${enc}/backtest`, { params });
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
  saved_run_id?: number | null;
  holdout_years_requested?: number | null;
  holdout_years_actual?: number | null;
};

export type BacktestJobStatus = {
  state: "idle" | "running" | "completed" | "error";
  total_tickers: number;
  completed_tickers: number;
  percent: number;
  current_ticker: string | null;
  current_index: number;
  message: string | null;
  started_at: string | null;
  finished_at: string | null;
  result: BulkBacktestResponse | null;
  error_detail: string | null;
};

export async function postBacktestAllStart(
  scenario: number = 5,
  opts?: { years?: number; maxHoldout?: boolean; persist?: boolean },
): Promise<{ status: string; detail: string; tickers: string[] }> {
  const params: Record<string, number | boolean> = { scenario };
  if (opts?.maxHoldout) params.max_holdout = true;
  else if (opts?.years != null) params.years = opts.years;
  if (opts?.persist === false) params.persist = false;
  const { data } = await api.post<{ status: string; detail: string; tickers: string[] }>(
    "/api/admin/backtest-all",
    {},
    { params },
  );
  return data;
}

export async function fetchBacktestJobStatus(): Promise<BacktestJobStatus> {
  const { data } = await api.get<BacktestJobStatus>("/api/admin/backtest-status");
  return data;
}

export type SavedRunListItem = {
  id: number;
  created_at: string | null;
  run_type: string;
  scenario: number;
  years_requested: number | null;
  max_holdout: boolean;
  holdout_years_requested: number | null;
  holdout_years_actual: number | null;
  source_ticker: string | null;
  ticker_count: number;
};

export async function fetchSavedRuns(): Promise<{ runs: SavedRunListItem[] }> {
  const { data } = await api.get<{ runs: SavedRunListItem[] }>("/api/backtest-runs");
  return data;
}

export type SavedRunDetail = {
  id: number;
  created_at: string | null;
  run_type: string;
  scenario: number;
  years_requested: number | null;
  max_holdout: boolean;
  holdout_years_requested: number | null;
  holdout_years_actual: number | null;
  source_ticker: string | null;
  aggregate: Record<string, unknown> | null;
  tickers: Array<{
    ticker: string;
    status: string;
    combined_verdict: string | null;
    mape_30d: number | null;
    direction_accuracy_7d: number | null;
    holdout_days: number;
    has_detail: boolean;
  }>;
};

export async function fetchSavedRun(runId: number): Promise<SavedRunDetail> {
  const { data } = await api.get<SavedRunDetail>(`/api/backtest-runs/${runId}`);
  return data;
}

export async function fetchSavedTickerDetail(runId: number, ticker: string): Promise<BacktestResponse> {
  const enc = encodeURIComponent(ticker);
  const { data } = await api.get<BacktestResponse>(`/api/backtest-runs/${runId}/ticker/${enc}`);
  return data;
}

export type CompareRunsResponse = {
  run_a: Record<string, unknown>;
  run_b: Record<string, unknown>;
  common_tickers: string[];
  comparison_rows: Array<{
    ticker: string;
    run_a: { verdict?: string | null; mape_30d?: number | null; direction_accuracy_7d?: number | null; status?: string };
    run_b: { verdict?: string | null; mape_30d?: number | null; direction_accuracy_7d?: number | null; status?: string };
    delta_mape_30d: number | null;
    delta_direction_7d: number | null;
  }>;
};

export async function fetchCompareRuns(a: number, b: number): Promise<CompareRunsResponse> {
  const { data } = await api.get<CompareRunsResponse>("/api/backtest-runs/compare/summary", { params: { a, b } });
  return data;
}

export async function deleteSavedRun(runId: number): Promise<void> {
  await api.delete(`/api/backtest-runs/${runId}`);
}

export type RefreshJobStatus = {
  state: "idle" | "running" | "completed" | "error";
  total_tickers: number;
  completed_tickers: number;
  percent: number;
  current_ticker: string | null;
  current_index: number;
  message: string | null;
  started_at: string | null;
  finished_at: string | null;
  result: Record<string, unknown> | null;
  error_detail: string | null;
};

export async function fetchRefreshJobStatus(): Promise<RefreshJobStatus> {
  const { data } = await api.get<RefreshJobStatus>("/api/admin/refresh-status");
  return data;
}

export async function postRefresh(): Promise<{ status: string; detail: string; tickers: string[] }> {
  const { data } = await api.post<{ status: string; detail: string; tickers: string[] }>("/api/admin/refresh");
  return data;
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
