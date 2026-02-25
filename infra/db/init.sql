CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =============================================================================
-- MARKET_RAW_MINUTE (Polygon minute OHLCV bars)
-- Columns from CSV: ticker,volume,open,close,high,low,window_start,transactions
-- =============================================================================
CREATE TABLE IF NOT EXISTS market_raw_minute (
  ticker TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  open NUMERIC(18,6) NOT NULL,
  high NUMERIC(18,6) NOT NULL,
  low NUMERIC(18,6) NOT NULL,
  close NUMERIC(18,6) NOT NULL,
  volume NUMERIC(18,2) NOT NULL,
  transactions INT NOT NULL,
  vwap NUMERIC(18,6),
  idempotency_key TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (ts, ticker)
);

SELECT create_hypertable('market_raw_minute', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS market_raw_minute_ticker_idx
  ON market_raw_minute (ticker, ts DESC);

-- =============================================================================
-- MARKET_RAW_DAY (Polygon daily OHLCV bars)
-- =============================================================================
CREATE TABLE IF NOT EXISTS market_raw_day (
  ticker TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  open NUMERIC(18,6) NOT NULL,
  high NUMERIC(18,6) NOT NULL,
  low NUMERIC(18,6) NOT NULL,
  close NUMERIC(18,6) NOT NULL,
  volume NUMERIC(18,2) NOT NULL,
  transactions INT NOT NULL,
  vwap NUMERIC(18,6),
  idempotency_key TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (ts, ticker)
);

SELECT create_hypertable('market_raw_day', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS market_raw_day_ticker_idx
  ON market_raw_day (ticker, ts DESC);

-- =============================================================================
-- FEATURES_TFT (engineered features for TFT model training)
-- =============================================================================
CREATE TABLE IF NOT EXISTS features_tft (
  ticker TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  time_idx INT NOT NULL,
  split TEXT NOT NULL DEFAULT 'train',
  -- Price features
  price NUMERIC(18,6) NOT NULL,
  open NUMERIC(18,6),
  high NUMERIC(18,6),
  low NUMERIC(18,6),
  close NUMERIC(18,6),
  volume NUMERIC(18,2),
  -- Returns
  returns_1 NUMERIC(12,8),
  returns_5 NUMERIC(12,8),
  returns_15 NUMERIC(12,8),
  target_return NUMERIC(12,8),
  -- Volatility
  volatility_5 NUMERIC(12,8),
  volatility_15 NUMERIC(12,8),
  -- Moving averages
  sma_5 NUMERIC(18,6),
  sma_20 NUMERIC(18,6),
  ema_5 NUMERIC(18,6),
  ema_20 NUMERIC(18,6),
  -- Indicators
  rsi_14 NUMERIC(8,4),
  volume_zscore NUMERIC(10,6),
  -- Temporal
  minute_of_day INT,
  hour_of_day INT,
  day_of_week INT,
  -- Social (Reddit)
  reddit_mentions_count INT DEFAULT 0,
  reddit_sentiment_mean NUMERIC(8,6) DEFAULT 0.0,
  reddit_sentiment_weighted NUMERIC(8,6) DEFAULT 0.0,
  reddit_positive_ratio NUMERIC(8,6) DEFAULT 0.0,
  reddit_negative_ratio NUMERIC(8,6) DEFAULT 0.0,
  -- Metadata
  idempotency_key TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (ts, ticker)
);

SELECT create_hypertable('features_tft', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS features_tft_ticker_idx
  ON features_tft (ticker, ts DESC);

CREATE INDEX IF NOT EXISTS features_tft_split_idx
  ON features_tft (split);

-- =============================================================================
-- REDDIT_POSTS (raw reddit posts/comments with extracted tickers)
-- =============================================================================
CREATE TABLE IF NOT EXISTS reddit_posts (
  post_id TEXT NOT NULL,
  ticker TEXT NOT NULL,
  created_utc TIMESTAMPTZ,
  subreddit TEXT,
  title TEXT,
  body TEXT,
  score INT DEFAULT 0,
  num_comments INT DEFAULT 0,
  post_type TEXT,
  source TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (post_id, ticker)
);

CREATE INDEX IF NOT EXISTS reddit_posts_ticker_idx
  ON reddit_posts (ticker, created_utc DESC);

-- =============================================================================
-- REDDIT_DAILY_MENTIONS (daily ticker mention counts + sentiment)
-- =============================================================================
CREATE TABLE IF NOT EXISTS reddit_daily_mentions (
  day DATE NOT NULL,
  ticker TEXT NOT NULL,
  mentions_count INT NOT NULL DEFAULT 0,
  sentiment_mean NUMERIC(8,6) DEFAULT 0.0,
  sentiment_weighted NUMERIC(8,6) DEFAULT 0.0,
  positive_ratio NUMERIC(8,6) DEFAULT 0.0,
  negative_ratio NUMERIC(8,6) DEFAULT 0.0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (day, ticker)
);

CREATE INDEX IF NOT EXISTS reddit_daily_mentions_ticker_idx
  ON reddit_daily_mentions (ticker, day DESC);

-- =============================================================================
-- REDDIT_MENTIONS (raw mentions with extracted tickers)
-- =============================================================================
CREATE TABLE IF NOT EXISTS reddit_mentions (
  ticker TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  source TEXT NOT NULL,
  subreddit TEXT,
  author TEXT,
  score INT DEFAULT 0,
  num_comments INT DEFAULT 0,
  post_id TEXT NOT NULL,
  permalink TEXT,
  title TEXT,
  body TEXT,
  created_utc TIMESTAMPTZ,
  raw_json JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (post_id, source, ticker)
);

CREATE INDEX IF NOT EXISTS reddit_mentions_ticker_idx
  ON reddit_mentions (ticker, ts DESC);

CREATE INDEX IF NOT EXISTS reddit_mentions_ts_idx
  ON reddit_mentions (ts DESC);

-- =============================================================================
-- PREDICTIONS_TFT (TFT model predictions)
-- =============================================================================
CREATE TABLE IF NOT EXISTS predictions_tft (
  event_id UUID NOT NULL,
  correlation_id UUID NOT NULL,
  idempotency_key TEXT NOT NULL,
  ticker TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  horizon_minutes INT NOT NULL,
  direction TEXT NOT NULL,
  confidence NUMERIC(6,4) NOT NULL,
  q10 NUMERIC(10,6),
  q50 NUMERIC(10,6),
  q90 NUMERIC(10,6),
  model_version TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (ts, idempotency_key)
);

SELECT create_hypertable('predictions_tft', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS predictions_tft_ticker_ts_idx
  ON predictions_tft (ticker, ts DESC);

CREATE INDEX IF NOT EXISTS predictions_tft_correlation_idx
  ON predictions_tft (correlation_id);

-- =============================================================================
-- PAPER_TRADES (paper execution results)
-- =============================================================================
CREATE TABLE IF NOT EXISTS paper_trades (
  event_id UUID NOT NULL,
  correlation_id UUID NOT NULL,
  idempotency_key TEXT NOT NULL,
  ticker TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  side TEXT NOT NULL,
  qty NUMERIC(18,8) NOT NULL,
  limit_price NUMERIC(18,8),
  fill_price NUMERIC(18,8),
  slippage_bps NUMERIC(10,4),
  status TEXT NOT NULL,
  reason TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (ts, idempotency_key)
);

SELECT create_hypertable('paper_trades', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS paper_trades_ticker_ts_idx
  ON paper_trades (ticker, ts DESC);

CREATE INDEX IF NOT EXISTS paper_trades_correlation_idx
  ON paper_trades (correlation_id);

CREATE INDEX IF NOT EXISTS paper_trades_status_idx
  ON paper_trades (status);

-- =============================================================================
-- AUDIT_EVENTS (pipeline trace log)
-- =============================================================================
CREATE TABLE IF NOT EXISTS audit_events (
  event_id UUID NOT NULL,
  correlation_id UUID NOT NULL,
  idempotency_key TEXT NOT NULL,
  symbol TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  schema_version INT NOT NULL,
  source TEXT NOT NULL,
  event_type TEXT NOT NULL,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (ts, idempotency_key)
);

SELECT create_hypertable('audit_events', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS audit_events_correlation_idx
  ON audit_events (correlation_id);

CREATE INDEX IF NOT EXISTS audit_events_event_type_idx
  ON audit_events (event_type);

CREATE INDEX IF NOT EXISTS audit_events_symbol_idx
  ON audit_events (symbol);

-- =============================================================================
-- RISK_DECISIONS (risk gate decisions - GAP 10.1)
-- =============================================================================
CREATE TABLE IF NOT EXISTS risk_decisions (
  event_id UUID NOT NULL,
  correlation_id UUID NOT NULL,
  idempotency_key TEXT NOT NULL,
  ticker TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  approved BOOLEAN NOT NULL,
  reason TEXT,
  risk_score NUMERIC(10,6),
  cvar_95 NUMERIC(10,6),
  max_position NUMERIC(18,8),
  limits_snapshot JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (ts, idempotency_key)
);

SELECT create_hypertable('risk_decisions', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS risk_decisions_ticker_ts_idx
  ON risk_decisions (ticker, ts DESC);

CREATE INDEX IF NOT EXISTS risk_decisions_correlation_idx
  ON risk_decisions (correlation_id);

CREATE INDEX IF NOT EXISTS risk_decisions_approved_idx
  ON risk_decisions (approved);

-- =============================================================================
-- BACKTEST_RUNS (backtest results)
-- =============================================================================
CREATE TABLE IF NOT EXISTS backtest_runs (
  run_id UUID PRIMARY KEY,
  params JSONB NOT NULL,
  metrics JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS backtest_runs_created_idx
  ON backtest_runs (created_at DESC);

-- =============================================================================
-- BACKFILL_CHECKPOINTS (resumable backfill tracking - GAP 10.2)
-- =============================================================================
CREATE TABLE IF NOT EXISTS backfill_checkpoints (
  source TEXT NOT NULL,
  filename TEXT NOT NULL,
  last_offset BIGINT NOT NULL DEFAULT 0,
  records_processed BIGINT NOT NULL DEFAULT 0,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (source, filename)
);

CREATE INDEX IF NOT EXISTS backfill_checkpoints_source_idx
  ON backfill_checkpoints (source);

-- =============================================================================
-- MODELS_REGISTRY (TFT model versions and deployment status)
-- =============================================================================
CREATE TABLE IF NOT EXISTS models_registry (
  model_id TEXT NOT NULL PRIMARY KEY,
  run_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('candidate', 'green', 'deprecated')),
  onnx_path TEXT NOT NULL,
  checkpoint_path TEXT NOT NULL,
  train_loss NUMERIC(12,8),
  val_loss NUMERIC(12,8),
  epochs INT,
  samples INT,
  train_duration_seconds NUMERIC(10,2),
  config JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  promoted_at TIMESTAMPTZ,
  deprecated_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS models_registry_status_idx
  ON models_registry (status);

CREATE INDEX IF NOT EXISTS models_registry_created_idx
  ON models_registry (created_at DESC);

-- =============================================================================
-- TRADES (executed paper trades - positions and fills)
-- =============================================================================
CREATE TABLE IF NOT EXISTS trades (
  id BIGSERIAL,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  symbol TEXT NOT NULL,
  side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
  qty INT NOT NULL,
  filled_qty INT,
  intended_notional NUMERIC(18,2),
  avg_fill_price NUMERIC(18,6),
  status TEXT NOT NULL CHECK (status IN ('filled', 'blocked', 'rejected', 'pending', 'closed')),
  alpaca_order_id TEXT,
  alpaca_status TEXT,
  order_class TEXT,
  tp_order_id TEXT,
  sl_order_id TEXT,
  exit_fill_price NUMERIC(18,6),
  exit_filled_at TIMESTAMPTZ,
  alpaca_raw JSONB,
  filled_at TIMESTAMPTZ,
  error_message TEXT,
  correlation_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  paper_trade BOOLEAN NOT NULL DEFAULT TRUE,
  dry_run BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (ts, id)
);

SELECT create_hypertable('trades', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS trades_symbol_idx ON trades (symbol, ts DESC);
CREATE INDEX IF NOT EXISTS trades_correlation_idx ON trades (correlation_id);
CREATE INDEX IF NOT EXISTS trades_status_idx ON trades (status);
CREATE INDEX IF NOT EXISTS trades_alpaca_order_idx ON trades (alpaca_order_id);

-- ===========================================================================
-- LLM TRACES (observability for all LLM calls)
-- ===========================================================================
-- Stores every LLM request/response for debugging and analytics.
-- View with: make llm-last or python scripts/show_llm_traces.py

CREATE TABLE IF NOT EXISTS llm_traces (
  id BIGSERIAL,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  correlation_id TEXT NOT NULL,
  agent_name TEXT NOT NULL,
  model TEXT NOT NULL,
  base_url TEXT NOT NULL,
  latency_ms INT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
  error_message TEXT,
  request JSONB NOT NULL,
  response JSONB NOT NULL,
  prompt_tokens INT,
  completion_tokens INT,
  PRIMARY KEY (ts, id)
);

SELECT create_hypertable('llm_traces', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS llm_traces_agent_idx ON llm_traces (agent_name, ts DESC);
CREATE INDEX IF NOT EXISTS llm_traces_model_idx ON llm_traces (model, ts DESC);
CREATE INDEX IF NOT EXISTS llm_traces_correlation_idx ON llm_traces (correlation_id);
CREATE INDEX IF NOT EXISTS llm_traces_status_idx ON llm_traces (status, ts DESC);

-- ===========================================================================
-- LLM DAILY SUMMARY (aggregated stats per day)
-- ===========================================================================

CREATE TABLE IF NOT EXISTS llm_daily_summary (
  summary_date DATE PRIMARY KEY,
  total_calls INT NOT NULL DEFAULT 0,
  error_count INT NOT NULL DEFAULT 0,
  avg_latency_ms NUMERIC(10,2) DEFAULT 0,
  p95_latency_ms NUMERIC(10,2) DEFAULT 0,
  top_agents JSONB,
  top_models JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ===========================================================================
-- POSITIONS (portfolio truth - current holdings)
-- ===========================================================================
-- Synced from Alpaca via reconcile_positions.py
-- ExecutionAgent uses this for SELL validation and exposure checks

CREATE TABLE IF NOT EXISTS positions (
  symbol TEXT PRIMARY KEY,
  qty NUMERIC(18,8) NOT NULL DEFAULT 0,
  avg_entry_price NUMERIC(18,6),
  market_value NUMERIC(18,2),
  unrealized_pl NUMERIC(18,2),
  cost_basis NUMERIC(18,2),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  raw JSONB
);

CREATE INDEX IF NOT EXISTS positions_updated_idx ON positions (updated_at DESC);

-- ===========================================================================
-- DAILY_PNL_SUMMARY (aggregated daily PnL and performance metrics)
-- ===========================================================================
-- Computed end-of-day by scripts/daily_pnl_report.py
-- View with: make pnl-today or make pnl-week

CREATE TABLE IF NOT EXISTS daily_pnl_summary (
  report_date DATE PRIMARY KEY,
  realized_pnl_usd NUMERIC(18,2) NOT NULL DEFAULT 0,
  unrealized_pnl_usd NUMERIC(18,2) NOT NULL DEFAULT 0,
  total_pnl_usd NUMERIC(18,2) NOT NULL DEFAULT 0,
  num_trades INT NOT NULL DEFAULT 0,
  num_buys INT NOT NULL DEFAULT 0,
  num_sells INT NOT NULL DEFAULT 0,
  num_wins INT NOT NULL DEFAULT 0,
  num_losses INT NOT NULL DEFAULT 0,
  win_rate NUMERIC(5,2) DEFAULT 0,
  avg_win_usd NUMERIC(18,2) DEFAULT 0,
  avg_loss_usd NUMERIC(18,2) DEFAULT 0,
  max_drawdown_usd NUMERIC(18,2) DEFAULT 0,
  best_symbol TEXT,
  best_symbol_pnl NUMERIC(18,2),
  worst_symbol TEXT,
  worst_symbol_pnl NUMERIC(18,2),
  total_volume_usd NUMERIC(18,2) DEFAULT 0,
  symbols_traded JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS daily_pnl_summary_date_idx ON daily_pnl_summary (report_date DESC);

-- ===========================================================================
-- TRADE_DECISIONS (trade signal decisions: BUY/SELL/HOLD)
-- ===========================================================================
-- Stores decision logic output from predictions -> trade signals
-- Used by backtester and paper trading execution

CREATE TABLE IF NOT EXISTS trade_decisions (
  id BIGSERIAL,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  ticker TEXT NOT NULL,
  decision TEXT NOT NULL CHECK (decision IN ('BUY', 'SELL', 'HOLD')),
  reason TEXT,
  confidence NUMERIC(6,4),
  pred_return NUMERIC(10,6),
  q10 NUMERIC(10,6),
  q50 NUMERIC(10,6),
  q90 NUMERIC(10,6),
  model_version TEXT,
  correlation_id TEXT,
  idempotency_key TEXT NOT NULL UNIQUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (ts, id)
);

SELECT create_hypertable('trade_decisions', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS trade_decisions_ticker_idx ON trade_decisions (ticker, ts DESC);
CREATE INDEX IF NOT EXISTS trade_decisions_decision_idx ON trade_decisions (decision, ts DESC);
CREATE INDEX IF NOT EXISTS trade_decisions_correlation_idx ON trade_decisions (correlation_id);

-- =============================================================================
-- Retention Policies — prevent unbounded disk growth
-- =============================================================================
-- Raw market data: 90 days (regenerable via backfill)
SELECT add_retention_policy('market_raw_minute', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('market_raw_day', INTERVAL '365 days', if_not_exists => TRUE);

-- Features: 90 days (regenerable from raw data)
SELECT add_retention_policy('features_tft', INTERVAL '90 days', if_not_exists => TRUE);

-- Predictions: 90 days
SELECT add_retention_policy('predictions_tft', INTERVAL '90 days', if_not_exists => TRUE);

-- Trades & audit: 1 year (compliance/audit requirement)
SELECT add_retention_policy('paper_trades', INTERVAL '365 days', if_not_exists => TRUE);
SELECT add_retention_policy('audit_events', INTERVAL '365 days', if_not_exists => TRUE);
SELECT add_retention_policy('risk_decisions', INTERVAL '365 days', if_not_exists => TRUE);
SELECT add_retention_policy('trade_decisions', INTERVAL '365 days', if_not_exists => TRUE);

-- LLM traces: 30 days (high volume, low long-term value)
SELECT add_retention_policy('llm_traces', INTERVAL '30 days', if_not_exists => TRUE);

-- =============================================================================
-- SIGNALS (unified ensemble signal output from SignalEngineService) [NEW]
-- =============================================================================
CREATE TABLE IF NOT EXISTS signals (
  ticker TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  tft_direction TEXT,
  tft_confidence NUMERIC(6,4),
  ensemble_direction TEXT NOT NULL,
  ensemble_confidence NUMERIC(6,4) NOT NULL,
  signal_score NUMERIC(8,6) NOT NULL,
  active_alphas JSONB,
  regime TEXT NOT NULL DEFAULT 'unknown',
  idempotency_key TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (ticker, ts)
);

SELECT create_hypertable('signals', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS signals_ticker_ts_idx  ON signals (ticker, ts DESC);
CREATE INDEX IF NOT EXISTS signals_direction_idx  ON signals (ensemble_direction, ts DESC);
CREATE INDEX IF NOT EXISTS signals_regime_idx     ON signals (regime, ts DESC);

SELECT add_retention_policy('signals', INTERVAL '90 days', if_not_exists => TRUE);

-- =============================================================================
-- WALK_FORWARD_RESULTS (model evaluation per OOS fold) [NEW]
-- =============================================================================
CREATE TABLE IF NOT EXISTS walk_forward_results (
  run_id UUID NOT NULL,
  fold INT NOT NULL,
  model_id TEXT NOT NULL,
  in_sample_start TIMESTAMPTZ NOT NULL,
  in_sample_end TIMESTAMPTZ NOT NULL,
  oos_start TIMESTAMPTZ NOT NULL,
  oos_end TIMESTAMPTZ NOT NULL,
  params JSONB,
  metrics JSONB NOT NULL,
  promoted BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, fold)
);

CREATE INDEX IF NOT EXISTS walk_forward_model_idx   ON walk_forward_results (model_id);
CREATE INDEX IF NOT EXISTS walk_forward_promoted_idx ON walk_forward_results (promoted, created_at DESC);

