# 04 — Data Contracts & Schemas

---

## Kafka Topics & Avro Schemas

### market.raw
```json
{
  "type": "record",
  "name": "MarketRawEvent",
  "namespace": "unified.market",
  "fields": [
    {"name": "event_id", "type": "string"},
    {"name": "symbol", "type": "string"},
    {"name": "ts", "type": {"type": "long", "logicalType": "timestamp-millis"}},
    {"name": "open", "type": "double"},
    {"name": "high", "type": "double"},
    {"name": "low", "type": "double"},
    {"name": "close", "type": "double"},
    {"name": "volume", "type": "long"},
    {"name": "source", "type": "string"},
    {"name": "bar_type", "type": "string"}
  ]
}
```

### market.engineered
```json
{
  "type": "record",
  "name": "MarketEngineeredEvent",
  "namespace": "unified.features",
  "fields": [
    {"name": "event_id", "type": "string"},
    {"name": "symbol", "type": "string"},
    {"name": "ts", "type": {"type": "long", "logicalType": "timestamp-millis"}},
    {"name": "price", "type": "double"},
    {"name": "volume", "type": "long"},
    {"name": "returns_1", "type": "double"},
    {"name": "returns_5", "type": "double"},
    {"name": "returns_15", "type": "double"},
    {"name": "volatility_5", "type": "double"},
    {"name": "volatility_15", "type": "double"},
    {"name": "sma_5", "type": "double"},
    {"name": "sma_20", "type": "double"},
    {"name": "ema_5", "type": "double"},
    {"name": "ema_20", "type": "double"},
    {"name": "rsi_14", "type": "double"},
    {"name": "volume_zscore", "type": "double"},
    {"name": "minute_of_day", "type": "int"},
    {"name": "day_of_week", "type": "int"}
  ]
}
```

### alpha.signals (NEW)
```json
{
  "type": "record",
  "name": "AlphaSignalEvent",
  "namespace": "unified.alpha",
  "fields": [
    {"name": "event_id", "type": "string"},
    {"name": "symbol", "type": "string"},
    {"name": "ts", "type": {"type": "long", "logicalType": "timestamp-millis"}},
    {"name": "indicator", "type": "string"},
    {"name": "direction", "type": "string"},
    {"name": "weight", "type": "double"},
    {"name": "raw_value", "type": "double"}
  ]
}
```

### signals.scored (NEW)
```json
{
  "type": "record",
  "name": "ScoredSignalEvent",
  "namespace": "unified.signals",
  "fields": [
    {"name": "event_id", "type": "string"},
    {"name": "correlation_id", "type": "string"},
    {"name": "symbol", "type": "string"},
    {"name": "ts", "type": {"type": "long", "logicalType": "timestamp-millis"}},
    {"name": "tft_direction", "type": "string"},
    {"name": "tft_confidence", "type": "double"},
    {"name": "ensemble_direction", "type": "string"},
    {"name": "ensemble_confidence", "type": "double"},
    {"name": "signal_score", "type": "double"},
    {"name": "regime", "type": "string"},
    {"name": "active_alphas", "type": "string"}
  ]
}
```

---

## TimescaleDB Tables

### Existing (from AWET — 20 tables)

| Table | Purpose | Hypertable? |
|-------|---------|-------------|
| `market_raw_minute` | Minute-bar OHLCV | Yes (ts) |
| `market_raw_day` | Daily-bar OHLCV | Yes (ts) |
| `features_tft` | Engineered features | Yes (ts) |
| `predictions_tft` | TFT quantile predictions | Yes (ts) |
| `paper_trades` | Active paper trade positions | No |
| `trades` | Completed trade history | Yes (executed_at) |
| `positions` | Current portfolio positions | No |
| `audit_events` | Full pipeline audit trail | Yes (ts) |
| `risk_decisions` | Risk engine approve/reject log | Yes (ts) |
| `trade_decisions` | Signal engine decision log | Yes (ts) |
| `backtest_runs` | Backtest parameter and result log | No |
| `backfill_checkpoints` | Backfill position tracking | No |
| `models_registry` | Model lifecycle tracking | No |
| `reddit_posts` | Reddit post content | Yes (created_at) |
| `reddit_daily_mentions` | Daily mention aggregates | Yes (ts) |
| `reddit_mentions` | Individual mentions | Yes (ts) |
| `llm_traces` | LLM API call logs | Yes (ts) |
| `llm_daily_summary` | Daily LLM usage summary | No |
| `daily_pnl_summary` | End-of-day P&L | No |

### New Tables (Phase 1-2)

```sql
-- Ensemble signal output
CREATE TABLE signals (
    ticker        TEXT        NOT NULL,
    ts            TIMESTAMPTZ NOT NULL,
    tft_direction TEXT,
    tft_confidence DOUBLE PRECISION,
    ensemble_direction TEXT   NOT NULL,
    ensemble_confidence DOUBLE PRECISION NOT NULL,
    signal_score  DOUBLE PRECISION NOT NULL,
    active_alphas JSONB,
    regime        TEXT DEFAULT 'unknown',
    idempotency_key TEXT UNIQUE,
    PRIMARY KEY (ticker, ts)
);
SELECT create_hypertable('signals','ts');

-- Walk-forward validation results
CREATE TABLE walk_forward_results (
    run_id UUID            NOT NULL,
    fold   INT             NOT NULL,
    model_id TEXT,
    in_sample_start TIMESTAMPTZ,
    in_sample_end   TIMESTAMPTZ,
    oos_start       TIMESTAMPTZ,
    oos_end         TIMESTAMPTZ,
    params JSONB,
    metrics JSONB,
    promoted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (run_id, fold)
);
```

### Retention Policies

| Table | Retention | Reason |
|-------|-----------|--------|
| `market_raw_minute` | 90 days | Only need recent for live pipeline |
| `features_tft` | 90 days | Regenerable from raw data |
| `signals` | 180 days | Need for model evaluation lookback |
| `audit_events` | 365 days | Compliance requirement |
| `walk_forward_results` | Indefinite | Research state |

---

## Redis Keys

| Key Pattern | TTL | Purpose |
|-------------|-----|---------|
| `risk:state:{date}` | 24h | Daily risk engine state (positions, P&L) |
| `risk:kill_switch` | None | Kill switch flag (persist across restarts) |
| `model:green:{model_id}` | None | Current green model metadata |
| `throttle:orders:{symbol}` | 60s | Order throttle counter |
