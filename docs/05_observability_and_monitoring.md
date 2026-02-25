# 05 — Observability & Monitoring Plan

---

## Structured Logging

All services use **structlog** with JSON output and correlation ID propagation.

```python
logger.info("event_processed",
    correlation_id=event.correlation_id,
    symbol=event.symbol,
    latency_ms=elapsed,
    agent="signal_engine",
    step="ensemble_vote")
```

**Required fields in every log line:**
- `correlation_id` — traces event end-to-end from ingest → execution
- `agent` — service name
- `ts` — ISO 8601 timestamp
- `level` — debug/info/warning/error/critical
- `event` — structured event name (snake_case)

---

## Prometheus Metrics (13 gauges/counters/histograms)

| Metric | Type | Labels | Source |
|--------|------|--------|--------|
| `events_processed_total` | Counter | agent, topic | All consumers |
| `events_failed_total` | Counter | agent, error_type | All consumers |
| `event_latency_seconds` | Histogram | agent, step | All consumers |
| `kafka_consumer_lag_total` | Gauge | group_id, topic, partition | MonitoringService |
| `risk_decisions_total` | Counter | outcome (approved/rejected), reason | RiskEngine |
| `model_confidence_score` | Histogram | model_id | ModelInference |
| `model_version_info` | Gauge (info) | model_id, status | ModelInference |
| `position_count` | Gauge | none | Execution |
| `daily_pnl_usd` | Gauge | none | Monitoring |
| `kill_switch_active` | Gauge | none | RiskEngine |
| `backtest_sharpe_ratio` | Gauge | model_id, fold | WalkForward |
| `execution_slippage_bps` | Histogram | symbol | Execution |
| `tool_gateway_requests_total` | Counter | tool, status | Orchestration |

---

## Prometheus Alert Rules (11)

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| `ConsumerLagHigh` | lag > 100 for 5m | warning | Investigate |
| `ConsumerLagCritical` | lag > 1000 for 2m | critical | Page on-call |
| `AgentDown` | up == 0 for 1m | critical | Auto-restart + page |
| `KillSwitchActivated` | kill_switch == 1 | critical | Halt all trading |
| `DailyLossWarning` | P&L < -$3000 | warning | Reduce position size |
| `DailyLossCritical` | P&L < -$5000 | critical | Approaching kill switch |
| `ModelConfidenceDrop` | p50(confidence) < 0.2 for 30m | warning | Model drift — retrain |
| `AlpacaAPIErrors` | error rate > 5% for 5m | critical | Execution degraded |
| `PipelineStale` | no events 10m in market hours | critical | Data feed down |
| `SlippageAbnormal` | avg slippage > 15 bps for 10m | warning | 3× expected |
| `ModelOOSSharpeLow` | OOS Sharpe < 0 | warning | Block promotion |

---

## Grafana Dashboards (5)

### 1. Pipeline Health
- Event throughput (events/sec per agent)
- Consumer lag per topic
- Error rate
- End-to-end latency (ingest → signal → execution)

### 2. Risk Dashboard
- Daily P&L curve
- Kill switch status
- Position count
- Sector exposure pie chart
- CVaR-95 value
- Drawdown from peak

### 3. Signal Quality
- Ensemble confidence distribution
- Signal score distribution
- Regime state (Bull/Bear/Sideways) timeline
- Alpha agreement rate (how often Lean alphas agree with TFT)
- Win rate rolling 20-trade

### 4. Equity Curve
- Cumulative P&L
- Sharpe ratio rolling 30-day
- Max drawdown overlay
- Benchmark comparison (SPY)

### 5. Model Monitor
- Training loss per fold
- OOS Sharpe per fold
- Model version timeline (candidate → green → deprecated)
- Prediction confidence histogram
- Feature importance heatmap

---

## Audit Trail

Every pipeline event is persisted to `audit_events` with:
```sql
(event_id UUID, correlation_id UUID, idempotency_key TEXT,
 symbol TEXT, ts TIMESTAMPTZ, schema_version INT,
 source TEXT, event_type TEXT, payload JSONB)
```

**Retention**: 365 days. **Purpose**: Full reconstructability of any trading decision.

---

## Alerting Routes

| Channel | Severity | Examples |
|---------|----------|---------|
| Telegram | critical | Kill switch, agent down, API failure |
| Email | warning | Consumer lag, model drift, slippage |
| Grafana annotation | info | Model promoted, backfill completed |
