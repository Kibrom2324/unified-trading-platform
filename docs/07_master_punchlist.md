# 07 — Master Punchlist (Hardening)

> **Mode**: Hardening only. No new features, no redesign.
> **Sources**: Quant Risk Review, SRE Reliability Review, Completeness Audit, code inspection of all service implementations.
> **Date**: 2026-02-25

---

## Definition of Done

All items in CRITICAL and HIGH must satisfy these gates before paper trading:

- [ ] **DoD-1**: Every CRITICAL item has a merged fix with passing test (T-# referenced)
- [ ] **DoD-2**: Every HIGH item has a merged fix with passing test
- [ ] **DoD-3**: All 13 existing + 11 new Prometheus metrics emit non-zero values in local docker-compose
- [ ] **DoD-4**: `PipelineStale` alert does NOT fire outside market hours (verify with `promtool test rules`)
- [ ] **DoD-5**: Risk engine survives Redis restart without losing kill switch state (integration test)
- [ ] **DoD-6**: Walk-forward OOS Sharpe is computed at correct bar frequency (unit test with synthetic data)
- [ ] **DoD-7**: No duplicate YAML keys in any config file (CI lint with `yamllint`)
- [ ] **DoD-8**: `pytest tests/` passes with 0 failures, 0 errors
- [ ] **DoD-9**: `docker compose up` starts all 13 services to healthy within 120s
- [ ] **DoD-10**: Manual walkthrough: ingest → features → inference → signal → risk → execution → exit produces a valid `execution.completed` event on `paper_trades` table

---

## Table of Contents

1. [CRITICAL — Must fix before any paper trading](#1-critical)
2. [HIGH — Must fix before declaring backtest results valid](#2-high)
3. [MEDIUM — Fix before production monitoring is trustworthy](#3-medium)
4. [NICE-TO-HAVE — Improvements for future phases](#4-nice-to-have)
5. [Test Matrix](#5-test-matrix)

---

## 1. CRITICAL

### CF-1: Walk-forward selects BEST fold, not LATEST fold (selection bias)

| Field | Value |
|-------|-------|
| **Description** | `select_best()` picks the fold with the highest OOS Sharpe across all 12 folds. This is selection bias — deploying the model that got lucky, not the one trained on the most recent data. Defeats the purpose of walk-forward validation. |
| **File:Line** | [walk_forward.py:301-302](file:///home/kironix/workspace/both/unified-trading-platform/services/model_training/walk_forward.py#L301-L302) |
| **Acceptance Criteria** | `select_best()` returns the model from the **highest `fold_number`** (most recent fold). Median OOS Sharpe across all folds remains the quality gate. The best-fold Sharpe is logged but not used for selection. |
| **Test** | T-4 |

---

### CF-2: No embargo period between IS and OOS windows

| Field | Value |
|-------|-------|
| **Description** | OOS starts the bar after IS ends (`oos_start = is_end`). With lookback=120 and horizon=60, the first 180 OOS bars have feature/target overlap with IS. This inflates OOS metrics via serial correlation leakage. |
| **File:Line** | [walk_forward.py:111-114](file:///home/kironix/workspace/both/unified-trading-platform/services/model_training/walk_forward.py#L111-L114) |
| **Acceptance Criteria** | New parameter `embargo_bars` (default = `lookback_window + max(horizons)` = 180). OOS start = IS end + embargo. `generate_folds()` enforces this gap. Config in `training.yaml`. |
| **Test** | T-1 |

---

### CF-3: Annualization factor wrong for minute data

| Field | Value |
|-------|-------|
| **Description** | Uses `√252` (daily annualization) but the platform processes minute bars. Correct factor is `√(252 × 390)` ≈ 313.5. Current code understates minute-bar Sharpe by ~20×, making the 0.5 OOS Sharpe gate trivially easy to pass. Affects both walk-forward and backtester. |
| **File:Line** | [walk_forward.py:166](file:///home/kironix/workspace/both/unified-trading-platform/services/model_training/walk_forward.py#L166), [metrics.py:94](file:///home/kironix/workspace/both/unified-trading-platform/services/backtester/metrics.py#L94) |
| **Acceptance Criteria** | New parameter `bars_per_day` (default=1 for daily, 390 for minute). `annualization_factor = √(252 × bars_per_day)`. Both `walk_forward.py` and `metrics.py` use the same logic. Config in `training.yaml`. |
| **Test** | T-3 |

---

### CF-4: Feature normalization stats not persisted for inference

| Field | Value |
|-------|-------|
| **Description** | `_normalize_features()` computes per-sample mean/std but doesn't store them. At inference time, features are normalized with different statistics than training, causing silent distribution shift. |
| **File:Line** | [dataset.py:62-71](file:///home/kironix/workspace/both/unified-trading-platform/services/model_training/dataset.py#L62-L71) |
| **Acceptance Criteria** | Per-fold normalization stats (mean, std per feature column) are saved as a JSON sidecar alongside the ONNX model. `model_inference/engine.py` loads and applies these stats. Round-trip test verifies <1e-6 error. |
| **Test** | T-2 |

---

### CF-5: CVaR calculation is fabricated

| Field | Value |
|-------|-------|
| **Description** | `_calculate_cvar()` uses `q10 × (1 + vol × 5) × 1.25` — an arbitrary formula with no statistical basis. Real CVaR-95 = expected loss in the 5% tail of historical returns. Current metric is meaningless for risk budgeting. |
| **File:Line** | [engine.py:298-309](file:///home/kironix/workspace/both/unified-trading-platform/services/risk_engine/engine.py#L298-L309) |
| **Acceptance Criteria** | Replace with actual CVaR-95: `mean(returns[returns < np.percentile(returns, 5)])` using ≥250 bars of return history. Accept `RiskInput` with a `return_history: np.ndarray` field. When history is unavailable, fall back to parametric CVaR using `σ × φ(z₀.₀₅) / 0.05`. |
| **Test** | T-5 |

---

### CF-6: Redis failure silently disables kill switch (ruin path)

| Field | Value |
|-------|-------|
| **Description** | On Redis failure, `load_state()` returns `False` and the engine starts with fresh state: `daily_pnl=0, kill_switch_active=False`. If risk engine restarts during a loss day, the 5% daily limit resets and the 10% kill switch is disabled. This is a real ruin path. |
| **File:Line** | [engine.py:393-412](file:///home/kironix/workspace/both/unified-trading-platform/services/risk_engine/engine.py#L393-L412) |
| **Acceptance Criteria** | On `load_state()` failure: (1) set `kill_switch_active=True` (fail-closed), (2) log at CRITICAL level, (3) emit `risk_state_load_failed` metric. Only resume trading after manual intervention or successful Redis reconnect and state load. |
| **Test** | T-6 |

---

### CF-7: Commit-before-flush in execution (data loss)

| Field | Value |
|-------|-------|
| **Description** | `execution/main.py` calls `producer.produce()` (async, buffered) then immediately `consumer.commit()`. If the Kafka broker is slow, the offset is committed but the execution event is never delivered. A restart will skip the message. |
| **File:Line** | [execution/main.py:588-590](file:///home/kironix/workspace/both/unified-trading-platform/services/execution/main.py#L588-L590) |
| **Acceptance Criteria** | Call `producer.flush(timeout=5.0)` before `consumer.commit()`. If flush fails (returns >0 undelivered), do NOT commit — log error, increment `kafka_producer_delivery_failures_total`, retry on next poll. |
| **Test** | T-13 |

---

### CF-8: No timeout on Alpaca API calls

| Field | Value |
|-------|-------|
| **Description** | `get_current_price()`, `submit_bracket_order()`, `submit_market_order()` have no timeout. If Alpaca hangs (common during market open), the execution agent blocks indefinitely, stalling the entire `risk.approved` consumer. |
| **File:Line** | [execution/main.py:452-545](file:///home/kironix/workspace/both/unified-trading-platform/services/execution/main.py#L452-L545) |
| **Acceptance Criteria** | All Alpaca HTTP calls have `connect_timeout=5s`, `read_timeout=10s`. On timeout, raise `AlpacaTimeoutError`, increment `alpaca_api_timeout_total` counter, reject the signal (don't retry — signal may be stale). |
| **Test** | T-14 |

---

### CF-9: Graceful shutdown timeout is never enforced

| Field | Value |
|-------|-------|
| **Description** | `GracefulShutdown.__init__` accepts `timeout_seconds=30.0` but `shutdown()` iterates handlers with no deadline. If any handler hangs (e.g., Kafka flush to unreachable broker), the process hangs forever until Docker SIGKILLs it. |
| **File:Line** | [graceful_shutdown.py:60-87](file:///home/kironix/workspace/both/unified-trading-platform/shared/core/graceful_shutdown.py#L60-L87) |
| **Acceptance Criteria** | Wrap the handler loop in a deadline: `signal.alarm(int(self.timeout))` or `asyncio.wait_for()`. If deadline expires, log remaining unexecuted handlers and `sys.exit(1)`. |
| **Test** | T-15 |

---

### CF-10: Duplicate YAML keys in app.yaml

| Field | Value |
|-------|-------|
| **Description** | `take_profit_pct` appears on lines 71 and 102. `stop_loss_pct` appears on lines 72 and 105. YAML spec: last value wins. Execution and exit services may silently use different values depending on parse order. |
| **File:Line** | [app.yaml:70-72 and 101-105](file:///home/kironix/workspace/both/unified-trading-platform/configs/app.yaml#L70-L105) |
| **Acceptance Criteria** | Remove duplicate keys. Single canonical `take_profit_pct: 0.03` and `stop_loss_pct: 0.015`. Add `yamllint` to CI. |
| **Test** | T-16 |

---

## 2. HIGH

### HI-1: UnifiedPortfolioRisk not wired into risk engine

| Field | Value |
|-------|-------|
| **Description** | `portfolio_risk.py` defines `UnifiedPortfolioRisk` (drawdown, trailing stop, sector cap) but `engine.py:evaluate()` never imports or calls it. The Lean risk models exist as dead code. Phase 3 in `06_implementation_phases.md` claims these are ✅ DONE. |
| **File:Line** | [engine.py:220-278](file:///home/kironix/workspace/both/unified-trading-platform/services/risk_engine/engine.py#L220-L278) (missing integration), [portfolio_risk.py:216-285](file:///home/kironix/workspace/both/unified-trading-platform/services/risk_engine/portfolio_risk.py#L216-L285) |
| **Acceptance Criteria** | `RiskEngine.evaluate()` instantiates or references `UnifiedPortfolioRisk`, calls `check_portfolio()` and `can_open_new()`, and includes results in `RiskOutput.checks_passed`. |
| **Test** | T-7 |

---

### HI-2: Correlation check is dead code

| Field | Value |
|-------|-------|
| **Description** | `_correlation_history` dict is initialized in `__init__` (line 164) but `_check_correlation_spike()` is never called in `evaluate()`. Inventory item A32 is documented but not functional. |
| **File:Line** | [engine.py:164](file:///home/kironix/workspace/both/unified-trading-platform/services/risk_engine/engine.py#L164) |
| **Acceptance Criteria** | Either wire `_check_correlation_spike()` into `evaluate()` as a check, or explicitly remove it and mark A32 as deferred in `01_logic_inventory.md`. |
| **Test** | T-11 |

---

### HI-3: Platt scaler silent pass-through when unfitted

| Field | Value |
|-------|-------|
| **Description** | If `_fitted=False`, `transform()` returns raw scores unchanged. The ensemble has no guard. A missing or stale calibration file silently uses uncalibrated confidence. |
| **File:Line** | [confidence_calibration.py:95-96](file:///home/kironix/workspace/both/unified-trading-platform/services/signal_engine/confidence_calibration.py#L95-L96) |
| **Acceptance Criteria** | `transform()` raises `CalibrationNotFittedError` if `_fitted=False`. Signal engine must handle this by either blocking or falling back with a logged warning and metric. |
| **Test** | T-8 |

---

### HI-4: Bear regime scaling is direction-agnostic

| Field | Value |
|-------|-------|
| **Description** | Bear regime scales ALL signals to 0.7×. But short signals in a bear market should have high confidence (1.0×). Long signals should be penalized (0.5×). Current scaling reduces short signal quality in the regime where shorts are most valuable. |
| **File:Line** | [regime.py:28-32](file:///home/kironix/workspace/both/unified-trading-platform/services/signal_engine/regime.py#L28-L32) |
| **Acceptance Criteria** | `confidence_scale()` accepts a `direction` parameter. Bear × short → 1.0×, Bear × long → 0.5×, Bull × long → 1.0×, Bull × short → 0.5×, Sideways → 0.5× for all. |
| **Test** | T-10 |

---

### HI-5: DST breaks market hours filter

| Field | Value |
|-------|-------|
| **Description** | `et_hour = (ts.hour - 5) % 24` assumes EST (UTC-5) year-round. During EDT (Mar–Nov), this is off by 1 hour. Signals are blocked or allowed at the wrong times for ~8 months per year. |
| **File:Line** | [filters.py:54-55](file:///home/kironix/workspace/both/unified-trading-platform/services/signal_engine/filters.py#L54-L55) |
| **Acceptance Criteria** | Use `zoneinfo.ZoneInfo("America/New_York")` for timezone conversion. Verify correct behavior across all four DST transition points (spring forward, fall back, day before, day after). |
| **Test** | T-7a |

---

### HI-6: PipelineStale alert fires 24/7 including weekends

| Field | Value |
|-------|-------|
| **Description** | Alert fires when `rate(events_processed_total[10m]) == 0`. No market-hours or weekday predicate. Will page at 2 AM Saturday. |
| **File:Line** | [alerts.yml:79-85](file:///home/kironix/workspace/both/unified-trading-platform/infra/prometheus/alerts.yml#L79-L85) |
| **Acceptance Criteria** | Add PromQL predicate: `and on() (hour() >= 14 and hour() < 20 and day_of_week() >= 1 and day_of_week() <= 5)` (UTC for 9:30–16:00 ET). Test with `promtool test rules`. |
| **Test** | T-17 |

---

### HI-7: DLQ `persist_to_db()` never called from `with_retry_and_dlq()`

| Field | Value |
|-------|-------|
| **Description** | `with_retry_and_dlq()` calls `dlq.send()` on final failure but never calls `dlq.persist_to_db()`. Failed messages are published to the DLQ Kafka topic but NOT persisted to the DB. If Kafka also fails, the message is silently lost. |
| **File:Line** | [dlq.py:142-157](file:///home/kironix/workspace/both/unified-trading-platform/shared/kafka/dlq.py#L142-L157) |
| **Acceptance Criteria** | `send()` calls `persist_to_db()` internally (or `with_retry_and_dlq` calls it after `send()`). Add exponential backoff between retries (currently retries immediately). |
| **Test** | T-18 |

---

### HI-8: Redis has no persistence or healthcheck

| Field | Value |
|-------|-------|
| **Description** | Redis runs with default config (RDB snapshots only, no AOF). No healthcheck in docker-compose. Risk state (kill switch, daily P&L) can be lost on Redis crash. Other services don't depend on Redis health. |
| **File:Line** | [docker-compose.yml:92-97](file:///home/kironix/workspace/both/unified-trading-platform/infra/docker-compose.yml#L92-L97) |
| **Acceptance Criteria** | Add `command: redis-server --appendonly yes`. Add healthcheck: `redis-cli ping`. Add `condition: service_healthy` to risk-engine's `depends_on`. |
| **Test** | T-19 |

---

### HI-9: Duplicate `to_dict`/`from_dict` on PositionState

| Field | Value |
|-------|-------|
| **Description** | `PositionState` in `engine.py` defines `to_dict()` twice (lines 101 and 127) and `from_dict()` twice (lines 111 and 137). Python uses the last definition. The two `from_dict` implementations have different logic — one uses constructor kwarg pattern, the other sets attributes directly. |
| **File:Line** | [engine.py:101-151](file:///home/kironix/workspace/both/unified-trading-platform/services/risk_engine/engine.py#L101-L151) |
| **Acceptance Criteria** | Remove the first pair (lines 101-125). Keep the second pair (lines 127-151) which is more robust. Add round-trip test: `assert PositionState.from_dict(state.to_dict()) == state`. |
| **Test** | T-20 |

---

## 3. MEDIUM

### MD-1: No Kafka retention limits — disk will fill

| Field | Value |
|-------|-------|
| **Description** | Kafka has no `KAFKA_LOG_RETENTION_BYTES` or `KAFKA_LOG_RETENTION_HOURS`. Topics grow unbounded. Single-node Kafka with unbounded topics will OOM or fill disk. |
| **File:Line** | [docker-compose.yml:42-51](file:///home/kironix/workspace/both/unified-trading-platform/infra/docker-compose.yml#L42-L51) |
| **Acceptance Criteria** | Add `KAFKA_LOG_RETENTION_HOURS: 168` (7 days) and `KAFKA_LOG_SEGMENT_BYTES: 1073741824` (1 GB). |
| **Test** | Manual: verify `kafka-log-dirs` reports segment cleanup after 7 days. |

---

### MD-2: No TimescaleDB retention policies

| Field | Value |
|-------|-------|
| **Description** | `market_raw_minute` grows indefinitely. At ~2.5M rows/day for 7 symbols × 390 bars, this is ~900M rows/year. No `add_retention_policy()` configured. |
| **File:Line** | `infra/db/init.sql` |
| **Acceptance Criteria** | Add `SELECT add_retention_policy('market_raw_minute', INTERVAL '30 days')` and similar for `audit_events` (90 days). |
| **Test** | Verify `timescaledb_information.jobs` shows active retention policies. |

---

### MD-3: Alpha votes not confidence-scaled

| Field | Value |
|-------|-------|
| **Description** | TFT vote is `weight × confidence`, but alpha votes are `weight × ±1`. RSI at 35 (weak) has same vote magnitude as RSI at 10 (extreme). |
| **File:Line** | [ensemble.py:117](file:///home/kironix/workspace/both/unified-trading-platform/services/signal_engine/ensemble.py#L117) |
| **Acceptance Criteria** | Alpha vote = `weight × signal_strength` where strength is derived from indicator value (e.g., RSI: `(50 - rsi) / 50` clipped to [-1, 1]). |
| **Test** | Unit test: RSI=10 produces stronger vote than RSI=35. |

---

### MD-4: Spread cost missing from backtester

| Field | Value |
|-------|-------|
| **Description** | Backtester accounts for 5 bps slippage and $1 fee but no bid-ask spread cost. For 0.05% typical spread, effective round-trip cost is underestimated by 5 bps. |
| **File:Line** | [metrics.py:61-171](file:///home/kironix/workspace/both/unified-trading-platform/services/backtester/metrics.py#L61-L171) |
| **Acceptance Criteria** | Add `spread_cost_bps` parameter (default=5). Deduct from each return: `adjusted_ret = raw_ret - spread_cost_bps / 10000`. |
| **Test** | T-9 (slippage sensitivity, extended to include spread). |

---

### MD-5: Schema Registry has no healthcheck or cache

| Field | Value |
|-------|-------|
| **Description** | Schema Registry has no healthcheck in docker-compose. No local schema cache. If SR goes down, all Avro deserialization fails → every consumer stops. |
| **File:Line** | [docker-compose.yml:64-73](file:///home/kironix/workspace/both/unified-trading-platform/infra/docker-compose.yml#L64-L73) |
| **Acceptance Criteria** | Add healthcheck (`curl http://localhost:8081/subjects`). Cache schemas after first fetch in `shared/kafka/`. Add `SchemaRegistryDown` alert. |
| **Test** | Manual: stop Schema Registry container, verify consumers continue with cached schemas for ≥5 minutes. |

---

### MD-6: ARCHITECTURE.md table count is wrong

| Field | Value |
|-------|-------|
| **Description** | §4 header says "22 tables" but enumerating the list gives 19 existing + 2 new = 21. Off by one. |
| **File:Line** | [ARCHITECTURE.md:60](file:///home/kironix/workspace/both/unified-trading-platform/ARCHITECTURE.md#L60) |
| **Acceptance Criteria** | Recount and fix header to actual count (21), or add the missing table. |
| **Test** | Grep `init.sql` for `CREATE TABLE` and count. |

---

### MD-7: "Zero Orphans" claim is false

| Field | Value |
|-------|-------|
| **Description** | ARCHITECTURE.md §5 and §10, inventory line 195, and mapping line 100 all claim "Zero Orphans." Audit found 4 true orphans (L6, L11, L13, L18), 6 inconsistent deferrals, and 5 implicit/vague mappings. |
| **File:Line** | [ARCHITECTURE.md:78](file:///home/kironix/workspace/both/unified-trading-platform/ARCHITECTURE.md#L78), [01_logic_inventory.md:195](file:///home/kironix/workspace/both/unified-trading-platform/docs/01_logic_inventory.md#L195), [02_old_to_new_mapping.md:100](file:///home/kironix/workspace/both/unified-trading-platform/docs/02_old_to_new_mapping.md#L100) |
| **Acceptance Criteria** | Add explicit mapping rows for L6, L11, L13, L18 (either "deferred" or "dropped with rationale"). Update orphan count. Add deferral flags to inventory for all items deferred in mapping. |
| **Test** | Script: cross-reference every L-# and A-# in inventory with mapping and flag unmatched. |

---

### MD-8: Missing 11 Prometheus metrics

| Field | Value |
|-------|-------|
| **Description** | SRE review identified 11 metrics not in `05_observability_and_monitoring.md` or code: `kafka_producer_delivery_failures_total`, `db_query_duration_seconds`, `db_connection_pool_available`, `alpaca_api_latency_seconds`, `alpaca_api_circuit_breaker_state`, `redis_command_duration_seconds`, `signal_age_seconds`, `model_inference_latency_seconds`, `dlq_messages_total`, `graceful_shutdown_duration_seconds`, `schema_registry_cache_hit_ratio`. |
| **File:Line** | [shared/monitoring/metrics.py](file:///home/kironix/workspace/both/unified-trading-platform/shared/monitoring/metrics.py) |
| **Acceptance Criteria** | Add definitions for at least the top 6 (producer failures, DB latency, Alpaca latency, Redis latency, signal age, DLQ count). Instrument in respective services. |
| **Test** | Verify metrics emit non-zero values via `curl localhost:PORT/metrics`. |

---

## 4. NICE-TO-HAVE

### NH-1: Walk-forward on the ensemble itself

| Field | Value |
|-------|-------|
| **Description** | Only TFT is walk-forward validated. Ensemble weights, direction threshold (0.15), confidence formula are never tested OOS. |
| **File:Line** | [ensemble.py:20-23](file:///home/kironix/workspace/both/unified-trading-platform/services/signal_engine/ensemble.py#L20-L23) |
| **Acceptance Criteria** | Add outer walk-forward loop testing full pipeline (TFT + alphas + ensemble → PnL) on each fold. |
| **Test** | T-12 |

---

### NH-2: Feature importance monitoring

| Field | Value |
|-------|-------|
| **Description** | TFT has attention weights but they are not logged. If one feature dominates, it may indicate overfitting. |
| **File:Line** | `services/model_inference/engine.py` |
| **Acceptance Criteria** | Log variable selection weights per inference batch. Alert if any feature >50% average importance. |
| **Test** | Log inspection in Grafana Model Monitor dashboard. |

---

### NH-3: Exactly-once Kafka processing

| Field | Value |
|-------|-------|
| **Description** | Listed as "future work" in `06_implementation_phases.md` (line 103). Currently at-least-once with idempotency key dedup. |
| **File:Line** | [06_implementation_phases.md:103](file:///home/kironix/workspace/both/unified-trading-platform/docs/06_implementation_phases.md#L103) |
| **Acceptance Criteria** | Use Kafka transactional producer: `begin_transaction()` → produce → `send_offsets_to_transaction()` → `commit_transaction()`. |
| **Test** | Kill and restart consumer mid-batch; verify no duplicates in output topic. |

---

### NH-4: Add stop_grace_period to docker-compose services

| Field | Value |
|-------|-------|
| **Description** | Docker default `stop_grace_period` is 10s. `GracefulShutdown` uses 30s timeout (once CF-9 is fixed). Services may be SIGKILLed before completing shutdown. |
| **File:Line** | [docker-compose.yml](file:///home/kironix/workspace/both/unified-trading-platform/infra/docker-compose.yml) (all service blocks) |
| **Acceptance Criteria** | Add `stop_grace_period: 45s` to all service definitions in `x-agent-base`. |
| **Test** | `docker compose stop` completes within 45s with clean shutdown logs from all services. |

---

## 5. Test Matrix

| Test ID | Description | Type | Validates |
|---------|-----------|------|-----------|
| **T-1** | Generate folds with embargo=180. Verify no OOS sample's feature window overlaps IS. | Unit | CF-2 |
| **T-2** | Save norm stats from training; load at inference. Verify match <1e-6. | Unit | CF-4 |
| **T-3** | Compute Sharpe on synthetic returns at minute and daily. Verify annualized values agree. | Unit | CF-3 |
| **T-4** | Verify `select_best()` returns latest fold, not max-Sharpe fold. | Unit | CF-1 |
| **T-5** | Compute CVaR on Normal(0, 0.01). Verify against analytical CVaR-95. Tolerance ±5%. | Unit | CF-5 |
| **T-6** | Start risk engine, process trades, kill Redis, restart risk engine. Verify kill switch = True. | Integration | CF-6 |
| **T-7** | Submit signal through risk engine. Verify `UnifiedPortfolioRisk` checks appear in output. | Integration | HI-1 |
| **T-7a** | Run signal filter with timestamps across EDT→EST transition. Verify correct behavior. | Unit | HI-5 |
| **T-8** | Create ensemble, don't fit Platt scaler, call transform. Verify error raised. | Unit | HI-3 |
| **T-9** | Run backtest with slippage = {0, 5, 10, 20, 30} bps. Verify Sharpe degrades monotonically. | Integration | MD-4 |
| **T-10** | In bear regime: short signal → 1.0× scale, long signal → 0.5× scale. | Unit | HI-4 |
| **T-11** | Submit 5 correlated signals (r > 0.8). Verify some rejected. | Integration | HI-2 |
| **T-12** | Ensemble walk-forward: IS→OOS PnL positive after costs. | Integration | NH-1 |
| **T-13** | Mock Kafka producer to return deliver failure. Verify consumer does NOT commit offset. | Unit | CF-7 |
| **T-14** | Mock Alpaca to hang for 15s. Verify execution agent returns within 10s with timeout error. | Unit | CF-8 |
| **T-15** | Register a handler that sleeps 60s. Verify `shutdown()` exits within 35s (30s timeout + margin). | Unit | CF-9 |
| **T-16** | `yamllint configs/*.yaml` returns 0 errors. | Lint | CF-10 |
| **T-17** | `promtool test rules` with PipelineStale at 3 AM Saturday → no alert. | Unit | HI-6 |
| **T-18** | Fail msg processing 4× (max_retries=3). Verify DLQ message in both Kafka topic AND DB. | Integration | HI-7 |
| **T-19** | `docker compose up redis`. Kill Redis process. Verify auto-restart and healthcheck recovery. | Integration | HI-8 |
| **T-20** | `PositionState` round-trip: `from_dict(state.to_dict()) == state` for all field types. | Unit | HI-9 |

---

> **Total: 10 CRITICAL, 9 HIGH, 8 MEDIUM, 4 NICE-TO-HAVE = 31 items**
> **Total tests: 22** (14 unit, 7 integration, 1 lint)
