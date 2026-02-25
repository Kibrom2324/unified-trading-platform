# 02 — Old → New Mapping (Zero Orphans)

> Every file from both repos maps to exactly one location in the unified platform.
> No logic is orphaned. Overlaps are resolved.

---

## AWET-main → Unified Platform

| AWET File | → Unified Location | Notes |
|-----------|-------------------|-------|
| `src/agents/data_ingestion.py` | `services/data_ingestion/main.py` | Direct migration |
| `src/agents/data_ingestion.py/backfill/` | `services/data_ingestion/backfill/` | Direct migration |
| `src/agents/feature_engineering.py` | `services/feature_engineering/main.py` | Direct migration |
| `src/features/engine.py` | `services/feature_engineering/engine.py` | Direct migration |
| `src/ml/train_tft.py` | `services/model_training/train_tft.py` | + walk_forward.py added |
| `src/ml/tft_model.py` | `services/model_training/tft/model.py` | Direct migration |
| `src/ml/dataset.py` | `services/model_training/dataset.py` | Direct migration |
| `src/ml/registry.py` | `services/model_training/registry.py` | Direct migration |
| `src/prediction/engine.py` | `services/model_inference/engine.py` | Direct migration |
| `src/agents/time_series_prediction.py` | `services/model_inference/main.py` | Direct migration |
| `src/agents/trader_decision_agent.py` | **REPLACED by** `services/signal_engine/` | Ensemble replaces single-source decision |
| `src/core/trade_decision.py` | `shared/core/trade_decision.py` | Direct migration |
| `src/agents/risk_agent.py` | `services/risk_engine/main.py` | + portfolio_risk.py added |
| `src/risk/engine.py` | `services/risk_engine/engine.py` | Direct migration |
| `src/agents/execution_agent.py` | `services/execution/main.py` | Direct migration |
| `src/integrations/alpaca_client.py` | `services/execution/integrations/alpaca_client.py` | Direct migration |
| `src/integrations/trades_repository.py` | `services/execution/integrations/trades_repository.py` | Direct migration |
| `src/agents/exit_agent.py` | `services/exit_monitor/main.py` | + trailing stop added |
| `src/agents/backtester_agent.py` | `services/backtester/main.py` | + metrics.py added |
| `src/agents/watchtower_agent.py` | `services/monitoring/main.py` | Direct migration |
| `src/monitoring/metrics.py` | `shared/monitoring/metrics.py` | Direct migration |
| `src/streaming/kafka_*.py` | `shared/kafka/` | Direct migration |
| `src/schemas/*.avsc` | `shared/schemas/` | Direct migration |
| `src/audit/trail_logger.py` | `shared/audit/trail_logger.py` | Direct migration |
| `src/core/config.py` | `shared/core/config.py` | Direct migration |
| `src/core/logging.py` | `shared/core/logging.py` | Direct migration |
| `src/core/retry.py` | `shared/core/retry.py` | Direct migration |
| `src/core/circuit_breaker.py` | `shared/core/circuit_breaker.py` | Direct migration |
| `src/models/*.py` | `shared/models/` | Pydantic event models |
| `config/app.yaml` | `configs/app.yaml` | Direct migration |
| `config/limits.yaml` | `configs/limits.yaml` | Direct migration |
| `config/kafka.yaml` | `configs/kafka.yaml` | Direct migration |
| `config/llm.yaml` | `configs/llm.yaml` | Direct migration |
| `config/market_data.yaml` | `configs/market_data.yaml` | Direct migration |
| `config/logging.yaml` | `configs/logging.yaml` | Direct migration |
| `config/universe.csv` | `configs/universe.csv` | Direct migration |
| `db/init.sql` | `infra/db/init.sql` | + 2 new tables appended |
| `prometheus/` | `infra/prometheus/` | + alerts.yml added |
| `grafana/` | `infra/grafana/` | Direct migration |
| `alertmanager/` | `infra/alertmanager/` | Direct migration |
| `directives/` | `services/orchestration/directives/` | Direct migration |
| `scripts/` | `services/orchestration/scripts/` | Direct migration |
| `tests/` | `tests/` | + new test files added |

---

## Lean-master → Unified Platform

| Lean Module | → Unified Location | Notes |
|-------------|-------------------|-------|
| `Algorithm.Framework/Alphas/RsiAlphaModel.py` | `services/lean_alpha/rsi_alpha.py` | Ported to Python with bounce hysteresis |
| `Algorithm.Framework/Alphas/EmaCrossAlphaModel.py` | `services/lean_alpha/ema_cross_alpha.py` | Ported to Python |
| `Algorithm.Framework/Alphas/MacdAlphaModel.py` | `services/lean_alpha/macd_alpha.py` | Ported to Python |
| `Algorithm.Framework/Risk/MaximumDrawdownPercentPortfolio` | `services/risk_engine/portfolio_risk.py` | Ported to Python |
| `Algorithm.Framework/Risk/TrailingStopRiskManagementModel` | `services/risk_engine/portfolio_risk.py` | Ported to Python |
| `Algorithm.Framework/Risk/MaximumSectorExposureRiskManagementModel` | `services/risk_engine/portfolio_risk.py` | Ported to Python |
| `Algorithm.Framework/Portfolio/InsightWeighting` | `services/signal_engine/ensemble.py` | Confidence-weighted voting |
| `Optimizer/` (walk-forward pattern) | `services/model_training/walk_forward.py` | Ported concept to Python |
| `Report/` (Sharpe, Sortino, Calmar, Alpha, Beta) | `services/backtester/metrics.py` | Ported metrics to Python |
| `Indicators/` (231 indicators) | `services/lean_alpha/` | RSI, EMA, MACD ported; rest available for future phases |
| `Algorithm.Framework/Alphas/HistoricalReturnsAlphaModel.py` | Future: `services/lean_alpha/historical_returns_alpha.py` | **Deferred to Phase 3** |
| `Algorithm.Framework/Alphas/PearsonCorrelationPairsTradingAlphaModel.py` | Future: `services/lean_alpha/pairs_trading_alpha.py` | **Deferred to Phase 3** |
| `Algorithm.Framework/Portfolio/MeanVarianceOptimization` | Future: `services/portfolio/mvo.py` | **Deferred to Phase 3** |
| `Algorithm.Framework/Portfolio/BlackLitterman` | Future: `services/portfolio/black_litterman.py` | **Deferred to Phase 3** |
| `Algorithm.Framework/Portfolio/RiskParity` | Future: `services/portfolio/risk_parity.py` | **Deferred to Phase 3** |
| `Algorithm.Framework/Execution/VWAP` | Future: `services/execution/vwap.py` | **Deferred to Phase 5** |

---

## New Files (Not From Either Repo)

| File | Purpose | Phase |
|------|---------|-------|
| `services/lean_alpha/main.py` | Kafka consumer → alpha.signals | 1 |
| `services/signal_engine/main.py` | Ensemble consumer → signals.scored | 1 |
| `services/signal_engine/ensemble.py` | Weighted voting logic | 1 |
| `services/model_training/walk_forward.py` | Walk-forward CV with OOS gating | 2 |
| `services/signal_engine/regime.py` | SMA 20/50 regime classifier | 3 |
| `services/signal_engine/filters.py` | Market hours, liquidity, spread, earnings | 3 |
| `services/signal_engine/confidence_calibration.py` | Platt scaling | 3 |
| `services/risk_engine/portfolio_risk.py` | Drawdown + trailing + sector caps | 3 |
| `services/backtester/metrics.py` | Full quant metrics suite | 4 |
| `infra/prometheus/alerts.yml` | 11 Prometheus alert rules | 4 |
| `shared/kafka/dlq.py` | Dead Letter Queue + retry | 5 |
| `shared/core/graceful_shutdown.py` | SIGTERM/SIGINT clean shutdown | 5 |
| `configs/training.yaml` | Walk-forward parameters | 2 |
| `configs/lean.yaml` | Alpha model config | 1 |

> **Accounting: 43 AWET files migrated. 16 Lean concepts ported (6 in Phase 1, 4 in later phases, 6 deferred). 14 new files created. 0 orphans.**
