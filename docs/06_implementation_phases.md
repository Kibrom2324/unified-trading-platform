# 06 — Implementation Phases

---

## Phase 1 — Unified Skeleton (End-to-End Running) ✅ DONE

**Goal**: Single monorepo with all AWET agents migrated + LeanAlpha + SignalEngine wired.

| Item | Status |
|------|--------|
| Create monorepo directory structure | ✅ |
| Migrate AWET shared library (kafka, schemas, models, audit, core) | ✅ |
| Migrate all 11 AWET agents → services/ | ✅ |
| Port Lean RSI/EMA/MACD alphas to Python → lean_alpha/ | ✅ |
| Build SignalEngine ensemble (weighted voting) → signal_engine/ | ✅ |
| Add `signals` + `walk_forward_results` DB tables | ✅ |
| Create unified docker-compose.yml (13 services + infra) | ✅ |
| Create Dockerfiles for new services | ✅ |
| Create Prometheus alerts.yml (11 rules) | ✅ |
| Create configs/training.yaml + configs/lean.yaml | ✅ |
| Write README.md | ✅ |
| Write unit tests for lean_alpha + signal_engine (17 tests) | ✅ |

---

## Phase 2 — Walk-Forward TFT Training ✅ DONE

**Goal**: Replace naive train/test split with quant-grade walk-forward CV.

| Item | Status |
|------|--------|
| `walk_forward.py` — fold generation (rolling windows) | ✅ |
| OOS metrics: Sharpe, Sortino, Calmar, max DD, win rate | ✅ |
| Model selection gating (median OOS Sharpe ≥ 0.5, win rate ≥ 45%) | ✅ |
| MLflow integration (nested runs per fold) | ✅ |
| DB persistence to `walk_forward_results` table | ✅ |
| Unit tests (fold generation, metric calculation, accept/reject) | ✅ |

**Config**: `configs/training.yaml`
```yaml
walk_forward:
  train_months: 12
  oos_months: 1
  min_folds: 6
  min_oos_sharpe: 0.5
  min_win_rate: 0.45
```

---

## Phase 3 — Signal Quality + Risk Models ✅ DONE

**Goal**: Add regime awareness, signal filters, calibration, Lean risk models.

| Item | Status |
|------|--------|
| Regime classifier (SMA 20/50 → Bull/Bear/Sideways) | ✅ |
| Regime confidence scaling (1.0x / 0.7x / 0.5x) | ✅ |
| Platt scaling confidence calibration | ✅ |
| Market hours filter (9:30-15:45 ET) | ✅ |
| Liquidity filter (10d avg vol ≥ 500k) | ✅ |
| Spread filter (bid-ask < 0.1%) | ✅ |
| Earnings blackout filter (±2 days) | ✅ |
| MaximumDrawdownPortfolio (Lean → Python) | ✅ |
| TrailingStopRisk (Lean → Python) | ✅ |
| MaximumSectorExposure (Lean → Python) | ✅ |
| UnifiedPortfolioRisk aggregator | ✅ |
| Unit tests (regime, filters, calibration, portfolio risk) | ✅ |

---

## Phase 4 — Monitoring + Metrics + Dashboards ✅ DONE

**Goal**: Comprehensive quant reporting and production monitoring.

| Item | Status |
|------|--------|
| `metrics.py` — Sharpe, Sortino, Calmar, Alpha, Beta, IR, profit factor | ✅ |
| Max drawdown depth + duration calculation | ✅ |
| Benchmark-relative metrics (Alpha, Beta) | ✅ |
| 11 Prometheus alert rules (alerts.yml) | ✅ |
| Unit tests for all metrics | ✅ |

**Remaining** (future work):
- [ ] Build 5 Grafana dashboard JSON files
- [ ] Telegram alertmanager route
- [ ] Daily HTML equity report

---

## Phase 5 — Hardening ✅ DONE

**Goal**: Production resilience — no message loss, clean restarts.

| Item | Status |
|------|--------|
| Dead Letter Queue (DLQ) — dlq.failed Kafka topic + DB persistence | ✅ |
| Retry wrapper with configurable max attempts | ✅ |
| Graceful shutdown (SIGTERM/SIGINT → flush in reverse order) | ✅ |
| Unit tests (retry, DLQ, shutdown handler ordering) | ✅ |

**Remaining** (future work):
- [ ] Exactly-once Kafka processing (transactional producers)
- [ ] Kubernetes Helm charts
- [ ] VWAP execution model (from Lean)
- [ ] Historical Returns alpha (from Lean)
- [ ] Pairs Trading alpha (from Lean)
- [ ] Mean-Variance Optimization portfolio construction (from Lean)
- [ ] Black-Litterman portfolio construction (from Lean)

---

## Dependencies Between Phases

```
Phase 1 (skeleton) ─→ Phase 2 (walk-forward)
                   ─→ Phase 3 (signal quality + risk)
                   ─→ Phase 4 (observability)
                   ─→ Phase 5 (hardening)
```
Phases 2-5 are independent of each other; only Phase 1 is a prerequisite.
