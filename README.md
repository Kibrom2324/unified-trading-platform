# Unified Trading Platform

An autonomous algorithmic trading platform integrating **AWET** (streaming ML pipeline) with **QuantConnect Lean** (backtesting & alpha engine) into a single unified system.

## Architecture

```
Data Ingestion → Feature Engineering → TFT Inference → ┐
                                                         ├→ Signal Engine → Risk Engine → Execution
Lean Alpha (RSI/EMA/MACD) ──────────────────────────────┘
```

**New in this unified platform vs AWET-main:**
- `lean-alpha` service: RSI-14, EMA-cross (12/26), MACD (12/26/9) signals ported from [Lean Algorithm.Framework](https://github.com/QuantConnect/Lean)
- `signal-engine` service: Weighted ensemble (TFT=0.5, RSI=0.2, EMA=0.2, MACD=0.1) with confidence voting
- Walk-forward model validation (12 folds, OOS Sharpe gate ≥ 0.5)
- 2 new DB tables: `signals`, `walk_forward_results`
- 11 Prometheus alert rules (kill switch, consumer lag, model drift, etc.)
- MLflow tracking (mandatory, not optional)

Full architecture: see [`docs/architecture.md`](docs/architecture.md)

---

## Quick Start

### Prerequisites
- Docker + Docker Compose v2
- Python 3.11+ (for local dev only)
- API keys: Polygon.io, Alpaca (paper)

### 1. Configure secrets
```bash
cp .env.example .env
# Edit .env — add POLYGON_API_KEY, ALPACA_API_KEY, ALPACA_SECRET
```

### 2. Start infrastructure + all agents
```bash
cd infra
docker compose up -d
```

### 3. Verify pipeline health
```bash
docker compose ps                    # All services running
curl http://localhost:8001/health    # DataIngestion
curl http://localhost:8015/health    # SignalEngine (NEW)
curl http://localhost:8014/health    # LeanAlpha (NEW)
open http://localhost:3000           # Grafana dashboard
```

### 4. Backfill historical data
```bash
docker compose run --rm data-ingestion python backfill_polygon.py \
  --data-dir /home/kironix/train --resume
```

### 5. Train TFT model
```bash
docker compose run --profile training model-training python train.py \
  --walk-forward --min-folds 6
```

### 6. Promote model + enable paper trading
```bash
docker compose run --rm model-training python -m src.ml.registry promote <model_id>
make approve        # Enable execution approval gate
make demo           # Verify end-to-end signal flow
```

---

## Service Ports

| Service | Port | Description |
|---|---|---|
| data-ingestion | 8001 | Market data ingest (Polygon/Alpaca/yfinance) |
| feature-engineering | 8002 | Feature compute (returns, vol, RSI, MACD, BB) |
| model-inference | 8003 | ONNX TFT inference + hot-reload |
| risk-engine | 8004 | CVaR, position limits, kill switch |
| execution | 8005 | Alpaca paper orders |
| monitoring | 8006 | Kafka consumer lag |
| backtester | 8012 | REST backtest API |
| **lean-alpha** | **8014** | **Lean RSI/EMA/MACD alpha signals (NEW)** |
| **signal-engine** | **8015** | **Ensemble signal output (NEW)** |
| Prometheus | 9090 | Metrics |
| Grafana | 3000 | Dashboards |
| MLflow | 5000 | Experiment tracking |
| TimescaleDB | 5433 | Time-series database |

---

## Kafka Topics

| Topic | Producer | Consumer |
|---|---|---|
| `market.raw` | data-ingestion | feature-engineering, **lean-alpha** |
| `market.engineered` | feature-engineering | model-inference |
| `predictions.tft` | model-inference | **signal-engine** |
| `alpha.signals` | **lean-alpha** | **signal-engine** |
| `signals.scored` | **signal-engine** | risk-engine |
| `risk.approved` | risk-engine | execution |
| `risk.rejected` | risk-engine | monitoring |
| `execution.completed` | execution | monitoring |
| `execution.blocked` | execution | monitoring |

---

## Safety Gates

Three layers prevent accidental real-money trading:
1. `paper_trade=True` — hardcoded in execution service
2. `execution_dry_run: false` — set in `configs/app.yaml`
3. `.tmp/APPROVE_EXECUTION` — file must exist (`make approve` / `make revoke`)

```bash
make approve   # Enable paper trading
make revoke    # Immediately halt all trading
```

---

## Signal Quality

The platform enforces quant-grade signal validation:
- **Walk-forward CV**: 12 folds, 12-month train / 1-month OOS
- **Promotion gate**: OOS Sharpe ≥ 0.5, win rate ≥ 45%
- **Ensemble**: TFT(0.5) + RSI(0.2) + EMA-cross(0.2) + MACD(0.1)
- **Regime filter**: Bull/Bear/Sideways confidence scaling (Phase 3)
- **Model hot-reload**: Green model promoted without restart

---

## Directory Structure

```
unified-trading-platform/
├── services/           # All microservices (FastAPI + Kafka consumers)
│   ├── data_ingestion/ # Polygon/Alpaca/yfinance ingest
│   ├── feature_engineering/  # TFT features
│   ├── model_training/ # TFT train + walk-forward
│   ├── model_inference/# ONNX prediction
│   ├── lean_alpha/     # NEW: RSI/EMA/MACD from Lean
│   ├── signal_engine/  # NEW: TFT + Lean ensemble
│   ├── risk_engine/    # CVaR, kill switch, limits
│   ├── execution/      # Alpaca paper orders
│   ├── exit_monitor/   # TP/SL/trailing stop exits
│   ├── backtester/     # REST backtest API
│   ├── monitoring/     # Consumer lag + health
│   └── orchestration/  # SuperAGI tools + cron scripts
├── shared/             # Shared Python library (kafka, models, audit, core)
├── configs/            # YAML configuration files
├── infra/              # Docker Compose, DB schema, Prometheus, Grafana
├── lean/               # Lean engine submodule + custom algorithms
├── research/           # Jupyter notebooks
├── docs/               # Architecture docs + runbooks
└── tests/              # Unit, integration, regression tests
```

---

## Implementation Phases

| Phase | Goal | Status |
|---|---|---|
| **1** | Unified skeleton, LeanAlpha, SignalEngine wired | ✅ **DONE** |
| **2** | Walk-forward TFT training, MLflow mandatory | 🔜 Next |
| **3** | Regime filter, ensemble improvement, Lean Risk models | 🔜 Planned |
| **4** | Full Grafana dashboards, Telegram alerts | 🔜 Planned |
| **5** | Hardening: DLQ, exactly-once, k8s manifests | 🔜 Planned |
