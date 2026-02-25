# 00 — Repository Deep-Read Summaries

---

## Repo A: AWET-main (Python Streaming Trading Platform)

### How It Runs
- **Runtime**: Streaming + event-driven via Kafka (KRaft mode, no Zookeeper)
- **Entrypoints**: Each agent is a standalone FastAPI app (`src/agents/*.py`)
- **Scheduling**: Cron via SuperAGI — Night Trainer (02:00), Morning Deployer (08:30), Trade Watchdog (*/15 min)
- **Deployment**: Docker Compose (20+ containers)

### Data Sources & Storage
- **Ingest**: Polygon.io (minute/daily bars), Alpaca (real-time), yfinance (fallback), Reddit (sentiment)
- **Storage**: TimescaleDB (20 hypertables), Redis (risk state persistence)
- **Streaming**: Kafka with Avro schemas + Schema Registry

### Trading Logic

| Logic | File | Details |
|-------|------|---------|
| Feature Engineering | `src/agents/feature_engineering.py`, `src/features/engine.py` | Returns (1/5/15), Volatility (5/15), SMA (5/20), EMA (5/20), RSI-14, Volume Z-score, calendar features, Reddit sentiment |
| TFT Model | `src/ml/train_tft.py`, `src/ml/tft_model.py` | Temporal Fusion Transformer, quantile regression (q10/q50/q90), 3 horizons (30/45/60 bars), 120-bar lookback, PyTorch → ONNX export |
| Signal Generation | `src/agents/trader_decision_agent.py`, `src/core/trade_decision.py` | TFT prediction direction + confidence ≥ 0.65 + quantile return ≥ 0.5% |
| Risk Controls | `src/risk/engine.py` | Position limits (2% portfolio), daily loss (5%), kill switch (10%), CVaR-95, volatility filter, correlation spike detection, Redis state |
| Execution | `src/agents/execution_agent.py` | Alpaca paper API, bracket orders (TP 3%, SL 1.5%), 3 safety gates (paper_trade hardcoded, dry_run config, approval file), throttles (5/min, 3/symbol/day), exposure caps ($50k total, $10k/symbol) |
| Exit Logic | `src/agents/exit_agent.py` | TP/SL/max_hold (60 min) exit triggers, polls every 60s |
| Backtesting | `src/agents/backtester_agent.py` | Batch backtest via REST API, slippage (5bps), fees ($1/trade), stop-loss (2%), take-profit (5%), cooldown (3 bars) |

### ML Pipeline
- **Training**: `src/ml/train_tft.py` — builds dataset from TimescaleDB, trains TFT, exports ONNX
- **Inference**: `src/agents/time_series_prediction.py` + `src/prediction/engine.py` — loads ONNX, hot-reloads green model
- **Registry**: `src/ml/registry.py` — candidate → green → deprecated lifecycle, auto-promote by val_loss

### Observability
- Prometheus metrics exported by each agent (`src/monitoring/metrics.py`)
- Grafana dashboard (`grafana/provisioning/dashboards/awet-pipeline.json`)
- Structured logging via structlog with `correlation_id` propagation
- Audit trail in `audit_events` table (every pipeline event persisted)

### Config & Deployment
- **Config files**: `config/app.yaml` (execution gates, throttles, symbols), `config/limits.yaml` (risk limits with per-symbol overrides), `config/kafka.yaml`, `config/market_data.yaml`, `config/llm.yaml`
- **Environment overrides**: `RISK_MAX_POSITION_PCT`, `RISK_MAX_DAILY_LOSS_PCT`, `ALPACA_API_KEY`, etc.
- **Docker Compose**: 20+ containers (Kafka, TimescaleDB, Redis, Schema Registry, Prometheus, Grafana, SuperAGI, all agents)

### Strengths
- End-to-end streaming pipeline with full idempotency
- Robust multi-layer risk controls with Redis state persistence
- TFT/ONNX ML pipeline with model registry and hot-reload
- Triple safety gate execution (hardcoded + config + file)
- Complete GAP audit (16 gaps identified and fixed)
- SuperAGI LLM orchestration integration

### Weaknesses
- Basic backtester (no walk-forward, no out-of-sample evaluation)
- Hardcoded volatility threshold in risk engine
- Naive ML training split (no temporal awareness)
- Static symbol universe (7 symbols)
- No Sortino, Calmar, Alpha, Beta metrics
- No regime detection or market hours filtering
- MLflow integration incomplete

---

## Repo B: Lean-master (QuantConnect Lean Engine)

### How It Runs
- **Runtime**: Event-driven algorithmic trading engine
- **Language**: C# (.NET 9) primary, Python API available
- **Entrypoints**: `lean backtest` CLI, `QuantConnect.Lean.Launcher.dll`
- **Deployment**: Docker containers, Lean CLI

### Data Sources & Storage
- **Data**: Local flat files, QuantConnect cloud, custom data providers
- **Storage**: In-memory during backtest; results exported to JSON/HTML

### Trading Logic (Algorithm Framework)

| Logic | File/Directory | Details |
|-------|---------------|---------|
| Alpha Models | `Algorithm.Framework/Alphas/` | EMA Cross, RSI (30/70 with hysteresis), MACD (histogram direction), Historical Returns, Pairs Trading (Pearson correlation) |
| Portfolio Construction | `Algorithm.Framework/Portfolio/` | Equal Weight, Confidence-Weighted, Mean-Variance Optimization, Black-Litterman, Risk Parity, Sector Weighting |
| Risk Management | `Algorithm.Framework/Risk/` | Max Drawdown per Security, Max Drawdown per Portfolio, Trailing Stop, Max Sector Exposure |
| Execution | `Algorithm.Framework/Execution/` | Immediate execution, VWAP, standard deviation execution thresholds |
| Universe Selection | `Algorithm.Framework/Selection/` | Coarse/fine fundamental universe, manual, options, futures |

### Indicators Library
- **231 indicators** in `Indicators/` directory
- RSI, MACD, Bollinger Bands, Stochastic, ATR, ADX, Ichimoku, ARIMA, Keltner, Donchian, Williams %R, CCI, OBV, etc.
- Greeks calculations for options

### Backtesting & Evaluation
- **Engine**: Full event-driven simulation (`Engine/`)
- **Optimizer**: Walk-forward parameter optimization (`Optimizer/`)
- **Report**: Comprehensive HTML/JSON — Sharpe, Sortino, Calmar, Alpha, Beta, Information Ratio, Treynor, max drawdown, benchmark comparison

### Observability
- Logging framework in `Logging/`
- Results exported to JSON (`Report/`)
- No Prometheus/Grafana (desktop/cloud oriented)

### Strengths
- Production-grade backtesting engine
- 231 indicators and extensive framework
- Walk-forward optimizer
- Multi-brokerage support (Alpaca, IB, Binance, OANDA, Bybit, etc.)
- Comprehensive quant reporting (Sharpe, Sortino, Calmar, Alpha, Beta)
- Modular Alpha Framework architecture

### Weaknesses
- No native ML model training pipeline
- No streaming feature store
- C# primary (friction with Python ML ecosystem)
- No Kafka/event bus
- No persistent audit trail
- In-memory state (no restart recovery)
- No TFT/deep learning pipeline
- No Redis for risk state
