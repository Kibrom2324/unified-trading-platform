# 03 — Unified Target Architecture

---

## High-Level Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                     │
│  DataIngestionService ──→ Kafka[market.raw] ──→ TimescaleDB        │
│  (Polygon, Alpaca, yfinance, Reddit)                               │
└────────────┬────────────────────────────┬──────────────────────────┘
             ↓                            ↓
┌────────────────────────┐  ┌──────────────────────────────┐
│  FEATURE LAYER         │  │  LEAN ALPHA LAYER             │
│  FeatureEngineering    │  │  LeanAlphaService             │
│  → 15+ features        │  │  → RSI-14, EMA(12/26), MACD  │
│  → Kafka[engineered]   │  │  → Kafka[alpha.signals]       │
└────────┬───────────────┘  └──────────────┬───────────────┘
         ↓                                 ↓
┌────────────────────────┐                 │
│  ML INFERENCE LAYER    │                 │
│  ModelInferenceService │                 │
│  → ONNX TFT (3 qntl)  │                 │
│  → Kafka[predictions]  │                 │
└────────┬───────────────┘                 │
         ↓                                 ↓
┌──────────────────────────────────────────────────────────┐
│  SIGNAL LAYER                                            │
│  SignalEngineService                                     │
│  → Ensemble: TFT(0.5) + RSI(0.2) + EMA(0.2) + MACD(0.1)│
│  → Regime filter (Bull/Bear/Sideways)                    │
│  → Platt confidence calibration                          │
│  → Market hours / liquidity / spread filters             │
│  → Kafka[signals.scored]                                 │
└────────────┬─────────────────────────────────────────────┘
             ↓
┌──────────────────────────────────────────────────────────┐
│  RISK LAYER                                              │
│  RiskEngineService                                       │
│  → AWET: CVaR-95, position limits, daily loss, kill sw.  │
│  → Lean: portfolio drawdown, trailing stop, sector cap   │
│  → Kafka[risk.approved] or Kafka[risk.rejected]          │
└────────────┬────────────────────┬────────────────────────┘
             ↓                    ↓
┌────────────────────────┐  ┌────────────────────────────┐
│  EXECUTION LAYER       │  │  MONITORING LAYER          │
│  ExecutionService      │  │  MonitoringService         │
│  → Alpaca paper API    │  │  → Consumer lag            │
│  → 3 safety gates      │  │  → Prometheus metrics      │
│  → Bracket TP/SL       │  │  → Grafana dashboards      │
│  → Kafka[exec.done]    │  │  → 11 alert rules          │
└────────┬───────────────┘  └────────────────────────────┘
         ↓
┌────────────────────────┐
│  EXIT LAYER            │
│  ExitMonitorService    │
│  → TP / SL / trailing  │
│  → Max hold time       │
└────────────────────────┘
```

## Signal Contract

All services communicate through a **standardized signal schema**:

```python
@dataclass
class ScoredSignal:
    ticker: str
    ts: datetime
    # TFT component
    tft_direction: str        # "long" | "short" | "neutral"
    tft_confidence: float     # 0.0 – 1.0
    tft_q50: float            # Median predicted return
    # Ensemble output
    ensemble_direction: str   # "long" | "short" | "neutral"
    ensemble_confidence: float
    signal_score: float       # -1.0 to +1.0 (normalized)
    # Metadata
    active_alphas: dict       # {"rsi_14": "UP", "ema_cross": "DOWN", ...}
    regime: str               # "bull" | "bear" | "sideways"
    correlation_id: str       # End-to-end traceability
```

## Market Data Contract

```python
@dataclass
class MarketRawEvent:
    event_id: str
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str      # "polygon" | "alpaca" | "yfinance" | "stub"
    bar_type: str     # "minute" | "daily"
```

## Key Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **Kafka as event bus** | AWET already uses Kafka; Lean signals plug in as another producer |
| **Python-only services** | Lean C# ported to Python to avoid polyglot complexity |
| **Ensemble voting** | Reduces single-model risk; weights configurable at runtime |
| **Walk-forward mandatory** | Prevents overfitting; OOS Sharpe gate blocks promotion |
| **Regime-aware sizing** | Bull=1.0x, Bear=0.7x, Sideways=0.5x confidence scaling |
| **Triple safety gate** | Paper-only by default; 3 independent gates must all pass |
| **DLQ for all consumers** | No message silently lost; all failures auditable |
| **MLflow mandatory** | Every training run tracked; reproducibility guaranteed |

## Configuration Hierarchy

```
Vault/Secrets (highest priority)
  ↓
Environment Variables (.env)
  ↓
YAML Config Files (configs/*.yaml)
  ↓
Code Defaults (lowest priority)
```

## Deployment

```
Phase 1-3:  Docker Compose (single node)
Phase 4-5:  Kubernetes (Helm charts)
```
