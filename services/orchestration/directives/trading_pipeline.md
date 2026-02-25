# Trading Pipeline Directive

## Goal
Run the end-to-end paper-trading pipeline from ingestion to execution with full audit trail.

## Success Criteria
- Events flow through: market.raw → market.engineered → predictions.tft → trade.decisions → risk.approved|risk.rejected → execution.completed
- Every event includes contract fields and valid Avro schema
- Audit table contains at least one row per stage
- Idempotency enforced by idempotency_key

## Inputs
- Symbols: configured in config/universe.csv
- Time range: configurable via --start/--end
- Risk limits: config/limits.yaml
- Kafka topics: src/streaming/topics.py

## Quick Start Commands

### 1. Start Infrastructure
```bash
docker compose up -d
python scripts/verify_db.py  # Verify all tables exist
```

### 2. Run Full E2E Test (Single Command)
```bash
python scripts/e2e_test.py
# OR
pytest -m e2e -v
```

### 3. Run Full Pipeline (Single Command)
```bash
python scripts/run_pipeline.py --symbols AAPL --start 2024-01-01 --end 2024-01-10
# With training:
python scripts/run_pipeline.py --symbols AAPL --start 2024-01-01 --end 2024-01-10 --train
# With paper trading:
python scripts/run_pipeline.py --symbols AAPL --start 2024-01-01 --end 2024-01-10 --train --paper
```

### 4. Run Individual Stages

#### Ingest Market Data
```bash
python scripts/ingest_market_data.py --symbols AAPL --start 2024-01-01 --end 2024-01-31 --source synthetic
python scripts/ingest_market_data.py --symbols AAPL --start 2024-01-01 --end 2024-01-31 --source yfinance
```

#### Feature Engineering
```bash
python scripts/feature_engineer.py --symbols AAPL --start 2024-01-01 --end 2024-01-31
```

#### Train Model
```bash
python scripts/train_tft_baseline.py --max-epochs 30 --symbols AAPL
```

#### Generate Predictions
```bash
python scripts/predict.py --symbols AAPL --start 2024-01-01 --end 2024-01-31 --use-synthetic
```

#### Run Backtest
```bash
python scripts/run_backtest.py --symbols AAPL --start 2024-01-01 --end 2024-12-31 --initial-cash 10000
```

#### Paper Trade Smoke Test
```bash
python scripts/paper_trade_smoke.py --symbols AAPL --simulate
```

### 5. Smoke Tests (Quick Validation)
```bash
python scripts/smoke_ingest.py      # Verify ingestion works
python scripts/smoke_features.py    # Verify feature engineering
python scripts/smoke_train.py       # Verify model training
python scripts/smoke_predict.py     # Verify predictions
python scripts/smoke_backtest.py    # Verify backtesting
```

### 6. Debug Commands
```bash
python scripts/verify_db.py                    # Check all tables and row counts
python scripts/verify_db.py --tables market_raw_day predictions_tft  # Check specific tables
docker compose logs timescaledb -f             # Stream DB logs
docker compose ps                              # Check service status
```

## Tools/Services
- Orchestrator: scripts/run_pipeline.py
- E2E Test: scripts/e2e_test.py
- Agents: src/agents/*
- Kafka + Schema Registry via docker-compose
- TimescaleDB audit via db/init.sql

## Outputs
- Kafka topics: market.raw, market.engineered, predictions.tft, trade.decisions, risk.approved, risk.rejected, execution.completed
- TimescaleDB tables: market_raw_day, features_tft, predictions_tft, trade_decisions, trades, backtest_runs, audit_events
- Metrics: /metrics per agent

## Steps (Automated by run_pipeline.py)
1. Start infrastructure via docker compose.
2. Preflight checks (verify services running, DB connected).
3. Ingest market data (synthetic or real via yfinance).
4. Engineer features and store in features_tft.
5. Train model (optional, --train flag).
6. Generate predictions from model.
7. Run backtest with simulated trades.
8. Verify results stored in backtest_runs table.
9. Paper trade demo (optional, --paper flag).

## Edge Cases
- Duplicate idempotency_key → skip processing (ON CONFLICT DO NOTHING)
- Schema registry unavailable → retry with backoff
- Execution approval missing → emit blocked execution status
- Missing market data → suggest ingest command

## Rollback/Replay
- Re-run consumers from earliest offsets after clearing audit table or using new idempotency keys
- Backtest can be re-run with different parameters at any time
