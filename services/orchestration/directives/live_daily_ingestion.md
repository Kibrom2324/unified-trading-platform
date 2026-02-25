# Live Daily Ingestion Directive

## Goal
Fetch the latest daily OHLCV data from a FREE source (Yahoo Finance) and publish it to Kafka for processing by the trading pipeline.

## Success Criteria
- [ ] Latest bars fetched for all configured symbols
- [ ] Events published to `market.raw` topic with correct schema
- [ ] Audit trail recorded in TimescaleDB
- [ ] No duplicate events (idempotency enforced)
- [ ] Pipeline triggered successfully (if requested)

## Inputs
| Input | Source | Description |
|-------|--------|-------------|
| Symbols | `config/app.yaml` or CLI | Stock symbols to fetch (e.g., AAPL, MSFT) |
| Provider | `config/market_data.yaml` | Data source (default: yfinance) |
| Trigger Pipeline | CLI flag | Whether to run full pipeline after ingestion |

## Tools/Scripts
| Tool | Path | Description |
|------|------|-------------|
| Live Ingestion | `execution/run_live_ingestion.py` | Main script for live data fetch |
| SuperAGI Tool | `superagi/tools/run_live_ingestion.py` | Tool wrapper for orchestration |
| YFinance Provider | `src/market_data/providers.py` | Data provider implementation |

## Outputs
| Output | Destination | Format |
|--------|-------------|--------|
| Market events | Kafka `market.raw` | Avro (MarketRawEvent) |
| Audit records | TimescaleDB `audit_events` | JSON |
| Metrics | Prometheus `/metrics` | Counter/Histogram |

## Event Flow
```
[Yahoo Finance API]
        ↓
[YFinanceProvider.get_latest_bar()]
        ↓
[MarketRawEvent (Avro)]
        ↓
[Kafka: market.raw]
        ↓ (if --trigger-pipeline)
[Feature Engineering Agent] → [market.engineered]
        ↓
[Prediction Agent] → [predictions.tft]
        ↓
[Risk Agent] → [risk.approved / risk.rejected]
        ↓
[Execution Agent] → [execution.completed / execution.blocked]
```

## Usage

### Basic: Fetch latest data
```bash
# Use default symbols from config
make live-ingest

# Specify symbols
python -m execution.run_live_ingestion --symbols AAPL,MSFT,NVDA
```

### With Pipeline Trigger
```bash
# Fetch data and run full pipeline
make live-pipeline

# Or directly
python -m execution.run_live_ingestion --trigger-pipeline
```

### Dry Run (Testing)
```bash
python -m execution.run_live_ingestion --dry-run
```

### From SuperAGI
```python
# In SuperAGI agent
result = awet_run_live_ingestion(
    symbols="AAPL,MSFT",
    trigger_pipeline=True,
    dry_run=False
)
```

## Edge Cases & Failure Modes

### No Data Available
- **Symptom**: `get_latest_bar()` returns None
- **Cause**: Market closed, symbol invalid, or API issue
- **Action**: Log warning, skip symbol, continue with others
- **Recovery**: Retry on next run

### Rate Limiting
- **Symptom**: HTTP 429 or connection errors
- **Cause**: Too many requests to Yahoo Finance
- **Action**: Built-in rate limiting (30/min default)
- **Recovery**: Automatic backoff via circuit breaker

### Duplicate Events
- **Symptom**: Same bar already processed
- **Cause**: Script run multiple times same day
- **Action**: Skip via idempotency_key check
- **Recovery**: None needed (safe to ignore)

### Kafka Unavailable
- **Symptom**: Connection refused to Kafka
- **Cause**: Docker not running or Kafka down
- **Action**: Fail with clear error message
- **Recovery**: `docker compose up -d kafka schema-registry`

### Pipeline Timeout
- **Symptom**: Pipeline doesn't complete in time
- **Cause**: Slow processing or stuck agent
- **Action**: Return partial success, log warning
- **Recovery**: Check agent logs, restart if needed

## Rollback/Replay

### Replay Data
If data was missed or corrupted:
```bash
# Backfill specific date range
python -m execution.yfinance_backfill \
    --symbols AAPL,MSFT \
    --start 2025-01-10 \
    --end 2025-01-15
```

### Clear and Rerun
```bash
# Clear today's data (careful!)
make clean-db

# Re-run ingestion
make live-ingest
```

## Observability

### Metrics
- `live_bars_ingested_total{symbol, source}` - Count of bars ingested
- `live_ingestion_latency_seconds{source}` - Total ingestion duration
- `live_ingestion_failures_total{symbol, source}` - Failed fetches

### Logs
All logs include `correlation_id` for tracing:
```json
{
  "event": "live_ingestion_complete",
  "correlation_id": "abc-123",
  "bars_ingested": 5,
  "symbols": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
  "duration_seconds": 3.45
}
```

### Verify Data
```bash
# Check Kafka
docker exec kafka kafka-console-consumer \
    --bootstrap-server localhost:9092 \
    --topic market.raw \
    --from-beginning \
    --max-messages 5

# Check TimescaleDB
docker exec timescaledb psql -U awet -c \
    "SELECT topic, symbol, ts FROM audit_events ORDER BY ts DESC LIMIT 10;"
```

## Schedule Recommendations

For daily trading:
- **US Markets**: Run at 16:30 ET (after market close)
- **Cron**: `30 20 * * 1-5` (UTC, Mon-Fri)
- **Manual**: Run anytime for latest available data

## Configuration

### config/market_data.yaml
```yaml
market_data:
  provider: yfinance  # FREE, no API key
  timeframe: 1D
  yfinance:
    rate_limit_per_minute: 30
```

### config/app.yaml
```yaml
app:
  symbols:
    - AAPL
    - MSFT
    - GOOGL
    - AMZN
    - NVDA
```
