from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import asyncpg
import numpy as np

from src.agents.base_agent import BaseAgent
from src.audit.trail_logger import AuditTrailLogger
from src.core.config import load_settings
from src.core.logging import set_correlation_id
from src.features.engine import FeatureComputer, FeatureStore
from src.models.events_engineered import MarketEngineeredEvent
from src.models.events_market import MarketRawEvent
from src.monitoring.metrics import EVENTS_PROCESSED, EVENT_LATENCY
from src.streaming.kafka_consumer import AvroConsumer
from src.streaming.kafka_producer import AvroProducer
from src.streaming.topics import MARKET_ENGINEERED, MARKET_RAW

RAW_SCHEMA = "src/schemas/market_raw.avsc"
ENG_SCHEMA = "src/schemas/market_engineered.avsc"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class FeatureEngineeringAgent(BaseAgent):
    """Feature engineering agent with real calculations.

    Features computed:
    - returns_1, returns_5, returns_15: period returns
    - volatility_5, volatility_15: rolling volatility
    - sma_5, sma_20: simple moving averages
    - ema_5, ema_20: exponential moving averages
    - rsi_14: RSI
    - volume_zscore: volume z-score vs 20-bar window
    - minute_of_day, day_of_week: calendar features
    
    Persists features to TimescaleDB table `features_tft` for training.
    """

    def __init__(self) -> None:
        settings = load_settings()
        super().__init__("feature_engineering", settings.app.http.feature_engineering_port)
        with open(RAW_SCHEMA, "r", encoding="utf-8") as handle:
            raw_schema = handle.read()
        self.consumer = AvroConsumer(
            self.settings.kafka,
            self.settings.kafka.group_ids.feature_engineering,
            raw_schema,
            MARKET_RAW,
        )
        self.producer = AvroProducer(self.settings.kafka)
        self.audit = AuditTrailLogger(self.settings)
        self.feature_store = FeatureStore()
        self._db_pool: asyncpg.Pool | None = None
        self._time_idx_counter: dict[str, int] = {}
        with open(ENG_SCHEMA, "r", encoding="utf-8") as handle:
            self._eng_schema = handle.read()

    async def _connect_db(self) -> None:
        """Connect to TimescaleDB for feature persistence."""
        if self._db_pool is not None:
            return
        from shared.db.dsn import build_dsn
        dsn = _env("DATABASE_URL") or build_dsn()
        self._db_pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)
        self.logger.info("db_connected", dsn=dsn.split("@")[1])

    async def _persist_features(self, event: MarketEngineeredEvent, raw_event: MarketRawEvent) -> None:
        """Persist engineered features to features_tft table."""
        if self._db_pool is None:
            await self._connect_db()
        assert self._db_pool is not None

        ts_dt = self._parse_ts(event.ts)
        
        # Increment time index for this ticker
        if event.symbol not in self._time_idx_counter:
            self._time_idx_counter[event.symbol] = 0
        self._time_idx_counter[event.symbol] += 1
        time_idx = self._time_idx_counter[event.symbol]

        async with self._db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO features_tft (
                    ticker, ts, time_idx, split, price,
                    open, high, low, close, volume,
                    returns_1, returns_5, returns_15, target_return,
                    volatility_5, volatility_15,
                    sma_5, sma_20, ema_5, ema_20,
                    rsi_14, volume_zscore,
                    minute_of_day, hour_of_day, day_of_week,
                    idempotency_key
                ) VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8, $9, $10,
                    $11, $12, $13, $14,
                    $15, $16,
                    $17, $18, $19, $20,
                    $21, $22,
                    $23, $24, $25,
                    $26
                )
                ON CONFLICT (ts, ticker) DO UPDATE SET
                    time_idx = EXCLUDED.time_idx,
                    price = EXCLUDED.price,
                    returns_1 = EXCLUDED.returns_1,
                    returns_5 = EXCLUDED.returns_5,
                    returns_15 = EXCLUDED.returns_15,
                    volatility_5 = EXCLUDED.volatility_5,
                    volatility_15 = EXCLUDED.volatility_15,
                    sma_5 = EXCLUDED.sma_5,
                    sma_20 = EXCLUDED.sma_20,
                    ema_5 = EXCLUDED.ema_5,
                    ema_20 = EXCLUDED.ema_20,
                    rsi_14 = EXCLUDED.rsi_14,
                    volume_zscore = EXCLUDED.volume_zscore
                """,
                event.symbol,
                ts_dt,
                time_idx,
                "train",  # Default split
                event.price,
                getattr(raw_event, "open", event.price),
                getattr(raw_event, "high", event.price),
                getattr(raw_event, "low", event.price),
                getattr(raw_event, "close", event.price),
                event.volume,
                event.returns_1,
                event.returns_5,
                getattr(event, "returns_15", 0.0),
                None,  # target_return computed later
                getattr(event, "volatility_5", 0.0),
                getattr(event, "volatility_15", 0.0),
                getattr(event, "sma_5", event.price),
                getattr(event, "sma_20", event.price),
                getattr(event, "ema_5", event.price),
                getattr(event, "ema_20", event.price),
                getattr(event, "rsi_14", 50.0),
                getattr(event, "volume_zscore", 0.0),
                getattr(event, "minute_of_day", ts_dt.hour * 60 + ts_dt.minute),
                ts_dt.hour,
                ts_dt.weekday(),
                event.idempotency_key,
            )

    async def start(self) -> None:
        self.app.add_event_handler("shutdown", self._shutdown)
        await self.audit.connect()
        await self.feature_store.connect()
        await self._connect_db()
        self.track_task(asyncio.create_task(self._consume_loop()))

    async def _shutdown(self) -> None:
        self.consumer.close()
        self.producer.close()
        await self.feature_store.close()
        await self.audit.close()
        if self._db_pool:
            await self._db_pool.close()

    def _parse_ts(self, ts_value: str | datetime) -> datetime:
        """Parse timestamp from ISO string or datetime."""
        if isinstance(ts_value, datetime):
            return ts_value
        return datetime.fromisoformat(ts_value.replace("Z", "+00:00"))

    async def _compute_features(
        self, event: MarketRawEvent
    ) -> MarketEngineeredEvent:
        """Compute all features for the event."""
        state = await self.feature_store.get_state(event.symbol)
        ts_dt = self._parse_ts(event.ts)
        price = event.price
        volume = event.volume

        state.add_tick(price, volume, event.ts if isinstance(event.ts, str) else event.ts.isoformat())
        prices = list(state.prices)
        volumes = list(state.volumes)
        returns = list(state.returns)
        returns_1 = FeatureComputer.returns(prices, 1)
        returns_5 = FeatureComputer.returns(prices, 5)
        returns_15 = FeatureComputer.returns(prices, 15)
        volatility_5 = FeatureComputer.rolling_volatility(returns, 5)
        volatility_15 = FeatureComputer.rolling_volatility(returns, 15)
        sma_5 = FeatureComputer.sma(prices, 5)
        sma_20 = FeatureComputer.sma(prices, 20)
        ema_5 = FeatureComputer.ema(prices, 5)
        ema_20 = FeatureComputer.ema(prices, 20)
        rsi_14 = FeatureComputer.rsi(prices, 14)
        volume_zscore = FeatureComputer.volume_zscore(volumes, 20)
        minute_of_day = FeatureComputer.minute_of_day(ts_dt)
        day_of_week = FeatureComputer.day_of_week(ts_dt)
        await self.feature_store.save_state(state)

        return MarketEngineeredEvent(
            idempotency_key=event.idempotency_key,
            symbol=event.symbol,
            source=self.name,
            correlation_id=event.correlation_id,
            price=price,
            volume=volume,
            returns_1=returns_1,
            returns_5=returns_5,
            returns_15=returns_15,
            volatility_5=volatility_5,
            volatility_15=volatility_15,
            sma_5=sma_5,
            sma_20=sma_20,
            ema_5=ema_5,
            ema_20=ema_20,
            rsi_14=rsi_14,
            volume_zscore=volume_zscore,
            minute_of_day=minute_of_day,
            day_of_week=day_of_week,
        )

    async def _consume_loop(self) -> None:
        while not self.is_shutting_down:
            try:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    await asyncio.sleep(0.1)
                    continue
                payload = msg.value()
                event = MarketRawEvent.model_validate(payload)
                set_correlation_id(str(event.correlation_id))
                if await self.audit.is_duplicate(MARKET_ENGINEERED, event.idempotency_key):
                    self.consumer.commit()
                    continue
                start_ts = datetime.now(tz=timezone.utc)
                engineered = await self._compute_features(event)
                
                # Persist to features_tft table (idempotent UPSERT)
                try:
                    await self._persist_features(engineered, event)
                except Exception as e:
                    self.logger.error("persist_features_error", error=str(e), symbol=event.symbol)
                
                payload_out = engineered.to_avro_dict()
                self.producer.produce(MARKET_ENGINEERED, self._eng_schema, payload_out, key=event.symbol)
                await self.audit.write_event(MARKET_ENGINEERED, payload_out)
                self.consumer.commit()
                duration = (datetime.now(tz=timezone.utc) - start_ts).total_seconds()
                EVENTS_PROCESSED.labels(agent=self.name, event_type=MARKET_ENGINEERED).inc()
                EVENT_LATENCY.labels(agent=self.name, event_type=MARKET_ENGINEERED).observe(duration)
                self.logger.info(
                    "feature_computed",
                    symbol=event.symbol,
                    returns_1=engineered.returns_1,
                    volatility_5=engineered.volatility_5,
                    rsi_14=engineered.rsi_14,
                )
            except asyncio.CancelledError:
                self.logger.info("consume_loop_cancelled")
                break
            except Exception:
                self.logger.exception("consume_loop_error")
                await asyncio.sleep(1.0)


def main() -> None:
    FeatureEngineeringAgent().run()


async def run_feature_engineering_batch(
    symbols: list[str],
    start: datetime,
    end: datetime,
    source: str = "day",
) -> int:
    """Batch feature engineering for backfills or CLI.

    Reads market_raw_day or market_raw_minute and writes to features_tft.
    This is deterministic and idempotent.
    """
    table = "market_raw_day" if source == "day" else "market_raw_minute"
    from shared.db.dsn import build_dsn
    dsn = _env("DATABASE_URL") or build_dsn()
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=3)
    total_inserted = 0
    try:
        async with pool.acquire() as conn:
            for symbol in symbols:
                rows = await conn.fetch(
                    f"""
                    SELECT ticker, ts, open, high, low, close, volume
                    FROM {table}
                    WHERE ticker = $1 AND ts >= $2 AND ts <= $3
                    ORDER BY ts
                    """,
                    symbol,
                    start,
                    end,
                )
                if len(rows) < 5:
                    continue

                timestamps = [r["ts"] for r in rows]
                opens = np.array([float(r["open"]) for r in rows])
                highs = np.array([float(r["high"]) for r in rows])
                lows = np.array([float(r["low"]) for r in rows])
                closes = np.array([float(r["close"]) for r in rows])
                volumes = np.array([float(r["volume"]) for r in rows])
                n = len(rows)

                returns_1 = np.zeros(n)
                returns_1[1:] = np.log(closes[1:] / closes[:-1])
                returns_5 = np.zeros(n)
                if n > 5:
                    returns_5[5:] = np.log(closes[5:] / closes[:-5])

                volatility_5 = np.zeros(n)
                for i in range(5, n):
                    volatility_5[i] = np.std(returns_1[i - 5 : i])

                sma_5 = np.convolve(closes, np.ones(5) / 5, mode="same")
                sma_20 = np.convolve(closes, np.ones(min(20, n)) / min(20, n), mode="same")

                vol_mean = np.mean(volumes)
                vol_std = np.std(volumes) + 1e-8
                volume_zscore = (volumes - vol_mean) / vol_std

                for i in range(n):
                    ts = timestamps[i]
                    idempotency_key = f"{symbol}:{ts.isoformat()}:features"
                    await conn.execute(
                        """
                        INSERT INTO features_tft (
                            ticker, ts, time_idx, split, price, open, high, low, close, volume,
                            returns_1, returns_5, returns_15, target_return,
                            volatility_5, volatility_15,
                            sma_5, sma_20, ema_5, ema_20, rsi_14, volume_zscore,
                            minute_of_day, hour_of_day, day_of_week,
                            idempotency_key
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                            $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22,
                            $23, $24, $25, $26
                        )
                        ON CONFLICT (ts, ticker) DO NOTHING
                        """,
                        symbol,
                        ts,
                        i,
                        "train" if i < n * 0.8 else "val",
                        float(closes[i]),
                        float(opens[i]),
                        float(highs[i]),
                        float(lows[i]),
                        float(closes[i]),
                        float(volumes[i]),
                        float(returns_1[i]),
                        float(returns_5[i]) if n > 5 else 0.0,
                        0.0,
                        0.0,
                        float(volatility_5[i]),
                        0.0,
                        float(sma_5[i]),
                        float(sma_20[i]),
                        float(closes[i]),
                        float(closes[i]),
                        50.0,
                        float(volume_zscore[i]),
                        ts.hour * 60 + ts.minute,
                        ts.hour,
                        ts.weekday(),
                        idempotency_key,
                    )
                    total_inserted += 1
    finally:
        await pool.close()
    return total_inserted


if __name__ == "__main__":
    main()
