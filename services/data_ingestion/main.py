from __future__ import annotations

import asyncio
import os
import random
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from src.agents.base_agent import BaseAgent
from src.audit.trail_logger import AuditTrailLogger
from src.core.config import load_settings
from src.core.logging import set_correlation_id
from src.market_data.providers import MarketDataProvider, OHLCVBar, create_provider
from src.models.events_market import MarketRawEvent
from src.monitoring.metrics import EVENTS_PROCESSED, EVENT_LATENCY
from src.streaming.kafka_producer import AvroProducer
from src.streaming.topics import MARKET_RAW

SCHEMA_PATH = "src/schemas/market_raw.avsc"


class DataIngestionAgent(BaseAgent):
    def __init__(self) -> None:
        settings = load_settings()
        super().__init__("data_ingestion", settings.app.http.data_ingestion_port)
        self.producer = AvroProducer(self.settings.kafka)
        self.audit = AuditTrailLogger(self.settings)
        self._provider: MarketDataProvider | None = None
        self._use_real_data = self._should_use_real_data()
        self._last_bar_ts: dict[str, datetime] = {}

    def _should_use_real_data(self) -> bool:
        """Check if we have a valid data provider configured."""
        provider = os.getenv("MARKET_DATA_PROVIDER", self.settings.market_data.provider)
        # yfinance doesn't need API keys
        if provider == "yfinance":
            return True
        # Polygon/Alpaca need API keys
        return bool(os.getenv("POLYGON_API_KEY") or os.getenv("ALPACA_API_KEY"))

    def _create_provider(self) -> MarketDataProvider:
        """Create market data provider from config."""
        md_cfg = self.settings.market_data
        provider_name = os.getenv("MARKET_DATA_PROVIDER", md_cfg.provider)
        
        if provider_name == "yfinance":
            # yfinance doesn't need API key
            api_key = ""
            base_url = ""
            rate_limit = md_cfg.yfinance.rate_limit_per_minute
        elif provider_name == "polygon":
            api_key = os.getenv("POLYGON_API_KEY", "")
            base_url = md_cfg.polygon.base_url
            rate_limit = md_cfg.rate_limit_per_minute
        else:
            api_key = os.getenv("ALPACA_API_KEY", "")
            base_url = md_cfg.alpaca.base_url
            rate_limit = md_cfg.rate_limit_per_minute
            
        return create_provider(
            provider_name=provider_name,
            api_key=api_key,
            base_url=base_url,
            rate_limit_per_minute=rate_limit,
            timeout_seconds=md_cfg.request_timeout_seconds,
        )

    async def start(self) -> None:
        self.app.add_event_handler("shutdown", self._shutdown)
        await self.audit.connect()
        if self._use_real_data:
            self._provider = self._create_provider()
            self.track_task(asyncio.create_task(self._backfill()))
        self.track_task(asyncio.create_task(self._produce_loop()))

    async def _shutdown(self) -> None:
        if self._provider:
            await self._provider.close()
        self.producer.close()
        await self.audit.close()

    async def _backfill(self) -> None:
        """Backfill historical bars on startup."""
        if not self._provider:
            return
        md_cfg = self.settings.market_data
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(minutes=md_cfg.backfill_minutes)
        self.logger.info("backfill_start", start=start.isoformat(), end=end.isoformat())
        for symbol in self.settings.app.symbols:
            try:
                bars = await self._provider.get_bars(symbol, md_cfg.timeframe, start, end)
                for bar in bars:
                    await self._publish_bar(bar)
                if bars:
                    self._last_bar_ts[symbol] = bars[-1].timestamp
                self.logger.info("backfill_complete", symbol=symbol, bars=len(bars))
            except Exception as exc:
                self.logger.error("backfill_error", symbol=symbol, error=str(exc))
        self.logger.info("backfill_done")

    async def _publish_bar(self, bar: OHLCVBar) -> None:
        """Publish a single OHLCV bar to Kafka."""
        correlation_id = uuid4()
        set_correlation_id(str(correlation_id))
        ts_str = bar.timestamp.strftime("%Y%m%d%H%M%S")
        idempotency_key = f"{bar.symbol}:{ts_str}"
        if await self.audit.is_duplicate(MARKET_RAW, idempotency_key):
            return
        start_ts = datetime.now(tz=timezone.utc)
        event = MarketRawEvent(
            idempotency_key=idempotency_key,
            symbol=bar.symbol,
            source=self.name,
            correlation_id=correlation_id,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            price=bar.close,
            volume=bar.volume,
            vwap=bar.vwap,
            trades=bar.trades,
        )
        with open(SCHEMA_PATH, "r", encoding="utf-8") as handle:
            schema_str = handle.read()
        payload = event.to_avro_dict()
        self.producer.produce(MARKET_RAW, schema_str, payload, key=bar.symbol)
        await self.audit.write_event(MARKET_RAW, payload)
        duration = (datetime.now(tz=timezone.utc) - start_ts).total_seconds()
        EVENTS_PROCESSED.labels(agent=self.name, event_type=MARKET_RAW).inc()
        EVENT_LATENCY.labels(agent=self.name, event_type=MARKET_RAW).observe(duration)
        self.logger.info("produced_market_raw", symbol=bar.symbol, price=bar.close)

    async def _poll_real_data(self) -> None:
        """Poll real market data provider for new bars."""
        if not self._provider:
            return
        md_cfg = self.settings.market_data
        now = datetime.now(tz=timezone.utc)
        for symbol in self.settings.app.symbols:
            try:
                last_ts = self._last_bar_ts.get(symbol)
                if last_ts is None:
                    last_ts = now - timedelta(minutes=5)
                bars = await self._provider.get_bars(symbol, md_cfg.timeframe, last_ts, now)
                new_bars = [b for b in bars if b.timestamp > last_ts]
                for bar in new_bars:
                    await self._publish_bar(bar)
                if new_bars:
                    self._last_bar_ts[symbol] = new_bars[-1].timestamp
            except Exception as exc:
                self.logger.error("poll_error", symbol=symbol, error=str(exc))

    async def _generate_stub_data(self) -> None:
        """Generate stub data when no API key is available (for demo/testing)."""
        with open(SCHEMA_PATH, "r", encoding="utf-8") as handle:
            schema_str = handle.read()
        for symbol in self.settings.app.symbols:
            correlation_id = uuid4()
            set_correlation_id(str(correlation_id))
            now = datetime.now(tz=timezone.utc)
            price = round(100 + random.random() * 5, 4)
            event = MarketRawEvent(
                idempotency_key=f"{symbol}:{int(now.timestamp())}",
                symbol=symbol,
                source=self.name,
                correlation_id=correlation_id,
                open=price * 0.999,
                high=price * 1.002,
                low=price * 0.998,
                close=price,
                price=price,
                volume=random.random() * 1000,
            )
            payload = event.to_avro_dict()
            self.producer.produce(MARKET_RAW, schema_str, payload, key=symbol)
            await self.audit.write_event(MARKET_RAW, payload)
            self.logger.info("produced_stub_market_raw", symbol=symbol, price=price)

    async def _produce_loop(self) -> None:
        """Main loop: poll real data or generate stubs."""
        await self.audit.connect()
        md_cfg = self.settings.market_data
        while not self.is_shutting_down:
          try:
            if self._use_real_data:
                await self._poll_real_data()
                await asyncio.sleep(md_cfg.polling_interval_seconds)
            else:
                await self._generate_stub_data()
                await asyncio.sleep(1)
          except asyncio.CancelledError:
              self.logger.info("produce_loop_cancelled")
              break
          except Exception:
              self.logger.exception("produce_loop_error")
              await asyncio.sleep(1.0)


def main() -> None:
    DataIngestionAgent().run()


if __name__ == "__main__":
    main()
