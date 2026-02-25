from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class DummyLogging:
    level = "INFO"


class DummyLLM:
    model = "test-model"
    base_url = "http://localhost"
    trace_enabled = True
    trace_preview_chars = 800
    trace_log_level = "INFO"
    trace_store_db = False


class DummyKafka:
    bootstrap_servers = ["localhost:9092"]
    schema_registry_url = "http://localhost:8081"


class DummySettings:
    logging = DummyLogging()
    llm = DummyLLM()
    kafka = DummyKafka()


@pytest.mark.asyncio
async def test_paper_trade_live_traces_and_trade():
    from scripts import paper_trade_live

    tracer = MagicMock()
    tracer.trace = AsyncMock()

    with patch("scripts.paper_trade_live.load_settings", return_value=DummySettings()):
        with patch("scripts.paper_trade_live.get_tracer", return_value=tracer):
            with patch("scripts.paper_trade_live._publish_risk_event", new=AsyncMock()):
                with patch("scripts.paper_trade_live._wait_for_trade", new=AsyncMock(return_value={
                    "alpaca_order_id": "order-1",
                    "status": "filled",
                    "filled_qty": 1,
                    "avg_fill_price": 100.0,
                })):
                    result = await paper_trade_live.run_once("NVDA", 1, "buy", 1)

    assert result == 0
    tracer.trace.assert_called_once()


@pytest.mark.asyncio
async def test_wait_for_trade_polls_until_filled():
    from scripts import paper_trade_live

    calls = {"count": 0}

    async def reconcile_fn():
        calls["count"] += 1

    class DummyConn:
        _shared_calls = 0

        async def fetchrow(self, *args, **kwargs):
            DummyConn._shared_calls += 1
            if DummyConn._shared_calls < 2:
                return None
            return {
                "alpaca_order_id": "order-1",
                "status": "filled",
                "filled_qty": 1,
                "avg_fill_price": 100.0,
            }

        async def close(self):
            return None

    DummyConn._shared_calls = 0

    async def fake_connect():
        return DummyConn()

    with patch("scripts.paper_trade_live._connect_db", new=fake_connect):
        result = await paper_trade_live._wait_for_trade("corr-1", 5, reconcile_fn)

    assert result is not None
    assert calls["count"] >= 1
