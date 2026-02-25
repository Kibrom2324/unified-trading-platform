from __future__ import annotations

import pytest
from unittest.mock import AsyncMock


class DummyResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.content = b"{}"

    def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_submit_bracket_payload():
    from src.integrations.alpaca_client import AlpacaClient

    client = AlpacaClient.__new__(AlpacaClient)

    captured = {}

    async def fake_request(method, path, json_data=None):
        captured["method"] = method
        captured["path"] = path
        captured["json"] = json_data
        return DummyResponse(
            201,
            {
                "id": "parent-1",
                "status": "accepted",
                "legs": [
                    {"id": "tp-1", "type": "limit"},
                    {"id": "sl-1", "type": "stop"},
                ],
            },
        )

    client._request_with_retry = AsyncMock(side_effect=fake_request)

    result = await AlpacaClient.submit_bracket_order(
        client,
        symbol="AAPL",
        qty=10,
        side="buy",
        take_profit_price=103.0,
        stop_loss_price=98.5,
    )

    assert result.success is True
    assert captured["path"] == "/v2/orders"
    payload = captured["json"]
    assert payload["order_class"] == "bracket"
    assert payload["take_profit"]["limit_price"] == "103.00"
    assert payload["stop_loss"]["stop_price"] == "98.50"
