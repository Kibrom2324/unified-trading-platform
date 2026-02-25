from __future__ import annotations

from typing import Any

import httpx
import pytest

from src.core.config import LLMConfig
from src.core.logging import set_correlation_id
from src.llm.client import LLMClient


@pytest.mark.asyncio
async def test_llm_client_chat_sends_payload_and_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class DummyAsyncClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        async def __aenter__(self) -> "DummyAsyncClient":
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def post(self, url: str, json: dict[str, Any]) -> httpx.Response:
            captured["url"] = url
            captured["json"] = json
            request = httpx.Request("POST", url)
            return httpx.Response(
                200,
                json={
                    "id": "test",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}}
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                },
                request=request,
            )

    monkeypatch.setattr("src.llm.client.httpx.AsyncClient", DummyAsyncClient)

    config = LLMConfig(
        provider="local",
        base_url="http://localhost:11434/v1",
        model="llama-3.1-70b-instruct-q4_k_m",
        timeout_seconds=60,
        max_tokens=4096,
        temperature=0.2,
        top_p=0.9,
        concurrency_limit=1,
    )
    client = LLMClient(config)

    set_correlation_id("test-correlation")
    response = await client.chat(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
        ]
    )

    assert response == "ok"
    assert captured["json"]["model"] == config.model
    assert captured["json"]["messages"][1]["content"] == "hello"
