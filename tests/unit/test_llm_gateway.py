"""Tests for LLM gateway orchestration functions."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import LLMConfig


def _make_mock_settings() -> MagicMock:
    """Create mock settings with valid LLM config."""
    mock = MagicMock()
    mock.llm = LLMConfig(
        provider="local",
        base_url="http://localhost:11434/v1",
        model="llama-3.1-70b-instruct-q4_k_m",
    )
    return mock


@pytest.mark.asyncio
async def test_generate_plan_loads_directive_and_returns_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test generate_plan loads directive file and calls LLM."""
    # Create a temporary directive file
    directive_content = "# Test Directive\nDo something."
    directive_dir = tmp_path / "directives"
    directive_dir.mkdir(parents=True, exist_ok=True)
    directive_file = directive_dir / "trading_pipeline.md"
    directive_file.write_text(directive_content)

    # Import after setting up mocks
    from src.orchestration import llm_gateway

    async def fake_chat(
        self: Any, messages: list[dict[str, str]], *, tools: list | None = None
    ) -> str:
        assert messages
        # Check that directive content is in the message
        assert "Test Directive" in messages[1]["content"]
        return "plan"

    # Mock settings
    monkeypatch.setattr(llm_gateway, "load_settings", lambda: _make_mock_settings())

    # Mock LLMClient.chat
    monkeypatch.setattr(llm_gateway.LLMClient, "chat", fake_chat)

    # Mock the path resolution to use tmp_path
    class MockPath:
        """Mock Path that returns tmp_path for directive resolution."""

        def __init__(self, path: str) -> None:
            self._path = Path(path)

        def resolve(self) -> "MockPath":
            return self

        @property
        def parents(self) -> list[Path]:
            # Return tmp_path as the project root
            return [tmp_path, tmp_path, tmp_path]

    monkeypatch.setattr(llm_gateway, "Path", MockPath)

    result = await llm_gateway.generate_plan("trading_pipeline", {"symbol": "AAPL"})
    assert result == "plan"


@pytest.mark.asyncio
async def test_explain_run_calls_audit_and_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test explain_run fetches audit trail and summarizes via LLM."""
    from src.orchestration import llm_gateway

    calls = {"llm": 0, "audit": 0}

    async def fake_chat(
        self: Any, messages: list[dict[str, str]], *, tools: list | None = None
    ) -> str:
        calls["llm"] += 1
        return "summary"

    async def fake_audit(_: str) -> list[dict[str, Any]]:
        calls["audit"] += 1
        return [{"event_type": "ingestion", "ts": "2024-01-01T00:00:00Z", "payload": {}}]

    monkeypatch.setattr(llm_gateway, "load_settings", lambda: _make_mock_settings())
    monkeypatch.setattr(llm_gateway.LLMClient, "chat", fake_chat)
    monkeypatch.setattr(llm_gateway, "_fetch_audit_trail", fake_audit)

    result = await llm_gateway.explain_run("cid-123")
    assert result == "summary"
    assert calls["llm"] == 1
    assert calls["audit"] == 1
