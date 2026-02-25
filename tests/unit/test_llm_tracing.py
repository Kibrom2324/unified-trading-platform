"""
Unit tests for LLM tracing.

Tests cover:
1. Tracing ON emits LLM_TRACE log line
2. Tracing OFF emits nothing
3. Redaction masks keys/emails/long numbers
4. DB persistence (mocked)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.tracing import (
    LLMTraceRecord,
    LLMTracer,
    LLMTracerConfig,
    redact_sensitive,
    truncate_preview,
)


class TestRedaction:
    """Test sensitive data redaction."""
    
    def test_redact_api_key_sk_pattern(self):
        """API keys with sk- prefix should be redacted."""
        text = "My key is sk-1234567890abcdefghijklmnop"
        result = redact_sensitive(text)
        assert "sk-123456" not in result
        assert "[REDACTED" in result
    
    def test_redact_email(self):
        """Email addresses should be redacted."""
        text = "Contact me at john.doe@example.com for details"
        result = redact_sensitive(text)
        assert "john.doe@example.com" not in result
        assert "[REDACTED_EMAIL]" in result
    
    def test_redact_long_numbers(self):
        """Numbers with 12+ digits should be redacted."""
        text = "Card number: 4111111111111111"
        result = redact_sensitive(text)
        assert "4111111111111111" not in result
        assert "[REDACTED_NUMBER]" in result
    
    def test_redact_short_numbers_preserved(self):
        """Numbers with fewer than 12 digits should be preserved."""
        text = "Order ID: 12345678"
        result = redact_sensitive(text)
        assert "12345678" in result
    
    def test_redact_bearer_token(self):
        """Bearer tokens should be redacted."""
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result = redact_sensitive(text)
        assert "eyJhbGci" not in result
        assert "[REDACTED_TOKEN]" in result
    
    def test_redact_env_var_values(self):
        """Environment variable values should be redacted."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-secret-key-12345'}):
            text = "Using key: test-secret-key-12345"
            result = redact_sensitive(text)
            assert "test-secret-key-12345" not in result
            assert "[REDACTED_OPENAI_API_KEY]" in result
    
    def test_redact_preserves_normal_text(self):
        """Normal text without sensitive data should be preserved."""
        text = "This is a normal message about trading AAPL stock"
        result = redact_sensitive(text)
        assert result == text


class TestTruncatePreview:
    """Test text truncation for previews."""
    
    def test_short_text_unchanged(self):
        """Text shorter than max_chars should be unchanged."""
        text = "Short message"
        result = truncate_preview(text, max_chars=100)
        assert result == text
    
    def test_long_text_truncated(self):
        """Text longer than max_chars should be truncated."""
        text = "A" * 200
        result = truncate_preview(text, max_chars=50)
        assert len(result) < 200
        assert "... [" in result
        assert "more chars]" in result
    
    def test_empty_text(self):
        """Empty text should return empty string."""
        assert truncate_preview("", max_chars=100) == ""
        assert truncate_preview(None, max_chars=100) == ""


class TestLLMTraceRecord:
    """Test LLMTraceRecord dataclass."""
    
    def test_to_log_dict_basic(self):
        """to_log_dict should return properly formatted dict."""
        record = LLMTraceRecord(
            ts=datetime(2026, 1, 16, 12, 0, 0, tzinfo=timezone.utc),
            correlation_id="test-123",
            agent_name="TestAgent",
            model="llama3.2:1b",
            base_url="http://localhost:11434/v1",
            latency_ms=150.5,
            status="ok",
            error_message=None,
            request={"messages": [{"role": "user", "content": "Hello"}]},
            response={"content": "Hi there!"},
            prompt_tokens=10,
            completion_tokens=5,
        )
        
        log_dict = record.to_log_dict(preview_chars=100)
        
        assert log_dict["correlation_id"] == "test-123"
        assert log_dict["agent_name"] == "TestAgent"
        assert log_dict["model"] == "llama3.2:1b"
        assert log_dict["latency_ms"] == 150.5
        assert log_dict["status"] == "ok"
        assert log_dict["prompt_tokens"] == 10
        assert log_dict["completion_tokens"] == 5
    
    def test_to_log_dict_redacts_sensitive(self):
        """to_log_dict should redact sensitive data in previews."""
        record = LLMTraceRecord(
            ts=datetime.now(tz=timezone.utc),
            correlation_id="test-123",
            agent_name="TestAgent",
            model="llama3.2:1b",
            base_url="http://localhost:11434/v1",
            latency_ms=100,
            status="ok",
            error_message=None,
            request={"messages": [{"role": "user", "content": "My email is test@example.com"}]},
            response={"content": "Got it"},
        )
        
        log_dict = record.to_log_dict()
        
        assert "test@example.com" not in log_dict["request_preview"]
        assert "[REDACTED_EMAIL]" in log_dict["request_preview"]


class TestLLMTracer:
    """Test LLMTracer class."""
    
    @pytest.mark.asyncio
    async def test_trace_when_enabled_logs(self):
        """When enabled, trace should emit log."""
        config = LLMTracerConfig(
            enabled=True,
            preview_chars=100,
            store_db=False,
        )
        tracer = LLMTracer(config)
        
        with patch.object(tracer, '_logger') as mock_logger:
            await tracer.trace(
                agent_name="TestAgent",
                correlation_id="test-123",
                model="llama3.2:1b",
                base_url="http://localhost:11434/v1",
                request_messages=[{"role": "user", "content": "Hello"}],
                response_content="Hi!",
                latency_ms=50,
                status="ok",
            )
            
            # Should have called logger.info with LLM_TRACE
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert call_args[0][0] == "LLM_TRACE"
    
    @pytest.mark.asyncio
    async def test_trace_when_disabled_no_log(self):
        """When disabled, trace should not emit log."""
        config = LLMTracerConfig(
            enabled=False,
            store_db=False,
        )
        tracer = LLMTracer(config)
        
        with patch.object(tracer, '_logger') as mock_logger:
            await tracer.trace(
                agent_name="TestAgent",
                correlation_id="test-123",
                model="llama3.2:1b",
                base_url="http://localhost:11434/v1",
                request_messages=[{"role": "user", "content": "Hello"}],
                response_content="Hi!",
                latency_ms=50,
                status="ok",
            )
            
            # Should NOT have called logger
            mock_logger.info.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_trace_error_status(self):
        """Trace should handle error status correctly."""
        config = LLMTracerConfig(
            enabled=True,
            store_db=False,
        )
        tracer = LLMTracer(config)
        
        with patch.object(tracer, '_logger') as mock_logger:
            await tracer.trace(
                agent_name="TestAgent",
                correlation_id="test-123",
                model="llama3.2:1b",
                base_url="http://localhost:11434/v1",
                request_messages=[{"role": "user", "content": "Hello"}],
                response_content=None,
                latency_ms=100,
                status="error",
                error_message="Connection timeout",
            )
            
            mock_logger.info.assert_called_once()
            call_kwargs = mock_logger.info.call_args[1]
            assert call_kwargs["status"] == "error"
            assert call_kwargs["error_message"] == "Connection timeout"
    
    @pytest.mark.asyncio
    async def test_persist_trace_when_db_enabled(self):
        """When store_db is True and pool exists, should attempt persist."""
        config = LLMTracerConfig(
            enabled=True,
            store_db=True,
            db_dsn="postgresql://test:test@localhost/test",
        )
        tracer = LLMTracer(config)
        
        # Just test that _persist_trace handles missing pool gracefully
        tracer._pool = None  # No pool = no persist
        
        record = LLMTraceRecord(
            ts=datetime.now(tz=timezone.utc),
            correlation_id="test-123",
            agent_name="TestAgent",
            model="llama3.2:1b",
            base_url="http://localhost:11434/v1",
            latency_ms=100,
            status="ok",
            error_message=None,
            request={"messages": []},
            response={"content": "test"},
        )
        
        # Should not raise when pool is None
        await tracer._persist_trace(record)


class TestLLMTracerConfig:
    """Test LLMTracerConfig defaults."""
    
    def test_default_values(self):
        """Default config values should be sensible."""
        config = LLMTracerConfig()
        
        assert config.enabled is True
        assert config.preview_chars == 800
        assert config.log_level == "INFO"
        assert config.store_db is True
    
    def test_custom_values(self):
        """Custom config values should be applied."""
        config = LLMTracerConfig(
            enabled=False,
            preview_chars=200,
            log_level="DEBUG",
            store_db=False,
        )
        
        assert config.enabled is False
        assert config.preview_chars == 200
        assert config.log_level == "DEBUG"
        assert config.store_db is False
