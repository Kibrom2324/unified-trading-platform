#!/usr/bin/env python3
"""Tests for the SuperAGI agent scheduler."""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

# Import the module we're testing
import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from scripts.schedule_agents import (
    AGENT_CONFIG,
    SuperAGIClient,
    get_agent_id,
    run_agent,
    run_night_trainer,
    run_morning_deployer,
    run_trade_watchdog,
    run_orchestrator,
)


class TestSuperAGIClient:
    """Test SuperAGIClient class."""

    def test_client_requires_api_key(self):
        """Test that client raises error without API key."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing key
            os.environ.pop("SUPERAGI_API_KEY", None)
            with pytest.raises(ValueError, match="SUPERAGI_API_KEY"):
                SuperAGIClient()

    def test_client_uses_env_vars(self):
        """Test that client reads from environment variables."""
        with patch.dict(os.environ, {
            "SUPERAGI_API_KEY": "test-key-123",
            "SUPERAGI_BASE_URL": "http://test:8100",
        }):
            client = SuperAGIClient()
            assert client.api_key == "test-key-123"
            assert client.base_url == "http://test:8100"


class TestGetAgentId:
    """Test get_agent_id function."""

    def test_returns_default_id(self):
        """Test that default ID is returned when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing env vars
            for config in AGENT_CONFIG.values():
                os.environ.pop(config["env_id"], None)
            
            agent_id = get_agent_id("orchestrator")
            assert agent_id == AGENT_CONFIG["orchestrator"]["default_id"]

    def test_returns_env_id(self):
        """Test that env var ID is preferred over default."""
        with patch.dict(os.environ, {
            "AWET_ORCHESTRATOR_AGENT_ID": "42",
        }):
            agent_id = get_agent_id("orchestrator")
            assert agent_id == 42

    def test_unknown_agent_raises(self):
        """Test that unknown agent raises ValueError."""
        with pytest.raises(ValueError, match="Unknown agent"):
            get_agent_id("nonexistent_agent")


class TestRunAgent:
    """Test run_agent function."""

    @patch("scripts.schedule_agents.SuperAGIClient")
    def test_run_agent_success(self, mock_client_class):
        """Test successful agent run."""
        # Set up mock
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_client.find_agent_by_name.return_value = {"id": 3, "name": "AWET Orchestrator"}
        mock_client.create_run.return_value = {"id": 123, "run_id": 123}

        with patch.dict(os.environ, {"SUPERAGI_API_KEY": "test-key"}):
            result = run_agent("orchestrator")

        assert result == 0
        mock_client.find_agent_by_name.assert_called_once_with("AWET Orchestrator")
        mock_client.create_run.assert_called_once()
        
        # Verify correct payload
        call_args = mock_client.create_run.call_args
        assert call_args.kwargs["agent_id"] == 3
        assert "Run the AWET trading pipeline end-to-end" in call_args.kwargs["goals"]

    @patch("scripts.schedule_agents.SuperAGIClient")
    def test_run_agent_uses_default_id_when_not_found(self, mock_client_class):
        """Test that default ID is used when agent not found by name."""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_client.find_agent_by_name.return_value = None  # Agent not found
        mock_client.create_run.return_value = {"id": 456}

        with patch.dict(os.environ, {"SUPERAGI_API_KEY": "test-key"}):
            result = run_agent("night_trainer")

        assert result == 0
        # Should use default ID from config
        call_args = mock_client.create_run.call_args
        assert call_args.kwargs["agent_id"] == AGENT_CONFIG["night_trainer"]["default_id"]

    @patch("scripts.schedule_agents.SuperAGIClient")
    def test_run_agent_handles_connection_error(self, mock_client_class):
        """Test that connection errors are handled gracefully."""
        import httpx
        
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_client.find_agent_by_name.side_effect = httpx.ConnectError("Connection refused")

        with patch.dict(os.environ, {"SUPERAGI_API_KEY": "test-key"}):
            result = run_agent("orchestrator")

        assert result == 1  # Should return error code

    def test_run_unknown_agent(self):
        """Test that unknown agent returns error."""
        with patch.dict(os.environ, {"SUPERAGI_API_KEY": "test-key"}):
            result = run_agent("nonexistent")
        assert result == 1


class TestConvenienceFunctions:
    """Test the convenience functions."""

    @patch("scripts.schedule_agents.run_agent")
    def test_run_night_trainer(self, mock_run):
        """Test run_night_trainer calls run_agent correctly."""
        mock_run.return_value = 0
        result = run_night_trainer()
        mock_run.assert_called_once_with("night_trainer")
        assert result == 0

    @patch("scripts.schedule_agents.run_agent")
    def test_run_morning_deployer(self, mock_run):
        """Test run_morning_deployer calls run_agent correctly."""
        mock_run.return_value = 0
        result = run_morning_deployer()
        mock_run.assert_called_once_with("morning_deployer")
        assert result == 0

    @patch("scripts.schedule_agents.run_agent")
    def test_run_trade_watchdog(self, mock_run):
        """Test run_trade_watchdog calls run_agent correctly."""
        mock_run.return_value = 0
        result = run_trade_watchdog()
        mock_run.assert_called_once_with("trade_watchdog")
        assert result == 0

    @patch("scripts.schedule_agents.run_agent")
    def test_run_orchestrator(self, mock_run):
        """Test run_orchestrator calls run_agent correctly."""
        mock_run.return_value = 0
        result = run_orchestrator()
        mock_run.assert_called_once_with("orchestrator")
        assert result == 0


class TestAgentConfig:
    """Test agent configuration."""

    def test_all_agents_have_required_fields(self):
        """Test that all agents have required configuration fields."""
        required_fields = ["name", "env_id", "default_id", "goals", "instructions"]
        
        for agent_key, config in AGENT_CONFIG.items():
            for field in required_fields:
                assert field in config, f"Agent '{agent_key}' missing field '{field}'"

    def test_agent_ids_are_unique(self):
        """Test that default agent IDs are unique."""
        ids = [config["default_id"] for config in AGENT_CONFIG.values()]
        assert len(ids) == len(set(ids)), "Duplicate default agent IDs found"
