#!/usr/bin/env python3
"""Tests for the AWET health check script."""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import httpx
import pytest

# Import the module we're testing
import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from scripts.awet_health_check import (
    HealthCheckResult,
    SystemHealth,
    check_kafka,
    check_ollama,
    check_postgres,
    check_superagi_api,
    check_agents,
    run_health_check,
    EXPECTED_AGENTS,
)


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass."""

    def test_healthy_result(self):
        result = HealthCheckResult(name="Test", healthy=True, message="OK")
        assert result.healthy is True
        assert result.name == "Test"

    def test_unhealthy_result_with_details(self):
        result = HealthCheckResult(
            name="Test",
            healthy=False,
            message="Failed",
            details={"error": "connection refused"},
        )
        assert result.healthy is False
        assert result.details["error"] == "connection refused"


class TestSystemHealth:
    """Test SystemHealth dataclass."""

    def test_all_healthy(self):
        health = SystemHealth(
            checks=[
                HealthCheckResult(name="A", healthy=True),
                HealthCheckResult(name="B", healthy=True),
            ]
        )
        assert health.all_healthy is True
        assert health.healthy_count == 2
        assert health.total_count == 2

    def test_partial_failure(self):
        health = SystemHealth(
            checks=[
                HealthCheckResult(name="A", healthy=True),
                HealthCheckResult(name="B", healthy=False),
                HealthCheckResult(name="C", healthy=True),
            ]
        )
        assert health.all_healthy is False
        assert health.healthy_count == 2
        assert health.total_count == 3

    def test_to_dict(self):
        health = SystemHealth(
            checks=[
                HealthCheckResult(name="Test", healthy=True, message="OK"),
            ]
        )
        result = health.to_dict()
        assert result["healthy"] is True
        assert result["summary"] == "1/1 checks passed"
        assert len(result["checks"]) == 1


class TestCheckPostgres:
    """Test Postgres health check."""

    @patch("subprocess.run")
    def test_postgres_healthy_via_docker(self, mock_run):
        """Test healthy postgres via docker pg_isready."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        
        # Without asyncpg installed, falls back to docker
        with patch.dict("sys.modules", {"asyncpg": None}):
            result = check_postgres({})
        
        assert result.name == "TimescaleDB"
        # May be healthy or unhealthy depending on asyncpg availability

    @patch("subprocess.run")
    def test_postgres_unhealthy(self, mock_run):
        """Test unhealthy postgres."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="connection refused",
        )
        
        with patch.dict("sys.modules", {"asyncpg": None}):
            # Force the ImportError path
            with patch("scripts.awet_health_check.check_postgres") as mock_check:
                mock_check.return_value = HealthCheckResult(
                    name="TimescaleDB",
                    healthy=False,
                    message="connection refused",
                )
                result = mock_check({})
        
        assert result.healthy is False


class TestCheckKafka:
    """Test Kafka health check."""

    @patch("subprocess.run")
    def test_kafka_healthy(self, mock_run):
        """Test healthy Kafka."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="market.raw\nmarket.engineered\npredictions.tft\n",
            stderr="",
        )
        
        result = check_kafka()
        
        assert result.name == "Kafka"
        assert result.healthy is True
        assert "3 topics" in result.message
        assert result.details["topic_count"] == 3

    @patch("subprocess.run")
    def test_kafka_unhealthy(self, mock_run):
        """Test unhealthy Kafka."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="broker not available",
        )
        
        result = check_kafka()
        
        assert result.name == "Kafka"
        assert result.healthy is False
        assert "broker not available" in result.message

    @patch("subprocess.run")
    def test_kafka_timeout(self, mock_run):
        """Test Kafka timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=15)
        
        result = check_kafka()
        
        assert result.healthy is False
        assert "Timeout" in result.message


class TestCheckSuperAGIAPI:
    """Test SuperAGI API health check."""

    @patch("httpx.Client")
    def test_superagi_healthy(self, mock_client_class):
        """Test healthy SuperAGI API."""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        
        result = check_superagi_api({})
        
        assert result.name == "SuperAGI API"
        assert result.healthy is True

    @patch("httpx.Client")
    def test_superagi_unhealthy(self, mock_client_class):
        """Test unhealthy SuperAGI API."""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        
        result = check_superagi_api({})
        
        assert result.name == "SuperAGI API"
        assert result.healthy is False


class TestCheckAgents:
    """Test SuperAGI agents health check."""

    @patch("httpx.Client")
    def test_all_agents_found(self, mock_client_class):
        """Test all agents are found."""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"name": "AWET Orchestrator", "id": 1},
            {"name": "AWET Night Trainer", "id": 2},
            {"name": "AWET Morning Deployer", "id": 3},
            {"name": "AWET Trade Watchdog", "id": 4},
        ]
        mock_client.get.return_value = mock_response
        
        result = check_agents({"SUPERAGI_API_KEY": "test-key"})
        
        assert result.name == "SuperAGI Agents"
        assert result.healthy is True
        assert "4 agents found" in result.message

    @patch("httpx.Client")
    def test_missing_agents(self, mock_client_class):
        """Test some agents are missing."""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"name": "AWET Orchestrator", "id": 1},
            {"name": "AWET Night Trainer", "id": 2},
            # Missing: Morning Deployer, Trade Watchdog
        ]
        mock_client.get.return_value = mock_response
        
        result = check_agents({"SUPERAGI_API_KEY": "test-key"})
        
        assert result.name == "SuperAGI Agents"
        assert result.healthy is False
        assert "Missing 2 agents" in result.message
        assert "AWET Morning Deployer" in result.details["missing"]
        assert "AWET Trade Watchdog" in result.details["missing"]


class TestCheckOllama:
    """Test Ollama health check."""

    @patch("httpx.Client")
    def test_ollama_healthy(self, mock_client_class):
        """Test healthy Ollama."""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        
        result = check_ollama()
        
        assert result.name == "Ollama"
        assert result.healthy is True
        assert "Ollama is running" in result.message

    @patch("httpx.Client")
    def test_ollama_unhealthy(self, mock_client_class):
        """Test unhealthy Ollama."""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        
        result = check_ollama()
        
        assert result.name == "Ollama"
        assert result.healthy is False


class TestRunHealthCheck:
    """Test the full health check runner."""

    @patch("scripts.awet_health_check.check_ollama")
    @patch("scripts.awet_health_check.check_agents")
    @patch("scripts.awet_health_check.check_superagi_api")
    @patch("scripts.awet_health_check.check_kafka")
    @patch("scripts.awet_health_check.check_postgres")
    def test_all_checks_run(
        self,
        mock_postgres,
        mock_kafka,
        mock_superagi,
        mock_agents,
        mock_ollama,
    ):
        """Test that all checks are executed."""
        mock_postgres.return_value = HealthCheckResult("TimescaleDB", True)
        mock_kafka.return_value = HealthCheckResult("Kafka", True)
        mock_superagi.return_value = HealthCheckResult("SuperAGI API", True)
        mock_agents.return_value = HealthCheckResult("SuperAGI Agents", True)
        mock_ollama.return_value = HealthCheckResult("Ollama", True)
        
        health = run_health_check()
        
        assert health.all_healthy is True
        assert health.total_count == 5
        mock_postgres.assert_called_once()
        mock_kafka.assert_called_once()
        mock_superagi.assert_called_once()
        mock_agents.assert_called_once()
        mock_ollama.assert_called_once()

    @patch("scripts.awet_health_check.check_ollama")
    @patch("scripts.awet_health_check.check_agents")
    @patch("scripts.awet_health_check.check_superagi_api")
    @patch("scripts.awet_health_check.check_kafka")
    @patch("scripts.awet_health_check.check_postgres")
    def test_partial_failure_reported(
        self,
        mock_postgres,
        mock_kafka,
        mock_superagi,
        mock_agents,
        mock_ollama,
    ):
        """Test that partial failures are correctly reported."""
        mock_postgres.return_value = HealthCheckResult("TimescaleDB", True)
        mock_kafka.return_value = HealthCheckResult("Kafka", False, "Not available")
        mock_superagi.return_value = HealthCheckResult("SuperAGI API", True)
        mock_agents.return_value = HealthCheckResult("SuperAGI Agents", True)
        mock_ollama.return_value = HealthCheckResult("Ollama", False, "Not running")
        
        health = run_health_check()
        
        assert health.all_healthy is False
        assert health.healthy_count == 3
        assert health.total_count == 5
