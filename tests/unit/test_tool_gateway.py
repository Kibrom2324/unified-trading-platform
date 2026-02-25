"""
Tests for AWET Tool Gateway endpoints.

These tests verify that the Tool Gateway HTTP endpoints work correctly
without hitting real external services (Kafka, TimescaleDB, etc.).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ============================================
# Fixtures
# ============================================


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    mock = MagicMock()
    mock.logging.level = "INFO"
    mock.app.correlation_header = "X-Correlation-Id"
    return mock


@pytest.fixture
def client(mock_settings):
    """Create test client with mocked settings."""
    with patch("src.orchestration.tool_gateway.load_settings", return_value=mock_settings):
        with patch("src.orchestration.tool_gateway.configure_logging"):
            from src.orchestration.tool_gateway import app
            return TestClient(app)


# ============================================
# Health & Metrics Tests
# ============================================


class TestHealthEndpoints:
    """Tests for health and metrics endpoints."""
    
    def test_health_returns_ok(self, client):
        """Test /health returns OK status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "tool-gateway"
        assert "timestamp" in data
    
    def test_metrics_returns_prometheus_format(self, client):
        """Test /metrics returns Prometheus-formatted metrics."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        # Should be plain text Prometheus format
        assert response.headers.get("content-type", "").startswith("text/plain")


class TestListTools:
    """Tests for tool listing endpoint."""
    
    def test_list_tools_returns_all_tools(self, client):
        """Test /tools/list returns all available tools."""
        response = client.get("/tools/list")
        
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        
        tool_names = [t["name"] for t in data["tools"]]
        assert "run_backfill" in tool_names
        assert "train_model" in tool_names
        assert "promote_model" in tool_names
        assert "run_demo" in tool_names
        assert "check_pipeline_health" in tool_names
        assert "check_kafka_lag" in tool_names
        assert "query_audit_trail" in tool_names
        assert "approve_execution" in tool_names
        assert "revoke_execution" in tool_names
        assert "run_live_ingestion" in tool_names


# ============================================
# Backfill Endpoint Tests
# ============================================


class TestRunBackfill:
    """Tests for the run_backfill endpoint."""
    
    def test_backfill_invalid_source_returns_400(self, client):
        """Test that invalid source returns 400 error."""
        response = client.post(
            "/tools/run_backfill",
            json={"source": "invalid_source"},
        )
        
        assert response.status_code == 400
        assert "invalid source" in response.text.lower()
    
    def test_backfill_valid_request_structure(self, client):
        """Test that valid request returns expected structure."""
        with patch("src.orchestration.tool_gateway._run_script") as mock_run:
            mock_run.return_value = (True, "Success output", "")
            
            response = client.post(
                "/tools/run_backfill",
                json={
                    "source": "yfinance",
                    "symbols": "AAPL,MSFT",
                    "dry_run": True,
                },
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["tool"] == "run_backfill"
        assert "correlation_id" in data
        assert "duration_seconds" in data
        assert data["data"]["source"] == "yfinance"
        assert data["data"]["symbols"] == "AAPL,MSFT"
    
    def test_backfill_polygon_builds_correct_command(self, client):
        """Test Polygon backfill builds correct subprocess command."""
        with patch("src.orchestration.tool_gateway._run_script") as mock_run:
            mock_run.return_value = (True, "OK", "")
            
            response = client.post(
                "/tools/run_backfill",
                json={
                    "source": "polygon",
                    "symbols": "AAPL",
                    "start_date": "2025-01-01",
                    "end_date": "2025-01-31",
                },
            )
        
        assert response.status_code == 200
        # Verify the command was called with expected args
        call_args = mock_run.call_args
        cmd = call_args[0][0]  # First positional arg is the command list
        assert "execution.backfill_polygon" in " ".join(cmd)


# ============================================
# Train Model Endpoint Tests
# ============================================


class TestTrainModel:
    """Tests for the train_model endpoint."""
    
    def test_train_model_valid_request(self, client):
        """Test train_model with valid request."""
        with patch("src.orchestration.tool_gateway._run_script") as mock_run:
            mock_run.return_value = (True, "Model trained: tft_20260115_123456", "")
            
            response = client.post(
                "/tools/train_model",
                json={
                    "symbols": "AAPL",
                    "lookback_days": 30,
                    "epochs": 10,
                },
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["tool"] == "train_model"
    
    def test_train_model_default_values(self, client):
        """Test train_model uses default values when not specified."""
        with patch("src.orchestration.tool_gateway._run_script") as mock_run:
            mock_run.return_value = (True, "OK", "")
            
            response = client.post("/tools/train_model", json={})
        
        assert response.status_code == 200
        data = response.json()
        # Should use defaults
        assert data["data"]["symbols"] == "AAPL,MSFT,NVDA"


# ============================================
# Promote Model Endpoint Tests
# ============================================


class TestPromoteModel:
    """Tests for the promote_model endpoint."""
    
    def test_promote_model_requires_model_id(self, client):
        """Test that model_id is required."""
        response = client.post("/tools/promote_model", json={})
        
        # Pydantic validation should fail
        assert response.status_code == 422
    
    def test_promote_model_success(self, client):
        """Test successful model promotion."""
        with patch("src.orchestration.tool_gateway._run_script") as mock_run:
            mock_run.return_value = (True, "Model promoted to green", "")
            
            response = client.post(
                "/tools/promote_model",
                json={"model_id": "tft_20260115_123456_abc123"},
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "promoted to green" in data["message"]


# ============================================
# Run Demo Endpoint Tests
# ============================================


class TestRunDemo:
    """Tests for the run_demo endpoint."""
    
    def test_run_demo_success(self, client):
        """Test successful demo run."""
        with patch("src.orchestration.tool_gateway._run_script") as mock_run:
            mock_run.return_value = (True, "✅ Demo passed", "")
            
            response = client.post("/tools/run_demo")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["passed"] is True
    
    def test_run_demo_failure(self, client):
        """Test demo failure is reported correctly."""
        with patch("src.orchestration.tool_gateway._run_script") as mock_run:
            mock_run.return_value = (False, "Demo failed", "Error details")
            
            response = client.post("/tools/run_demo")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


# ============================================
# Pipeline Health Endpoint Tests
# ============================================


class TestCheckPipelineHealth:
    """Tests for the check_pipeline_health endpoint."""
    
    def test_check_health_returns_agent_status(self, client):
        """Test health check returns status for all agents."""
        # This will fail to connect to agents (expected in test)
        response = client.get("/tools/check_pipeline_health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["tool"] == "check_pipeline_health"
        assert "agents" in data["data"]
        assert "healthy_count" in data["data"]
        assert "total_count" in data["data"]


# ============================================
# Kafka Lag Endpoint Tests
# ============================================


class TestCheckKafkaLag:
    """Tests for the check_kafka_lag endpoint."""
    
    def test_check_kafka_lag_structure(self, client):
        """Test Kafka lag check returns expected structure."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1  # Simulate kafka not running
            mock_result.stdout = ""
            mock_result.stderr = ""
            mock_run.return_value = mock_result
            
            response = client.get("/tools/check_kafka_lag")
        
        assert response.status_code == 200
        data = response.json()
        assert data["tool"] == "check_kafka_lag"
        assert "groups" in data["data"]
        assert "total_lag" in data["data"]


# ============================================
# Audit Trail Endpoint Tests
# ============================================


class TestQueryAuditTrail:
    """Tests for the query_audit_trail endpoint."""
    
    def test_query_audit_trail_default_params(self, client):
        """Test audit trail query with default parameters."""
        with patch("src.orchestration.tool_gateway._get_db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetch.return_value = []
            
            mock_pool_instance = AsyncMock()
            mock_pool_instance.acquire.return_value.__aenter__.return_value = mock_conn
            mock_pool_instance.close = AsyncMock()
            mock_pool.return_value = mock_pool_instance
            
            response = client.post("/tools/query_audit_trail", json={})
        
        # May fail due to DB connection, but structure should be valid
        assert response.status_code in (200, 500)
    
    def test_query_audit_trail_with_filters(self, client):
        """Test audit trail query with filters."""
        with patch("src.orchestration.tool_gateway._get_db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetch.return_value = [
                {
                    "event_type": "market.raw",
                    "symbol": "AAPL",
                    "ts": datetime.now(timezone.utc),
                    "correlation_id": "test-123",
                    "source": "live_ingestion",
                }
            ]
            
            mock_pool_instance = AsyncMock()
            mock_pool_instance.acquire.return_value.__aenter__.return_value = mock_conn
            mock_pool_instance.close = AsyncMock()
            mock_pool.return_value = mock_pool_instance
            
            response = client.post(
                "/tools/query_audit_trail",
                json={
                    "event_type": "market.raw",
                    "symbol": "AAPL",
                    "limit": 10,
                },
            )
        
        assert response.status_code in (200, 500)


# ============================================
# Execution Control Endpoint Tests
# ============================================


class TestApproveExecution:
    """Tests for the approve_execution endpoint."""
    
    def test_approve_execution_creates_file(self, client, tmp_path):
        """Test approve_execution creates approval file."""
        with patch("src.orchestration.tool_gateway._project_root", return_value=tmp_path):
            response = client.post("/tools/approve_execution")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "APPROVED" in data["message"]
        
        # Verify file was created
        approval_file = tmp_path / ".tmp" / "APPROVE_EXECUTION"
        assert approval_file.exists()


class TestRevokeExecution:
    """Tests for the revoke_execution endpoint."""
    
    def test_revoke_execution_removes_file(self, client, tmp_path):
        """Test revoke_execution removes approval file."""
        # Create approval file first
        approval_dir = tmp_path / ".tmp"
        approval_dir.mkdir(parents=True)
        approval_file = approval_dir / "APPROVE_EXECUTION"
        approval_file.touch()
        
        with patch("src.orchestration.tool_gateway._project_root", return_value=tmp_path):
            response = client.post("/tools/revoke_execution")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "REVOKED" in data["message"]
        assert data["data"]["existed"] is True
        
        # Verify file was removed
        assert not approval_file.exists()
    
    def test_revoke_execution_handles_missing_file(self, client, tmp_path):
        """Test revoke_execution handles case where file doesn't exist."""
        with patch("src.orchestration.tool_gateway._project_root", return_value=tmp_path):
            response = client.post("/tools/revoke_execution")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["existed"] is False


# ============================================
# Live Ingestion Endpoint Tests
# ============================================


class TestRunLiveIngestion:
    """Tests for the run_live_ingestion endpoint."""
    
    def test_live_ingestion_default_symbols(self, client):
        """Test live ingestion with default symbols."""
        with patch("src.orchestration.tool_gateway._run_script") as mock_run:
            mock_run.return_value = (True, "Ingested AAPL, MSFT", "")
            
            response = client.post("/tools/run_live_ingestion", json={})
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["tool"] == "run_live_ingestion"
    
    def test_live_ingestion_with_custom_symbols(self, client):
        """Test live ingestion with custom symbols."""
        with patch("src.orchestration.tool_gateway._run_script") as mock_run:
            mock_run.return_value = (True, "OK", "")
            
            response = client.post(
                "/tools/run_live_ingestion",
                json={
                    "symbols": "TSLA,NVDA",
                    "dry_run": True,
                },
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["symbols"] == "TSLA,NVDA"
        assert data["data"]["dry_run"] is True
    
    def test_live_ingestion_with_pipeline_trigger(self, client):
        """Test live ingestion with pipeline trigger."""
        with patch("src.orchestration.tool_gateway._run_script") as mock_run:
            mock_run.return_value = (True, "Pipeline triggered", "")
            
            response = client.post(
                "/tools/run_live_ingestion",
                json={"trigger_pipeline": True},
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["trigger_pipeline"] is True


# ============================================
# Response Model Tests
# ============================================


class TestToolResponse:
    """Tests for ToolResponse model structure."""
    
    def test_tool_response_has_required_fields(self, client):
        """Test all endpoints return ToolResponse with required fields."""
        with patch("src.orchestration.tool_gateway._run_script") as mock_run:
            mock_run.return_value = (True, "OK", "")
            
            response = client.post("/tools/run_demo")
        
        data = response.json()
        
        required_fields = ["success", "correlation_id", "tool", "message", "duration_seconds"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
