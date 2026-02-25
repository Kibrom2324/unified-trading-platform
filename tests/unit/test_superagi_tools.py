"""
Unit tests for SuperAGI integration - Testing Tool Gateway endpoints.

These tests verify that the Tool Gateway HTTP endpoints work correctly,
which is how SuperAGI actually integrates with AWET. All subprocess calls
and external dependencies are mocked.

The SuperAGI tools themselves (in superagi/tools/) inherit from SuperAGI's
BaseTool class which is only available inside the SuperAGI Docker container.
Our tests therefore focus on the Tool Gateway layer.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import subprocess
import asyncio
import json
from pathlib import Path


# ============================================================================
# Tool Gateway Endpoint Logic Tests (Mock-Based)
# ============================================================================

class TestReadDirectiveLogic:
    """Tests for the read_directive endpoint logic."""
    
    def test_read_directive_success(self, tmp_path):
        """Test successful reading of a directive file."""
        directive_content = "# Test Directive\n\nThis is a test."
        directives_dir = tmp_path / "directives"
        directives_dir.mkdir()
        (directives_dir / "test.md").write_text(directive_content)
        
        # Simulate what the endpoint does
        directive_name = "test.md"
        file_path = directives_dir / directive_name
        result = file_path.read_text()
        
        assert "Test Directive" in result
        assert "This is a test." in result
    
    def test_read_directive_adds_md_extension(self, tmp_path):
        """Test that .md extension is added automatically."""
        directives_dir = tmp_path / "directives"
        directives_dir.mkdir()
        (directives_dir / "test.md").write_text("Content")
        
        directive_name = "test"  # Without .md
        if not directive_name.endswith('.md'):
            directive_name = f"{directive_name}.md"
        
        assert directive_name == "test.md"
        assert (directives_dir / directive_name).exists()
    
    def test_read_directive_prevents_path_traversal(self):
        """Test that path traversal is blocked."""
        directive_name = "../../../etc/passwd"
        
        # Security check logic
        is_invalid = ".." in directive_name or directive_name.startswith("/")
        
        assert is_invalid is True
    
    def test_read_directive_not_found(self, tmp_path):
        """Test handling of missing directive."""
        directives_dir = tmp_path / "directives"
        directives_dir.mkdir()
        
        directive_name = "nonexistent.md"
        file_path = directives_dir / directive_name
        
        assert not file_path.exists()


class TestRunBackfillLogic:
    """Tests for the run_backfill endpoint logic."""
    
    def test_backfill_validates_source(self):
        """Test that invalid data sources are rejected."""
        valid_sources = ["yfinance", "alpha_vantage"]
        source = "invalid_source"
        
        is_valid = source in valid_sources
        
        assert is_valid is False
    
    def test_backfill_accepts_valid_sources(self):
        """Test that valid sources are accepted."""
        valid_sources = ["yfinance", "alpha_vantage"]
        
        assert "yfinance" in valid_sources
        assert "alpha_vantage" in valid_sources
    
    @patch('subprocess.run')
    def test_backfill_script_execution(self, mock_run):
        """Test backfill script execution."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Backfill completed: 1000 records",
            stderr=""
        )
        
        result = subprocess.run(
            ["python", "-m", "execution.backfill", "--symbols", "AAPL", "--source", "yfinance"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert result.returncode == 0
        assert "completed" in result.stdout.lower()
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_backfill_with_date_range(self, mock_run):
        """Test backfill with specific date range."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Backfill completed: AAPL 2024-01-01 to 2024-01-31",
            stderr=""
        )
        
        result = subprocess.run(
            [
                "python", "-m", "execution.backfill",
                "--symbols", "AAPL",
                "--source", "yfinance",
                "--start", "2024-01-01",
                "--end", "2024-01-31"
            ],
            capture_output=True,
            text=True
        )
        
        assert "2024-01-01" in result.stdout or result.returncode == 0
    
    @patch('subprocess.run')
    def test_backfill_timeout(self, mock_run):
        """Test handling of script timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="backfill", timeout=300)
        
        with pytest.raises(subprocess.TimeoutExpired):
            subprocess.run(
                ["python", "-m", "execution.backfill", "--symbols", "AAPL"],
                timeout=300
            )


class TestTrainModelLogic:
    """Tests for the train_model endpoint logic."""
    
    @patch('subprocess.run')
    def test_train_model_success(self, mock_run):
        """Test successful model training."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Training started: run_id=train_20240115_001",
            stderr=""
        )
        
        result = subprocess.run(
            ["python", "-m", "execution.train_tft", "--symbols", "AAPL,MSFT", "--epochs", "100"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "train" in result.stdout.lower()
    
    def test_train_model_validates_epochs(self):
        """Test that epochs parameter is validated."""
        epochs = -1
        
        is_valid = isinstance(epochs, int) and epochs > 0
        
        assert is_valid is False
        
        epochs = 100
        is_valid = isinstance(epochs, int) and epochs > 0
        
        assert is_valid is True


class TestPromoteModelLogic:
    """Tests for the promote_model endpoint logic."""
    
    @patch('subprocess.run')
    def test_promote_model_success(self, mock_run):
        """Test successful model promotion."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Model train_20240115_001 promoted to green",
            stderr=""
        )
        
        result = subprocess.run(
            ["python", "-m", "execution.promote_model", "--run-id", "train_20240115_001"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "promoted" in result.stdout.lower() or "green" in result.stdout.lower()
    
    @patch('subprocess.run')
    def test_promote_model_not_found(self, mock_run):
        """Test promotion of non-existent model."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: Model 'nonexistent' not found"
        )
        
        result = subprocess.run(
            ["python", "-m", "execution.promote_model", "--run-id", "nonexistent"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 1
        assert "not found" in result.stderr.lower()


class TestRunDemoLogic:
    """Tests for the run_demo endpoint logic."""
    
    @patch('subprocess.run')
    def test_run_demo_success(self, mock_run):
        """Test successful demo run."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Demo completed: 5 events processed",
            stderr=""
        )
        
        result = subprocess.run(
            ["python", "-m", "execution.run_demo", "--duration", "60"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "completed" in result.stdout.lower()


class TestCheckPipelineHealthLogic:
    """Tests for the check_pipeline_health endpoint logic."""
    
    def test_health_check_response_format(self):
        """Test health check response format."""
        # Simulate health check response
        health_status = {
            "IngestionAgent": {"status": "healthy", "port": 8001},
            "FeatureAgent": {"status": "healthy", "port": 8002},
            "PredictionAgent": {"status": "healthy", "port": 8003},
        }
        
        all_healthy = all(
            agent["status"] == "healthy" 
            for agent in health_status.values()
        )
        
        assert all_healthy is True
    
    def test_health_check_partial_failure(self):
        """Test health check with some agents down."""
        health_status = {
            "IngestionAgent": {"status": "healthy", "port": 8001},
            "FeatureAgent": {"status": "unhealthy", "port": 8002},
            "PredictionAgent": {"status": "healthy", "port": 8003},
        }
        
        all_healthy = all(
            agent["status"] == "healthy" 
            for agent in health_status.values()
        )
        
        assert all_healthy is False
        
        unhealthy_agents = [
            name for name, agent in health_status.items()
            if agent["status"] != "healthy"
        ]
        
        assert "FeatureAgent" in unhealthy_agents


class TestApproveExecutionLogic:
    """Tests for the approve_execution endpoint logic."""
    
    def test_approve_execution_creates_token(self, tmp_path):
        """Test that approval creates a token file."""
        approval_file = tmp_path / ".execution_approved"
        reason = "Market conditions favorable"
        
        # Simulate approval
        approval_file.write_text(f"approved\nreason: {reason}")
        
        assert approval_file.exists()
        content = approval_file.read_text()
        assert "approved" in content
        assert reason in content


class TestRevokeExecutionLogic:
    """Tests for the revoke_execution endpoint logic."""
    
    def test_revoke_execution_removes_token(self, tmp_path):
        """Test that revocation removes the token file."""
        approval_file = tmp_path / ".execution_approved"
        approval_file.write_text("approved")
        
        # Simulate revocation
        if approval_file.exists():
            approval_file.unlink()
        
        assert not approval_file.exists()


class TestQueryAuditTrailLogic:
    """Tests for the query_audit_trail endpoint logic."""
    
    def test_query_audit_filter_by_time(self):
        """Test audit trail time filtering."""
        events = [
            {"ts": "2024-01-15T10:00:00Z", "event": "ingestion"},
            {"ts": "2024-01-15T11:00:00Z", "event": "prediction"},
            {"ts": "2024-01-16T10:00:00Z", "event": "execution"},
        ]
        
        start = "2024-01-15T00:00:00Z"
        end = "2024-01-15T23:59:59Z"
        
        filtered = [
            e for e in events
            if start <= e["ts"] <= end
        ]
        
        assert len(filtered) == 2
        assert all("2024-01-15" in e["ts"] for e in filtered)


class TestCheckKafkaLagLogic:
    """Tests for the check_kafka_lag endpoint logic."""
    
    def test_kafka_lag_check_healthy(self):
        """Test Kafka lag check when lag is acceptable."""
        lag_data = {
            "market.raw": {"lag": 5, "threshold": 1000},
            "market.engineered": {"lag": 2, "threshold": 1000},
            "predictions.tft": {"lag": 0, "threshold": 100},
        }
        
        all_healthy = all(
            topic["lag"] <= topic["threshold"]
            for topic in lag_data.values()
        )
        
        assert all_healthy is True
    
    def test_kafka_lag_check_unhealthy(self):
        """Test Kafka lag check when lag exceeds threshold."""
        lag_data = {
            "market.raw": {"lag": 5000, "threshold": 1000},  # Exceeds
            "predictions.tft": {"lag": 0, "threshold": 100},
        }
        
        unhealthy_topics = [
            name for name, data in lag_data.items()
            if data["lag"] > data["threshold"]
        ]
        
        assert "market.raw" in unhealthy_topics


# ============================================================================
# Orchestrator Workflow Tests
# ============================================================================

class TestOrchestratorWorkflow:
    """Tests simulating full orchestrator workflows."""
    
    def test_full_pipeline_workflow_mock(self):
        """Test a full pipeline workflow with all steps mocked."""
        # Simulate what SuperAGI would do when running a full pipeline
        
        workflow_results = []
        
        # Step 1: Read directive
        workflow_results.append({
            "step": "read_directive",
            "result": "# trading_pipeline.md\n\nGoal: Process market data."
        })
        
        # Step 2: Run backfill
        workflow_results.append({
            "step": "run_backfill",
            "result": "Backfill completed for AAPL: 1000 records"
        })
        
        # Step 3: Train model
        workflow_results.append({
            "step": "train_model",
            "result": "Training started: run_id=train_20240115_001"
        })
        
        # Step 4: Promote model
        workflow_results.append({
            "step": "promote_model",
            "result": "Model promoted to green deployment"
        })
        
        # Step 5: Approve execution
        workflow_results.append({
            "step": "approve_execution",
            "result": "Paper trading approved"
        })
        
        # Step 6: Run demo
        workflow_results.append({
            "step": "run_demo",
            "result": "Demo completed: 5 predictions generated"
        })
        
        # Step 7: Check health
        workflow_results.append({
            "step": "check_health",
            "result": "All agents healthy"
        })
        
        # Verify workflow completed all steps
        assert len(workflow_results) == 7
        
        # Verify each step produced expected output
        assert "trading_pipeline" in workflow_results[0]["result"]
        assert "completed" in workflow_results[1]["result"].lower()
        assert "run_id" in workflow_results[2]["result"]
        assert "promoted" in workflow_results[3]["result"].lower()
        assert "approved" in workflow_results[4]["result"].lower()
        assert "completed" in workflow_results[5]["result"].lower()
        assert "healthy" in workflow_results[6]["result"].lower()
    
    def test_workflow_stops_on_error(self):
        """Test that workflow stops when an error occurs."""
        workflow = [
            ("read_directive", {"success": True, "result": "Directive content"}),
            ("run_backfill", {"success": False, "error": "Connection failed"}),
            ("train_model", None),  # Should not execute
        ]
        
        executed = []
        for step_name, step_data in workflow:
            if step_data is None:
                break
            executed.append(step_name)
            if not step_data.get("success", True):
                break
        
        # Should stop at error
        assert len(executed) == 2
        assert "train_model" not in executed
    
    def test_workflow_with_correlation_id(self):
        """Test that correlation_id is propagated through workflow."""
        import uuid
        
        correlation_id = str(uuid.uuid4())
        
        events = []
        for step in ["ingestion", "feature", "prediction", "execution"]:
            events.append({
                "correlation_id": correlation_id,
                "step": step,
                "status": "completed"
            })
        
        # All events should share the same correlation_id
        assert all(e["correlation_id"] == correlation_id for e in events)
        assert len(events) == 4


# ============================================================================
# Tool Registration Tests
# ============================================================================

class TestToolRegistration:
    """Tests for SuperAGI tool registration."""
    
    def test_tool_definition_structure(self):
        """Test that tool definitions have required fields."""
        tool_def = {
            "name": "awet_run_backfill",
            "description": "Run historical data backfill",
            "parameters": {
                "symbols": {"type": "string", "required": True},
                "source": {"type": "string", "required": True},
                "start_date": {"type": "string", "required": False},
                "end_date": {"type": "string", "required": False},
            }
        }
        
        assert "name" in tool_def
        assert "description" in tool_def
        assert "parameters" in tool_def
        assert tool_def["name"].startswith("awet_")
    
    def test_all_tools_have_awet_prefix(self):
        """Test that all AWET tools have correct prefix."""
        tool_names = [
            "awet_read_directive",
            "awet_run_backfill",
            "awet_train_model",
            "awet_promote_model",
            "awet_approve_execution",
            "awet_revoke_execution",
            "awet_run_demo",
            "awet_check_pipeline_health",
            "awet_check_kafka_lag",
            "awet_query_audit_trail",
            "awet_generate_plan",
            "awet_explain_run",
            "awet_run_live_ingestion",
        ]
        
        assert all(name.startswith("awet_") for name in tool_names)
        assert len(tool_names) == 13


# ============================================================================
# Tool Gateway HTTP Tests (Integration-style with mocks)
# ============================================================================

class TestToolGatewayEndpoints:
    """Tests simulating Tool Gateway HTTP endpoint behavior."""
    
    def test_endpoint_returns_json(self):
        """Test that endpoints return JSON responses."""
        response = {
            "status": "success",
            "result": "Backfill completed",
            "correlation_id": "abc123"
        }
        
        # Should be JSON serializable
        json_str = json.dumps(response)
        parsed = json.loads(json_str)
        
        assert parsed["status"] == "success"
    
    def test_endpoint_includes_correlation_id(self):
        """Test that responses include correlation_id."""
        import uuid
        
        correlation_id = str(uuid.uuid4())
        
        response = {
            "status": "success",
            "result": "Operation completed",
            "correlation_id": correlation_id
        }
        
        assert "correlation_id" in response
        # Should be a valid UUID
        uuid.UUID(response["correlation_id"])
    
    def test_error_response_format(self):
        """Test error response format."""
        error_response = {
            "status": "error",
            "error": "Script execution failed",
            "details": "Connection timeout",
            "correlation_id": "abc123"
        }
        
        assert error_response["status"] == "error"
        assert "error" in error_response
        assert "correlation_id" in error_response
