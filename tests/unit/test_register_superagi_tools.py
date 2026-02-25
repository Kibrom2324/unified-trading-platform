"""
Tests for SuperAGI tool registration script.

These tests verify the registration logic without actually
connecting to SuperAGI.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestToolDefinitions:
    """Tests for tool definitions."""
    
    def test_all_tools_have_required_fields(self):
        """Test all tool definitions have required fields."""
        from execution.register_superagi_tools import AWET_TOOLS
        
        required_fields = ["name", "display_name", "description", "endpoint", "method", "parameters"]
        
        for tool in AWET_TOOLS:
            for field in required_fields:
                assert hasattr(tool, field), f"Tool {tool.name} missing field: {field}"
    
    def test_tool_names_follow_convention(self):
        """Test tool names follow awet_ prefix convention."""
        from execution.register_superagi_tools import AWET_TOOLS
        
        for tool in AWET_TOOLS:
            assert tool.name.startswith("awet_"), f"Tool {tool.name} should start with 'awet_'"
    
    def test_endpoints_are_valid(self):
        """Test all endpoints start with /tools/."""
        from execution.register_superagi_tools import AWET_TOOLS
        
        for tool in AWET_TOOLS:
            assert tool.endpoint.startswith("/tools/"), f"Tool {tool.name} endpoint should start with '/tools/'"
    
    def test_methods_are_valid(self):
        """Test all methods are valid HTTP methods."""
        from execution.register_superagi_tools import AWET_TOOLS
        
        valid_methods = {"GET", "POST", "PUT", "DELETE", "PATCH"}
        
        for tool in AWET_TOOLS:
            assert tool.method in valid_methods, f"Tool {tool.name} has invalid method: {tool.method}"
    
    def test_tool_count(self):
        """Test expected number of tools are defined."""
        from execution.register_superagi_tools import AWET_TOOLS
        
        # We should have at least 10 tools
        assert len(AWET_TOOLS) >= 10, f"Expected at least 10 tools, got {len(AWET_TOOLS)}"


class TestSuperAGIClient:
    """Tests for SuperAGI client."""
    
    def test_client_initialization(self):
        """Test client can be initialized."""
        from execution.register_superagi_tools import SuperAGIClient
        
        client = SuperAGIClient(
            api_url="http://localhost:8100/api",
            tool_gateway_url="http://localhost:8200",
        )
        
        assert client.api_url == "http://localhost:8100/api"
        assert client.tool_gateway_url == "http://localhost:8200"
        client.close()
    
    def test_health_check_fails_gracefully(self):
        """Test health check returns False when SuperAGI is not running."""
        from execution.register_superagi_tools import SuperAGIClient
        
        client = SuperAGIClient(
            api_url="http://localhost:99999/api",  # Invalid port
            tool_gateway_url="http://localhost:8200",
        )
        
        # Should return False, not raise exception
        result = client.health_check()
        assert result is False
        client.close()


class TestCheckToolGateway:
    """Tests for tool gateway health check."""
    
    def test_check_tool_gateway_fails_gracefully(self):
        """Test gateway check returns False when not running."""
        from execution.register_superagi_tools import check_tool_gateway
        
        result = check_tool_gateway("http://localhost:99999")
        assert result is False
    
    def test_check_tool_gateway_success(self):
        """Test gateway check with mock success."""
        from execution.register_superagi_tools import check_tool_gateway
        
        with patch("execution.register_superagi_tools.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            result = check_tool_gateway("http://localhost:8200")
            assert result is True


class TestRegisterAllTools:
    """Tests for registration function."""
    
    def test_register_fails_when_superagi_not_running(self):
        """Test registration fails gracefully when SuperAGI is down."""
        from execution.register_superagi_tools import register_all_tools
        
        success, failed = register_all_tools(
            superagi_url="http://localhost:99999/api",
            tool_gateway_url="http://localhost:8200",
        )
        
        # All tools should fail
        assert success == 0
        assert failed >= 10
