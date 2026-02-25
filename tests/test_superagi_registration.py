import os

import responses

from tools.register_awet_agent import register_awet_agent


def _set_env(monkeypatch):
    monkeypatch.setenv("SUPERAGI_BASE_URL", "http://localhost:3001/api")
    monkeypatch.setenv("SUPERAGI_API_KEY", "test-key")
    monkeypatch.setenv("TOOL_GATEWAY_URL", "http://host.docker.internal:8200")


def _tool_list():
    return [
        {"id": 1, "name": "awet_read_directive"},
        {"id": 2, "name": "awet_run_backfill"},
        {"id": 3, "name": "awet_train_model"},
        {"id": 4, "name": "awet_promote_model"},
        {"id": 5, "name": "awet_run_demo"},
        {"id": 6, "name": "awet_check_pipeline_health"},
    ]


@responses.activate
def test_creates_agent_when_missing(monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.setattr(
        "tools.register_awet_agent.register_all_tools",
        lambda **kwargs: (6, 0, {t["name"]: t for t in _tool_list()}),
    )

    responses.add(
        responses.GET,
        "http://localhost:3001/api/agents/get/project/1",
        json=[],
        status=200,
    )
    responses.add(
        responses.POST,
        "http://localhost:3001/api/agents/create",
        json={"id": 123},
        status=201,
    )

    result = register_awet_agent()
    assert result["agent_id"] == 123
    assert result["status"] == "created"


@responses.activate
def test_updates_agent_when_exists(monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.setattr(
        "tools.register_awet_agent.register_all_tools",
        lambda **kwargs: (6, 0, {t["name"]: t for t in _tool_list()}),
    )

    responses.add(
        responses.GET,
        "http://localhost:3001/api/agents/get/project/1",
        json=[{"id": 99, "name": "AWET Orchestrator"}],
        status=200,
    )
    responses.add(
        responses.PUT,
        "http://localhost:3001/api/v1/agent/99",
        json={"id": 99},
        status=200,
    )

    result = register_awet_agent()
    assert result["agent_id"] == 99
    assert result["status"] == "updated"


def test_fails_without_api_key(monkeypatch):
    monkeypatch.delenv("SUPERAGI_BASE_URL", raising=False)
    monkeypatch.delenv("SUPERAGI_API_KEY", raising=False)
    monkeypatch.delenv("TOOL_GATEWAY_URL", raising=False)

    try:
        register_awet_agent()
    except SystemExit as exc:
        assert "SUPERAGI_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected SystemExit when API key missing")
