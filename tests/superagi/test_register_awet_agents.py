import json
import os

import responses

from superagi.register_awet_agents import main as register_all
from tools.register_awet_agent import AWET_AGENT_TEAM


def _tool_records():
    return {
        "awet_read_directive": {"id": 1, "name": "awet_read_directive"},
        "awet_run_backfill": {"id": 2, "name": "awet_run_backfill"},
        "awet_train_model": {"id": 3, "name": "awet_train_model"},
        "awet_promote_model": {"id": 4, "name": "awet_promote_model"},
        "awet_run_demo": {"id": 5, "name": "awet_run_demo"},
        "awet_check_pipeline_health": {"id": 6, "name": "awet_check_pipeline_health"},
    }


def _set_env(monkeypatch):
    monkeypatch.setenv("SUPERAGI_BASE_URL", "http://localhost:3001/api")
    monkeypatch.setenv("SUPERAGI_API_KEY", "test-key")
    monkeypatch.setenv("TOOL_GATEWAY_URL", "http://host.docker.internal:8200")


@responses.activate
def test_registers_agent_team(monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.setattr(
        "tools.register_awet_agent.register_all_tools",
        lambda **kwargs: (6, 0, _tool_records()),
    )

    responses.add(
        responses.GET,
        "http://localhost:3001/api/agents/get/project/1",
        json=[],
        status=200,
    )

    created_payloads = []

    def _create_callback(request):
        body = request.body if isinstance(request.body, str) else request.body.decode("utf-8")
        created_payloads.append(json.loads(body))
        return (201, {"Content-Type": "application/json"}, "{\"id\": 10}")

    responses.add_callback(
        responses.POST,
        "http://localhost:3001/api/agents/create",
        callback=_create_callback,
    )

    register_all()

    assert len(created_payloads) == len(AWET_AGENT_TEAM)
    created_names = {payload["name"] for payload in created_payloads}
    expected_names = {agent.name for agent in AWET_AGENT_TEAM}
    assert created_names == expected_names
