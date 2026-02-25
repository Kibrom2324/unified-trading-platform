#!/usr/bin/env python3
"""
pytest wrapper for E2E test.

Runs scripts/e2e_test.py as a subprocess and asserts exit code 0.

Usage:
    pytest tests/test_e2e.py -v
    pytest -m e2e -v
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


def _docker_compose_services_running() -> tuple[bool, str]:
    """Check if required services are running via docker compose ps."""
    required = {"timescaledb", "kafka", "redis"}
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            return False, f"docker compose ps failed: {result.stderr}"

        running = set()
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                service = data.get("Service", "")
                state = data.get("State", "")
                if state == "running":
                    running.add(service)
            except json.JSONDecodeError:
                continue

        missing = required - running
        if missing:
            return False, f"Services not running: {', '.join(sorted(missing))}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "docker compose ps timed out"
    except FileNotFoundError:
        return False, "docker compose not found"
    except Exception as exc:
        return False, str(exc)


@pytest.mark.e2e
def test_e2e_pipeline() -> None:
    """
    Run the full E2E pipeline test.

    This test is skipped if docker compose services are not running.
    """
    ok, reason = _docker_compose_services_running()
    if not ok:
        pytest.skip(f"Docker services not available: {reason}")

    script_path = REPO_ROOT / "scripts" / "e2e_test.py"
    assert script_path.exists(), f"E2E script not found: {script_path}"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
        cwd=REPO_ROOT,
    )

    # Print output for debugging
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    assert result.returncode == 0, (
        f"E2E test failed with exit code {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


@pytest.mark.e2e
def test_e2e_pipeline_no_ingest() -> None:
    """
    Run E2E pipeline with --no-ingest (assumes data exists from previous run).
    """
    ok, reason = _docker_compose_services_running()
    if not ok:
        pytest.skip(f"Docker services not available: {reason}")

    script_path = REPO_ROOT / "scripts" / "e2e_test.py"
    assert script_path.exists(), f"E2E script not found: {script_path}"

    result = subprocess.run(
        [sys.executable, str(script_path), "--no-ingest"],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
        cwd=REPO_ROOT,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    assert result.returncode == 0, (
        f"E2E test (--no-ingest) failed with exit code {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
