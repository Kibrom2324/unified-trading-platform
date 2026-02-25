#!/usr/bin/env python3
"""
SuperAGI Agent Scheduler

Schedule and trigger SuperAGI agent runs from cron or systemd timers.
This script is designed to be idempotent and safe for automated execution.

Usage:
    python scripts/schedule_agents.py night_trainer
    python scripts/schedule_agents.py morning_deployer
    python scripts/schedule_agents.py trade_watchdog
    python scripts/schedule_agents.py orchestrator

Environment Variables:
    SUPERAGI_API_KEY: API key for SuperAGI (required)
    SUPERAGI_BASE_URL: Base URL for SuperAGI API (default: http://localhost:8100)

Example cron entries (add via `crontab -e`):
    # Night Trainer at 2:00 AM on weekdays
    0 2 * * 1-5  cd /home/kironix/Awet && .venv/bin/python scripts/schedule_agents.py night_trainer

    # Morning Deployer at 8:30 AM on weekdays
    30 8 * * 1-5 cd /home/kironix/Awet && .venv/bin/python scripts/schedule_agents.py morning_deployer

    # Trade Watchdog every 15 minutes during market hours (9 AM - 4 PM) on weekdays
    */15 9-16 * * 1-5 cd /home/kironix/Awet && .venv/bin/python scripts/schedule_agents.py trade_watchdog
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Agent configuration - maps friendly names to agent details
# These can be overridden via environment variables
AGENT_CONFIG = {
    "orchestrator": {
        "name": "AWET Orchestrator",
        "env_id": "AWET_ORCHESTRATOR_AGENT_ID",
        "default_id": 3,  # Fallback ID if not in env
        "goals": [
            "Run the AWET trading pipeline end-to-end",
            "Report results clearly",
        ],
        "instructions": [
            "Execute tools in sequence: backfill → train → promote → demo",
            "Check pipeline health after operations",
        ],
    },
    "night_trainer": {
        "name": "AWET Night Trainer",
        "env_id": "AWET_NIGHT_TRAINER_AGENT_ID",
        "default_id": 4,
        "goals": [
            "Run nightly backfill of market data",
            "Train TFT model on latest data",
            "Report training metrics",
        ],
        "instructions": [
            "Run backfill for polygon data",
            "Train model with default parameters",
            "Check pipeline health after training",
        ],
    },
    "morning_deployer": {
        "name": "AWET Morning Deployer",
        "env_id": "AWET_MORNING_DEPLOYER_AGENT_ID",
        "default_id": 5,
        "goals": [
            "Check pipeline health",
            "Promote latest trained model to production",
            "Run smoke test demo",
        ],
        "instructions": [
            "Check all agents are healthy",
            "Promote the latest yellow model to green",
            "Run demo to verify pipeline flow",
        ],
    },
    "trade_watchdog": {
        "name": "AWET Trade Watchdog",
        "env_id": "AWET_TRADE_WATCHDOG_AGENT_ID",
        "default_id": 6,
        "goals": [
            "Monitor pipeline health during market hours",
            "Report any issues immediately",
        ],
        "instructions": [
            "Check pipeline health",
            "Report status clearly",
        ],
    },
}


class SuperAGIClient:
    """Simple HTTP client for SuperAGI API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url or os.getenv(
            "SUPERAGI_BASE_URL", "http://localhost:8100"  # Direct backend, not proxy
        )
        self.api_key = api_key or os.getenv("SUPERAGI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SUPERAGI_API_KEY environment variable is required"
            )
        self.timeout = timeout
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                headers={
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> "SuperAGIClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def get_agents(self, project_id: int = 1) -> list[dict[str, Any]]:
        """Get list of all agents for a project."""
        response = self.client.get(f"/agents/get/project/{project_id}")
        response.raise_for_status()
        return response.json()

    def find_agent_by_name(self, name: str, project_id: int = 1) -> dict[str, Any] | None:
        """Find an agent by name."""
        agents = self.get_agents(project_id)
        for agent in agents:
            if agent.get("name") == name:
                return agent
        return None

    def create_run(
        self,
        agent_id: int,
        name: str = "Scheduled Run",
        goals: list[str] | None = None,
        instructions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new run for an agent using /agentexecutions/add."""
        payload: dict[str, Any] = {
            "agent_id": agent_id,
            "name": name,
        }
        if goals:
            payload["goal"] = goals
        if instructions:
            payload["instruction"] = instructions

        response = self.client.post("/agentexecutions/add", json=payload)
        response.raise_for_status()
        return response.json()

    def get_run_status(self, run_id: int) -> dict[str, Any]:
        """Get the status of a run."""
        response = self.client.get(f"/agentexecutions/get/{run_id}")
        response.raise_for_status()
        return response.json()


def get_agent_id(agent_key: str) -> int:
    """Get agent ID from environment or default config."""
    config = AGENT_CONFIG.get(agent_key)
    if not config:
        raise ValueError(f"Unknown agent: {agent_key}")

    # Try environment variable first
    env_id = os.getenv(config["env_id"])
    if env_id:
        return int(env_id)

    # Fall back to default
    return config["default_id"]


def run_agent(agent_key: str) -> int:
    """
    Trigger a new run for the specified agent.
    
    Returns:
        Exit code: 0 for success, 1 for failure
    """
    config = AGENT_CONFIG.get(agent_key)
    if not config:
        logger.error(f"Unknown agent: {agent_key}")
        logger.info(f"Available agents: {', '.join(AGENT_CONFIG.keys())}")
        return 1

    agent_name = config["name"]
    timestamp = datetime.now(timezone.utc).isoformat()
    logger.info(f"[{timestamp}] Starting run for {agent_name}")

    try:
        with SuperAGIClient() as client:
            # Try to find agent by name first
            agent = client.find_agent_by_name(agent_name)
            if agent:
                agent_id = agent["id"]
                logger.info(f"Found agent '{agent_name}' with ID {agent_id}")
            else:
                agent_id = get_agent_id(agent_key)
                logger.warning(
                    f"Agent '{agent_name}' not found by name, using ID {agent_id}"
                )

            # Create the run
            result = client.create_run(
                agent_id=agent_id,
                goals=config.get("goals"),
                instructions=config.get("instructions"),
            )

            run_id = result.get("id") or result.get("run_id")
            logger.info(f"✅ Run created successfully: run_id={run_id}")
            logger.info(f"   Agent: {agent_name} (ID: {agent_id})")
            logger.info(f"   Timestamp: {timestamp}")

            # Output JSON for machine parsing
            print(json.dumps({
                "status": "success",
                "agent": agent_name,
                "agent_id": agent_id,
                "run_id": run_id,
                "timestamp": timestamp,
            }))

            return 0

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        return 1
    except httpx.ConnectError as e:
        logger.error(f"Connection error: {e}")
        logger.error("Is SuperAGI running? Try: make superagi-up")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


# Convenience functions for each agent
def run_night_trainer() -> int:
    """Run the Night Trainer agent."""
    return run_agent("night_trainer")


def run_morning_deployer() -> int:
    """Run the Morning Deployer agent."""
    return run_agent("morning_deployer")


def run_trade_watchdog() -> int:
    """Run the Trade Watchdog agent."""
    return run_agent("trade_watchdog")


def run_orchestrator() -> int:
    """Run the Orchestrator agent."""
    return run_agent("orchestrator")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Schedule and trigger SuperAGI agent runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "agent",
        choices=list(AGENT_CONFIG.keys()),
        help="Agent to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing",
    )

    args = parser.parse_args()

    if args.dry_run:
        config = AGENT_CONFIG[args.agent]
        agent_id = get_agent_id(args.agent)
        print(f"Would run agent: {config['name']} (ID: {agent_id})")
        print(f"Goals: {config.get('goals', [])}")
        print(f"Instructions: {config.get('instructions', [])}")
        return 0

    return run_agent(args.agent)


if __name__ == "__main__":
    sys.exit(main())
