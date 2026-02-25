#!/usr/bin/env python3
"""
AWET System Health Check

Checks the health of all AWET system components:
- TimescaleDB connectivity
- Kafka connectivity (via Docker)
- SuperAGI API availability
- All four agents exist in SuperAGI

Usage:
    python scripts/awet_health_check.py
    python scripts/awet_health_check.py --json   # Machine-readable output

Exit Codes:
    0 - All systems healthy
    1 - One or more systems unhealthy
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# Expected agents
EXPECTED_AGENTS = [
    "AWET Orchestrator",
    "AWET Night Trainer",
    "AWET Morning Deployer",
    "AWET Trade Watchdog",
]


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    healthy: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health status."""
    checks: list[HealthCheckResult] = field(default_factory=list)
    
    @property
    def all_healthy(self) -> bool:
        return all(c.healthy for c in self.checks)
    
    @property
    def healthy_count(self) -> int:
        return sum(1 for c in self.checks if c.healthy)
    
    @property
    def total_count(self) -> int:
        return len(self.checks)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.all_healthy,
            "summary": f"{self.healthy_count}/{self.total_count} checks passed",
            "checks": [
                {
                    "name": c.name,
                    "healthy": c.healthy,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


def load_env() -> dict[str, str]:
    """Load environment variables from .env file."""
    env = dict(os.environ)
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env.setdefault(key.strip(), value.strip())
    return env


def check_postgres(env: dict[str, str]) -> HealthCheckResult:
    """Check TimescaleDB/Postgres connectivity."""
    try:
        import asyncpg
        import asyncio
        
        database_url = env.get("DATABASE_URL", "postgresql://awet:awet@localhost:5433/awet")
        
        async def _check():
            conn = await asyncpg.connect(database_url, timeout=5.0)
            version = await conn.fetchval("SELECT version()")
            await conn.close()
            return version
        
        version = asyncio.run(_check())
        return HealthCheckResult(
            name="TimescaleDB",
            healthy=True,
            message="Connected successfully",
            details={"version": version[:50] + "..." if len(version) > 50 else version},
        )
    except ImportError:
        # Fall back to psql via docker
        try:
            result = subprocess.run(
                ["docker", "exec", "timescaledb", "pg_isready", "-U", "awet"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return HealthCheckResult(
                    name="TimescaleDB",
                    healthy=True,
                    message="pg_isready OK",
                )
            return HealthCheckResult(
                name="TimescaleDB",
                healthy=False,
                message=f"pg_isready failed: {result.stderr}",
            )
        except Exception as e:
            return HealthCheckResult(
                name="TimescaleDB",
                healthy=False,
                message=f"Check failed: {e}",
            )
    except Exception as e:
        return HealthCheckResult(
            name="TimescaleDB",
            healthy=False,
            message=str(e),
        )


def check_kafka() -> HealthCheckResult:
    """Check Kafka connectivity via Docker."""
    try:
        # Confluent images use 'kafka-topics' (no .sh extension)
        result = subprocess.run(
            [
                "docker", "exec", "kafka",
                "kafka-topics", "--bootstrap-server", "localhost:9092", "--list",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            topics = [t for t in result.stdout.strip().split("\n") if t]
            return HealthCheckResult(
                name="Kafka",
                healthy=True,
                message=f"{len(topics)} topics available",
                details={"topic_count": len(topics)},
            )
        return HealthCheckResult(
            name="Kafka",
            healthy=False,
            message=f"Failed to list topics: {result.stderr}",
        )
    except subprocess.TimeoutExpired:
        return HealthCheckResult(
            name="Kafka",
            healthy=False,
            message="Timeout connecting to Kafka",
        )
    except Exception as e:
        return HealthCheckResult(
            name="Kafka",
            healthy=False,
            message=str(e),
        )


def check_superagi_api(env: dict[str, str]) -> HealthCheckResult:
    """Check SuperAGI API availability."""
    # Use direct backend port (8100) which is more reliable
    base_url = env.get("SUPERAGI_BASE_URL", "http://localhost:8100")
    if "3001" in base_url:
        base_url = "http://localhost:8100"  # Use direct backend
    
    try:
        with httpx.Client(timeout=10.0) as client:
            # Try health endpoint first, then agents endpoint
            for endpoint in ["/health", "/agents/get/project/1"]:
                try:
                    resp = client.get(f"{base_url}{endpoint}")
                    if resp.status_code < 500:
                        return HealthCheckResult(
                            name="SuperAGI API",
                            healthy=True,
                            message=f"Responding on {base_url}",
                            details={"endpoint": endpoint, "status_code": resp.status_code},
                        )
                except httpx.RequestError:
                    continue
            
            return HealthCheckResult(
                name="SuperAGI API",
                healthy=False,
                message=f"No healthy endpoints found at {base_url}",
            )
    except Exception as e:
        return HealthCheckResult(
            name="SuperAGI API",
            healthy=False,
            message=str(e),
        )


def check_agents(env: dict[str, str]) -> HealthCheckResult:
    """Check that all expected agents exist in SuperAGI."""
    base_url = env.get("SUPERAGI_BASE_URL", "http://localhost:8100")
    if "3001" in base_url:
        base_url = "http://localhost:8100"
    
    api_key = env.get("SUPERAGI_API_KEY", "")
    
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{base_url}/agents/get/project/1",
                headers={"X-API-Key": api_key} if api_key else {},
            )
            resp.raise_for_status()
            agents = resp.json()
            
            agent_names = {a.get("name") for a in agents}
            found = [name for name in EXPECTED_AGENTS if name in agent_names]
            missing = [name for name in EXPECTED_AGENTS if name not in agent_names]
            
            if len(missing) == 0:
                return HealthCheckResult(
                    name="SuperAGI Agents",
                    healthy=True,
                    message=f"All {len(EXPECTED_AGENTS)} agents found",
                    details={"found": found},
                )
            else:
                return HealthCheckResult(
                    name="SuperAGI Agents",
                    healthy=False,
                    message=f"Missing {len(missing)} agents",
                    details={"found": found, "missing": missing},
                )
    except Exception as e:
        return HealthCheckResult(
            name="SuperAGI Agents",
            healthy=False,
            message=str(e),
        )


def check_ollama() -> HealthCheckResult:
    """Check Ollama LLM service availability."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get("http://localhost:11434/")
            if resp.status_code == 200:
                return HealthCheckResult(
                    name="Ollama",
                    healthy=True,
                    message="Ollama is running",
                )
            return HealthCheckResult(
                name="Ollama",
                healthy=False,
                message=f"Unexpected status: {resp.status_code}",
            )
    except Exception as e:
        return HealthCheckResult(
            name="Ollama",
            healthy=False,
            message=str(e),
        )


def run_health_check() -> SystemHealth:
    """Run all health checks and return results."""
    env = load_env()
    health = SystemHealth()
    
    # Run all checks
    health.checks.append(check_postgres(env))
    health.checks.append(check_kafka())
    health.checks.append(check_superagi_api(env))
    health.checks.append(check_agents(env))
    health.checks.append(check_ollama())
    
    return health


def print_results(health: SystemHealth, as_json: bool = False) -> None:
    """Print health check results."""
    if as_json:
        print(json.dumps(health.to_dict(), indent=2))
        return
    
    print("=" * 50)
    print("🏥 AWET System Health Check")
    print("=" * 50)
    print()
    
    for check in health.checks:
        status = "✅" if check.healthy else "❌"
        print(f"{status} {check.name}: {check.message}")
        if check.details and not check.healthy:
            for key, value in check.details.items():
                print(f"   {key}: {value}")
    
    print()
    print("-" * 50)
    
    if health.all_healthy:
        print(f"✅ All {health.total_count} checks passed - System is HEALTHY")
    else:
        failed = health.total_count - health.healthy_count
        print(f"❌ {failed}/{health.total_count} checks failed - System is UNHEALTHY")
    
    print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check AWET system health",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()
    
    health = run_health_check()
    print_results(health, as_json=args.json)
    
    return 0 if health.all_healthy else 1


if __name__ == "__main__":
    sys.exit(main())
