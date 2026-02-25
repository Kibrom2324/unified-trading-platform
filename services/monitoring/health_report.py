from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.config import load_settings


def _run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return 1, "", "docker not found in PATH"
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _parse_rows(output: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not output:
        return rows
    for line in output.splitlines():
        parts = line.split("\t")
        while len(parts) < 4:
            parts.append("")
        rows.append(
            {
                "name": parts[0],
                "status": parts[1],
                "image": parts[2],
                "ports": parts[3],
            }
        )
    return rows


def _container_report() -> dict[str, Any]:
    code, stdout, stderr = _run(
        ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Ports}}"]
    )
    if code != 0:
        return {"error": stderr or "Failed to run docker ps"}

    rows = _parse_rows(stdout)
    running = [row for row in rows if row.get("status", "").startswith("Up")]
    down = [row for row in rows if not row.get("status", "").startswith("Up")]
    return {"running_containers": running, "down_containers": down}


class HealthReportAgent(BaseAgent):
    def __init__(self) -> None:
        settings = load_settings()
        super().__init__("health_report", settings.app.http.health_report_port)
        self.add_protected_route("/health/report", self.report, methods=["GET"])

    async def start(self) -> None:
        return None

    async def report(self) -> dict[str, Any]:
        payload = _container_report()
        payload["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
        return payload


def main() -> None:
    HealthReportAgent().run()


if __name__ == "__main__":
    main()
