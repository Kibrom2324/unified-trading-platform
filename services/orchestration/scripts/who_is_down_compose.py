#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, List


@dataclass
class ServiceStatus:
    service: str
    name: str
    state: str
    status: str

    @property
    def is_up(self) -> bool:
        if self.state and self.state.lower() != "running":
            return False
        lowered = (self.status or "").lower()
        if "exit" in lowered or "restarting" in lowered:
            return False
        return True


def _run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return 1, "", "docker not found in PATH"
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _parse_json(output: str) -> list[ServiceStatus]:
    items = json.loads(output)
    rows: list[ServiceStatus] = []
    for item in items:
        rows.append(
            ServiceStatus(
                service=item.get("Service", ""),
                name=item.get("Name", ""),
                state=item.get("State", ""),
                status=item.get("Status", ""),
            )
        )
    return rows


def _parse_text(output: str) -> list[ServiceStatus]:
    rows: list[ServiceStatus] = []
    if not output:
        return rows
    lines = output.splitlines()
    if len(lines) > 1 and lines[0].lower().startswith("name"):
        lines = lines[1:]
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        name = parts[0]
        service = parts[1]
        state = parts[2]
        status = " ".join(parts[3:])
        rows.append(ServiceStatus(service=service, name=name, state=state, status=status))
    return rows


def _print_text(down: list[ServiceStatus], up: list[ServiceStatus]) -> None:
    print("DOWN services:")
    if down:
        for item in down:
            print(f"- {item.service} | {item.name} | {item.state} | {item.status}")
    else:
        print("- None")

    print("\nUP services:")
    if up:
        for item in up:
            print(f"- {item.service} | {item.name} | {item.state} | {item.status}")
    else:
        print("- None")


def _print_json(down: list[ServiceStatus], up: list[ServiceStatus]) -> None:
    payload: dict[str, Any] = {
        "down": [item.__dict__ for item in down],
        "up": [item.__dict__ for item in up],
    }
    print(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Report down services for a compose project")
    parser.add_argument("--project", default="awet", help="Compose project name")
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    args = parser.parse_args()

    code, stdout, stderr = _run(["docker", "compose", "-p", args.project, "ps", "--format", "json"])
    rows: list[ServiceStatus] = []
    if code == 0 and stdout:
        try:
            rows = _parse_json(stdout)
        except json.JSONDecodeError:
            rows = []
    if not rows:
        code, stdout, stderr = _run(["docker", "compose", "-p", args.project, "ps"])
        if code != 0:
            err = stderr or "Failed to run docker compose ps"
            print(f"Error: {err}", file=sys.stderr)
            return 1
        rows = _parse_text(stdout)

    down = [row for row in rows if not row.is_up]
    up = [row for row in rows if row.is_up]

    if args.json:
        _print_json(down, up)
    else:
        _print_text(down, up)

    return 2 if down else 0


if __name__ == "__main__":
    raise SystemExit(main())
