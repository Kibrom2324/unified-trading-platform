#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import List


@dataclass
class ContainerStatus:
    name: str
    status: str
    image: str
    ports: str

    @property
    def is_up(self) -> bool:
        return self.status.startswith("Up")


def _run_docker_ps() -> tuple[int, str, str]:
    cmd = [
        "docker",
        "compose",
        "ps",
        "--format",
        "json",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return 1, "", "docker not found in PATH"
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _parse_rows(output: str) -> List[ContainerStatus]:
    rows: List[ContainerStatus] = []
    if not output:
        return rows
    for line in output.splitlines():
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = data.get("Service") or data.get("Name") or "unknown"
        status = data.get("State") or data.get("Status") or "unknown"
        image = data.get("Image") or ""
        ports = data.get("Ports") or ""
        rows.append(ContainerStatus(name=name, status=status, image=image, ports=ports))
    return rows


def _print_text_report(down: List[ContainerStatus], up: List[ContainerStatus], only_down: bool) -> None:
    print("DOWN containers:")
    if down:
        for item in down:
            ports = f" | {item.ports}" if item.ports else ""
            print(f"- {item.name} | {item.status} | {item.image}{ports}")
    else:
        print("- None")

    if only_down:
        return

    print("\nUP containers:")
    if up:
        for item in up:
            print(f"- {item.name} | {item.status}")
    else:
        print("- None")


def _print_json_report(down: List[ContainerStatus], up: List[ContainerStatus], only_down: bool) -> None:
    payload = {
        "down": [item.__dict__ for item in down],
        "up": [item.__dict__ for item in up],
    }
    if only_down:
        payload = {"down": [item.__dict__ for item in down]}
    print(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Report Docker containers that are down")
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    parser.add_argument("--only-down", action="store_true", help="Print only down containers")
    args = parser.parse_args()

    code, stdout, stderr = _run_docker_ps()
    if code != 0:
        err_msg = stderr or "Failed to run docker ps"
        print(f"Error: {err_msg}", file=sys.stderr)
        return 1

    rows = _parse_rows(stdout)
    down = [row for row in rows if not row.is_up]
    up = [row for row in rows if row.is_up]

    if args.json:
        _print_json_report(down, up, args.only_down)
    else:
        _print_text_report(down, up, args.only_down)

    return 2 if down else 0


if __name__ == "__main__":
    raise SystemExit(main())
