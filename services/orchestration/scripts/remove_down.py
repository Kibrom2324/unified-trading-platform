#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List


@dataclass
class ContainerStatus:
    name: str
    status: str

    @property
    def is_up(self) -> bool:
        return self.status.startswith("Up")


def _run(cmd: list[str]) -> tuple[int, str, str]:
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
        parts = line.split("\t")
        while len(parts) < 2:
            parts.append("")
        rows.append(ContainerStatus(name=parts[0], status=parts[1]))
    return rows


def _parse_minutes(status: str) -> int | None:
    lowered = status.lower()
    if "exit" not in lowered and "exited" not in lowered:
        return None
    tokens = lowered.split()
    for idx, token in enumerate(tokens):
        if token.isdigit() and idx + 1 < len(tokens):
            unit = tokens[idx + 1]
            value = int(token)
            if unit.startswith("minute"):
                return value
            if unit.startswith("hour"):
                return value * 60
            if unit.startswith("day"):
                return value * 24 * 60
            if unit.startswith("second"):
                return max(1, value // 60)
    return None


def _confirm(names: list[str]) -> bool:
    print("About to remove:")
    for name in names:
        print(f"- {name}")
    reply = input("Type YES to continue: ").strip()
    return reply == "YES"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate or apply docker rm for down containers")
    parser.add_argument("--apply", action="store_true", help="Actually remove containers")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Print only (default)")
    parser.add_argument("--older-than-minutes", type=int, default=0, help="Only remove if exited this long")
    args = parser.parse_args()

    code, stdout, stderr = _run(["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}"])
    if code != 0:
        err = stderr or "Failed to run docker ps"
        print(f"Error: {err}", file=sys.stderr)
        return 1

    rows = _parse_rows(stdout)
    down = [row for row in rows if not row.is_up]

    if args.older_than_minutes:
        filtered = []
        for row in down:
            minutes = _parse_minutes(row.status)
            if minutes is None:
                continue
            if minutes >= args.older_than_minutes:
                filtered.append(row)
        down = filtered

    if not down:
        print("No down containers.")
        return 0

    names = [row.name for row in down]
    commands = [f"docker rm -f {name}" for name in names]

    if args.apply:
        if not _confirm(names):
            print("Aborted.")
            return 2
        for name in names:
            _run(["docker", "rm", "-f", name])
        return 2

    for cmd in commands:
        print(cmd)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
