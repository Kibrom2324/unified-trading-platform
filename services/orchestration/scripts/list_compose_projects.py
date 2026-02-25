#!/usr/bin/env python3
"""
List Docker containers grouped by Compose project.

Usage:
    python scripts/list_compose_projects.py
    make stack-list

Shows which containers belong to AWET vs other projects (superagi, monitoring, etc.)
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict


def get_containers() -> list[dict]:
    """Get all containers with labels."""
    try:
        result = subprocess.run(
            [
                "docker", "ps", "-a",
                "--format", "{{json .}}"
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        containers = []
        for line in result.stdout.strip().split("\n"):
            if line:
                containers.append(json.loads(line))
        return containers
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running docker ps: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Docker not found. Is Docker installed?", file=sys.stderr)
        sys.exit(1)


def get_container_labels(container_id: str) -> dict:
    """Get labels for a specific container."""
    try:
        result = subprocess.run(
            ["docker", "inspect", container_id, "--format", "{{json .Config.Labels}}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout.strip())
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return {}


def format_ports(ports_str: str) -> str:
    """Format ports string for display."""
    if not ports_str:
        return "-"
    # Shorten long port strings
    parts = ports_str.split(", ")
    if len(parts) > 2:
        return f"{parts[0]}, ... (+{len(parts)-1})"
    return ports_str


def format_status(status: str) -> str:
    """Add emoji to status."""
    status_lower = status.lower()
    if "up" in status_lower:
        return f"🟢 {status}"
    elif "exited" in status_lower:
        return f"🔴 {status}"
    elif "created" in status_lower:
        return f"🟡 {status}"
    elif "paused" in status_lower:
        return f"⏸️  {status}"
    return f"⚪ {status}"


def main():
    containers = get_containers()
    
    if not containers:
        print("No containers found.")
        return
    
    # Group by compose project
    projects: dict[str, list[dict]] = defaultdict(list)
    standalone: list[dict] = []
    
    for c in containers:
        labels = get_container_labels(c["ID"])
        project = labels.get("com.docker.compose.project", "")
        service = labels.get("com.docker.compose.service", "")
        
        container_info = {
            "id": c["ID"][:12],
            "name": c["Names"],
            "image": c["Image"].split(":")[0].split("/")[-1],  # Short image name
            "status": c["Status"],
            "ports": c.get("Ports", ""),
            "service": service,
        }
        
        if project:
            projects[project].append(container_info)
        else:
            standalone.append(container_info)
    
    # Print header
    print("=" * 80)
    print("🐳 Docker Containers by Compose Project")
    print("=" * 80)
    print()
    
    # Sort projects, putting "awet" first
    sorted_projects = sorted(projects.keys(), key=lambda x: (x != "awet", x))
    
    for project in sorted_projects:
        containers_in_project = projects[project]
        is_awet = project == "awet"
        marker = "⭐" if is_awet else "📦"
        
        print(f"{marker} Project: {project}")
        print(f"   Services: {len(containers_in_project)}")
        if is_awet:
            print("   Stop with: docker compose down  (from repo root)")
        else:
            print(f"   Stop with: docker compose -p {project} down")
        print("-" * 80)
        print(f"   {'SERVICE':<20} {'CONTAINER':<25} {'STATUS':<25} {'PORTS'}")
        print("-" * 80)
        
        for c in sorted(containers_in_project, key=lambda x: x["service"]):
            service = c["service"] or c["image"]
            name = c["name"][:24]
            status = format_status(c["status"])
            ports = format_ports(c["ports"])
            print(f"   {service:<20} {name:<25} {status:<25} {ports}")
        
        print()
    
    # Standalone containers (not managed by compose)
    if standalone:
        print("📋 Standalone Containers (not Compose-managed)")
        print("-" * 80)
        print(f"   {'NAME':<25} {'IMAGE':<20} {'STATUS':<25} {'PORTS'}")
        print("-" * 80)
        
        for c in sorted(standalone, key=lambda x: x["name"]):
            name = c["name"][:24]
            image = c["image"][:19]
            status = format_status(c["status"])
            ports = format_ports(c["ports"])
            print(f"   {name:<25} {image:<20} {status:<25} {ports}")
        
        print()
    
    # Summary
    print("=" * 80)
    print("📊 Summary")
    print("-" * 80)
    total_containers = sum(len(v) for v in projects.values()) + len(standalone)
    awet_count = len(projects.get("awet", []))
    other_count = total_containers - awet_count
    
    print(f"   Total containers: {total_containers}")
    print(f"   AWET project:     {awet_count}")
    print(f"   Other projects:   {other_count}")
    print()
    print("💡 Quick Commands:")
    print("   Stop AWET only:        docker compose down")
    print("   Stop specific project: docker compose -p <project> down")
    print("   Stop all containers:   docker stop $(docker ps -q)")
    print("=" * 80)


if __name__ == "__main__":
    main()
