#!/usr/bin/env python3
"""AWET Trading Platform — Full-Detail Dashboard API Server.

Serves a comprehensive dashboard and REST API that surfaces every detail
of the trading pipeline: infrastructure, agents, models, schemas, database
columns, Docker services, Prometheus alerts, Alertmanager config, risk
parameters, directives, Makefile commands, dependencies, safety gates,
feature engineering metadata, and more.

Usage:
    python scripts/dashboard_server.py          # http://localhost:8888
    python scripts/dashboard_server.py --port 9999
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

logger = structlog.get_logger("dashboard")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
MODELS_DIR = ROOT / "models"
TEMPLATES_DIR = ROOT / "templates"
DB_INIT = ROOT / "db" / "init.sql"
SCHEMAS_DIR = ROOT / "src" / "schemas"
DIRECTIVES_DIR = ROOT / "directives"

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="AWET Dashboard API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _safe_load_yaml(name: str) -> dict[str, Any]:
    try:
        return _load_yaml(CONFIG_DIR / name)
    except Exception:
        return {}


def _load_registry() -> list[dict[str, Any]]:
    reg_path = MODELS_DIR / "registry.json"
    if not reg_path.exists():
        return []
    try:
        with open(reg_path) as f:
            data = json.load(f)
        return data.get("models", [])
    except Exception:
        return []


def _tcp_check(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


async def _http_health(url: str, timeout: float = 2.0) -> dict[str, Any]:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
            return {"status": "up", "code": r.status_code,
                    "body": r.json() if r.status_code == 200 else None}
    except Exception as e:
        return {"status": "down", "error": str(e)}


def _run_git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "--no-pager", *args], cwd=str(ROOT), text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def _file_age(p: Path) -> str | None:
    try:
        mtime = p.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. /api/overview — high-level platform stats
# ---------------------------------------------------------------------------
@app.get("/api/overview")
async def overview():
    app_cfg = _safe_load_yaml("app.yaml")
    app_s = app_cfg.get("app", {})
    symbols = app_s.get("symbols", [])
    registry = _load_registry()
    green = [m for m in registry if m.get("status") == "green"]
    candidates = [m for m in registry if m.get("status") == "candidate"]

    src_lines = src_files = 0
    for p in (ROOT / "src").rglob("*.py"):
        src_files += 1
        src_lines += sum(1 for _ in open(p, errors="ignore"))

    test_files = sum(1 for _ in (ROOT / "tests").rglob("*.py"))
    script_files = sum(1 for _ in (ROOT / "scripts").rglob("*.py"))
    exec_files = sum(1 for _ in (ROOT / "execution").rglob("*.py")) if (ROOT / "execution").exists() else 0
    directive_files = sum(1 for _ in DIRECTIVES_DIR.glob("*.md")) if DIRECTIVES_DIR.exists() else 0
    schema_files = sum(1 for _ in SCHEMAS_DIR.glob("*.avsc")) if SCHEMAS_DIR.exists() else 0

    approval_file = ROOT / app_s.get("execution_approval_file", ".tmp/APPROVE_EXECUTION")

    return {
        "project": "AWET Trading Platform",
        "version": "0.1.0",
        "environment": app_s.get("environment", "unknown"),
        "symbols": symbols,
        "num_symbols": len(symbols),
        "src_files": src_files,
        "src_lines": src_lines,
        "test_files": test_files,
        "script_files": script_files,
        "execution_files": exec_files,
        "directive_files": directive_files,
        "schema_files": schema_files,
        "models_total": len(registry),
        "models_green": len(green),
        "models_candidate": len(candidates),
        "green_model_id": green[0].get("model_id") or green[0].get("run_id") if green else None,
        "execution_dry_run": app_s.get("execution_dry_run", True),
        "approval_file_exists": approval_file.exists(),
        "paper_trade_only": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# 2. /api/infrastructure — TCP probes for 8 services
# ---------------------------------------------------------------------------
@app.get("/api/infrastructure")
async def infrastructure():
    services = {
        "kafka": {"host": "localhost", "port": 9092, "display": "Kafka Broker", "image": "confluentinc/cp-kafka:7.6.1"},
        "schema_registry": {"host": "localhost", "port": 8081, "display": "Schema Registry", "image": "confluentinc/cp-schema-registry:7.6.1"},
        "timescaledb": {"host": "localhost", "port": 5433, "display": "TimescaleDB", "image": "timescale/timescaledb:2.13.1-pg15"},
        "redis": {"host": "localhost", "port": 6379, "display": "Redis", "image": "redis:7.2"},
        "prometheus": {"host": "localhost", "port": 9090, "display": "Prometheus", "image": "prom/prometheus:v2.54.1"},
        "grafana": {"host": "localhost", "port": 3000, "display": "Grafana", "image": "grafana/grafana:11.1.0"},
        "alertmanager": {"host": "localhost", "port": 9093, "display": "Alertmanager", "image": "prom/alertmanager:v0.27.0"},
        "kafka_ui": {"host": "localhost", "port": 8080, "display": "Kafka UI", "image": "provectuslabs/kafka-ui:v0.7.2"},
    }
    results = {}
    for key, svc in services.items():
        up = _tcp_check(svc["host"], svc["port"])
        results[key] = {
            "display_name": svc["display"],
            "host": svc["host"],
            "port": svc["port"],
            "image": svc["image"],
            "status": "up" if up else "down",
        }
    return {"services": results}


# ---------------------------------------------------------------------------
# 3. /api/agents — HTTP health probes for 9 agents
# ---------------------------------------------------------------------------
@app.get("/api/agents")
async def agents():
    app_cfg = _safe_load_yaml("app.yaml")
    app_s = app_cfg.get("app", {})
    http_cfg = app_s.get("http", {})
    agent_ports = {
        "data_ingestion": http_cfg.get("data_ingestion_port", 8001),
        "feature_engineering": http_cfg.get("feature_engineering_port", 8002),
        "prediction": http_cfg.get("prediction_port", 8003),
        "risk": http_cfg.get("risk_port", 8004),
        "execution": http_cfg.get("execution_port", 8005),
        "watchtower": http_cfg.get("watchtower_port", 8006),
        "orchestrator": http_cfg.get("orchestrator_port", 8007),
        "trader_decision": 8011,
        "backtester": http_cfg.get("backtester_port", 8012),
    }
    results = {}
    coros = {name: _http_health(f"http://localhost:{port}/health") for name, port in agent_ports.items()}
    for name, coro in coros.items():
        results[name] = await coro
        results[name]["port"] = agent_ports[name]
    return {"agents": results}


# ---------------------------------------------------------------------------
# 4. /api/models — registry with full metrics, ONNX, feature_meta
# ---------------------------------------------------------------------------
@app.get("/api/models")
async def models():
    registry = _load_registry()
    for m in registry:
        onnx_path = m.get("onnx_path") or ""
        if not onnx_path:
            path_dir = m.get("path", "")
            onnx_path = str(ROOT / path_dir / "model.onnx")
        m["onnx_exists"] = os.path.exists(onnx_path)
        meta_path = Path(onnx_path).parent / "feature_meta.json" if onnx_path else None
        m["has_feature_meta"] = meta_path.exists() if meta_path else False
        if m["has_feature_meta"]:
            try:
                with open(meta_path) as f:
                    m["feature_meta"] = json.load(f)
            except Exception:
                m["feature_meta"] = None
        # file sizes
        for key in ["onnx_path", "checkpoint_path"]:
            p = Path(m.get(key, ""))
            if p.exists():
                m[f"{key}_size_mb"] = round(p.stat().st_size / 1e6, 2)
                m[f"{key}_modified"] = _file_age(p)
    return {"models": registry}


# ---------------------------------------------------------------------------
# 5. /api/config — full config with ALL sections
# ---------------------------------------------------------------------------
@app.get("/api/config")
async def config_endpoint():
    app_cfg = _safe_load_yaml("app.yaml")
    kafka_cfg = _safe_load_yaml("kafka.yaml")
    limits_cfg = _safe_load_yaml("limits.yaml")
    logging_cfg = _safe_load_yaml("logging.yaml")
    market_cfg = _safe_load_yaml("market_data.yaml")
    llm_cfg = _safe_load_yaml("llm.yaml")
    app_s = app_cfg.get("app", {})

    risk_limits = {
        "max_trade_notional_usd": app_s.get("max_trade_notional_usd"),
        "max_qty_per_trade": app_s.get("max_qty_per_trade"),
        "min_qty_per_trade": app_s.get("min_qty_per_trade"),
        "max_orders_per_minute": app_s.get("max_orders_per_minute"),
        "max_orders_per_symbol_per_day": app_s.get("max_orders_per_symbol_per_day"),
        "max_open_orders_total": app_s.get("max_open_orders_total"),
        "cooldown_seconds_per_symbol": app_s.get("cooldown_seconds_per_symbol"),
        "max_total_exposure_usd": app_s.get("max_total_exposure_usd"),
        "max_exposure_per_symbol_usd": app_s.get("max_exposure_per_symbol_usd"),
        "take_profit_pct": app_s.get("take_profit_pct"),
        "stop_loss_pct": app_s.get("stop_loss_pct"),
        "max_holding_minutes": app_s.get("max_holding_minutes"),
        "use_bracket_orders": app_s.get("use_bracket_orders"),
        "enable_exit_logic": app_s.get("enable_exit_logic"),
        "exit_check_interval_seconds": app_s.get("exit_check_interval_seconds"),
        "check_position_before_sell": app_s.get("check_position_before_sell"),
        "allow_add_to_position": app_s.get("allow_add_to_position"),
        "execution_dry_run": app_s.get("execution_dry_run"),
        "execution_default_qty": app_s.get("execution_default_qty"),
    }

    safety_gates = {
        "layer_1_paper_trade": "HARDCODED True — no real-money code exists",
        "layer_2_dry_run": app_s.get("execution_dry_run", True),
        "layer_3_approval_file": app_s.get("execution_approval_file", ".tmp/APPROVE_EXECUTION"),
        "approval_file_exists": (ROOT / app_s.get("execution_approval_file", ".tmp/APPROVE_EXECUTION")).exists(),
    }

    trader_decision = app_cfg.get("trader_decision_agent", {})

    return {
        "app": app_s,
        "kafka": kafka_cfg.get("kafka", kafka_cfg),
        "training": app_cfg.get("training", {}),
        "data": app_cfg.get("data", {}),
        "risk_limits": risk_limits,
        "safety_gates": safety_gates,
        "trader_decision": trader_decision,
        "limits": limits_cfg,
        "logging": logging_cfg,
        "market_data": market_cfg,
        "llm": llm_cfg,
    }


# ---------------------------------------------------------------------------
# 6. /api/kafka — topics with FULL Avro field details
# ---------------------------------------------------------------------------
@app.get("/api/kafka")
async def kafka_topics():
    topics_meta = [
        {"name": "market.raw", "desc": "Raw OHLCV market bars from Polygon/yfinance", "schema": "market_raw.avsc"},
        {"name": "market.engineered", "desc": "Feature-engineered data (RSI, SMA, EMA, volatility)", "schema": "market_engineered.avsc"},
        {"name": "predictions.tft", "desc": "TFT model quantile predictions (3 horizons × 3 quantiles)", "schema": "prediction.avsc"},
        {"name": "trade.decisions", "desc": "Trader decision agent BUY/SELL/HOLD signals", "schema": "trade_decision.avsc"},
        {"name": "risk.approved", "desc": "Risk-approved trade signals", "schema": "risk.avsc"},
        {"name": "risk.rejected", "desc": "Risk-rejected signals with detailed reasons", "schema": "risk.avsc"},
        {"name": "execution.completed", "desc": "Paper trade execution confirmations", "schema": "execution.avsc"},
        {"name": "execution.blocked", "desc": "Blocked execution events (dry_run / safety gate)", "schema": "execution.avsc"},
    ]
    for t in topics_meta:
        schema_file = SCHEMAS_DIR / t["schema"]
        if schema_file.exists():
            with open(schema_file) as f:
                schema_data = json.load(f)
            t["schema_name"] = schema_data.get("name", "")
            t["namespace"] = schema_data.get("namespace", "")
            t["schema_fields"] = []
            for fld in schema_data.get("fields", []):
                ft = fld["type"]
                if isinstance(ft, list):
                    ft = " | ".join(str(x) for x in ft)
                elif isinstance(ft, dict):
                    ft = ft.get("type", str(ft))
                t["schema_fields"].append({
                    "name": fld["name"],
                    "type": str(ft),
                    "default": fld.get("default", "—"),
                    "doc": fld.get("doc", ""),
                })
        else:
            t["schema_fields"] = []
    return {"topics": topics_meta}


# ---------------------------------------------------------------------------
# 7. /api/pipeline — stage-by-stage flow with full detail
# ---------------------------------------------------------------------------
@app.get("/api/pipeline")
async def pipeline():
    stages = [
        {"id": 1, "name": "Data Ingestion", "agent": "DataIngestionAgent", "port": 8001,
         "input": "Polygon / yfinance API", "output": "market.raw",
         "description": "Fetches OHLCV bars, publishes Avro events with idempotency_key",
         "source_file": "src/agents/data_ingestion_agent.py",
         "features": ["Polygon backfill", "yfinance live", "Reddit sentiment", "Resumable checkpoints"]},
        {"id": 2, "name": "Feature Engineering", "agent": "FeatureEngineeringAgent", "port": 8002,
         "input": "market.raw", "output": "market.engineered",
         "description": "Computes returns_1/5/15, volatility_5/15, SMA_5/20, EMA_5/20, RSI_14, volume z-score, temporal features",
         "source_file": "src/agents/feature_engineering_agent.py",
         "features": ["14 engineered features", "Redis feature cache", "Temporal: minute_of_day, day_of_week"]},
        {"id": 3, "name": "TFT Prediction", "agent": "TimeSeriesPredictionAgent", "port": 8003,
         "input": "market.engineered", "output": "predictions.tft",
         "description": "ONNX inference → 10 outputs: 3 horizons × 3 quantiles (q10/q50/q90) + confidence",
         "source_file": "src/agents/prediction_agent.py",
         "features": ["300-bar lookback window", "Dynamic feature loading from feature_meta.json", "QuantileLoss training", "Blue/green model switching"]},
        {"id": 4, "name": "Trade Decision", "agent": "TraderDecisionAgent", "port": 8011,
         "input": "predictions.tft", "output": "trade.decisions",
         "description": "Converts quantile predictions to BUY/SELL/HOLD with direction + confidence",
         "source_file": "src/agents/trader_decision_agent.py",
         "features": ["Configurable buy/sell thresholds", "Price + prediction propagation", "min_confidence filter"]},
        {"id": 5, "name": "Risk Gate", "agent": "RiskAgent", "port": 8004,
         "input": "trade.decisions", "output": "risk.approved / risk.rejected",
         "description": "Position limits, exposure caps, CVaR, cooldown, kill switch, volatility filter",
         "source_file": "src/agents/risk_agent.py",
         "features": ["Redis state persistence", "Portfolio-level exposure check", "Per-symbol cooldown", "Rate limiting", "Daily loss limit"]},
        {"id": 6, "name": "Execution", "agent": "ExecutionAgent", "port": 8005,
         "input": "risk.approved", "output": "execution.completed / execution.blocked",
         "description": "Paper trade via Alpaca API, bracket orders (TP/SL), position tracking, TimescaleDB audit",
         "source_file": "src/agents/execution_agent.py",
         "features": ["3-layer safety gates", "Bracket orders (take-profit + stop-loss)", "Loop detection", "Throttle (5 orders/min)", "Dry run mode"]},
        {"id": 7, "name": "Watchtower", "agent": "WatchtowerAgent", "port": 8006,
         "input": "(all topics)", "output": "Prometheus metrics",
         "description": "Monitors consumer lag, agent health, pipeline throughput, fires alerts",
         "source_file": "src/agents/watchtower_agent.py",
         "features": ["Kafka consumer lag tracking", "Agent HTTP health probes", "Prometheus metrics export", "Graceful shutdown awareness"]},
    ]
    return {"stages": stages}


# ---------------------------------------------------------------------------
# 8. /api/database — full schema with columns, types, indexes
# ---------------------------------------------------------------------------
@app.get("/api/database")
async def database():
    tables_from_sql: list[dict] = []
    if DB_INIT.exists():
        content = DB_INIT.read_text()
        # Parse CREATE TABLE blocks
        table_pattern = re.compile(
            r"CREATE TABLE IF NOT EXISTS (\w+)\s*\((.*?)\);",
            re.DOTALL | re.IGNORECASE
        )
        for match in table_pattern.finditer(content):
            tname = match.group(1)
            body = match.group(2)
            columns = []
            for line in body.split("\n"):
                line = line.strip().rstrip(",")
                if not line or line.upper().startswith(("PRIMARY KEY", "UNIQUE", "CHECK", "CONSTRAINT", "CREATE", "SELECT")):
                    continue
                parts = line.split()
                if len(parts) >= 2 and not parts[0].upper().startswith(("PRIMARY", "UNIQUE", "CHECK", "CONSTRAINT")):
                    col_name = parts[0]
                    col_type = parts[1]
                    nullable = "NOT NULL" not in line.upper()
                    default_match = re.search(r"DEFAULT\s+(.+?)(?:\s*,?\s*$)", line, re.IGNORECASE)
                    default_val = default_match.group(1).strip().rstrip(",") if default_match else None
                    columns.append({
                        "name": col_name,
                        "type": col_type,
                        "nullable": nullable,
                        "default": default_val,
                    })
            tables_from_sql.append({"name": tname, "columns": columns})

        # Parse retention policies
        retention = []
        for match in re.finditer(r"add_retention_policy\('(\w+)',\s*INTERVAL\s*'([^']+)'", content):
            retention.append({"table": match.group(1), "interval": match.group(2)})

        # Parse hypertables
        hypertables = []
        for match in re.finditer(r"create_hypertable\('(\w+)',\s*'(\w+)'", content):
            hypertables.append({"table": match.group(1), "time_column": match.group(2)})

        # Parse indexes
        indexes = []
        for match in re.finditer(r"CREATE INDEX IF NOT EXISTS (\w+)\s+ON\s+(\w+)\s*\(([^)]+)\)", content):
            indexes.append({"name": match.group(1), "table": match.group(2), "columns": match.group(3).strip()})
    else:
        retention = []
        hypertables = []
        indexes = []

    # Live DB stats
    db_stats: dict = {}
    try:
        import asyncpg
        conn = await asyncpg.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5433")),
            user=os.getenv("POSTGRES_USER", "awet"),
            password=os.getenv("POSTGRES_PASSWORD", "awet"),
            database=os.getenv("POSTGRES_DB", "awet"),
        )
        try:
            rows = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename")
            for r in rows:
                tn = r["tablename"]
                try:
                    cnt = await conn.fetchval(f"SELECT COUNT(*) FROM {tn}")
                    sz = await conn.fetchval(f"SELECT pg_total_relation_size('{tn}')")
                    db_stats[tn] = {"rows": cnt, "size_bytes": sz, "size_mb": round(sz / 1e6, 2)}
                except Exception:
                    db_stats[tn] = {"rows": "error"}

            # DB size total
            total_size = await conn.fetchval("SELECT pg_database_size('awet')")
            db_stats["_database_total"] = {"size_bytes": total_size, "size_mb": round(total_size / 1e6, 2)}
        finally:
            await conn.close()
    except Exception as e:
        db_stats = {"error": str(e)}

    return {
        "tables": tables_from_sql,
        "live_stats": db_stats,
        "retention_policies": retention,
        "hypertables": hypertables,
        "indexes": indexes,
    }


# ---------------------------------------------------------------------------
# 9. /api/tests — test file listing with paths
# ---------------------------------------------------------------------------
@app.get("/api/tests")
async def tests():
    test_dir = ROOT / "tests"
    unit_tests = sorted((test_dir / "unit").rglob("test_*.py")) if (test_dir / "unit").exists() else []
    integration_tests = sorted((test_dir / "integration").rglob("test_*.py")) if (test_dir / "integration").exists() else []
    # Count test functions
    def count_tests(path: Path) -> int:
        try:
            return sum(1 for ln in open(path, errors="ignore") if ln.strip().startswith(("def test_", "async def test_")))
        except Exception:
            return 0

    unit_detail = [{"file": p.name, "path": str(p.relative_to(ROOT)), "test_count": count_tests(p)} for p in unit_tests]
    integ_detail = [{"file": p.name, "path": str(p.relative_to(ROOT)), "test_count": count_tests(p)} for p in integration_tests]

    return {
        "unit_test_files": len(unit_tests),
        "integration_test_files": len(integration_tests),
        "total_test_functions": sum(t["test_count"] for t in unit_detail) + sum(t["test_count"] for t in integ_detail),
        "unit_tests": unit_detail,
        "integration_tests": integ_detail,
    }


# ---------------------------------------------------------------------------
# 10. /api/universe — trading symbols with enriched data
# ---------------------------------------------------------------------------
@app.get("/api/universe")
async def universe():
    csv_path = CONFIG_DIR / "universe.csv"
    if not csv_path.exists():
        return {"symbols": []}
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return {"symbols": rows, "count": len(rows)}


# ---------------------------------------------------------------------------
# 11. /api/git — detailed git info
# ---------------------------------------------------------------------------
@app.get("/api/git")
async def git_info():
    try:
        branch = _run_git("branch", "--show-current")
        log_lines = _run_git("log", "--oneline", "--format=%h %s", "-10").split("\n")
        status_lines = _run_git("status", "--short").split("\n")
        status_lines = [s for s in status_lines if s.strip()]
        modified = [s for s in status_lines if s.startswith(" M") or s.startswith("M ")]
        untracked = [s for s in status_lines if s.startswith("??")]
        added = [s for s in status_lines if s.startswith("A ")]
        deleted = [s for s in status_lines if s.startswith(" D") or s.startswith("D ")]

        # Commit count
        commit_count = _run_git("rev-list", "--count", "HEAD")
        # Last commit date
        last_commit_date = _run_git("log", "-1", "--format=%ci")
        # Remote
        remote = _run_git("remote", "get-url", "origin")
        # Tags
        tags = _run_git("tag", "--sort=-creatordate").split("\n")[:5]
        tags = [t for t in tags if t.strip()]

        return {
            "branch": branch,
            "recent_commits": log_lines,
            "modified_files": len(modified),
            "modified_list": [s[3:] for s in modified[:20]],
            "untracked_files": len(untracked),
            "untracked_list": [s[3:] for s in untracked[:20]],
            "added_files": len(added),
            "deleted_files": len(deleted),
            "total_changes": len(status_lines),
            "commit_count": commit_count,
            "last_commit_date": last_commit_date,
            "remote": remote,
            "tags": tags,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 12. /api/docker — Docker Compose service inventory
# ---------------------------------------------------------------------------
@app.get("/api/docker")
async def docker_services():
    dc_path = ROOT / "docker-compose.yml"
    if not dc_path.exists():
        return {"services": []}
    dc = _load_yaml(dc_path)
    svcs = dc.get("services", {})
    result = []
    for name, cfg in svcs.items():
        ports = cfg.get("ports", [])
        image = cfg.get("image", "")
        if not image and "build" in cfg:
            image = f"(build: {cfg['build'].get('context', '.')})"
        depends = cfg.get("depends_on", [])
        if isinstance(depends, dict):
            depends = list(depends.keys())
        volumes = cfg.get("volumes", [])
        healthcheck = cfg.get("healthcheck", {})
        hc_test = healthcheck.get("test", [])
        if isinstance(hc_test, list):
            hc_test = " ".join(hc_test[1:]) if len(hc_test) > 1 else ""
        result.append({
            "name": name,
            "image": image,
            "ports": [str(p) for p in ports],
            "depends_on": depends,
            "volumes": [str(v) for v in volumes][:5],
            "healthcheck": hc_test,
            "restart": cfg.get("restart", ""),
            "profiles": cfg.get("profiles", []),
        })
    volumes = list(dc.get("volumes", {}).keys())
    networks = list(dc.get("networks", {}).keys())
    return {"services": result, "volumes": volumes, "networks": networks}


# ---------------------------------------------------------------------------
# 13. /api/alerts — Prometheus alert rules
# ---------------------------------------------------------------------------
@app.get("/api/alerts")
async def alert_rules():
    rules_path = ROOT / "prometheus" / "alert_rules.yml"
    if not rules_path.exists():
        return {"groups": []}
    data = _load_yaml(rules_path)
    groups = data.get("groups", [])
    result = []
    for g in groups:
        rules = []
        for r in g.get("rules", []):
            rules.append({
                "alert": r.get("alert", ""),
                "expr": r.get("expr", "").strip(),
                "for": r.get("for", ""),
                "severity": r.get("labels", {}).get("severity", ""),
                "summary": r.get("annotations", {}).get("summary", ""),
                "description": r.get("annotations", {}).get("description", ""),
            })
        result.append({"name": g.get("name", ""), "rules": rules})
    return {"groups": result}


# ---------------------------------------------------------------------------
# 14. /api/alertmanager — Alertmanager routing config
# ---------------------------------------------------------------------------
@app.get("/api/alertmanager")
async def alertmanager_config():
    cfg_path = ROOT / "alertmanager" / "config.yml"
    if not cfg_path.exists():
        return {"config": {}}
    data = _load_yaml(cfg_path)
    return {
        "global": data.get("global", {}),
        "route": data.get("route", {}),
        "receivers": data.get("receivers", []),
        "inhibit_rules": data.get("inhibit_rules", []),
    }


# ---------------------------------------------------------------------------
# 15. /api/directives — SOPs listing with content preview
# ---------------------------------------------------------------------------
@app.get("/api/directives")
async def directives():
    if not DIRECTIVES_DIR.exists():
        return {"directives": []}
    result = []
    for p in sorted(DIRECTIVES_DIR.glob("*.md")):
        content = p.read_text(errors="ignore")
        # Extract title from first H1
        title = p.stem
        for line in content.split("\n"):
            if line.startswith("# "):
                title = line[2:].strip()
                break
        result.append({
            "file": p.name,
            "title": title,
            "size_bytes": len(content),
            "lines": content.count("\n") + 1,
            "preview": content[:500],
            "full_content": content,
        })
    return {"directives": result}


# ---------------------------------------------------------------------------
# 16. /api/makefile — all Make targets with descriptions
# ---------------------------------------------------------------------------
@app.get("/api/makefile")
async def makefile_targets():
    mk_path = ROOT / "Makefile"
    if not mk_path.exists():
        return {"targets": []}
    content = mk_path.read_text(errors="ignore")
    targets = []
    lines = content.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_-]*):", line)
        if m:
            target = m.group(1)
            # Look at previous line for comment
            desc = ""
            if i > 0 and lines[i-1].strip().startswith("#"):
                desc = lines[i-1].strip().lstrip("#").strip()
            # Get the command (next non-empty line after target)
            cmd = ""
            for j in range(i+1, min(i+5, len(lines))):
                if lines[j].strip() and lines[j].startswith("\t"):
                    cmd = lines[j].strip()
                    break
            targets.append({"name": target, "description": desc, "command": cmd})
    return {"targets": targets, "total": len(targets)}


# ---------------------------------------------------------------------------
# 17. /api/dependencies — from pyproject.toml
# ---------------------------------------------------------------------------
@app.get("/api/dependencies")
async def dependencies():
    toml_path = ROOT / "pyproject.toml"
    if not toml_path.exists():
        return {"dependencies": []}
    content = toml_path.read_text()
    # Parse dependencies (simple regex-based)
    deps = []
    in_deps = False
    in_optional: str | None = None
    for line in content.split("\n"):
        if line.strip() == "dependencies = [":
            in_deps = True
            continue
        if in_deps:
            if line.strip() == "]":
                in_deps = False
                continue
            m = re.match(r'\s*"([^"]+)"', line)
            if m:
                deps.append({"package": m.group(1), "group": "core"})

    # Optional deps
    for group_match in re.finditer(r'\[project\.optional-dependencies\]\s*\n(\w+)\s*=\s*\[', content):
        pass  # Fall through to simpler parsing
    for section in ["ml", "dev"]:
        pattern = rf'{section}\s*=\s*\[(.*?)\]'
        m = re.search(pattern, content, re.DOTALL)
        if m:
            for pkg in re.findall(r'"([^"]+)"', m.group(1)):
                deps.append({"package": pkg, "group": section})

    # Project metadata
    name_match = re.search(r'name\s*=\s*"([^"]+)"', content)
    version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
    python_match = re.search(r'requires-python\s*=\s*"([^"]+)"', content)

    return {
        "project_name": name_match.group(1) if name_match else "",
        "version": version_match.group(1) if version_match else "",
        "requires_python": python_match.group(1) if python_match else "",
        "dependencies": deps,
        "total": len(deps),
    }


# ---------------------------------------------------------------------------
# 18. /api/prometheus-metrics — Prometheus metric definitions
# ---------------------------------------------------------------------------
@app.get("/api/prometheus-metrics")
async def prometheus_metrics():
    metrics_file = ROOT / "src" / "monitoring" / "metrics.py"
    if not metrics_file.exists():
        return {"metrics": []}
    content = metrics_file.read_text(errors="ignore")
    metrics = []
    # Parse metric definitions
    for m in re.finditer(
        r'(Counter|Histogram|Gauge)\(\s*"([^"]+)",\s*"([^"]+)"(?:,\s*\[([^\]]*)\])?',
        content
    ):
        mtype, mname, mdesc, labels = m.groups()
        label_list = [l.strip().strip('"') for l in labels.split(",")] if labels else []
        metrics.append({
            "name": mname,
            "type": mtype.lower(),
            "description": mdesc,
            "labels": label_list,
        })
    return {"metrics": metrics, "total": len(metrics)}


# ---------------------------------------------------------------------------
# 19. /api/features — feature engineering column inventory
# ---------------------------------------------------------------------------
@app.get("/api/features")
async def features():
    """Enumerate all engineered features from init.sql features_tft table."""
    if not DB_INIT.exists():
        return {"features": []}
    content = DB_INIT.read_text()
    m = re.search(r"CREATE TABLE IF NOT EXISTS features_tft\s*\((.*?)\);", content, re.DOTALL)
    if not m:
        return {"features": []}
    body = m.group(1)
    features_list = []
    current_category = "metadata"
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("--"):
            current_category = stripped.lstrip("- ").strip()
            continue
        if not stripped or stripped.upper().startswith(("PRIMARY", "UNIQUE")):
            continue
        parts = stripped.rstrip(",").split()
        if len(parts) >= 2 and not parts[0].upper().startswith(("PRIMARY", "CONSTRAINT")):
            features_list.append({
                "column": parts[0],
                "type": parts[1],
                "category": current_category,
            })
    return {"features": features_list, "total": len(features_list)}


# ---------------------------------------------------------------------------
# 20. /api/logs — recent log files
# ---------------------------------------------------------------------------
@app.get("/api/logs")
async def logs():
    logs_dir = ROOT / "logs"
    if not logs_dir.exists():
        return {"files": []}
    files = []
    for p in sorted(logs_dir.iterdir()):
        if p.is_file():
            files.append({
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "size_kb": round(p.stat().st_size / 1024, 1),
                "modified": _file_age(p),
            })
    return {"files": files, "total": len(files)}


# ---------------------------------------------------------------------------
# 21. /api/source-tree — src directory structure
# ---------------------------------------------------------------------------
@app.get("/api/source-tree")
async def source_tree():
    """Return src/ directory structure as nested dict."""
    result = {}
    src = ROOT / "src"
    if not src.exists():
        return {"tree": {}}
    for p in sorted(src.rglob("*.py")):
        rel = p.relative_to(src)
        parts = list(rel.parts)
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        fname = parts[-1]
        lines = sum(1 for _ in open(p, errors="ignore"))
        node[fname] = {"lines": lines, "size_kb": round(p.stat().st_size / 1024, 1)}
    return {"tree": result}


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(TEMPLATES_DIR / "awet_dashboard.html")


@app.get("/favicon.ico")
async def favicon():
    return HTMLResponse("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"\n  🚀  AWET Dashboard → http://localhost:{args.port}")
    print(f"  📡  21 API endpoints available\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
