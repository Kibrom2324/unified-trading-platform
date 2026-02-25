#!/usr/bin/env bash
# =============================================================================
# AWET System Startup Script
# =============================================================================
# Starts the complete AWET + SuperAGI stack and waits until healthy.
# Idempotent: safe to run even if services are already up.
#
# Usage:
#   ./scripts/awet_start.sh
#   ./scripts/awet_start.sh --no-wait   # Skip health checks
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
SKIP_WAIT=false
for arg in "$@"; do
    case $arg in
        --no-wait)
            SKIP_WAIT=true
            shift
            ;;
    esac
done

echo "=============================================="
echo "🚀 AWET System Startup"
echo "=============================================="
echo "Repo root: $REPO_ROOT"
echo ""

# Step 1: Start AWET infrastructure (Kafka, TimescaleDB, etc.)
log_info "Starting AWET infrastructure (Kafka, TimescaleDB, Redis, Prometheus, Grafana)..."
make up

# Step 2: Start SuperAGI
log_info "Starting SuperAGI..."
make superagi-up

if [ "$SKIP_WAIT" = true ]; then
    log_warn "Skipping health checks (--no-wait specified)"
    echo ""
    echo "=============================================="
    echo "✅ SYSTEM STARTED (health checks skipped)"
    echo "=============================================="
    exit 0
fi

# Step 3: Wait for services to be healthy
log_info "Waiting for services to become healthy..."

# Function to check if a service is ready
wait_for_service() {
    local name=$1
    local check_cmd=$2
    local max_attempts=${3:-30}
    local attempt=1

    echo -n "  Waiting for $name..."
    while [ $attempt -le $max_attempts ]; do
        if eval "$check_cmd" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    echo -e " ${RED}✗ (timeout after $max_attempts attempts)${NC}"
    return 1
}

# Check TimescaleDB (port 5433)
wait_for_service "TimescaleDB" \
    "docker exec timescaledb pg_isready -U awet" \
    30

# Check Kafka (port 9092) - Confluent images use 'kafka-topics' not 'kafka-topics.sh'
wait_for_service "Kafka" \
    "docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null" \
    30

# Check Redis (port 6379)
wait_for_service "Redis" \
    "docker exec redis redis-cli ping 2>/dev/null | grep -q PONG" \
    15

# Check SuperAGI API (port 8100 - direct backend)
wait_for_service "SuperAGI API" \
    "curl -sf http://localhost:8100/health 2>/dev/null || curl -sf http://localhost:8100/agents/get/project/1 2>/dev/null" \
    45

# Check Ollama (port 11434)
wait_for_service "Ollama" \
    "curl -sf http://localhost:11434/ 2>/dev/null" \
    15

echo ""
log_info "All services are up!"

# Step 4: Verify agents exist
log_info "Verifying SuperAGI agents..."
if [ -n "$SUPERAGI_API_KEY" ] || [ -f ".env" ]; then
    # Source .env if SUPERAGI_API_KEY not set
    if [ -z "$SUPERAGI_API_KEY" ] && [ -f ".env" ]; then
        export $(grep -E '^SUPERAGI_API_KEY=' .env | xargs)
    fi
    
    AGENT_COUNT=$(curl -sf "http://localhost:8100/agents/get/project/1" 2>/dev/null | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
    log_info "Found $AGENT_COUNT agents in SuperAGI"
else
    log_warn "SUPERAGI_API_KEY not set - skipping agent verification"
fi

echo ""
echo "=============================================="
echo -e "${GREEN}✅ SYSTEM READY${NC}"
echo "=============================================="
echo ""
echo "Services:"
echo "  • Grafana:    http://localhost:3000  (admin/admin)"
echo "  • Prometheus: http://localhost:9090"
echo "  • SuperAGI:   http://localhost:3001"
echo "  • Kafka:      localhost:9092"
echo "  • TimescaleDB: localhost:5433"
echo ""
echo "Next steps:"
echo "  • Run health check:  .venv/bin/python scripts/awet_health_check.py"
echo "  • Trigger agent:     .venv/bin/python scripts/schedule_agents.py orchestrator"
echo "  • View logs:         docker compose logs -f"
echo ""
