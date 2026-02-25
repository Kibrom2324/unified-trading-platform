#!/usr/bin/env bash
# =============================================================================
# AWET Doctor - Comprehensive System Diagnostics
# =============================================================================
# Usage: ./scripts/awet_doctor.sh [--json] [--quiet]
#
# Exit codes:
#   0 = All checks passed
#   1 = Critical failures (system not operational)
#   2 = Warnings (system degraded but functional)
# =============================================================================

# Don't use set -e, we want to continue on failures
set -uo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Flags
JSON_OUTPUT=false
QUIET=false
for arg in "$@"; do
    case $arg in
        --json) JSON_OUTPUT=true ;;
        --quiet|-q) QUIET=true ;;
    esac
done

# Counters
PASSED=0
FAILED=0
WARNED=0

# Kafka UI host port (supports remapped host ports in docker-compose)
KAFKA_UI_PORT="${KAFKA_UI_PORT:-8080}"

# Results array for JSON
declare -a RESULTS=()

log() {
    if [ "$QUIET" = false ]; then
        echo -e "$1"
    fi
}

check_pass() {
    ((PASSED++))
    RESULTS+=("{\"check\":\"$1\",\"status\":\"pass\",\"message\":\"$2\"}")
    log "${GREEN}✅ $1${NC}: $2"
}

check_fail() {
    ((FAILED++))
    RESULTS+=("{\"check\":\"$1\",\"status\":\"fail\",\"message\":\"$2\"}")
    log "${RED}❌ $1${NC}: $2"
}

check_warn() {
    ((WARNED++))
    RESULTS+=("{\"check\":\"$1\",\"status\":\"warn\",\"message\":\"$2\"}")
    log "${YELLOW}⚠️  $1${NC}: $2"
}

section() {
    if [ "$QUIET" = false ]; then
        echo ""
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}  $1${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    fi
}

# =============================================================================
# HEADER
# =============================================================================
if [ "$JSON_OUTPUT" = false ] && [ "$QUIET" = false ]; then
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║           🩺 AWET SYSTEM DOCTOR                           ║"
    echo "║           $(date '+%Y-%m-%d %H:%M:%S')                           ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
fi

# =============================================================================
# 1. DOCKER SERVICES
# =============================================================================
section "1. Docker Services"

# Check Docker is running
if ! docker info &>/dev/null; then
    check_fail "Docker" "Docker daemon not running"
else
    check_pass "Docker" "Docker daemon running"
fi

# Core services
CORE_SERVICES=(kafka schema-registry timescaledb redis prometheus grafana alertmanager kafka-ui)
for svc in "${CORE_SERVICES[@]}"; do
    status=$(docker inspect -f '{{.State.Status}}' "$svc" 2>/dev/null || echo "not_found")
    if [ "$status" = "running" ]; then
        check_pass "$svc" "running"
    elif [ "$status" = "not_found" ]; then
        check_fail "$svc" "container not found"
    else
        check_fail "$svc" "status: $status"
    fi
done

# SuperAGI services (optional)
SUPERAGI_SERVICES=(superagi-backend superagi-celery superagi-gui superagi-proxy)
for svc in "${SUPERAGI_SERVICES[@]}"; do
    status=$(docker inspect -f '{{.State.Status}}' "$svc" 2>/dev/null || echo "not_found")
    if [ "$status" = "running" ]; then
        check_pass "$svc" "running"
    elif [ "$status" = "not_found" ]; then
        check_warn "$svc" "not started (optional)"
    else
        check_warn "$svc" "status: $status"
    fi
done

# Detect mapped Kafka UI host port when available
kafka_ui_port_from_docker=$(docker port kafka-ui 8080/tcp 2>/dev/null | head -1 | awk -F: '{print $NF}' || true)
if [ -n "${kafka_ui_port_from_docker:-}" ]; then
    KAFKA_UI_PORT="$kafka_ui_port_from_docker"
fi

# =============================================================================
# 2. PORT USAGE
# =============================================================================
section "2. Port Bindings"

declare -A EXPECTED_PORTS=(
    [3000]="Grafana"
    [3001]="SuperAGI"
    [5433]="TimescaleDB"
    [6379]="Redis"
    ["$KAFKA_UI_PORT"]="Kafka-UI"
    [8081]="Schema-Registry"
    [8100]="SuperAGI-API"
    [9090]="Prometheus"
    [9092]="Kafka"
    [9093]="Alertmanager"
    [11434]="Ollama"
)

for port in "${!EXPECTED_PORTS[@]}"; do
    svc="${EXPECTED_PORTS[$port]}"
    if ss -lntp 2>/dev/null | grep -q ":${port} "; then
        check_pass "Port $port" "$svc listening"
    else
        if [ "$svc" = "Ollama" ] || [ "$svc" = "SuperAGI" ] || [ "$svc" = "SuperAGI-API" ]; then
            check_warn "Port $port" "$svc not listening (optional)"
        else
            check_fail "Port $port" "$svc not listening"
        fi
    fi
done

# =============================================================================
# 3. DATABASE CONNECTIVITY
# =============================================================================
section "3. Database (TimescaleDB)"

if command -v psql &>/dev/null; then
    DB_URL="postgresql://awet:awet@localhost:5433/awet"
    
    # Connection test
    if psql "$DB_URL" -c "SELECT 1" &>/dev/null; then
        check_pass "DB Connection" "Connected to awet database"
        
        # Table counts
        for table in market_raw_minute features_tft trades positions; do
            count=$(psql "$DB_URL" -t -c "SELECT COUNT(*) FROM $table" 2>/dev/null | tr -d ' ')
            if [ -n "$count" ] && [ "$count" -gt 0 ]; then
                check_pass "Table $table" "$count rows"
            else
                check_warn "Table $table" "empty or missing"
            fi
        done
    else
        check_fail "DB Connection" "Cannot connect to TimescaleDB"
    fi
else
    check_warn "psql" "psql not installed, skipping DB checks"
fi

# =============================================================================
# 4. KAFKA
# =============================================================================
section "4. Kafka"

if docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list &>/dev/null; then
    topics=$(docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null | wc -l)
    check_pass "Kafka Topics" "$topics topics found"
    
    # Check key topics
    for topic in market.raw market.engineered predictions.tft risk.approved execution.completed; do
        if docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null | grep -q "^${topic}$"; then
            check_pass "Topic $topic" "exists"
        else
            check_warn "Topic $topic" "not found"
        fi
    done
    
    # Consumer lag (simplified)
    lag_output=$(docker exec kafka kafka-consumer-groups --bootstrap-server localhost:9092 --all-groups --describe 2>/dev/null | grep -v "^$" | head -20)
    if [ -n "$lag_output" ]; then
        check_pass "Consumer Groups" "found consumer groups"
    else
        check_warn "Consumer Groups" "no consumer groups (may be normal)"
    fi
else
    check_fail "Kafka" "Cannot connect to Kafka broker"
fi

# =============================================================================
# 5. HEALTH ENDPOINTS
# =============================================================================
section "5. HTTP Health Endpoints"

declare -A HEALTH_ENDPOINTS=(
    ["Prometheus"]="http://localhost:9090/-/healthy"
    ["Grafana"]="http://localhost:3000/api/health"
    ["Alertmanager"]="http://localhost:9093/-/healthy"
    ["Schema-Registry"]="http://localhost:8081/subjects"
    ["Kafka-UI"]="http://localhost:${KAFKA_UI_PORT}/api/clusters"
    ["SuperAGI"]="http://localhost:8100/health"
    ["Ollama"]="http://localhost:11434/api/tags"
)

for svc in "${!HEALTH_ENDPOINTS[@]}"; do
    url="${HEALTH_ENDPOINTS[$svc]}"
    if curl -sf --max-time 5 "$url" &>/dev/null; then
        check_pass "$svc Health" "responding"
    else
        if [ "$svc" = "SuperAGI" ] || [ "$svc" = "Ollama" ]; then
            check_warn "$svc Health" "not responding (optional)"
        else
            check_fail "$svc Health" "not responding at $url"
        fi
    fi
done

# Agent ports (only if running locally)
for port in 8001 8002 8003 8004 8005 8006; do
    if curl -sf --max-time 2 "http://localhost:$port/health" &>/dev/null; then
        agent=$(curl -sf "http://localhost:$port/health" 2>/dev/null | grep -o '"agent":"[^"]*"' | cut -d'"' -f4)
        check_pass "Agent :$port" "${agent:-running}"
    fi
done

# =============================================================================
# 6. MODEL REGISTRY
# =============================================================================
section "6. Model Registry"

REGISTRY_FILE="models/registry.json"
if [ -f "$REGISTRY_FILE" ]; then
    models_count=$(jq '.models | length' "$REGISTRY_FILE" 2>/dev/null || echo "0")
    green_model=$(jq -r '.models[] | select(.status=="green") | .run_id' "$REGISTRY_FILE" 2>/dev/null | head -1)
    
    if [ "$models_count" -gt 0 ]; then
        check_pass "Model Registry" "$models_count models registered"
    else
        check_warn "Model Registry" "no models registered"
    fi
    
    if [ -n "$green_model" ]; then
        check_pass "Production Model" "$green_model"
    else
        check_warn "Production Model" "no green model (run training)"
    fi
else
    check_warn "Model Registry" "registry.json not found"
fi

# =============================================================================
# 7. APPROVAL GATE
# =============================================================================
section "7. Trading Safety"

APPROVAL_FILE=".tmp/APPROVE_EXECUTION"
if [ -f "$APPROVAL_FILE" ]; then
    check_warn "Approval Gate" "ENABLED - trades can execute"
else
    check_pass "Approval Gate" "DISABLED - trades blocked (safe mode)"
fi

# Check dry_run setting
if grep -q "execution_dry_run: true" config/app.yaml 2>/dev/null; then
    check_pass "Dry Run" "enabled - all trades blocked"
elif grep -q "execution_dry_run: false" config/app.yaml 2>/dev/null; then
    check_warn "Dry Run" "disabled - paper trades allowed"
else
    check_warn "Dry Run" "setting not found in config"
fi

# =============================================================================
# 8. DISK SPACE
# =============================================================================
section "8. Disk Space"

disk_usage=$(df -h . | tail -1 | awk '{print $5}' | tr -d '%')
disk_avail=$(df -h . | tail -1 | awk '{print $4}')

if [ "$disk_usage" -lt 80 ]; then
    check_pass "Disk Space" "${disk_avail} available (${disk_usage}% used)"
elif [ "$disk_usage" -lt 90 ]; then
    check_warn "Disk Space" "${disk_avail} available (${disk_usage}% used)"
else
    check_fail "Disk Space" "CRITICAL: ${disk_avail} available (${disk_usage}% used)"
fi

# =============================================================================
# 9. PORTAL URLs
# =============================================================================
section "9. Portal URLs"

if [ "$QUIET" = false ] && [ "$JSON_OUTPUT" = false ]; then
    echo ""
    echo "  📊 Grafana:        http://localhost:3000  (admin/admin)"
    echo "  📈 Prometheus:     http://localhost:9090"
    echo "  🚨 Alertmanager:   http://localhost:9093"
    echo "  📬 Kafka UI:       http://localhost:${KAFKA_UI_PORT}"
    echo "  📋 Schema Registry: http://localhost:8081"
    echo "  🤖 SuperAGI:       http://localhost:3001"
    echo "  🧠 Ollama:         http://localhost:11434"
    echo "  🗄️  TimescaleDB:    postgresql://awet:awet@localhost:5433/awet"
    echo ""
fi

# =============================================================================
# SUMMARY
# =============================================================================
section "Summary"

TOTAL=$((PASSED + FAILED + WARNED))

if [ "$JSON_OUTPUT" = true ]; then
    echo "{"
    echo "  \"timestamp\": \"$(date -Iseconds)\","
    echo "  \"passed\": $PASSED,"
    echo "  \"failed\": $FAILED,"
    echo "  \"warned\": $WARNED,"
    echo "  \"total\": $TOTAL,"
    echo "  \"status\": \"$([ $FAILED -eq 0 ] && echo 'healthy' || echo 'unhealthy')\","
    echo "  \"checks\": ["
    printf '%s\n' "${RESULTS[@]}" | sed 's/$/,/' | sed '$ s/,$//'
    echo "  ]"
    echo "}"
else
    echo ""
    echo -e "  ${GREEN}Passed:${NC}  $PASSED"
    echo -e "  ${RED}Failed:${NC}  $FAILED"
    echo -e "  ${YELLOW}Warned:${NC}  $WARNED"
    echo "  Total:   $TOTAL"
    echo ""
    
    if [ $FAILED -eq 0 ] && [ $WARNED -eq 0 ]; then
        echo -e "  ${GREEN}🎉 System is HEALTHY${NC}"
    elif [ $FAILED -eq 0 ]; then
        echo -e "  ${YELLOW}⚠️  System is DEGRADED (warnings only)${NC}"
    else
        echo -e "  ${RED}❌ System is UNHEALTHY${NC}"
    fi
    echo ""
fi

# Exit code
if [ $FAILED -gt 0 ]; then
    exit 1
elif [ $WARNED -gt 0 ]; then
    exit 2
else
    exit 0
fi
