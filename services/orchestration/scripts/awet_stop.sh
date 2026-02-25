#!/usr/bin/env bash
# =============================================================================
# AWET System Shutdown Script
# =============================================================================
# Cleanly stops the complete AWET + SuperAGI stack.
#
# Usage:
#   ./scripts/awet_stop.sh          # Stop all services
#   ./scripts/awet_stop.sh --clean  # Stop and remove volumes (DESTRUCTIVE)
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

# Parse arguments
CLEAN_VOLUMES=false
for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN_VOLUMES=true
            shift
            ;;
    esac
done

echo "=============================================="
echo "🛑 AWET System Shutdown"
echo "=============================================="
echo ""

# Step 1: Stop SuperAGI services
log_info "Stopping SuperAGI services..."
make superagi-down 2>/dev/null || log_warn "SuperAGI services may not have been running"

# Step 2: Stop AWET infrastructure
log_info "Stopping AWET infrastructure..."
if [ "$CLEAN_VOLUMES" = true ]; then
    log_warn "Removing volumes (--clean specified)..."
    docker compose down -v
else
    docker compose down
fi

# Step 3: Verify everything is stopped
log_info "Verifying shutdown..."
RUNNING=$(docker compose ps -q 2>/dev/null | wc -l)
if [ "$RUNNING" -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo -e "${GREEN}✅ SYSTEM STOPPED${NC}"
    echo "=============================================="
else
    log_warn "$RUNNING containers still running. You may need to run: docker compose down --remove-orphans"
fi

if [ "$CLEAN_VOLUMES" = true ]; then
    echo ""
    log_warn "Volumes have been removed. Data will be reset on next start."
fi
echo ""
