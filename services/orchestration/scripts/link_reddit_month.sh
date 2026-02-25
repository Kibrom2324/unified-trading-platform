#!/usr/bin/env bash
#
# link_reddit_month.sh - Create symlinks for Reddit dump files from SSD
# Usage: ./scripts/link_reddit_month.sh YYYY-MM
# Example: ./scripts/link_reddit_month.sh 2025-07
#

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 YYYY-MM"
    echo "Example: $0 2025-07"
    exit 1
fi

MONTH="$1"

# Validate month format
if ! [[ "$MONTH" =~ ^[0-9]{4}-[0-9]{2}$ ]]; then
    echo "ERROR: Invalid month format. Use YYYY-MM (e.g., 2025-07)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate venv if available
if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

echo "=== Linking Reddit Data for $MONTH ==="
echo

# Get data roots
ROOTS_JSON=$(python "$SCRIPT_DIR/find_data_roots.py" 2>/dev/null) || {
    echo "ERROR: Could not detect SSD."
    exit 1
}

REDDIT_ROOT=$(echo "$ROOTS_JSON" | python -c "import sys,json; print(json.load(sys.stdin).get('reddit_root') or '')")
SSD_ROOT=$(echo "$ROOTS_JSON" | python -c "import sys,json; print(json.load(sys.stdin).get('ssd_root') or '')")

if [[ -z "$REDDIT_ROOT" ]]; then
    echo "WARNING: No reddit_root detected. Searching entire SSD..."
    REDDIT_ROOT="$SSD_ROOT"
fi

if [[ -z "$REDDIT_ROOT" || ! -d "$REDDIT_ROOT" ]]; then
    echo "ERROR: Reddit root not found: $REDDIT_ROOT"
    exit 1
fi

echo "Searching in: $REDDIT_ROOT"
echo

# Search for submission file
RS_FILE=$(find "$REDDIT_ROOT" -name "RS_${MONTH}.zst" -type f 2>/dev/null | head -1)
RC_FILE=$(find "$REDDIT_ROOT" -name "RC_${MONTH}.zst" -type f 2>/dev/null | head -1)

# Also try alternate naming patterns
if [[ -z "$RS_FILE" ]]; then
    RS_FILE=$(find "$REDDIT_ROOT" -iname "*submission*${MONTH}*.zst" -type f 2>/dev/null | head -1)
fi
if [[ -z "$RC_FILE" ]]; then
    RC_FILE=$(find "$REDDIT_ROOT" -iname "*comment*${MONTH}*.zst" -type f 2>/dev/null | head -1)
fi

# Destination directories
DEST_SUBS="/home/kironix/train/reddit/submissions"
DEST_COMM="/home/kironix/train/reddit/comments"

# Create destination directories
mkdir -p "$DEST_SUBS" "$DEST_COMM"

ERRORS=0

# Link submissions
if [[ -n "$RS_FILE" && -f "$RS_FILE" ]]; then
    RS_SIZE=$(ls -lh "$RS_FILE" | awk '{print $5}')
    LINK_PATH="$DEST_SUBS/RS_${MONTH}.zst"

    if [[ -L "$LINK_PATH" ]]; then
        rm "$LINK_PATH"
    fi

    ln -s "$RS_FILE" "$LINK_PATH"
    echo "✓ Submissions: RS_${MONTH}.zst ($RS_SIZE)"
    echo "  Source: $RS_FILE"
    echo "  Link:   $LINK_PATH"
else
    echo "✗ Submissions: RS_${MONTH}.zst NOT FOUND"
    ERRORS=$((ERRORS + 1))
fi

echo

# Link comments
if [[ -n "$RC_FILE" && -f "$RC_FILE" ]]; then
    RC_SIZE=$(ls -lh "$RC_FILE" | awk '{print $5}')
    LINK_PATH="$DEST_COMM/RC_${MONTH}.zst"

    if [[ -L "$LINK_PATH" ]]; then
        rm "$LINK_PATH"
    fi

    ln -s "$RC_FILE" "$LINK_PATH"
    echo "✓ Comments: RC_${MONTH}.zst ($RC_SIZE)"
    echo "  Source: $RC_FILE"
    echo "  Link:   $LINK_PATH"
else
    echo "✗ Comments: RC_${MONTH}.zst NOT FOUND"
    ERRORS=$((ERRORS + 1))
fi

echo

if [[ $ERRORS -eq 0 ]]; then
    echo "=== OK: Both files linked successfully ==="
    echo
    echo "You can now run:"
    echo "  python scripts/ingest_reddit_month.py --month $MONTH --write-db"
else
    echo "=== WARNING: $ERRORS file(s) not found ==="
    echo
    echo "Available .zst files on SSD:"
    find "$REDDIT_ROOT" -name "*.zst" -type f 2>/dev/null | head -20 | while read -r f; do
        echo "  $(basename "$f")"
    done
    exit 1
fi
