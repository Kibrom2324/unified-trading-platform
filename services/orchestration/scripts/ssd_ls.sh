#!/usr/bin/env bash
#
# ssd_ls.sh - List contents of external SSD data folders
# Usage: ./scripts/ssd_ls.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate venv if available
if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

echo "=== Detecting SSD Data Roots ==="
echo

# Get data roots as JSON
ROOTS_JSON=$(python "$SCRIPT_DIR/find_data_roots.py" 2>/dev/null) || {
    echo "ERROR: Could not detect SSD. Check if external drive is mounted."
    echo "Searched: /media/\$USER/*, /run/media/\$USER/*, /mnt/*"
    exit 1
}

SSD_ROOT=$(echo "$ROOTS_JSON" | python -c "import sys,json; print(json.load(sys.stdin).get('ssd_root') or '')")
POLYGON_ROOT=$(echo "$ROOTS_JSON" | python -c "import sys,json; print(json.load(sys.stdin).get('polygon_root') or '')")
REDDIT_ROOT=$(echo "$ROOTS_JSON" | python -c "import sys,json; print(json.load(sys.stdin).get('reddit_root') or '')")

echo "SSD Root:     $SSD_ROOT"
echo "Polygon Root: ${POLYGON_ROOT:-'(not found)'}"
echo "Reddit Root:  ${REDDIT_ROOT:-'(not found)'}"
echo

# List Polygon data
if [[ -n "$POLYGON_ROOT" && -d "$POLYGON_ROOT" ]]; then
    echo "=== Polygon Data ==="
    echo "Location: $POLYGON_ROOT"
    echo

    DAY_AGG="$POLYGON_ROOT/Day Aggregates"
    if [[ -d "$DAY_AGG" ]]; then
        echo "Day Aggregates by month:"
        for month_dir in "$DAY_AGG"/*/; do
            if [[ -d "$month_dir" ]]; then
                month_name=$(basename "$month_dir")
                file_count=$(find "$month_dir" -maxdepth 1 -name "*.csv.gz" 2>/dev/null | wc -l)
                printf "  %-15s %3d files\n" "$month_name" "$file_count"
            fi
        done
    else
        echo "  (No 'Day Aggregates' folder found)"
        echo "  Top-level contents:"
        ls -1 "$POLYGON_ROOT" 2>/dev/null | head -20 | sed 's/^/    /'
    fi
    echo
fi

# List Reddit data
if [[ -n "$REDDIT_ROOT" && -d "$REDDIT_ROOT" ]]; then
    echo "=== Reddit Data ==="
    echo "Location: $REDDIT_ROOT"
    echo

    # Check for submissions/comments structure
    if [[ -d "$REDDIT_ROOT/submissions" ]]; then
        echo "Submissions:"
        find "$REDDIT_ROOT/submissions" -name "*.zst" -exec ls -lh {} \; 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
    fi

    if [[ -d "$REDDIT_ROOT/comments" ]]; then
        echo "Comments:"
        find "$REDDIT_ROOT/comments" -name "*.zst" -exec ls -lh {} \; 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
    fi

    # Also check for flat .zst files
    ZST_FILES=$(find "$REDDIT_ROOT" -maxdepth 2 -name "*.zst" 2>/dev/null | head -20)
    if [[ -n "$ZST_FILES" ]]; then
        echo "Found .zst files:"
        echo "$ZST_FILES" | while read -r f; do
            size=$(ls -lh "$f" 2>/dev/null | awk '{print $5}')
            echo "  $(basename "$f") ($size)"
        done
    fi
    echo
fi

# Show local train folder status
echo "=== Local Train Folder Status ==="
LOCAL_TRAIN="/home/kironix/train"
if [[ -d "$LOCAL_TRAIN" ]]; then
    echo "Location: $LOCAL_TRAIN"
    echo
    if [[ -d "$LOCAL_TRAIN/poligon/Day Aggregates" ]]; then
        echo "Polygon (local):"
        for month_dir in "$LOCAL_TRAIN/poligon/Day Aggregates"/*/; do
            if [[ -d "$month_dir" ]]; then
                month_name=$(basename "$month_dir")
                file_count=$(find "$month_dir" -maxdepth 1 -name "*.csv.gz" 2>/dev/null | wc -l)
                printf "  %-15s %3d files\n" "$month_name" "$file_count"
            fi
        done
    fi
    echo
    if [[ -d "$LOCAL_TRAIN/reddit" ]]; then
        echo "Reddit (local):"
        find "$LOCAL_TRAIN/reddit" -name "*.zst" -exec ls -lh {} \; 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
    fi
else
    echo "  (not found at $LOCAL_TRAIN)"
fi

echo
echo "Done."
