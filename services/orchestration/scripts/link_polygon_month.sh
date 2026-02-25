#!/usr/bin/env bash
#
# link_polygon_month.sh - Create symlinks for Polygon data from SSD
# Usage: ./scripts/link_polygon_month.sh YYYY-MM
# Example: ./scripts/link_polygon_month.sh 2025-01
#

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 YYYY-MM"
    echo "Example: $0 2025-01"
    exit 1
fi

MONTH="$1"

# Validate month format
if ! [[ "$MONTH" =~ ^[0-9]{4}-[0-9]{2}$ ]]; then
    echo "ERROR: Invalid month format. Use YYYY-MM (e.g., 2025-01)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate venv if available
if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Extract year and month number
YEAR="${MONTH:0:4}"
MONTH_NUM="${MONTH:5:2}"

# Map month number to name
declare -A MONTH_NAMES=(
    [01]="January"   [02]="February"  [03]="March"
    [04]="April"     [05]="May"       [06]="June"
    [07]="July"      [08]="August"    [09]="September"
    [10]="October"   [11]="November"  [12]="December"
)

MONTH_NAME="${MONTH_NAMES[$MONTH_NUM]:-}"
if [[ -z "$MONTH_NAME" ]]; then
    echo "ERROR: Invalid month number: $MONTH_NUM"
    exit 1
fi

echo "=== Linking Polygon Data for $MONTH ($MONTH_NAME $YEAR) ==="
echo

# Get data roots
ROOTS_JSON=$(python "$SCRIPT_DIR/find_data_roots.py" 2>/dev/null) || {
    echo "ERROR: Could not detect SSD."
    exit 1
}

POLYGON_ROOT=$(echo "$ROOTS_JSON" | python -c "import sys,json; print(json.load(sys.stdin).get('polygon_root') or '')")
SSD_ROOT=$(echo "$ROOTS_JSON" | python -c "import sys,json; print(json.load(sys.stdin).get('ssd_root') or '')")

if [[ -z "$POLYGON_ROOT" ]]; then
    echo "WARNING: No polygon_root detected. Searching entire SSD..."
    POLYGON_ROOT="$SSD_ROOT"
fi

if [[ -z "$POLYGON_ROOT" || ! -d "$POLYGON_ROOT" ]]; then
    echo "ERROR: Polygon root not found: $POLYGON_ROOT"
    exit 1
fi

echo "Polygon root: $POLYGON_ROOT"

# Find the month folder
DAY_AGG="$POLYGON_ROOT/Day Aggregates"
SOURCE_DIR=""

# Try exact month name match first
if [[ -d "$DAY_AGG/$MONTH_NAME" ]]; then
    SOURCE_DIR="$DAY_AGG/$MONTH_NAME"
elif [[ -d "$DAY_AGG/${MONTH_NAME,,}" ]]; then
    SOURCE_DIR="$DAY_AGG/${MONTH_NAME,,}"
else
    # Search for folder containing files with YYYY-MM pattern
    for dir in "$DAY_AGG"/*/; do
        if [[ -d "$dir" ]]; then
            # Check if any file matches the date pattern
            if find "$dir" -maxdepth 1 -name "${MONTH}-*.csv.gz" -type f 2>/dev/null | head -1 | grep -q .; then
                SOURCE_DIR="$dir"
                break
            fi
        fi
    done
fi

if [[ -z "$SOURCE_DIR" || ! -d "$SOURCE_DIR" ]]; then
    echo "ERROR: Could not find folder for $MONTH_NAME in $DAY_AGG"
    echo
    echo "Available month folders:"
    ls -1 "$DAY_AGG" 2>/dev/null | sed 's/^/  /'
    exit 1
fi

# Count files
FILE_COUNT=$(find "$SOURCE_DIR" -maxdepth 1 -name "*.csv.gz" -type f 2>/dev/null | wc -l)

echo "Source:     $SOURCE_DIR"
echo "Files:      $FILE_COUNT .csv.gz files"
echo

# Create local destination
DEST_BASE="/home/kironix/train/poligon/day_aggs"
DEST_LINK="$DEST_BASE/$MONTH"

mkdir -p "$DEST_BASE"

# Remove existing link if present
if [[ -L "$DEST_LINK" ]]; then
    rm "$DEST_LINK"
elif [[ -e "$DEST_LINK" ]]; then
    echo "WARNING: $DEST_LINK exists and is not a symlink. Skipping."
    exit 1
fi

# Create symlink
ln -s "$SOURCE_DIR" "$DEST_LINK"

echo "✓ Created symlink:"
echo "  $DEST_LINK -> $SOURCE_DIR"
echo
echo "=== OK: $FILE_COUNT files linked ==="
echo
echo "You can now access data at:"
echo "  $DEST_LINK/*.csv.gz"
