#!/usr/bin/env python3
"""Check Avro schema compatibility (forward & backward).

Validates that every *.avsc file under ``--schemas-dir`` is:
  1. Valid Avro JSON according to ``fastavro``.
  2. Self-consistent (can parse/round-trip a dummy record).
  3. Backward-compatible with the previous version if one exists
     (naming convention: ``<topic>.v<N>.avsc``).

Exit codes:
  0 — all schemas pass
  1 — at least one schema has errors

Usage:
    python scripts/check_schema_compatibility.py --schemas-dir schemas/avro
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path

try:
    import fastavro
    from fastavro.schema import parse_schema
except ImportError:
    sys.exit("fastavro is required: pip install fastavro")


REQUIRED_ENVELOPE_FIELDS = {
    "event_id",
    "correlation_id",
    "schema_version",
    "source",
}


def _load_schema(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _field_names(schema: dict) -> set[str]:
    return {f["name"] for f in schema.get("fields", [])}


def validate_schema(path: Path) -> list[str]:
    """Return a list of error strings (empty = ok)."""
    errors: list[str] = []

    # ---- 1) JSON parse -------------------------------------------------------
    try:
        schema = _load_schema(path)
    except json.JSONDecodeError as exc:
        return [f"Invalid JSON: {exc}"]

    # ---- 2) fastavro parse ---------------------------------------------------
    try:
        parsed = parse_schema(schema)
    except Exception as exc:  # noqa: BLE001
        return [f"fastavro parse error: {exc}"]

    # ---- 3) Round-trip with dummy data ---------------------------------------
    try:
        # Build a minimal record from field defaults / type heuristics
        record = _dummy_record(schema)
        buf = BytesIO()
        fastavro.writer(buf, parsed, [record])
        buf.seek(0)
        rows = list(fastavro.reader(buf))
        if len(rows) != 1:
            errors.append(f"Round-trip returned {len(rows)} rows, expected 1")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Round-trip failed: {exc}")

    # ---- 4) Envelope fields present -----------------------------------------
    names = _field_names(schema)
    missing = REQUIRED_ENVELOPE_FIELDS - names
    if missing:
        errors.append(f"Missing required envelope fields: {sorted(missing)}")

    return errors


def _dummy_record(schema: dict) -> dict:  # noqa: C901 — simple heuristic
    """Produce a minimal valid record for round-trip testing."""
    record: dict = {}
    for field in schema.get("fields", []):
        ftype = field["type"]
        name = field["name"]

        if "default" in field:
            record[name] = field["default"]
            continue

        # Union ["null", X] → None
        if isinstance(ftype, list):
            record[name] = None
            continue

        if isinstance(ftype, dict):
            logical = ftype.get("logicalType", "")
            atype = ftype.get("type", "")
            if logical == "timestamp-millis":
                record[name] = 0
            elif atype == "record":
                record[name] = _dummy_record(ftype)
            elif atype == "map":
                record[name] = {}
            elif atype == "array":
                record[name] = []
            elif atype == "enum":
                record[name] = ftype["symbols"][0]
            else:
                record[name] = None
            continue

        # Primitive types
        _DEFAULTS = {
            "string": "",
            "int": 0,
            "long": 0,
            "float": 0.0,
            "double": 0.0,
            "boolean": False,
            "bytes": b"",
        }
        record[name] = _DEFAULTS.get(ftype, None)
    return record


_VERSION_RE = re.compile(r"\.v(\d+)\.avsc$")


def group_by_topic(paths: list[Path]) -> dict[str, list[tuple[int, Path]]]:
    """Group schemas by topic name, sorted by version."""
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in paths:
        m = _VERSION_RE.search(p.name)
        if m:
            version = int(m.group(1))
            topic = p.name[: m.start()]
        else:
            version = 1
            topic = p.stem
        groups[topic].append((version, p))
    for v in groups.values():
        v.sort()
    return groups


def check_backward_compat(old_path: Path, new_path: Path) -> list[str]:
    """Minimal backward-compatibility check between two schema versions.

    Checks:
    - No required field was *removed* (would break readers still on old schema).
    - Newly added fields have a default value.
    """
    errors: list[str] = []
    old = _load_schema(old_path)
    new = _load_schema(new_path)

    old_fields = {f["name"]: f for f in old.get("fields", [])}
    new_fields = {f["name"]: f for f in new.get("fields", [])}

    # Removed fields
    for name in old_fields:
        if name not in new_fields:
            errors.append(f"Field '{name}' removed (breaks backward compat)")

    # New fields without default
    for name, field in new_fields.items():
        if name not in old_fields and "default" not in field:
            # Union with null first element counts as having a default
            ftype = field.get("type")
            if isinstance(ftype, list) and ftype and ftype[0] == "null":
                continue
            errors.append(
                f"New field '{name}' has no default (breaks backward compat)"
            )

    return errors


def _check_one_dir(schema_dir: Path) -> int:
    """Validate all .avsc files in a single directory. Returns error count."""
    if not schema_dir.is_dir():
        print(f"ERROR: {schema_dir} is not a directory")
        return 1

    avsc_files = sorted(schema_dir.glob("*.avsc"))
    if not avsc_files:
        print(f"No .avsc files found in {schema_dir}")
        return 1

    print(f"Checking {len(avsc_files)} schema(s) in {schema_dir}/\n")

    total_errors = 0

    # Per-file validation
    for path in avsc_files:
        errors = validate_schema(path)
        status = "PASS" if not errors else "FAIL"
        print(f"  [{status}] {path.name}")
        for e in errors:
            print(f"         ↳ {e}")
            total_errors += 1

    # Cross-version compatibility
    groups = group_by_topic(avsc_files)
    print()
    for topic, versions in groups.items():
        if len(versions) < 2:
            continue
        for i in range(1, len(versions)):
            old_ver, old_path = versions[i - 1]
            new_ver, new_path = versions[i]
            compat_errors = check_backward_compat(old_path, new_path)
            status = "PASS" if not compat_errors else "FAIL"
            print(f"  [{status}] {topic} v{old_ver} → v{new_ver}")
            for e in compat_errors:
                print(f"         ↳ {e}")
                total_errors += 1

    return total_errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schemas-dir",
        nargs="+",
        type=Path,
        default=[Path("schemas/avro")],
        help="One or more directories containing .avsc files (default: schemas/avro)",
    )
    args = parser.parse_args()

    grand_total = 0
    for schema_dir in args.schemas_dir:
        grand_total += _check_one_dir(schema_dir)

    print()
    if grand_total:
        print(f"FAILED: {grand_total} error(s)")
        return 1

    print("ALL SCHEMAS PASS ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
