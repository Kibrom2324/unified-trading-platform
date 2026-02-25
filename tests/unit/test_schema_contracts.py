"""Contract tests for social.reddit.* Avro schemas.

Validates:
  1. Every schema is valid Avro (fastavro parse + round-trip).
  2. Required envelope fields are present.
  3. Required payload fields are present per topic.
  4. schema_version defaults match the filename convention.
  5. DLQ envelope schema is valid and has required DLQ fields.
  6. Trace sub-record has required fields.

Run::

    pytest tests/unit/test_schema_contracts.py -v
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

try:
    import fastavro
    from fastavro.schema import parse_schema
except ImportError:
    pytest.skip("fastavro not installed", allow_module_level=True)

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas" / "avro"

# ── Helpers ──────────────────────────────────────────────────────────────────


def _load(name: str) -> dict[str, Any]:
    path = SCHEMAS_DIR / name
    assert path.exists(), f"Schema file not found: {path}"
    with open(path) as f:
        return json.load(f)


def _field_names(schema: dict[str, Any]) -> set[str]:
    return {f["name"] for f in schema.get("fields", [])}


def _field_by_name(schema: dict[str, Any], name: str) -> dict[str, Any] | None:
    for f in schema.get("fields", []):
        if f["name"] == name:
            return f
    return None


def _nested_field_names(schema: dict[str, Any], record_name: str) -> set[str]:
    """Return field names of a nested record type field."""
    field = _field_by_name(schema, record_name)
    if field is None:
        return set()
    ftype = field["type"]
    if isinstance(ftype, dict) and ftype.get("type") == "record":
        return {f["name"] for f in ftype.get("fields", [])}
    return set()


def _dummy_record(schema: dict[str, Any]) -> dict[str, Any]:
    """Build a minimal valid record for round-trip testing."""
    record: dict[str, Any] = {}
    for field in schema.get("fields", []):
        name = field["name"]
        ftype = field["type"]

        if "default" in field:
            record[name] = field["default"]
            continue

        if isinstance(ftype, list):
            # Union — pick null if available, else first type
            record[name] = None
            continue

        if isinstance(ftype, dict):
            atype = ftype.get("type", "")
            if atype == "record":
                record[name] = _dummy_record(ftype)
                continue
            if atype == "array":
                record[name] = []
                continue
            if atype == "map":
                record[name] = {}
                continue
            if atype == "enum":
                record[name] = ftype["symbols"][0]
                continue
            if ftype.get("logicalType") == "timestamp-millis":
                record[name] = 0
                continue
            record[name] = None
            continue

        _DEFAULTS = {
            "string": "test",
            "int": 0,
            "long": 0,
            "float": 0.0,
            "double": 0.0,
            "boolean": False,
            "bytes": b"",
        }
        record[name] = _DEFAULTS.get(ftype, None)
    return record


# ── Required field sets ──────────────────────────────────────────────────────

ENVELOPE_FIELDS = {
    "event_id",
    "correlation_id",
    "event_version",
    "produced_at_utc",
    "schema_version",
    "source",
    "trace",
    "payload",
}

TRACE_FIELDS = {
    "service_name",
    "host",
    "attempt",
    "parent_event_id",
}

RAW_PAYLOAD_FIELDS = {
    "source_id",
    "subreddit",
    "post_id",
    "comment_id",
    "author_hash",
    "created_at_utc",
    "title",
    "body",
    "url",
    "score",
    "num_comments",
    "content_hash",
    "fetched_at_utc",
}

ENRICHED_PAYLOAD_FIELDS = {
    "source_id",
    "cleaned_text",
    "chunk_ids",
    "entities",
    "tickers",
    "embedding_model",
    "embedding_model_version",
    "content_hash",
}

SUMMARY_PAYLOAD_FIELDS = {
    "source_id",
    "summary",
    "key_points",
    "risks",
    "citations",
    "content_hash",
}

DLQ_FIELDS = {
    "event_id",
    "correlation_id",
    "original_topic",
    "original_value",
    "error_message",
    "error_class",
    "retry_count",
    "max_retries",
    "first_failed_at_utc",
    "last_failed_at_utc",
    "source",
    "schema_version",
}

# ── Schema file list ─────────────────────────────────────────────────────────

SOCIAL_SCHEMAS = [
    "social.reddit.raw.v1.avsc",
    "social.reddit.enriched.v1.avsc",
    "social.reddit.summary.v1.avsc",
]

PAYLOAD_FIELDS_MAP = {
    "social.reddit.raw.v1.avsc": RAW_PAYLOAD_FIELDS,
    "social.reddit.enriched.v1.avsc": ENRICHED_PAYLOAD_FIELDS,
    "social.reddit.summary.v1.avsc": SUMMARY_PAYLOAD_FIELDS,
}

SCHEMA_VERSION_MAP = {
    "social.reddit.raw.v1.avsc": "social.reddit.raw.v1",
    "social.reddit.enriched.v1.avsc": "social.reddit.enriched.v1",
    "social.reddit.summary.v1.avsc": "social.reddit.summary.v1",
}


# ── Tests ────────────────────────────────────────────────────────────────────


class TestSocialRedditSchemas:
    """Contract tests for social.reddit.* Avro schemas."""

    @pytest.mark.parametrize("schema_file", SOCIAL_SCHEMAS)
    def test_schema_parses_as_valid_avro(self, schema_file: str) -> None:
        schema = _load(schema_file)
        parsed = parse_schema(schema)
        assert parsed is not None

    @pytest.mark.parametrize("schema_file", SOCIAL_SCHEMAS)
    def test_round_trip_serialisation(self, schema_file: str) -> None:
        schema = _load(schema_file)
        parsed = parse_schema(schema)
        record = _dummy_record(schema)
        buf = BytesIO()
        fastavro.writer(buf, parsed, [record])
        buf.seek(0)
        rows = list(fastavro.reader(buf))
        assert len(rows) == 1

    @pytest.mark.parametrize("schema_file", SOCIAL_SCHEMAS)
    def test_envelope_fields_present(self, schema_file: str) -> None:
        schema = _load(schema_file)
        names = _field_names(schema)
        missing = ENVELOPE_FIELDS - names
        assert not missing, f"Missing envelope fields: {sorted(missing)}"

    @pytest.mark.parametrize("schema_file", SOCIAL_SCHEMAS)
    def test_trace_fields_present(self, schema_file: str) -> None:
        schema = _load(schema_file)
        trace_names = _nested_field_names(schema, "trace")
        missing = TRACE_FIELDS - trace_names
        assert not missing, f"Missing trace fields: {sorted(missing)}"

    @pytest.mark.parametrize("schema_file", SOCIAL_SCHEMAS)
    def test_payload_fields_present(self, schema_file: str) -> None:
        schema = _load(schema_file)
        expected = PAYLOAD_FIELDS_MAP[schema_file]
        payload_names = _nested_field_names(schema, "payload")
        missing = expected - payload_names
        assert not missing, f"Missing payload fields: {sorted(missing)}"

    @pytest.mark.parametrize("schema_file", SOCIAL_SCHEMAS)
    def test_schema_version_default_matches_filename(
        self, schema_file: str
    ) -> None:
        schema = _load(schema_file)
        sv_field = _field_by_name(schema, "schema_version")
        assert sv_field is not None, "schema_version field missing"
        expected = SCHEMA_VERSION_MAP[schema_file]
        assert sv_field.get("default") == expected, (
            f"schema_version default {sv_field.get('default')!r} "
            f"does not match expected {expected!r}"
        )

    @pytest.mark.parametrize("schema_file", SOCIAL_SCHEMAS)
    def test_source_default_is_reddit(self, schema_file: str) -> None:
        schema = _load(schema_file)
        src_field = _field_by_name(schema, "source")
        assert src_field is not None
        assert src_field.get("default") == "reddit"

    @pytest.mark.parametrize("schema_file", SOCIAL_SCHEMAS)
    def test_correlation_id_is_required_string(self, schema_file: str) -> None:
        schema = _load(schema_file)
        cid_field = _field_by_name(schema, "correlation_id")
        assert cid_field is not None
        assert cid_field["type"] == "string", (
            "correlation_id must be a required string, not nullable"
        )

    @pytest.mark.parametrize("schema_file", SOCIAL_SCHEMAS)
    def test_event_id_is_required_string(self, schema_file: str) -> None:
        schema = _load(schema_file)
        eid_field = _field_by_name(schema, "event_id")
        assert eid_field is not None
        assert eid_field["type"] == "string"


class TestDlqSocialRedditSchema:
    """Contract tests for the DLQ envelope schema."""

    def test_schema_parses_as_valid_avro(self) -> None:
        schema = _load("dlq.social.reddit.v1.avsc")
        parsed = parse_schema(schema)
        assert parsed is not None

    def test_round_trip_serialisation(self) -> None:
        schema = _load("dlq.social.reddit.v1.avsc")
        parsed = parse_schema(schema)
        record = _dummy_record(schema)
        # original_value is bytes — needs actual bytes
        record["original_value"] = b'{"test": true}'
        buf = BytesIO()
        fastavro.writer(buf, parsed, [record])
        buf.seek(0)
        rows = list(fastavro.reader(buf))
        assert len(rows) == 1

    def test_required_dlq_fields_present(self) -> None:
        schema = _load("dlq.social.reddit.v1.avsc")
        names = _field_names(schema)
        missing = DLQ_FIELDS - names
        assert not missing, f"Missing DLQ fields: {sorted(missing)}"

    def test_original_value_is_bytes(self) -> None:
        schema = _load("dlq.social.reddit.v1.avsc")
        ov_field = _field_by_name(schema, "original_value")
        assert ov_field is not None
        assert ov_field["type"] == "bytes"

    def test_schema_version_default(self) -> None:
        schema = _load("dlq.social.reddit.v1.avsc")
        sv_field = _field_by_name(schema, "schema_version")
        assert sv_field is not None
        assert sv_field.get("default") == "dlq.social.reddit.v1"


class TestSchemaBackwardCompatibility:
    """Verify the existing compatibility checker accepts all social.reddit schemas."""

    @pytest.mark.parametrize(
        "schema_file",
        SOCIAL_SCHEMAS + ["dlq.social.reddit.v1.avsc"],
    )
    def test_compatibility_checker_passes(self, schema_file: str) -> None:
        """Runs the same validation logic as check_schema_compatibility.py."""
        import importlib.util
        import sys

        script = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "check_schema_compatibility.py"
        )
        spec = importlib.util.spec_from_file_location(
            "check_schema_compat", script,
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["check_schema_compat"] = mod
        spec.loader.exec_module(mod)

        path = SCHEMAS_DIR / schema_file
        errors = mod.validate_schema(path)
        assert errors == [], f"Compatibility checker errors: {errors}"
