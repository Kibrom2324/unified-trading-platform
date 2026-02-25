"""Verify no script bypasses centralized logging.

This test scans the codebase for direct ``structlog.configure()`` calls
outside of ``src/core/logging.py``.  Any match is a violation because it
skips ``add_correlation_id`` and ``merge_contextvars`` processors.
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PATTERN = re.compile(r"\bstructlog\.configure\s*\(")
ALLOWED = {PROJECT_ROOT / "src" / "core" / "logging.py"}


def _scan_dirs() -> list[str]:
    """Return list of 'file:line' strings that violate the rule."""
    violations: list[str] = []
    for directory in ("scripts", "execution", "src"):
        base = PROJECT_ROOT / directory
        if not base.is_dir():
            continue
        for py in base.rglob("*.py"):
            if py in ALLOWED:
                continue
            for lineno, line in enumerate(py.read_text().splitlines(), start=1):
                if PATTERN.search(line):
                    violations.append(f"{py.relative_to(PROJECT_ROOT)}:{lineno}")
    return violations


def test_no_direct_structlog_configure():
    """Ensure nobody calls structlog.configure() outside src/core/logging.py."""
    violations = _scan_dirs()
    assert violations == [], (
        "Direct structlog.configure() calls found (use "
        "src.core.logging.configure_logging instead):\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_configure_logging_has_console_param():
    """The centralized configure_logging must accept a ``console`` kwarg."""
    import inspect

    from src.core.logging import configure_logging

    sig = inspect.signature(configure_logging)
    assert "console" in sig.parameters, "configure_logging must accept 'console' kwarg"
    assert sig.parameters["console"].default is False, "console should default to False"
