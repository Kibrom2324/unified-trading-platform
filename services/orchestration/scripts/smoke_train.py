#!/usr/bin/env python3
"""
Smoke test for model training.

Trains a small model and verifies registry update.

Usage:
    python scripts/smoke_train.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"


def get_model_count() -> int:
    """Count models in registry."""
    if not REGISTRY_PATH.exists():
        return 0
    registry = json.loads(REGISTRY_PATH.read_text())
    return len(registry.get("models", []))


def main() -> int:
    print("=" * 60)
    print("SMOKE TEST: Model Training")
    print("=" * 60)

    # Get initial model count
    initial_count = get_model_count()
    print(f"\n[1/2] Initial model count: {initial_count}")

    # Run training with minimal epochs
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_tft_baseline.py"),
        "--max-epochs", "2",
        "--batch-size", "32",
        "--hidden-size", "16",
        "--attention-heads", "1",
        "--symbols", "AAPL",
    ]

    print(f"\n[2/2] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)

    # Print output
    if result.stdout:
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

    if result.returncode != 0:
        print(f"❌ Training failed (exit {result.returncode})")
        if result.stderr:
            print(result.stderr[-1000:])
        return 1

    # Verify registry updated
    final_count = get_model_count()
    if final_count > initial_count:
        print(f"\n✅ SMOKE_TRAIN_PASS: Registry updated ({initial_count} → {final_count} models)")
        return 0
    else:
        print(f"\n❌ SMOKE_TRAIN_FAIL: Registry not updated (still {final_count} models)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
