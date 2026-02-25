#!/usr/bin/env python3
"""
Detect external SSD mountpoint and locate polygon/reddit data roots.
Outputs JSON: { "ssd_root": "...", "polygon_root": "...", "reddit_root": "..." }
"""

import json
import os
import sys
from pathlib import Path


def find_mount_candidates() -> list[Path]:
    """Scan common mount locations for external drives AND local train folder."""
    user = os.environ.get("USER", os.environ.get("LOGNAME", ""))
    candidates = []

    # Check local train folder FIRST (highest priority)
    local_train = Path(f"/home/{user}/train")
    if local_train.exists() and local_train.is_dir():
        candidates.append(local_train)

    # Then check external mount points
    search_roots = [
        Path(f"/media/{user}"),
        Path(f"/run/media/{user}"),
        Path("/mnt"),
    ]

    for root in search_roots:
        if root.exists() and root.is_dir():
            try:
                for child in root.iterdir():
                    if child.is_dir() and not child.name.startswith("."):
                        candidates.append(child)
            except PermissionError:
                continue

    return candidates


def score_candidate(path: Path) -> tuple[int, dict]:
    """
    Score a mount candidate based on presence of polygon/reddit data.
    Returns (score, {"polygon_root": ..., "reddit_root": ...})
    """
    score = 0
    roots = {"polygon_root": None, "reddit_root": None}

    # Search for polygon/poligon folders (up to 3 levels deep)
    for pattern in ["**/poligon", "**/polygon", "**/Polygon", "**/Poligon"]:
        try:
            for match in path.glob(pattern):
                if match.is_dir():
                    # Check for Day Aggregates subfolder
                    day_agg = match / "Day Aggregates"
                    if day_agg.exists():
                        roots["polygon_root"] = str(match)
                        score += 10
                        break
                    elif any(match.glob("*.csv.gz")) or any(match.glob("*/*.csv.gz")):
                        roots["polygon_root"] = str(match)
                        score += 5
                        break
        except (PermissionError, OSError):
            continue
        if roots["polygon_root"]:
            break

    # Search for reddit folders
    for pattern in ["**/reddit", "**/Reddit"]:
        try:
            for match in path.glob(pattern):
                if match.is_dir():
                    # Check for .zst files
                    has_zst = (
                        any(match.glob("**/*.zst"))
                        or (match / "submissions").exists()
                        or (match / "comments").exists()
                    )
                    if has_zst or any(match.iterdir()):
                        roots["reddit_root"] = str(match)
                        score += 10
                        break
        except (PermissionError, OSError):
            continue
        if roots["reddit_root"]:
            break

    # Also check for train folder structure
    train_folder = path / "train"
    if train_folder.exists():
        for sub in ["poligon", "polygon", "Polygon"]:
            p = train_folder / sub
            if p.exists():
                roots["polygon_root"] = str(p)
                score += 8
                break
        for sub in ["reddit", "Reddit"]:
            r = train_folder / sub
            if r.exists():
                roots["reddit_root"] = str(r)
                score += 8
                break

    return score, roots


def find_data_roots() -> dict:
    """Find the best SSD root and data locations."""
    candidates = find_mount_candidates()

    if not candidates:
        return {
            "ssd_root": None,
            "polygon_root": None,
            "reddit_root": None,
            "error": "No external drives found in /media, /run/media, or /mnt",
            "candidates": [],
        }

    best_score = -1
    best_root = None
    best_paths = {"polygon_root": None, "reddit_root": None}
    all_candidates = []

    for candidate in candidates:
        score, roots = score_candidate(candidate)
        all_candidates.append(
            {
                "path": str(candidate),
                "score": score,
                "polygon_root": roots["polygon_root"],
                "reddit_root": roots["reddit_root"],
            }
        )
        if score > best_score:
            best_score = score
            best_root = candidate
            best_paths = roots

    result = {
        "ssd_root": str(best_root) if best_root else None,
        "polygon_root": best_paths["polygon_root"],
        "reddit_root": best_paths["reddit_root"],
        "candidates": all_candidates,
    }

    if best_score == 0:
        result["warning"] = "SSD found but no polygon/reddit data detected"

    return result


def main():
    result = find_data_roots()

    # Pretty print for humans, compact for piping
    if sys.stdout.isatty():
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))

    # Exit with error if no SSD found
    if result.get("error") or result["ssd_root"] is None:
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
