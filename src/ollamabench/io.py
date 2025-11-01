"""Filesystem helpers for ollamabench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_directory(path: Path) -> None:
    """Create a directory hierarchy if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON payload with UTF-8 encoding."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write newline-delimited JSON rows."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write CSV rows using dictionary headers."""
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


__all__ = ["ensure_directory", "write_csv", "write_json", "write_jsonl"]
