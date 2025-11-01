"""Reporting helpers that turn raw rows into CSV and Markdown summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from . import __version__
from .io import ensure_directory, write_csv, write_jsonl

try:
    import pandas as pd  # type: ignore

    PANDAS_AVAILABLE = True
except Exception:  # pragma: no cover - pandas import optional
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False


def _markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return "_no data_"
    headers = list(rows[0].keys())
    lines = [
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header)
            if value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("|" + "|".join(values) + "|")
    return "\n".join(lines)


def write_reports(rows: Iterable[Dict[str, Any]], out_dir: Path) -> Dict[str, Path]:
    """Write JSONL, CSV, and optional Markdown summary for a benchmark run."""
    rows_list = list(rows)
    ensure_directory(out_dir)

    jsonl_path = out_dir / "results.jsonl"
    write_jsonl(jsonl_path, rows_list)

    csv_path = out_dir / "results.csv"
    write_csv(csv_path, rows_list)

    outputs: Dict[str, Path] = {"jsonl": jsonl_path, "csv": csv_path}

    if not PANDAS_AVAILABLE:
        return outputs

    df = pd.DataFrame(rows_list)
    df.to_csv(csv_path, index=False)

    summary_columns = ["kind", "model", "tag"]
    present_summary_cols = [col for col in summary_columns if col in df.columns]

    if present_summary_cols:
        grouped = (
            df.groupby(present_summary_cols, dropna=False)
            .median(numeric_only=True)
            .reset_index()
        )
    else:
        grouped = df.median(numeric_only=True).to_frame().T

    summary_csv = out_dir / "results_summary.csv"
    grouped.to_csv(summary_csv, index=False)
    outputs["summary_csv"] = summary_csv

    summary_md = out_dir / "SUMMARY.md"
    aggregates_table = _markdown_table(grouped.fillna("").to_dict(orient="records"))
    sample_rows = df.head(20).fillna("").to_dict(orient="records")
    sample_table = _markdown_table(sample_rows)

    summary_md.write_text(
        "\n".join(
            [
                f"# ollama-bench summary (v{__version__})",
                "",
                "## Aggregates (median)",
                "",
                aggregates_table,
                "",
                "## First 20 raw rows",
                "",
                sample_table,
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs["summary_md"] = summary_md
    return outputs


__all__ = ["PANDAS_AVAILABLE", "write_reports"]
