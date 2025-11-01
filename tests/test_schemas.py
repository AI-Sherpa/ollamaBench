from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ollamabench import schemas


def write_suite(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_load_suite_success(tmp_path: Path) -> None:
    path = write_suite(
        tmp_path,
        """
        name: demo
        generation:
          - model: llama3
            prompt: hello
            repeats: 2
        embeddings:
          - model: embed
            text: hi
        """,
    )
    suite = schemas.load_suite(path)
    assert suite.name == "demo"
    assert suite.ollama_url == schemas.DEFAULT_OLLAMA_URL
    assert len(suite.generation) == 1
    assert len(suite.embeddings) == 1


def test_load_suite_requires_jobs(tmp_path: Path) -> None:
    path = write_suite(tmp_path, "name: empty\n")
    with pytest.raises(schemas.SuiteValidationError):
        schemas.load_suite(path)


def test_load_suite_rejects_invalid_warmup(tmp_path: Path) -> None:
    path = write_suite(
        tmp_path,
        """
        name: bad
        generation:
          - model: llama3
            prompt: hello
            warmup: -1
        """,
    )
    with pytest.raises(schemas.SuiteValidationError):
        schemas.load_suite(path)
