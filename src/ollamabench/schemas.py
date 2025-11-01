"""Data models and validation helpers for ollamabench suites."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OUT_DIR = "bench_out"


class SuiteValidationError(ValueError):
    """Raised when a suite definition fails validation."""


@dataclass(frozen=True)
class GenerationJobConfig:
    """Configuration for a generation benchmark job."""

    model: str
    prompt: str
    tag: Optional[str] = None
    warmup: int = 1
    repeats: int = 3
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingJobConfig:
    """Configuration for an embedding benchmark job."""

    model: str
    text: str
    tag: Optional[str] = None
    repeats: int = 3


@dataclass(frozen=True)
class SuiteConfig:
    """Valid suite specification used by the runner."""

    name: str
    ollama_url: str
    out_dir: str
    generation: List[GenerationJobConfig] = field(default_factory=list)
    embeddings: List[EmbeddingJobConfig] = field(default_factory=list)


def _validate_positive_int(name: str, value: Any, minimum: int) -> int:
    if not isinstance(value, int):
        raise SuiteValidationError(f"'{name}' must be an integer (got {type(value).__name__})")
    if value < minimum:
        suffix = "1" if minimum == 1 else str(minimum)
        raise SuiteValidationError(f"'{name}' must be >= {suffix} (got {value})")
    return value


def _validate_non_empty_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SuiteValidationError(f"'{name}' must be a non-empty string")
    return value


def _coerce_options(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SuiteValidationError("'options' must be a mapping if provided")
    return dict(value)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except OSError as exc:
        raise SuiteValidationError(f"Failed to read suite file '{path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise SuiteValidationError(f"Suite file '{path}' is not valid YAML: {exc}") from exc
    if data is None:
        raise SuiteValidationError("Suite file was empty")
    if not isinstance(data, dict):
        raise SuiteValidationError("Suite file must define a YAML mapping at top level")
    return data


def load_suite(path: str | Path) -> SuiteConfig:
    """Load and validate a suite definition from YAML."""
    path = Path(path)
    raw = _load_yaml(path)

    name = _validate_non_empty_str("name", raw.get("name", path.stem))
    ollama_url_raw = raw.get("ollama_url", DEFAULT_OLLAMA_URL)
    ollama_url = _validate_non_empty_str("ollama_url", ollama_url_raw)
    out_dir_raw = raw.get("out_dir", DEFAULT_OUT_DIR)
    out_dir = _validate_non_empty_str("out_dir", out_dir_raw)

    gen_jobs: List[GenerationJobConfig] = []
    raw_generation = raw.get("generation", [])
    if raw_generation is None:
        raw_generation = []
    if not isinstance(raw_generation, list):
        raise SuiteValidationError("'generation' must be a list if provided")
    for idx, job in enumerate(raw_generation):
        if not isinstance(job, dict):
            raise SuiteValidationError(f"generation[{idx}] must be a mapping")
        model = _validate_non_empty_str(f"generation[{idx}].model", job.get("model"))
        prompt = _validate_non_empty_str(f"generation[{idx}].prompt", job.get("prompt"))
        tag = job.get("tag")
        if tag is not None:
            tag = _validate_non_empty_str(f"generation[{idx}].tag", tag)
        warmup = _validate_positive_int(f"generation[{idx}].warmup", job.get("warmup", 1), minimum=0)
        repeats = _validate_positive_int(f"generation[{idx}].repeats", job.get("repeats", 3), minimum=1)
        options = _coerce_options(job.get("options"))
        gen_jobs.append(
            GenerationJobConfig(
                model=model,
                prompt=prompt,
                tag=tag,
                warmup=warmup,
                repeats=repeats,
                options=options,
            )
        )

    emb_jobs: List[EmbeddingJobConfig] = []
    raw_embeddings = raw.get("embeddings", [])
    if raw_embeddings is None:
        raw_embeddings = []
    if not isinstance(raw_embeddings, list):
        raise SuiteValidationError("'embeddings' must be a list if provided")
    for idx, job in enumerate(raw_embeddings):
        if not isinstance(job, dict):
            raise SuiteValidationError(f"embeddings[{idx}] must be a mapping")
        model = _validate_non_empty_str(f"embeddings[{idx}].model", job.get("model"))
        text = _validate_non_empty_str(f"embeddings[{idx}].text", job.get("text"))
        tag = job.get("tag")
        if tag is not None:
            tag = _validate_non_empty_str(f"embeddings[{idx}].tag", tag)
        repeats = _validate_positive_int(f"embeddings[{idx}].repeats", job.get("repeats", 3), minimum=1)
        emb_jobs.append(
            EmbeddingJobConfig(
                model=model,
                text=text,
                tag=tag,
                repeats=repeats,
            )
        )

    if not gen_jobs and not emb_jobs:
        raise SuiteValidationError("Suite must define at least one generation or embedding job")

    return SuiteConfig(
        name=name,
        ollama_url=ollama_url,
        out_dir=out_dir,
        generation=gen_jobs,
        embeddings=emb_jobs,
    )


__all__ = [
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_OUT_DIR",
    "EmbeddingJobConfig",
    "GenerationJobConfig",
    "SuiteConfig",
    "SuiteValidationError",
    "load_suite",
]
