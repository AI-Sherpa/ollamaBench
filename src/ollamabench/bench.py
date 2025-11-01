"""Benchmark execution logic for ollamabench."""

from __future__ import annotations

import json
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import requests
from requests import Response, Session

from . import __version__
from .schemas import EmbeddingJobConfig, GenerationJobConfig

DEFAULT_OPTIONS = {"temperature": 0, "seed": 1}
DEFAULT_TIMEOUT: Tuple[int, int] = (3, 120)


class OllamaClient:
    """Lightweight HTTP client for the Ollama REST API."""

    def __init__(
        self,
        base_url: str,
        *,
        session: Optional[Session] = None,
        timeout: Tuple[int, int] = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    def close(self) -> None:
        self.session.close()

    def _post(self, path: str, **kwargs: Any) -> Response:
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", self.timeout)
        return self.session.post(url, **kwargs)

    def _get(self, path: str, **kwargs: Any) -> Response:
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", self.timeout)
        return self.session.get(url, **kwargs)

    def tokenize_count(self, model: str, text: str) -> Optional[int]:
        payload = {"model": model, "prompt": text}
        try:
            response = self._post("/api/tokenize", json=payload)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, json.JSONDecodeError):
            return None
        count = data.get("count")
        if isinstance(count, int):
            return count
        tokens = data.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
        return None

    def generate_stream(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": options or {},
        }
        try:
            response = self._post("/api/generate", json=payload, stream=True)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network errors
            raise RuntimeError(f"Ollama generate request failed: {exc}") from exc

        with closing(response):
            for chunk in response.iter_lines():
                if not chunk:
                    continue
                try:
                    decoded = json.loads(chunk.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                yield decoded

    def create_embeddings(self, model: str, text: str) -> Dict[str, Any]:
        payload = {"model": model, "prompt": text, "input": text}
        try:
            response = self._post("/api/embeddings", json=payload)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama embeddings request failed: {exc}") from exc

    def show_model(self, model: str) -> Optional[Dict[str, Any]]:
        payload = {"model": model}
        try:
            response = self._post("/api/show", json=payload)
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama show request failed: {exc}") from exc

        if response.status_code == 404:
            return None

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"Ollama show request failed: {exc}") from exc

        try:
            data = response.json()
        except json.JSONDecodeError:
            return {}
        if isinstance(data, dict):
            return data
        return {}

    def fetch_ollama_version(self) -> Optional[str]:
        try:
            response = self._get("/api/version")
            response.raise_for_status()
            data = response.json()
            version = data.get("version")
            if isinstance(version, str):
                return version
        except (requests.RequestException, json.JSONDecodeError):
            return None
        return None

    def is_model_available(self, model: str) -> bool:
        try:
            return self.show_model(model) is not None
        except RuntimeError:
            return False

    def check_connectivity(self) -> bool:
        try:
            response = self._get("/api/tags")
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rate(count: Optional[int], duration: float) -> Optional[float]:
    if count is None or count < 0:
        return None
    if duration <= 0:
        return None
    return count / duration


def _merge_options(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    return merged


@dataclass
class GenerationTrialOutcome:
    row: Dict[str, Any]
    prompt_tokens: Optional[int]


class BenchmarkRunner:
    """Coordinates benchmark execution for both generation and embeddings."""

    def __init__(
        self,
        client: OllamaClient,
        *,
        quiet: bool = False,
    ) -> None:
        self.client = client
        self.quiet = quiet

    def run_generation_job(self, job: GenerationJobConfig) -> List[Dict[str, Any]]:
        prompt_tokens = self.client.tokenize_count(job.model, job.prompt)
        tag = job.tag

        outcomes: List[Dict[str, Any]] = []
        benchmark_rows: List[Dict[str, Any]] = outcomes
        if not self.quiet:
            print(
                f"[ollama-bench] generation job model={job.model} tag={tag or '-'} "
                f"warmup={job.warmup} repeats={job.repeats}"
            )

        merged_options = _merge_options(DEFAULT_OPTIONS, job.options)

        total_iterations = job.warmup + job.repeats
        try:
            for iteration in range(total_iterations):
                record = iteration >= job.warmup
                outcome = self._run_single_generation(
                    job=job,
                    prompt_tokens=prompt_tokens,
                    merged_options=merged_options,
                    record=record,
                    iteration=iteration,
                )
                if record and outcome:
                    outcomes.append(outcome.row)
                    if outcome.prompt_tokens is not None:
                        prompt_tokens = outcome.prompt_tokens
        except KeyboardInterrupt as exc:
            raise BenchmarkInterrupted(benchmark_rows) from exc

        return outcomes

    def _run_single_generation(
        self,
        job: GenerationJobConfig,
        *,
        prompt_tokens: Optional[int],
        merged_options: Dict[str, Any],
        record: bool,
        iteration: int,
    ) -> Optional[GenerationTrialOutcome]:
        start_monotonic = time.monotonic()
        first_chunk_time: Optional[float] = None
        completion_parts: List[str] = []
        prompt_eval_count: Optional[int] = None
        completion_eval_count: Optional[int] = None

        if not self.quiet:
            phase = "warmup" if not record else "trial"
            print(f"  → {phase} {iteration + 1}")

        for chunk in self.client.generate_stream(job.model, job.prompt, merged_options):
            now = time.monotonic()
            response_text = chunk.get("response") or ""
            if response_text:
                completion_parts.append(response_text)
            if first_chunk_time is None and (response_text or chunk.get("done")):
                first_chunk_time = now

            if chunk.get("prompt_eval_count") is not None:
                prompt_eval_count = chunk.get("prompt_eval_count")

            if chunk.get("eval_count") is not None:
                completion_eval_count = chunk.get("eval_count")

            if chunk.get("done"):
                break

        end_monotonic = time.monotonic()
        ttft = (first_chunk_time or end_monotonic) - start_monotonic
        total_time = end_monotonic - start_monotonic
        decode_time = max(total_time - ttft, 0.0)
        completion_text = "".join(completion_parts)

        if prompt_tokens is None and prompt_eval_count is not None:
            prompt_tokens = int(prompt_eval_count)

        completion_tokens = self.client.tokenize_count(job.model, completion_text)
        if completion_tokens is None and completion_eval_count is not None:
            completion_tokens = int(completion_eval_count)

        ingest_rate = _safe_rate(prompt_tokens, ttft)
        decode_rate = _safe_rate(completion_tokens, decode_time)

        if not record:
            return None

        row: Dict[str, Any] = {
            "kind": "generate",
            "ts": _timestamp(),
            "bench_version": __version__,
            "model": job.model,
            "tag": job.tag,
            "prompt_chars": len(job.prompt),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft_sec": round(ttft, 6),
            "total_time_sec": round(total_time, 6),
            "decode_time_sec": round(decode_time, 6),
            "ingest_toks_per_sec": round(ingest_rate, 6) if ingest_rate is not None else None,
            "decode_toks_per_sec": round(decode_rate, 6) if decode_rate is not None else None,
            "options": dict(merged_options),
        }

        return GenerationTrialOutcome(row=row, prompt_tokens=prompt_tokens)

    def run_embedding_job(self, job: EmbeddingJobConfig) -> List[Dict[str, Any]]:
        if not self.quiet:
            print(
                f"[ollama-bench] embedding job model={job.model} tag={job.tag or '-'} "
                f"repeats={job.repeats}"
            )
        rows: List[Dict[str, Any]] = []
        try:
            for iteration in range(job.repeats):
                if not self.quiet:
                    print(f"  → trial {iteration + 1}")
                start = time.monotonic()
                payload = self.client.create_embeddings(job.model, job.text)
                end = time.monotonic()
                elapsed = end - start

                embeddings = payload.get("embedding")
                if embeddings is None:
                    data = payload.get("data")
                    if isinstance(data, list) and data:
                        embeddings = data[0].get("embedding")
                dim = len(embeddings) if isinstance(embeddings, list) else None

                throughput = _safe_rate(len(job.text), elapsed)

                rows.append(
                    {
                        "kind": "embedding",
                        "ts": _timestamp(),
                        "bench_version": __version__,
                        "model": job.model,
                        "tag": job.tag,
                        "text_chars": len(job.text),
                        "elapsed_sec": round(elapsed, 6),
                        "throughput_text_chars_per_sec": round(throughput, 6) if throughput else None,
                        "dim": dim,
                    }
                )
        except KeyboardInterrupt as exc:
            raise BenchmarkInterrupted(rows) from exc
        return rows


__all__ = ["BenchmarkInterrupted", "BenchmarkRunner", "DEFAULT_OPTIONS", "DEFAULT_TIMEOUT", "OllamaClient"]
class BenchmarkInterrupted(KeyboardInterrupt):
    """Carries partial benchmark rows when a run is interrupted."""

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        super().__init__("Benchmark interrupted")
        self.rows = rows
