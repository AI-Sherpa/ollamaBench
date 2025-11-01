from __future__ import annotations

from typing import Any, Dict, Iterator, List

import pytest

from ollamabench.bench import BenchmarkRunner
from ollamabench.schemas import EmbeddingJobConfig, GenerationJobConfig


class StubClient:
    def __init__(
        self,
        *,
        prompt: str,
        stream_chunks: List[Dict[str, Any]],
        prompt_tokens: int,
        completion_tokens: int,
        embedding_payload: Dict[str, Any],
    ) -> None:
        self.prompt = prompt
        self.stream_chunks = stream_chunks
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.embedding_payload = embedding_payload

    def tokenize_count(self, model: str, text: str) -> int:
        return self.prompt_tokens if text == self.prompt else self.completion_tokens

    def generate_stream(self, model: str, prompt: str, options: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        yield from self.stream_chunks

    def create_embeddings(self, model: str, text: str) -> Dict[str, Any]:
        return self.embedding_payload

    def fetch_ollama_version(self) -> None:  # pragma: no cover - unused in tests
        return None

    def check_connectivity(self) -> bool:  # pragma: no cover - unused in tests
        return True

    def close(self) -> None:  # pragma: no cover - unused in tests
        return None


def test_generation_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = "prompt text"
    chunks = [
        {"response": "hello", "done": False},
        {"response": " world", "done": True, "eval_count": 18, "prompt_eval_count": 12},
    ]
    client = StubClient(
        prompt=prompt,
        stream_chunks=chunks,
        prompt_tokens=12,
        completion_tokens=18,
        embedding_payload={"embedding": [0.1, 0.2]},
    )

    times = iter([0.0, 0.2, 0.9, 1.0])
    monkeypatch.setattr("ollamabench.bench.time.monotonic", lambda: next(times))

    runner = BenchmarkRunner(client, quiet=True)
    job = GenerationJobConfig(model="llama3", prompt=prompt, tag="S", warmup=0, repeats=1, options={})
    rows = runner.run_generation_job(job)
    assert len(rows) == 1
    row = rows[0]
    assert row["ttft_sec"] == pytest.approx(0.2)
    assert row["total_time_sec"] == pytest.approx(1.0)
    assert row["decode_time_sec"] == pytest.approx(0.8)
    assert row["ingest_toks_per_sec"] == pytest.approx(60.0)
    assert row["decode_toks_per_sec"] == pytest.approx(22.5)


def test_generation_metrics_zero_ttft(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = "hi"
    chunks = [
        {"response": "", "done": True, "eval_count": 5, "prompt_eval_count": 5},
    ]
    client = StubClient(
        prompt=prompt,
        stream_chunks=chunks,
        prompt_tokens=5,
        completion_tokens=5,
        embedding_payload={"embedding": [0.1]},
    )

    times = iter([1.0, 1.0, 2.0])
    monkeypatch.setattr("ollamabench.bench.time.monotonic", lambda: next(times))

    runner = BenchmarkRunner(client, quiet=True)
    job = GenerationJobConfig(model="llama3", prompt=prompt, tag=None, warmup=0, repeats=1, options={})
    row = runner.run_generation_job(job)[0]
    assert row["ttft_sec"] == 0.0
    assert row["ingest_toks_per_sec"] is None


def test_embedding_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = "prompt"
    chunks = [{"response": "", "done": True}]
    client = StubClient(
        prompt=prompt,
        stream_chunks=chunks,
        prompt_tokens=1,
        completion_tokens=1,
        embedding_payload={"embedding": list(range(10))},
    )

    times = iter([10.0, 11.0])
    monkeypatch.setattr("ollamabench.bench.time.monotonic", lambda: next(times))

    runner = BenchmarkRunner(client, quiet=True)
    job = EmbeddingJobConfig(model="embed", text="abcdef", tag="E", repeats=1)
    row = runner.run_embedding_job(job)[0]
    assert row["elapsed_sec"] == pytest.approx(1.0)
    assert row["throughput_text_chars_per_sec"] == pytest.approx(6.0)
    assert row["dim"] == 10
