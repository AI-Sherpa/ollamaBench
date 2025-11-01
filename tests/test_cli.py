from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Set

import pytest

from ollamabench import __version__, cli


class DummyClient:
    available_models: Set[str] = {"llama3"}

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def tokenize_count(self, model: str, text: str) -> int:
        return 12

    def fetch_ollama_version(self) -> str:
        return "0.0.1"

    def check_connectivity(self) -> bool:
        return True

    def is_model_available(self, model: str) -> bool:
        return self.show_model(model) is not None

    def show_model(self, model: str) -> Dict[str, Any] | None:
        if model not in self.available_models:
            return None
        return {
            "model": model,
            "details": {
                "parameter_size": "1B",
                "quantization_level": "q4_K_M",
            },
        }

    def close(self) -> None:
        return None


class DummyRunner:
    def __init__(self, client: DummyClient, quiet: bool = False) -> None:
        self.client = client
        self.quiet = quiet

    def run_generation_job(self, job) -> List[Dict[str, Any]]:
        return [
            {
                "kind": "generate",
                "ts": "2024-01-01T00:00:00+00:00",
                "bench_version": __version__,
                "model": job.model,
                "tag": job.tag,
                "prompt_chars": len(job.prompt),
                "prompt_tokens": 12,
                "completion_tokens": 24,
                "ttft_sec": 0.2,
                "total_time_sec": 1.2,
                "decode_time_sec": 1.0,
                "ingest_toks_per_sec": 60.0,
                "decode_toks_per_sec": 24.0,
                "options": job.options,
            }
        ]

    def run_embedding_job(self, job) -> List[Dict[str, Any]]:
        return [
            {
                "kind": "embedding",
                "ts": "2024-01-01T00:00:00+00:00",
                "bench_version": __version__,
                "model": job.model,
                "tag": job.tag,
                "text_chars": len(job.text),
                "elapsed_sec": 1.0,
                "throughput_text_chars_per_sec": 40.0,
                "dim": 10,
            }
        ]


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DummyClient, "available_models", {"llama3"})
    monkeypatch.setattr(cli, "OllamaClient", DummyClient)
    monkeypatch.setattr(cli, "BenchmarkRunner", DummyRunner)
    monkeypatch.setattr(cli, "_command_version", lambda cmd: "ollama version 0.test" if cmd and cmd[0] == "ollama" else None)
    monkeypatch.setattr(cli, "_list_local_models", lambda: {"llama3"})
    monkeypatch.setattr(cli, "_pull_model", lambda model, quiet=False: True)
    monkeypatch.setattr(cli, "_remove_model", lambda model, quiet=False: None)
    monkeypatch.setattr(
        cli.shutil,
        "which",
        lambda tool: "/usr/sbin/system_profiler" if tool == "system_profiler" else None,
    )
    monkeypatch.setattr(
        cli,
        "collect_system_info",
        lambda: {
            "timestamp_utc": "2024-01-01T00:00:00+00:00",
            "os": "Darwin",
            "machine": "arm64",
            "ram_total_gib": 32,
            "cpu_count_logical": 8,
            "cpu_count_physical": 4,
        },
    )
    monkeypatch.setattr(cli, "_query_gpu_memory", lambda: [])
    monkeypatch.setattr(
        cli.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=32 * (1024**3), available=24 * (1024**3)),
    )


def test_cli_quick_writes_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_dir = tmp_path / "quick_out"
    rc = cli.main(
        [
            "--json",
            "quick",
            "--model",
            "llama3",
            "--prompt",
            "hello world",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    output = json.loads(capsys.readouterr().out)
    assert output["rows_written"] == 1
    assert (out_dir / "results.jsonl").exists()
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "meta.json").exists()


def test_cli_run_suite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite_path = tmp_path / "suite.yaml"
    out_dir = tmp_path / "suite_out"
    suite_path.write_text(
        f"""
        name: demo
        out_dir: "{out_dir}"
        generation:
          - model: llama3
            prompt: hi
        """,
        encoding="utf-8",
    )
    rc = cli.main(["--json", "run", "--suite", str(suite_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows_written"] == 1
    assert payload["out_dir"] == str(out_dir)
    assert (out_dir / "results.jsonl").exists()


def test_cli_quick_requires_local_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(DummyClient, "available_models", set())
    monkeypatch.setattr(cli, "_list_local_models", lambda: set())
    out_dir = tmp_path / "quick_out"
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "quick",
                "--model",
                "llama3",
                "--prompt",
                "hello world",
                "--out-dir",
                str(out_dir),
            ]
        )
    message = str(exc.value)
    assert "not loaded locally" in message
    assert "ollama pull" in message


def test_cli_validate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
        name: ok
        generation:
          - model: llama3
            prompt: hi
        """,
        encoding="utf-8",
    )
    rc = cli.main(["--json", "validate", "--suite", str(suite_path)])
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["generation_jobs"] == 1


def test_cli_validate_json_alias(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
        name: alias
        generation:
          - model: llama3
            prompt: hi
        """,
        encoding="utf-8",
    )
    rc = cli.main(["validate", "--suite", str(suite_path), "--json"])
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"


def test_cli_doctor_suite_preflight(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
        name: preflight
        generation:
          - model: llama3
            prompt: hi
          - model: missing-model
            prompt: there
        """,
        encoding="utf-8",
    )
    rc = cli.main(["--json", "doctor", "--suite", str(suite_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["suite_path"] == str(suite_path)
    preflight = payload["preflight"]
    assert "missing-model" in preflight["models_missing"]
    assert not preflight["ok"]
    tools = payload["tools"]
    assert tools["system_profiler"]["status"] == "found"
    assert tools["system_profiler"]["detail"] == "/usr/sbin/system_profiler"
    assert tools["nvidia-smi"]["status"] == "not_applicable"
    assert tools["tegrastats"]["status"] == "not_applicable"


def test_cli_doctor_auto_pull(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    available: Set[str] = set()
    pulled: List[str] = []
    removed: List[str] = []

    monkeypatch.setattr(
        cli,
        "_list_local_models",
        lambda: set(available),
    )
    monkeypatch.setattr(DummyClient, "available_models", available)

    def fake_pull(model: str, quiet: bool = False) -> bool:
        if model == "missing-model":
            return False
        available.add(model)
        pulled.append(model)
        return True

    def fake_remove(model: str, quiet: bool = False) -> None:
        removed.append(model)
        available.discard(model)

    monkeypatch.setattr(cli, "_pull_model", fake_pull)
    monkeypatch.setattr(cli, "_remove_model", fake_remove)

    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
        name: auto
        generation:
          - model: llama3
            prompt: hi
          - model: missing-model
            prompt: \"missing\"
        """,
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--json", "doctor", "--suite", str(suite_path), "--auto-pull"])
    assert "Failed to pull" in str(excinfo.value)
    # doctor retains successfully pulled models on failure for inspection
    assert removed == []


def test_cli_quick_auto_pull(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    available: Set[str] = set()
    removals: List[str] = []

    monkeypatch.setattr(DummyClient, "available_models", available)
    monkeypatch.setattr(
        cli,
        "_list_local_models",
        lambda: set(available),
    )

    def fake_pull(model: str, quiet: bool = False) -> bool:
        available.add(model)
        return True

    def fake_remove(model: str, quiet: bool = False) -> None:
        removals.append(model)
        available.discard(model)

    monkeypatch.setattr(cli, "_pull_model", fake_pull)
    monkeypatch.setattr(cli, "_remove_model", fake_remove)

    out_dir = tmp_path / "quick_out"
    rc = cli.main(
        [
            "--json",
            "quick",
            "--model",
            "llama3",
            "--prompt",
            "auto pull test",
            "--out-dir",
            str(out_dir),
            "--auto-pull",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows_written"] == 1
    assert removals == ["llama3"]
    assert not available


def test_cli_run_sync_models_updates_suite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # initial suite with outdated identifiers
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
        name: sync
        out_dir: "{out}"
        generation:
          - model: old-model
            prompt: hi
        embeddings:
          - model: old-embed
            text: hi
        """.replace("{out}", str(tmp_path / "out")),
        encoding="utf-8",
    )

    def fake_get(url: str, params: Dict[str, Any], timeout: int) -> Any:
        class MockResponse:
            def __init__(self, data: Dict[str, Any]) -> None:
                self._data = data
                self.status_code = 200

            def json(self) -> Dict[str, Any]:
                return self._data

        query = params.get("q")
        if query == "old-model":
            return MockResponse({"models": [{"model": "new-model"}]})
        if query == "old-embed":
            return MockResponse({"models": [{"model": "new-embed"}]})
        return MockResponse({"models": []})

    monkeypatch.setattr(cli.requests, "get", fake_get)
    monkeypatch.setattr(cli, "_list_local_models", lambda: {"new-model", "new-embed"})
    monkeypatch.setattr(DummyClient, "available_models", {"new-model", "new-embed"})

    rc = cli.main(["--json", "run", "--suite", str(suite_path), "--sync-models"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["suite"] == "sync"
    updated_yaml = suite_path.read_text(encoding="utf-8")
    assert "new-model" in updated_yaml
    assert "new-embed" in updated_yaml


def test_cli_run_auto_pull_executes_jobs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    available: Set[str] = set()
    removals: List[str] = []

    monkeypatch.setattr(DummyClient, "available_models", available)
    monkeypatch.setattr(cli, "_list_local_models", lambda: set(available))

    def fake_pull(model: str, quiet: bool = False) -> bool:
        available.add(model)
        return True

    def fake_remove(model: str, quiet: bool = False) -> None:
        removals.append(model)
        available.discard(model)

    monkeypatch.setattr(cli, "_pull_model", fake_pull)
    monkeypatch.setattr(cli, "_remove_model", fake_remove)

    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
        name: run
        out_dir: "{out}"
        generation:
          - model: llama3
            prompt: hi
        """.replace("{out}", str(tmp_path / "out")),
        encoding="utf-8",
    )

    rc = cli.main(["--json", "run", "--suite", str(suite_path), "--auto-pull"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows_written"] == 1
    assert removals == ["llama3"]
