"""Command line interface for ollama-bench."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests
import shutil
import yaml

import psutil

from . import __version__
from .bench import BenchmarkInterrupted, BenchmarkRunner, DEFAULT_OPTIONS, OllamaClient
from .io import ensure_directory, write_json
from .report import PANDAS_AVAILABLE, write_reports
from .schemas import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_OUT_DIR,
    EmbeddingJobConfig,
    GenerationJobConfig,
    SuiteConfig,
    SuiteValidationError,
    load_suite,
)
from .system import collect_system_info


def _parse_json_options(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse --options-json: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--options-json must decode to a JSON object")
    return parsed


def _classify_prompt(size_hint: Optional[int], prompt: str) -> str:
    """Return a coarse tag based on prompt length."""
    estimate = size_hint if size_hint is not None else max(len(prompt) // 4, 1)
    if estimate <= 80:
        return "S"
    if estimate <= 400:
        return "M"
    return "L"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ollama-bench",
        description="Benchmark Ollama models with reproducible workloads.",
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    parser.add_argument("--quiet", action="store_true", help="reduce console output")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON on stdout (where applicable)",
    )

    subparsers = parser.add_subparsers(dest="command")

    quick = subparsers.add_parser("quick", help="run a quick single prompt benchmark")
    quick.add_argument("--model", required=True, help="Ollama model name (e.g. llama3:8b)")
    quick.add_argument("--prompt", required=True, help="Prompt text")
    quick.add_argument("--tag", help="Optional tag label for this prompt size")
    quick.add_argument("--repeats", type=int, default=3, help="Number of recorded trials (default: 3)")
    quick.add_argument("--warmup", type=int, default=1, help="Warmup iterations to discard (default: 1)")
    quick.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})",
    )
    quick.add_argument(
        "--out-dir",
        default="bench_out_quick",
        help="Output directory (default: bench_out_quick)",
    )
    quick.add_argument(
        "--options-json",
        help="JSON object with Ollama generation options (merged over deterministic defaults)",
    )
    quick.add_argument("--json", dest="json_alias", action="store_true", help="Emit machine-readable JSON (alias)")
    quick.add_argument(
        "--auto-pull",
        action="store_true",
        help="Automatically pull missing models before running (removed afterwards by default)",
    )
    quick.add_argument(
        "--keep-pulled",
        action="store_true",
        help="Keep models fetched via --auto-pull instead of removing them after completion",
    )

    run = subparsers.add_parser("run", help="execute a YAML benchmark suite")
    run.add_argument("--suite", required=True, help="Path to suite YAML file")
    run.add_argument(
        "--auto-pull",
        action="store_true",
        help="Automatically pull missing models listed in the suite before running",
    )
    run.add_argument(
        "--keep-pulled",
        action="store_true",
        help="Keep models fetched via --auto-pull instead of removing them after completion",
    )
    run.add_argument(
        "--sync-models",
        action="store_true",
        help="Query the Ollama catalog and update suite model names before running",
    )
    run.add_argument("--json", dest="json_alias", action="store_true", help="Emit machine-readable JSON (alias)")

    validate = subparsers.add_parser("validate", help="validate a suite without running it")
    validate.add_argument("--suite", required=True, help="Path to suite YAML file")
    validate.add_argument("--json", dest="json_alias", action="store_true", help="Emit machine-readable JSON (alias)")

    doctor = subparsers.add_parser("doctor", help="diagnose environment and Ollama connectivity")
    doctor.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL to probe (default: {DEFAULT_OLLAMA_URL})",
    )
    doctor.add_argument(
        "--suite",
        help="Optional suite file to preflight model availability and resource requirements",
    )
    doctor.add_argument(
        "--auto-pull",
        action="store_true",
        help="Pull missing models referenced by --suite during preflight",
    )
    doctor.add_argument(
        "--keep-pulled",
        action="store_true",
        help="Keep models fetched via --auto-pull after doctor completes",
    )
    doctor.add_argument(
        "--sync-models",
        action="store_true",
        help="Update suite model identifiers to the latest catalog entries before preflight",
    )
    doctor.add_argument("--json", dest="json_alias", action="store_true", help="Emit machine-readable JSON (alias)")

    return parser


def _median(values: List[float]) -> Optional[float]:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return float(statistics.median(cleaned))


def _print_warning(message: str) -> None:
    sys.stderr.write(f"[ollama-bench] warning: {message}\n")


def _parse_parameter_count(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        if raw > 1e6:
            return float(raw)
        return None
    if not isinstance(raw, str):
        return None
    cleaned = raw.replace(",", "").strip()
    match = re.match(r"(?P<num>\d+(?:\.\d+)?)(?:\s*)(?P<unit>[kKmMgGtTbB]?)(?:[pP]aram[s]?)?", cleaned)
    if not match:
        return None
    value = float(match.group("num"))
    unit = match.group("unit").lower()
    multipliers = {
        "": 1e9,  # assume billions by default
        "k": 1e3,
        "m": 1e6,
        "g": 1e9,
        "b": 1e9,
        "t": 1e12,
    }
    multiplier = multipliers.get(unit, 1e9)
    return value * multiplier


def _parse_quantization_bits(raw: Any) -> int:
    if not raw:
        return 16
    text = str(raw).lower()
    match = re.search(r"q(\d+)", text)
    if match:
        val = int(match.group(1))
        return max(val, 1)
    for hint, bits in (
        ("int4", 4),
        ("int8", 8),
        ("fp16", 16),
        ("f16", 16),
        ("bf16", 16),
        ("fp32", 32),
    ):
        if hint in text:
            return bits
    return 16


def _estimate_model_memory_gib(model_info: Dict[str, Any]) -> Optional[float]:
    details = model_info.get("details") or {}
    parameter_fields = [
        details.get("parameter_size"),
        details.get("parameters"),
        model_info.get("parameter_size"),
        model_info.get("parameters"),
    ]
    param_count = None
    for field in parameter_fields:
        param_count = _parse_parameter_count(field)
        if param_count:
            break
    if not param_count:
        return None

    bits = _parse_quantization_bits(details.get("quantization_level") or details.get("quantization"))
    bytes_required = param_count * (bits / 8.0)
    overhead_factor = 1.15 if bits <= 8 else 1.1
    gib = bytes_required * overhead_factor / (1024**3)
    return gib


def _query_gpu_memory() -> List[Tuple[str, float, float]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    if result.returncode != 0 or not result.stdout:
        return []

    devices: List[Tuple[str, float, float]] = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        name = parts[0]
        try:
            total_mb = float(parts[1])
            used_mb = float(parts[2])
        except ValueError:
            continue
        total_gib = total_mb / 1024.0
        free_gib = max(total_mb - used_mb, 0.0) / 1024.0
        devices.append((name, total_gib, free_gib))
    return devices


def _evaluate_model_resources(model_infos: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    if not model_infos:
        return ([], [])
    try:
        virtual_mem = psutil.virtual_memory()
    except Exception:
        return ([], [])

    available_gib = virtual_mem.available / (1024**3)
    total_gib = virtual_mem.total / (1024**3)

    errors: List[str] = []
    warnings_set: set[str] = set()

    gpu_devices = _query_gpu_memory()
    max_gpu_total = max((device[1] for device in gpu_devices), default=0.0)
    max_gpu_free = max((device[2] for device in gpu_devices), default=0.0)

    for model, info in model_infos.items():
        requirement = _estimate_model_memory_gib(info)
        if requirement is None:
            continue
        if requirement + 0.5 > available_gib:
            errors.append(
                f"{model} requires ~{requirement:.1f} GiB but only {available_gib:.1f} GiB is currently free"
            )
        elif requirement > total_gib * 0.9:
            warnings_set.add(
                f"{model} consumes most system memory (~{requirement:.1f} GiB of {total_gib:.1f} GiB total)"
            )

        if gpu_devices:
            if requirement > max_gpu_total + 0.1:
                warnings_set.add(
                    f"{model} (~{requirement:.1f} GiB) exceeds largest GPU memory ({max_gpu_total:.1f} GiB)"
                )
            elif requirement + 0.5 > max_gpu_free:
                warnings_set.add(
                    f"{model} (~{requirement:.1f} GiB) may not fit into currently free GPU memory "
                    f"(max free {max_gpu_free:.1f} GiB)"
                )

    return (sorted(warnings_set), sorted(errors))


def _collect_model_infos(
    client: OllamaClient,
    models: Iterable[str],
    available_models: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
    unique_models: List[str] = []
    for model in models:
        if model not in unique_models:
            unique_models.append(model)

    missing: List[str] = []
    infos: Dict[str, Dict[str, Any]] = {}
    query_errors: List[str] = []

    available_set: Optional[set[str]] = None
    if available_models is not None:
        available_set = {name.strip() for name in available_models if name}

    for model in unique_models:
        installed_hint = None
        if available_set is not None:
            installed_hint = model in available_set

        if installed_hint is False:
            missing.append(model)
            continue

        try:
            info = client.show_model(model)
        except RuntimeError as exc:
            if installed_hint is True:
                infos[model] = {"details": {}}
                continue
            query_errors.append(str(exc))
            missing.append(model)
            continue
        if info is None:
            if installed_hint is True:
                infos[model] = {"details": {}}
                continue
            missing.append(model)
            continue
        infos[model] = info

    return infos, missing, query_errors


def _ensure_models_available(
    client: OllamaClient,
    models: Iterable[str],
    *,
    auto_pull: bool = False,
    pulled_models: Optional[Set[str]] = None,
    quiet: bool = False,
) -> Dict[str, Dict[str, Any]]:
    available_models = _list_local_models()
    model_infos, missing, query_errors = _collect_model_infos(client, models, available_models)

    if missing and auto_pull:
        pulled_now: List[str] = []
        for model in missing:
            if not _pull_model(model, quiet=quiet):
                raise SystemExit(f"Failed to pull '{model}' from the Ollama registry")
            pulled_now.append(model)
            if pulled_models is not None:
                pulled_models.add(model)
        if pulled_now:
            available_models = _list_local_models()
        model_infos, missing, query_errors = _collect_model_infos(client, models, available_models)

    if query_errors:
        lines = ["Encountered errors while querying Ollama for model metadata:", *(f"  - {err}" for err in query_errors)]
        raise SystemExit("\n".join(lines))

    if missing:
        lines = [
            "The following Ollama models are not loaded locally:",
            *(f"  - {model}" for model in missing),
            "",
            (
                "Please load them with `ollama pull <model>` (or `ollama run <model>` once) "
                "before rerunning, or update your benchmark configuration to use available models."
            ),
        ]
        raise SystemExit("\n".join(lines))

    warnings, errors = _evaluate_model_resources(model_infos)
    if errors:
        lines = [
            "Resource checks failed for requested models:",
            *(f"  - {err}" for err in errors),
            "",
            "Free up memory or choose a smaller / more aggressively quantized model before benchmarking.",
        ]
        raise SystemExit("\n".join(lines))

    for warn in warnings:
        _print_warning(warn)

    return model_infos


def _doctor_model_preflight(
    client: OllamaClient,
    models: Iterable[str],
    *,
    auto_pull: bool = False,
    pulled_models: Optional[Set[str]] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    available_models = _list_local_models()
    model_infos, missing, query_errors = _collect_model_infos(client, models, available_models)

    if missing and auto_pull:
        pulled_now: List[str] = []
        for model in missing:
            if not _pull_model(model, quiet=quiet):
                raise SystemExit(f"Failed to pull '{model}' from the Ollama registry")
            pulled_now.append(model)
            if pulled_models is not None:
                pulled_models.add(model)
        if pulled_now:
            available_models = _list_local_models()
        model_infos, missing, query_errors = _collect_model_infos(client, models, available_models)
    warnings, errors = _evaluate_model_resources(model_infos)
    requested = sorted({model for model in models})
    return {
        "requested_models": requested,
        "models_available": sorted(model_infos.keys()),
        "models_missing": sorted(missing),
        "model_query_errors": sorted(query_errors),
        "resource_warnings": warnings,
        "resource_errors": errors,
        "autopulled_models": sorted(pulled_models) if pulled_models else [],
        "ok": not missing and not query_errors and not errors,
    }


def _list_local_models() -> Set[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return set()

    if result.returncode != 0 or not result.stdout:
        return set()

    names: Set[str] = set()
    for idx, line in enumerate(result.stdout.strip().splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if idx == 0 and stripped.lower().startswith("name"):
            # skip header row
            continue
        name = stripped.split()[0]
        if name:
            names.add(name)
    return names


def _catalog_search_queries(model: str) -> List[str]:
    queries: List[str] = []
    if model:
        queries.append(model)
    base, _, remainder = model.partition(":")
    if base and base not in queries:
        queries.append(base)
    if remainder:
        segments = re.split(r"[-_]", remainder)
        while segments:
            candidate = base + ":" + "-".join(segments)
            if candidate not in queries:
                queries.append(candidate)
            segments = segments[:-1]
    seen: Set[str] = set()
    ordered: List[str] = []
    for query in queries:
        if query and query not in seen:
            ordered.append(query)
            seen.add(query)
    return ordered


def _fetch_catalog_names(query: str) -> Tuple[List[str], Optional[Exception]]:
    try:
        response = requests.get("https://ollama.com/search", params={"q": query}, timeout=10)
    except requests.RequestException as exc:
        return [], exc
    if response.status_code >= 400:
        return [], RuntimeError(f"https://ollama.com/search returned HTTP {response.status_code}")
    names: List[str] = []
    try:
        payload = response.json()
    except ValueError:
        text = response.text
        matches = re.findall(r'"model"\s*:\s*"([^"\\]+)"', text)
        if not matches:
            matches = re.findall(r'/library/([^"\\]+)"', text)
        names = [match.replace("\\u003A", ":").strip() for match in matches]
    else:
        raw_models = (
            payload.get("models")
            or payload.get("data")
            or payload.get("items")
            or payload.get("results")
            or []
        )
        for entry in raw_models:
            if isinstance(entry, str):
                names.append(entry.strip())
                continue
            if isinstance(entry, dict):
                for key in ("model", "name", "slug"):
                    value = entry.get(key)
                    if isinstance(value, str):
                        names.append(value.strip())
                        break
    names = [name for name in names if name]
    return names, None


def _select_best_catalog_match(model: str, names: List[str]) -> Optional[str]:
    if not names:
        return None
    base = model.split(":", 1)[0]
    for candidate in names:
        if candidate == model:
            return candidate
    for candidate in names:
        if candidate.startswith(model):
            return candidate
    for candidate in names:
        if candidate.startswith(base + ":"):
            return candidate
    return names[0]


def _search_model_catalog(model: str) -> Optional[str]:
    last_error: Optional[Exception] = None
    all_candidates: List[str] = []
    for query in _catalog_search_queries(model):
        names, error = _fetch_catalog_names(query)
        if error:
            last_error = error
            continue
        match = _select_best_catalog_match(model, names)
        if match:
            return match
        all_candidates.extend(names)
    if all_candidates:
        return _select_best_catalog_match(model, all_candidates)
    if last_error:
        raise SystemExit(f"Failed to query Ollama catalog for '{model}': {last_error}")
    return None


def _sync_suite_models(path: Path, *, quiet: bool = False) -> List[Tuple[str, str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except OSError as exc:
        raise SystemExit(f"Failed to read suite file '{path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"Suite file '{path}' is not valid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise SystemExit(f"Suite file '{path}' must contain a mapping")

    updated: List[Tuple[str, str]] = []

    def update_jobs(section: str) -> None:
        jobs = data.get(section, []) or []
        if not isinstance(jobs, list):
            raise SystemExit(f"'{section}' must be a list in suite '{path}'")
        for job in jobs:
            if not isinstance(job, dict):
                continue
            model = job.get("model")
            if not isinstance(model, str) or not model.strip():
                continue
            resolved = _search_model_catalog(model)
            if resolved is None:
                raise SystemExit(
                    f"Model '{model}' in suite '{path}' was not found in the Ollama catalog; "
                    "please update the configuration."
                )
            if resolved != model:
                job["model"] = resolved
                updated.append((model, resolved))

    update_jobs("generation")
    update_jobs("embeddings")

    if updated:
        try:
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(data, handle, sort_keys=False)
        except OSError as exc:
            raise SystemExit(f"Failed to write updated suite file '{path}': {exc}") from exc
        if not quiet:
            changes = ", ".join(f"{old}->{new}" for old, new in updated)
            print(f"[ollama-bench] updated suite models: {changes}", file=sys.stderr)

    return updated


def _pull_model(model: str, *, quiet: bool = False) -> bool:
    if not quiet:
        print(f"[ollama-bench] pulling model '{model}'…", file=sys.stderr)
    try:
        process = subprocess.Popen(
            ["ollama", "pull", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (FileNotFoundError, OSError) as exc:
        _print_warning(f"Failed to execute 'ollama pull {model}': {exc}")
        return False

    assert process.stdout is not None
    try:
        for line in process.stdout:
            if not quiet:
                stripped = line.rstrip()
                if stripped:
                    print(f"[ollama-pull] {stripped}", file=sys.stderr)
        process.wait()
    except subprocess.SubprocessError as exc:
        _print_warning(f"ollama pull {model} interrupted: {exc}")
        process.kill()
        return False

    if process.returncode != 0:
        _print_warning(f"ollama pull {model} failed with exit code {process.returncode}")
        return False
    return True


def _remove_model(model: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"[ollama-bench] removing model '{model}'…", file=sys.stderr)
    try:
        result = subprocess.run(
            ["ollama", "rm", model],
            capture_output=True,
            text=True,
            timeout=None,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        _print_warning(f"Failed to execute 'ollama rm {model}': {exc}")
        return
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        _print_warning(f"ollama rm {model} failed: {detail}")


def _cleanup_pulled_models(models: Set[str], *, quiet: bool = False) -> None:
    for model in sorted(models):
        _remove_model(model, quiet=quiet)


def _tool_applicability(os_name: str, machine: str) -> Dict[str, bool]:
    os_name = os_name.lower()
    machine = machine.lower()
    return {
        "ollama": True,
        "system_profiler": os_name == "darwin",
        "nvidia-smi": os_name == "linux",
        "tegrastats": os_name == "linux" and ("aarch64" in machine or "arm" in machine),
    }


def _gather_tool_status(system_info: Dict[str, Any]) -> Dict[str, Dict[str, Optional[str]]]:
    os_name = str(system_info.get("os") or "").lower()
    machine = str(system_info.get("machine") or "").lower()
    applicability = _tool_applicability(os_name, machine)

    tools: Dict[str, Dict[str, Optional[str]]] = {}

    if applicability["ollama"]:
        version = _command_version(["ollama", "--version"])
        tools["ollama"] = {
            "status": "found" if version else "missing",
            "detail": version,
        }

    for tool in ("system_profiler", "nvidia-smi", "tegrastats"):
        applicable = applicability.get(tool, False)
        if not applicable:
            tools[tool] = {"status": "not_applicable", "detail": None}
            continue
        present = bool(shutil.which(tool))
        detail = None
        if present:
            detail = shutil.which(tool)
        tools[tool] = {
            "status": "found" if present else "missing",
            "detail": detail,
        }

    return tools


def _run_quick(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    ensure_directory(out_dir)

    user_options = _parse_json_options(args.options_json)
    merged_options = {**DEFAULT_OPTIONS, **user_options}
    tag: Optional[str] = args.tag

    client = OllamaClient(args.ollama_url)
    runner = BenchmarkRunner(client, quiet=args.quiet)
    version: Optional[str] = None
    pulled_models: Set[str] = set()
    rows: List[Dict[str, Any]] = []
    interrupted = False

    try:
        _ensure_models_available(
            client,
            [args.model],
            auto_pull=args.auto_pull,
            pulled_models=pulled_models,
            quiet=args.quiet,
        )
        prompt_tokens_hint = client.tokenize_count(args.model, args.prompt)
        tag = args.tag or _classify_prompt(prompt_tokens_hint, args.prompt)
        job = GenerationJobConfig(
            model=args.model,
            prompt=args.prompt,
            tag=tag,
            warmup=max(args.warmup, 0),
            repeats=max(args.repeats, 1),
            options=merged_options,
        )

        try:
            rows = runner.run_generation_job(job)
        except BenchmarkInterrupted as exc:
            rows = exc.rows
            interrupted = True
        version = client.fetch_ollama_version()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        client.close()
        if args.auto_pull and pulled_models and not args.keep_pulled:
            _cleanup_pulled_models(pulled_models, quiet=args.quiet)

    status = "interrupted" if interrupted else "completed"
    if interrupted and not args.quiet:
        print("[ollama-bench] benchmark interrupted; partial results saved", file=sys.stderr)

    if not rows and not interrupted:
        raise SystemExit("No benchmark rows produced")

    outputs = write_reports(rows, out_dir)

    system_snapshot = collect_system_info()

    started_ts = rows[0]["ts"] if rows else system_snapshot["timestamp_utc"]
    prompt_tag = tag
    if rows and rows[0].get("tag") is not None:
        prompt_tag = rows[0]["tag"]

    meta = {
        "bench_version": __version__,
        "mode": "quick",
        "started": started_ts,
        "model": args.model,
        "prompt_chars": len(args.prompt),
        "prompt_tag": prompt_tag,
        "ollama_url": args.ollama_url,
        "system": system_snapshot,
        "status": status,
        "rows_recorded": len(rows),
    }

    if version:
        meta["ollama_version"] = version

    write_json(out_dir / "meta.json", meta)

    metrics = {
        "ttft_sec": _median([row["ttft_sec"] for row in rows if row.get("ttft_sec") is not None]),
        "decode_toks_per_sec": _median(
            [row.get("decode_toks_per_sec") for row in rows if row.get("decode_toks_per_sec") is not None]
        ),
        "ingest_toks_per_sec": _median(
            [row.get("ingest_toks_per_sec") for row in rows if row.get("ingest_toks_per_sec") is not None]
        ),
        "total_time_sec": _median([row["total_time_sec"] for row in rows if row.get("total_time_sec") is not None]),
    }

    if not PANDAS_AVAILABLE:
        _print_warning("pandas not installed - CSV summarisation limited")

    return {
        "out_dir": str(out_dir),
        "rows_written": len(rows),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "metrics_median": metrics,
        "status": status,
    }


def _suite_to_jobs(suite_path: str) -> "SuiteConfig":
    try:
        suite = load_suite(suite_path)
    except SuiteValidationError as exc:
        raise SystemExit(str(exc)) from exc
    return suite


def _run_suite(args: argparse.Namespace) -> Dict[str, Any]:
    suite_path = Path(args.suite)
    if getattr(args, "sync_models", False):
        _sync_suite_models(suite_path, quiet=args.quiet)

    suite = _suite_to_jobs(suite_path)
    out_dir = Path(suite.out_dir or DEFAULT_OUT_DIR)
    ensure_directory(out_dir)

    client = OllamaClient(suite.ollama_url)
    runner = BenchmarkRunner(client, quiet=args.quiet)

    system_info = collect_system_info()
    ollama_version = client.fetch_ollama_version()

    rows: List[Dict[str, Any]] = []
    pulled_models: Set[str] = set()
    interrupted = False
    try:
        models = [job.model for job in suite.generation] + [job.model for job in suite.embeddings]
        _ensure_models_available(
            client,
            models,
            auto_pull=args.auto_pull,
            pulled_models=pulled_models,
            quiet=args.quiet,
        )
        try:
            for job in suite.generation:
                try:
                    rows.extend(runner.run_generation_job(job))
                except BenchmarkInterrupted as exc:
                    rows.extend(exc.rows)
                    interrupted = True
                    break
            if not interrupted:
                for job in suite.embeddings:
                    try:
                        rows.extend(runner.run_embedding_job(job))
                    except BenchmarkInterrupted as exc:
                        rows.extend(exc.rows)
                        interrupted = True
                        break
        except KeyboardInterrupt:
            interrupted = True
    finally:
        client.close()
        if args.auto_pull and pulled_models and not args.keep_pulled:
            _cleanup_pulled_models(pulled_models, quiet=args.quiet)

    status = "interrupted" if interrupted else "completed"
    if interrupted and not args.quiet:
        print("[ollama-bench] benchmark interrupted; partial results saved", file=sys.stderr)

    outputs = write_reports(rows, out_dir)

    meta = {
        "bench_version": __version__,
        "started": rows[0]["ts"] if rows else system_info["timestamp_utc"],
        "suite_name": suite.name,
        "ollama_url": suite.ollama_url,
        "system": system_info,
        "status": status,
        "rows_recorded": len(rows),
    }
    if ollama_version:
        meta["ollama_version"] = ollama_version

    write_json(out_dir / "meta.json", meta)

    return {
        "suite": suite.name,
        "out_dir": str(out_dir),
        "rows_written": len(rows),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "status": status,
    }


def _run_validate(args: argparse.Namespace) -> Dict[str, Any]:
    suite = _suite_to_jobs(args.suite)
    return {
        "suite": suite.name,
        "generation_jobs": len(suite.generation),
        "embedding_jobs": len(suite.embeddings),
        "ollama_url": suite.ollama_url,
        "status": "ok",
    }
def _run_doctor(args: argparse.Namespace) -> Dict[str, Any]:
    client = OllamaClient(args.ollama_url)

    suite_path = getattr(args, "suite", None)
    suite_error: Optional[str] = None
    preflight: Optional[Dict[str, Any]] = None
    system_info: Dict[str, Any] = {}
    connectivity = False
    ollama_version: Optional[str] = None
    pulled_models: Set[str] = set()

    if args.auto_pull and not suite_path:
        raise SystemExit("--auto-pull requires --suite")

    try:
        system_info = collect_system_info()
        connectivity = client.check_connectivity()
        ollama_version = client.fetch_ollama_version()

        if suite_path:
            if getattr(args, "sync_models", False):
                _sync_suite_models(Path(suite_path), quiet=args.quiet)
            try:
                suite = load_suite(suite_path)
            except SuiteValidationError as exc:
                suite_error = str(exc)
            else:
                suite_models = [job.model for job in suite.generation] + [job.model for job in suite.embeddings]
                preflight = _doctor_model_preflight(
                    client,
                    suite_models,
                    auto_pull=args.auto_pull,
                    pulled_models=pulled_models,
                    quiet=args.quiet,
                )
    finally:
        client.close()

    tools = _gather_tool_status(system_info)

    diagnostics = {
        "ollama_url": args.ollama_url,
        "ollama_connectivity": connectivity,
        "ollama_version": ollama_version,
        "tools": tools,
        "system": system_info,
        "pandas_available": PANDAS_AVAILABLE,
        "suite_path": suite_path,
        "suite_error": suite_error,
        "preflight": preflight,
    }
    return diagnostics


def _command_version(cmd: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return None
        return (result.stdout or result.stderr).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _print_doctor(diagnostics: Dict[str, Any]) -> None:
    print("Ollama doctor report")
    print(f"  Ollama URL: {diagnostics['ollama_url']}")
    version = diagnostics.get("ollama_version")
    if version:
        print(f"  Ollama version: {version}")
    print(f"  Connectivity: {'ok' if diagnostics['ollama_connectivity'] else 'failed'}")
    print("  pandas installed:", "yes" if diagnostics["pandas_available"] else "no")
    print("  Tools:")
    for tool, info in diagnostics["tools"].items():
        status = info.get("status")
        detail = info.get("detail")
        if status == "found" and detail:
            print(f"    - {tool}: {status} ({detail})")
        else:
            print(f"    - {tool}: {status}")
    print("  System overview:")
    for key, value in diagnostics["system"].items():
        if key in {"system_profiler", "nvidia_smi", "tegrastats", "powermetrics_hint"}:
            continue
        print(f"    {key}: {value}")
    suite_path = diagnostics.get("suite_path")
    if suite_path:
        print(f"  Suite preflight ({suite_path}):")
        suite_error = diagnostics.get("suite_error")
        if suite_error:
            print(f"    invalid suite: {suite_error}")
        else:
            preflight = diagnostics.get("preflight") or {}
            if not preflight:
                print("    no models listed in suite")
            else:
                status = "ok" if preflight.get("ok") else "issues detected"
                print(f"    status: {status}")
                missing = preflight.get("models_missing") or []
                if missing:
                    print("    missing models:")
                    for model in missing:
                        print(f"      - {model}")
                query_errors = preflight.get("model_query_errors") or []
                if query_errors:
                    print("    model query errors:")
                    for err in query_errors:
                        print(f"      - {err}")
                resource_errors = preflight.get("resource_errors") or []
                if resource_errors:
                    print("    resource blockers:")
                    for err in resource_errors:
                        print(f"      - {err}")
                resource_warnings = preflight.get("resource_warnings") or []
                if resource_warnings:
                    print("    resource warnings:")
                    for warn in resource_warnings:
                        print(f"      - {warn}")
                autopulled = preflight.get("autopulled_models") or []
                if autopulled:
                    print("    auto-pulled models (removed after run unless kept):")
                    for model in autopulled:
                        print(f"      - {model}")


def _wants_json(args: argparse.Namespace) -> bool:
    return any(getattr(args, attr, False) for attr in ("json", "json_alias"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "quick":
        result = _run_quick(args)
    elif args.command == "run":
        result = _run_suite(args)
    elif args.command == "validate":
        result = _run_validate(args)
    elif args.command == "doctor":
        result = _run_doctor(args)
        if not _wants_json(args):
            _print_doctor(result)
    else:
        parser.error(f"Unknown command {args.command}")
        return 2

    if _wants_json(args):
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command in {"quick", "run"} and not args.quiet:
        print(f"[ollama-bench] outputs written to {result['out_dir']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
