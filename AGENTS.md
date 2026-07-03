# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- codegraph-stanza:start v2 -->
## Code intelligence: use CodeGraph first

This repo has a CodeGraph index at `.codegraph/codegraph.db`. For symbol/structure questions, prefer the `codegraph_*` MCP tools — they resolve in 1 call what Read/Grep/Glob need 3–8 calls for, at a fraction of the context cost. The "Architecture" section below tells you *what* lives where; CodeGraph tells you *how* it connects.

### First-time setup this session (one ToolSearch call)

Before your first `codegraph_*` call this session, load the tool schemas in one shot:

```
ToolSearch query="select:codegraph_search,codegraph_callers,codegraph_callees,codegraph_node,codegraph_explore,codegraph_context,codegraph_impact,codegraph_files,codegraph_status"
```

The codegraph MCP tools are deferred by default — their schemas aren't in scope until you fetch them. Skip this and your first call fails with `InputValidationError`, and you'll be tempted to give up and `Grep`. Don't.

### Always start with CodeGraph for these intents

| Intent | Use this | Don't use |
|---|---|---|
| "What's this repo about?" / get oriented | `codegraph_explore topic="overview"` | recursive `Glob` + `Read` |
| Find a function, class, type, or interface by name | `codegraph_search query="<name>"` (filter with `kind:"class"`, etc.) | `Grep "class <name>"` |
| Who calls `foo`? | `codegraph_callers symbol="foo"` | `Grep "foo("` |
| What does `foo` call? | `codegraph_callees symbol="foo"` | reading `foo`'s file |
| Show me `foo`'s implementation | `codegraph_node symbol="foo" includeCode=true` | `Grep` + `Read` whole file |
| Blast radius before changing `Foo` | `codegraph_impact symbol="Foo"` | manual cross-grepping |
| Build context for a multi-file feature | `codegraph_context task="<short task>"` | multiple `Read`s |

### Cross-repo queries

Sibling repos under `/opt/dev/*` may have their own CodeGraph indexes. To query a sibling without `cd`, pass `projectPath`:

```
codegraph_search query="dashboard" projectPath="/opt/dev/projIndex"
```

If unsure which sibling holds a feature, run the same `codegraph_search` against the few likely candidates in parallel — still cheaper than recursive `Glob`.

### CodeGraph answers structure, not product requirements

CodeGraph provides **code context**, not product requirements. For new features, still ask the user about:

- UX preferences and behavior
- Edge cases and error handling
- Acceptance criteria

### Do NOT use CodeGraph for

CodeGraph indexes AST symbol *names*, not raw bytes. Use `Grep` for:

- Config keys/values in `pyproject.toml`, env-var names, URLs, error strings, regex patterns
- Markdown/prose in `README.md`, `docs/`, code comments
- Python `import` / `from ... import` *strings* — symbol search misses these


If a CodeGraph call returns nothing useful, **then** fall back to Read/Grep/Glob — not before.
<!-- codegraph-stanza:end v2 -->

## Project Overview

ollama-bench is a cross-platform CLI utility for benchmarking Ollama models. It measures streaming text generation (TTFT, decode throughput, ingest throughput) and embedding throughput, with system metadata capture and multiple output formats (JSONL, CSV, Markdown, HTML).

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e .            # installs as editable with 'ollama-bench' CLI entry point
pip install -e ".[dev]"     # adds pytest

# Run tests (all mocked, no Ollama needed)
pytest
pytest tests/test_schemas.py          # single test file
pytest tests/test_metrics.py::test_generation_metrics  # single test

# CLI usage
ollama-bench quick --model llama3:8b --prompt "Hello"
ollama-bench run --suite examples/bench_suite.yaml
ollama-bench validate --suite examples/bench_suite.yaml
ollama-bench doctor
ollama-bench show --out-dir bench_out --print-results
```

## Architecture

### Package layout (`src/ollamabench/`)

- **cli.py** — Entry point (`main()`). Argparse-based CLI with subcommands: `quick`, `run`, `validate`, `doctor`, `show`. Contains HTML report generation, model resource estimation, Ollama catalog sync, and auto-pull/cleanup logic. This is by far the largest module (~1600 lines).
- **bench.py** — `OllamaClient` (HTTP wrapper for Ollama API) and `BenchmarkRunner` (orchestrates warmup iterations, timed trials, metric collection). Raises `BenchmarkInterrupted` on Ctrl+C with partial results preserved.
- **schemas.py** — Frozen dataclasses (`SuiteConfig`, `GenerationJobConfig`, `EmbeddingJobConfig`) and `load_suite()` YAML loader with strict validation. Raises `SuiteValidationError`.
- **report.py** — `write_reports()` produces JSONL + CSV + optional Markdown/pandas summary. Conditionally imports pandas (`PANDAS_AVAILABLE` flag).
- **io.py** — Low-level filesystem helpers: `write_json`, `write_jsonl`, `write_csv`, `ensure_directory`.
- **system.py** — `collect_system_info()` gathers hardware metadata (RAM, CPU, GPU via nvidia-smi, macOS system_profiler, Jetson tegrastats).

### Key design patterns

- **Deterministic defaults**: `temperature=0`, `seed=1` in `DEFAULT_OPTIONS` (bench.py), overridable via `--options-json`.
- **Graceful interruption**: `BenchmarkInterrupted` exception carries partial `rows` so Ctrl+C still writes results with `status: interrupted`.
- **Test isolation**: Tests use `StubClient` and `monkeypatch` for `time.monotonic` — no real HTTP calls. The bench module is patched at `ollamabench.bench.time.monotonic`.
- **Output directories**: `bench_out/` (suite runs), `bench_out_quick/` (quick runs), `comp_bench_out/` (multi-system comparisons). All gitignored.

### Data flow

1. CLI parses args → loads/validates suite YAML via `schemas.load_suite()`
2. `_ensure_models_available()` checks local models, optionally auto-pulls
3. `BenchmarkRunner` runs warmup + timed trials via `OllamaClient`
4. Results go through `report.write_reports()` → JSONL/CSV/Markdown
5. `collect_system_info()` + meta written to `meta.json`
6. Optional HTML dashboard generated and opened in browser

## Conventions

- Python >=3.10, uses `from __future__ import annotations` everywhere
- Frozen dataclasses for config objects (immutable after creation)
- All modules export `__all__`
- Timestamps in UTC ISO 8601
- Suite YAML schema documented in `examples/bench_suite.yaml`

<!-- projindex-cascaded-must-dos v2 -->

## Must-Dos — cascaded standards for every /opt/dev/ repo

These are hard requirements inherited by every project created via projIndex.
Treat them as non-negotiable, not suggestions.

### 1. Secrets live in the macOS Keychain behind Touch ID — never in plaintext

Any API key, token, or secret this project needs MUST be retrieved from the macOS
Keychain via Touch ID (Secure Enclave) — NEVER from a plaintext `.env`, a shell rc,
or a plaintext `security add-generic-password` item.

- Store and retrieve secrets with the Touch-ID-gated pattern from the
  `elevenlabs-touchid-keychain` skill (or `bitwarden-touchid-macos` for
  Bitwarden-backed secrets).
- Never commit secrets. Any `.env*` file holding real values must be gitignored.
- Don't use `SecItemAdd` biometric ACLs — they fail with
  `errSecMissingEntitlement -34018` under an ad-hoc code signature.
- If a secret is ever committed, pushed, or found exposed on disk, ROTATE it
  immediately (once, across every repo sharing it) — rotation is the real fix;
  gitignore entries and history rewrites do not un-leak an exposed key.

### 2. Confirm destructive ops — ask before anything irreversible

Ask the user before `rm -rf`, force-push, history rewrites, schema drops,
dependency removal, bulk overwrites, or anything that affects shared state or
may destroy the only copy of something.

- State the exact scope (paths, count of affected items) when asking.
- Never emit a delete without a paired backup/preservation mechanism when one
  exists (e.g. `--delete` always with `--backup-dir`).
- Never silently auto-apply bulk or regenerative writes — surface the plan
  first and let the user pick the mode.

### 3. Verify before you commit — the repo's gate must pass, and logic changes carry tests

Run the repo's declared verification suite (type-check, lint, format, tests —
whatever the repo's CLAUDE.md/AGENTS.md or CI defines) and require it to pass
before every commit or push that touches logic.

- New or changed logic lands with tests in the same change; "I'll add tests
  later" and "it's a one-liner" are not justifications. If a change is truly
  test-neutral, say so in the commit message.
- Never weaken or disable a failing check (linter, hook, test) to get green —
  fix the code.

### 4. If a change makes anything else stale, update it in the same change

A change is not done while any duplicated copy, derived artifact, or doc that
describes the changed thing is stale.

- Declared duplicates (version in two files, config + its `.example`, copies-not-
  symlinks, parallel generators): apply the edit to EVERY copy in the same commit.
- The repo's instruction file (CLAUDE.md/AGENTS.md) and its designated
  source-of-truth docs: correct any port, status, convention, or architecture
  line your change invalidates, in the same commit.
- Stale instructions poison every future agent session; wrong is worse than missing.
