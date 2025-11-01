# ollama-bench

`ollama-bench` is a cross-platform CLI utility for benchmarking [Ollama](https://ollama.ai) models. It focuses on reproducible measurements, portable telemetry, and a user-friendly experience across macOS, Linux (x86/NVIDIA), and Jetson devices.

## Features
- Streaming text generation benchmarks: warmups, TTFT, decode throughput, ingest throughput, total latency
- Embedding throughput trials
- System metadata capture with optional GPU/SoC telemetry
- JSONL, CSV, and Markdown outputs with median aggregates (when `pandas` is installed)
- YAML suite runner and quick single-prompt benchmark
- Deterministic defaults (`temperature=0`, `seed=1`) with opt-in overrides

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Quickstart
Run a quick single-prompt benchmark:
```bash
ollama-bench quick --model llama3:8b --prompt "Hello, Ollama!"
```

If the model is not already loaded locally, the command will exit and prompt you to run `ollama pull <model>` (or `ollama run <model>` once) before retrying.
ollama-bench also estimates whether the model can fit into current system/GPU memory and exits early with guidance if resources look too tight.
Add `--auto-pull` to let the tool fetch missing models automatically; by default they are removed after the run (use `--keep-pulled` if you want to retain them). If you interrupt a run with `Ctrl+C`, completed trials are still written out and the summary reports `status: interrupted` so you can resume later.

Run a full suite defined in YAML:
```bash
ollama-bench run --suite examples/bench_suite.yaml
ollama-bench run --suite examples/bench_suite.yaml --sync-models --auto-pull --keep-pulled
```
The sample suite covers multiple model families (LLaMA, Mistral/Mixtral, Qwen) plus two embedding backends so you can benchmark alternatives side by side—copy it and prune to match the models you have pulled locally.

Validate a suite without executing it:
```bash
ollama-bench validate --suite examples/bench_suite.yaml
```

Check environment readiness:
```bash
ollama-bench doctor
```
Add `--suite path/to/bench.yaml` to preflight model availability and memory requirements before a long run.
Use `--sync-models` to refresh suite entries against the live Ollama catalog before validation, and `--auto-pull` to fetch anything missing (use `--keep-pulled` if you want to retain them afterward).
`doctor` only surfaces tooling expectations that match your OS (e.g., `system_profiler` on macOS, `nvidia-smi` on Linux/NVIDIA boxes).
Run `ollama-bench doctor --suite path/to/bench.yaml --sync-models --auto-pull --keep-pulled` to verify setup and keep any freshly pulled models cached.

## Suite Schema
See `examples/bench_suite.yaml` for a full example. Each suite can include:
- `name`, `ollama_url`, and `out_dir`
- `generation`: list of generation jobs (model, tag, prompt, warmup, repeats, options)
- `embeddings`: list of embedding jobs (model, tag, text, repeats)

## Methodology
- Warmup iterations are discarded and not written to results
- `TTFT` (time-to-first-token) is measured from request start to the first streamed token
- `decode_toks_per_sec` uses completion tokens divided by decode time
- `ingest_toks_per_sec` uses prompt tokens divided by TTFT (falls back to `None` under very small TTFT)
- Token counts use `/api/tokenize` when available, otherwise fall back to streamed token chunks
- All timestamps are recorded in UTC ISO 8601

## Telemetry Tips
- macOS: `system_profiler SPDisplaysDataType` is captured automatically (truncated to 2 KiB)
- NVIDIA GPUs: `nvidia-smi` one-shot query is attempted
- Jetson boards: `tegrastats --interval 1000 --count 1` is attempted
- If `powermetrics` is installed, a hint is included so you can launch it manually alongside benchmarks

## Reproducibility Checklist
- Pin Ollama model variants (e.g., `llama3:8b-instruct-q4_K_M`)
- Use consistent `options` (threads/ctx) via `--options-json`
- Record hardware metadata produced in `meta.json`
- Repeat runs to ensure medians converge
- Capture sidecar telemetry (`nvidia-smi --loop=1`, `powermetrics`, etc.) into the output directory

## Development
Run tests (HTTP interactions are fully mocked):
```bash
pytest
```

## License
MIT License. See `LICENSE` for details.
