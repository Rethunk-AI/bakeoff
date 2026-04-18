# Local LLM N-vs-N Benchmark (LM Studio GGUFs via llama.cpp)

[![ci](https://github.com/Rethunk-AI/bakeoff/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Rethunk-AI/bakeoff/actions/workflows/ci.yml)
[![license](https://img.shields.io/github/license/Rethunk-AI/bakeoff)](LICENSE)
[![python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](pyproject.toml)

Small harness that serves models from `~/.lmstudio/models/` through a `llama.cpp` podman container and benchmarks them on **quality**, **latency**, and **cost** (energy). Supports any number of models: round-robin tournament (`pairwise_all`) or absolute rubric (`scored`). Emits JSON, Markdown, and a single-file HTML dashboard.

Matrix: `tasks × prompt_variants × models`.

## Documentation

**Start here:**

- **[HUMANS.md](HUMANS.md)** — operators & developers: prerequisites, install, run, configure, troubleshoot, clean up.
- **[AGENTS.md](AGENTS.md)** — LLMs & contributors: design invariants, hardware caveats, judge-mode selection, editing conventions. `CLAUDE.md` is a symlink here.

**Reference:**

- **[config.yaml](config.yaml)** — single source of truth for server, models, prompts, dataset, judge, cost, output. Inline comments describe every knob.

## Design choices (explicit)

| Concern | Choice | Reason |
|---|---|---|
| Serving | `podman` + `ghcr.io/ggml-org/llama.cpp:server-vulkan` | Works on AMD (Strix Halo tested), NVIDIA, Intel without per-backend wrangling. No `llama-server` binary ships in LM Studio's own `~/.lmstudio/extensions/backends/*/` — only `.so` libraries for LM Studio's internal runtime. |
| One model at a time | Sequential phase per model: boot → run all calls → tear down → next | Unified-memory APUs (and modest-VRAM discrete GPUs) can't hold A + B + judge concurrently. Serial is honest. |
| Transport | OpenAI-compatible `/v1/chat/completions` | llama.cpp server exposes it; one client class works for Ollama, vLLM, LM Studio, llama.cpp. |
| Quality scorer | `pairwise_all` tournament (default) or `scored` 1-5 rubric via LLM judge; plus heuristic (`exact` / `contains` / `regex`) for structured tasks | Tournament gives sharp ranking on small N (2-4); rubric scales linearly for larger N. Heuristics catch hard ground-truth items without a judge round trip. |
| Pairwise positional-bias mitigation | Order randomized per call (seeded from `run.seed`); swapped verdicts inverted before counting | Judges show a 5-15% preference for slot A; flipping per call averages it out across the matrix. `order: "AB" \| "BA"` is stored on every judgement. |
| Default context | `4096` | Benchmark prompts are short; keeping `ctx` small cuts load time + memory. Override per model or in `server.ctx`. |
| "Cost" for local models | Energy estimate via `nvidia-smi --query-gpu=power.draw` **or** `rocm-smi --showpower` sampled at call start + end; average × wall time × `$/kWh` | No per-token price for local. Energy is the honest cost axis. |
| Cost fallback | `energy_wh` / `cost_usd` = `null` when neither tool works | No silent substitution with latency. On Strix Halo, `rocm-smi` often fails on `libdrm_amdgpu.so` — expect `null`. |
| Dataset | Generated from seeded templates across `qa` / `code` / `summarize` / `classify` | Simple stack, no external corpus. Deterministic via `run.seed`. |
| Stack | Python + `httpx` + `pyyaml`, stdlib for everything else | No promptfoo / lm-eval / framework. |
| Dashboard | One static HTML file with Chart.js via CDN, reads embedded run JSON | No build step, opens from disk. |

## Layout

```
config.yaml          server, models, prompts, dataset, judge, cost, output
bin/serve.sh         minimal podman launcher for llama.cpp:server-vulkan
bench/
  clients.py         OpenAI-compat HTTP client (httpx)
  dataset.py         seeded synthetic task generator
  download.py        fetch GGUFs from Hugging Face into models_dir
  metrics.py         heuristic scoring, pairwise judge, nvidia/rocm power sampling
  runner.py          sequential boot → run → teardown orchestrator
  report.py          markdown + HTML dashboard emit
run.sh               uv venv + install + run
datasets/            generated tasks-<ts>.jsonl
results/             run-<ts>.json / .md / .html
```

## Quick start

```sh
./run.sh fetch    # pull GGUFs referenced in config.yaml from Hugging Face
./run.sh          # dataset + all phases + reports
```

For prerequisites, configuration, and troubleshooting see [HUMANS.md](HUMANS.md).
