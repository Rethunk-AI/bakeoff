# Local LLM N-vs-N Benchmark (LM Studio GGUFs via llama.cpp)

Small harness that serves models from `~/.lmstudio/models/` through a
`llama.cpp` podman container and benchmarks them on **quality**, **latency**,
and **cost** (energy). Supports any number of models: round-robin tournament
(`pairwise_all`) or absolute rubric (`scored`). Emits JSON, Markdown, and a
single-file HTML dashboard.

Matrix: `tasks × prompt_variants × models`.

## Design choices (explicit)

| Concern | Choice | Reason |
|---|---|---|
| Serving | `podman` + `ghcr.io/ggml-org/llama.cpp:server-vulkan` | Works on AMD (Strix Halo tested), NVIDIA, Intel without per-backend wrangling. No `llama-server` binary ships in LM Studio's own `~/.lmstudio/extensions/backends/*/` — only `.so` libraries for LM Studio's internal runtime. |
| One model at a time | Sequential phase per model: boot → run all calls → tear down → next | Unified-memory APUs (and modest-VRAM discrete GPUs) can't hold A + B + judge concurrently. Serial is honest. |
| Transport | OpenAI-compatible `/v1/chat/completions` | llama.cpp server exposes it; one client class works for Ollama, vLLM, LM Studio, llama.cpp. |
| Quality scorer | `pairwise_all` tournament (default) or `scored` 1-5 rubric, both via LLM-judge; plus heuristic (`exact` / `contains` / `regex`) for structured tasks | Tournament gives sharp ranking on small N (2-4); rubric scales linearly for larger N. Heuristics catch hard ground-truth items without a judge round trip. |
| Pairwise positional-bias mitigation | Order randomized per call (seeded from `run.seed`); swapped verdicts are inverted before counting | Judges show a 5-15% preference for slot A; flipping per call averages it out across the matrix and keeps the record honest. `order: "AB" \| "BA"` is stored on every judgement. |
| Default context | `4096` | User explicitly keeps `ctx` small when testing directly — cuts load time + memory. Override per model or in `server.ctx`. |
| "Cost" for local models | Energy estimate: sample `nvidia-smi --query-gpu=power.draw` **or** `rocm-smi --showpower` at start + end of each call, average × wall time × `$/kWh` | No per-token price for local. Energy is the honest cost axis. |
| Cost fallback | `energy_wh` / `cost_usd` = `null` when neither tool works | No silent substitution with latency. On Strix Halo, `rocm-smi` often fails on `libdrm_amdgpu.so` — expect `null` and live with it. |
| Dataset | Generated from seeded templates across `qa` / `code` / `summarize` / `classify` | Simple stack, no external corpus. Deterministic via `run.seed`. |
| Stack | Python + `httpx` + `pyyaml`, stdlib for everything else | No promptfoo / lm-eval / framework. |
| Dashboard | One static HTML file with Chart.js via CDN, reads embedded run JSON | No build step, opens from disk. |

## Layout

```
config.yaml          # server, models, prompts, dataset, judge, cost, output
bin/
  serve.sh           # minimal podman launcher for llama.cpp:server-vulkan
bench/
  clients.py         # OpenAI-compat HTTP client (httpx)
  dataset.py         # seeded synthetic task generator
  metrics.py         # heuristic scoring, pairwise judge, nvidia/rocm power sampling
  runner.py          # sequential boot → run → teardown orchestrator
  report.py          # markdown + HTML dashboard emit
run.sh               # uv venv + install + run
datasets/            # generated tasks-<ts>.jsonl
results/             # run-<ts>.json / .md / .html
```

## Prerequisites

- `podman` (to run the llama.cpp container)
- `uv` (Python env management — see [installation](https://docs.astral.sh/uv/getting-started/installation/))
- One or more GGUFs under `~/.lmstudio/models/` (or wherever `server.models_dir` points)
- First container run pulls `ghcr.io/ggml-org/llama.cpp:server-vulkan` (~1 GB)

AMD users: image uses Vulkan, works on ROCm-supported GPUs and APUs without needing the ROCm userspace stack to be fully functional.

## Usage

```sh
./run.sh                       # dataset + all phases + reports
./run.sh --dry-run             # parse config + generate dataset only
./run.sh --config other.yaml   # different config
```

Without the wrapper:

```sh
uv venv .venv
uv pip install -r requirements.txt
uv run python -m bench.runner --config config.yaml
```

## Configuration pointers

- **Add models**: append entries under `models:`. No two-model limit.
- **Prompt variants**: `prompts:` list — every task runs against every prompt against every model.
- **Judge mode**: `judge.mode: pairwise_all` (default) or `judge.mode: scored`. Pick based on N:
  - N=2-4 → pairwise_all (sharp ranking; cost `C(N,2) × tasks × prompts`)
  - N≥5 → scored (linear cost `N × tasks × prompts`)
- **Skip judge entirely**: `judge.enabled: false` → quality column only shows heuristic scores; tasks with `scorer: "judge"` come out `null`.
- **Per-model context**: set `ctx:` under a model entry to override `server.ctx` (useful if one model has larger KV demands).
- **MoE model OOM**: uncomment `n_cpu_moe: 999` under the model entry to spill experts to CPU.
- **Skip `mmproj-*` files**: these are vision projectors that ship alongside multimodal GGUFs in LM Studio — not standalone text models. Do not list them under `models:`.

## Output

- `results/run-<ts>.json` — full record (config, tasks, per-call metrics, judgements — each judgement tagged `mode: "pairwise" | "scored"` with slot `order` for pairwise entries).
- `results/run-<ts>.md` — per-model rollup; pairwise mode adds W/L/T table + NxN win-rate matrix; scored mode adds a `mean ± sd` column.
- `results/run-<ts>.html` — dashboard (open in browser). Shows matrix + overall win-rate chart in pairwise mode, mean-score chart in scored mode.

## Troubleshooting

- `serve.sh: port N already in use` — another benchmark run, `lazy-local`, or LM Studio itself holds the port. Stop it or change `server.port`.
- `serve.sh: gguf not found: ...` — relative path typo. List real files with `fd -e gguf . ~/.lmstudio/models/` and copy a matching entry into `config.yaml`.
- `HTTPError 404 /v1/chat/completions` — container is up but the `/v1` router not ready. Raise `server.boot_timeout_s`.
- `cost_usd: null` everywhere — normal on Strix Halo (broken `libdrm_amdgpu.so`) and on non-GPU hosts. Document only, no action needed unless you also have `nvidia-smi`.
- Judge returns mostly TIE — judge model too weak. Swap `judge.gguf` to a stronger local model or increase `judge.ctx`.
- Judge call cost feels high with >4 models — switch `judge.mode: scored`. Cost drops from `C(N,2)` to `N` per `(task, prompt)`.
- Reasoning models (Qwen3, DeepSeek-R1, etc.) return their final answer in `content` and their chain-of-thought in `reasoning_content`. The client prefers `content` and falls back to `reasoning_content` if empty. Pair two reasoning models for a fair fight, or set `--reasoning-budget` via `EXTRA_ARGS` when booting the server (edit `bin/serve.sh` to pass it through) if you need tighter control.
- Teardown didn't kill a container — run `./bin/serve.sh down-all` to nuke every `bench-llama-*` container left behind.

## Clean-up

```sh
./bin/serve.sh down-all        # remove all bench-llama-* containers
rm -rf .venv results datasets  # nuke generated state
```
