# bakeoff — project notes for agents

Local LLM N-vs-N benchmark. LM Studio GGUFs served via `podman` + `ghcr.io/ggml-org/llama.cpp:server-vulkan`. OpenAI-compatible `/v1/chat/completions`.

Global rules (tooling, git, commits, MCP preference) live in `~/.claude/CLAUDE.md`. Do not restate them here. This file is project-specific context only.

## Layout

```
config.yaml          single source of truth (server, models, prompts, dataset, judge, cost, output)
bin/serve.sh         podman launcher / teardown (supports `down-all`)
bench/
  clients.py         httpx OpenAI-compat client; prefers `content`, falls back to `reasoning_content`
  dataset.py         seeded synthetic tasks (qa / code / summarize / classify)
  metrics.py         heuristic scorers + judge prompts + nvidia-smi / rocm-smi power sampling
  runner.py          boot → run all calls → teardown, per model; then judge phase
  report.py          JSON + Markdown + single-file HTML dashboard (Chart.js via CDN)
run.sh               uv venv + uv pip install + uv run
datasets/ results/   generated artifacts (gitignored)
```

## Design invariants (don't break silently)

- **One model in VRAM at a time.** Unified-memory APU (Strix Halo / Radeon 8060S) can't hold A + B + judge concurrently. Phases are sequential by design — don't "optimize" to parallel boot.
- **Judge runs as its own phase** after every model has completed and been torn down. Judge results reference stored per-call outputs, not live model state.
- **Pairwise order randomized per call** (seeded from `run.seed`); swapped verdicts inverted before counting. Every judgement records `order: "AB" | "BA"`. Mitigates 5-15% positional bias.
- **Cost axis is energy, not tokens.** `nvidia-smi --query-gpu=power.draw` or `rocm-smi --showpower` sampled at call start + end. Neither available → `energy_wh` / `cost_usd` set to `null`. Do not substitute latency.
- **`mmproj-*` files are vision projectors, not standalone models.** Never list under `models:`.

## Hardware caveats

- Strix Halo `rocm-smi` typically fails on `libdrm_amdgpu.so`. `cost_usd: null` in results is expected, not a bug.
- Vulkan image works on AMD/NVIDIA/Intel without per-backend wrangling. Don't switch to a ROCm-specific image "to fix" Strix Halo — that regresses portability.
- MoE models (`Qwen3.6-35B-A3B` etc.): if boot OOMs, set `n_cpu_moe: 999` on the model entry to spill experts to CPU.

## Judge mode selection

- N ∈ {2, 3, 4} → `pairwise_all` (cost `C(N,2) × tasks × prompts`, sharp ranking).
- N ≥ 5 → `scored` (cost `N × tasks × prompts`, linear).
- `judge.enabled: false` → heuristic scores only; `scorer: "judge"` tasks emit `null`.

## Running

```sh
./run.sh                       # dataset + all phases + reports
./run.sh --dry-run             # parse config + generate dataset only
./run.sh --config other.yaml   # alternate config
./bin/serve.sh down-all        # nuke every bench-llama-* container left behind
```

Raw path (no wrapper):

```sh
uv venv .venv
uv pip install -r requirements.txt
uv run python -m bench.runner --config config.yaml
```

## Gotchas

- `serve.sh: port N already in use` — `lazy-local`, prior bench run, or LM Studio itself holds it. Change `server.port` or stop the other process. Do not `podman rm -f` blindly — `lazy-local` may own the container.
- `HTTPError 404 /v1/chat/completions` — container booted but router not ready. Raise `server.boot_timeout_s`, don't retry-loop faster.
- Judge returns mostly TIE — judge too weak for the task pool. Swap `judge.gguf` to a stronger model or bump `judge.ctx`.
- Reasoning models (Qwen3, DeepSeek-R1) put chain-of-thought in `reasoning_content`. Client prefers `content`; pair two reasoning models for a fair fight, or thread `--reasoning-budget` through `EXTRA_ARGS` in `bin/serve.sh`.
- Teardown leak — always follow up failed runs with `./bin/serve.sh down-all` before launching the next.

## When editing

- `config.yaml` is the contract. Add new knobs there first, then wire through `runner.py` / `serve.sh`. Don't hard-code.
- Every new scorer/judge mode must preserve the JSON record shape in `results/run-<ts>.json` — the HTML dashboard reads it verbatim.
- Python env: `uv`. No `python -m venv`, no bare `pip`.
