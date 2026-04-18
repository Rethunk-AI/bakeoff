# HUMANS.md — End-user guide

This guide covers **prerequisites, installation, running, configuring, and troubleshooting** the bakeoff benchmark. For **LLM/contributor onboarding, design invariants, and editing conventions**, see [`AGENTS.md`](AGENTS.md).

## What it does

Serves one or more GGUFs from `~/.lmstudio/models/` through a `llama.cpp` podman container and benchmarks them on **quality**, **latency**, and **cost** (energy). Supports any number of models with two judge modes: `pairwise_all` (round-robin tournament) and `scored` (absolute 1-5 rubric). Pick one — see [AGENTS § Judge mode selection](AGENTS.md#judge-mode-selection) for thresholds and cost formulas.

Emits JSON, Markdown, and a single-file HTML dashboard under `results/`.

Matrix: `tasks × prompt_variants × models`.

## Prerequisites

- **`podman`** — runs the llama.cpp container.
- **`uv`** — Python env management. See [installation](https://docs.astral.sh/uv/getting-started/installation/).
- One or more GGUFs under `~/.lmstudio/models/` (or wherever `server.models_dir` points). Fetch them with `./run.sh fetch` — see [Downloading models](#downloading-models).
- First container run pulls `ghcr.io/ggml-org/llama.cpp:server-vulkan` (~1 GB).

AMD users: the image uses Vulkan, works on ROCm-supported GPUs and APUs without needing the ROCm userspace stack to be fully functional.

## Install & run

```sh
./run.sh                       # dataset + all phases + reports
./run.sh --dry-run             # parse config + generate dataset only
./run.sh --config other.yaml   # alternate config
```

Without the wrapper:

```sh
uv venv .venv
uv pip install -r requirements.txt
uv run python -m bench.runner --config config.yaml
```

## Downloading models

`./run.sh fetch` pulls GGUFs from Hugging Face into `server.models_dir`, matching the `<repo_id>/<filename>` layout that `config.yaml`'s `gguf:` paths already assume — so no config edits after fetching.

```sh
./run.sh fetch                               # every missing gguf in config.yaml
./run.sh fetch --list                        # dry-run: show plan + total size
./run.sh fetch <repo_id> <filename>          # ad-hoc, one file
./run.sh fetch --config other.yaml           # alternate config
```

Ad-hoc example (pulls `<models_dir>/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf`):

```sh
./run.sh fetch lmstudio-community/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf
```

- Existing files are skipped (idempotent).
- Gated repos (Llama, Gemma): `hf auth login` first, or export `HF_TOKEN=<token>`.
- Faster transfers: `export HF_HUB_ENABLE_HF_TRANSFER=1` (optional; installs separately).

## Configuration

`config.yaml` is the single source of truth. Common edits:

- **Add models** — append entries under `models:`. No two-model limit. First two entries are A / B for pairwise judging; additional models are added to the tournament (pairwise) or the absolute rubric (scored).
- **Prompt variants** — `prompts:` list; every task runs against every prompt against every model.
- **Judge mode** — `judge.mode: pairwise_all` (default) or `judge.mode: scored`. Threshold + cost formulas live in [AGENTS § Judge mode selection](AGENTS.md#judge-mode-selection).
- **Skip the judge entirely** — `judge.enabled: false`. Quality column shows only heuristic scores; tasks with `scorer: "judge"` emit `null`.
- **Per-model context** — set `ctx:` on a model entry to override `server.ctx` (useful when one model has larger KV demands).
- **MoE model OOM** — uncomment `n_cpu_moe: 999` on the model entry to spill experts to CPU.
- **Skip `mmproj-*` files** — vision projectors that ship alongside multimodal GGUFs in LM Studio. Not standalone text models. Do not list under `models:`.

## Output

- **`results/run-<ts>.json`** — full record: config snapshot, tasks, per-call metrics, judgements. Each judgement is tagged `mode: "pairwise" | "scored"`; pairwise entries also carry slot `order`.
- **`results/run-<ts>.md`** — per-model rollup. Pairwise mode adds W/L/T table + N×N win-rate matrix. Scored mode adds a `mean ± sd` column.
- **`results/run-<ts>.html`** — dashboard (open in browser). Matrix + overall win-rate chart in pairwise mode, mean-score chart in scored mode.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `serve.sh: port N already in use` | Another benchmark run, `lazy-local`, or LM Studio itself holds the port. Stop it or change `server.port`. Don't `podman rm -f` blindly — `lazy-local` may own the container. |
| `serve.sh: gguf not found: ...` | Relative path typo. List real files with `fd -e gguf . ~/.lmstudio/models/` and copy a matching entry into `config.yaml`. |
| `HTTPError 404 /v1/chat/completions` | Container is up but the `/v1` router isn't ready yet. Raise `server.boot_timeout_s`. |
| `cost_usd: null` everywhere | Normal on Strix Halo (broken `libdrm_amdgpu.so`) and on non-GPU hosts. No action unless you also have `nvidia-smi`. |
| Judge returns mostly TIE | Judge model too weak for the task pool. Swap `judge.gguf` to a stronger local model or increase `judge.ctx`. |
| Judge call cost feels high with >4 models | Switch to `judge.mode: scored`. Cost drops from `C(N,2)` to `N` per `(task, prompt)`. |
| Reasoning model answers missing | Qwen3, DeepSeek-R1, etc. return the final answer in `content` and chain-of-thought in `reasoning_content`. The client prefers `content` and falls back to `reasoning_content` if empty. Pair two reasoning models for a fair fight, or thread `--reasoning-budget` through `EXTRA_ARGS` in `bin/serve.sh`. |
| Teardown didn't kill a container | Run `./bin/serve.sh down-all` to nuke every `bench-llama-*` container left behind. |

## Clean-up

```sh
./bin/serve.sh down-all        # remove all bench-llama-* containers
rm -rf .venv results datasets  # nuke generated state
```

## More detail

- **Design choices & invariants:** [`AGENTS.md`](AGENTS.md)
- **Code layout:** [`AGENTS.md` § Layout](AGENTS.md#layout)
- **Configuration contract:** [`config.yaml`](config.yaml) (inline comments)
