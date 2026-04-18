# HUMANS.md — End-user guide

This guide covers **prerequisites, installation, running, configuring, and troubleshooting** the bakeoff benchmark. For **LLM/contributor onboarding, design invariants, and editing conventions**, see [`AGENTS.md`](AGENTS.md).

## What it does

Serves one or more GGUFs from `~/.lmstudio/models/` through a [`llama-swap`](https://github.com/mostlygeek/llama-swap) proxy sitting in front of `llama.cpp` podman containers, and benchmarks them on **quality**, **latency**, and **cost** (energy). Supports any number of models with two judge modes: `pairwise_all` (round-robin tournament) and `scored` (absolute 1-5 rubric). Pick one — see [AGENTS § Judge mode selection](AGENTS.md#judge-mode-selection) for thresholds and cost formulas.

Emits JSON, Markdown, and a single-file HTML dashboard under `results/`.

Matrix: `tasks × prompt_variants × models`.

## Prerequisites

- **`podman`** — runs the llama.cpp container that `llama-swap` drives.
- **`uv`** — Python env management. See [installation](https://docs.astral.sh/uv/getting-started/installation/).
- **`curl`** + `sha256sum` (or `shasum`) — used by the bootstrap to fetch the pinned `llama-swap` binary.
- One or more GGUFs under `~/.lmstudio/models/` (or wherever `server.models_dir` points). Fetch them with `./run.sh fetch` — see [Downloading models](#downloading-models).
- First container run pulls `ghcr.io/ggml-org/llama.cpp:server-vulkan` (~1 GB).

The pinned `llama-swap` binary is fetched on the first benchmark run into `.cache/llama-swap/` and verified against an in-repo SHA256. Bumping the version is a `run.sh` edit — see [`AGENTS.md` § When editing](AGENTS.md#when-editing).

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

- **Add models** — append entries under `models:`. No two-model limit. First two entries are A / B for pairwise judging; additional models are added to the tournament (pairwise) or the absolute rubric (scored). `id` is the routing key the runner sends to `llama-swap`; `alias` is only what llama.cpp reports via `-a`.
- **Prompt variants** — `prompts:` list; every task runs against every prompt against every model.
- **Judge mode** — `judge.mode: pairwise_all` (default) or `judge.mode: scored`. Threshold + cost formulas live in [AGENTS § Judge mode selection](AGENTS.md#judge-mode-selection).
- **Skip the judge entirely** — `judge.enabled: false`. Quality column shows only heuristic scores; tasks with `scorer: "judge"` emit `null`.
- **Per-model context** — set `ctx:` on a model entry to override `server.ctx` (useful when one model has larger KV demands).
- **MoE model OOM** — uncomment `n_cpu_moe: 999` on the model entry to spill experts to CPU.
- **Proxy port** — `server.swap_port` (default `8080`) is the public listener the runner talks to. `server.backend_start_port` (default `5800`) is the base for `llama-swap`'s `${PORT}` allocation to backend containers. Change these only if another service already holds the ports.
- **Skip `mmproj-*` files** — vision projectors that ship alongside multimodal GGUFs in LM Studio. Not standalone text models. The harness refuses them in the generator.

## Output

- **`results/run-<ts>.json`** — full record: config snapshot, tasks, per-call metrics, judgements. Each judgement is tagged `mode: "pairwise" | "scored"`; pairwise entries also carry slot `order`.
- **`results/run-<ts>.md`** — per-model rollup. Pairwise mode adds W/L/T table + N×N win-rate matrix. Scored mode adds a `mean ± sd` column.
- **`results/run-<ts>.html`** — dashboard (open in browser). Matrix + overall win-rate chart in pairwise mode, mean-score chart in scored mode.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `llama-swap.sh: binary not found` | Bootstrap hasn't run yet. Execute `./run.sh` once; it fetches and SHA-verifies the pinned binary into `.cache/llama-swap/`. |
| `SHA256 mismatch for ...` during bootstrap | The pinned version's checksum no longer matches the downloaded tarball. Check the release on GitHub; update both `LLAMA_SWAP_VERSION` and the matching `LLAMA_SWAP_SHA256_*` line in `run.sh`. Never bypass the check. |
| Port `server.swap_port` already in use | `llama-swap` itself, another benchmark instance, or LM Studio holds it. Stop the other process or change `server.swap_port`. |
| Dry-run: `gguf must be '<org>/<repo>/<file>.gguf' form` | Path shape typo. List real files with `fd -e gguf . ~/.lmstudio/models/` and copy a matching entry into `config.yaml`. |
| `HTTPError 404 /v1/chat/completions` | Proxy is up but the requested model's backend is still loading. Raise `server.boot_timeout_s`. |
| `cost_usd: null` everywhere | Normal on Strix Halo (broken `libdrm_amdgpu.so`) and on non-GPU hosts. No action unless you also have `nvidia-smi`. |
| Judge returns mostly TIE | Judge model too weak for the task pool. Swap `judge.gguf` to a stronger local model or increase `judge.ctx`. |
| Judge call cost feels high with >4 models | Switch to `judge.mode: scored`. Cost drops from `C(N,2)` to `N` per `(task, prompt)`. |
| Reasoning model answers missing | Qwen3, DeepSeek-R1, etc. return the final answer in `content` and chain-of-thought in `reasoning_content`. The client prefers `content` and falls back to `reasoning_content` if empty. Pair two reasoning models for a fair fight. |
| `bench-llama-*` container left behind | Run `./bin/llama-swap.sh sweep` (or `down`, which also stops the proxy). Orphans mean an abnormal exit — check the proxy logs in stderr. |

## Clean-up

```sh
./bin/llama-swap.sh down              # stop proxy + sweep bench-llama-* containers
rm -rf .venv results datasets .cache  # nuke generated state incl. pinned llama-swap binary
```

## More detail

- **Design choices & invariants:** [`AGENTS.md`](AGENTS.md)
- **Code layout:** [`AGENTS.md` § Layout](AGENTS.md#layout)
- **Configuration contract:** [`config.yaml`](config.yaml) (inline comments)
