# HUMANS.md — End-user guide

Operator runbook: prerequisites, install, run, configure, troubleshoot, clean up. Design invariants and code layout: [`AGENTS.md`](AGENTS.md).

## Prerequisites

- **A running [`engined`](https://github.com/Rethunk-Tech/engined)** on this box, reachable at `engined.url` (default `http://127.0.0.1:29200`) — it owns the llama.cpp build and container lifecycle. bakeoff never starts, stops, or builds anything on its behalf.
- **`uv`** — Python env management. See [installation](https://docs.astral.sh/uv/getting-started/installation/).
- `systemctl --user` access to reload the `engined` unit after bakeoff writes its config fragment.
- One or more GGUFs under `engined.models_dir` (default `~/.local/share/engined-models/llm-bench`). Fetch with `./run.sh fetch`.

AMD: engined's Vulkan image works on ROCm-supported GPUs/APUs without a fully functional ROCm userspace stack.

## Install & run

```sh
./run.sh                       # dataset + all phases + reports
./run.sh --dry-run             # validate config + gen dataset; no engined calls
./run.sh --config other.yaml   # alternate config
```

Without the wrapper:

```sh
uv sync
uv run python -m bench.runner --config config.yaml
```

**Dry-run** validates config, generates the dataset, and exercises the engined fragment generator without reloading engined or making HTTP calls. Run after every `config.yaml` edit.

## Downloading models

```sh
./run.sh fetch                               # every missing gguf in config.yaml
./run.sh fetch --list                        # dry-run: show plan + total size
./run.sh fetch <repo_id> <filename>          # ad-hoc, one file
./run.sh fetch --config other.yaml           # alternate config
```

Existing files are skipped. Gated repos: `hf auth login` or `HF_TOKEN=<token>`. Optional faster transfers: `export HF_HUB_ENABLE_HF_TRANSFER=1`.

## Configuration

`config.yaml` is the single source of truth. Common edits:

- **Add models** — append under `models:`. First two entries are A/B for pairwise judging. `id` is the engined route's model id (lowercase, `[a-z0-9_-]`).
- **Prompt variants** — `prompts:` list; every task runs against every prompt against every model.
- **Judge mode** — `judge.mode: pairwise_all` (default) or `scored`. Thresholds in [AGENTS § Judge mode selection](AGENTS.md#judge-mode-selection).
- **Skip judge** — `judge.enabled: false`. Tasks with `scorer: "judge"` emit `null`.
- **Per-model context** — `ctx:` overrides `server.ctx`.
- **MoE OOM** — `n_cpu_moe: 999` spills experts to CPU.
- **engined address** — `engined.url` (default `http://127.0.0.1:29200`), `engined.engine` (default `llama-bench`).
- **Hold other engines during a run** — `engined.hold: [comfy]` stops and keeps `comfy` (or any engine id) stopped for the run's duration.
- **Skip `mmproj-*`** — vision projectors, not standalone text models.

### Benchmark profiles

| Goal | Key edits |
| --- | --- |
| Smoke test | `dataset.n: 5`, `judge.enabled: false`, one model |
| Two-model A/B | Default shape; edit two `gguf:` paths; `judge.mode: pairwise_all` |
| 3–4 models | Add entries; `pairwise_all` (judge cost grows as C(N,2)) |
| ≥5 models | `judge.mode: scored` (linear judge cost) |
| MoE OOM | Lower `ctx`, set `n_cpu_moe: 999` on the entry |

One model in VRAM at a time — see [AGENTS § Design invariants](AGENTS.md#design-invariants-dont-break-silently).

## Output

- **`results/run-<ts>.json`** — full record: config snapshot, tasks, per-call metrics, judgements.
- **`results/run-<ts>.md`** — per-model rollup; W/L/T or `mean ± sd` depending on judge mode.
- **`results/run-<ts>.html`** — single-file dashboard (open in browser).

## Publishing results

`results/` stays gitignored. Package a run for `Rethunk-AI/bakeoff-results`:

```sh
uv run python -m bench.publish validate results/run-<ts>.json
uv run python -m bench.publish package results/run-<ts>.json --output-dir /tmp/bakeoff-bundle
uv run python -m bench.publish submit /tmp/bakeoff-bundle --dry-run
```

Add `--sign` when `cosign` is installed and you accept Sigstore/Rekor public records. Submission opens a review PR; results-repo CI owns schema/hash/signature checks.

## Troubleshooting

| Symptom | Cause / fix |
| --------- | ------------- |
| `engined config_error after reload` | The fragment engined loaded is broken — check `GET /engined/v1/engines` on `engined.url` for the named field. |
| `engined never listed: @/llama-bench/...` | engined never brought the route up within timeout — check `engined`'s own logs, not bakeoff's. |
| Dry-run: `gguf must be '<org>/<repo>/<file>.gguf' form` | List files with `fd -e gguf . ~/.local/share/engined-models/llm-bench/` and fix path shape. |
| `[config] ...` errors on startup | Fix named field in `config.yaml`; re-run `--dry-run`. |
| `[config] model IDs must be unique` | Give each model a distinct `id:`. |
| `cost_usd: null` everywhere | Normal on Strix Halo and non-GPU hosts. |
| Judge returns mostly TIE | Swap `judge.gguf` to a stronger model or increase `judge.ctx`. |
| Judge cost high with >4 models | Switch to `judge.mode: scored`. |
| Reasoning model answers missing | Client prefers `content`, falls back to `reasoning_content`. |
| Fragment left behind after a crash | `rm ~/.config/engined/config.d/bakeoff.toml && systemctl --user reload engined`. |

## Clean-up

```sh
rm ~/.config/engined/config.d/bakeoff.toml   # only if a crash left it behind
rm -rf .venv results datasets                # nuke bakeoff's own generated state
```

## HuggingFace metadata enrichment

Set `run.hf_enrichment` in `config.yaml` (or `--hf-enrichment` on the runner):

```yaml
run:
  hf_enrichment: "best-effort"   # off (default) | best-effort | strict
```

`off` — no HF calls. `best-effort` — failures append to `provenance.warnings`. `strict` — lookup failure aborts. Gated repos need `HF_TOKEN` or `huggingface-cli login`.

## Comparing two runs

```sh
uv run python -m bench.compare results/run-base.json results/run-cand.json
uv run python -m bench.compare base.json cand.json --output report.md --strict
```

Delta report for latency, tokens/sec, heuristic quality, energy/cost, and judge scores. Warnings to stderr when seeds, tasks, prompts, models, or judge modes differ.

## Resuming a partial run

```sh
./run.sh --resume-from results/run-20260101-120000.json
```

Completed model rows are copied; missing/errored cells re-run. Judge phase always re-runs. Emits a fresh `results/run-<ts>.json`; prior file untouched.

## Disk persistence (`BAKEOFF_DATA_DIR`)

```sh
export BAKEOFF_DATA_DIR=/data/bakeoff   # default ~/.local/share/bakeoff
```

Store/queue modules write JSON under `models/`, `run_queue/pending/`, `run_queue/completed/`. The standalone runner does **not** use this by default — opt-in for multi-runner scenarios.

## Distributed worker mode

Opt-in pull client against a `bakeoff-results` queue server. The default `./run.sh` matrix is unchanged.

```sh
# Approve this runner's public key on the queue host first (admin /runners UI).
uv run python -m bench.worker \
  --queue-url http://queue-host:8765 \
  --vram-mb 32768 \
  --quantization q4_k_m \
  --poll-seconds 30
```

`--no-execute` registers, claims, heartbeats, and submits a signed stub without starting llama.cpp (useful for wiring tests). `--once` exits after one claim attempt. `--models` on `bench.runner` is the execute path the worker uses for a single claimed model. If the runner exits non-zero, the worker reports `/fail` (queue retry/backoff) and keeps polling. A paused runner sleeps on the poll interval until an admin sets it idle again. Use `--results-dir` when `bench.runner` writes somewhere other than `./results`.
