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
./run.sh --dry-run             # validate config + gen dataset; no proxy or network
./run.sh --config other.yaml   # alternate config
```

Without the wrapper:

```sh
uv venv .venv
uv pip install -r requirements.txt
uv run python -m bench.runner --config config.yaml
```

**Dry-run** validates config, generates the dataset, and exercises the `llama-swap` proxy config generator without starting a container or making any HTTP calls. On success it prints a one-line summary:

```
[dry-run] ok · validation passed · 20 tasks · 3 backends (incl. judge) · 2 prompts · 40 cells/model · judge=pairwise_all
```

Run `--dry-run` after every `config.yaml` edit and before committing benchmark configs to CI.

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

## Recommended benchmark profiles

Each profile maps a goal to the minimal `config.yaml` edits required. Judge mode cost formulas and N thresholds live in [AGENTS § Judge mode selection](AGENTS.md#judge-mode-selection); the one-model-in-VRAM constraint is documented in [AGENTS § Design invariants](AGENTS.md#design-invariants-dont-break-silently).

### Quick smoke — verify setup

**When:** Just installed the harness; want end-to-end confirmation before a long run.

```yaml
dataset:
  n: 5                   # 5 tasks instead of 20

judge:
  enabled: false         # skip judge; heuristic scores only
```

Keep one model in `models:`. Run `./run.sh --dry-run` first, then `./run.sh`. Total: 5 tasks × 2 prompts = 10 model calls, no judge calls.

**Inspect:** `results/run-<ts>.md` — confirm all rows have `latency_s` values and no `null` answers.

### Two-model A/B — serious comparison

**When:** Two candidate models; want a sharp, statistically clean head-to-head.

This is the default `config.yaml` shape. Only edit the two `gguf:` paths:

```yaml
models:
  - id: model_a
    gguf: "org/repo-GGUF/model-a-Q4_K_M.gguf"
  - id: model_b
    gguf: "org/repo-GGUF/model-b-Q4_K_M.gguf"

judge:
  mode: "pairwise_all"   # default; both orders are tried per call to de-bias
```

**Cost at defaults (n=20, prompts=2):** 80 model calls + 40 judge calls = 120 total.

**Inspect:** W/L/T table in `results/run-<ts>.md`; win-rate chart in the HTML dashboard.

### Three-to-four-model pairwise tournament

**When:** Ranking 3–4 models with the sharpest signal the harness can provide.

```yaml
models:
  - id: model_a
    gguf: "..."
  - id: model_b
    gguf: "..."
  - id: model_c          # add up to 4 total
    gguf: "..."

judge:
  mode: "pairwise_all"
```

**Cost at n=20, prompts=2:** 3 models → 120 judge calls; 4 models → 240 judge calls. Judge call count grows as C(N,2) — budget several extra minutes of wall-clock time per added model.

**Inspect:** N×N win-rate matrix in the HTML dashboard; look for a clear Condorcet winner.

### Five-or-more-model scored run

**When:** ≥ 5 models and `pairwise_all` judge call count would be impractical.

```yaml
models:
  - id: model_a
    gguf: "..."
  # ... 4+ more entries

judge:
  mode: "scored"         # absolute 1-5 rubric; cost is linear in N
```

**Cost at n=20, prompts=2:** N × 40 judge calls — 200 at N=5 versus 400 for `pairwise_all` at the same N. Trade-off: scored judges compress differences between strong models. Use the `mean ± sd` column to spot statistically significant gaps.

**Inspect:** `mean ± sd` column in the Markdown report; bar chart in the HTML dashboard.

### Low-memory MoE run

**When:** A MoE model (e.g. Qwen3.6-35B-A3B) OOMs on boot despite unified memory.

```yaml
models:
  - id: moe_model
    gguf: "org/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf"
    ctx: 2048              # shrink KV cache to reduce activation memory
    n_cpu_moe: 999         # spill expert layers to CPU; throughput drops ~30–50%
```

Only one model loads at a time (see [AGENTS § Design invariants](AGENTS.md#design-invariants-dont-break-silently)), so `n_cpu_moe` on one entry does not affect others. Budget extra wall-clock time. `cost_usd` remains `null` on Strix Halo regardless.

**Inspect:** `latency_s` — expect 3–5× slower than a fully-offloaded model of similar size.

## Output

- **`results/run-<ts>.json`** — full record: config snapshot, tasks, per-call metrics, judgements. Each judgement is tagged `mode: "pairwise" | "scored"`; pairwise entries also carry slot `order`.
- **`results/run-<ts>.md`** — per-model rollup. Pairwise mode adds W/L/T table + N×N win-rate matrix. Scored mode adds a `mean ± sd` column.
- **`results/run-<ts>.html`** — dashboard (open in browser). Matrix + overall win-rate chart in pairwise mode, mean-score chart in scored mode.

## Publishing results

`results/` stays gitignored. Publication is an explicit operator action that
packages one run into a reviewable bundle for `Rethunk-AI/bakeoff-results`.

```sh
uv run python -m bench.publish validate results/run-<ts>.json
uv run python -m bench.publish package results/run-<ts>.json --output-dir /tmp/bakeoff-bundle
uv run python -m bench.publish validate /tmp/bakeoff-bundle
uv run python -m bench.publish submit /tmp/bakeoff-bundle --dry-run
```

Bundle layout:

- `result.json` — canonical result payload.
- `manifest.json` — schema version, hashes, run metadata, and signature metadata.
- `summary.md` — generated Markdown rollup.
- `dashboard.html` — generated single-file HTML dashboard.
- `signature.sigstore.json` — optional Sigstore bundle when packaged with `--sign`.

Use `--sign` only when `cosign` is installed and you are ready for the OIDC
identity used by Sigstore/Rekor to appear in public transparency records:

```sh
uv run python -m bench.publish package results/run-<ts>.json --sign
```

Submission copies the bundle into a branch in `Rethunk-AI/bakeoff-results` and
opens a review PR. The results repository CI owns schema/hash/signature checks
and static leaderboard generation.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `llama-swap.sh: binary not found` | Bootstrap hasn't run yet. Execute `./run.sh` once; it fetches and SHA-verifies the pinned binary into `.cache/llama-swap/`. |
| `SHA256 mismatch for ...` during bootstrap | The pinned version's checksum no longer matches the downloaded tarball. Check the release on GitHub; update both `LLAMA_SWAP_VERSION` and the matching `LLAMA_SWAP_SHA256_*` line in `run.sh`. Never bypass the check. |
| Port `server.swap_port` already in use | `llama-swap` itself, another benchmark instance, or LM Studio holds it. Stop the other process or change `server.swap_port`. |
| Dry-run: `gguf must be '<org>/<repo>/<file>.gguf' form` | Path shape typo. List real files with `fd -e gguf . ~/.lmstudio/models/` and copy a matching entry into `config.yaml`. |
| `[config] ...` errors on startup | Validation failed. Read the messages — each names the field and the problem. Fix `config.yaml` and re-run `--dry-run` to confirm. |
| `[config] model IDs must be unique` | Two models share the same `id:`. Give each a distinct routing key. |
| `[config] gguf must be '<org>/<repo>/<file>.gguf' form` (validation) | Path shape typo. See the troubleshooting row below for the dry-run variant. |
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

## HuggingFace metadata enrichment

By default no network calls are made post-benchmark. Set `run.hf_enrichment` in `config.yaml` (or pass `--hf-enrichment` to the runner) to add model card metadata to `model_metadata` in the result JSON:

```yaml
run:
  hf_enrichment: "best-effort"   # off (default) | best-effort | strict
```

| Mode | Behaviour |
|------|-----------|
| `off` | No HF calls. Safe for offline and air-gapped environments. |
| `best-effort` | Failures append to `provenance.warnings` and the run continues. |
| `strict` | Any lookup failure aborts the run with an error. |

Enriched fields added to each `model_metadata` entry: `hf_sha`, `hf_tags`, `hf_pipeline_tag`, `hf_private`.

**Credentials:** Public repos work without authentication. For gated repos (Llama 3, Gemma, etc.), set `HF_TOKEN` or run `huggingface-cli login` first — same as for `./run.sh fetch`.

**Offline:** Use `hf_enrichment: "off"` (the default). No network calls are made, and no warnings are emitted.

## Comparing two runs

`bench.compare` produces a Markdown delta report between any two result JSON files — useful after swapping a GGUF quantization, editing a prompt, or adding a model.

```sh
uv run python -m bench.compare results/run-base.json results/run-cand.json
uv run python -m bench.compare base.json cand.json --output report.md
uv run python -m bench.compare base.json cand.json --strict   # exit 1 on any warning
```

The report includes:
- **Core metrics** — latency, tokens/sec, heuristic quality deltas per model.
- **Energy and cost** — total Wh and USD deltas per model (columns are `—` when energy was not measured).
- **Judge scores / win rates** — score delta (scored mode) or win-rate delta (pairwise mode), only when both runs used the same judge mode.
- **Compatibility warnings** — emitted to stderr when seeds, task sets, prompt IDs, model IDs, or judge modes differ between runs.

### Common comparison scenarios

**Quantization change** — same model, different quant level:

```sh
# Run Q4_K_M, rename result; swap gguf to Q8_0, re-run.
mv results/run-20260101-120000.json results/run-q4.json
# edit config.yaml gguf path: ...Q4_K_M.gguf → ...Q8_0.gguf
./run.sh
uv run python -m bench.compare results/run-q4.json results/run-20260101-130000.json
```

Expect: latency ↑, tokens/sec ↓, quality_heuristic ≈ same for a well-calibrated quant.

**Prompt variant change** — same models, different system prompt:

```sh
# Change prompts[].system in config.yaml, re-run.
uv run python -m bench.compare results/run-plain.json results/run-cot.json
```

The tool warns "prompt IDs differ" if the prompt `id:` fields changed between runs. If only the `system:` text changed (same IDs), no warning is emitted — the delta speaks for itself.

**Model swap** — replace one model with another:

```sh
uv run python -m bench.compare results/run-qwen35.json results/run-qwen36.json
```

When model ID sets differ, the tool warns and still renders both model rows — using `—` for whichever run lacks the model.

## Resuming a partial run

If a benchmark fails mid-run (proxy crash, OOM, network timeout), resume it instead of re-running everything:

```sh
./run.sh --resume-from results/run-20260101-120000.json
# or without the wrapper:
uv run python -m bench.runner --resume-from results/run-20260101-120000.json
```

Resume behaviour:
- **Reused rows** — model rows that completed without error in the prior run are copied into the new result and tagged `resumed_from: <prior_run_id>`.
- **Pending cells** — rows that are missing or errored in the prior run are re-executed and tagged `source_run_id: <prior_run_id>`.
- **Models with all cells complete** — skipped entirely (no proxy swap, no warmup call).
- **Judge phase** — always re-runs fully after the model phase completes (judge resume is not yet implemented).
- **New result file** — emitted as a fresh `results/run-<ts>.json` with `resumed_from` in the payload. The prior file is untouched.

Compatibility warnings are printed to stderr when the prior and current configs differ (seed, task set, prompt IDs, model IDs). Warnings do not abort the run.

## More detail

- **Design choices & invariants:** [`AGENTS.md`](AGENTS.md)
- **Code layout:** [`AGENTS.md` § Layout](AGENTS.md#layout)
- **Configuration contract:** [`config.yaml`](config.yaml) (inline comments)
