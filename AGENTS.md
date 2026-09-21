# AGENTS.md — LLM onboarding

Local LLM N-vs-N benchmark harness. Serves LM Studio GGUFs through a `llama-swap` proxy in front of `podman` + `ghcr.io/ggml-org/llama.cpp:server-vulkan` containers, over OpenAI-compatible `/v1/chat/completions`. Matrix is `tasks × prompt_variants × models`; the runner iterates per-model-sequentially and relies on `llama-swap`'s singleton swap to unload the previous backend before the next boots. Judge runs as its own swap target after the model phases.

**Claude Code:** `CLAUDE.md` is `@AGENTS.md`. Edit **AGENTS.md**.

**Operator runbook:** [`HUMANS.md`](HUMANS.md). Global harness rules live in `~/.claude/CLAUDE.md` — not restated here.

## Layout

```
config.yaml          single source of truth (server, models, prompts, dataset, judge, cost, output)
bin/llama-swap.sh    llama-swap launcher: up / down / sweep / wait
bench/
  clients.py         httpx OpenAI-compat client; prefers content, falls back to reasoning_content
  compare.py         diff two result JSON files → Markdown report
  config.py          config loading + validation gate
  dataset.py         seeded synthetic tasks (qa / code / summarize / classify)
  descriptor.py      model descriptor reader/validator/persister
  download.py        huggingface_hub fetcher
  failure.py         failure-reason taxonomy (9 codes)
  hardware.py        best-effort hardware context collector
  llama_swap.py      bakeoff config.yaml → llama-swap proxy config
  metrics.py         heuristic scorers + judge prompts + power sampling
  provenance.py      run provenance collector (git SHA, platform, optional HF enrichment)
  publish.py         validate/package/sign/submit result bundles for bakeoff-results
  queue.py           opt-in disk-backed run queue (pending/ + completed/)
  report.py          JSON + Markdown + single-file HTML dashboard
  resume.py          resume support from a prior partial result
  runner.py          start proxy → warmup + matrix per model → judge → stop proxy
  worker.py          opt-in distributed pull client (claim / heartbeat / submit / fail)
  scoring.py         completeness-weighted partial score rollup
  signing.py         Ed25519 sign/verify for result envelopes
  store.py           atomic JSON record I/O under BAKEOFF_DATA_DIR
migrate/             Go module: bakeoff migration runner (#27)
run.sh               uv sync + pinned llama-swap bootstrap + uv run; fetch → bench.download
.cache/ datasets/ results/   generated artifacts (gitignored)
```

## Design invariants (don't break silently)

- **One model in VRAM at a time.** Unified-memory APU (Strix Halo / Radeon 8060S) can't hold A + B + judge concurrently. `llama-swap`'s default behaviour — unload current before starting next — enforces this at the proxy. No groups, no exclusive profiles; the default applies. `globalTTL: 0` (and per-model `ttl: 0`) in the generated config stops an idle model from silently unloading mid-matrix and forcing a silent re-boot inside a timed call.
- **Runner iterates per-model-sequentially.** All (task × prompt) cells for model A finish before any call lands for model B. A full benchmark incurs exactly N swaps (N+1 with judge), not one per cell. A round-robin outer loop would turn every cell into a swap and invalidate the energy + latency numbers. Preserve this iteration order if you touch `run_model_phase` / `main`.
- **Warmup absorbs swap + first-batch cost.** The first call to a model id is made outside the `PowerSampler` wrapper so the swap (which can pay graph-build + weight-page-in) does not leak into any recorded row.
- **`sendLoadingState: false` in the generated proxy config.** Otherwise `llama-swap` injects a loading message into `reasoning_content` during boot; the `ChatClient` falls back to `reasoning_content` when `content` is empty, so warmup could silently capture the loading text as the answer.
- **Judge runs as its own proxy entry** (`models[<judge_id>]`), not a separate subprocess of the runner. The judge swap follows the last A/B model's teardown exactly once.
- **Pairwise order randomized per call** (seeded from `run.seed`); swapped verdicts inverted before counting. Every judgement records `order: "AB" | "BA"`. Mitigates 5-15% positional bias.
- **Cost axis is energy, not tokens.** `nvidia-smi --query-gpu=power.draw` or `rocm-smi --showpower` sampled during the call. Neither available → `energy_wh` / `cost_usd` set to `null`. Do not substitute latency.
- **`mmproj-*` files are vision projectors, not standalone models.** Never list under `models:`. The generator rejects them outright.
- **Disk-persistence layer is directory-per-table, UUID-filename JSON.** `bench/store.py` owns all atomic I/O under `BAKEOFF_DATA_DIR` (env-configurable; default `~/.local/share/bakeoff`). `schema_version` is a plain integer (currently 1); missing or unexpected values are hard errors. `run_queue/` is the only ephemeral sub-tree. The runner writes each completed run to `runs/<run_id>.json` (canonical, addressable by run ID) and maintains a `run_queue/pending` → `run_queue/completed` lifecycle record per real run. `results/run-<ts>.json` is retained for backwards compatibility. `--resume-run-id <id>` loads from the store; `--resume-from <file>` loads from the flat file.
- **No database in `bench/`.** Never add psycopg/sqlite imports to any `bench/` module. Database access lives in `migrate/` (Go), which is the sole runtime consumer of `schema/schema.sql`. The `migrate/` package is a separate Go module (`go.mod` at `migrate/`) — it does not import any Python packages and is not reachable from `bench/`.
- **Worker mode is opt-in.** `bench.worker` polls a bakeoff-results queue; it must not be wired into the default `bench.runner` matrix loop. Empty queue and paused runners sleep. Execute failures report `/fail` rather than abandoning the claim. `--models` on the runner is the execute seam so a claimed job still loads one model at a time.

## Hardware caveats

- Strix Halo `rocm-smi` typically fails on `libdrm_amdgpu.so`. `cost_usd: null` in results is expected, not a bug.
- Vulkan image works on AMD/NVIDIA/Intel without per-backend wrangling. Don't switch to a ROCm-specific image "to fix" Strix Halo — that regresses portability.
- MoE models (`Qwen3.6-35B-A3B` etc.): if boot OOMs, set `n_cpu_moe: 999` on the model entry to spill experts to CPU.

## Judge mode selection

- N ∈ {2, 3, 4} → `pairwise_all` (cost `C(N,2) × tasks × prompts`, sharp ranking).
- N ≥ 5 → `scored` (cost `N × tasks × prompts`, linear).
- `judge.enabled: false` → heuristic scores only; `scorer: "judge"` tasks emit `null`.

## When editing

- `config.yaml` is the contract. Add new knobs there first, then wire through `runner.py` and/or `llama_swap.py`. Don't hard-code.
- Backend container flags (image args, ctx, ngl, etc.) are rendered into `cmd` strings inside `bench/llama_swap.py`. Changes there are covered by `tests/test_llama_swap.py` — keep the structural assertions current.
- Bumping the pinned `llama-swap` version means updating `LLAMA_SWAP_VERSION` **and** the matching per-platform SHA256 constants in `run.sh`. A mismatch aborts the bootstrap; never silence the check.
- Every new scorer/judge mode must preserve the JSON record shape in `results/run-<ts>.json` — the HTML dashboard reads it verbatim.
- Publication is explicit: `bench.publish` packages a completed result for `Rethunk-AI/bakeoff-results`; normal benchmark runs still leave `results/` gitignored and local.
- Python env: `uv`. No `python -m venv`, no bare `pip`.
- Match style in touched files; no drive-by refactors.
