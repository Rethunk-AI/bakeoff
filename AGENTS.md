# AGENTS.md — LLM onboarding

Local LLM N-vs-N benchmark harness. Every candidate model and the judge are served by [`engined`](https://github.com/Rethunk-Tech/engined) — bakeoff declares its own engine (`llama-bench`) and one route per model in a `config.d/*.toml` fragment, over OpenAI-compatible `/v1/chat/completions`. Matrix is `tasks × prompt_variants × models`; the runner iterates per-model-sequentially and relies on the `llama-bench` engine's `models_max = 1` to unload the previous backend before the next boots. Judge runs as its own route after the model phases.

**Claude Code:** `CLAUDE.md` is a symlink to `AGENTS.md`. Edit **AGENTS.md**.

**Operator runbook:** [`HUMANS.md`](HUMANS.md). Global harness rules live in `~/.claude/CLAUDE.md` — not restated here.

## Layout

```text
config.yaml          single source of truth (engined, server, models, prompts, dataset, judge, cost, output)
bench/
  clients.py         httpx OpenAI-compat client; prefers content, falls back to reasoning_content
  compare.py         diff two result JSON files → Markdown report
  config.py          config loading + validation gate
  dataset.py         seeded synthetic tasks (qa / code / summarize / classify)
  descriptor.py      model descriptor reader/validator/persister
  download.py        huggingface_hub fetcher
  engined.py         bakeoff config.yaml → engined config.d fragment; up/down lifecycle
  failure.py         failure-reason taxonomy (9 codes)
  hardware.py        best-effort hardware context collector
  metrics.py         heuristic scorers + judge prompts + power sampling
  provenance.py      run provenance collector (git SHA, platform, optional HF enrichment)
  publish.py         validate/package/sign/submit result bundles for bakeoff-results
  queue.py           opt-in disk-backed run queue (pending/ + completed/)
  report.py          JSON + Markdown + single-file HTML dashboard
  resume.py          resume support from a prior partial result
  runner.py          declare fragment → warmup + matrix per model → judge → release fragment
  worker.py          opt-in distributed pull client (claim / heartbeat / submit / fail)
  scoring.py         completeness-weighted partial score rollup
  signing.py         Ed25519 sign/verify for result envelopes
  store.py           atomic JSON record I/O under BAKEOFF_DATA_DIR
migrate/             Go module: bakeoff migration runner
run.sh               uv sync + uv run; fetch → bench.download
datasets/ results/   generated artifacts (gitignored)
```

## Design invariants (don't break silently)

- **One model in VRAM at a time.** Unified-memory APU (Strix Halo / Radeon 8060S) can't hold A + B + judge concurrently. The `llama-bench` engine's `models_max = 1` in the generated fragment enforces this inside engined: no groups, no exclusive profiles. `parallel = 1` in every route's `[route.args]` stops a batched concurrent caller from sharing a resident model's ctx pool mid-matrix.
- **Runner iterates per-model-sequentially.** All (task × prompt) cells for model A finish before any call lands for model B. A full benchmark incurs exactly N swaps (N+1 with judge), not one per cell. A round-robin outer loop would turn every cell into a swap and invalidate the energy + latency numbers. Preserve this iteration order if you touch `run_model_phase` / `main`.
- **Warmup absorbs swap + first-batch cost.** `POST /engined/v1/start` pays the swap explicitly; the throwaway chat call that follows, made outside the `PowerSampler` wrapper, absorbs only first-batch JIT cost. Neither contaminates a recorded row.
- **The runner checks `x-engined-route` on every response.** A row whose answering address doesn't match what was asked for is a failure, not a silent misattribution — see `bench/clients.py` and `call_one`'s `expected_route`.
- **Judge runs as its own route**, not a separate subprocess of the runner. The judge swap follows the last A/B model's teardown exactly once.
- **Pairwise order randomized per call** (seeded from `run.seed`); swapped verdicts inverted before counting. Every judgement records `order: "AB" | "BA"`. Mitigates 5-15% positional bias.
- **Cost axis is energy, not tokens.** `nvidia-smi --query-gpu=power.draw` or `rocm-smi --showpower` sampled during the call. Neither available → `energy_wh` / `cost_usd` set to `null`. Do not substitute latency.
- **`mmproj-*` files are vision projectors, not standalone models.** Never list under `models:`. The generator rejects them outright.
- **Disk-persistence layer is directory-per-table, UUID-filename JSON.** `bench/store.py` owns all atomic I/O under `BAKEOFF_DATA_DIR` (env-configurable; default `~/.local/share/bakeoff`). `schema_version` is a plain integer (currently 1); missing or unexpected values are hard errors. `run_queue/` is the only ephemeral sub-tree. The runner writes each completed run to `runs/<run_id>.json` (canonical, addressable by run ID) and maintains a `run_queue/pending` → `run_queue/completed` lifecycle record per real run. `results/run-<ts>.json` is the flat, self-contained per-run file: it is the portable output, easy to copy, diff and hand to `--resume-from <file>`, which resumes a run from that file alone without the store. `--resume-run-id <id>` resumes from the store instead.
- **No database in `bench/`.** Never add psycopg/sqlite imports to any `bench/` module. Database access lives in `migrate/` (Go), which is the sole runtime consumer of `schema/schema.sql`. The `migrate/` package is a separate Go module (`go.mod` at `migrate/`) — it does not import any Python packages and is not reachable from `bench/`.
- **Worker mode is opt-in.** `bench.worker` polls a bakeoff-results queue; it must not be wired into the default `bench.runner` matrix loop. Empty queue and paused runners sleep. Execute failures report `/fail` rather than abandoning the claim. `--models` on the runner is the execute seam so a claimed job still loads one model at a time.

## Hardware caveats

- Strix Halo `rocm-smi` typically fails on `libdrm_amdgpu.so`. `cost_usd: null` in results is expected, not a bug.
- MoE models (`Qwen3.6-35B-A3B` etc.): if boot OOMs, set `n_cpu_moe: 999` on the model entry to spill experts to CPU.

## Judge mode selection

- N ∈ {2, 3, 4} → `pairwise_all` (cost `C(N,2) × tasks × prompts`, sharp ranking).
- N ≥ 5 → `scored` (cost `N × tasks × prompts`, linear).
- `judge.enabled: false` → heuristic scores only; `scorer: "judge"` tasks emit `null`.

## When editing

- `config.yaml` is the contract. Add new knobs there first, then wire through `runner.py` and/or `bench/engined.py`. Don't hard-code.
- Backend flags (ctx, ngl, ubatch, etc.) are rendered into each route's `[route.args]` inside `bench/engined.py`'s `render_fragment`. Changes there are covered by `tests/test_engined.py` — keep the `tomllib` structural assertions current.
- `bench/engined.up`/`down` write and delete `~/.config/engined/config.d/bakeoff.toml` and reload the running `engined` — never start, stop, or build it. `down` must stay idempotent and exception-safe: it runs from both `finally` and `atexit`, and a fragment left behind whose GGUF later vanished makes engined refuse to boot.
- Exclusivity (holding other engines like `comfy` off the box) is `engined.hold` in config.yaml, renewed by a background thread in `bench/engined.py` — not anything engined itself decides.
- Every new scorer/judge mode must preserve the JSON record shape in `results/run-<ts>.json` — the HTML dashboard reads it verbatim.
- Publication is explicit: `bench.publish` packages a completed result for `Rethunk-AI/bakeoff-results`; normal benchmark runs still leave `results/` gitignored and local.
- Python env: `uv`. No `python -m venv`, no bare `pip`.
- Match style in touched files; no drive-by refactors.
