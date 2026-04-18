# AGENTS.md — LLM onboarding

Local LLM N-vs-N benchmark harness. Serves LM Studio GGUFs through a `llama-swap` proxy sitting in front of `podman` + `ghcr.io/ggml-org/llama.cpp:server-vulkan` containers, over OpenAI-compatible `/v1/chat/completions`. Matrix is `tasks × prompt_variants × models`; the runner iterates per-model-sequentially and relies on `llama-swap`'s singleton swap to unload the previous backend before the next boots. Judge runs as its own swap target after the A/B phases.

**Claude Code:** `CLAUDE.md` is a symlink to this file. Edit **AGENTS.md**.

**Global rules** (shell tools, git/MCP preference, commit discipline, sandbox) live in `~/.claude/CLAUDE.md`. Do not restate them here. This file is project-specific LLM onboarding only.

**Operator/developer run/use** (prereqs, install, usage, troubleshooting, cleanup): [`HUMANS.md`](HUMANS.md).

## Layout

```
config.yaml          single source of truth (server, models, prompts, dataset, judge, cost, output)
bin/llama-swap.sh    llama-swap launcher: up (sweep stragglers + exec binary) / down / sweep / wait
bench/
  clients.py         httpx OpenAI-compat client; prefers `content`, falls back to `reasoning_content`
  dataset.py         seeded synthetic tasks (qa / code / summarize / classify)
  download.py        huggingface_hub fetcher; writes `<models_dir>/<repo_id>/<filename>`
  llama_swap.py      pure config generator: bakeoff config.yaml → llama-swap proxy config
  metrics.py         heuristic scorers + judge prompts + nvidia-smi / rocm-smi power sampling
  runner.py          start proxy → warmup + matrix per model → judge → stop proxy
  report.py          JSON + Markdown + single-file HTML dashboard (Chart.js via CDN)
run.sh               uv venv + uv pip install + pinned llama-swap bootstrap + uv run;
                     `fetch` subcommand → bench.download
.cache/              vendored llama-swap binary + generated proxy config (gitignored)
datasets/ results/   generated artifacts (gitignored)
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
- Python env: `uv`. No `python -m venv`, no bare `pip`.
- Match style in touched files; no drive-by refactors.

## Documentation governance

Three-tier split prevents duplication and routes content to the right audience. Modeled after Rethunk-Tech/Bastion ([`documentation-governance.mdc`](https://github.com/Rethunk-Tech/Bastion/blob/main/.cursor/rules/documentation-governance.mdc)).

| Tier | File | Audience | Content |
|------|------|----------|---------|
| **README** | [`README.md`](README.md) | Everyone (entry point) | Hook + basic product description + layout + links. **No** full install procedures, env catalogs, troubleshooting tables, or LLM internals — those link out. |
| **HUMANS** | [`HUMANS.md`](HUMANS.md) | Operators, developers, end-users | Prerequisites, install, usage, config walkthrough, output description, troubleshooting, cleanup. Everything needed to **run and use** the harness. |
| **AGENTS** | [`AGENTS.md`](AGENTS.md) (this file) | LLMs, contributors, reviewers | Design invariants, hardware caveats, judge mode selection, editing conventions, governance. `CLAUDE.md` is a symlink here. |

### When to update which file

- **New CLI flag, env var, config key, or usage mode** → **HUMANS.md** (operator-visible)
- **New design constraint, invariant, or hardware quirk affecting code** → **AGENTS.md**
- **New troubleshooting recipe** → **HUMANS.md**
- **New scorer / judge mode / runner phase** → **AGENTS.md** (invariants section) + **HUMANS.md** (usage)
- **Typo / broken link** → fix in place
- **Re-shuffled files** → update **AGENTS.md** § Layout (canonical, per-module behavior) + confirm **README.md**'s compact tree still reflects reality. README points to AGENTS for module notes; do **not** duplicate per-file annotations back into README.

### Drift patterns to consolidate

- ✗ Install / prereq detail in **README** → move to **HUMANS**
- ✗ Troubleshooting table in **README** → move to **HUMANS**
- ✗ Design invariants in **HUMANS** → move to **AGENTS**
- ✗ Global shell/git rules in any project file → delete; they live in `~/.claude/CLAUDE.md`
- ✗ Duplicated command blocks in **README** and **HUMANS** → keep in **HUMANS**; README links
- ✗ Per-module file annotations in **README** → trim to bare tree; canonical annotated layout is **AGENTS** § Layout
- ✗ Environment / install instructions in **CONTRIBUTING** → keep in **HUMANS**; CONTRIBUTING only adds the dev-extras delta (`.[dev]`) and the PR checklist

### Keep it lean

Single-repo project. No submodules, no nested package-manager split. Three files stay flat at repo root; no per-subdirectory `AGENTS.md`. If the project ever grows a submodule or a separable library, add a local `AGENTS.md` there following the same split.

### Community-profile files (outside the three-tier split)

These are **not** product documentation and are not governed by the three-tier split above. They are standard GitHub community-profile files that live at repo root by convention and are consumed by GitHub flows (PR surfacing, security-advisory prompts, community-health badge):

- [`LICENSE`](LICENSE) — terms of use.
- [`SECURITY.md`](SECURITY.md) — private disclosure policy.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — PR routing + checklist. **Thin pointer only** — must not duplicate HUMANS (setup) or AGENTS (invariants).

Do not fold any of these into README / HUMANS / AGENTS, and do not move them into a `docs/` directory — GitHub looks for them at root.

A `docs/` directory is **not** warranted for this project. The three-tier split already covers product documentation at the right granularity; adding a folder would fragment without consolidating. Revisit only if a fourth distinct audience emerges that the existing tiers can't absorb.
