# AGENTS.md — LLM onboarding

Local LLM N-vs-N benchmark harness. Serves LM Studio GGUFs through a `podman` + `ghcr.io/ggml-org/llama.cpp:server-vulkan` container over OpenAI-compatible `/v1/chat/completions`. Matrix is `tasks × prompt_variants × models`; runner boots one model at a time (unified-memory APU constraint), runs all calls, tears down, then runs a separate judge phase.

**Claude Code:** `CLAUDE.md` is a symlink to this file. Edit **AGENTS.md**.

**Global rules** (shell tools, git/MCP preference, commit discipline, sandbox) live in `~/.claude/CLAUDE.md`. Do not restate them here. This file is project-specific LLM onboarding only.

**Operator/developer run/use** (prereqs, install, usage, troubleshooting, cleanup): [`HUMANS.md`](HUMANS.md).

## Layout

```
config.yaml          single source of truth (server, models, prompts, dataset, judge, cost, output)
bin/serve.sh         podman launcher / teardown (`down-all` nukes every bench-llama-* container)
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

## When editing

- `config.yaml` is the contract. Add new knobs there first, then wire through `runner.py` / `serve.sh`. Don't hard-code.
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
- **Re-shuffled files** → update **README.md** layout + **AGENTS.md** layout + pointers

### Drift patterns to consolidate

- ✗ Install / prereq detail in **README** → move to **HUMANS**
- ✗ Troubleshooting table in **README** → move to **HUMANS**
- ✗ Design invariants in **HUMANS** → move to **AGENTS**
- ✗ Global shell/git rules in any project file → delete; they live in `~/.claude/CLAUDE.md`
- ✗ Duplicated command blocks in **README** and **HUMANS** → keep in **HUMANS**; README links

### Keep it lean

Single-repo project. No submodules, no nested package-manager split. Three files stay flat at repo root; no per-subdirectory `AGENTS.md`. If the project ever grows a submodule or a separable library, add a local `AGENTS.md` there following the same split.
