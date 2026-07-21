# Changelog

All notable changes to this project are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- **Disk-persistence layer** — `bench/store.py` (atomic JSON record I/O under `BAKEOFF_DATA_DIR`; directory-per-table / UUID-filename layout; `schema_version` audit stamping; deterministic UUID5 helpers), `bench/descriptor.py` (model descriptor reader/validator with a hard `schema_version` gate), and `bench/queue.py` (opt-in disk-backed run queue: `pending/` + `completed/` DR layout, race-safe rename-as-mutex `claim()`, retry backoff + terminal failure, stale-claim reaping). The standalone runner default is unchanged; the queue is strictly opt-in. (#13, #15)
- **`interface_types` seed** — `schema/seeds/interface_types.json`, completing the seed-file pattern for the last lookup table that was only seeded inline in `schema.sql`. (#22)
- **Failure-reason taxonomy** — `bench/failure.py` introduces a `failure_code` enum (`timeout`, `refusal`, `malformed_output`, `oom`, `load_failure`, `capability_gap`, `infra_error`, `cancelled`, `unknown`) and a classifier; replaces the previous free-text `error` field. The runner now emits a structured `failure_code` + `failure_detail` per cell. (#23)
- **Completeness-weighted scoring** — `bench/scoring.py` adds a `partial_score` formula (`S(m)/C` treating unattempted cells as 0), a `floor_score` rollup for the `dumb_model` tier, per-model `model_rollup()`, and `run_status_from_scores()`. (#23)
- **Floor-tier dataset** — `datasets/dumb_model_tasks.jsonl`: 12 fixed tasks spanning arithmetic, instruction-following, QA, and summarisation with deterministic exact/contains scorers; the runner executes the `dumb_model` tier before the main matrix when configured. Schema additions: `failure_detail TEXT` on `run_model_metrics`, `run_status TEXT` on `runs`, and a `dumb_model` row in `task_categories`. (#23)
- **`manifest.json` score projection** — `bench/publish.py` now writes `run_status` and `model_scores_summary` into the published manifest. A `--strict` flag warns (without erroring) when new optional fields are absent in older bundles. (#23)
- **Ed25519 result signing** — `bench/signing.py` provides sign/verify for result envelopes using Ed25519; the `runners` schema table gains a `public_key TEXT` (base64) column; signing is gated on `config.signing.enabled`; `bench/publish.py` enforces mandatory Ed25519 verification on ingest. Unit tests cover sign/verify round-trips and failure modes. (#16)
- **Hardware context collector** — `bench/hardware.py` captures GPU, CPU, RAM, and OS details and wires them into the runner to populate `run_hardware_metrics`. (closes #8 partial)
- **Hardware schema tables** — `interface_type`, `gpu_hardware`, `system_hardware`, `system_software`, `system_gpu_link`, and `run_hardware_metrics` tables. (#17–#21)
- **Schema-version tracking** — `schema_versions`, `schema_tables`, and `schema_tables_join` tables for schema-version tracking and migration-graph bookkeeping. (#25)

### Changed

- Refactored runner to route all inference through `llama-swap` proxy; retired `bin/serve.sh` in favour of `bin/llama-swap.sh`.
- Extracted `bench.config` module; validation and `judge_id` helper are now shared across runner, report, and llama-swap generator.
- Consolidated `_fmt`, `resolve_models_dir`, `DEFAULT_CONFIG`, and several small helpers that had drifted to multiple sites.
- Updated repo-ops dependency and GitHub Actions versions.
- **Dev tooling:** replaced `mypy` with `pyrefly` for type checking (CI, pre-commit docs, `.gitignore`); repository-only change, no user-facing effect.

### Fixed

- Removed duplicate `_fmt` definition in `bench/report.py`.
- Applied current ruff formatting fleet-wide.

---

## [0.1.0] — 2026-04-18

First complete benchmark harness.

### Added

- **Core harness** — `bench/runner.py`: sequential per-model matrix (`tasks × prompt_variants × models`); `llama-swap` proxy lifecycle (up / warmup / matrix / judge / down); warmup call excluded from `PowerSampler` to avoid absorbing swap cost.
- **llama-swap integration** — `bin/llama-swap.sh` launcher (up / down / sweep / wait) and `bench/llama_swap.py` config generator; produces a validated proxy config from `config.yaml` with `globalTTL: 0` and `sendLoadingState: false` enforced.
- **Judge modes** — `pairwise_all` (round-robin tournament, order-randomized per call to mitigate positional bias) and `scored` (absolute 1–5 rubric); threshold guidance: N ∈ {2–4} → pairwise, N ≥ 5 → scored.
- **Heuristic scorers** — `exact`, `contains`, `regex`; tasks with `scorer: "judge"` skip the judge when `judge.enabled: false`.
- **Energy-based cost** — `bench/metrics.py`: background `PowerSampler` using `nvidia-smi --query-gpu=power.draw` or `rocm-smi --showpower`; `energy_wh` / `cost_usd` set to `null` when neither tool is available.
- **GGUF download** — `bench/download.py` and `./run.sh fetch`; pulls from Hugging Face into `<models_dir>/<repo_id>/<filename>`; idempotent; supports `HF_TOKEN` for gated repos.
- **SSE streaming client** — `bench/clients.py`: records time-to-first-token; prefers `content`, falls back to `reasoning_content` (Qwen3, DeepSeek-R1 support).
- **Reports** — `bench/report.py`: JSON, Markdown (W/L/T table + N×N win-rate matrix for pairwise; `mean ± sd` for scored), and single-file HTML dashboard (Chart.js via CDN; opens from disk with no build step).
- **Latency percentiles** — p50 / p95 / p99 latency and TTFT recorded per model.
- **Quality-vs-energy Pareto view** — scatter chart in the HTML dashboard.
- **Run comparison** — `bench/compare.py` and `uv run python -m bench.compare`; Markdown delta report across any two result JSON files; warns on seed / task-set / prompt / model / judge-mode mismatches.
- **Provenance** — `bench/provenance.py`: `collect()` captures git SHA, Python version, OS, and dependency pins; `build_model_metadata()` adds per-model GGUF size and mtime; result JSON carries a `provenance` block.
- **HuggingFace enrichment** — optional `run.hf_enrichment: off | best-effort | strict`; adds `hf_sha`, `hf_tags`, `hf_pipeline_tag`, `hf_private` to `model_metadata`.
- **Config validation** — `bench/config.py`: `load()` + `validate_config()` checked on every entry point including `--dry-run`; rejects malformed GGUF paths and duplicate model IDs.
- **Dry-run mode** — `--dry-run`: validates config, generates dataset, exercises the proxy config generator; no containers or HTTP calls; prints a one-line summary on success.
- **Result publication** — `bench/publish.py`: validate / package / sign (Sigstore / cosign) / submit workflow targeting `Rethunk-AI/bakeoff-results`.
- **Resume** — `bench/resume.py` and `--resume-from`; stable row keys; completed model rows are copied with `resumed_from` tag; errored / missing rows are re-executed; judge phase always re-runs.
- **CI** — GitHub Actions: ruff lint + format, mypy, shellcheck, actionlint, pytest, dry-run; uses `uv` throughout.
- **Three-tier governance** — `README.md` (entry point), `HUMANS.md` (operators), `AGENTS.md` / `CLAUDE.md` (LLMs + contributors).

### Technical constraints

- One model in VRAM at a time (unified-memory APU support; Strix Halo / Radeon 8060S tested).
- `pairwise_all` judge order randomized per call (seeded from `run.seed`); swapped verdicts inverted before counting.
- Cost axis is energy (Wh / USD), not token count; `null` on Strix Halo due to `libdrm_amdgpu.so` failures in `rocm-smi`.
- Vulkan container image (`ghcr.io/ggml-org/llama.cpp:server-vulkan`) works on AMD / NVIDIA / Intel without per-backend wrangling.

[Unreleased]: https://github.com/Rethunk-AI/bakeoff/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Rethunk-AI/bakeoff/releases/tag/v0.1.0
