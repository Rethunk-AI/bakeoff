"""Benchmark runner: sequential model phases over a llama-swap proxy.

Phases:
  1. Generate dataset.
  2. Start llama-swap proxy in front of podman + llama.cpp backends.
  3. For each model: warmup (triggers the swap) -> run all
     (task x prompt) calls -> next model.
  4. Judge phase (optional): warmup the judge model -> run judgements
     over stored A/B responses.
  5. Stop the proxy (SIGTERM + sweep any bench-llama-* containers).
  6. Emit reports (JSON + Markdown + static HTML dashboard).

Invariants (AGENTS.md):

  - **One model in VRAM at a time.** llama-swap's default behaviour
    unloads the current backend before starting the next; we never
    configure groups or exclusive profiles, so the default applies.
    The runner additionally iterates per-model-sequentially — all
    calls for model A complete before any call for model B — so a
    full benchmark incurs exactly N swaps (N+1 with judge), not one
    per matrix cell. Changing to a round-robin outer loop would
    silently trigger a swap per call and invalidate the energy and
    latency numbers.
  - **Warmup absorbs swap + first-batch cost.** The first call to a
    previously-unseen model name triggers llama-swap's boot. We make
    that call outside the PowerSampler so the swap energy does not
    leak into any row.
  - **Judge is its own model entry**, not a live subprocess of the
    runner. Running it through llama-swap keeps the timing model
    identical to the A/B phase.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from bench import llama_swap
from bench.clients import ChatClient, ChatResult
from bench.config import (
    DEFAULT_CONFIG,
    ConfigError,
    judge_id,
    load_config,
    resolve_models_dir,
    validate_config,
)
from bench.dataset import Task, generate, load_floor_tasks, write_jsonl
from bench.failure import classify as classify_failure
from bench.hardware import collect_hardware_context
from bench.metrics import (
    PowerSampler,
    detect_hardware_id,
    flops_per_token,
    gpu_weighted_seconds,
    invert_winner,
    judge_pair_randomized,
    judge_score_prompt,
    lookup_peak_tflops,
    parse_judge,
    parse_score,
    score_heuristic,
    tflops_utilization_pct,
)
from bench.provenance import build_model_metadata, enrich_model_metadata
from bench.provenance import collect as collect_provenance
from bench.resume import (
    ResumeError,
    build_pending,
    build_pending_judge_pairs,
    build_pending_judge_scores,
    check_compat,
    load_prior,
    tag_fresh,
    tag_reused,
)
from bench.scoring import model_rollup, run_status_from_scores

HERE = Path(__file__).resolve().parent.parent
LAUNCHER = HERE / "bin" / "llama-swap.sh"
LLAMA_SWAP_CONFIG = HERE / ".cache" / "llama-swap" / "config.yaml"


# --- llama-swap lifecycle ---------------------------------------------------


def _write_proxy_config(
    bakeoff_cfg: dict[str, Any],
    models_dir: Path,
    target: Path,
) -> None:
    ls_cfg = llama_swap.build(bakeoff_cfg, str(models_dir))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(ls_cfg, sort_keys=False))


def _launcher_args(*extra: str) -> list[str]:
    return [str(LAUNCHER), *extra]


def _proxy_start(listen: str, config_path: Path, boot_timeout_s: int) -> subprocess.Popen[bytes]:
    """Launch llama-swap as a child process; block until it is accepting requests.

    The launcher script handles the pre-start sweep of stale
    `bench-llama-*` containers and then execs the binary, so the
    Popen pid tracks the binary once exec completes.
    """
    print(f"[proxy] starting llama-swap on {listen}", file=sys.stderr)
    proc = subprocess.Popen(
        _launcher_args("up", str(config_path), listen),
        stdout=sys.stderr.fileno(),
        stderr=subprocess.STDOUT,
    )
    try:
        subprocess.run(
            _launcher_args("wait", listen, str(boot_timeout_s)),
            check=True,
        )
    except Exception:
        _proxy_stop(proc)
        raise
    print("[proxy] ready", file=sys.stderr)
    return proc


def _proxy_stop(proc: subprocess.Popen[bytes]) -> None:
    print("[proxy] stopping llama-swap", file=sys.stderr)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    # Defensive: even if the proxy exited cleanly, sweep orphans that a
    # crashed `cmd` might have left behind.
    subprocess.run(_launcher_args("sweep"), check=False)


# --- Call wrapper -----------------------------------------------------------


def call_one(
    client: ChatClient,
    system: str,
    user: str,
    gpu_index: int,
    cost_enabled: bool,
    kwh_rate: float,
    sample_hz: float = 10.0,
) -> tuple[ChatResult, float | None, float | None, float | None, float | None, float | None]:
    """Run one inference call.

    Returns (result, energy_wh, peak_vram_mb, mean_sm_pct,
             cpu_seconds_user, cpu_seconds_sys).

    cost_usd is not returned — it is a derived value computed at display time
    from energy_wh × kwh_rate. The PowerSampler always runs so VRAM and SM
    utilization are captured regardless of cost_enabled. CPU timing via
    getrusage brackets the call.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    ru_before = resource.getrusage(resource.RUSAGE_SELF)
    with PowerSampler(hz=sample_hz, gpu_index=gpu_index) as sampler:
        res = client.chat(messages)
    ru_after = resource.getrusage(resource.RUSAGE_SELF)

    peak_vram = sampler.peak_vram_mb
    mean_sm = sampler.mean_sm_pct
    cpu_user_ms = (ru_after.ru_utime - ru_before.ru_utime) * 1000.0
    cpu_sys_ms = (ru_after.ru_stime - ru_before.ru_stime) * 1000.0

    wh = sampler.energy_wh if cost_enabled else None
    return res, wh, peak_vram, mean_sm, cpu_user_ms, cpu_sys_ms


WARMUP_SYSTEM = "You are a helpful assistant. Answer concisely."
WARMUP_USER = "Say 'ready' and nothing else."


def _warmup(client: ChatClient) -> None:
    """Fire one throwaway call so the timed matrix starts hot.

    With llama-swap in front, the first call to a given model name
    triggers an unload-of-previous + boot-of-requested backend. On
    unified-memory APUs that can add seconds of graph-build and
    weight-page-in cost. We absorb that here, outside the PowerSampler
    wrapper, so it never contaminates a recorded row.
    """
    try:
        client.chat(
            [
                {"role": "system", "content": WARMUP_SYSTEM},
                {"role": "user", "content": WARMUP_USER},
            ]
        )
    except Exception as e:
        print(f"[warmup-err] {e}", file=sys.stderr)


# --- Phases -----------------------------------------------------------------


def run_model_phase(
    model_cfg: dict[str, Any],
    tasks: list[Task],
    prompts: list[dict[str, Any]],
    base_url: str,
    cost_cfg: dict[str, Any],
    timeout_s: float,
    warmup: bool = True,
    pending: set[tuple[str, str]] | None = None,
    hardware_id: str | None = None,
    peak_tflops: float | None = None,
    floor_tasks: list[Task] | None = None,
) -> list[dict[str, Any]]:
    """Run every (task x prompt) cell for one model against the proxy.

    `model_cfg["id"]` is the llama-swap routing key (not `alias` — the
    alias stays inside the generated `cmd` as llama.cpp's `-a` flag).

    When `pending` is provided, only (task_id, prompt_id) pairs in that
    set are executed; all others are silently skipped.

    When `floor_tasks` is provided, the minimal-capability ("dumb_model")
    floor suite runs FIRST, while this model is already loaded — so it adds
    no extra swap (invariant: exactly N swaps per run) and a model that
    later crashes the main matrix still has a floor score if it booted at
    all (Rethunk-AI/bakeoff#23). Floor cells use a single fixed prompt
    (prompt_id "floor", empty system) and deterministic scorers only.
    """
    mid = model_cfg["id"]
    print(f"[phase] {mid}", file=sys.stderr)

    client = ChatClient(
        base_url=base_url,
        model=mid,
        api_key="none",
        timeout_s=timeout_s,
    )
    if warmup:
        print(f"[warmup] {mid}", file=sys.stderr)
        _warmup(client)

    cost_enabled = bool(cost_cfg.get("enabled"))
    kwh = float(cost_cfg.get("kwh_rate_usd", 0.0))
    gpu_i = int(cost_cfg.get("gpu_index", 0))
    sample_hz = float(cost_cfg.get("sample_hz", 10.0))

    # TFLOPS utilization: needs per-model param counts + hardware peak.
    num_params: int | None = model_cfg.get("num_params")
    num_active: int | None = model_cfg.get("num_active_params")
    fpt: int | None = flops_per_token(num_params, num_active) if num_params is not None else None
    _peak_tflops = peak_tflops

    records: list[dict[str, Any]] = []

    # Minimal-capability floor suite — runs first, within this single model
    # load (no extra swap). Single fixed prompt, deterministic scorers, binary
    # per-cell score. See run-level docstring + Rethunk-AI/bakeoff#23.
    for ft in floor_tasks or []:
        try:
            res, _wh, _vram, _sm, _cu, _cs = call_one(
                client, "", ft.user_prompt, gpu_i, False, kwh, sample_hz
            )
            records.append(
                {
                    "task_id": ft.id,
                    "domain": ft.domain,
                    "prompt_id": "floor",
                    "model_id": mid,
                    "hardware_id": hardware_id,
                    "tier": "dumb_model",
                    "text": res.text,
                    "wall_clock_seconds": res.latency_s,
                    "quality_heuristic": score_heuristic(ft.scorer, ft.expected, res.text),
                    "failure_code": None,
                    "failure_detail": None,
                    "error": None,
                }
            )
            print(f"[floor] {mid} {ft.id}", file=sys.stderr)
        except Exception as e:
            records.append(
                {
                    "task_id": ft.id,
                    "domain": ft.domain,
                    "prompt_id": "floor",
                    "model_id": mid,
                    "tier": "dumb_model",
                    "failure_code": classify_failure(e),
                    "failure_detail": str(e),
                    "error": str(e),
                }
            )
            print(f"[floor-err] {mid} {ft.id}: {e}", file=sys.stderr)

    for task, prm in itertools.product(tasks, prompts):
        if pending is not None and (str(task.id), str(prm["id"])) not in pending:
            continue
        try:
            res, wh, peak_vram, mean_sm, cpu_user_ms, cpu_sys_ms = call_one(
                client, prm["system"], task.user_prompt, gpu_i, cost_enabled, kwh, sample_hz
            )
            records.append(
                {
                    "task_id": task.id,
                    "domain": task.domain,
                    "prompt_id": prm["id"],
                    "model_id": mid,
                    "hardware_id": hardware_id,
                    "text": res.text,
                    "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                    "wall_clock_seconds": res.latency_s,
                    "seconds_to_first_token": res.ttft_s,
                    "tokens_per_second": res.tokens_per_sec,
                    "energy_wh": wh,
                    "peak_vram_mb": peak_vram,
                    "gpu_sm_utilization_pct": mean_sm,
                    # Path 1: kernel wall time from cudaEventElapsedTime() /
                    # hipEventElapsedTime(). None until the CUDA/ROCm event API
                    # path is wired; the field exists now so the schema is stable.
                    "gpu_event_seconds": None,
                    # Path 2: utilization-weighted GPU time.
                    # wall_clock_seconds x mean(gpu_sm_utilization_pct / 100).
                    # None when SM utilization is unavailable (non-NVML hosts).
                    "gpu_weighted_seconds": gpu_weighted_seconds(res.latency_s, mean_sm),
                    "cpu_seconds_user": (cpu_user_ms / 1000.0) if cpu_user_ms is not None else None,
                    "cpu_seconds_sys": (cpu_sys_ms / 1000.0) if cpu_sys_ms is not None else None,
                    "flops_per_token_theoretical": fpt,
                    "tflops_utilization_pct": None,  # filled below when possible
                    "quality_heuristic": score_heuristic(task.scorer, task.expected, res.text),
                    "tier": "main",
                    "failure_code": None,
                    "failure_detail": None,
                    "error": None,
                }
            )
            # Compute TFLOPS utilization when we have all inputs.
            rec = records[-1]
            if fpt is not None and res.tokens_per_sec is not None and _peak_tflops is not None:
                rec["tflops_utilization_pct"] = tflops_utilization_pct(
                    res.tokens_per_sec, fpt, _peak_tflops
                )
            print(f"[run] {mid} {prm['id']} {task.id} {res.latency_s:.2f}s", file=sys.stderr)
        except Exception as e:
            records.append(
                {
                    "task_id": task.id,
                    "domain": task.domain,
                    "prompt_id": prm["id"],
                    "model_id": mid,
                    "tier": "main",
                    "failure_code": classify_failure(e),
                    "failure_detail": str(e),
                    "error": str(e),
                }
            )
            print(f"[err] {mid} {prm['id']} {task.id}: {e}", file=sys.stderr)
    return records


def _run_model_phases(
    models: list[dict[str, Any]],
    tasks: list[Task],
    prompts: list[dict[str, Any]],
    base_url: str,
    cost_cfg: dict[str, Any],
    timeout_s: float,
    warmup: bool,
    pending_by_model: dict[str, set[tuple[str, str]]] | None,
    prior_run_id: str | None,
    hardware_id: str | None = None,
    peak_tflops: float | None = None,
    floor_tasks: list[Task] | None = None,
) -> list[dict[str, Any]]:
    """Iterate per-model-sequentially, honouring resume pending sets.

    Returns only the fresh records; caller merges with reused records.
    """
    fresh: list[dict[str, Any]] = []
    for m in models:
        mid = str(m["id"])
        pending = pending_by_model[mid] if pending_by_model is not None else None
        if pending is not None and not pending:
            print(f"[resume] {mid}: all cells complete, skipping", file=sys.stderr)
            continue
        recs = run_model_phase(
            m,
            tasks,
            prompts,
            base_url,
            cost_cfg,
            timeout_s,
            warmup=warmup,
            pending=pending,
            hardware_id=hardware_id,
            peak_tflops=peak_tflops,
            floor_tasks=floor_tasks,
        )
        if prior_run_id is not None:
            recs = tag_fresh(recs, prior_run_id)
        fresh.extend(recs)
    return fresh


def assemble_model_scores(
    models: list[dict[str, Any]],
    records: list[dict[str, Any]],
    cells_total: int,
) -> tuple[list[dict[str, Any]], str]:
    """Post-hoc per-model rollup + run-level status (Rethunk-AI/bakeoff#23).

    Pure: no network, no proxy. `cells_total` is the main-suite cell count C
    per model (len(tasks) * len(prompts)). Floor records (tier == "dumb_model")
    are separated from main records before rollup so partial_score reflects the
    main suite and floor_score the floor suite. Returns (model_scores list,
    run_status word)."""
    model_scores: list[dict[str, Any]] = []
    for m in models:
        mid = str(m["id"])
        mine = [r for r in records if r.get("model_id") == mid]
        main_recs = [r for r in mine if r.get("tier", "main") != "dumb_model"]
        floor_recs = [r for r in mine if r.get("tier") == "dumb_model"]
        model_scores.append(model_rollup(mid, main_recs, floor_recs, cells_total))
    return model_scores, run_status_from_scores(model_scores)


def _pairwise_all_phase(
    judge: ChatClient,
    models: list[dict[str, Any]],
    tasks: list[Task],
    prompts: list[dict[str, Any]],
    records: list[dict[str, Any]],
    rng: random.Random,
    pending_pairs: set | None = None,
) -> list[dict[str, Any]]:
    """Round-robin all unordered pairs (N choose 2). Randomize slot order
    per call to de-bias positional preference; invert winner on swap.

    When pending_pairs is provided, only those (task_id, prompt_id,
    frozenset({a,b})) keys are run; others are skipped.
    """
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {
        (r["task_id"], r["prompt_id"], r["model_id"]): r for r in records if not r.get("error")
    }
    judgements: list[dict[str, Any]] = []
    for a_m, b_m in itertools.combinations(models, 2):
        a_id, b_id = a_m["id"], b_m["id"]
        for task, prm in itertools.product(tasks, prompts):
            if pending_pairs is not None:
                from bench.resume import pairwise_key as _pk

                if (
                    _pk(
                        {
                            "task_id": task.id,
                            "prompt_id": prm["id"],
                            "a_model": a_id,
                            "b_model": b_id,
                        }
                    )
                    not in pending_pairs
                ):
                    continue
            a = by_key.get((task.id, prm["id"], a_id))
            b = by_key.get((task.id, prm["id"], b_id))
            if not a or not b:
                continue
            msgs, order = judge_pair_randomized(task.user_prompt, a["text"], b["text"], rng)
            try:
                jr = judge.chat(msgs)
                raw_winner = parse_judge(jr.text)
                winner = invert_winner(raw_winner) if order == "BA" else raw_winner
                judgements.append(
                    {
                        "mode": "pairwise",
                        "task_id": task.id,
                        "prompt_id": prm["id"],
                        "a_model": a_id,
                        "b_model": b_id,
                        "order": order,
                        "winner": winner,
                        "judge_raw": jr.text,
                    }
                )
                print(
                    f"[judge-pair] {a_id} vs {b_id} {task.id} {prm['id']} ({order}) -> {winner}",
                    file=sys.stderr,
                )
            except Exception as e:
                judgements.append(
                    {
                        "mode": "pairwise",
                        "task_id": task.id,
                        "prompt_id": prm["id"],
                        "a_model": a_id,
                        "b_model": b_id,
                        "order": order,
                        "winner": None,
                        "error": str(e),
                    }
                )
                print(f"[judge-err] {a_id} vs {b_id} {task.id} {prm['id']}: {e}", file=sys.stderr)
    return judgements


def _scored_phase(
    judge: ChatClient,
    models: list[dict[str, Any]],
    tasks: list[Task],
    prompts: list[dict[str, Any]],
    records: list[dict[str, Any]],
    pending_scores: set | None = None,
) -> list[dict[str, Any]]:
    """Absolute 1-5 rubric score per (model, task, prompt). Linear in N models.

    When pending_scores is provided, only (task_id, prompt_id, model_id)
    keys in that set are run; others are skipped.
    """
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {
        (r["task_id"], r["prompt_id"], r["model_id"]): r for r in records if not r.get("error")
    }
    judgements: list[dict[str, Any]] = []
    for m in models:
        mid = m["id"]
        for task, prm in itertools.product(tasks, prompts):
            if pending_scores is not None and (task.id, prm["id"], mid) not in pending_scores:
                continue
            rec = by_key.get((task.id, prm["id"], mid))
            if not rec:
                continue
            try:
                jr = judge.chat(judge_score_prompt(task.user_prompt, rec["text"]))
                score = parse_score(jr.text)
                judgements.append(
                    {
                        "mode": "scored",
                        "task_id": task.id,
                        "prompt_id": prm["id"],
                        "model_id": mid,
                        "score": score,
                        "judge_raw": jr.text,
                    }
                )
                print(f"[judge-score] {mid} {task.id} {prm['id']} -> {score}", file=sys.stderr)
            except Exception as e:
                judgements.append(
                    {
                        "mode": "scored",
                        "task_id": task.id,
                        "prompt_id": prm["id"],
                        "model_id": mid,
                        "score": None,
                        "error": str(e),
                    }
                )
                print(f"[judge-err] {mid} {task.id} {prm['id']}: {e}", file=sys.stderr)
    return judgements


def run_judge_phase(
    judge_cfg: dict[str, Any],
    models: list[dict[str, Any]],
    tasks: list[Task],
    prompts: list[dict[str, Any]],
    records: list[dict[str, Any]],
    base_url: str,
    timeout_s: float,
    seed: int,
    warmup: bool = True,
    pending_pairs: set | None = None,
    pending_scores: set | None = None,
) -> list[dict[str, Any]]:
    """Dispatch judge phase based on judge.mode.

    Modes:
      pairwise_all (default) — every unordered pair, order-randomized.
        Cost: C(N,2) * tasks * prompts judge calls.
      scored — 1-5 absolute rubric per (model, task, prompt).
        Cost: N * tasks * prompts judge calls. Scales linearly in N.
    """
    if not judge_cfg.get("enabled"):
        return []
    mode = judge_cfg.get("mode", "pairwise_all")
    if mode not in {"pairwise_all", "scored"}:
        print(f"[warn] unknown judge.mode {mode!r}; skipping judge phase", file=sys.stderr)
        return []
    if mode == "pairwise_all" and len(models) < 2:
        print("[warn] pairwise_all needs >= 2 models; skipping judge phase", file=sys.stderr)
        return []
    if mode == "scored" and len(models) < 1:
        return []

    jid = judge_id(judge_cfg)
    print(f"[phase] judge ({jid}), mode={mode}", file=sys.stderr)

    judge = ChatClient(
        base_url=base_url,
        model=jid,
        api_key="none",
        timeout_s=timeout_s,
    )
    if warmup:
        print(f"[warmup] {judge_id}", file=sys.stderr)
        _warmup(judge)

    if mode == "pairwise_all":
        rng = random.Random(seed)
        return _pairwise_all_phase(
            judge, models, tasks, prompts, records, rng, pending_pairs=pending_pairs
        )
    return _scored_phase(judge, models, tasks, prompts, records, pending_scores=pending_scores)


# --- Entry point ------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse + gen dataset, no proxy startup or network calls.",
    )
    _resume_group = ap.add_mutually_exclusive_group()
    _resume_group.add_argument(
        "--resume-from",
        metavar="RESULT_JSON",
        help="Prior result JSON. Reuses complete rows, reruns errored/missing.",
    )
    _resume_group.add_argument(
        "--resume-run-id",
        metavar="RUN_ID",
        help="Resume from a run stored in BAKEOFF_DATA_DIR (alternative to --resume-from).",
    )
    ap.add_argument(
        "--rerun-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rerun errored rows on resume (default: true).",
    )
    ap.add_argument(
        "--rerun-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rerun missing rows on resume (default: true).",
    )
    ap.add_argument(
        "--resume-models", nargs="+", metavar="MODEL_ID", help="Limit resume to these model IDs."
    )
    ap.add_argument(
        "--resume-tasks", nargs="+", metavar="TASK_ID", help="Limit resume to these task IDs."
    )
    ap.add_argument(
        "--resume-prompts", nargs="+", metavar="PROMPT_ID", help="Limit resume to these prompt IDs."
    )
    ap.add_argument(
        "--hf-enrichment",
        choices=["off", "best-effort", "strict"],
        default=None,
        help="HuggingFace metadata enrichment (overrides run.hf_enrichment in config).",
    )
    args = ap.parse_args()

    try:
        cfg = load_config(Path(args.config))
    except ConfigError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    issues = validate_config(cfg)
    if issues:
        for issue in issues:
            print(f"[config] {issue}", file=sys.stderr)
        return 1

    run_cfg = cfg.get("run", {})
    ds_cfg = cfg["dataset"]
    server_cfg = cfg["server"]
    cost_cfg = cfg.get("cost", {})
    judge_cfg = cfg.get("judge", {})
    out_cfg = cfg.get("output", {})
    hardware_cfg = cfg.get("hardware", {})
    gpu_index = int((cfg.get("cost") or {}).get("gpu_index", 0))
    # Auto-detect hardware identity from GPU tooling; config is fallback only.
    hardware_id: str | None = detect_hardware_id(gpu_index) or hardware_cfg.get("id") or None
    # Peak TFLOPS: table lookup on detected id first, then config value.
    peak_tflops: float | None = None
    if hardware_id:
        peak_tflops = lookup_peak_tflops(hardware_id)
    if peak_tflops is None:
        cfg_tflops = hardware_cfg.get("peak_tflops")
        if cfg_tflops is not None:
            peak_tflops = float(cfg_tflops)
    if hardware_id:
        print(f"[hardware] id={hardware_id} peak_tflops={peak_tflops}", file=sys.stderr)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_cfg.get("dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset
    tasks = generate(
        n=int(ds_cfg["n"]),
        domains=list(ds_cfg["domains"]),
        seed=int(run_cfg.get("seed", 42)),
    )
    ds_path = Path("datasets") / f"tasks-{ts}.jsonl"
    write_jsonl(tasks, ds_path)
    print(f"[dataset] {len(tasks)} tasks -> {ds_path}", file=sys.stderr)

    if args.dry_run:
        # Also exercise the llama-swap generator so a misconfigured
        # models[] block trips CI's dry-run step instead of a live
        # benchmark run. The yaml round-trip guards against a future
        # regression where a non-primitive (e.g. pathlib.Path) leaks
        # into the emitted config — safe_dump would fail at runtime
        # otherwise; here it fails in CI.
        models_dir = resolve_models_dir(server_cfg)
        ls_cfg = llama_swap.build(cfg, str(models_dir))
        yaml.safe_dump(ls_cfg, sort_keys=False)
        n_backends = len(ls_cfg.get("models", {}))
        n_prompts = len(cfg.get("prompts") or [])
        n_cells = len(tasks) * n_prompts
        judge_mode = "off"
        if (cfg.get("judge") or {}).get("enabled"):
            judge_mode = (cfg.get("judge") or {}).get("mode", "pairwise_all")
        print(
            f"[dry-run] ok"
            f" · validation passed"
            f" · {len(tasks)} tasks"
            f" · {n_backends} backends (incl. judge)"
            f" · {n_prompts} prompts"
            f" · {n_cells} cells/model"
            f" · judge={judge_mode}",
            file=sys.stderr,
        )
        return 0

    if not LAUNCHER.exists():
        print(f"[error] missing launcher: {LAUNCHER}", file=sys.stderr)
        return 2

    models_dir = resolve_models_dir(server_cfg)
    prompts = cfg["prompts"]
    models = cfg["models"]
    timeout_s = float(run_cfg.get("timeout_s", 180))
    warmup = bool(run_cfg.get("warmup", True))
    swap_port = int(server_cfg.get("swap_port", 8080))
    boot_timeout = int(server_cfg.get("boot_timeout_s", 300))
    listen = f"127.0.0.1:{swap_port}"
    base_url = f"http://{listen}/v1"

    # 2. Resume: load prior run, plan pending cells.
    seed = int(run_cfg.get("seed", 42))
    reused_records: list[dict[str, Any]] = []
    reused_judgements: list[dict[str, Any]] = []
    pending_by_model: dict[str, set[tuple[str, str]]] | None = None
    pending_pairs: set | None = None
    pending_scores: set | None = None
    prior_run_id: str | None = None

    prior: dict[str, Any] | None = None
    if args.resume_run_id:
        from bench.store import StoreError as _StoreError
        from bench.store import read_record

        try:
            prior = read_record("runs", args.resume_run_id)
        except _StoreError as e:
            print(f"[error] resume-run-id: {e}", file=sys.stderr)
            return 1
        # Validate minimum required fields (same contract as load_prior).
        for field in ("tasks", "records"):
            if field not in prior:
                print(f"[error] stored run missing required field: {field!r}", file=sys.stderr)
                return 1
    elif args.resume_from:
        try:
            prior = load_prior(Path(args.resume_from))
        except ResumeError as e:
            print(f"[error] resume: {e}", file=sys.stderr)
            return 1

    if prior is not None:
        prior_run_id = prior.get("run_id")
        task_ids = [str(t.id) for t in tasks]
        prompt_ids = [str(p["id"]) for p in prompts]
        compat_errors = check_compat(cfg, seed=seed, task_ids=task_ids, prior=prior)
        for err in compat_errors:
            print(f"[resume-warn] {err}", file=sys.stderr)
        pending_by_model = build_pending(
            models,
            task_ids,
            prompt_ids,
            prior["records"],
            rerun_errors=args.rerun_errors,
            rerun_missing=args.rerun_missing,
            filter_models=set(args.resume_models) if args.resume_models else None,
            filter_tasks=set(args.resume_tasks) if args.resume_tasks else None,
            filter_prompts=set(args.resume_prompts) if args.resume_prompts else None,
        )
        complete = [r for r in prior["records"] if not r.get("error")]
        reused_records = tag_reused(complete, prior_run_id or "")
        n_reused = len(reused_records)
        n_pending = sum(len(v) for v in pending_by_model.values())
        # Judge resume planning
        prior_judgements = prior.get("judgements") or []
        if prior_judgements and judge_cfg.get("enabled"):
            judge_mode = judge_cfg.get("mode", "pairwise_all")
            if judge_mode == "pairwise_all":
                pending_pairs = build_pending_judge_pairs(
                    prior_judgements, models, task_ids, prompt_ids
                )
                reused_judgements = [
                    j
                    for j in prior_judgements
                    if j.get("mode") == "pairwise"
                    and not j.get("error")
                    and j.get("winner") is not None
                ]
                print(
                    f"[resume] judge: {len(reused_judgements)} pairs reused,"
                    f" {len(pending_pairs)} pending",
                    file=sys.stderr,
                )
            elif judge_mode == "scored":
                pending_scores = build_pending_judge_scores(
                    prior_judgements, models, task_ids, prompt_ids
                )
                reused_judgements = [
                    j
                    for j in prior_judgements
                    if j.get("mode") == "scored"
                    and not j.get("error")
                    and j.get("score") is not None
                ]
                print(
                    f"[resume] judge: {len(reused_judgements)} scores reused,"
                    f" {len(pending_scores)} pending",
                    file=sys.stderr,
                )
        print(
            f"[resume] prior={prior_run_id} reused={n_reused} pending_cells={n_pending}",
            file=sys.stderr,
        )

    # Minimal-capability floor tier (Rethunk-AI/bakeoff#23): runs per model
    # within its existing load (no extra swap). Fresh runs only — resume reruns
    # focus on pending main cells. Config-gated, default enabled.
    dumb_cfg = cfg.get("dumb_model_tier", {})
    floor_tasks = (
        load_floor_tasks() if bool(dumb_cfg.get("enabled", True)) and prior is None else None
    )
    if floor_tasks:
        print(f"[floor] dumb_model tier: {len(floor_tasks)} tasks", file=sys.stderr)

    # Collect hardware context once before proxy startup so it doesn't
    # interfere with GPU power/VRAM sampling during the benchmark.
    hardware_context = collect_hardware_context()
    if any(v is not None for v in hardware_context.values()):
        print(f"[hardware-ctx] {hardware_context}", file=sys.stderr)

    # Write a run-queue pending record so reporting can enumerate historical
    # runs without scanning the flat results/ directory.
    import uuid as _uuid

    from bench.store import delete_record as _store_delete
    from bench.store import write_record as _store_write

    _queue_id = str(_uuid.uuid4())
    _run_id_for_queue = run_cfg.get("name", ts)
    _store_write(
        "run_queue/pending",
        _queue_id,
        {"queue_id": _queue_id, "run_id": _run_id_for_queue, "status": "pending"},
    )
    _queue_terminal_status = "error"

    _write_proxy_config(cfg, models_dir, LLAMA_SWAP_CONFIG)
    proxy = _proxy_start(listen, LLAMA_SWAP_CONFIG, boot_timeout)

    try:
        # 3. Per-model phases (sequential — see invariant above).
        fresh_records = _run_model_phases(
            models,
            tasks,
            prompts,
            base_url,
            cost_cfg,
            timeout_s,
            warmup,
            pending_by_model,
            prior_run_id,
            hardware_id=hardware_id,
            peak_tflops=peak_tflops,
            floor_tasks=floor_tasks,
        )
        all_records = reused_records + fresh_records

        # 4. Judge phase
        fresh_judgements = run_judge_phase(
            judge_cfg,
            models,
            tasks,
            prompts,
            all_records,
            base_url,
            timeout_s,
            seed=seed,
            warmup=warmup,
            pending_pairs=pending_pairs,
            pending_scores=pending_scores,
        )
        judgements = reused_judgements + fresh_judgements
        _queue_terminal_status = "complete"
    finally:
        _proxy_stop(proxy)
        # Move queue record from pending to completed regardless of outcome.
        _store_write(
            "run_queue/completed",
            _queue_id,
            {
                "queue_id": _queue_id,
                "run_id": _run_id_for_queue,
                "status": _queue_terminal_status,
            },
        )
        import contextlib as _contextlib

        with _contextlib.suppress(Exception):
            _store_delete("run_queue/pending", _queue_id)

    # 5. Emit
    out_json = out_dir / f"run-{ts}.json"
    binary_dir = LLAMA_SWAP_CONFIG.parent
    provenance = collect_provenance(cfg, seed=seed, repo_root=HERE, binary_dir=binary_dir)
    hf_mode = args.hf_enrichment or run_cfg.get("hf_enrichment", "off")
    model_metadata = build_model_metadata(cfg)
    model_metadata = enrich_model_metadata(model_metadata, hf_mode, provenance["warnings"])
    # Post-hoc completeness-weighted rollup + run-level status (#23).
    cells_total = len(tasks) * len(prompts)
    model_scores, run_status = assemble_model_scores(models, all_records, cells_total)
    payload = {
        "run_id": run_cfg.get("name", ts),
        "timestamp": ts,
        "run_status": run_status,
        "config": cfg,
        "provenance": provenance,
        "model_metadata": model_metadata,
        "tasks": [asdict(t) for t in tasks],
        "records": all_records,
        "model_scores": model_scores,
        "judgements": judgements,
        "resumed_from": prior_run_id,
        "hardware": hardware_context,
    }

    # Persist the run record to the store so it is addressable by run ID.
    # The store record IS the payload dict; the flat results/ file remains
    # for backwards compatibility (e.g. existing --resume-from callers).
    _store_write("runs", payload["run_id"], payload)
    print(f"[store] runs/{payload['run_id']}.json", file=sys.stderr)

    # Config-gated Ed25519 signing.
    # Enable by adding a `signing` stanza to config.yaml:
    #
    #   signing:
    #     enabled: true
    #     key_path: "~/.bakeoff/runner_key.pem"   # Ed25519 private key PEM
    #     runner_id: "amd-8060s"                   # defaults to hardware.id if omitted
    signing_cfg = cfg.get("signing", {})
    if signing_cfg.get("enabled"):
        from bench.signing import load_private_key, sign_result

        key_path = Path(signing_cfg["key_path"]).expanduser()
        runner_id = (
            signing_cfg.get("runner_id")
            or (cfg.get("hardware") or {}).get("id")
            or "unknown-runner"
        )
        private_key = load_private_key(key_path)
        output = sign_result(payload, private_key, runner_id)
    else:
        output = payload

    with out_json.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"[out] {out_json}", file=sys.stderr)

    if out_cfg.get("emit_markdown") or out_cfg.get("emit_html"):
        from bench.report import emit_reports

        emit_reports(
            payload,
            out_dir,
            ts,
            md=bool(out_cfg.get("emit_markdown", True)),
            html=bool(out_cfg.get("emit_html", True)),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
