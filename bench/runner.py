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
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from bench import llama_swap
from bench.clients import ChatClient, ChatResult
from bench.config import ConfigError, load_config, validate_config
from bench.dataset import Task, generate, write_jsonl
from bench.metrics import (
    PowerSampler,
    cost_usd,
    invert_winner,
    judge_pair_randomized,
    judge_score_prompt,
    parse_judge,
    parse_score,
    score_heuristic,
)
from bench.provenance import build_model_metadata, enrich_model_metadata
from bench.provenance import collect as collect_provenance
from bench.resume import (
    ResumeError,
    build_pending,
    check_compat,
    load_prior,
    tag_fresh,
    tag_reused,
)

HERE = Path(__file__).resolve().parent.parent
LAUNCHER = HERE / "bin" / "llama-swap.sh"
LLAMA_SWAP_CONFIG = HERE / ".cache" / "llama-swap" / "config.yaml"


def resolve_models_dir(server_cfg: dict[str, Any]) -> Path:
    p = server_cfg.get("models_dir", "~/.lmstudio/models")
    return Path(os.path.expanduser(p)).resolve()


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
) -> tuple[ChatResult, float | None, float | None]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if not cost_enabled:
        res = client.chat(messages)
        return res, None, None
    with PowerSampler(hz=sample_hz, gpu_index=gpu_index) as sampler:
        res = client.chat(messages)
    wh = sampler.energy_wh
    usd = cost_usd(wh, kwh_rate)
    return res, wh, usd


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
        client.chat([
            {"role": "system", "content": WARMUP_SYSTEM},
            {"role": "user", "content": WARMUP_USER},
        ])
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
) -> list[dict[str, Any]]:
    """Run every (task x prompt) cell for one model against the proxy.

    `model_cfg["id"]` is the llama-swap routing key (not `alias` — the
    alias stays inside the generated `cmd` as llama.cpp's `-a` flag).

    When `pending` is provided, only (task_id, prompt_id) pairs in that
    set are executed; all others are silently skipped.
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

    records: list[dict[str, Any]] = []
    for task, prm in itertools.product(tasks, prompts):
        if pending is not None and (str(task.id), str(prm["id"])) not in pending:
            continue
        try:
            res, wh, usd = call_one(client, prm["system"], task.user_prompt,
                                    gpu_i, cost_enabled, kwh, sample_hz)
            records.append({
                "task_id": task.id, "domain": task.domain,
                "prompt_id": prm["id"], "model_id": mid,
                "text": res.text,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "latency_s": res.latency_s,
                "ttft_s": res.ttft_s,
                "tokens_per_sec": res.tokens_per_sec,
                "energy_wh": wh, "cost_usd": usd,
                "quality_heuristic": score_heuristic(task.scorer, task.expected, res.text),
                "error": None,
            })
            print(f"[run] {mid} {prm['id']} {task.id} {res.latency_s:.2f}s",
                  file=sys.stderr)
        except Exception as e:
            records.append({
                "task_id": task.id, "domain": task.domain,
                "prompt_id": prm["id"], "model_id": mid,
                "error": str(e),
            })
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
        recs = run_model_phase(m, tasks, prompts, base_url, cost_cfg, timeout_s,
                               warmup=warmup, pending=pending)
        if prior_run_id is not None:
            recs = tag_fresh(recs, prior_run_id)
        fresh.extend(recs)
    return fresh


def _pairwise_all_phase(
    judge: ChatClient,
    models: list[dict[str, Any]],
    tasks: list[Task],
    prompts: list[dict[str, Any]],
    records: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Round-robin all unordered pairs (N choose 2). Randomize slot order
    per call to de-bias positional preference; invert winner on swap."""
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {
        (r["task_id"], r["prompt_id"], r["model_id"]): r
        for r in records if not r.get("error")
    }
    judgements: list[dict[str, Any]] = []
    for a_m, b_m in itertools.combinations(models, 2):
        a_id, b_id = a_m["id"], b_m["id"]
        for task, prm in itertools.product(tasks, prompts):
            a = by_key.get((task.id, prm["id"], a_id))
            b = by_key.get((task.id, prm["id"], b_id))
            if not a or not b:
                continue
            msgs, order = judge_pair_randomized(task.user_prompt, a["text"], b["text"], rng)
            try:
                jr = judge.chat(msgs)
                raw_winner = parse_judge(jr.text)
                winner = invert_winner(raw_winner) if order == "BA" else raw_winner
                judgements.append({
                    "mode": "pairwise",
                    "task_id": task.id, "prompt_id": prm["id"],
                    "a_model": a_id, "b_model": b_id,
                    "order": order, "winner": winner,
                    "judge_raw": jr.text,
                })
                print(f"[judge-pair] {a_id} vs {b_id} {task.id} {prm['id']} "
                      f"({order}) -> {winner}", file=sys.stderr)
            except Exception as e:
                judgements.append({
                    "mode": "pairwise",
                    "task_id": task.id, "prompt_id": prm["id"],
                    "a_model": a_id, "b_model": b_id,
                    "order": order, "winner": None,
                    "error": str(e),
                })
                print(f"[judge-err] {a_id} vs {b_id} {task.id} {prm['id']}: {e}",
                      file=sys.stderr)
    return judgements


def _scored_phase(
    judge: ChatClient,
    models: list[dict[str, Any]],
    tasks: list[Task],
    prompts: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Absolute 1-5 rubric score per (model, task, prompt). Linear in N models."""
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {
        (r["task_id"], r["prompt_id"], r["model_id"]): r
        for r in records if not r.get("error")
    }
    judgements: list[dict[str, Any]] = []
    for m in models:
        mid = m["id"]
        for task, prm in itertools.product(tasks, prompts):
            rec = by_key.get((task.id, prm["id"], mid))
            if not rec:
                continue
            try:
                jr = judge.chat(judge_score_prompt(task.user_prompt, rec["text"]))
                score = parse_score(jr.text)
                judgements.append({
                    "mode": "scored",
                    "task_id": task.id, "prompt_id": prm["id"],
                    "model_id": mid,
                    "score": score, "judge_raw": jr.text,
                })
                print(f"[judge-score] {mid} {task.id} {prm['id']} -> {score}",
                      file=sys.stderr)
            except Exception as e:
                judgements.append({
                    "mode": "scored",
                    "task_id": task.id, "prompt_id": prm["id"],
                    "model_id": mid,
                    "score": None, "error": str(e),
                })
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
        print(f"[warn] unknown judge.mode {mode!r}; skipping judge phase",
              file=sys.stderr)
        return []
    if mode == "pairwise_all" and len(models) < 2:
        print("[warn] pairwise_all needs >= 2 models; skipping judge phase",
              file=sys.stderr)
        return []
    if mode == "scored" and len(models) < 1:
        return []

    judge_id = str(judge_cfg.get("id") or "judge")
    print(f"[phase] judge ({judge_id}), mode={mode}", file=sys.stderr)

    judge = ChatClient(
        base_url=base_url,
        model=judge_id,
        api_key="none",
        timeout_s=timeout_s,
    )
    if warmup:
        print(f"[warmup] {judge_id}", file=sys.stderr)
        _warmup(judge)

    if mode == "pairwise_all":
        rng = random.Random(seed)
        return _pairwise_all_phase(judge, models, tasks, prompts, records, rng)
    return _scored_phase(judge, models, tasks, prompts, records)


# --- Entry point ------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse + gen dataset, no proxy startup or network calls.")
    ap.add_argument("--resume-from", metavar="RESULT_JSON",
                    help="Prior result JSON. Reuses complete rows, reruns errored/missing.")
    ap.add_argument("--hf-enrichment", choices=["off", "best-effort", "strict"],
                    default=None,
                    help="HuggingFace metadata enrichment (overrides run.hf_enrichment in config).")
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
    pending_by_model: dict[str, set[tuple[str, str]]] | None = None
    prior_run_id: str | None = None

    if args.resume_from:
        try:
            prior = load_prior(Path(args.resume_from))
        except ResumeError as e:
            print(f"[error] resume: {e}", file=sys.stderr)
            return 1
        prior_run_id = prior.get("run_id")
        task_ids = [str(t.id) for t in tasks]
        prompt_ids = [str(p["id"]) for p in prompts]
        compat_errors = check_compat(cfg, seed=seed, task_ids=task_ids, prior=prior)
        for err in compat_errors:
            print(f"[resume-warn] {err}", file=sys.stderr)
        pending_by_model = build_pending(models, task_ids, prompt_ids, prior["records"])
        complete = [r for r in prior["records"] if not r.get("error")]
        reused_records = tag_reused(complete, prior_run_id or "")
        n_reused = len(reused_records)
        n_pending = sum(len(v) for v in pending_by_model.values())
        print(f"[resume] prior={prior_run_id} reused={n_reused} pending_cells={n_pending}",
              file=sys.stderr)

    _write_proxy_config(cfg, models_dir, LLAMA_SWAP_CONFIG)
    proxy = _proxy_start(listen, LLAMA_SWAP_CONFIG, boot_timeout)

    try:
        # 3. Per-model phases (sequential — see invariant above).
        fresh_records = _run_model_phases(
            models, tasks, prompts, base_url, cost_cfg, timeout_s, warmup,
            pending_by_model, prior_run_id,
        )
        all_records = reused_records + fresh_records

        # 4. Judge phase
        judgements = run_judge_phase(
            judge_cfg, models, tasks, prompts, all_records,
            base_url, timeout_s,
            seed=seed,
            warmup=warmup,
        )
    finally:
        _proxy_stop(proxy)

    # 5. Emit
    out_json = out_dir / f"run-{ts}.json"
    binary_dir = LLAMA_SWAP_CONFIG.parent
    provenance = collect_provenance(cfg, seed=seed, repo_root=HERE, binary_dir=binary_dir)
    hf_mode = args.hf_enrichment or run_cfg.get("hf_enrichment", "off")
    model_metadata = build_model_metadata(cfg)
    model_metadata = enrich_model_metadata(model_metadata, hf_mode, provenance["warnings"])
    payload = {
        "run_id": run_cfg.get("name", ts),
        "timestamp": ts,
        "config": cfg,
        "provenance": provenance,
        "model_metadata": model_metadata,
        "tasks": [asdict(t) for t in tasks],
        "records": all_records,
        "judgements": judgements,
        "resumed_from": prior_run_id,
    }
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[out] {out_json}", file=sys.stderr)

    if out_cfg.get("emit_markdown") or out_cfg.get("emit_html"):
        from bench.report import emit_reports
        emit_reports(payload, out_dir, ts,
                     md=bool(out_cfg.get("emit_markdown", True)),
                     html=bool(out_cfg.get("emit_html", True)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
