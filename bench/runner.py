"""Benchmark runner: sequential model phases (one boot at a time).

Phases:
  1. Generate dataset.
  2. For each model: boot llama.cpp podman server -> run all (task x prompt)
     calls -> tear down -> next model.
  3. Judge phase (optional): boot judge model -> run pairwise judge on stored
     A/B responses -> tear down.
  4. Emit reports (JSON + Markdown + static HTML dashboard).

The unified-memory APU (AMD Strix Halo) makes concurrent model serving wasteful;
sequential phases keep VRAM usage bounded to one model at a time.
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

from bench.clients import ChatClient, ChatResult
from bench.dataset import Task, generate, write_jsonl
from bench.metrics import (
    cost_usd,
    energy_wh,
    invert_winner,
    judge_pair_randomized,
    judge_score_prompt,
    parse_judge,
    parse_score,
    sample_power,
    score_heuristic,
)

HERE = Path(__file__).resolve().parent.parent
SERVE_SH = HERE / "bin" / "serve.sh"


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_models_dir(server_cfg: dict[str, Any]) -> Path:
    p = server_cfg.get("models_dir", "~/.lmstudio/models")
    return Path(os.path.expanduser(p)).resolve()


def _serve_env(server_cfg: dict[str, Any], models_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["MODELS_DIR"] = str(models_dir)
    env["IMAGE"] = str(server_cfg.get("image", "ghcr.io/ggml-org/llama.cpp:server-vulkan"))
    env["NGL"] = str(server_cfg.get("ngl", 99))
    env["UBATCH"] = str(server_cfg.get("ubatch", 512))
    env["CACHE_TYPE_K"] = str(server_cfg.get("cache_type_k", "q8_0"))
    env["CACHE_TYPE_V"] = str(server_cfg.get("cache_type_v", "q8_0"))
    env["FLASH_ATTN"] = "1" if server_cfg.get("flash_attn", True) else "0"
    env["JINJA"] = "1" if server_cfg.get("jinja", True) else "0"
    return env


def boot(
    gguf_rel: str,
    alias: str,
    port: int,
    ctx: int,
    server_cfg: dict[str, Any],
    models_dir: Path,
    n_cpu_moe: int | None = None,
) -> str:
    """Start container. Returns container name on success; raises on failure."""
    args = [str(SERVE_SH), "up", gguf_rel, alias, str(port), str(ctx)]
    if n_cpu_moe is not None:
        args.append(str(n_cpu_moe))
    res = subprocess.run(args, env=_serve_env(server_cfg, models_dir),
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"serve.sh up failed: {res.stderr.strip() or res.stdout.strip()}")
    name = res.stdout.strip().splitlines()[-1].strip()
    return name


def wait_ready(port: int, timeout_s: int) -> None:
    res = subprocess.run(
        [str(SERVE_SH), "wait", str(port), str(timeout_s)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"server not ready on :{port}: {res.stderr.strip()}")


def teardown(container_name: str) -> None:
    subprocess.run([str(SERVE_SH), "down", container_name],
                   capture_output=True, text=True, check=False)


def call_one(
    client: ChatClient,
    system: str,
    user: str,
    gpu_index: int,
    cost_enabled: bool,
    kwh_rate: float,
) -> tuple[ChatResult, float | None, float | None]:
    p0 = sample_power(gpu_index) if cost_enabled else None
    res = client.chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    p1 = sample_power(gpu_index) if cost_enabled else None
    avg_w: float | None = None
    if p0 and p1 and p0.ok and p1.ok:
        avg_w = (p0.watts + p1.watts) / 2.0  # type: ignore[operator]
    wh = energy_wh(avg_w, res.latency_s)
    usd = cost_usd(wh, kwh_rate) if cost_enabled else None
    return res, wh, usd


def run_model_phase(
    model_cfg: dict[str, Any],
    tasks: list[Task],
    prompts: list[dict[str, Any]],
    server_cfg: dict[str, Any],
    models_dir: Path,
    cost_cfg: dict[str, Any],
    timeout_s: float,
) -> list[dict[str, Any]]:
    mid = model_cfg["id"]
    port = int(server_cfg["port"])
    ctx = int(model_cfg.get("ctx", server_cfg.get("ctx", 4096)))
    boot_timeout = int(server_cfg.get("boot_timeout_s", 300))

    print(f"[phase] booting {mid} ({model_cfg['gguf']})", file=sys.stderr)
    container = boot(
        model_cfg["gguf"], model_cfg["alias"], port, ctx,
        server_cfg, models_dir,
        n_cpu_moe=model_cfg.get("n_cpu_moe"),
    )
    records: list[dict[str, Any]] = []
    try:
        wait_ready(port, boot_timeout)
        client = ChatClient(
            base_url=f"http://127.0.0.1:{port}/v1",
            model=model_cfg["alias"],
            api_key="none",
            timeout_s=timeout_s,
        )
        cost_enabled = bool(cost_cfg.get("enabled"))
        kwh = float(cost_cfg.get("kwh_rate_usd", 0.0))
        gpu_i = int(cost_cfg.get("gpu_index", 0))
        for task, prm in itertools.product(tasks, prompts):
            try:
                res, wh, usd = call_one(client, prm["system"], task.user_prompt,
                                        gpu_i, cost_enabled, kwh)
                records.append({
                    "task_id": task.id, "domain": task.domain,
                    "prompt_id": prm["id"], "model_id": mid,
                    "text": res.text,
                    "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                    "latency_s": res.latency_s,
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
    finally:
        print(f"[phase] tearing down {mid}", file=sys.stderr)
        teardown(container)
    return records


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
    server_cfg: dict[str, Any],
    models_dir: Path,
    timeout_s: float,
    seed: int,
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

    port = int(server_cfg["port"])
    ctx = int(judge_cfg.get("ctx", server_cfg.get("ctx", 8192)))
    boot_timeout = int(server_cfg.get("boot_timeout_s", 300))

    print(f"[phase] booting judge ({judge_cfg['gguf']}), mode={mode}", file=sys.stderr)
    container = boot(
        judge_cfg["gguf"], judge_cfg["alias"], port, ctx,
        server_cfg, models_dir,
        n_cpu_moe=judge_cfg.get("n_cpu_moe"),
    )
    judgements: list[dict[str, Any]] = []
    try:
        wait_ready(port, boot_timeout)
        judge = ChatClient(
            base_url=f"http://127.0.0.1:{port}/v1",
            model=judge_cfg["alias"],
            api_key="none",
            timeout_s=timeout_s,
        )
        if mode == "pairwise_all":
            rng = random.Random(seed)
            judgements = _pairwise_all_phase(judge, models, tasks, prompts, records, rng)
        else:  # scored
            judgements = _scored_phase(judge, models, tasks, prompts, records)
    finally:
        print("[phase] tearing down judge", file=sys.stderr)
        teardown(container)
    return judgements


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse + gen dataset, no container or network calls.")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
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
        print("[dry-run] dataset ok, skipping container phases", file=sys.stderr)
        return 0

    if not SERVE_SH.exists():
        print(f"[error] missing launcher: {SERVE_SH}", file=sys.stderr)
        return 2

    models_dir = resolve_models_dir(server_cfg)
    prompts = cfg["prompts"]
    models = cfg["models"]
    timeout_s = float(run_cfg.get("timeout_s", 180))

    # 2. Per-model phases (sequential)
    all_records: list[dict[str, Any]] = []
    for m in models:
        recs = run_model_phase(m, tasks, prompts, server_cfg, models_dir,
                               cost_cfg, timeout_s)
        all_records.extend(recs)

    # 3. Judge phase
    judgements = run_judge_phase(
        judge_cfg, models, tasks, prompts, all_records,
        server_cfg, models_dir, timeout_s,
        seed=int(run_cfg.get("seed", 42)),
    )

    # 4. Emit
    out_json = out_dir / f"run-{ts}.json"
    payload = {
        "run_id": run_cfg.get("name", ts),
        "timestamp": ts,
        "config": cfg,
        "tasks": [asdict(t) for t in tasks],
        "records": all_records,
        "judgements": judgements,
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
