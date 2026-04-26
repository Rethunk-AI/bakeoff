"""Resume support for bakeoff benchmark runs.

Stable row keys:
  model row:    (task_id, prompt_id, model_id)
  pairwise key: (task_id, prompt_id, frozenset({a_model, b_model}))
  scored key:   (task_id, prompt_id, model_id)

A model row is "complete" when it has no error field. Missing rows and
errored rows are both treated as pending by default; pass rerun_errors=False
or rerun_missing=False to build_pending to narrow the scope.

Judge resume: build_pending_judge_pairs / build_pending_judge_scores compute
the subset of judge cells not yet completed in a prior run.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class ResumeError(ValueError):
    """Raised when a prior result cannot be used for resume."""


# --- Stable row keys ---------------------------------------------------------


def row_key(record: dict[str, Any]) -> tuple[str, str, str]:
    """Return (task_id, prompt_id, model_id) — the stable identity of one model row."""
    return (str(record["task_id"]), str(record["prompt_id"]), str(record["model_id"]))


def pairwise_key(j: dict[str, Any]) -> tuple[str, str, frozenset]:
    """Return (task_id, prompt_id, frozenset({a_model, b_model})) for a pairwise judgement."""
    return (str(j["task_id"]), str(j["prompt_id"]),
            frozenset({str(j["a_model"]), str(j["b_model"])}))


def scored_key(j: dict[str, Any]) -> tuple[str, str, str]:
    """Return (task_id, prompt_id, model_id) for a scored judgement."""
    return (str(j["task_id"]), str(j["prompt_id"]), str(j["model_id"]))


# --- Load prior result -------------------------------------------------------


def load_prior(path: Path) -> dict[str, Any]:
    """Load a prior result JSON. Raises ResumeError on any failure."""
    try:
        with path.open() as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ResumeError(f"prior result not found: {path}") from None
    except json.JSONDecodeError as e:
        raise ResumeError(f"malformed JSON in {path}: {e}") from e
    if not isinstance(data, dict):
        raise ResumeError(f"{path} is not a JSON object")
    for field in ("tasks", "records"):
        if field not in data:
            raise ResumeError(f"prior result missing required field: {field!r}")
    return data


# --- Compatibility check -----------------------------------------------------


def check_compat(
    cfg: dict[str, Any],
    seed: int,
    task_ids: list[str],
    prior: dict[str, Any],
) -> list[str]:
    """Return a list of incompatibility reasons. Empty means compatible."""
    errors: list[str] = []

    # Seed
    prov = prior.get("provenance") or {}
    if prov.get("seed") is not None and int(prov["seed"]) != seed:
        errors.append(
            f"seed mismatch: prior={prov['seed']} current={seed}"
        )

    # Dataset n + domains (via task ID set identity)
    prior_task_ids = sorted(str(t.get("id", "")) for t in prior.get("tasks") or [])
    current_task_ids = sorted(task_ids)
    if prior_task_ids != current_task_ids:
        errors.append(
            f"task set mismatch: {len(prior_task_ids)} prior tasks vs "
            f"{len(current_task_ids)} current tasks"
        )

    # Prompt IDs
    current_pids = sorted(str(p["id"]) for p in cfg.get("prompts") or [])
    prior_pids = sorted(
        str(p["id"]) for p in (prior.get("config") or {}).get("prompts") or []
    )
    if current_pids != prior_pids:
        errors.append(
            f"prompt ID mismatch: prior={prior_pids} current={current_pids}"
        )

    # Model IDs
    current_mids = sorted(str(m["id"]) for m in cfg.get("models") or [])
    prior_mids = sorted(
        str(m["id"]) for m in (prior.get("config") or {}).get("models") or []
    )
    if set(current_mids) != set(prior_mids):
        errors.append(
            f"model ID mismatch: prior={prior_mids} current={current_mids}"
        )

    return errors


# --- Pending-cell planning ---------------------------------------------------


def build_pending(
    models: list[dict[str, Any]],
    task_ids: list[str],
    prompt_ids: list[str],
    prior_records: list[dict[str, Any]],
    *,
    rerun_errors: bool = True,
    rerun_missing: bool = True,
    filter_models: set[str] | None = None,
    filter_tasks: set[str] | None = None,
    filter_prompts: set[str] | None = None,
) -> dict[str, set[tuple[str, str]]]:
    """Return {model_id → set of (task_id, prompt_id) cells that still need running}.

    By default both missing and errored cells are pending. Pass
    rerun_errors=False to skip errored rows, rerun_missing=False to skip
    missing rows. filter_models/tasks/prompts restrict which IDs are considered.
    """
    complete: set[tuple[str, str, str]] = {
        row_key(r) for r in prior_records if not r.get("error")
    }
    errored: set[tuple[str, str, str]] = {
        row_key(r) for r in prior_records if r.get("error")
    }
    pending: dict[str, set[tuple[str, str]]] = {}
    for m in models:
        mid = str(m["id"])
        if filter_models is not None and mid not in filter_models:
            pending[mid] = set()
            continue
        cells: set[tuple[str, str]] = set()
        for tid in task_ids:
            if filter_tasks is not None and tid not in filter_tasks:
                continue
            for pid in prompt_ids:
                if filter_prompts is not None and pid not in filter_prompts:
                    continue
                key = (tid, pid, mid)
                if key in complete:
                    continue
                if key in errored:
                    if rerun_errors:
                        cells.add((tid, pid))
                else:
                    if rerun_missing:
                        cells.add((tid, pid))
        pending[mid] = cells
    return pending


def build_pending_judge_pairs(
    prior_judgements: list[dict[str, Any]],
    models: list[dict[str, Any]],
    task_ids: list[str],
    prompt_ids: list[str],
) -> set[tuple[str, str, frozenset]]:
    """Return pairwise keys not yet judged in the prior run.

    Keys: (task_id, prompt_id, frozenset({a_model, b_model}))
    Errored judgements (winner=None with error field) are treated as pending.
    """
    done: set[tuple[str, str, frozenset]] = {
        pairwise_key(j)
        for j in prior_judgements
        if j.get("mode") == "pairwise" and not j.get("error") and j.get("winner") is not None
    }
    pending: set[tuple[str, str, frozenset]] = set()
    for a_m, b_m in itertools.combinations(models, 2):
        a_id, b_id = str(a_m["id"]), str(b_m["id"])
        for tid in task_ids:
            for pid in prompt_ids:
                key: tuple[str, str, frozenset] = (tid, pid, frozenset({a_id, b_id}))
                if key not in done:
                    pending.add(key)
    return pending


def build_pending_judge_scores(
    prior_judgements: list[dict[str, Any]],
    models: list[dict[str, Any]],
    task_ids: list[str],
    prompt_ids: list[str],
) -> set[tuple[str, str, str]]:
    """Return scored keys not yet judged in the prior run.

    Keys: (task_id, prompt_id, model_id)
    Errored judgements are treated as pending.
    """
    done: set[tuple[str, str, str]] = {
        scored_key(j)
        for j in prior_judgements
        if j.get("mode") == "scored" and not j.get("error") and j.get("score") is not None
    }
    pending: set[tuple[str, str, str]] = set()
    for m in models:
        mid = str(m["id"])
        for tid in task_ids:
            for pid in prompt_ids:
                key = (tid, pid, mid)
                if key not in done:
                    pending.add(key)
    return pending


# --- Metadata tagging --------------------------------------------------------


def tag_reused(records: list[dict[str, Any]], source_run_id: str) -> list[dict[str, Any]]:
    """Return copies of records tagged with resumed_from."""
    return [{**r, "resumed_from": source_run_id} for r in records]


def tag_fresh(records: list[dict[str, Any]], source_run_id: str) -> list[dict[str, Any]]:
    """Tag freshly-measured records with source_run_id (the run they resume)."""
    return [{**r, "source_run_id": source_run_id} for r in records]
