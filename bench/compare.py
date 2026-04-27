"""Comparison of two bakeoff result JSON files.

Usage:
  python -m bench.compare <base.json> <candidate.json> [--output FILE]
  python -m bench.compare --help

Emits Markdown to stdout (or --output file). Compatibility warnings are
printed to stderr. Mismatched task/prompt/judge/seed are warnings, not
hard failures, unless --strict is passed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from bench.report import _detect_mode, _pairwise_rollup, _rollup, _scored_rollup

# --- Compatibility check -----------------------------------------------------


def _compat_warnings(base: dict[str, Any], cand: dict[str, Any]) -> list[str]:
    """Return a list of compatibility warning strings between two payloads."""
    warnings: list[str] = []

    def _sorted_ids(payload: dict[str, Any], key: str, id_field: str) -> list[str]:
        return sorted(str(e.get(id_field, "")) for e in payload.get(key) or [])

    base_tasks = sorted(str(t.get("id", "")) for t in base.get("tasks") or [])
    cand_tasks = sorted(str(t.get("id", "")) for t in cand.get("tasks") or [])
    if base_tasks != cand_tasks:
        warnings.append(
            f"task sets differ ({len(base_tasks)} base vs {len(cand_tasks)} candidate)"
        )

    base_prompts = _sorted_ids(base.get("config") or {}, "prompts", "id")
    cand_prompts = _sorted_ids(cand.get("config") or {}, "prompts", "id")
    if base_prompts != cand_prompts:
        warnings.append(
            f"prompt IDs differ: base={base_prompts} candidate={cand_prompts}"
        )

    base_models = _sorted_ids(base.get("config") or {}, "models", "id")
    cand_models = _sorted_ids(cand.get("config") or {}, "models", "id")
    if set(base_models) != set(cand_models):
        warnings.append(
            f"model ID sets differ: base={base_models} candidate={cand_models}"
        )

    base_mode = _detect_mode(base.get("judgements") or [])
    cand_mode = _detect_mode(cand.get("judgements") or [])
    if base_mode != cand_mode:
        warnings.append(f"judge mode differs: base={base_mode!r} candidate={cand_mode!r}")

    base_prov = base.get("provenance") or {}
    cand_prov = cand.get("provenance") or {}
    if (
        base_prov.get("seed") is not None
        and cand_prov.get("seed") is not None
        and base_prov["seed"] != cand_prov["seed"]
    ):
<<<<<<< HEAD
        warnings.append(f"seed differs: base={base_prov['seed']} candidate={cand_prov['seed']}")
=======
        warnings.append(
            f"seed differs: base={base_prov['seed']} candidate={cand_prov['seed']}"
        )
>>>>>>> 029e297 (fix(compare): clear lint blockers)

    if (
        base_prov.get("config_hash")
        and cand_prov.get("config_hash")
        and base_prov["config_hash"] != cand_prov["config_hash"]
    ):
        warnings.append(
            f"config hash differs: base={base_prov['config_hash']} "
            f"candidate={cand_prov['config_hash']}"
        )

    return warnings


# --- Delta helpers -----------------------------------------------------------


def _delta(a: float | None, b: float | None) -> str:
    """Format a signed delta string, or '—' if either value is missing."""
    if a is None or b is None:
        return "—"
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def _delta2(a: float | None, b: float | None, nd: int = 3) -> str:
    if a is None or b is None:
        return "—"
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.{nd}f}"


def _fmt(v: float | None, nd: int = 3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


# --- Markdown comparison output ----------------------------------------------


def compare_markdown(
    base: dict[str, Any],
    cand: dict[str, Any],
    base_label: str = "base",
    cand_label: str = "candidate",
    warnings: list[str] | None = None,
) -> str:
    lines: list[str] = []

    base_id = base.get("run_id", base.get("timestamp", base_label))
    cand_id = cand.get("run_id", cand.get("timestamp", cand_label))
    lines.append(f"# Comparison: {base_id} vs {cand_id}")
    lines.append("")

    if warnings:
        for w in warnings:
            lines.append(f"> **Warning:** {w}")
        lines.append("")

    base_roll = _rollup(base.get("records") or [])
    cand_roll = _rollup(cand.get("records") or [])
    all_models = sorted(set(base_roll) | set(cand_roll))

    # --- Core metrics delta --------------------------------------------------
    lines.append("## Core metrics")
    lines.append("")
    lines.append(
        f"| Model | Latency {base_label} (s) | Latency {cand_label} (s) | Δ lat | "
        f"TPS {base_label} | TPS {cand_label} | Δ TPS | "
        f"Heuristic {base_label} | Heuristic {cand_label} | Δ heur |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mid in all_models:
        br = base_roll.get(mid, {})
        cr = cand_roll.get(mid, {})
        lines.append(
            f"| {mid} "
            f"| {_fmt(br.get('latency_mean_s'))} "
            f"| {_fmt(cr.get('latency_mean_s'))} "
            f"| {_delta(br.get('latency_mean_s'), cr.get('latency_mean_s'))} "
            f"| {_fmt(br.get('tokens_per_sec_mean'), 2)} "
            f"| {_fmt(cr.get('tokens_per_sec_mean'), 2)} "
            f"| {_delta2(br.get('tokens_per_sec_mean'), cr.get('tokens_per_sec_mean'), 2)} "
            f"| {_fmt(br.get('quality_heuristic_mean'), 3)} "
            f"| {_fmt(cr.get('quality_heuristic_mean'), 3)} "
            f"| {_delta(br.get('quality_heuristic_mean'), cr.get('quality_heuristic_mean'))} |"
        )
    lines.append("")

    # --- Energy / cost delta -------------------------------------------------
    lines.append("## Energy and cost")
    lines.append("")
    lines.append(
        f"| Model | Energy {base_label} (Wh) | Energy {cand_label} (Wh) | Δ Wh | "
        f"Cost {base_label} (USD) | Cost {cand_label} (USD) | Δ USD |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for mid in all_models:
        br = base_roll.get(mid, {})
        cr = cand_roll.get(mid, {})
        lines.append(
            f"| {mid} "
            f"| {_fmt(br.get('energy_wh_total'))} "
            f"| {_fmt(cr.get('energy_wh_total'))} "
            f"| {_delta(br.get('energy_wh_total'), cr.get('energy_wh_total'))} "
            f"| {_fmt(br.get('cost_usd_total'), 4)} "
            f"| {_fmt(cr.get('cost_usd_total'), 4)} "
            f"| {_delta2(br.get('cost_usd_total'), cr.get('cost_usd_total'), 4)} |"
        )
    lines.append("")

    # --- Judge comparison ----------------------------------------------------
    base_mode = _detect_mode(base.get("judgements") or [])
    cand_mode = _detect_mode(cand.get("judgements") or [])

    if base_mode == "scored" and cand_mode == "scored":
        bsr = _scored_rollup(base.get("judgements") or [])
        csr = _scored_rollup(cand.get("judgements") or [])
        judge_models = sorted(set(bsr["per_model"]) | set(csr["per_model"]))
        if judge_models:
            lines.append("## Judge scores (scored mode)")
            lines.append("")
            lines.append(
                f"| Model | Score {base_label} | Score {cand_label} | Δ score |"
            )
            lines.append("|---|---:|---:|---:|")
            for mid in judge_models:
                bm = bsr["per_model"].get(mid, {})
                cm = csr["per_model"].get(mid, {})
                lines.append(
                    f"| {mid} "
                    f"| {_fmt(bm.get('mean'), 2)} "
                    f"| {_fmt(cm.get('mean'), 2)} "
                    f"| {_delta2(bm.get('mean'), cm.get('mean'), 2)} |"
                )
            lines.append("")

    elif base_mode == "pairwise" and cand_mode == "pairwise":
        bpr = _pairwise_rollup(base.get("judgements") or [])
        cpr = _pairwise_rollup(cand.get("judgements") or [])
        pw_models = sorted(set(bpr["models"]) | set(cpr["models"]))
        if pw_models:
            lines.append("## Judge win rates (pairwise mode)")
            lines.append("")
            lines.append(
                f"| Model | Win rate {base_label} | Win rate {cand_label} | Δ win rate |"
            )
            lines.append("|---|---:|---:|---:|")
            for mid in pw_models:
                bm = bpr["per_model"].get(mid, {})
                cm = cpr["per_model"].get(mid, {})
                bwr = bm.get("win_rate")
                cwr = cm.get("win_rate")
<<<<<<< HEAD
                base_win_rate = bwr * 100 if bwr is not None else None
                cand_win_rate = cwr * 100 if cwr is not None else None
                lines.append(
                    f"| {mid} "
                    f"| {_fmt(base_win_rate, 1)}% "
                    f"| {_fmt(cand_win_rate, 1)}% "
                    f"| {_delta2(base_win_rate, cand_win_rate, 1)}% |"
=======
                bwr_pct = bwr * 100 if bwr is not None else None
                cwr_pct = cwr * 100 if cwr is not None else None
                lines.append(
                    f"| {mid} "
                    f"| {_fmt(bwr_pct, 1)}% "
                    f"| {_fmt(cwr_pct, 1)}% "
                    f"| {_delta2(bwr_pct, cwr_pct, 1)}% |"
>>>>>>> 029e297 (fix(compare): clear lint blockers)
                )
            lines.append("")

    return "\n".join(lines)


# --- Public API --------------------------------------------------------------


def load_result(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def compare(
    base_path: Path,
    cand_path: Path,
    strict: bool = False,
    output: Path | None = None,
) -> int:
    base = load_result(base_path)
    cand = load_result(cand_path)

    warns = _compat_warnings(base, cand)
    for w in warns:
        print(f"[compare] warning: {w}", file=sys.stderr)

    if strict and warns:
        print("[compare] incompatible runs; aborting (--strict)", file=sys.stderr)
        return 1

    md = compare_markdown(
        base, cand,
        base_label=base_path.stem,
        cand_label=cand_path.stem,
        warnings=warns,
    )

    if output:
        output.write_text(md + "\n")
        print(f"[compare] written to {output}", file=sys.stderr)
    else:
        print(md)

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare two bakeoff result JSON files and emit Markdown."
    )
    ap.add_argument("base", type=Path, help="Base result JSON file")
    ap.add_argument("candidate", type=Path, help="Candidate result JSON file")
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="Write Markdown to this file (default: stdout)")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero on any compatibility warning")
    args = ap.parse_args()
    return compare(args.base, args.candidate, strict=args.strict, output=args.output)


if __name__ == "__main__":
    sys.exit(main())
