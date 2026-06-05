"""CLI entry point: bakeoff-report — generate reports for stored runs.

Usage::

    bakeoff-report [--run-id RUN_ID] [--list] [--format {md,html,both}] [--out-dir DIR]

  --list            Print stored run IDs newest-first (reversed ascending sort).
  --run-id RUN_ID   Generate a report for this run. Default: most recent.
  --format          Output format: md, html, or both (default: both).
  --out-dir DIR     Write output files here (default: current directory).

The report files are named run-<timestamp>.md and/or run-<timestamp>.html,
matching the naming convention used by the inline runner path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="bakeoff-report",
        description="Generate markdown/HTML reports from stored bakeoff runs.",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="Print stored run IDs newest-first and exit.",
    )
    ap.add_argument(
        "--run-id",
        metavar="RUN_ID",
        default=None,
        help="Run ID to report. Default: most recent stored run.",
    )
    ap.add_argument(
        "--format",
        choices=["md", "html", "both"],
        default="both",
        help="Output format (default: both).",
    )
    ap.add_argument(
        "--out-dir",
        metavar="DIR",
        default=".",
        help="Directory to write report files (default: current directory).",
    )
    args = ap.parse_args()

    from bench.store import StoreError, list_runs, read_record

    run_ids = list_runs()  # ascending by stem; reverse for newest-first

    if args.list:
        for rid in reversed(run_ids):
            print(rid)
        return 0

    if not run_ids and args.run_id is None:
        print("[bakeoff-report] no stored runs found in BAKEOFF_DATA_DIR", file=sys.stderr)
        return 1

    run_id = args.run_id if args.run_id is not None else run_ids[-1]

    try:
        payload = read_record("runs", run_id)
    except StoreError as e:
        print(f"[bakeoff-report] {e}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = payload.get("timestamp", run_id)
    emit_md = args.format in ("md", "both")
    emit_html = args.format in ("html", "both")

    from bench.report import emit_reports

    emit_reports(payload, out_dir, ts=ts, md=emit_md, html=emit_html)

    if emit_md:
        print(f"[out] {out_dir / f'run-{ts}.md'}")
    if emit_html:
        print(f"[out] {out_dir / f'run-{ts}.html'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
