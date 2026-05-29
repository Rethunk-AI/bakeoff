"""Disk-backed run queue for the bakeoff harness (OPT-IN, standalone default unchanged).

The standalone runner (bench.runner) reads config.yaml directly and does not
interact with this module. The queue is an opt-in overlay for multi-runner /
multi-host scenarios. See AGENTS.md "Design invariants" — do not wire this
into bench.runner's default matrix loop.

Disk layout::

    <BAKEOFF_DATA_DIR>/
      run_queue/
        pending/<queue_id>.json    — PENDING, CLAIMED, IN_PROGRESS items
        completed/<queue_id>.json  — COMPLETE, FAILED (terminal), CANCELLED items

Claim race-safety (concurrent runners):
    The claim mutex is the *atomic rename* of the pending file to a
    per-runner lock name in the same directory. Only the process that wins
    ``os.rename(pending/<id>.json → pending/<id>.lck-<runner_id>)`` proceeds;
    all other processes that try to rename the same source get FileNotFoundError
    and move on to the next candidate. After the winner updates the record, it
    renames the lock file back to the original name (still in pending/ —
    status is now CLAIMED). This gives mutual exclusion without a DB or
    advisory lock daemon.

    Test: enqueue one item, claim from N concurrent threads — exactly one
    wins (rest return None).

OPT-IN seam for queue workers:
    A future queue-driven runner can call:
        item = queue.claim(runner_id)
        if item: queue.mark_in_progress(item["queue_id"], runner_id)
        # ... run the benchmark cell ...
        queue.complete(item["queue_id"])
    The bench.runner.run_model_phase / main functions are the natural attach
    points, but that wiring is NOT done here to preserve the standalone default.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_PENDING = "pending"
_COMPLETED = "completed"

# Backoff constants (per #13 design).
_RETRY_BACKOFF_MINUTES = 5    # retry_after = now + N * attempt_count minutes
_RETRY_PRIORITY_BUMP = 5      # priority += N * attempt_count

_STATUS_PENDING = "PENDING"
_STATUS_CLAIMED = "CLAIMED"
_STATUS_IN_PROGRESS = "IN_PROGRESS"
_STATUS_COMPLETE = "COMPLETE"
_STATUS_FAILED = "FAILED"
_STATUS_CANCELLED = "CANCELLED"


# ---------------------------------------------------------------------------
# Datetime helpers (py310-safe: fromisoformat() rejects trailing 'Z' on <3.11)
# ---------------------------------------------------------------------------

_ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime(_ISO_FMT)


def _parse_dt(s: str) -> datetime:
    """Parse an ISO-8601 UTC string (always ends in Z) into an aware datetime."""
    return datetime.strptime(s, _ISO_FMT).replace(tzinfo=timezone.utc)


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Internal directory helpers (run_queue is a nested sub-tree, not a flat table)
# ---------------------------------------------------------------------------


def _queue_dir() -> Path:
    from bench.store import data_dir

    return data_dir() / "run_queue"


def _pending_dir() -> Path:
    d = _queue_dir() / _PENDING
    d.mkdir(parents=True, exist_ok=True)
    return d


def _completed_dir() -> Path:
    d = _queue_dir() / _COMPLETED
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pending_path(queue_id: str) -> Path:
    return _pending_dir() / f"{queue_id}.json"


def _completed_path(queue_id: str) -> Path:
    return _completed_dir() / f"{queue_id}.json"


def _lock_path(queue_id: str, runner_id: str) -> Path:
    return _pending_dir() / f"{queue_id}.lck-{runner_id}"


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    """Atomically write *data* as JSON to *path* (temp + os.replace)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _stamp(item: dict[str, Any]) -> dict[str, Any]:
    """Stamp updated_at and ensure created_at is set. Does NOT mutate the input."""
    out = dict(item)
    now = _utc_now()
    out.setdefault("created_at", now)
    out["updated_at"] = now
    return out


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data: dict[str, Any] = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enqueue(item: dict[str, Any]) -> str:
    """Write *item* to the pending queue. Returns the queue_id.

    Caller must supply at least ``queue_id``, ``run_id``, and ``prompt_id``.
    Defaults are applied for optional fields (priority, status, attempt_count,
    max_attempts) if absent.
    """
    out = dict(item)
    out.setdefault("priority", 100)
    out.setdefault("status", _STATUS_PENDING)
    out.setdefault("attempt_count", 0)
    out.setdefault("max_attempts", 5)
    out = _stamp(out)
    queue_id = str(out["queue_id"])
    _atomic_write(_pending_path(queue_id), out)
    return queue_id


def claim(runner_id: str) -> dict[str, Any] | None:
    """Claim the next eligible PENDING item for *runner_id*.

    Eligibility: status == PENDING, retry_after is null or in the past.
    Selection: lowest priority first, then oldest created_at.

    Race-safe protocol (rename-as-mutex):
      1. ``os.rename(src, lock)`` — src disappears; only one thread wins this.
         Losers get FileNotFoundError and try the next candidate.
      2. Winner reads from lock, stamps CLAIMED, writes back to lock atomically.
      3. ``os.rename(lock, src)`` — single atomic commit; src now has CLAIMED.

    This keeps src absent for the entire duration of steps 1-3, so any
    concurrent thread that scanned the PENDING file and tries to rename src
    will always get FileNotFoundError. There is no window where src exists
    with CLAIMED status while lock also exists.

    Returns the claimed item dict, or None if no eligible items.
    """
    pending = _pending_dir()
    now = _now_dt()

    # Collect candidates: PENDING files whose retry_after is not in the future.
    candidates: list[tuple[int, str, str]] = []  # (priority, created_at, queue_id)
    for p in pending.glob("*.json"):
        try:
            data = _read_json(p)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("status") != _STATUS_PENDING:
            continue
        retry_after = data.get("retry_after")
        if retry_after:
            try:
                if _parse_dt(retry_after) > now:
                    continue
            except ValueError:
                pass
        pri = int(data.get("priority", 100))
        created = str(data.get("created_at", ""))
        queue_id = str(data.get("queue_id", p.stem))
        candidates.append((pri, created, queue_id))

    # Sort: lowest priority first, then oldest created_at.
    candidates.sort(key=lambda x: (x[0], x[1]))

    for _, _, queue_id in candidates:
        src = _pending_path(queue_id)
        lock = _lock_path(queue_id, runner_id)
        try:
            # Step 1: atomic claim — src disappears, only one thread wins.
            os.rename(src, lock)
        except FileNotFoundError:
            # Another runner already claimed (or file disappeared). Try next.
            continue
        except OSError:
            continue

        # Step 2: re-verify precondition, mutate, write back to lock path.
        # Guard: a late-scheduled thread may win os.rename(src, its_lock) AFTER
        # a previous winner has already restored src with status=CLAIMED. Reading
        # status from the lock file and backing off if it is not PENDING makes
        # the claim monotonic: once any thread sets CLAIMED, every straggler
        # that grabs src sees non-PENDING, restores it, and returns None.
        try:
            data = _read_json(lock)
            if data.get("status") != _STATUS_PENDING:
                # Someone else already claimed and restored this item.
                with contextlib.suppress(OSError):
                    os.rename(lock, src)  # put it back for others
                continue
            now_str = _utc_now()
            data["status"] = _STATUS_CLAIMED
            data["claimed_by"] = runner_id
            data["claimed_at"] = now_str
            data["updated_at"] = now_str
            # Write to a temp file in the same dir, then replace the lock file.
            fd, tmp = tempfile.mkstemp(dir=pending, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                    f.write("\n")
                os.replace(tmp, lock)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
        except BaseException:
            # Restore the lock file to src so other runners can try.
            with contextlib.suppress(OSError):
                os.rename(lock, src)
            raise

        # Step 3: atomic commit — rename lock back to src; src now has CLAIMED.
        # src is absent from step 1 through here, so concurrent renames fail
        # with FileNotFoundError. The post-rename PENDING guard (step 2) catches
        # the rare straggler that wins a rename of an already-CLAIMED src.
        os.rename(lock, src)

        return data

    return None


def mark_in_progress(queue_id: str, runner_id: str) -> dict[str, Any]:
    """Transition a CLAIMED item to IN_PROGRESS. Returns the updated item."""
    path = _pending_path(queue_id)
    data = _read_json(path)
    if data.get("claimed_by") != runner_id:
        raise ValueError(
            f"queue item {queue_id!r} is not claimed by {runner_id!r} "
            f"(claimed_by={data.get('claimed_by')!r})"
        )
    now_str = _utc_now()
    data["status"] = _STATUS_IN_PROGRESS
    data["started_at"] = now_str
    data["updated_at"] = now_str
    _atomic_write(path, data)
    return data


def complete(queue_id: str) -> dict[str, Any]:
    """Mark a queue item COMPLETE and move it to the completed directory.

    Returns the final item dict.
    """
    src = _pending_path(queue_id)
    data = _read_json(src)
    now_str = _utc_now()
    data["status"] = _STATUS_COMPLETE
    data["completed_at"] = now_str
    data["updated_at"] = now_str
    dest = _completed_path(queue_id)
    _atomic_write(dest, data)
    with contextlib.suppress(FileNotFoundError):
        src.unlink()
    return data


def fail(queue_id: str, error: str) -> dict[str, Any]:
    """Record a failure for *queue_id*.

    If ``attempt_count < max_attempts``: requeue with incremented attempt_count,
    exponential backoff retry_after, and bumped priority (per #13 design).
    Otherwise: mark terminal FAILED and move to completed/.

    Returns the updated item dict.
    """
    src = _pending_path(queue_id)
    data = _read_json(src)
    now_str = _utc_now()
    attempt = int(data.get("attempt_count", 0))
    max_att = int(data.get("max_attempts", 5))
    data["error_detail"] = error

    if attempt < max_att:
        # Requeue with backoff.
        attempt += 1
        data["attempt_count"] = attempt
        data["status"] = _STATUS_PENDING
        data["claimed_by"] = None
        data["claimed_at"] = None
        data["started_at"] = None
        # retry_after = now + 5 * attempt_count minutes
        retry_dt = _now_dt() + timedelta(minutes=_RETRY_BACKOFF_MINUTES * attempt)
        data["retry_after"] = retry_dt.strftime(_ISO_FMT)
        # priority += 5 * attempt_count (lower = more important; bump = deprioritize)
        data["priority"] = int(data.get("priority", 100)) + _RETRY_PRIORITY_BUMP * attempt
        data["updated_at"] = now_str
        _atomic_write(src, data)
    else:
        # Terminal failure: move to completed/.
        data["status"] = _STATUS_FAILED
        data["completed_at"] = now_str
        data["updated_at"] = now_str
        dest = _completed_path(queue_id)
        _atomic_write(dest, data)
        with contextlib.suppress(FileNotFoundError):
            src.unlink()

    return data


def cancel(queue_id: str) -> dict[str, Any]:
    """Cancel a queue item (pending or in-progress) and move to completed/.

    Returns the updated item dict.
    """
    src = _pending_path(queue_id)
    data = _read_json(src)
    now_str = _utc_now()
    data["status"] = _STATUS_CANCELLED
    data["completed_at"] = now_str
    data["updated_at"] = now_str
    dest = _completed_path(queue_id)
    _atomic_write(dest, data)
    with contextlib.suppress(FileNotFoundError):
        src.unlink()
    return data


def reap_stale_claims(timeout_minutes: int = 10) -> list[str]:
    """Return stale CLAIMED/IN_PROGRESS items to PENDING.

    A claim is stale when claimed_at is more than *timeout_minutes* ago.
    Call this from the worker loop (probabilistic embedded reaper — no daemon
    required). Returns a list of queue_ids that were reaped.

    Per #13: simple function the worker calls, not a background thread.
    """
    now = _now_dt()
    timeout = timedelta(minutes=timeout_minutes)
    reaped: list[str] = []

    for path in _pending_dir().glob("*.json"):
        try:
            data = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        status = data.get("status")
        if status not in (_STATUS_CLAIMED, _STATUS_IN_PROGRESS):
            continue
        claimed_at_str = data.get("claimed_at")
        if not claimed_at_str:
            continue
        try:
            claimed_dt = _parse_dt(claimed_at_str)
        except ValueError:
            continue
        if now - claimed_dt < timeout:
            continue

        # Stale: return to PENDING.
        queue_id = str(data.get("queue_id", path.stem))
        now_str = _utc_now()
        data["status"] = _STATUS_PENDING
        data["claimed_by"] = None
        data["claimed_at"] = None
        data["started_at"] = None
        data["updated_at"] = now_str
        with contextlib.suppress(OSError):
            _atomic_write(path, data)
            reaped.append(queue_id)

    return reaped


def list_pending() -> list[dict[str, Any]]:
    """Return all items currently in the pending queue (any status)."""
    items = []
    for path in _pending_dir().glob("*.json"):
        with contextlib.suppress(OSError, json.JSONDecodeError):
            items.append(_read_json(path))
    return items


def list_completed() -> list[dict[str, Any]]:
    """Return all items in the completed directory."""
    items = []
    for path in _completed_dir().glob("*.json"):
        with contextlib.suppress(OSError, json.JSONDecodeError):
            items.append(_read_json(path))
    return items
