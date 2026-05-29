"""Tests for bench.queue — disk-backed run queue.

Coverage strategy: one happy-path end-to-end (enqueue → claim → in_progress →
complete), then critical edges: fail retry with backoff, retry exhaustion
→ terminal FAILED, stale-claim reaping, and a genuine concurrent claim race.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from bench import queue


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Point BAKEOFF_DATA_DIR at a per-test temp directory."""
    monkeypatch.setenv("BAKEOFF_DATA_DIR", str(tmp_path))
    return tmp_path


def _item(extra: dict | None = None) -> dict:
    """Minimal valid queue item."""
    base = {
        "queue_id": str(uuid.uuid4()),
        "run_id": str(uuid.uuid4()),
        "prompt_id": "p1",
    }
    if extra:
        base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Happy path: enqueue → claim → mark_in_progress → complete
# ---------------------------------------------------------------------------


def test_enqueue_claim_complete_happy():
    item = _item()
    qid = queue.enqueue(item)
    assert qid == item["queue_id"]

    # File must be in pending.
    pending = queue.list_pending()
    assert len(pending) == 1
    assert pending[0]["status"] == "PENDING"

    # Claim.
    claimed = queue.claim("runner-1")
    assert claimed is not None
    assert claimed["queue_id"] == qid
    assert claimed["status"] == "CLAIMED"
    assert claimed["claimed_by"] == "runner-1"

    # mark_in_progress.
    in_prog = queue.mark_in_progress(qid, "runner-1")
    assert in_prog["status"] == "IN_PROGRESS"
    assert "started_at" in in_prog

    # complete → file moves to completed/.
    done = queue.complete(qid)
    assert done["status"] == "COMPLETE"
    assert "completed_at" in done

    # pending must be empty; completed must have one entry.
    assert queue.list_pending() == []
    completed = queue.list_completed()
    assert len(completed) == 1
    assert completed[0]["status"] == "COMPLETE"


def test_claim_returns_none_when_empty():
    assert queue.claim("runner-x") is None


def test_claim_priority_ordering():
    """Lower priority value is claimed first."""
    qid_lo = queue.enqueue(_item({"priority": 10}))
    qid_hi = queue.enqueue(_item({"priority": 200}))

    first = queue.claim("r1")
    assert first is not None
    assert first["queue_id"] == qid_lo

    second = queue.claim("r1")
    assert second is not None
    assert second["queue_id"] == qid_hi


# ---------------------------------------------------------------------------
# fail(): retry with backoff, then terminal FAILED
# ---------------------------------------------------------------------------


def test_fail_retry_increments_attempt_and_sets_backoff():
    item = _item({"max_attempts": 3})
    qid = queue.enqueue(item)
    queue.claim("r1")

    result = queue.fail(qid, "transient error")
    assert result["status"] == "PENDING"
    assert result["attempt_count"] == 1
    assert result["retry_after"] is not None
    # retry_after must be in the future.
    retry_dt = datetime.strptime(result["retry_after"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    assert retry_dt > datetime.now(timezone.utc)
    # priority bumped.
    assert result["priority"] > 100


def test_fail_exhaustion_moves_to_terminal_failed():
    """With max_attempts=1: first fail requeues (attempt 1/1); second fail is terminal."""
    import json as _json

    item = _item({"max_attempts": 1})
    qid = queue.enqueue(item)
    queue.claim("r1")
    queue.mark_in_progress(qid, "r1")

    # First failure: attempt_count goes 0→1, still < max? No: 0 < 1, so requeue.
    result1 = queue.fail(qid, "transient")
    assert result1["status"] == "PENDING"
    assert result1["attempt_count"] == 1

    # Reset retry_after so item is eligible again.
    path = queue._pending_path(qid)
    data = _json.loads(path.read_text())
    data["retry_after"] = None
    path.write_text(_json.dumps(data))

    # Second attempt: claim → fail; attempt_count=1, max_attempts=1 → 1 < 1 is False → terminal.
    queue.claim("r2")
    result2 = queue.fail(qid, "unrecoverable")
    assert result2["status"] == "FAILED"
    assert "completed_at" in result2

    # Must be in completed/, not pending.
    assert queue.list_pending() == []
    completed = queue.list_completed()
    assert any(c["queue_id"] == qid for c in completed)


def test_fail_multiple_retries_accumulate():
    """Second failure further increments attempt_count and bumps priority again."""
    item = _item({"max_attempts": 5, "priority": 100})
    qid = queue.enqueue(item)
    queue.claim("r1")
    result1 = queue.fail(qid, "err1")
    assert result1["attempt_count"] == 1

    # Reset retry_after so it can be claimed again.
    import json
    path = queue._pending_path(qid)
    data = json.loads(path.read_text())
    data["retry_after"] = None
    path.write_text(json.dumps(data))

    queue.claim("r2")
    result2 = queue.fail(qid, "err2")
    assert result2["attempt_count"] == 2
    assert result2["priority"] > result1["priority"]


# ---------------------------------------------------------------------------
# cancel()
# ---------------------------------------------------------------------------


def test_cancel_moves_to_completed():
    item = _item()
    qid = queue.enqueue(item)
    result = queue.cancel(qid)
    assert result["status"] == "CANCELLED"
    assert queue.list_pending() == []
    completed = queue.list_completed()
    assert any(c["queue_id"] == qid for c in completed)


# ---------------------------------------------------------------------------
# retry_after gate in claim()
# ---------------------------------------------------------------------------


def test_claim_respects_retry_after():
    """An item with retry_after in the future must not be claimed."""
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    item = _item({"retry_after": future})
    queue.enqueue(item)

    assert queue.claim("r1") is None


def test_claim_picks_up_past_retry_after():
    """An item with retry_after in the past is eligible."""
    past = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    item = _item({"retry_after": past})
    qid = queue.enqueue(item)

    claimed = queue.claim("r1")
    assert claimed is not None
    assert claimed["queue_id"] == qid


# ---------------------------------------------------------------------------
# Stale-claim reaping
# ---------------------------------------------------------------------------


def test_reap_stale_claims():
    item = _item()
    qid = queue.enqueue(item)
    queue.claim("r1")

    # Backdate claimed_at to make it stale.
    import json

    path = queue._pending_path(qid)
    data = json.loads(path.read_text())
    stale_time = (datetime.now(timezone.utc) - timedelta(minutes=20)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    data["claimed_at"] = stale_time
    path.write_text(json.dumps(data))

    reaped = queue.reap_stale_claims(timeout_minutes=10)
    assert qid in reaped

    # Item must now be PENDING again.
    fresh_data = json.loads(path.read_text())
    assert fresh_data["status"] == "PENDING"
    assert fresh_data["claimed_by"] is None


def test_reap_does_not_affect_fresh_claims():
    item = _item()
    qid = queue.enqueue(item)
    queue.claim("r1")

    reaped = queue.reap_stale_claims(timeout_minutes=10)
    assert qid not in reaped


# ---------------------------------------------------------------------------
# Concurrent claim race: exactly one winner
# ---------------------------------------------------------------------------


def test_concurrent_claim_exactly_one_winner():
    """16 threads racing to claim a single item — exactly one must win.

    This tests the rename-as-mutex + post-rename PENDING guard that prevents
    late-scheduled threads from re-claiming an already-CLAIMED record.
    Run this assertion many times within the test to surface intermittent races.
    """
    workers = 16
    rounds = 20

    for round_num in range(rounds):
        queue.enqueue(_item())

        def try_claim(runner_id: str) -> dict | None:
            return queue.claim(runner_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(try_claim, f"runner-{i}") for i in range(workers)]
            results = [f.result() for f in concurrent.futures.as_completed(futs)]

        winners = [r for r in results if r is not None]
        assert len(winners) == 1, (
            f"Round {round_num}: expected exactly 1 winner, got {len(winners)}"
        )
        # Clean up the claimed item so it doesn't pollute the next round.
        if winners:
            with contextlib.suppress(Exception):
                queue.complete(winners[0]["queue_id"])
