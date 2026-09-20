"""Distributed worker: pull jobs from bakeoff-results and run them locally.

The standalone ``bench.runner`` matrix loop is unchanged. This module is the
opt-in pull client for Rethunk-AI/bakeoff#37 — poll, claim, heartbeat, execute,
sign, submit. Empty queue sleeps rather than busy-looping.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from bench.signing import (
    encode_public_key,
    generate_keypair,
    load_private_key,
    save_private_key,
    sign_result,
)
from bench.store import data_dir, write_record

WaitFn = Callable[[float], None]


class WorkerError(RuntimeError):
    """Raised when the queue API rejects a worker request."""


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_or_create_key(path: Path):
    if path.is_file():
        return load_private_key(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    private_key, _ = generate_keypair()
    save_private_key(private_key, path)
    path.chmod(0o600)
    return private_key


class QueueClient:
    """HTTP client for the bakeoff-results queue API."""

    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.token: str | None = None
        self.runner_id: str | None = None
        self.heartbeat_interval_s = 30
        self.heartbeat_ttl_s = 120

    def _headers(self, authed: bool = True) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if authed and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _raise_for_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        try:
            payload = response.json()
            message = payload.get("error") if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            message = None
        raise WorkerError(message or f"HTTP {response.status_code}: {response.text}")

    def register(
        self,
        public_key: str,
        *,
        hostname: str,
        process_id: int,
        effective_user: str,
        capabilities: dict[str, Any],
    ) -> dict[str, Any]:
        response = httpx.post(
            self._url("/api/runners/register"),
            headers=self._headers(authed=False),
            json={
                "public_key": public_key,
                "hostname": hostname,
                "process_id": process_id,
                "effective_user": effective_user,
                "capabilities": capabilities,
            },
            timeout=self.timeout_s,
        )
        self._raise_for_error(response)
        payload = response.json()
        self.token = str(payload["token"])
        self.runner_id = str(payload["runner"]["runner_id"])
        self.heartbeat_interval_s = int(payload.get("heartbeat_interval_s", 30))
        self.heartbeat_ttl_s = int(payload.get("heartbeat_ttl_s", 120))
        return payload

    def claim(self, capabilities: dict[str, Any] | None = None) -> dict[str, Any] | None:
        response = httpx.post(
            self._url("/api/queue/claim"),
            headers=self._headers(),
            json={"capabilities": capabilities or {}},
            timeout=self.timeout_s,
        )
        if response.status_code == 204:
            return None
        self._raise_for_error(response)
        return response.json()

    def heartbeat(self, job_id: str) -> dict[str, Any]:
        response = httpx.post(
            self._url(f"/api/queue/{job_id}/heartbeat"),
            headers=self._headers(),
            timeout=self.timeout_s,
        )
        self._raise_for_error(response)
        return response.json()

    def submit(self, job_id: str, envelope: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(
            self._url(f"/api/queue/{job_id}/submit"),
            headers=self._headers(),
            json=envelope,
            timeout=self.timeout_s,
        )
        self._raise_for_error(response)
        return response.json()


def stub_result(job: dict[str, Any], runner_id: str) -> dict[str, Any]:
    model_id = str(job.get("model_id", "unknown"))
    run_id = str(job.get("run_id") or job.get("queue_id"))
    return {
        "run_id": run_id,
        "timestamp": _utc_now(),
        "provenance": {
            "source_repository": "Rethunk-AI/bakeoff",
            "source_commit": os.environ.get("BAKEOFF_SOURCE_COMMIT", "local-worker"),
            "runner_id": runner_id,
        },
        "models": [{"id": model_id}],
        "worker": {"execute": False, "queue_id": job.get("queue_id")},
    }


def load_latest_result(results_dir: Path) -> dict[str, Any] | None:
    files = sorted(results_dir.glob("run-*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    inner = data.get("result")
    if "sig" in data and isinstance(inner, dict):
        return inner
    return data


def execute_job(config: Path, model_id: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "bench.runner",
        "--config",
        str(config),
        "--models",
        model_id,
    ]
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise WorkerError(f"runner exited {completed.returncode} for model {model_id}")


def _heartbeat_loop(
    client: QueueClient,
    job_id: str,
    stop: threading.Event,
    interval_s: float,
) -> None:
    while not stop.wait(interval_s):
        try:
            client.heartbeat(job_id)
        except WorkerError as exc:
            print(f"[worker] heartbeat failed: {exc}", file=sys.stderr)
            return


def run_once(
    client: QueueClient,
    *,
    private_key,
    capabilities: dict[str, Any],
    execute: bool,
    config: Path,
) -> bool:
    """Claim at most one job. Returns True when a job was processed."""
    claimed = client.claim(capabilities)
    if claimed is None:
        return False
    job = claimed["job"]
    job_id = str(job.get("queue_id") or job.get("run_id"))
    model_id = str(job.get("model_id"))
    runner_id = client.runner_id or "unknown"
    interval = float(claimed.get("heartbeat_interval_s") or client.heartbeat_interval_s)
    print(f"[worker] claimed {job_id} model={model_id}", file=sys.stderr)

    stop = threading.Event()
    beat = threading.Thread(
        target=_heartbeat_loop,
        args=(client, job_id, stop, interval),
        daemon=True,
    )
    beat.start()
    try:
        client.heartbeat(job_id)
        if execute:
            execute_job(config, model_id)
            result = load_latest_result(Path("results")) or stub_result(job, runner_id)
            result["run_id"] = str(job.get("run_id") or job_id)
        else:
            result = stub_result(job, runner_id)
        envelope = sign_result(result, private_key, runner_id)
        client.submit(job_id, envelope)
        print(f"[worker] submitted {job_id}", file=sys.stderr)
    finally:
        stop.set()
    return True


def run_loop(
    client: QueueClient,
    *,
    private_key,
    capabilities: dict[str, Any],
    execute: bool,
    config: Path,
    poll_seconds: float,
    wait: WaitFn = time.sleep,
    once: bool = False,
) -> None:
    while True:
        processed = run_once(
            client,
            private_key=private_key,
            capabilities=capabilities,
            execute=execute,
            config=config,
        )
        if not processed:
            print(f"[worker] queue empty, sleeping {poll_seconds}s", file=sys.stderr)
            wait(poll_seconds)
        if once:
            return


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue-url", required=True, help="bakeoff-results queue base URL")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--key-file",
        type=Path,
        help="Ed25519 PEM (created if missing). Default: $BAKEOFF_DATA_DIR/runner.pem",
    )
    ap.add_argument("--hostname", default=socket.gethostname())
    ap.add_argument("--vram-mb", type=int)
    ap.add_argument("--quantization", action="append", default=[])
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument("--heartbeat-seconds", type=float, default=30.0)
    ap.add_argument(
        "--execute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run bench.runner for the claimed model (default: true).",
    )
    ap.add_argument(
        "--once",
        action="store_true",
        help="Claim at most one job (or sleep once if empty) and exit.",
    )
    args = ap.parse_args(argv)

    key_path = args.key_file or (data_dir() / "runner.pem")
    private_key = load_or_create_key(key_path)
    public_key = encode_public_key(private_key.public_key())
    capabilities: dict[str, Any] = {}
    if args.vram_mb is not None:
        capabilities["vram_mb"] = args.vram_mb
    if args.quantization:
        capabilities["quantization"] = args.quantization

    client = QueueClient(args.queue_url)
    client.heartbeat_interval_s = int(args.heartbeat_seconds)
    try:
        registered = client.register(
            public_key,
            hostname=args.hostname,
            process_id=os.getpid(),
            effective_user=getpass.getuser(),
            capabilities=capabilities,
        )
    except WorkerError as exc:
        print(f"[worker] register failed: {exc}", file=sys.stderr)
        return 1

    runner = registered["runner"]
    write_record(
        "runners",
        str(runner["runner_id"]),
        {
            "runner_id": runner["runner_id"],
            "public_key": public_key,
            "hostname": args.hostname,
            "status": "IDLE",
        },
    )
    print(f"[worker] registered {runner['runner_id']}", file=sys.stderr)

    try:
        run_loop(
            client,
            private_key=private_key,
            capabilities=capabilities,
            execute=args.execute,
            config=Path(args.config),
            poll_seconds=args.poll_seconds,
            once=args.once,
        )
    except WorkerError as exc:
        print(f"[worker] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
