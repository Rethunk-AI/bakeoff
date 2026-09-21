"""Distributed worker pull client against a real HTTP queue fake."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from bench.signing import generate_keypair, verify_result
from bench.worker import QueueClient, WorkerError, load_latest_result, run_loop, stub_result


class QueueFake:
    def __init__(self) -> None:
        self.jobs: list[dict[str, Any]] = []
        self.submitted: list[dict[str, Any]] = []
        self.failed: list[dict[str, Any]] = []
        self.heartbeats = 0
        self.registered: dict[str, Any] | None = None
        self.token = "runner-token"
        self.paused = False

    def add_job(self, model_id: str = "qwen3.5-9b") -> None:
        self.jobs.append(
            {
                "queue_id": "job-1",
                "run_id": "job-1",
                "model_id": model_id,
                "status": "PENDING",
                "priority": 100,
            }
        )


def _make_handler(fake: QueueFake) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if not length:
                return {}
            payload = json.loads(self.rfile.read(length))
            return payload if isinstance(payload, dict) else {}

        def _send(self, status: int, payload: dict[str, Any] | None = None) -> None:
            body = b"" if payload is None else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/api/runners/register":
                body = self._read_json()
                fake.registered = body
                self._send(
                    200,
                    {
                        "runner": {
                            "runner_id": "runner-test",
                            "public_key": body.get("public_key"),
                        },
                        "token": fake.token,
                        "expires_at": 0,
                        "heartbeat_interval_s": 30,
                        "heartbeat_ttl_s": 120,
                    },
                )
                return
            if path == "/api/queue/claim":
                if fake.paused:
                    self._send(403, {"error": "runner is paused"})
                    return
                if not fake.jobs:
                    self._send(204)
                    return
                job = fake.jobs.pop(0)
                job["status"] = "CLAIMED"
                self._send(200, {"job": job, "heartbeat_interval_s": 30})
                return
            if path.endswith("/heartbeat"):
                fake.heartbeats += 1
                self._send(200, {"job": {"status": "IN_PROGRESS"}})
                return
            if path.endswith("/submit"):
                fake.submitted.append(self._read_json())
                self._send(200, {"job": {"status": "COMPLETE"}})
                return
            if path.endswith("/fail"):
                fake.failed.append(self._read_json())
                self._send(200, {"job": {"status": "PENDING"}})
                return
            self._send(404, {"error": "not found"})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


@pytest.fixture
def queue_http():
    fake = QueueFake()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(fake))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_port}"
    try:
        yield fake, url
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_empty_queue_sleeps_once(queue_http, tmp_path):
    fake, url = queue_http
    private_key, public_key = generate_keypair()
    client = QueueClient(url)
    client.register(
        public_key,
        hostname="test",
        process_id=1,
        effective_user="tester",
        capabilities={"vram_mb": 8192},
    )
    slept: list[float] = []
    run_loop(
        client,
        private_key=private_key,
        capabilities={"vram_mb": 8192},
        execute=False,
        config=tmp_path / "config.yaml",
        poll_seconds=7,
        wait=slept.append,
        once=True,
    )
    assert slept == [7]
    assert fake.submitted == []


def test_claim_and_submit_signed_stub(queue_http, tmp_path):
    fake, url = queue_http
    fake.add_job("qwen3.5-9b")
    private_key, public_key = generate_keypair()
    client = QueueClient(url)
    client.register(
        public_key,
        hostname="test",
        process_id=1,
        effective_user="tester",
        capabilities={"vram_mb": 32768},
    )
    run_loop(
        client,
        private_key=private_key,
        capabilities={"vram_mb": 32768},
        execute=False,
        config=tmp_path / "config.yaml",
        poll_seconds=1,
        wait=lambda _: None,
        once=True,
    )
    assert len(fake.submitted) == 1
    envelope = fake.submitted[0]
    result = verify_result(envelope, public_key)
    assert result["models"][0]["id"] == "qwen3.5-9b"
    assert result["run_id"] == "job-1"
    assert fake.heartbeats >= 1


def test_stub_result_has_required_fields():
    job = {"queue_id": "q1", "run_id": "r1", "model_id": "m"}
    result = stub_result(job, "runner-x")
    assert result["run_id"] == "r1"
    assert result["provenance"]["runner_id"] == "runner-x"
    assert result["models"][0]["id"] == "m"


def test_paused_claim_sleeps_once(queue_http, tmp_path):
    fake, url = queue_http
    fake.paused = True
    fake.add_job("qwen3.5-9b")
    private_key, public_key = generate_keypair()
    client = QueueClient(url)
    client.register(
        public_key,
        hostname="test",
        process_id=1,
        effective_user="tester",
        capabilities={},
    )
    slept: list[float] = []
    run_loop(
        client,
        private_key=private_key,
        capabilities={},
        execute=False,
        config=tmp_path / "config.yaml",
        poll_seconds=7,
        wait=slept.append,
        once=True,
    )
    assert slept == [7]
    assert fake.submitted == []
    assert fake.failed == []
    assert fake.jobs[0]["model_id"] == "qwen3.5-9b"


def test_execute_failure_reports_fail(queue_http, tmp_path, monkeypatch):
    fake, url = queue_http
    fake.add_job("boom")

    def boom(_config, model_id: str) -> None:
        raise WorkerError(f"runner exited 1 for model {model_id}")

    monkeypatch.setattr("bench.worker.execute_job", boom)
    private_key, public_key = generate_keypair()
    client = QueueClient(url)
    client.register(
        public_key,
        hostname="test",
        process_id=1,
        effective_user="tester",
        capabilities={},
    )
    run_loop(
        client,
        private_key=private_key,
        capabilities={},
        execute=True,
        config=tmp_path / "config.yaml",
        poll_seconds=1,
        wait=lambda _: None,
        once=True,
        results_dir=tmp_path,
    )
    assert fake.submitted == []
    assert fake.failed == [{"error": "runner exited 1 for model boom"}]


def test_load_latest_result_skips_other_models(tmp_path):
    older = tmp_path / "run-old.json"
    newer = tmp_path / "run-new.json"
    older.write_text(
        json.dumps({"run_id": "old", "config": {"models": [{"id": "other"}]}}),
        encoding="utf-8",
    )
    newer.write_text(
        json.dumps(
            {
                "run_id": "new",
                "model_metadata": [{"id": "qwen3.5-9b"}],
            }
        ),
        encoding="utf-8",
    )
    result = load_latest_result(tmp_path, model_id="qwen3.5-9b")
    assert result is not None
    assert result["run_id"] == "new"
    assert load_latest_result(tmp_path, model_id="missing") is None
