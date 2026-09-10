from __future__ import annotations

import asyncio
import json

import aiohttp
import pytest
from aiohttp import web

from scripts.runtime_baseline import Runner, parse_args, run, schedule, summarize


def test_summary_counts_errors_and_shedding():
    records = [
        dict(operation="login", outcome=outcome, duration_ms=value, scheduler_lag_ms=0)
        for outcome, value in [("success", 1), ("error", 100), ("shed", 0)]
    ]
    summary = summarize(records)["login"]
    assert summary["measured"] == 2
    assert summary["scheduled"] == 3
    assert summary["p99_ms"] == 100
    assert summarize([]) == {}


@pytest.mark.asyncio
async def test_fixed_arrivals_shed_instead_of_queueing():
    records = []

    async def slow():
        await asyncio.sleep(0.12)
        return {"outcome": "success"}

    await schedule(100, 0.05, 1, "login", slow, records)
    assert len(records) == 5
    assert sum(r["outcome"] == "success" for r in records) == 1
    assert sum(r["outcome"] == "shed" for r in records) == 4


@pytest.fixture
async def target():
    app = web.Application()
    state = {"created": 0, "logins": 0, "commands": [], "fail": False}

    async def login(request):
        assert (await request.json())["password"] == "secret-test-password"
        state["logins"] += 1
        return web.json_response({"success": True, "access_token": "secret-test-token"})

    async def health(request):
        return web.json_response({"status": "ok"})

    async def create(request):
        assert request.headers["Authorization"] == "Bearer secret-test-token"
        state["created"] += 1
        if state.get("create_error"):
            return web.Response(status=500)
        return web.json_response({"task_id": state["created"]})

    async def status(request):
        return web.json_response({"status": "completed"})

    async def socket(request):
        assert request.query["token"] == "secret-test-token"
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        state["commands"].append(await ws.receive_json())
        await ws.send_json({"type": "error" if state["fail"] else "task_completed"})
        await ws.close()
        return ws

    app.router.add_post("/api/auth/login", login)
    app.router.add_get("/health", health)
    app.router.add_post("/api/chat/task/create", create)
    app.router.add_get("/api/chat/task/{id}/status", status)
    app.router.add_get("/ws/chat/{id}", socket)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        yield f"http://127.0.0.1:{port}", state
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_complete_runner_and_secret_free_report(target, tmp_path, monkeypatch):
    base, state = target
    monkeypatch.setenv("XAGENT_BENCH_USERNAME", "test")
    monkeypatch.setenv("XAGENT_BENCH_PASSWORD", "secret-test-password")
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps(dict(title="baseline", description="fixed", llm_ids=["test"] * 4))
    )
    args = parse_args(
        [
            "--base-url",
            base,
            "--task-payload",
            str(payload),
            "--output",
            str(tmp_path / "result"),
            "--confirm-test-environment",
            "--stages",
            "0,2",
            "--duration",
            "0.1",
            "--tasks-per-minute",
            "600",
            "--probe-rate",
            "20",
        ]
    )
    report = await run(args)
    assert report["status"] == "completed"
    assert state["created"] == 1
    assert state["commands"] == [{"type": "execute_task"}]
    assert state["logins"] >= 3
    assert report["stages"][1]["summary"]["task"]["outcomes"] == {"success": 1}
    text = (args.output / "report.json").read_text()
    assert "secret-test-password" not in text
    assert "secret-test-token" not in text
    assert (args.output / "report.md").exists()


@pytest.mark.asyncio
async def test_task_error_retains_uncertain_id(target):
    base, state = target
    state["fail"] = True
    runner = Runner(base, "test", "secret-test-password", {}, 1)
    async with aiohttp.ClientSession() as session:
        await runner.login(session)
        result = await runner.task(session)
    assert result["outcome"] == "task_error"
    assert runner.uncertain == {1}


@pytest.mark.asyncio
async def test_creation_error_is_not_assumed_side_effect_free(target):
    base, state = target
    state["create_error"] = True
    runner = Runner(base, "test", "secret-test-password", {}, 1)
    async with aiohttp.ClientSession() as session:
        await runner.login(session)
        result = await runner.task(session)
    assert result["outcome"] == "create_http_error"
    assert runner.unknown_creations == 1
    assert not runner.task_ids


@pytest.mark.asyncio
async def test_failed_stage_aborts_before_next_cap(target, tmp_path, monkeypatch):
    base, state = target
    state["fail"] = True
    monkeypatch.setenv("XAGENT_BENCH_USERNAME", "test")
    monkeypatch.setenv("XAGENT_BENCH_PASSWORD", "secret-test-password")
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps(dict(title="test", description="fixed", agent_id=1)))
    args = parse_args(
        [
            "--base-url",
            base,
            "--task-payload",
            str(payload),
            "--output",
            str(tmp_path / "result"),
            "--confirm-test-environment",
            "--stages",
            "1,2",
            "--duration",
            "0.1",
            "--probe-rate",
            "20",
        ]
    )
    report = await run(args)
    assert report["status"] == "aborted_unconfirmed_tasks"
    assert len(report["stages"]) == 1
    assert report["uncertain_task_ids"] == [1]


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--duration", "nan"),
        ("--probe-rate", "0"),
        ("--stages", "-1"),
        ("--base-url", "http://user:password@localhost"),
    ],
)
def test_invalid_parameters(flag, value, tmp_path):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--base-url",
                "http://localhost",
                "--task-payload",
                "unused",
                "--output",
                str(tmp_path),
                "--confirm-test-environment",
                flag,
                value,
            ]
        )
