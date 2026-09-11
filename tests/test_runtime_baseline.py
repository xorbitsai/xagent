from __future__ import annotations

import asyncio
import json
from fractions import Fraction

import aiohttp
import pytest
from aiohttp import web

from scripts.runtime_baseline import (
    MODEL_FIELDS,
    Runner,
    arrival_offsets,
    parse_args,
    pinned_models,
    run,
    schedule,
    summarize,
)

PINNED = {"llm_ids": ["test"] * 4}


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
    now = 0
    release = asyncio.Event()

    async def sleep(delay):
        nonlocal now
        now += delay
        if now >= 0.04:
            release.set()
        # Advance only the synthetic clock, not wall time.
        await asyncio.sleep(0)

    async def slow():
        await release.wait()
        return {"outcome": "success"}

    await schedule(
        100, 0.05, 1, "login", slow, records, clock=lambda: now, sleeper=sleep
    )
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
        return web.json_response(state.get("health", {"status": "ok"}))

    async def test_models(request):
        assert (await request.json())["model_ids"] == ["test"]
        return web.json_response(
            state.get("model_test", [{"model_id": "test", "status": "passed"}])
        )

    async def create(request):
        assert request.headers["Authorization"] == "Bearer secret-test-token"
        state["created"] += 1
        if state.get("create_error"):
            # Model a persisted creation whose post-commit hook/cleanup failed:
            # the client receives no task ID and must not infer rollback.
            return web.Response(status=state["create_error"])
        return web.json_response(
            {
                "task_id": state["created"],
                **dict(zip(MODEL_FIELDS, state.get("models", ["test"] * 4))),
            }
        )

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
    app.router.add_post("/api/models/test", test_models)
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
@pytest.mark.parametrize("probe_rate", ["20", "0.01"])
async def test_complete_runner_and_secret_free_report(
    target, tmp_path, monkeypatch, probe_rate
):
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
            probe_rate,
        ]
    )
    # Sparse probes must not sleep until their 100-second next tick, including
    # idle stages. This is a generous deadlock bound, not a latency assertion.
    report = await asyncio.wait_for(run(args), 5)
    assert report["status"] == "completed"
    assert state["created"] == 1
    assert state["commands"] == [{"type": "execute_task"}]
    assert state["logins"] >= 3
    assert report["stages"][1]["summary"]["task"]["outcomes"] == {"success": 1}
    assert report["expected_model_ids"] == ["test"] * 4
    task_record = next(
        r for r in report["stages"][1]["records"] if r["operation"] == "task"
    )
    assert task_record["persisted_model_ids"] == ["test"] * 4
    assert report["model_preflight"] == {
        "outcome": "success",
        "tested_model_ids": ["test"],
    }
    stage = report["stages"][1]
    assert stage["successful_tasks_per_second"] == pytest.approx(
        1 / (stage["ended_at"] - stage["started_at"])
    )
    text = (args.output / "report.json").read_text()
    assert "secret-test-password" not in text
    assert "secret-test-token" not in text
    assert (args.output / "report.md").exists()


@pytest.mark.asyncio
async def test_task_error_retains_uncertain_id(target):
    base, state = target
    state["fail"] = True
    runner = Runner(base, "test", "secret-test-password", PINNED, 1)
    async with aiohttp.ClientSession() as session:
        await runner.login(session)
        result = await runner.task(session)
    assert result["outcome"] == "task_error"
    assert runner.uncertain == {1}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 403, 500])
async def test_creation_error_is_not_assumed_side_effect_free(target, status):
    base, state = target
    state["create_error"] = status
    runner = Runner(base, "test", "secret-test-password", PINNED, 1)
    async with aiohttp.ClientSession() as session:
        await runner.login(session)
        result = await runner.task(session)
    assert result["outcome"] == "create_http_error"
    assert runner.unknown_creations == 1
    assert not runner.task_ids
    assert state["created"] == 1
    assert state["commands"] == []
    assert runner.stop_new_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["task", "model", 400, 403])
async def test_failed_stage_aborts_before_next_cap(
    target, tmp_path, monkeypatch, failure
):
    base, state = target
    if failure == "task":
        state["fail"] = True
    elif failure == "model":
        state["models"] = ["fallback"] * 4
    else:
        state["create_error"] = failure
    monkeypatch.setenv("XAGENT_BENCH_USERNAME", "test")
    monkeypatch.setenv("XAGENT_BENCH_PASSWORD", "secret-test-password")
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps(dict(title="test", description="fixed", **PINNED)))
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
    if isinstance(failure, int):
        assert report["unknown_creations"] == 1
        assert report["uncertain_task_ids"] == []
        assert state["commands"] == []
    else:
        assert report["uncertain_task_ids"] == [1]
    if failure == "model":
        assert state["commands"] == []
        assert report["stages"][0]["summary"]["task"]["measured"] == 0


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--duration", "nan"),
        ("--duration", "sNaN"),
        ("--duration", "not-a-number"),
        ("--probe-rate", "1e-9999"),
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


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"llm_ids": ["test"]},
        {"llm_ids": ["test", "test", "", "test"]},
        {"llm_ids": ["test", "test", None, "test"]},
        {"llm_ids": ["test", "test", " test", "test"]},
        {"agent_id": 1},
        {"agent_id": 1, **PINNED},
    ],
)
def test_unpinned_models_rejected(payload):
    with pytest.raises(ValueError):
        pinned_models(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "models", [["default"] * 4, ["test"] * 3, ["test", None, "test", "test"]]
)
async def test_model_fallback_stops_before_execution(target, models):
    base, state = target
    state["models"] = models
    runner = Runner(base, "test", "secret-test-password", PINNED, 1)
    async with aiohttp.ClientSession() as session:
        await runner.login(session)
        result = await runner.task(session)
    assert result["outcome"] == "model_mismatch"
    assert runner.stop_new_tasks
    assert runner.task_ids == [1]
    assert runner.uncertain == {1}
    assert state["commands"] == []
    assert (
        summarize([dict(result, operation="task", scheduler_lag_ms=0)])["task"][
            "measured"
        ]
        == 0
    )


@pytest.mark.asyncio
async def test_health_degradation_is_not_success(target):
    base, state = target
    state["health"] = {"status": "ok", "degradations": ["runtime_unavailable"]}
    runner = Runner(base, "test", "secret-test-password", PINNED, 1)
    async with aiohttp.ClientSession() as session:
        result = await runner.health(session)
    assert result == {
        "outcome": "degraded",
        "status": 200,
        "degradations": ["runtime_unavailable"],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"status": "bad"},
        [],
        {"status": "ok", "degradations": "bad"},
        {"status": "ok", "padding": "x" * 65536},
    ],
)
async def test_invalid_health_payload_is_not_success(target, body):
    base, state = target
    state["health"] = body
    async with aiohttp.ClientSession() as session:
        result = await Runner(base, "test", "secret-test-password", PINNED, 1).health(
            session
        )
    assert result["outcome"] == "invalid_health_response"


@pytest.mark.parametrize(
    "origin,expected",
    [
        ("HTTPS://example.test:443", "wss://example.test:443"),
        ("hTtPs://example.test", "wss://example.test"),
        ("http://example.test", "ws://example.test"),
    ],
)
def test_websocket_origin_preserves_transport(origin, expected):
    assert Runner(origin, "test", "password", PINNED, 1).ws_url == expected


def test_decimal_arrivals_exclude_duration_boundary():
    args = parse_args(
        [
            "--base-url",
            "http://localhost",
            "--task-payload",
            "unused",
            "--output",
            "unused",
            "--confirm-test-environment",
            "--tasks-per-minute",
            "4.2",
            "--duration",
            "100",
        ]
    )
    assert (
        len(list(arrival_offsets(Fraction(args.tasks_per_minute) / 60, args.duration)))
        == 7
    )
    assert len(list(arrival_offsets(0.07, 100))) == 7


@pytest.mark.asyncio
async def test_sparse_probe_stop_wakes_and_drains_admitted_work():
    stop = asyncio.Event()
    started = asyncio.Event()
    release = asyncio.Event()
    records = []

    async def invoke():
        started.set()
        await release.wait()
        return {"outcome": "success"}

    task = asyncio.create_task(
        schedule(0.01, 200, 1, "health", invoke, records, stop_event=stop)
    )
    await asyncio.wait_for(started.wait(), 2)
    stop.set()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    await asyncio.wait_for(task, 2)
    assert len(records) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["health", "unsupported_model", "missing_model", "wrong_model"]
)
async def test_degraded_preflight_is_saved_without_creating_tasks(
    target, tmp_path, monkeypatch, failure
):
    base, state = target
    if failure == "health":
        state["health"] = {"status": "ok", "degradations": ["runtime_unavailable"]}
    else:
        state["model_test"] = {
            "unsupported_model": [
                {
                    "model_id": "test",
                    "status": "failed",
                    "error": "secret-provider-error",
                }
            ],
            "missing_model": [],
            "wrong_model": [{"model_id": "fallback", "status": "passed"}],
        }[failure]
    monkeypatch.setenv("XAGENT_BENCH_USERNAME", "test")
    monkeypatch.setenv("XAGENT_BENCH_PASSWORD", "secret-test-password")
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps(dict(title="test", description="fixed", **PINNED)))
    output = tmp_path / "result"
    args = parse_args(
        [
            "--base-url",
            base,
            "--task-payload",
            str(payload),
            "--output",
            str(output),
            "--confirm-test-environment",
        ]
    )
    with pytest.raises(ValueError, match="preflight failed"):
        await run(args)
    report = json.loads((output / "report.json").read_text())
    if failure == "health":
        assert report["health_preflight"]["outcome"] == "degraded"
        assert report["health_preflight"]["degradations"] == ["runtime_unavailable"]
    else:
        assert report["model_preflight"]["outcome"] != "success"
        assert "secret-provider-error" not in json.dumps(report)
    assert state["created"] == 0


@pytest.mark.asyncio
async def test_stalled_generator_sheds_without_catchup_burst():
    records = []
    now = 0

    async def stalled_sleep(delay):
        nonlocal now
        now += delay + 1
        await asyncio.sleep(0)

    async def never_called():
        pytest.fail("late arrivals must be shed")

    await schedule(
        100,
        0.05,
        1,
        "task",
        never_called,
        records,
        clock=lambda: now,
        sleeper=stalled_sleep,
    )
    assert len(records) == 5
    assert all(r["outcome"] == "shed" for r in records)
