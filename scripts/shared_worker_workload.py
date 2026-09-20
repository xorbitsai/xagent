"""Real SDK workload used by shared_worker_benchmark; never run on production.

The caller supplies a dedicated configured account/model and PostgreSQL database.
We create and retain published agents, API keys, tasks and traces. Prompts are
arithmetic-only, but model instructions are NOT a tool sandbox: use an isolated
test deployment without external tool credentials. No fake model responses,
server delays, task retries, cancellation or data cleanup are injected.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from contextlib import closing
from datetime import timezone

import httpx

TERMINAL = frozenset({"completed", "failed", "paused", "waiting_for_user"})
PROMPT = (
    "Controlled runtime benchmark: use ONLY execute_python_code. Make exactly "
    "ten sequential tool calls, one call per assistant turn, waiting for each "
    "result. Begin with x=1. At each step call execute_python_code with code "
    "print(x*2+1), substituting x with the integer from the previous tool output. "
    "The first call is print(1*2+1), the second print(3*2+1). Do not combine "
    "steps, loop, parallelize calls, or compute later results yourself. Every "
    "call must contain only a print of this arithmetic expression. No imports, "
    "file access, network access, other tools, or questions. After the tenth "
    "result reply exactly BENCHMARK_DONE followed by the final integer. "
    "If execute_python_code is unavailable, say TOOL_UNAVAILABLE and stop."
)
QA = "Reply with exactly OK. Do not use tools."


def emit(value):
    print(json.dumps(value), flush=True)


def timestamp(value):
    # Database UTC timestamps can be naive. Never interpret them as local time.
    return (
        value.replace(tzinfo=timezone.utc).timestamp()
        if value.tzinfo is None
        else value.timestamp()
    )


def snapshot(ids):
    import psycopg2  # Optional PostgreSQL extra; pure guard tests do not need it.

    with closing(
        psycopg2.connect(
            os.environ["DATABASE_URL"],
            connect_timeout=5,
            options="-c default_transaction_read_only=on -c statement_timeout=5000",
        )
    ) as db:
        with db.cursor() as cursor:
            cursor.execute(
                "SELECT id,status::text,created_at,updated_at FROM tasks WHERE id=ANY(%s)",
                (ids,),
            )
            tasks = [
                {
                    "id": r[0],
                    "status": r[1].lower(),
                    "created": timestamp(r[2]),
                    "updated": timestamp(r[3]),
                }
                for r in cursor.fetchall()
            ]
            cursor.execute(
                "SELECT task_id,event_type,min(timestamp),count(*),"
                "sum(CASE WHEN data->>'success'='true' THEN 1 ELSE 0 END) "
                "FROM trace_events WHERE task_id=ANY(%s) GROUP BY task_id,event_type",
                (ids,),
            )
            events = [
                {
                    "id": r[0],
                    "type": r[1],
                    "first": timestamp(r[2]),
                    "count": r[3],
                    "success_count": r[4],
                }
                for r in cursor.fetchall()
            ]
    return {"tasks": tasks, "events": events}


def all_terminal(state, ids):
    tasks = state["tasks"]
    return (
        len(tasks) == len(ids)
        and {row["id"] for row in tasks} == set(ids)
        and all(row["status"] in TERMINAL for row in tasks)
    )


async def drain(rows, limit):
    ids = [r["id"] for r in rows if r.get("id") is not None]
    deadline = time.monotonic() + limit
    state = {"tasks": [], "events": []}
    while time.monotonic() < deadline:
        state = await asyncio.to_thread(snapshot, ids)
        if all_terminal(state, ids):
            return state
        await asyncio.sleep(1)
    return {"timeout": True, **state}


async def submit_background(submit, config, prompt, count, spacing=0):
    """Pacing is warmup-only; zero keeps simultaneous client submission."""
    pending = []
    try:
        for index in range(count):
            if index and spacing:
                await asyncio.sleep(spacing)
            pending.append(asyncio.create_task(submit(config, prompt)))
        return await asyncio.gather(*pending)
    finally:
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


async def submit(client, config, prompt):
    sent = time.time()
    started = time.monotonic()
    row = {"id": None, "sent": sent, "http": None}
    try:
        response = await client.post(
            "/v1/chat/tasks",
            headers=config[1],
            json={"agent_id": config[0], "message": {"content": prompt}},
        )
        row["http"] = response.status_code
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        task_id = payload.get("task_id") if isinstance(payload, dict) else None
        if isinstance(task_id, int) and not isinstance(task_id, bool) and task_id > 0:
            row["id"] = task_id
        if row["http"] != 202 or row["id"] is None:
            row["error"] = "rejected_or_missing_task_id"
    except httpx.HTTPError as exc:
        # Do not log response bodies, tokens, URLs or exception messages.
        row["error"] = type(exc).__name__
    row["api_ms"] = (time.monotonic() - started) * 1000
    row["unknown_creation"] = row["id"] is None
    return row


async def agent(client, headers, model_id, kind):
    response = await client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "worker-benchmark-" + kind + "-" + uuid.uuid4().hex[:12],
            "instructions": "Follow the task exactly. No network, files, imports or side effects.",
            "execution_mode": "flash" if kind == "probe" else "balanced",
            "models": {
                slot: model_id
                for slot in ("general", "small_fast", "visual", "compact")
            },
            "tool_categories": [] if kind == "probe" else ["basic"],
            "skills": [],
            "knowledge_bases": [],
        },
    )
    response.raise_for_status()
    aid = response.json()["id"]
    response = await client.post(f"/api/agents/{aid}/publish", headers=headers)
    response.raise_for_status()
    response = await client.post(f"/api/agents/{aid}/api-key", headers=headers)
    response.raise_for_status()
    return aid, {"Authorization": "Bearer " + response.json()["full_key"]}


async def continuous(client, config, background, count, interval):
    """Keep submission ordinals even when completions arrive out of order."""
    pending = []
    try:
        for ordinal in range(count):
            if ordinal:
                await asyncio.sleep(interval)
            state = await asyncio.to_thread(
                snapshot, [r["id"] for r in background if r.get("id")]
            )
            # RUNNING is admission state, not proof of concurrent execution.
            active = sum(r["status"] == "running" for r in state["tasks"])

            async def one(index=ordinal, running=active):
                row = await submit(client, config, QA)
                row.update(ordinal=index, active_background=running)
                emit({"type": "probe_submission", **row})
                return row

            pending.append(asyncio.create_task(one()))
        return await asyncio.gather(*pending)
    finally:
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


async def run(args):
    # Separate connection capacity prevents the background HTTP burst from
    # consuming the continuous probe client's pool.
    async with (
        httpx.AsyncClient(
            base_url=args.base_url,
            timeout=60,
            limits=httpx.Limits(max_connections=max(120, args.background)),
        ) as client,
        httpx.AsyncClient(
            base_url=args.base_url,
            timeout=60,
            limits=httpx.Limits(max_connections=max(1, args.probes)),
        ) as probe_client,
    ):
        response = await client.post(
            "/api/auth/login",
            json={
                "username": os.environ["XAGENT_BENCH_USERNAME"],
                "password": os.environ["XAGENT_BENCH_PASSWORD"],
            },
        )
        response.raise_for_status()
        headers = {"Authorization": "Bearer " + response.json()["access_token"]}
        probe = await agent(client, headers, args.model_id, "probe")
        worker = await agent(client, headers, args.model_id, "worker")
        emit(
            {
                "type": "setup",
                "agents": [probe[0], worker[0]],
                "model_id": args.model_id,
            }
        )
        warm = await submit(client, probe, QA)
        warm_state = await drain([warm], args.task_timeout)
        emit({"type": "warm", "rows": [warm], **warm_state})
        if (
            warm["http"] != 202
            or not warm_state["tasks"]
            or any(r["status"] != "completed" for r in warm_state["tasks"])
        ):
            raise RuntimeError("Connectivity task did not complete")

        async def submit_one(config, prompt):
            row = await submit(client, config, prompt)
            emit({"type": "submission", **row})
            return row

        background = await submit_background(
            submit_one, worker, PROMPT, args.background, args.spacing
        )
        await asyncio.sleep(3)
        probes = await asyncio.gather(*(submit_one(probe, QA) for _ in range(10)))
        emit(
            {
                "type": "admission",
                "background_n": args.background,
                "background": background,
                "probes": probes,
            }
        )
        # Drain and continuous arrival generation run concurrently. A failed
        # request is recorded once, never silently retried.
        task_state, continuous_rows = await asyncio.gather(
            drain(background + probes, args.task_timeout),
            continuous(
                probe_client, probe, background, args.probes, args.probe_interval
            ),
        )
        emit({"type": "completed", "background_n": args.background, **task_state})
        probe_state = await drain(continuous_rows, args.task_timeout)
        emit({"type": "probes", "rows": continuous_rows, **probe_state})
        all_rows = [warm] + background + probes + continuous_rows
        if (
            task_state.get("timeout")
            or probe_state.get("timeout")
            or any(r.get("http") != 202 or not r.get("id") for r in all_rows)
            or any(
                r["status"] != "completed"
                for r in task_state["tasks"] + probe_state["tasks"]
            )
        ):
            raise RuntimeError("Failed or unsettled task")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-id", type=int, required=True)
    parser.add_argument("--background", type=int, required=True)
    parser.add_argument("--probes", type=int, default=0)
    parser.add_argument("--probe-interval", type=float, default=2)
    parser.add_argument("--task-timeout", type=float, default=600)
    parser.add_argument("--spacing", type=float, default=0)
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except Exception as exc:
        emit({"type": "safety_stop", "reason": type(exc).__name__})
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
