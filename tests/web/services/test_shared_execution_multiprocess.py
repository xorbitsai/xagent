"""Production bridge, socket writer, START and scheduler across four processes.

Only model construction/execution is stubbed. Configure XAGENT_TEST_REDIS_URL
with a disposable Redis instance; this test uses a unique channel namespace.
"""

import asyncio
import json
import multiprocessing
import os
from types import SimpleNamespace
from uuid import uuid4

import pytest


def _web_host(pipe, env, task_id, command_id):
    os.environ.update(env)

    async def run():
        from websockets.asyncio.server import serve

        from xagent.web.api.websocket import ConnectionManager
        from xagent.web.models.database import configure_db
        from xagent.web.services.task_event_bridge import (
            start_task_event_bridge,
            stop_task_event_bridge,
        )
        from xagent.web.services.task_stream_snapshot import load_task_stream_snapshots

        configure_db()
        manager = ConnectionManager()
        bridge = await start_task_event_bridge(deliver=manager.deliver_shared_event)

        async def handler(connection):
            class Socket:
                async def send_text(self, text):
                    await connection.send(text)

                async def close(self, code=1000):
                    await connection.close(code=code)

            socket = Socket()
            # Authentication is outside ConnectionManager. These test sockets
            # represent the same already-authorized task audience on two hosts.
            manager.register_connection(socket, task_id)
            token = bridge.register_origin(
                task_id,
                command_id,
                lambda message: manager.send_personal_message(message, socket),
                recipient=socket,
            )
            await connection.send(json.dumps({"host": bridge.host_id, "origin": token}))
            try:
                async for message in connection:
                    if message == "snapshot":
                        snapshots = await asyncio.to_thread(
                            load_task_stream_snapshots, [task_id]
                        )
                        await manager.send_personal_message(snapshots[0], socket)
            finally:
                manager.disconnect(socket)
                bridge.discard_recipient(socket)

        try:
            async with serve(handler, "127.0.0.1", 0) as server:
                pipe.send({"port": server.sockets[0].getsockname()[1]})
                await asyncio.to_thread(pipe.recv)
        finally:
            await stop_task_event_bridge()

    asyncio.run(run())


def _execution_host(pipe, env, task_id):
    os.environ.update(env)

    async def run():
        from xagent.core.agent.trace import (
            TraceAction,
            TraceCategory,
            TraceEvent,
            TraceEventType,
            TraceScope,
        )
        from xagent.web.models.database import configure_db, get_session_local
        from xagent.web.models.task import Task, TaskStatus
        from xagent.web.services import agent_service_manager, task_orchestrator
        from xagent.web.services.task_command_transport import claim_task_command
        from xagent.web.services.task_event_bridge import (
            start_task_event_bridge,
            stop_task_event_bridge,
        )
        from xagent.web.services.task_execution import background_task_manager
        from xagent.web.services.task_execution_context_service import (
            TaskExecutionRecoverySnapshot,
        )
        from xagent.web.services.task_lease_service import get_runner_id
        from xagent.web.services.task_start_consumer import execute_task_start

        configure_db()
        bridge = await start_task_event_bridge()
        with get_session_local()() as db:
            owner_id = db.get(Task, task_id).user_id
        snapshot = SimpleNamespace(
            task=SimpleNamespace(user_id=owner_id, status=TaskStatus.RUNNING),
            runtime_user=object(),
            conversation_history=(),
            conversation_watermark=None,
            execution_recovery=TaskExecutionRecoverySnapshot(),
        )
        task_orchestrator.load_task_setup_snapshot_sync = (
            lambda *args, **kwargs: snapshot
        )
        handlers = []
        tracer = SimpleNamespace(
            add_handler=handlers.append, remove_handler=handlers.remove
        )
        service = SimpleNamespace(
            tracer=tracer,
            workspace=None,
            set_conversation_history=lambda *args, **kwargs: None,
            set_execution_context_messages=lambda *args: None,
            set_recovered_skill_context=lambda *args: None,
        )

        class ModelBoundary:
            async def get_agent_for_task(self, *args, **kwargs):
                return service

            async def execute_task(self, **kwargs):
                for text in ("hello ", "world"):
                    await bridge.publish({"type": "delta", "text": text}, task_id)
                event = TraceEvent(
                    TraceEventType(
                        TraceScope.TASK, TraceAction.START, TraceCategory.GENERAL
                    ),
                    task_id=str(task_id),
                )
                for handler in tuple(handlers):
                    await handler.handle_event(event)
                return {"success": True, "status": "completed", "output": "hello world"}

        agent_service_manager.get_agent_manager = lambda: ModelBoundary()
        background_task_manager.start_accepting()
        pipe.send({"ready": True})
        await asyncio.to_thread(pipe.recv)
        with get_session_local()() as db:
            command = claim_task_command(db, runner_id=get_runner_id())
        try:
            if command is not None:
                await execute_task_start(command)
                pending = background_task_manager.running_tasks.get(task_id)
                if pending is not None:
                    await pending
            pipe.send({"executed": command is not None})
            await asyncio.to_thread(pipe.recv)
        finally:
            await background_task_manager.shutdown()
            await stop_task_event_bridge()

    asyncio.run(run())


@pytest.mark.asyncio
@pytest.mark.parametrize("ingress_exits", [False, True])
async def test_two_web_two_worker_delivery_and_single_execution(
    tmp_path, monkeypatch, ingress_exits
):
    from unittest.mock import Mock

    from websockets.asyncio.client import connect

    from xagent.web.models.chat_message import TaskChatMessage
    from xagent.web.models.database import Base, get_engine, get_session_local, init_db
    from xagent.web.models.task import Task, TaskStatus
    from xagent.web.models.user import User
    from xagent.web.services import task_event_bridge
    from xagent.web.services.task_orchestrator import (
        TaskTurnOrchestrator,
        TaskTurnPayload,
    )

    redis_url = os.getenv("XAGENT_TEST_REDIS_URL")
    if not redis_url:
        pytest.skip("XAGENT_TEST_REDIS_URL is not set")
    db_url = f"sqlite:///{tmp_path / 'shared.db'}"
    init_db(db_url=db_url)
    with get_session_local()() as db:
        owner = User(username="owner", password_hash="unused")
        db.add(owner)
        db.flush()
        task = Task(
            user_id=owner.id, title="Shared", source="sdk", status=TaskStatus.PENDING
        )
        db.add(task)
        db.commit()
        task_id, owner_id = task.id, owner.id
    payload = TaskTurnPayload("hello")
    env = {
        "DATABASE_URL": db_url,
        "XAGENT_REDIS_URL": redis_url,
        "XAGENT_TASK_EVENT_CHANNEL_PREFIX": f"integration:{uuid4().hex}",
        "XAGENT_SHARED_TASK_EXECUTION_ENABLED": "true",
    }
    context = multiprocessing.get_context("spawn")
    processes, pipes = [], []

    async def recv(pipe):
        return await asyncio.wait_for(asyncio.to_thread(pipe.recv), 45)

    def start(target, *args):
        parent, child = context.Pipe()
        process = context.Process(
            target=target,
            args=(child, env, *args),
        )
        process.start()
        child.close()
        processes.append(process)
        pipes.append(parent)
        return parent

    sockets = []
    try:
        web1 = start(_web_host, task_id, payload.turn_id)
        web2 = start(_web_host, task_id, payload.turn_id)
        addresses = await asyncio.gather(recv(web1), recv(web2))
        for address in addresses:
            sockets.append(await connect(f"ws://127.0.0.1:{address['port']}"))
        for socket in sockets:
            await socket.recv()
        monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
        monkeypatch.setattr(task_event_bridge, "get_task_event_bridge", lambda: Mock())
        with get_session_local()() as db:
            accepted = TaskTurnOrchestrator.claim_created_turn_no_commit(
                db, task_id=task_id, task_owner_user_id=owner_id, payload=payload
            )
            db.commit()
        if ingress_exits:
            await sockets[0].close()
            web1.send("stop")
            await asyncio.to_thread(processes[0].join, 10)
            assert not processes[0].is_alive()
        workers = [start(_execution_host, task_id) for _ in range(2)]
        await asyncio.gather(*(recv(pipe) for pipe in workers))
        for pipe in workers:
            pipe.send("claim")
        results = await asyncio.gather(*(recv(pipe) for pipe in workers))
        assert sum(item["executed"] for item in results) == 1
        first = (
            [json.loads(await asyncio.wait_for(sockets[0].recv(), 5)) for _ in range(2)]
            if not ingress_exits
            else []
        )
        second = [
            json.loads(await asyncio.wait_for(sockets[1].recv(), 5)) for _ in range(2)
        ]
        if not ingress_exits:
            assert [item["text"] for item in first if item["type"] == "delta"] == [
                "hello ",
                "world",
            ]
        assert [item["text"] for item in second] == ["hello ", "world"]
        # Reconnect to an empty local socket registry and recover persisted
        # output; the bridge does not replay old token events.
        await sockets[1].close()
        sockets[1] = await connect(f"ws://127.0.0.1:{addresses[1]['port']}")
        await sockets[1].recv()  # new, distinct origin registration
        await sockets[1].send("snapshot")
        snapshot = json.loads(await asyncio.wait_for(sockets[1].recv(), 5))
        assert snapshot["output"] == "hello world"
        assert snapshot["run_id"] == accepted.run_id
        with get_session_local()() as db:
            assert db.get(Task, task_id).status == TaskStatus.COMPLETED
            assert (
                db.query(TaskChatMessage)
                .filter_by(task_id=task_id, role="assistant")
                .count()
                == 1
            )
    finally:
        for socket in sockets:
            await socket.close()
        for pipe in pipes:
            try:
                pipe.send("stop")
            except (OSError, EOFError):
                pass
        for process in processes:
            await asyncio.to_thread(process.join, 10)
            if process.is_alive():
                process.terminate()
                await asyncio.to_thread(process.join, 5)
        for pipe in pipes:
            pipe.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.mark.asyncio
async def test_real_redis_subscription_reconnect_exposes_gap(monkeypatch):
    from xagent.web.services.task_event_bridge import TaskEventBridge

    redis_url = os.getenv("XAGENT_TEST_REDIS_URL")
    if not redis_url:
        pytest.skip("XAGENT_TEST_REDIS_URL is not set")
    monkeypatch.setenv("XAGENT_REDIS_URL", redis_url)
    monkeypatch.setenv("XAGENT_TASK_EVENT_CHANNEL_PREFIX", f"reconnect:{uuid4().hex}")
    delivered, statuses = [], []
    unavailable = asyncio.Event()
    received = asyncio.Event()

    async def deliver(message, task_id):
        delivered.append(message["text"])
        received.set()

    async def status(kind):
        statuses.append(kind)
        if kind == "stream_unavailable":
            unavailable.set()

    subscriber = TaskEventBridge(deliver=deliver, stream_status=status)
    publisher = TaskEventBridge()
    try:
        await subscriber.start()
        await publisher.start()
        await publisher.publish({"type": "delta", "text": "before"}, 1)
        await asyncio.wait_for(received.wait(), 5)
        received.clear()
        # Disconnect only this test subscriber's connections; other Redis
        # clients and server data are untouched.
        await subscriber.redis.connection_pool.disconnect()
        await asyncio.wait_for(unavailable.wait(), 5)
        await publisher.publish({"type": "delta", "text": "lost"}, 1)
        await asyncio.wait_for(subscriber.ready.wait(), 5)
        await publisher.publish({"type": "delta", "text": "after"}, 1)
        await asyncio.wait_for(received.wait(), 5)
        assert delivered == ["before", "after"]
        assert "stream_unavailable" in statuses
        assert statuses.count("stream_resync_required") >= 2
    finally:
        await subscriber.close()
        await publisher.close()
