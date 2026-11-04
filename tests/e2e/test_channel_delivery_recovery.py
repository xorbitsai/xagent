"""SIGKILL ingress after execution, then recover through the platform renderer."""

import asyncio
import multiprocessing
import os
import signal
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


def ingress(pipe, environment, channel_id, delivery_log, restart=False):
    os.environ.update(environment)
    os.environ.pop("PYTEST_CURRENT_TEST", None)

    async def main():
        from xagent.web.models.database import configure_db

        configure_db()
        from xagent.web.services import shared_channel_execution as shared
        from xagent.web.services.task_event_bridge import (
            get_task_event_bridge,
            start_task_event_bridge,
            stop_task_event_bridge,
        )
        from xagent.web.services.task_orchestrator import TaskTurnPayload

        await start_task_event_bridge()
        if restart:
            pipe.send(
                {
                    "host": get_task_event_bridge().host_id,
                    "origins": len(get_task_event_bridge()._origins),
                }
            )
            from xagent.web.channels.feishu.bot import FeishuBotInstance
            from xagent.web.services.channel_delivery import recover_channel_results

            bot = FeishuBotInstance(
                "test-id", "test-secret", "e2e", channel_id, "Recovery"
            )

            async def update(chat_id, message_id, text, **kwargs):
                assert chat_id == "saved-chat" and message_id == "saved-loading"
                with Path(delivery_log).open("a") as output:
                    output.write(text + "\n")

            bot._update_text = update
            await recover_channel_results(channel_id, bot._deliver_shared_result)
            await recover_channel_results(channel_id, bot._deliver_shared_result)
            await stop_task_event_bridge()
            return
        read = shared._read_channel_result
        captured = False

        def hold_before_delivery(command_id, run_id):
            nonlocal captured
            result = read(command_id, run_id)
            if result is not None and not captured:
                captured = True
                pipe.send({"result_ready": result, "command_id": command_id})
            return None

        shared._read_channel_result = hold_before_delivery
        turn = await shared.prepare_shared_channel_turn(
            channel_id=channel_id,
            external_user_id="sender",
            active_task_id=None,
            text="Channel question",
            channel_name="Review",
        )
        turn.delivery_destination = {
            "chat_id": "saved-chat",
            "loading_message_id": "saved-loading",
        }
        pipe.send(
            {"task_id": turn.selection.task_id, "host": get_task_event_bridge().host_id}
        )
        result = await turn.execute(TaskTurnPayload("Channel question"), None)
        Path(delivery_log).write_text(str(result))

    asyncio.run(main())


def test_killed_ingress_recovers_final_reply_without_reexecuting_agent(
    shared_app, tmp_path
):
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task_command import TaskExecutionCommand
    from xagent.web.models.user_channel import UserChannel

    app = shared_app
    with get_session_local()() as db:
        channel = UserChannel(
            user_id=app.user_id,
            channel_type="feishu",
            channel_name="Review",
            is_active=True,
            config={"allowed_users": ["sender"]},
        )
        db.add(channel)
        db.commit()
        channel_id = channel.id
    context = multiprocessing.get_context("spawn")
    delivery_log = tmp_path / "delivered.txt"
    parent, child = context.Pipe()
    process = context.Process(
        target=ingress, args=(child, app.environment, channel_id, str(delivery_log))
    )
    restarted = None
    process.start()
    try:
        assert parent.poll(30), app.diagnostics()
        initial = parent.recv()
        assert parent.poll(40), app.diagnostics()
        completed = parent.recv()
        assert completed["result_ready"]["status"] == "completed"
        os.kill(process.pid, signal.SIGKILL)
        process.join(5)
        assert process.exitcode == -signal.SIGKILL
        p2, c2 = context.Pipe()
        restarted = context.Process(
            target=ingress,
            args=(c2, app.environment, channel_id, str(delivery_log), True),
        )
        restarted.start()
        assert p2.poll(20)
        after = p2.recv()
        assert after["host"] != initial["host"] and after["origins"] == 0
        with get_session_local()() as db:
            row = db.get(TaskExecutionCommand, completed["command_id"])
            assert row.result["channel_result"]["status"] == "completed"
            assert row.reply_host_id == initial["host"]
        restarted.join(20)
        assert restarted.exitcode == 0
        assert delivery_log.read_text().count("Shared E2E answer") == 1
        from xagent.web.models.task_channel_delivery import TaskChannelDelivery

        with get_session_local()() as db:
            assert db.query(TaskExecutionCommand).count() == 1
            assert (
                db.get(TaskChannelDelivery, completed["command_id"]).status
                == "delivered"
            )
    finally:
        for child_process in (process, restarted):
            if child_process is not None and child_process.is_alive():
                child_process.terminate()
                child_process.join(5)
