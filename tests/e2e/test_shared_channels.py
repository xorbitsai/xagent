"""Platform callbacks use durable START and real worker execution.

Only platform network I/O is replaced; selection, authorization, transcript,
Redis trace forwarding, AgentService and final rendering use production code.
"""

import asyncio
import importlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "platform,scenario",
    [
        (platform, scenario)
        for platform in ("slack", "feishu", "telegram")
        for scenario in ("text", "file", "reply")
    ]
    + [("telegram", "pending")],
)
async def test_channel_callback_runs_remotely_and_returns_answer(
    shared_app, monkeypatch, platform, scenario, tmp_path
):
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task import Task
    from xagent.web.models.user_channel import UserChannel
    from xagent.web.services.task_event_bridge import (
        start_task_event_bridge,
        stop_task_event_bridge,
    )

    app = shared_app
    pending = scenario == "pending"
    if pending:
        worker, pipe = app.processes[-1], app.pipes[-1]
        pipe.send("stop")
        await asyncio.to_thread(worker.join, 30)
        assert worker.exitcode == 0, app.diagnostics()
        from xagent.web.services import shared_channel_execution

        monkeypatch.setattr(
            shared_channel_execution,
            "get_task_reply_wait_timeout_seconds",
            lambda: 0.03,
        )
    with_file = scenario == "file"
    with_question = scenario == "reply"
    input_text = (
        "e2e:gate"
        if pending
        else "e2e:files"
        if with_file
        else "e2e:ask"
        if with_question
        else "Channel question"
    )
    expected_text = (
        "still being processed"
        if pending
        else "Which choice?"
        if with_question
        else "Shared E2E answer"
    )
    followup_text = "e2e:answer" if with_question else "Channel followup"
    monkeypatch.chdir(tmp_path)
    with get_session_local()() as db:
        channel = UserChannel(
            user_id=app.user_id,
            channel_type=platform,
            channel_name="E2E",
            is_active=True,
        )
        channel.config = {"allowed_users": ["sender", "123"]}
        db.add(channel)
        db.commit()
        channel_id = channel.id
    delivered = []
    forwarded = []
    module = importlib.import_module(f"xagent.web.channels.{platform}.bot")
    handler_type = getattr(
        module,
        {
            "slack": "SlackTraceHandler",
            "feishu": "FeishuTraceHandler",
            "telegram": "TelegramTraceHandler",
        }[platform],
    )
    handle_event = handler_type.handle_event

    async def observe_event(self, event):
        forwarded.append(event)
        await handle_event(self, event)

    monkeypatch.setattr(handler_type, "handle_event", observe_event)
    await start_task_event_bridge()
    bot = None
    try:
        if platform == "slack":
            from xagent.web.channels.slack.bot import SlackBotInstance

            bot = SlackBotInstance("test-token", None, "e2e", channel_id, "E2E", "bot")
            bot.web_client = SimpleNamespace(
                chat_postMessage=AsyncMock(return_value={"ts": "loading"}),
                chat_update=AsyncMock(return_value={"ok": True}),
                files_upload_v2=AsyncMock(
                    side_effect=lambda **kwargs: delivered.append(
                        Path(kwargs["file"]).read_bytes()
                    )
                ),
            )
            if with_file:
                import httpx

                original_client = httpx.AsyncClient
                transport = httpx.MockTransport(
                    lambda request: httpx.Response(
                        200, content=b"unique shared input\n"
                    )
                )
                monkeypatch.setattr(
                    module.httpx,
                    "AsyncClient",
                    lambda **kwargs: original_client(transport=transport, **kwargs),
                )
            await asyncio.wait_for(
                bot._process_event(
                    "conversation",
                    {"team_id": "T1"},
                    {
                        "type": "message",
                        "channel_type": "im",
                        "channel": "D1",
                        "user": "sender",
                        "ts": "1.0",
                        "text": input_text,
                        "files": [
                            {
                                "id": "platform-file",
                                "name": "source.txt",
                                "mimetype": "text/plain",
                                "url_private_download": "https://files.slack.com/source.txt",
                            }
                        ]
                        if with_file
                        else [],
                    },
                ),
                30,
            )
            assert expected_text in str(bot.web_client.chat_update.call_args_list)
        elif platform == "feishu":
            from xagent.web.channels.feishu.bot import FeishuBotInstance

            bot = FeishuBotInstance("test-id", "test-secret", "e2e", channel_id, "E2E")
            bot.api_client = Mock()
            bot.api_client.im.v1.message.patch.return_value.success.return_value = True
            bot.api_client.im.v1.message_resource.get.return_value = SimpleNamespace(
                success=lambda: True,
                file_name="source.txt",
                file=io.BytesIO(b"unique shared input\n"),
            )
            bot._send_text = AsyncMock(return_value="loading")
            bot._update_text = AsyncMock()
            message = SimpleNamespace(
                event=SimpleNamespace(
                    message=SimpleNamespace(
                        chat_id="chat",
                        message_id="message",
                        message_type="text",
                        content=json.dumps({"text": input_text}),
                    )
                )
            )
            messages = [message]
            if with_file:
                messages.append(
                    SimpleNamespace(
                        event=SimpleNamespace(
                            message=SimpleNamespace(
                                chat_id="chat",
                                message_id="attachment",
                                message_type="file",
                                content='{"file_key":"platform-file"}',
                            )
                        )
                    )
                )
            await asyncio.wait_for(bot._process_messages_batch("sender", messages), 30)
            assert expected_text in str(bot._update_text.call_args_list)
        else:
            from xagent.web.channels.telegram.bot import TelegramBotInstance

            bot = TelegramBotInstance(
                "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi", "e2e", channel_id, "E2E"
            )
            await bot.bot.session.close()

            async def download_file(_path, *, destination):
                Path(destination).write_bytes(b"unique shared input\n")

            bot.bot = SimpleNamespace(
                edit_message_text=AsyncMock(),
                send_message=AsyncMock(),
                get_file=AsyncMock(
                    return_value=SimpleNamespace(file_path="source.txt")
                ),
                download_file=download_file,
            )

            async def answer_document(document, **kwargs):
                delivered.append(Path(document.path).read_bytes())
                return SimpleNamespace(delete=AsyncMock())

            loading = SimpleNamespace(
                message_id=77, edit_text=AsyncMock(), delete=AsyncMock()
            )
            message = SimpleNamespace(
                message_id=1,
                message_thread_id=None,
                from_user=SimpleNamespace(id=123),
                chat=SimpleNamespace(id=456),
                answer=AsyncMock(return_value=loading),
                answer_document=answer_document,
                text=input_text,
                caption=None,
                document=SimpleNamespace(
                    file_id="platform-file",
                    file_name="source.txt",
                    mime_type="text/plain",
                    file_size=20,
                )
                if with_file
                else None,
                photo=None,
                audio=None,
                voice=None,
                video=None,
            )
            await asyncio.wait_for(bot._process_user_messages_batch(123, [message]), 30)
            assert expected_text in str(loading.edit_text.call_args_list)
            if pending:
                from xagent.web.services.channel_delivery import recover_channel_results

                task_id, turn = bot.user_active_executions[123]
                assert bot._stop_current_conversation(123)
                await turn.stop_task
                await asyncio.to_thread(app.start, "worker")
                await asyncio.to_thread(app.wait_task, task_id, status="paused")
                bot._deliver_telegram_result = AsyncMock()
                # The pending notice set the retry delay; make recovery ready now.
                from xagent.web.models.task_channel_delivery import TaskChannelDelivery

                with get_session_local()() as db:
                    db.get(TaskChannelDelivery, turn.command_db_id).available_at = None
                    db.commit()
                await recover_channel_results(channel_id, bot._deliver_shared_result)
                bot._deliver_telegram_result.assert_awaited_once()
                assert 123 not in bot.user_active_executions
                return
            if with_file:
                assert delivered == [b"UNIQUE SHARED INPUT\n"]
        with get_session_local()() as db:
            tasks = db.query(Task).all()
            assert len(tasks) == 1
            task_id = tasks[0].id
        first = app.wait_task(
            task_id, status="waiting_for_user" if with_question else "completed"
        )
        if with_file:
            if platform != "feishu":
                assert delivered == [b"UNIQUE SHARED INPUT\n"]
            else:
                # Feishu currently returns a managed file link in its final text.
                from xagent.web.models.uploaded_file import UploadedFile

                with get_session_local()() as db:
                    output_id = (
                        db.query(UploadedFile)
                        .filter_by(task_id=str(task_id), filename="derived.txt")
                        .one()
                        .file_id
                    )
                assert output_id in first["output"]
                assert output_id in str(bot._update_text.call_args_list)
                response = app.client.get(
                    f"/api/files/download/{output_id}", headers=app.headers
                )
                assert response.status_code == 200, response.text
                assert response.content == b"UNIQUE SHARED INPUT\n"
        else:
            if platform == "slack":
                await asyncio.wait_for(
                    bot._process_event(
                        "conversation",
                        {"team_id": "T1"},
                        {
                            "type": "message",
                            "channel_type": "im",
                            "channel": "D1",
                            "user": "sender",
                            "ts": "2.0",
                            "text": followup_text,
                        },
                    ),
                    30,
                )
            elif platform == "feishu":
                message.event.message.message_id = "followup"
                message.event.message.content = json.dumps({"text": followup_text})
                await asyncio.wait_for(
                    bot._process_messages_batch("sender", [message]), 30
                )
            else:
                message.text = followup_text
                await asyncio.wait_for(
                    bot._process_user_messages_batch(123, [message]), 30
                )
            second = app.wait_task(task_id)
            assert second["run_id"] != first["run_id"]
            assert "Shared E2E answer" in second["output"]
            with get_session_local()() as db:
                assert db.query(Task).count() == 1
        calls = [
            json.loads(line)
            for path in app.root.glob("model-*.jsonl")
            for line in path.read_text().splitlines()
        ]
        assert calls and {call["role"] for call in calls} == {"worker"}
        if not with_file:
            assert any(followup_text in json.dumps(call["messages"]) for call in calls)
            replies = (
                bot.web_client.chat_update.call_args_list
                if platform == "slack"
                else bot._update_text.call_args_list
                if platform == "feishu"
                else loading.edit_text.call_args_list
            )
            assert "Shared E2E answer" in str(replies[-1])
        assert forwarded
        assert all(
            event.task_id is None or str(event.task_id) == str(task_id)
            for event in forwarded
        )
    finally:
        await stop_task_event_bridge()
        from xagent.web.services.task_events import (
            set_task_command_delivery,
            set_task_event_sink,
        )

        set_task_command_delivery(None)
        set_task_event_sink(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["stop", "deactivate"])
async def test_queued_channel_control_is_applied_by_worker(shared_app, action):
    from xagent.web.models.database import get_session_local
    from xagent.web.models.task_command import TaskExecutionCommand
    from xagent.web.models.user_channel import UserChannel
    from xagent.web.services.shared_channel_execution import prepare_shared_channel_turn
    from xagent.web.services.task_event_bridge import (
        start_task_event_bridge,
        stop_task_event_bridge,
    )
    from xagent.web.services.task_events import (
        set_task_command_delivery,
        set_task_event_sink,
    )
    from xagent.web.services.task_orchestrator import TaskTurnPayload

    app = shared_app
    worker, pipe = app.processes[-1], app.pipes[-1]
    pipe.send("stop")
    worker.join(30)
    assert worker.exitcode == 0, app.diagnostics()
    with get_session_local()() as db:
        channel = UserChannel(
            user_id=app.user_id,
            channel_type="telegram",
            channel_name="Queued",
            is_active=True,
        )
        channel.config = {"allowed_users": ["123"]}
        db.add(channel)
        db.commit()
        channel_id = channel.id
    await start_task_event_bridge()
    turn = None
    execution = None
    try:
        turn = await prepare_shared_channel_turn(
            channel_id=channel_id,
            external_user_id="123",
            active_task_id=None,
            text="e2e:gate",
            channel_name="Queued",
        )
        assert turn is not None
        execution = asyncio.create_task(
            turn.execute(
                TaskTurnPayload(
                    transcript_message="e2e:gate", execution_message="e2e:gate"
                ),
                None,
            )
        )
        async with asyncio.timeout(10):
            while not turn.accepted:
                await asyncio.sleep(0.02)
        if action == "stop":
            await turn.stop()
        else:
            with get_session_local()() as db:
                db.get(UserChannel, channel_id).is_active = False
                db.commit()
        await asyncio.to_thread(app.start, "worker")
        result = await asyncio.wait_for(execution, 30)
        expected = "paused" if action == "stop" else "failed"
        await asyncio.to_thread(app.wait_task, turn.selection.task_id, status=expected)
        if action == "stop":
            assert result["status"] in {"paused", "interrupted"}
            with get_session_local()() as db:
                stop = db.query(TaskExecutionCommand).filter_by(kind="pause").one()
                assert stop.status == "completed"
                assert stop.target_run_id == turn.run_id
        else:
            assert not list(app.root.glob("model-*.jsonl"))
    finally:
        if execution is not None and not execution.done():
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        if turn is not None:
            await turn.close()
        await stop_task_event_bridge()
        set_task_command_delivery(None)
        set_task_event_sink(None)
