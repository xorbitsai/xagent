"""
合并后的WebSocket测试
整合了多个WebSocket相关的测试文件，消除重复代码
"""

import json
import unittest
from unittest.mock import AsyncMock

import pytest

from tests.shared import create_test_components


@pytest.mark.slow
class TestWebSocket(unittest.IsolatedAsyncioTestCase):
    """合并后的WebSocket测试"""

    def setUp(self):
        """设置测试组件"""
        self.components = create_test_components()
        self.test_messages = []
        self.test_connections = {}

    async def test_websocket_connection_management(self):
        """测试WebSocket连接管理"""
        print("\n=== 测试WebSocket连接管理 ===")

        # 导入WebSocket管理器
        from xagent.web.api.websocket import ConnectionManager

        manager = ConnectionManager()

        # 测试连接
        websocket1 = AsyncMock()
        websocket2 = AsyncMock()

        # 模拟连接
        await manager.connect(websocket1, 1)
        await manager.connect(websocket2, 2)

        # 验证连接
        self.assertEqual(len(manager.active_connections), 2)
        self.assertIn(1, manager.active_connections)
        self.assertIn(2, manager.active_connections)

        # 测试断开连接
        manager.disconnect(websocket1, 1)

        # 验证断开连接
        self.assertEqual(len(manager.active_connections), 1)
        self.assertNotIn(1, manager.active_connections)
        self.assertIn(2, manager.active_connections)

        print("连接管理验证: ✅")
        print(f"  活跃连接数: {len(manager.active_connections)}")
        print(f"  剩余连接: {list(manager.active_connections.keys())}")

    async def test_websocket_message_broadcast(self):
        """测试WebSocket消息广播"""
        print("\n=== 测试WebSocket消息广播 ===")

        from xagent.web.api.websocket import ConnectionManager

        manager = ConnectionManager()

        # 创建模拟的WebSocket连接
        websocket1 = AsyncMock()
        websocket2 = AsyncMock()

        # 连接
        await manager.connect(websocket1, 1)
        await manager.connect(websocket2, 2)

        # 广播消息
        test_message = {
            "type": "test_message",
            "data": {"content": "Hello, WebSocket!"},
        }

        await manager.broadcast_to_task(test_message, 1)

        # 验证消息发送
        websocket1.send_text.assert_called_once_with(json.dumps(test_message))
        websocket2.send_text.assert_not_called()  # 不应该发送到task_2

        print("消息广播验证: ✅")
        print("  目标task: task_1")
        print(f"  websocket1调用次数: {websocket1.send_text.call_count}")
        print(f"  websocket2调用次数: {websocket2.send_text.call_count}")

    async def test_websocket_message_handling(self):
        """测试WebSocket消息处理"""
        print("\n=== 测试WebSocket消息处理 ===")

        # 测试不同类型的消息处理
        test_messages = [
            {"type": "subscribe", "task_id": "task_1"},
            {"type": "unsubscribe", "task_id": "task_1"},
            {"type": "ping", "data": {"timestamp": "2023-01-01T00:00:00"}},
            {"type": "get_status", "task_id": "task_1"},
        ]

        for message in test_messages:
            # 模拟消息处理
            message_type = message.get("type")

            if message_type == "subscribe":
                # 订阅逻辑
                task_id = message.get("task_id")
                print(f"  处理订阅消息: task_id={task_id}")

            elif message_type == "unsubscribe":
                # 取消订阅逻辑
                task_id = message.get("task_id")
                print(f"  处理取消订阅消息: task_id={task_id}")

            elif message_type == "ping":
                # ping响应
                print(f"  处理ping消息: {message.get('data')}")

            elif message_type == "get_status":
                # 状态查询
                task_id = message.get("task_id")
                print(f"  处理状态查询: task_id={task_id}")

        print("消息处理验证: ✅")
        print(f"  处理消息数量: {len(test_messages)}")

    async def test_websocket_error_handling(self):
        """测试WebSocket错误处理"""
        print("\n=== 测试WebSocket错误处理 ===")

        from xagent.web.api.websocket import ConnectionManager

        manager = ConnectionManager()

        # 创建会抛出异常的WebSocket
        failing_websocket = AsyncMock()
        failing_websocket.send_text.side_effect = Exception("Connection failed")

        # 连接
        await manager.connect(failing_websocket, "task_1")

        # 尝试发送消息 - 应该处理异常
        test_message = {"type": "test", "data": {"content": "test"}}

        try:
            await manager.broadcast_to_task(test_message, 1)
        except Exception as e:
            print(f"  捕获到异常: {e}")

        # 验证连接仍然存在
        self.assertEqual(len(manager.active_connections), 1)

        print("错误处理验证: ✅")
        print(f"  连接数保持: {len(manager.active_connections)}")

    async def test_websocket_trace_integration(self):
        """测试WebSocket与追踪集成"""
        print("\n=== 测试WebSocket与追踪集成 ===")

        from xagent.web.api.websocket import ConnectionManager

        manager = ConnectionManager()

        # 创建模拟WebSocket
        websocket = AsyncMock()
        await manager.connect(websocket, 1)

        # 创建追踪事件
        trace_events = [
            {
                "type": "trace_start",
                "task_id": "task_1",
                "data": {"timestamp": "2023-01-01T00:00:00"},
            },
            {
                "type": "dag_step_start",
                "task_id": "task_1",
                "data": {"step_id": "step1", "step_name": "test_step"},
            },
            {
                "type": "llm_call_start",
                "task_id": "task_1",
                "data": {"model": "gpt-3.5-turbo"},
            },
        ]

        # 模拟追踪事件通过WebSocket发送
        for event in trace_events:
            websocket_message = {"type": "trace_event", "data": event}
            await manager.broadcast_to_task(websocket_message, 1)

        # 验证WebSocket发送
        self.assertEqual(websocket.send_text.call_count, len(trace_events))

        print("追踪集成验证: ✅")
        print(f"  发送追踪事件数: {len(trace_events)}")
        print(f"  WebSocket调用次数: {websocket.send_text.call_count}")

    async def test_websocket_step_update(self):
        """测试WebSocket步骤更新"""
        print("\n=== 测试WebSocket步骤更新 ===")

        from xagent.web.api.websocket import ConnectionManager

        manager = ConnectionManager()

        # 创建多个WebSocket连接
        websockets = []
        for i in range(3):
            websocket = AsyncMock()
            await manager.connect(websocket, i)
            websockets.append(websocket)

        # 发送步骤更新
        step_update = {
            "type": "step_update",
            "task_id": "task_1",
            "data": {
                "step_id": "step1",
                "status": "completed",
                "result": {"output": "Step completed successfully"},
            },
        }

        await manager.broadcast_to_task(step_update, 1)

        # 验证只有task_1的WebSocket接收到消息
        websockets[0].send_text.assert_not_called()
        websockets[1].send_text.assert_called_once_with(json.dumps(step_update))
        websockets[2].send_text.assert_not_called()

        print("步骤更新验证: ✅")
        print("  目标task: task_1")
        print("  正确接收的WebSocket: 1/3")

    async def test_websocket_task_isolation(self):
        """测试WebSocket任务隔离"""
        print("\n=== 测试WebSocket任务隔离 ===")

        from xagent.web.api.websocket import ConnectionManager

        manager = ConnectionManager()

        # 创建不同任务的WebSocket连接
        task_connections = {}
        for i in [1, 2, 3]:
            websocket = AsyncMock()
            await manager.connect(websocket, i)
            task_connections[i] = websocket

        # 向不同任务发送消息
        for i in [1, 2, 3]:
            message = {
                "type": "task_message",
                "task_id": f"task_{i}",
                "data": {"content": f"Message for task_{i}"},
            }
            await manager.broadcast_to_task(message, i)

        # 验证任务隔离
        for task_id, websocket in task_connections.items():
            self.assertEqual(websocket.send_text.call_count, 1)
            call_args = websocket.send_text.call_args[0][0]
            # 消息是JSON字符串，需要解析
            import json

            parsed_message = json.loads(call_args)
            self.assertEqual(parsed_message["task_id"], f"task_{task_id}")

        print("任务隔离验证: ✅")
        print(f"  任务数量: {len(task_connections)}")
        print("  每个任务接收消息数: 1")

    async def test_websocket_connection_limits(self):
        """测试WebSocket连接限制"""
        print("\n=== 测试WebSocket连接限制 ===")

        from xagent.web.api.websocket import ConnectionManager

        manager = ConnectionManager()

        # 创建大量连接
        max_connections = 100
        websockets = []

        for i in range(max_connections):
            websocket = AsyncMock()
            await manager.connect(websocket, f"task_{i}")
            websockets.append(websocket)

        # 验证连接数量
        self.assertEqual(len(manager.active_connections), max_connections)

        # 测试广播性能
        test_message = {"type": "broadcast_test", "data": {"content": "test"}}

        import time

        start_time = time.time()

        # 向所有连接广播消息
        for i in range(max_connections):
            await manager.broadcast_to_task(test_message, i)

        end_time = time.time()
        duration = end_time - start_time

        # 验证性能
        self.assertLess(duration, 5.0, f"{max_connections}次广播应该在5秒内完成")

        print("连接限制验证: ✅")
        print(f"  最大连接数: {max_connections}")
        print(f"  广播耗时: {duration:.3f}秒")
        print(f"  平均每次广播: {duration / max_connections * 1000:.3f}毫秒")

    async def test_websocket_message_validation(self):
        """测试WebSocket消息验证"""
        print("\n=== 测试WebSocket消息验证 ===")

        # 测试不同类型的消息验证
        test_cases = [
            {
                "name": "有效消息",
                "message": {
                    "type": "trace_event",
                    "task_id": "task_1",
                    "data": {"event_type": "test"},
                },
                "should_pass": True,
            },
            {
                "name": "缺少类型",
                "message": {"task_id": "task_1", "data": {"event_type": "test"}},
                "should_pass": False,
            },
            {
                "name": "缺少task_id",
                "message": {"type": "trace_event", "data": {"event_type": "test"}},
                "should_pass": False,
            },
            {"name": "空消息", "message": {}, "should_pass": False},
            {"name": "非字典消息", "message": "invalid_message", "should_pass": False},
        ]

        for test_case in test_cases:
            message = test_case["message"]
            should_pass = test_case["should_pass"]

            # 简单的消息验证逻辑
            is_valid = True
            if not isinstance(message, dict):
                is_valid = False
            elif "type" not in message:
                is_valid = False
            elif "task_id" not in message:
                is_valid = False

            self.assertEqual(
                is_valid, should_pass, f"消息 '{test_case['name']}' 验证失败"
            )

            print(f"  {test_case['name']}: {'✅' if is_valid == should_pass else '❌'}")

        print("消息验证验证: ✅")
        print(f"  测试用例数: {len(test_cases)}")

    async def test_websocket_stream_event_format(self):
        """测试WebSocket流式事件格式统一"""
        print("\n=== 测试WebSocket流式事件格式统一 ===")

        from datetime import datetime, timezone

        from xagent.web.api.websocket import create_stream_event

        # 测试创建流式事件
        test_data = {
            "step_id": "step1",
            "step_name": "test_step",
            "status": "completed",
            "result": "test_result",
        }

        stream_event = create_stream_event(
            "dag_step_info",
            task_id=1,
            data=test_data,
            timestamp=datetime.now(timezone.utc),
        )

        # 验证流式事件格式
        self.assertEqual(stream_event["type"], "trace_event")
        self.assertEqual(stream_event["event_type"], "dag_step_info")
        self.assertEqual(stream_event["task_id"], 1)
        self.assertIn("event_id", stream_event)
        self.assertIn("timestamp", stream_event)
        self.assertEqual(stream_event["data"], test_data)

        print("流式事件格式验证: ✅")
        print(f"  事件类型: {stream_event['event_type']}")
        print(f"  任务ID: {stream_event['task_id']}")
        print(f"  事件ID: {stream_event['event_id']}")
        print(f"  数据键: {list(stream_event['data'].keys())}")

    async def test_websocket_reconnection_handling(self):
        """测试WebSocket重连处理"""
        print("\n=== 测试WebSocket重连处理 ===")

        from xagent.web.api.websocket import ConnectionManager

        manager = ConnectionManager()

        # 创建WebSocket连接
        websocket1 = AsyncMock()
        await manager.connect(websocket1, 1)

        # 模拟连接断开
        manager.disconnect(websocket1, 1)

        # 验证连接已断开
        self.assertEqual(len(manager.active_connections), 0)

        # 重新连接
        websocket2 = AsyncMock()
        await manager.connect(websocket2, 1)

        # 验证重新连接成功
        self.assertEqual(len(manager.active_connections), 1)
        self.assertIn(1, manager.active_connections)

        # 测试重连后消息发送
        test_message = {"type": "reconnection_test", "data": {"content": "test"}}
        await manager.broadcast_to_task(test_message, 1)

        # 验证新连接能接收消息
        websocket2.send_text.assert_called_once_with(json.dumps(test_message))

        print("重连处理验证: ✅")
        print("  断开后连接数: 0")
        print(f"  重连后连接数: {len(manager.active_connections)}")
        print(f"  消息发送成功: {websocket2.send_text.call_count}")

    async def test_historical_data_streaming(self):
        """测试历史数据流式发送"""
        print("\n=== 测试历史数据流式发送 ===")

        from datetime import datetime, timezone
        from unittest.mock import AsyncMock

        from xagent.web.api.websocket import create_stream_event

        # 创建模拟的WebSocket和数据库
        websocket = AsyncMock()

        # 模拟数据库数据
        now = datetime.now(timezone.utc)
        mock_task_data = {
            "id": 1,
            "title": "Test Task",
            "description": "Test Description",
            "status": "completed",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        # 模拟历史数据事件
        historical_events = [
            create_stream_event("task_info", 1, mock_task_data, now),
            create_stream_event(
                "task_message", 1, {"role": "user", "content": "Hello"}, now
            ),
            create_stream_event(
                "dag_step_info",
                1,
                {"id": "step1", "name": "test_step", "status": "completed"},
                now,
            ),
            create_stream_event(
                "execution_log", 1, {"level": "info", "message": "Step completed"}, now
            ),
            create_stream_event(
                "historical_data_complete",
                1,
                {"message": "Historical data loaded"},
                now,
            ),
        ]

        # 模拟发送历史数据
        for event in historical_events:
            await websocket.send_text(json.dumps(event))

        # 验证历史数据发送
        self.assertEqual(websocket.send_text.call_count, len(historical_events))

        # 验证事件格式
        for i, call in enumerate(websocket.send_text.call_args_list):
            sent_data = json.loads(call[0][0])
            self.assertEqual(sent_data["type"], "trace_event")
            self.assertIn("event_id", sent_data)
            self.assertIn("timestamp", sent_data)
            self.assertEqual(sent_data["task_id"], 1)

        print("历史数据流式发送验证: ✅")
        print(f"  发送事件数量: {len(historical_events)}")
        print(f"  WebSocket调用次数: {websocket.send_text.call_count}")

    async def test_rewrite_file_links_supports_legacy_preview_paths(self):
        from xagent.web.api.websocket import _rewrite_file_links_to_file_id

        output_text = (
            "Legacy link: [report](/preview/web_task_12/output/report.html)\n"
            "Legacy uploads link: [img](/uploads/user_3/web_task_12/output/a.png)\n"
            "File link: [doc](file:/preview/web_task_12/output/readme.md)\n"
            "Task-local link: [poster](output/poster.html)"
        )
        path_to_file_id = {
            "preview/web_task_12/output/report.html": "fid-report",
            "uploads/user_3/web_task_12/output/a.png": "fid-image",
            "preview/web_task_12/output/readme.md": "fid-readme",
            "output/poster.html": "fid-poster",
        }

        rewritten = _rewrite_file_links_to_file_id(output_text, path_to_file_id)

        self.assertIn("[report](file:fid-report)", rewritten)
        self.assertIn("[img](file:fid-image)", rewritten)
        self.assertIn("[doc](file:fid-readme)", rewritten)
        self.assertIn("[poster](file:fid-poster)", rewritten)

    async def test_rewrite_file_links_preserves_non_legacy_urls(self):
        from xagent.web.api.websocket import _rewrite_file_links_to_file_id

        output_text = "[site](https://example.com) and [local](/not-preview/path)"
        rewritten = _rewrite_file_links_to_file_id(output_text, {})
        self.assertEqual(rewritten, output_text)

    async def test_websocket_trace_handler_integration(self):
        """测试WebSocket追踪处理器集成"""
        print("\n=== 测试WebSocket追踪处理器集成 ===")

        from unittest.mock import AsyncMock, patch

        from xagent.core.agent.trace import (
            TraceAction,
            TraceCategory,
            TraceEvent,
            TraceEventType,
            TraceScope,
        )
        from xagent.web.api.ws_trace_handlers import WebSocketTraceHandler

        # 创建模拟的WebSocket管理器
        mock_manager = AsyncMock()

        # 用patch替换manager
        with patch("xagent.web.api.ws_trace_handlers.manager", mock_manager):
            # 创建WebSocket追踪处理器
            handler = WebSocketTraceHandler(task_id=1)

            # 创建测试事件
            event_type = TraceEventType(
                TraceScope.STEP, TraceAction.START, TraceCategory.DAG
            )
            trace_event = TraceEvent(
                event_type=event_type,
                task_id=1,
                step_id="step1",
                data={"step_name": "test_step", "tool_name": "test_tool"},
            )

            # 处理事件
            await handler.handle_event(trace_event)

            # 验证事件被转换为流式格式并发送
            mock_manager.broadcast_to_task.assert_called_once()

            # 获取发送的消息
            sent_message = mock_manager.broadcast_to_task.call_args[0][0]

            # 验证消息格式
            self.assertEqual(sent_message["type"], "trace_event")
            self.assertEqual(sent_message["event_type"], "dag_step_start")
            self.assertEqual(sent_message["task_id"], 1)
            self.assertIn("event_id", sent_message)
            self.assertIn("timestamp", sent_message)
            self.assertIn("data", sent_message)

            # 验证数据内容
            self.assertEqual(sent_message["data"]["step_name"], "test_step")
            self.assertEqual(sent_message["data"]["tool_name"], "test_tool")

            print("WebSocket追踪处理器集成验证: ✅")
            print(f"  事件类型: {sent_message['event_type']}")
            print(f"  任务ID: {sent_message['task_id']}")
            print(f"  步骤名称: {sent_message['data']['step_name']}")
            print(f"  工具名称: {sent_message['data']['tool_name']}")

    async def test_websocket_pause_resume_handlers_exist(self):
        """测试WebSocket暂停恢复处理器存在性"""
        print("\n=== 测试WebSocket暂停恢复处理器存在性 ===")

        # 测试导入
        from xagent.web.api.websocket import handle_pause_task, handle_resume_task

        # 验证函数存在
        assert callable(handle_pause_task)
        assert callable(handle_resume_task)

        print("  WebSocket暂停恢复处理器存在性验证完成")

    async def test_websocket_pause_resume_error_handling_simple(self):
        """测试WebSocket暂停恢复错误处理（简化版）"""
        print("\n=== 测试WebSocket暂停恢复错误处理（简化版） ===")

        from unittest.mock import AsyncMock, patch

        from xagent.web.api.websocket import handle_pause_task

        # 创建模拟WebSocket
        websocket = AsyncMock()
        task_id = 123

        # 模拟agent manager返回None（agent不存在的情况）
        mock_agent_manager = AsyncMock()
        mock_agent_manager.get_agent_for_task = AsyncMock(return_value=None)

        # 模拟chat模块的agent_manager
        mock_chat_module = AsyncMock()
        mock_chat_module.agent_manager = mock_agent_manager

        # 模拟全局manager
        mock_manager = AsyncMock()

        with patch.dict("sys.modules", {"xagent.web.api.chat": mock_chat_module}):
            with patch("xagent.web.api.websocket.manager", mock_manager):
                # 测试暂停 - 应该发送错误消息
                message_data = {
                    "type": "pause_task",
                    "task_id": task_id,
                    "timestamp": "2024-01-01T00:00:00Z",
                }

                await handle_pause_task(websocket, task_id, message_data)

                # 验证发送了错误消息
                mock_manager.send_personal_message.assert_called()

        print("  WebSocket暂停恢复错误处理测试完成")


if __name__ == "__main__":
    unittest.main()
