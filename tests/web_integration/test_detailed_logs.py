#!/usr/bin/env python3

"""
详细测试日志数据接收和关联
"""

import asyncio
import json
from datetime import datetime

import pytest
import websockets


@pytest.mark.integration
@pytest.mark.slow
async def test_detailed_logs():
    """详细测试日志数据接收和关联"""
    uri = "ws://localhost:8000/ws/chat/1001"

    try:
        print(f"连接到WebSocket: {uri}")
        async with websockets.connect(uri) as websocket:
            # 发送测试消息
            test_message = {
                "type": "chat",
                "message": "分析销售数据并生成报告",
                "context": {},
            }

            print(f"发送测试消息: {test_message['message']}")
            await websocket.send(json.dumps(test_message))

            # 接收响应
            message_count = 0
            all_logs = []
            step_logs = []
            steps_with_log_data = {}

            start_time = datetime.now()
            timeout = 60  # 60秒超时

            while (datetime.now() - start_time).seconds < timeout:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    message_count += 1

                    if data.get("type") == "dag_status_update":
                        # 检查日志数据
                        if "logs" in data:
                            logs = data["logs"]
                            print(
                                f"\n--- 消息 {message_count}: 收到 {len(logs)} 条日志 ---"
                            )

                            for i, log in enumerate(logs):
                                print(f"  日志 {i + 1}:")
                                print(f"    级别: {log.get('level', 'unknown')}")
                                print(
                                    f"    消息: {log.get('message', 'no message')[:100]}..."
                                )
                                print(f"    步骤ID: {log.get('step_id', 'None')}")
                                print(f"    步骤名称: {log.get('step_name', 'None')}")
                                print(f"    时间戳: {log.get('timestamp', 'None')}")

                                # 添加到日志列表
                                all_logs.append(log)

                                # 如果有步骤信息，添加到步骤日志
                                if log.get("step_id") or log.get("step_name"):
                                    step_logs.append(log)
                                    step_id = log.get(
                                        "step_id", log.get("step_name", "unknown")
                                    )
                                    if step_id not in steps_with_log_data:
                                        steps_with_log_data[step_id] = []
                                    steps_with_log_data[step_id].append(log)

                    elif data.get("type") == "task_completed":
                        print("\n--- 任务完成 ---")
                        break

                except asyncio.TimeoutError:
                    print("等待消息超时，继续...")
                    continue
                except Exception as e:
                    print(f"处理消息时出错: {e}")
                    continue

            # 分析结果
            print("\n" + "=" * 60)
            print("详细日志分析结果:")
            print(f"总消息数: {message_count}")
            print(f"总日志数: {len(all_logs)}")
            print(f"步骤相关日志数: {len(step_logs)}")
            print(f"有日志数据的步骤数: {len(steps_with_log_data)}")

            if steps_with_log_data:
                print("\n步骤日志关联详情:")
                for step_id, logs in steps_with_log_data.items():
                    print(f"  步骤 {step_id}: {len(logs)} 条日志")
                    for log in logs:
                        print(
                            f"    - {log.get('level', 'unknown')}: {log.get('message', 'no message')[:60]}..."
                        )

            # 测试前端日志关联逻辑
            print("\n" + "=" * 60)
            print("模拟前端日志关联逻辑测试:")

            # 模拟步骤数据
            mock_steps = [
                {"id": "step1", "name": "Research Sales Analysis Methods"},
                {"id": "step2", "name": "Find Sample Sales Data"},
                {"id": "step3", "name": "Set Up Analysis Environment"},
                {"id": "step4", "name": "Collect and Load Data"},
                {"id": "step5", "name": "Data Cleaning and Preparation"},
            ]

            for step in mock_steps:
                # 模拟前端的日志过滤逻辑
                step_related_logs = []

                for log in all_logs:
                    # 检查结构化的步骤信息
                    if log.get("step_id") == step["id"]:
                        step_related_logs.append(log)
                    # 检查步骤名称匹配
                    elif log.get("step_name") == step["name"]:
                        step_related_logs.append(log)
                    # 回退到消息内容过滤
                    elif log.get("message"):
                        message = log["message"].lower()
                        step_name = step["name"].lower()

                        if (
                            step_name in message
                            or step["id"] in message
                            or any(
                                keyword in message
                                for keyword in [
                                    "research",
                                    "analysis",
                                    "data",
                                    "cleaning",
                                ]
                            )
                        ):
                            step_related_logs.append(log)

                print(f"步骤 {step['id']} ({step['name']}):")
                print(f"  关联日志数: {len(step_related_logs)}")
                if step_related_logs:
                    for log in step_related_logs[:2]:  # 只显示前2条
                        print(f"    - {log.get('message', 'no message')[:80]}...")
                else:
                    print("    ❌ 没有关联的日志")

            return len(step_logs) > 0

    except Exception as e:
        print(f"测试失败: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_detailed_logs())
    exit(0 if success else 1)
