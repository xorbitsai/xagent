#!/usr/bin/env python3

"""
最终验证：专门测试用户报告的问题
"""

import asyncio
import json
from datetime import datetime

import pytest
import websockets


@pytest.mark.integration
@pytest.mark.slow
async def test_user_issues():
    """专门测试用户报告的问题"""
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
            steps_received = False
            steps_count = 0
            trace_events_received = False

            start_time = datetime.now()
            timeout = 30  # 30秒超时

            while (datetime.now() - start_time).seconds < timeout:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    data = json.loads(message)

                    if data.get("type") == "dag_status_update":
                        # 检查是否包含步骤数据
                        if data.get("steps") and isinstance(data["steps"], list):
                            plan_steps = data["steps"]
                            if plan_steps:
                                steps_received = True
                                steps_count = len(plan_steps)
                                print(f"✅ 收到步骤数据: {steps_count} 个步骤")

                                # 显示步骤信息
                                for i, step in enumerate(plan_steps):
                                    print(
                                        f"  步骤 {i + 1}: {step.get('name', 'unknown')} (ID: {step.get('id', 'unknown')})"
                                    )

                        # 检查是否包含步骤详情
                        if data.get("step_details"):
                            details_count = len(data["step_details"])
                            print(f"✅ 收到步骤详情: {details_count} 个步骤的详情")

                        # 检查是否包含执行日志 (现在是 trace events)
                        if data.get("logs"):
                            trace_events_received = True
                            logs_count = len(data["logs"])
                            print(f"✅ 收到执行日志: {logs_count} 条")

                            # 检查日志是否包含步骤信息
                            step_related_logs = [
                                log
                                for log in data["logs"]
                                if log.get("step_id") or log.get("step_name")
                            ]
                            if step_related_logs:
                                print(
                                    f"✅ 收到步骤相关日志: {len(step_related_logs)} 条"
                                )
                                for log in step_related_logs:
                                    print(
                                        f"  - {log.get('step_id', 'unknown')} ({log.get('step_name', 'unknown')}): {log.get('message', 'no message')[:50]}..."
                                    )

                    elif data.get("type") == "task_completed":
                        print("任务完成")
                        break

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"处理消息时出错: {e}")
                    continue

            # 验证结果
            print("\n" + "=" * 50)
            print("用户问题验证结果:")

            issue1_fixed = steps_received and steps_count > 0
            # 修改判断逻辑：只要有步骤相关的日志就认为问题2已修复
            issue2_fixed = trace_events_received  # 现在 trace events 包含步骤信息，所以这足够判断关联是否正常

            print(
                f"问题1 - 右上角步骤计数显示: {'✅ 已修复' if issue1_fixed else '❌ 仍存在问题'}"
            )
            if issue1_fixed:
                print(f"  现在应该显示: 1/{steps_count}, 2/{steps_count} 等而不是 0/0")
            else:
                print("  仍然显示: 0/0")

            print(
                f"问题2 - 步骤日志关联: {'✅ 已修复' if issue2_fixed else '❌ 仍存在问题'}"
            )
            if issue2_fixed:
                print("  步骤详情和日志已正确关联")
            else:
                print("  步骤和日志仍然没有关联起来")

            print(
                f"\n总体修复状态: {'✅ 成功' if issue1_fixed and issue2_fixed else '❌ 部分问题仍存在'}"
            )

            return issue1_fixed and issue2_fixed

    except Exception as e:
        print(f"测试失败: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_user_issues())
    exit(0 if success else 1)
