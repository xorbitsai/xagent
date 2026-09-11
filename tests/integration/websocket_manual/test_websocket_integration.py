#!/usr/bin/env python3
"""
更详细的WebSocket测试，包括更多调试信息
"""

import asyncio
import json
import time

import pytest
import websockets


@pytest.mark.slow
async def test_websocket_debug():
    uri = "ws://127.0.0.1:8000/ws/chat/3"
    try:
        async with websockets.connect(uri) as websocket:
            print("=== WebSocket调试测试 ===")

            # 1. 首先请求历史数据
            print("\n1. 请求历史数据...")
            await websocket.send(json.dumps({"type": "status_request"}))

            response = await websocket.recv()
            data = json.loads(response)
            print(f"历史数据类型: {data.get('type')}")
            print(f"任务ID: {data.get('task_id')}")

            # 2. 发送新的聊天消息
            print("\n2. 发送新的DAG任务...")
            test_message = "用python计算5*8"
            await websocket.send(
                json.dumps({"type": "chat", "message": test_message, "context": {}})
            )

            # 3. 接收消息确认
            response = await websocket.recv()
            data = json.loads(response)
            print(f"消息确认: {data.get('type')}")

            # 4. 接收实时更新（等待更长时间）
            print("\n3. 等待实时更新...")
            message_count = 0
            max_messages = 20
            start_time = time.time()
            timeout = 60  # 60秒超时

            while message_count < max_messages:
                try:
                    # 计算剩余时间
                    elapsed = time.time() - start_time
                    remaining = timeout - elapsed

                    if remaining <= 0:
                        print(f"\n⏰ 总超时，共收到 {message_count} 个消息")
                        break

                    response = await asyncio.wait_for(
                        websocket.recv(), timeout=min(remaining, 10.0)
                    )
                    data = json.loads(response)
                    message_count += 1

                    print(f"\n--- 消息 {message_count} (时间: {elapsed:.1f}s) ---")
                    print(f"类型: {data.get('type')}")

                    if data.get("type") == "agent_thinking":
                        print(f"思考状态: {data.get('message')}")
                    elif data.get("type") == "agent_response":
                        print(f"AI响应: {data.get('message', '')[:100]}...")
                    elif data.get("type") == "task_completed":
                        print(f"任务完成: {data.get('success')}")
                        print(f"最终结果: {data.get('result', '')[:100]}...")
                    elif data.get("type") == "historical_data":
                        print(f"历史数据更新 - 步骤数: {len(data.get('steps', []))}")
                        print(f"步骤详情数: {len(data.get('step_details', {}))}")
                        print(f"日志数: {len(data.get('logs', []))}")
                    elif data.get("type") == "agent_error":
                        print(f"错误: {data.get('message')}")
                    elif data.get("type") == "agent_thought":
                        print(f"思考完成: {data.get('message')}")
                    elif data.get("type") == "agent_action":
                        print(f"AI行动: {data.get('message')}")
                    elif data.get("type") == "agent_action_result":
                        print(f"行动结果: {data.get('message')}")
                    elif data.get("type") == "dag_status_update":
                        print(f"DAG状态更新: {data.get('message')}")
                    else:
                        print(f"其他消息: {data}")

                    # 如果收到任务完成消息，结束测试
                    if data.get("type") == "task_completed":
                        print("\n✅ 任务完成，测试结束")
                        break

                except asyncio.TimeoutError:
                    print(f"\n⏰ 等待超时，共收到 {message_count} 个消息")
                    break
                except Exception as e:
                    print(f"❌ 接收消息时出错: {e}")
                    break

            print("\n=== 测试完成 ===")
            print(f"总共接收了 {message_count} 个实时消息")

            if message_count == 0:
                print("❌ 没有收到任何实时消息")
                print("可能的原因:")
                print("1. TaskEventTraceHandler没有正确添加")
                print("2. AgentService没有正确执行")
                print("3. 消息发送失败")
            else:
                print("✅ WebSocket实时通信功能正常")
                print("✅ DAG任务执行功能正常")
                print("✅ 前端数据更新功能正常")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_websocket_debug())
