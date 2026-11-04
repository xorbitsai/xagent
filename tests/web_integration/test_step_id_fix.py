#!/usr/bin/env python3

"""
测试历史数据中step_id的修复
"""

import asyncio
import json
from datetime import datetime

import pytest
import websockets


@pytest.mark.integration
@pytest.mark.slow
async def test_historical_step_id_fix():
    """测试历史数据中step_id的修复"""
    uri = "ws://localhost:8000/ws/chat/1001"

    try:
        print(f"连接到WebSocket: {uri}")
        async with websockets.connect(uri) as websocket:
            # 发送测试消息
            test_message = {
                "type": "chat",
                "message": "2 * 3 = ?",
                "context": {},
            }

            print(f"发送测试消息: {test_message['message']}")
            await websocket.send(json.dumps(test_message))

            # 接收响应
            step_id_found = False
            step_events_with_id = []

            start_time = datetime.now()
            timeout = 30  # 30秒超时

            while (datetime.now() - start_time).seconds < timeout:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    data = json.loads(message)

                    if data.get("type") == "trace_event":
                        # 检查是否包含step_id
                        if data.get("step_id"):
                            step_id_found = True
                            step_events_with_id.append(
                                {
                                    "event_type": data.get("event_type"),
                                    "step_id": data.get("step_id"),
                                    "has_step_name": "step_name"
                                    in data.get("data", {}),
                                }
                            )
                            print(
                                f"✅ 找到带step_id的事件: {data.get('event_type')} (step_id: {data.get('step_id')})"
                            )

                        # 特别检查dag_step_start事件
                        if data.get("event_type") == "dag_step_start":
                            if data.get("step_id"):
                                print(
                                    f"✅ dag_step_start事件包含step_id: {data.get('step_id')}"
                                )
                            else:
                                print("❌ dag_step_start事件缺少step_id")

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
            print("历史数据step_id修复验证结果:")

            print(f"找到带step_id的事件数量: {len(step_events_with_id)}")
            for event in step_events_with_id:
                print(
                    f"  - {event['event_type']}: step_id={event['step_id']}, step_name={event['has_step_name']}"
                )

            if step_id_found:
                print("✅ 历史数据step_id修复成功")
                return True
            else:
                print("❌ 历史数据中仍然没有step_id")
                return False

    except Exception as e:
        print(f"测试失败: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_historical_step_id_fix())
    exit(0 if success else 1)
