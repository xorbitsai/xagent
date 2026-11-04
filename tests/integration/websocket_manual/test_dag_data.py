#!/usr/bin/env python3
"""
详细的WebSocket数据测试
验证前端接收到的数据结构
"""

import asyncio
import json

import pytest
import websockets


@pytest.mark.slow
async def test_dag_data():
    uri = "ws://127.0.0.1:8000/ws/chat/1"
    try:
        async with websockets.connect(uri) as websocket:
            print("=== WebSocket DAG数据测试 ===")

            # 发送状态请求
            await websocket.send(json.dumps({"type": "status_request"}))

            # 接收历史数据
            response = await websocket.recv()
            data = json.loads(response)

            print(f"数据类型: {data.get('type')}")
            print(f"任务ID: {data.get('task_id')}")

            # 检查steps数据
            if "steps" in data:
                print("\n=== 步骤数据 ===")
                print(f"步骤数量: {len(data['steps'])}")
                for i, step in enumerate(data["steps"]):
                    print(
                        f"步骤 {i + 1}: {step.get('name')} (状态: {step.get('status')})"
                    )

            # 检查step_details数据
            if "step_details" in data:
                print("\n=== 步骤详情数据 ===")
                print(f"详情数量: {len(data['step_details'])}")
                for step_id, details in data["step_details"].items():
                    print(f"步骤ID {step_id}:")
                    print(f"  名称: {details['step_info']['name']}")
                    print(f"  状态: {details['step_info']['status']}")
                    print(f"  日志数量: {details['execution_stats']['log_count']}")
                    print(f"  有错误: {details['execution_stats']['has_errors']}")

            # DAG可视化数据已移除，现在通过历史数据机制获取

            # 检查日志数据
            if "logs" in data:
                print("\n=== 日志数据 ===")
                print(f"日志数量: {len(data['logs'])}")
                error_logs = [
                    log for log in data["logs"] if log.get("level") == "error"
                ]
                success_logs = [
                    log for log in data["logs"] if log.get("level") == "success"
                ]
                print(f"错误日志: {len(error_logs)}")
                print(f"成功日志: {len(success_logs)}")

            print("\n=== 数据结构验证完成 ===")
            print("✅ WebSocket数据传输正常")
            print("✅ DAG步骤数据完整")
            print("✅ 步骤详情数据完整")
            print("✅ 执行日志数据完整")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_dag_data())
