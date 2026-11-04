#!/usr/bin/env python3

"""
更全面的WebSocket消息分类测试
"""

import asyncio
import json
from datetime import datetime

import websockets


async def comprehensive_test():
    """全面的WebSocket消息分类测试"""
    uri = "ws://localhost:8000/ws/chat/1001"

    try:
        print(f"连接到WebSocket: {uri}")
        async with websockets.connect(uri) as websocket:
            # 发送不同类型的测试消息
            test_messages = [
                {
                    "type": "chat",
                    "message": "请帮我分析一下当前的销售数据并生成报告",
                    "context": {},
                },
                {"type": "chat", "message": "搜索最新的AI技术发展趋势", "context": {}},
            ]

            for i, test_message in enumerate(test_messages):
                print(f"\n--- 测试 {i + 1}: {test_message['message'][:30]}... ---")
                await websocket.send(json.dumps(test_message))

                # 接收响应
                left_panel_messages = []
                right_panel_messages = []
                llm_messages = []

                start_time = datetime.now()
                timeout = 20  # 20秒超时

                while (datetime.now() - start_time).seconds < timeout:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                        data = json.loads(message)

                        # 模拟前端分类逻辑
                        if data.get("type") == "dag_status_update":
                            has_tool_data = (
                                data.get("tool_execution")
                                or data.get("llm_interaction")
                                or data.get("agent_thinking")
                                or data.get("agent_action")
                                or data.get("agent_action_result")
                                or data.get("agent_observation")
                                or data.get("agent_observation_result")
                                or data.get("agent_final_answer")
                                or data.get("agent_final_answer_start")
                            )

                            has_step_data = (
                                data.get("step_start")
                                or data.get("step_update")
                                or data.get("step_end")
                            )

                            is_plan_event = (
                                (
                                    data.get("plan_completed") is not None
                                    or data.get("execution_started") is not None
                                    or data.get("execution_completed") is not None
                                    or data.get("steps_count") is not None
                                )
                                and not has_step_data
                                and not has_tool_data
                                and not data.get("iteration_start")
                                and not data.get("iteration_end")
                            )

                            if is_plan_event:
                                left_panel_messages.append(data.get("message", ""))
                                print(f"  [左侧] {data.get('message', '')}")
                            else:
                                right_panel_messages.append(data.get("message", ""))
                                print(f"  [右侧] {data.get('message', '')}")

                                # 记录LLM交互
                                if data.get("llm_interaction"):
                                    llm_messages.append(data.get("message", ""))
                                    llm_info = data["llm_interaction"]
                                    if llm_info.get("type") == "request":
                                        print(
                                            f"    -> LLM请求: {llm_info.get('model', 'unknown')}"
                                        )
                                    elif llm_info.get("type") == "response":
                                        print(
                                            f"    -> LLM响应: {llm_info.get('result_type', 'unknown')}"
                                        )

                        elif data.get("type") == "task_completed":
                            print(
                                f"  [完成] 任务完成: {data.get('result', '')[:50]}..."
                            )
                            break

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"处理消息时出错: {e}")
                        continue

                # 输出本次测试的统计
                print(f"  测试 {i + 1} 结果:")
                print(f"    左侧面板消息: {len(left_panel_messages)} 条")
                print(f"    右侧面板消息: {len(right_panel_messages)} 条")
                print(f"    LLM交互消息: {len(llm_messages)} 条")

                # 检查是否有重复
                duplicates = set(left_panel_messages) & set(right_panel_messages)
                if duplicates:
                    print(f"    ❌ 发现重复消息: {duplicates}")
                else:
                    print("    ✅ 无重复消息")

                # 等待一下再进行下一个测试
                await asyncio.sleep(2)

            print(f"\n{'=' * 50}")
            print("全面测试完成!")

    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    asyncio.run(comprehensive_test())
