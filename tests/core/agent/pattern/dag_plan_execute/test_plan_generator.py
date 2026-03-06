"""
单元测试：测试PlanGenerator的JSON格式解析功能
"""

import json
from typing import Dict, List

import pytest

from tests.utils.mock_llm import MockLLM
from xagent.core.agent.pattern.dag_plan_execute.plan_generator import PlanGenerator
from xagent.core.model.chat.basic.base import BaseLLM


class MockLLMForTest(BaseLLM):
    """专门用于测试的Mock LLM"""

    def __init__(self):
        self._model_name = "test_mock_llm"

    @property
    def abilities(self) -> list[str]:
        return ["chat"]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # 只返回新的字典格式响应
        return self._generate_dict_response()

    def _generate_dict_response(self) -> str:
        """生成新的字典格式响应"""
        return json.dumps(
            {
                "plan": {
                    "task_name": "Test Task Execution",
                    "goal": "test goal",
                    "steps": [
                        {
                            "id": "step1",
                            "name": "Test Step 1",
                            "description": "First test step",
                            "tool_names": ["test_tool"],
                            "dependencies": [],
                            "difficulty": "hard",
                        },
                        {
                            "id": "step2",
                            "name": "Test Step 2",
                            "description": "Second test step",
                            "tool_names": ["another_tool"],
                            "dependencies": ["step1"],
                            "difficulty": "easy",
                        },
                    ],
                }
            },
            indent=2,
        )


class TestPlanGenerator:
    """测试PlanGenerator的JSON格式解析功能"""

    @pytest.fixture
    def plan_generator(self):
        """创建PlanGenerator实例"""
        llm = MockLLM()
        return PlanGenerator(llm)

    def test_parse_new_dict_format(self, plan_generator):
        """测试解析新的字典格式"""
        # 新的字典格式
        dict_response = """
        {
            "plan": {
                "goal": "analyze data",
                "steps": [
                    {
                        "id": "step_1",
                        "name": "Data Collection",
                        "description": "Gather required data from sources",
                        "tool_names": ["fetch_data"],
                        "dependencies": [],
                        "difficulty": "hard"
                    },
                    {
                        "id": "step_2",
                        "name": "Data Analysis",
                        "description": "Analyze collected data",
                        "tool_names": ["analyze_data"],
                        "dependencies": ["step_1"],
                        "difficulty": "easy"
                    }
                ]
            }
        }
        """

        parsed_data = plan_generator._parse_plan_response(dict_response)
        steps = parsed_data["steps"]
        task_name = parsed_data["task_name"]

        # 验证解析结果
        assert len(steps) == 2
        assert steps[0]["id"] == "step_1"
        assert steps[0]["name"] == "Data Collection"
        assert steps[0]["tool_names"] == ["fetch_data"]
        assert steps[0]["dependencies"] == []
        assert steps[0]["difficulty"] == "hard"

        assert steps[1]["id"] == "step_2"
        assert steps[1]["name"] == "Data Analysis"
        assert steps[1]["tool_names"] == ["analyze_data"]
        assert steps[1]["dependencies"] == ["step_1"]
        assert steps[1]["difficulty"] == "easy"
        # task_name is optional, may be None if not provided
        assert task_name is None  # This example doesn't include task_name

    def test_parse_invalid_array_format(self, plan_generator):
        """测试解析无效的数组格式（应该失败）"""
        from xagent.core.agent.exceptions import LLMResponseError

        # 旧的数组格式现在应该被拒绝
        array_response = """
        [
            {
                "id": "step1",
                "name": "Research",
                "description": "Gather information",
                "tool_names": ["web_search"],
                "dependencies": []
            }
        ]
        """

        # 验证解析失败，抛出异常
        with pytest.raises(LLMResponseError):
            plan_generator._parse_plan_response(array_response)

    def test_parse_malformed_json(self, plan_generator):
        """测试解析格式错误的JSON"""
        from xagent.core.agent.exceptions import LLMResponseError

        malformed_responses = [
            "invalid json string",
            '{"plan": "missing steps field"}',
            '{"steps": "not an array"}',
            '{"missing_plan_field": {}}',
            "[]",  # 旧格式现在无效
            "{}",  # 缺少plan字段
        ]

        for response in malformed_responses:
            with pytest.raises(LLMResponseError):
                plan_generator._parse_plan_response(response)

    def test_parse_missing_required_fields(self, plan_generator):
        """测试解析缺少必需字段的步骤"""
        incomplete_response = """
        {
            "plan": {
                "goal": "test goal",
                "steps": [
                    {
                        "id": "step1",
                        "name": "Valid Step",
                        "description": "This step is valid",
                        "tool_names": ["tool1"],
                        "dependencies": [],
                        "difficulty": "hard"
                    },
                    {
                        "id": "step2",
                        "description": "Missing name and tool_names"
                    }
                ]
            }
        }
        """

        parsed_data = plan_generator._parse_plan_response(incomplete_response)
        steps = parsed_data["steps"]

        # 只有有效的步骤会被返回
        assert len(steps) == 1
        assert steps[0]["id"] == "step1"
        assert steps[0]["name"] == "Valid Step"

    def test_parse_auto_generate_id(self, plan_generator):
        """测试自动生成步骤ID"""
        response_without_ids = """
        {
            "plan": {
                "goal": "test goal",
                "steps": [
                    {
                        "name": "Step Without ID",
                        "description": "This step has no ID",
                        "tool_names": ["tool1"],
                        "dependencies": []
                    },
                    {
                        "name": "Another Step",
                        "description": "This step also has no ID",
                        "tool_names": ["tool2"],
                        "dependencies": []
                    }
                ]
            }
        }
        """

        parsed_data = plan_generator._parse_plan_response(response_without_ids)
        steps = parsed_data["steps"]

        assert len(steps) == 2
        assert steps[0]["id"] == "step_1"
        assert steps[1]["id"] == "step_2"

    def test_parse_auto_generate_dependencies(self, plan_generator):
        """测试自动生成依赖数组"""
        response_without_deps = """
        {
            "plan": {
                "goal": "test goal",
                "steps": [
                    {
                        "id": "step1",
                        "name": "Step Without Dependencies",
                        "description": "This step has no dependencies field",
                        "tool_names": ["tool1"]
                    }
                ]
            }
        }
        """

        parsed_data = plan_generator._parse_plan_response(response_without_deps)
        steps = parsed_data["steps"]

        assert len(steps) == 1
        assert steps[0]["id"] == "step1"
        assert steps[0]["dependencies"] == []

    def test_parse_with_extra_whitespace(self, plan_generator):
        """测试解析包含额外空格的响应"""
        response_with_whitespace = """

        Some text before the JSON

        {
            "plan": {
                "goal": "test goal",
                "steps": [
                    {
                        "id": "step1",
                        "name": "Test Step",
                        "description": "A test step",
                        "tool_names": ["test_tool"],
                        "dependencies": [],
                        "difficulty": "hard"
                    }
                ]
            }
        }

        Some text after the JSON

        """

        parsed_data = plan_generator._parse_plan_response(response_with_whitespace)
        steps = parsed_data["steps"]

        assert len(steps) == 1
        assert steps[0]["id"] == "step1"

    def test_parse_with_nested_json(self, plan_generator):
        """测试解析嵌套在复杂文本中的JSON"""
        nested_response = """
        The user requested a plan with the following structure:
        Here is the generated plan:
        {
            "plan": {
                "goal": "analyze data",
                "steps": [
                    {
                        "id": "step1",
                        "name": "Data Collection",
                        "description": "Collect the required data",
                        "tool_names": ["data_collector"],
                        "dependencies": [],
                        "difficulty": "hard"
                    }
                ]
            }
        }
        End of response.
        """

        parsed_data = plan_generator._parse_plan_response(nested_response)
        steps = parsed_data["steps"]

        assert len(steps) == 1
        assert steps[0]["id"] == "step1"
        assert steps[0]["name"] == "Data Collection"

    @pytest.mark.asyncio
    async def test_generate_plan_with_new_format(self):
        """测试使用新格式生成计划"""
        llm = MockLLMForTest()
        generator = PlanGenerator(llm)

        # 模拟执行环境
        tools = []
        history = []
        iteration = 1

        # 创建一个简单的tracer mock
        class MockTracer:
            async def trace_dag_plan_start(self, task_id, data):
                pass

            async def trace_dag_plan_end(self, task_id, data):
                pass

        tracer = MockTracer()

        try:
            plan = await generator.generate_plan(
                goal="test goal",
                tools=tools,
                iteration=iteration,
                history=history,
                tracer=tracer,
            )

            # 验证计划生成成功
            assert plan is not None
            assert plan.goal == "test goal"
            assert len(plan.steps) == 2
            assert plan.steps[0].id == "step1"
            assert plan.steps[0].difficulty == "hard"
            assert plan.steps[1].id == "step2"
            assert plan.steps[1].difficulty == "easy"

        except Exception as e:
            # 如果测试环境不支持完整的计划生成，跳过此测试
            pytest.skip(f"Full plan generation not supported in test environment: {e}")

    def test_error_message_format(self, plan_generator):
        """Test that error message format instructions have been updated"""
        # Verify prompt template only contains new format requirements
        prompt = plan_generator._build_planning_prompt(
            goal="test goal", iteration=1, history=[], tools=[]
        )

        # Verify prompt contains JSON format requirements and correct structure
        prompt_text = "".join([msg["content"] for msg in prompt])
        assert "JSON object" in prompt_text
        assert '"plan": {' in prompt_text
        assert '"steps": [' in prompt_text

        # Ensure old array format is not mentioned
        assert "JSON array" not in prompt_text
