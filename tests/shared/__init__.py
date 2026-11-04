"""
共享测试组件 - 提供通用的mock类和测试工具
避免在多个测试文件中重复定义相同的组件
"""

import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from xagent.core.agent.trace import (
    ACTION_END_LLM,
    ACTION_END_TOOL,
    ACTION_START_LLM,
    ACTION_START_TOOL,
    STEP_END_DAG,
    STEP_END_REACT,
    STEP_START_DAG,
    STEP_START_REACT,
    TASK_END_DAG,
    TASK_END_REACT,
    TASK_START_DAG,
    TASK_START_REACT,
    TraceEvent,
    TraceEventType,
    TraceHandler,
    Tracer,
)
from xagent.core.memory.base import MemoryResponse, MemoryStore
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.tools.adapters.vibe import Tool, ToolMetadata
from xagent.core.workspace import TaskWorkspace


class SharedMockLLM(BaseLLM):
    """共享的Mock LLM类"""

    def __init__(self, responses: Optional[List[str]] = None):
        self.call_count = 0
        self.responses = responses or [
            # Plan generation response
            """[
                {
                    "id": "step1",
                    "name": "get_weather",
                    "description": "Check the weather in a city",
                    "tool_name": "get_weather",
                    "tool_args": {"city": "Singapore", "date": "today"},
                    "dependencies": []
                },
                {
                    "id": "step2",
                    "name": "analyze_weather",
                    "description": "Analyze the weather data",
                    "tool_name": "analyze_data",
                    "tool_args": {"data": "weather_data"},
                    "dependencies": ["step1"]
                }
            ]""",
            # Goal achievement check response
            '{"achieved": true, "reason": "Weather analysis completed successfully"}',
            # ReAct responses
            "I need to check the weather in Singapore first.",
            "Now I'll analyze the weather data I collected.",
            "Task completed successfully.",
        ]
        self._model_name = "shared_mock_llm"

    @property
    def abilities(self) -> List[str]:
        return ["chat"]

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        else:
            return "Task completed successfully."


class SharedMockWeatherTool(Tool):
    """共享的Mock天气工具"""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name="get_weather", description="Mock weather tool")

    def args_type(self):
        return dict

    def return_type(self):
        return dict

    def state_type(self):
        return None

    def is_async(self):
        return True

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        return {
            "forecast": "sunny",
            "city": args.get("city", "Unknown"),
            "temperature": 28,
        }

    def run_json_sync(self, args: dict[str, Any]) -> Any:
        return {"forecast": "sunny", "city": args.get("city", "Unknown")}

    async def save_state_json(self):
        return {}

    async def load_state_json(self, state: dict[str, Any]):
        pass

    def return_value_as_string(self, value: Any) -> str:
        return str(value)


class SharedMockAnalysisTool(Tool):
    """共享的Mock分析工具"""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name="analyze_data", description="Mock analysis tool")

    def args_type(self):
        return dict

    def return_type(self):
        return dict

    def state_type(self):
        return None

    def is_async(self):
        return True

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        return {"analysis": "Weather is favorable", "recommendation": "Go outside"}

    def run_json_sync(self, args: dict[str, Any]) -> Any:
        return {"analysis": "Weather is favorable", "recommendation": "Go outside"}

    async def save_state_json(self):
        return {}

    async def load_state_json(self, state: dict[str, Any]):
        pass

    def return_value_as_string(self, value: Any) -> str:
        return str(value)


class SharedDummyMemoryStore(MemoryStore):
    """共享的虚拟内存存储"""

    def add(self, note):
        return MemoryResponse(success=True)

    def get(self, note_id: str):
        return MemoryResponse(success=True)

    def update(self, note):
        return MemoryResponse(success=True)

    def delete(self, note_id: str):
        return MemoryResponse(success=True)

    def search(self, query: str, k: int = 5, filters=None):
        return []

    def list_all(self, filters=None):
        return []

    def get_stats(self):
        return {
            "total_count": 0,
            "category_counts": {},
            "tag_counts": {},
            "memory_store_type": "SharedDummyMemoryStore",
        }

    def clear(self):
        pass


class CaptureTraceHandler(TraceHandler):
    """捕获追踪事件的处理器"""

    def __init__(self):
        self.events: List[TraceEvent] = []

    async def handle_event(self, event: TraceEvent) -> None:
        self.events.append(event)

    def clear(self):
        """清除所有捕获的事件"""
        self.events.clear()

    def get_events_by_type(self, event_type: TraceEventType) -> List[TraceEvent]:
        """根据事件类型获取事件"""
        return [event for event in self.events if event.event_type == event_type]

    def get_events_with_task_id(self, task_id: str) -> List[TraceEvent]:
        """根据task_id获取事件"""
        return [event for event in self.events if event.task_id == task_id]


class MockDatabaseSession:
    """模拟数据库会话"""

    def __init__(self):
        self.add = MagicMock()
        self.commit = MagicMock()
        self.rollback = MagicMock()
        self.query = MagicMock()
        self.query.return_value.filter.return_value.first.return_value = None
        self.add_call_count = 0
        # Make add callable increment counter
        self.add.side_effect = self._track_add

    def _track_add(self, item):
        """Track add calls"""
        self.add_call_count += 1

    def close(self):
        """模拟数据库会话关闭"""
        pass


class EventOwnershipValidator:
    """事件归属验证器"""

    def __init__(self):
        self.events: List[TraceEvent] = []
        self.ownership_issues: List[str] = []

    def add_event(self, event: TraceEvent):
        """添加事件到验证器"""
        self.events.append(event)

    def validate_event_ownership(self) -> bool:
        """验证所有事件都正确归属于plan或step"""
        self.ownership_issues.clear()

        for event in self.events:
            # 基础验证：所有事件都必须有task_id
            if not event.task_id:
                self.ownership_issues.append(
                    f"Event {event.id} ({event.event_type.value}) missing task_id"
                )
                continue

            # 根据事件类型进行特定验证
            if self._is_plan_level_event(event):
                self._validate_plan_event_ownership(event)

            elif self._is_step_level_event(event):
                self._validate_step_event_ownership(event)

            elif self._is_llm_or_tool_event(event):
                self._validate_llm_tool_event_ownership(event)

        return len(self.ownership_issues) == 0

    def _is_plan_level_event(self, event: TraceEvent) -> bool:
        """判断是否为plan级别事件"""
        plan_events = [
            TASK_START_DAG,
            TASK_END_DAG,
            TASK_START_REACT,
            TASK_END_REACT,
        ]
        return event.event_type in plan_events

    def _is_step_level_event(self, event: TraceEvent) -> bool:
        """判断是否为step级别事件"""
        step_events = [
            STEP_START_DAG,
            STEP_END_DAG,
            STEP_START_REACT,
            STEP_END_REACT,
        ]
        return event.event_type in step_events

    def _is_llm_or_tool_event(self, event: TraceEvent) -> bool:
        """判断是否为LLM或工具事件"""
        llm_tool_events = [
            ACTION_START_LLM,
            ACTION_END_LLM,
            ACTION_START_TOOL,
            ACTION_END_TOOL,
        ]
        return event.event_type in llm_tool_events

    def _validate_plan_event_ownership(self, event: TraceEvent) -> bool:
        """验证plan事件归属"""
        if not event.task_id:
            self.ownership_issues.append(
                f"Plan event {event.id} ({event.event_type.value}) missing task_id"
            )
            return False
        return True

    def _validate_step_event_ownership(self, event: TraceEvent) -> bool:
        """验证step事件归属"""
        if not event.task_id:
            self.ownership_issues.append(
                f"Step event {event.id} ({event.event_type.value}) missing task_id"
            )
            return False

        # Step事件应该有step信息
        if not event.step_id:
            self.ownership_issues.append(
                f"Step event {event.id} ({event.event_type.value}) missing step_id"
            )
            return False

        return True

    def _validate_llm_tool_event_ownership(self, event: TraceEvent) -> bool:
        """验证LLM或工具事件归属"""
        if not event.task_id:
            self.ownership_issues.append(
                f"LLM/Tool event {event.id} ({event.event_type.value}) missing task_id"
            )
            return False

        # LLM/工具事件必须有parent_id
        if not event.parent_id:
            self.ownership_issues.append(
                f"LLM/Tool event {event.id} ({event.event_type.value}) missing parent_id"
            )
            return False

        # 验证parent_id确实指向一个step事件
        parent_event = self._find_event_by_id(event.parent_id)
        if not parent_event or not self._is_step_level_event(parent_event):
            self.ownership_issues.append(
                f"LLM/Tool event {event.id} ({event.event_type.value}) has invalid parent_id: {event.parent_id}"
            )
            return False

        return True

    def _find_event_by_id(self, event_id: str) -> Optional[TraceEvent]:
        """根据ID查找事件"""
        for event in self.events:
            if event.id == event_id:
                return event
        return None

    def get_ownership_report(self) -> Dict[str, Any]:
        """获取归属验证报告"""
        # 执行验证
        self.validate_event_ownership()

        total_events = len(self.events)
        events_with_task_id = sum(1 for e in self.events if e.task_id)
        events_with_parent_id = sum(1 for e in self.events if e.parent_id)

        return {
            "total_events": total_events,
            "events_with_task_id": events_with_task_id,
            "events_with_parent_id": events_with_parent_id,
            "ownership_issues": self.ownership_issues,
            "validation_passed": len(self.ownership_issues) == 0,
            "coverage_percentage": (events_with_task_id / total_events * 100)
            if total_events > 0
            else 0,
        }


def create_test_tracer() -> Tracer:
    """创建测试用的追踪器"""
    tracer = Tracer()
    # 移除默认的console handler，避免测试时产生过多日志
    tracer.handlers.clear()
    return tracer


def create_mock_db_session():
    """
    创建模拟数据库会话和正确的patch配置.

    Returns:
        tuple: (mock_db_session, mock_get_db_generator)
            mock_db_session: The mock database session
            mock_get_db_generator: A generator function that yields the mock session
    """
    mock_db = MockDatabaseSession()

    def mock_get_db_generator():
        """Mock generator that yields the mock database session"""
        yield mock_db

    return mock_db, mock_get_db_generator


def create_test_components():
    """创建测试所需的所有组件"""
    llm = SharedMockLLM()
    memory = SharedDummyMemoryStore()
    tools = [SharedMockWeatherTool(), SharedMockAnalysisTool()]
    tracer = create_test_tracer()
    # Create a temporary workspace for testing
    temp_dir = tempfile.mkdtemp()
    workspace = TaskWorkspace(id="test_workspace", base_dir=temp_dir)

    return {
        "llm": llm,
        "memory": memory,
        "tools": tools,
        "tracer": tracer,
        "workspace": workspace,
        "temp_dir": temp_dir,
    }
