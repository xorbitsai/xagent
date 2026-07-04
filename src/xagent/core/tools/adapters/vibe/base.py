from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Type, runtime_checkable

from pydantic import BaseModel


class ToolVisibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    INTERNAL = "internal"


class ToolCategory(str, Enum):
    """Centralized enum for tool categories.

    All tools should declare their category using this enum to ensure
    consistency across the codebase and enable auto-discovery in the UI.
    """

    VISION = "vision"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    KNOWLEDGE = "knowledge"
    FILE = "file"
    WEB_SEARCH = "web_search"
    BASIC = "basic"
    BROWSER = "browser"
    PPT = "ppt"
    AGENT = "agent"
    MCP = "mcp"
    DATABASE = "database"
    SKILL = "skill"
    OTHER = "other"


class ToolMetadata(BaseModel):
    name: str
    description: Optional[str] = None
    tags: list[str] = []
    visibility: ToolVisibility = ToolVisibility.PRIVATE
    allow_users: Optional[list[str]] = None  # Explicitly allowed user IDs
    has_state: bool = False
    category: ToolCategory = ToolCategory.OTHER  # Default category
    # Normalized identity of the MCP server / Custom-API config a tool was
    # generated from (via ``normalize_mcp_server_name``), or ``None`` for
    # tools with no server origin. Set once at generation so server-scoped
    # selection (``ToolSelectionSpec.compute_allowed_names``) matches by
    # structured equality instead of re-parsing the tool name.
    source_server: Optional[str] = None
    # Optional logical group used by generic agent-control heuristics that
    # should treat related tools as one repeated action stream.
    decision_group: Optional[str] = None
    # Concurrent-execution semantics (consumed by the ReAct in-turn scheduler).
    # ``read_only`` is the stronger declaration: the tool is guaranteed not to
    # write the filesystem / DB / any shared state. ``concurrency_safe`` is the
    # only field the scheduler reads: it is safe to run this tool concurrently
    # with other concurrency-safe tools in the same turn (idempotent, no
    # process-global side effects, independent of other tools' results in the
    # same turn). ``read_only`` implies ``concurrency_safe`` (enforced where the
    # metadata is constructed).
    read_only: bool = False
    concurrency_safe: bool = False


@runtime_checkable
class Tool(Protocol):
    @property
    def metadata(self) -> ToolMetadata: ...

    def args_type(self) -> Type[BaseModel]: ...
    def return_type(self) -> Type[BaseModel]: ...
    def state_type(self) -> Optional[Type[BaseModel]]: ...

    def is_async(self) -> bool: ...

    def return_value_as_string(self, value: Any) -> str: ...

    async def run_json_async(self, args: Mapping[str, Any]) -> Any: ...
    def run_json_sync(self, args: Mapping[str, Any]) -> Any: ...

    async def save_state_json(self) -> Mapping[str, Any]: ...
    async def load_state_json(self, state: Mapping[str, Any]) -> None: ...


class AbstractBaseTool(ABC, Tool):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    def tags(self) -> list[str]:
        return []

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description=self.description,
            tags=self.tags,
            visibility=getattr(self, "_visibility", ToolVisibility.PRIVATE),
            allow_users=getattr(self, "_allow_users", None),
            has_state=self.state_type() is not None,
            category=getattr(self, "category", ToolCategory.OTHER),
            source_server=getattr(self, "source_server", None),
            decision_group=getattr(self, "decision_group", None),
            read_only=getattr(self, "read_only", False),
            # read_only implies concurrency_safe; fall back to the explicit flag.
            concurrency_safe=(
                getattr(self, "concurrency_safe", False)
                or getattr(self, "read_only", False)
            ),
        )

    @abstractmethod
    def args_type(self) -> Type[BaseModel]: ...

    @abstractmethod
    def return_type(self) -> Type[BaseModel]: ...

    def state_type(self) -> Optional[Type[BaseModel]]:
        return None

    def return_value_as_string(self, value: Any) -> str:
        return str(value)

    def is_async(self) -> bool:
        return callable(getattr(self, "run_json_async", None))

    @abstractmethod
    def run_json_sync(self, args: Mapping[str, Any]) -> Any: ...

    @abstractmethod
    async def run_json_async(self, args: Mapping[str, Any]) -> Any: ...

    async def save_state_json(self) -> Mapping[str, Any]:
        return {}

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        pass

    async def setup(self, task_id: Optional[str] = None) -> None:
        """
        Setup method called when a task starts.

        Override this method to initialize resources (connections, sessions, etc.)
        for the duration of a task. This is called once per task execution.

        Args:
            task_id: Optional task identifier for resource tracking
        """
        pass

    async def teardown(self, task_id: Optional[str] = None) -> None:
        """
        Teardown method called when a task completes.

        Override this method to clean up resources (close connections, release sessions, etc.)
        that were created during setup. This is called once per task completion,
        even if the task failed or was interrupted.

        Args:
            task_id: Optional task identifier for resource tracking
        """
        pass
