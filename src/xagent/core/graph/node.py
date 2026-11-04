from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union, cast

from pydantic import BaseModel, Field, field_validator

from .node_factory import NodeFactory


class Node(BaseModel):
    id: str


class InputOverride(BaseModel):
    source: Literal["state_args"]
    value: str


class InputSchema(BaseModel):
    input_schema: str | None = None
    input_overrides: Optional[dict[str, InputOverride]] = None


class OutputSchema(BaseModel):
    output_schema: str | None = None


class BoundSchema(BaseModel):
    bound_schema: str | None = None


class WithNextIds(ABC):
    @abstractmethod
    def get_node_ids(self) -> list[str]: ...


class SingleDestNode(Node, WithNextIds):
    next: str

    def get_node_ids(self) -> list[str]:
        return [self.next]


class MultiDestNode(Node, WithNextIds):
    next: Union[str, list[str]]

    @field_validator("next", mode="before")
    @classmethod
    def prevent_char_iteration(cls, v: Any) -> Any:
        # If input is a string, wrap it in a list immediately.
        # This prevents Pydantic from iterating "END" into ['E', 'N', 'D']
        if isinstance(v, str):
            return [v]
        return v

    def get_node_ids(self) -> list[str]:
        # Always a list due to the @field_validator wrapping strings
        return cast(list[str], self.next)


class CondDestNode(Node, WithNextIds):
    next: list[dict[str, Any]]

    def get_node_ids(self) -> list[str]:
        """Return a list of node IDs from the next rules."""
        node_ids: list[str] = []
        for rule in self.next:
            node_id = rule.get("default") or rule.get("branch")
            if isinstance(node_id, str):
                node_ids.append(node_id)
        return node_ids


class DeterministicEdge:
    pass


class Parallelizable:
    """Marker class for nodes that can be executed in parallel.

    Nodes that inherit from this class can be used as targets in parallel execution
    patterns such as map and fork operations.

    """

    pass


@NodeFactory.register("start")
class StartNode(MultiDestNode, OutputSchema, DeterministicEdge):
    id: str = "START"


@NodeFactory.register("end")
class EndNode(Node, InputSchema):
    id: str = "END"


class BaseAgent(OutputSchema):
    model_id: str
    temperature: float | None = Field(default=None, ge=0, le=2)
    system_prompt: str


@NodeFactory.register("agent")
class AgentNode(
    BaseAgent, MultiDestNode, InputSchema, DeterministicEdge, Parallelizable
):
    ai_state_args: Optional[dict[str, str]] = None


@NodeFactory.register("tool_agent")
class ToolAgentNode(AgentNode, InputSchema, Parallelizable):
    tools: list[
        tuple[str, Optional[dict[str, str]]]  # tuple is ("tool name", tool_init_info)
    ]
    max_tool_call: int = 5
    need_summary: bool = False
    summary_prompt: Optional[str] = None
    summary_exclude_tool: list[str] = []


@NodeFactory.register("tool_conditional")
class ToolConditionalNode(CondDestNode, InputSchema, OutputSchema):
    """A node that executes a tool and routes based on the result using a custom router function."""

    tool_name: str
    tool_args: Optional[dict[str, Any]] = None  # Static arguments from YAML
    state_args: Optional[dict[str, str]] = (
        None  # Argument mappings from state (key: state_path)
    )
    state_rets: Optional[dict[str, str]] = (
        None  # Argument mappings from state (key: state_path)
    )
    init_info: Optional[dict[str, str]] = None  # Tool initialization information
    include_message_output: bool = (
        True  # Whether this tool's output should be passed to AI messages
    )
    include_tool_output: bool = (
        True  # Whether this tool's output should be passed to other tools
    )


@NodeFactory.register("agent_conditional")
class AgentConditionalNode(BaseAgent, MultiDestNode, InputSchema):
    ai_state_args: Optional[dict[str, str]] = None


@NodeFactory.register("tool")
class ToolNode(
    MultiDestNode, InputSchema, OutputSchema, DeterministicEdge, Parallelizable
):
    """Node that executes a single tool with arguments from YAML and/or upstream state."""

    tool_name: str
    init_info: Optional[dict[str, str]] = None  # Tool initialization information
    include_message_output: bool = (
        True  # Whether this tool's output should be passed to AI messages
    )


@NodeFactory.register("embedding")
class EmbeddingNode(MultiDestNode, DeterministicEdge, Parallelizable):
    """Node that generates embeddings from state messages using an embedding model."""

    model_id: str
    input_field: str = "messages.-1.content"  # State path to extract input text


@NodeFactory.register("dashscope_rerank")
class DashScopeRerankNode(MultiDestNode, DeterministicEdge, Parallelizable):
    """Node that reranks documents using DashScope Rerank API."""

    model_id: str  # For API key/config management via model hub
    input_field: str = "messages.-1.content"  # State path to extract documents list
    query_field: Optional[str] = (
        None  # State path to query for relevance scoring; if None, uses first document from input
    )
    top_k: int = -1  # Number of top documents to return; -1 returns all documents


@NodeFactory.register("structured_agent")
class StructuredAgentNode(
    SingleDestNode, BaseAgent, InputSchema, OutputSchema, DeterministicEdge
):
    pass


@NodeFactory.register("map")
class MapNode(MultiDestNode, DeterministicEdge, InputSchema, OutputSchema, BoundSchema):
    node: dict[str, Any]
    chunk_size: int = Field(default=1, gt=0)
