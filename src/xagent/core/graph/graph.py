from enum import Enum
from typing import Any, Callable

from jsonschema import Draft7Validator, SchemaError  # type: ignore[import-untyped]
from pydantic import BaseModel, field_validator

from .error import (
    DanglingNodeError,
    DuplicatedStartNodeError,
    DuplicateNodeError,
    MissingEndNodeError,
    MissingStartNodeError,
)
from .io_validation import get_validate_input_output_fn
from .node import (
    BaseAgent,
    CondDestNode,
    DeterministicEdge,
    EndNode,
    MultiDestNode,
    Node,
    SingleDestNode,
    StartNode,
)
from .schema import SchemaParser


class Graph(BaseModel):
    graph_id: str = "main"
    nodes: list[Node]
    schemas: dict[str, dict[str, Any]] = {}
    # Do not force DAG execution if enabled
    allow_cycles: bool = False

    @field_validator("schemas", mode="before")
    @classmethod
    def parse_schemas(
        cls, schemas: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Validate schemas using SchemaParser."""
        for schema_id, schema in schemas.items():
            try:
                Draft7Validator.check_schema(schema)
            except SchemaError as e:
                raise SchemaError(f"Invalid JSON schema for '{schema_id}': {e.message}")

        parser = SchemaParser(schemas)
        return parser.schemas

    _node_map: dict[str, Node] = {}

    def model_post_init(self, context: Any) -> None:
        self._node_map = {node.id: node for node in self.nodes}

    def node_map(self) -> dict[str, Node]:
        return {node.id: node for node in self.nodes}

    def get_duplicated_node_ids(self) -> list[str]:
        """Get the list of duplicated node IDs in the graph.

        Returns:
            list[str]: List of duplicated node IDs (empty if none).
        """
        node_ids = [node.id for node in self.nodes]
        return [id for id in set(node_ids) if node_ids.count(id) > 1]

    def _traverse_graph(
        self, visitor_func: Callable[[Node, Any], Any], initial_data: Any = None
    ) -> Any:
        """Generic graph traversal function with custom visitor logic.

        Args:
            visitor_func: Function called for each node. Should return updated data.
            initial_data: Initial data passed to visitor function.

        Returns:
            Final data after traversal.
        """
        visited: set[str] = set()
        if not self.nodes:
            return visited

        start_nodes = [node for node in self.nodes if isinstance(node, StartNode)]
        if not start_nodes:
            return visited

        stack: list[tuple[Node, Any]] = [(start_nodes[0], initial_data)]

        while stack:
            current, data = stack.pop()
            if current.id in visited:
                continue
            visited.add(current.id)

            data = visitor_func(current, data)

            if isinstance(current, SingleDestNode):
                if current.next and current.next in self.node_map():
                    node = self.node_map()[current.next]
                    stack.append((node, data))
            elif isinstance(current, MultiDestNode):
                if current.next:
                    for node_id in current.next:
                        if node_id and node_id in self.node_map():
                            node = self.node_map()[node_id]
                            stack.append((node, data))
            elif isinstance(current, CondDestNode):
                for rule in current.next:
                    node_id = None
                    if "default" in rule:
                        node_id = rule["default"]
                    elif "branch" in rule:
                        node_id = rule["branch"]
                    if node_id and node_id in self.node_map():
                        node = self.node_map()[node_id]
                        stack.append((node, data))

        return visited

    def get_dangling_nodes(self) -> list[str]:
        """Get nodes not connected to the main graph (from StartNode to EndNode)."""
        visited = self._traverse_graph(
            lambda *args, **kwargs: (args, kwargs) and None, None
        )
        return [node.id for node in self.nodes if node.id not in visited]

    def validate_input_output(self) -> None:
        self._traverse_graph(get_validate_input_output_fn(self.schemas), {})

    def validate_schemas(self) -> None:
        """Validate that all schemas conform to JSON Schema format.

        Raises:
            jsonschema.SchemaError: If any schema is invalid.
        """
        import jsonschema
        from jsonschema import Draft7Validator

        for schema_id, schema in self.schemas.items():
            try:
                Draft7Validator.check_schema(schema)
            except jsonschema.SchemaError as e:
                raise jsonschema.SchemaError(
                    f"Invalid JSON schema for '{schema_id}': {e.message}"
                )

    def validate_graph(self) -> None:
        """Validate the graph and raise errors if issues are found.

        Raises:
            DuplicateNodeError: If duplicate node IDs exist.
            MissingStartNodeError: If no start node exists.
            DuplicatedStartNodeError: If more than 1 start nodes exist.
            MissingEndNodeError: If no end node exists.
            DanglingNodeError: If dangling nodes exist.
        """
        GraphValidator(self).validate()

        self.validate_input_output()
        self.validate_schemas()

    def get_start_node(self) -> StartNode:
        """Return the start node of the graph.

        Returns:
            StartNode: The start node of the graph.

        Raises:
            MissingStartNodeError: If no start node exists.
        """
        start_nodes = [node for node in self.nodes if isinstance(node, StartNode)]
        if not start_nodes:
            raise MissingStartNodeError()
        return start_nodes[0]

    def get_next_nodes(self, current: Node) -> list[Node]:
        """Return the next nodes of the given `current` node.

        Args:
            current: The node whose next nodes are to be retrieved.

        Returns:
            list[Node]: A list of next nodes. Empty if no next nodes exist.
        """
        next_nodes = []

        if isinstance(current, SingleDestNode):
            if current.next and current.next in self.node_map():
                next_nodes.append(self.node_map()[current.next])
        elif isinstance(current, MultiDestNode):
            if current.next:
                for node_id in current.next:
                    if node_id and node_id in self.node_map():
                        next_nodes.append(self.node_map()[node_id])
        elif isinstance(current, CondDestNode):
            for rule in current.next:
                node_id = None
                if "default" in rule:
                    node_id = rule["default"]
                elif "branch" in rule:
                    node_id = rule["branch"]
                if isinstance(node_id, str):
                    next_nodes.append(self.node_map()[node_id])

        return next_nodes

    def is_start_node(self, node: Node) -> bool:
        """Check if the given node is a start node.

        Args:
            node: The node to check.

        Returns:
            bool: True if the node is a StartNode, False otherwise.
        """
        return isinstance(node, StartNode)

    def is_end_node(self, node: Node) -> bool:
        """Check if the given node is an end node.

        Args:
            node: The node to check.

        Returns:
            bool: True if the node is an EndNode, False otherwise.
        """
        return isinstance(node, EndNode)

    def get_node_by_id(self, node_id: str) -> Node:
        """Return the node with the given ID.

        Args:
            node_id: The ID of the node to retrieve.

        Returns:
            Node: The node with the specified ID.

        Raises:
            KeyError: If the node ID is not found in the graph.
        """
        if node_id not in self.node_map():
            raise KeyError(f"Node with ID '{node_id}' not found in the graph.")
        return self.node_map()[node_id]


class StateMode(Enum):
    NONE = 0
    READ = 1
    WRITE = 2
    READWRITE = 3

    @classmethod
    def from_string(cls, value: str) -> "StateMode":
        """Parse StateMode from string (case-insensitive)."""
        try:
            return cls[value.upper()]
        except KeyError:
            valid_values = ", ".join(m.name for m in cls)
            raise ValueError(
                f"Invalid StateMode: '{value}'. Valid values: {valid_values}"
            )


class GraphNode(MultiDestNode, DeterministicEdge):
    graph: Graph
    compact: BaseAgent | None = None
    mode: StateMode = StateMode.READWRITE
    human_message: str | None = None

    @field_validator("mode", mode="before")
    @classmethod
    def parse_mode(cls, v: Any) -> StateMode:
        if isinstance(v, str):
            return StateMode.from_string(v)
        if isinstance(v, StateMode):
            return v
        if isinstance(v, int) and not isinstance(v, bool):
            return StateMode(v)
        raise ValueError(f"Cannot parse StateMode from {type(v).__name__}: {v}")


class GraphWalker:
    graph: Graph
    _node_map: dict[str, Node]

    def __init__(self, graph: Graph):
        self.graph = graph
        self._node_map = {node.id: node for node in graph.nodes}

    def walk(
        self, visitor_func: Callable[[Node, Any], Any], initial_data: Any = None
    ) -> Any:
        """Generic graph traversal function with custom visitor logic.

        Args:
            visitor_func: Function called for each node. Should return updated data.
            initial_data: Initial data passed to visitor function.

        Returns:
            Final data after traversal.
        """
        visited: set[str] = set()
        if not self._node_map:
            return visited

        start_nodes = [
            node for node in self._node_map.values() if isinstance(node, StartNode)
        ]
        if not start_nodes:
            return visited

        stack: list[tuple[Node, Any]] = [(start_nodes[0], initial_data)]

        while stack:
            current, data = stack.pop()
            if current.id in visited:
                continue
            visited.add(current.id)

            data = visitor_func(current, data)

            if isinstance(current, SingleDestNode):
                if current.next and current.next in self._node_map:
                    node = self._node_map[current.next]
                    stack.append((node, data))
            elif isinstance(current, MultiDestNode):
                for node_id in current.next:
                    if node_id and node_id in self._node_map:
                        node = self._node_map[node_id]
                        stack.append((node, data))
            elif isinstance(current, CondDestNode):
                for node_id in current.get_node_ids():
                    if node_id and node_id in self._node_map:
                        node = self._node_map[node_id]
                        stack.append((node, data))
        return visited


class GraphValidator:
    graph: GraphWalker
    _nodes: list[Node]
    _node_map: dict[str, Node]
    _schemas: dict[str, dict[str, Any]]

    def __init__(self, graph: Graph):
        self.graph = GraphWalker(graph)
        self._node_map = {node.id: node for node in graph.nodes}
        self._nodes = graph.nodes

    def _get_duplicated_node_ids(self) -> list[str]:
        """Get the list of duplicated node IDs in the graph.

        Returns:
            list[str]: List of duplicated node IDs (empty if none).
        """
        node_ids = [node.id for node in self._nodes]
        return [id for id in set(node_ids) if node_ids.count(id) > 1]

    def _get_dangling_nodes(self) -> list[str]:
        """Get nodes not connected to the main graph (from StartNode to EndNode)."""
        visited = self.graph.walk(lambda *args, **kwargs: (args, kwargs) and None, None)
        return [node.id for node in self._node_map.values() if node.id not in visited]

    def validate(self) -> None:
        """Validate the graph and raise errors if issues are found.

        Raises:
            DuplicateNodeError: If duplicate node IDs exist.
            MissingStartNodeError: If no start node exists.
            DuplicatedStartNodeError: If more than 1 start nodes exist.
            MissingEndNodeError: If no end node exists.
            DanglingNodeError: If dangling nodes exist.
        """
        # Check duplicate node IDs
        if duplicate_ids := self._get_duplicated_node_ids():
            raise DuplicateNodeError(duplicate_ids)

        # Check start/end nodes
        start_node_cnt = sum(
            1 for node in self._node_map.values() if isinstance(node, StartNode)
        )
        end_node_cnt = sum(
            1 for node in self._node_map.values() if isinstance(node, EndNode)
        )
        if start_node_cnt < 1:
            raise MissingStartNodeError()
        elif start_node_cnt > 1:
            raise DuplicatedStartNodeError()
        if end_node_cnt < 1:
            raise MissingEndNodeError()

        # Check dangling nodes
        if dangling_ids := self._get_dangling_nodes():
            raise DanglingNodeError(dangling_ids)
