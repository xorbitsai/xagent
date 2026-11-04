class GraphValidationError(Exception):
    """Base class for graph validation errors."""


class DuplicateNodeError(GraphValidationError):
    def __init__(self, node_ids: list[str]):
        super().__init__(f"Duplicate node IDs: {node_ids}")


class MissingStartNodeError(GraphValidationError):
    """Raised when no start node is found."""


class DuplicatedStartNodeError(GraphValidationError):
    """Raised when more than 1 start nodes are found."""


class MissingEndNodeError(GraphValidationError):
    """Raised when no end node is found."""


class DanglingNodeError(GraphValidationError):
    """Raised when dangling nodes are found."""

    def __init__(self, node_ids: list[str]):
        super().__init__(f"Dangling nodes' IDs: {node_ids}")


class InvalidSchemaIDError(GraphValidationError):
    def __init__(self, schema_id: str, available_schema_ids: list[str]):
        super().__init__(
            f"Invalid schema id: {schema_id}, available: {available_schema_ids}"
        )


class MismatchInputOutputError(GraphValidationError):
    def __init__(self, node_id: str, message: str):
        super().__init__(f"Node '{node_id}': {message}")


class InvalidInputOutputPathError(GraphValidationError):
    def __init__(self, path: str):
        super().__init__(f"Invalid path: {path}")


class NonParallelizableNodeError(GraphValidationError):
    """Raised when the node is not parallelizable."""

    from .node import Node

    def __init__(self, node: Node) -> None:
        super().__init__(
            f"Non parallelizable node '{node.id}', type: {type(node).__name__}"
        )
