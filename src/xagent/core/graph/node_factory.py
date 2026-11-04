from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type, TypeVar

if TYPE_CHECKING:
    from .node import Node

T = TypeVar("T", bound="Node")


class NodeFactory:
    """Factory for creating nodes from type strings with automatic registration."""

    _registry: Dict[str, Type["Node"]] = {}

    @classmethod
    def register(cls, node_type: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a node class with the factory.

        Args:
            node_type: The string identifier for this node type

        Returns:
            Decorator function that registers the class
        """

        def decorator(node_class: Type[T]) -> Type[T]:
            cls._registry[node_type] = node_class
            return node_class

        return decorator

    @classmethod
    def create_node(cls, node_type: str, **kwargs: Any) -> Optional["Node"]:
        """Create a node instance from its type string.

        Args:
            node_type: The string identifier for the node type
            **kwargs: Arguments to pass to the node constructor

        Returns:
            Node: Instance of the requested node type

        Raises:
            ValueError: If node_type is not registered
        """
        if node_type not in cls._registry:
            return None
        return cls._registry[node_type](**kwargs)

    @classmethod
    def get_registered_types(cls) -> list[str]:
        """Get all registered node types.

        Returns:
            List of registered node type strings
        """
        return list(cls._registry.keys())
