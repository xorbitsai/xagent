import logging
from typing import Any

from pydantic import ValidationError

from .graph import Graph, GraphNode
from .node import Node
from .node_factory import NodeFactory

logger = logging.getLogger(__name__)


def validate_tools(node_config: dict[str, Any]) -> None:
    """Validate and transform tools configuration in YAML data.

    Args:
        yaml_data: Parsed YAML data dictionary containing nodes

    Raises:
        ValueError: If tool validation fails
    """
    node_type = node_config.get("type")
    node_id = node_config.get("id", "unknown")

    if node_type == "tool_agent" and "tools" in node_config:
        tools = node_config["tools"]
        if not isinstance(tools, list):
            raise ValueError(f"Node '{node_id}': 'tools' must be a list")

        tools_list = []
        for tool in tools:
            if not isinstance(tool, dict):
                raise ValueError(f"Node '{node_id}': each tool must be a dictionary")

            if "name" not in tool:
                raise ValueError(
                    f"Node '{node_id}': each tool must have a 'name' field"
                )

            tool_name = tool["name"]
            if not isinstance(tool_name, str):
                raise ValueError(f"Node '{node_id}': tool 'name' must be a string")

            tool_init_info = tool.get("init_info")
            tools_list.append((tool_name, tool_init_info))

        # Update the node_config with transformed tools
        node_config["tools"] = tools_list


def parse_node(yaml_data: dict[str, Any], graphs: dict[str, Graph]) -> Node:
    node_id: str = yaml_data["id"]
    node_type: str = yaml_data["type"]

    logger.debug(f"parse node, graphs: {graphs}")

    try:
        # Makes it compatible with old naming
        if node_type == "agent_conditional":
            nexts = yaml_data.get("next")
            if nexts is not None:
                yaml_data["nexts"] = nexts

        # Builtin nodes
        validate_tools(yaml_data)
        node = NodeFactory.create_node(node_type, **yaml_data)
        if node is not None:
            return node
        # Sub-graphs
        if node_type in graphs:
            graph = graphs[node_type]
            next_id: str = yaml_data["next"]
            return GraphNode(
                id=node_id,
                graph=graph,
                next=next_id,
            )

        raise ValueError(f"Unknown node type: {node_type}")

    except (ValueError, ValidationError) as e:
        raise ValueError(f"Failed to create node {node_id}: {e}")


def parse_graphs(yaml_data: dict[str, Any]) -> dict[str, Graph]:
    graphs: dict[str, Graph] = {}

    # Pass 1: Create graph shells
    for graph_id, graph_data in yaml_data.items():
        logger.debug(f"graph_id: {graph_id}")
        schemas = graph_data.get("schemas") or {}
        allow_cycles = graph_data.get("allow_cycles", False)
        graphs[graph_id] = Graph(
            graph_id=graph_id,
            allow_cycles=allow_cycles,
            schemas=schemas,
            nodes=[],
        )

    # Pass 2: Parse all nodes
    for graph_id, graph_data in yaml_data.items():
        if not isinstance(graph_data, dict):
            raise ValueError(
                f"Graph data for '{graph_id}' must be a dictionary, got {type(graph_data).__name__}"
            )
        nodes = [parse_node(data, graphs) for data in graph_data.get("nodes", [])]
        graphs[graph_id].nodes = nodes

    # Pass 3: Resolve GraphNode references
    for graph in graphs.values():
        for node in graph.nodes:
            if isinstance(node, GraphNode):
                node.graph.nodes = graphs[node.graph.graph_id].nodes

    return graphs
