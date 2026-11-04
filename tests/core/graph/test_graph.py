import pytest
import yaml
from pydantic import ValidationError

from xagent.core.graph.error import (
    DanglingNodeError,
    DuplicatedStartNodeError,
    DuplicateNodeError,
    MissingEndNodeError,
    MissingStartNodeError,
)
from xagent.core.graph.graph import Graph, GraphNode, StateMode
from xagent.core.graph.node import EndNode, MultiDestNode, SingleDestNode, StartNode


def test_valid_graph() -> None:
    """Test a valid graph passes all checks."""
    nodes = [
        StartNode(
            next="END",
        ),
        EndNode(),
    ]
    graph = Graph(nodes=nodes)
    graph.validate_graph()  # Should not raise


def test_duplicate_node_ids() -> None:
    """Test detection of duplicate node IDs."""
    nodes = [
        StartNode(
            id="dup",
            next="END",
        ),
        EndNode(
            id="dup",
        ),
    ]
    graph = Graph(nodes=nodes)
    with pytest.raises(DuplicateNodeError):
        graph.validate_graph()


def test_missing_start_node() -> None:
    """Test detection of missing start node."""
    nodes = [EndNode()]
    graph = Graph(nodes=nodes)
    with pytest.raises(MissingStartNodeError):
        graph.validate_graph()


def test_missing_end_node() -> None:
    """Test detection of missing end node."""
    nodes = [
        StartNode(
            next="END",
        )
    ]
    graph = Graph(nodes=nodes)
    with pytest.raises(MissingEndNodeError):
        graph.validate_graph()


def test_dangling_nodes() -> None:
    """Test detection of dangling nodes."""
    nodes = [
        StartNode(
            next="END",
        ),
        EndNode(),
        SingleDestNode(
            id="dangling",
            next="fake",
        ),
    ]
    graph = Graph(nodes=nodes)
    with pytest.raises(DanglingNodeError):
        graph.validate_graph()


def test_duplicate_start_nodes() -> None:
    """Test detection of multiple start nodes."""
    nodes = [
        StartNode(
            id="start1",
            next="end",
        ),
        StartNode(
            id="start2",
            next="end",
        ),
        EndNode(
            id="end",
        ),
    ]
    graph = Graph(nodes=nodes)
    with pytest.raises(DuplicatedStartNodeError):
        graph.validate_graph()


def test_empty_graph() -> None:
    """Test empty graph validation."""
    graph = Graph(nodes=[])
    with pytest.raises(MissingStartNodeError):
        graph.validate_graph()


def test_get_start_node() -> None:
    """Test retrieval of the start node."""
    nodes = [
        StartNode(
            id="start",
            next="end",
        ),
        EndNode(
            id="end",
        ),
    ]
    graph = Graph(nodes=nodes)
    start_node = graph.get_start_node()
    assert isinstance(start_node, StartNode)
    assert start_node.id == "start"


def test_get_start_node_missing() -> None:
    """Test retrieval of start node when none exists."""
    nodes = [
        EndNode(
            id="end",
        )
    ]
    graph = Graph(nodes=nodes)
    with pytest.raises(MissingStartNodeError):
        graph.get_start_node()


def test_get_next_nodes_single_dest() -> None:
    """Test retrieval of next nodes for a SingleDestNode."""
    nodes = [
        StartNode(
            id="start",
            next="middle",
        ),
        SingleDestNode(
            id="middle",
            next="end",
        ),
        EndNode(
            id="end",
        ),
    ]
    graph = Graph(nodes=nodes)
    middle_node = graph.node_map()["middle"]
    next_nodes = graph.get_next_nodes(middle_node)
    assert len(next_nodes) == 1
    assert next_nodes[0].id == "end"


def test_get_next_nodes_multi_dest() -> None:
    """Test retrieval of next nodes for a MultiDestNode."""
    nodes = [
        StartNode(
            id="start",
            next="middle1",
        ),
        MultiDestNode(
            id="middle1",
            next=["end1", "end2"],
        ),
        EndNode(
            id="end1",
        ),
        EndNode(
            id="end2",
        ),
    ]
    graph = Graph(nodes=nodes)
    middle_node = graph.node_map()["middle1"]
    next_nodes = graph.get_next_nodes(middle_node)
    assert len(next_nodes) == 2
    assert {node.id for node in next_nodes} == {"end1", "end2"}


def test_get_next_nodes_no_next() -> None:
    """Test retrieval of next nodes when none exist."""
    nodes = [
        EndNode(
            id="end",
        )
    ]
    graph = Graph(nodes=nodes)
    end_node = graph.node_map()["end"]
    next_nodes = graph.get_next_nodes(end_node)
    assert len(next_nodes) == 0


def test_node_map_initialization() -> None:
    """Test that `node_map()` is correctly initialized."""
    nodes = [
        StartNode(
            id="start",
            next="middle",
        ),
        SingleDestNode(
            id="middle",
            next="end",
        ),
        EndNode(
            id="end",
        ),
    ]
    graph = Graph(nodes=nodes)
    assert len(graph.node_map()) == 3
    assert "start" in graph.node_map()
    assert "middle" in graph.node_map()
    assert "end" in graph.node_map()
    assert isinstance(graph.node_map()["start"], StartNode)
    assert isinstance(graph.node_map()["middle"], SingleDestNode)
    assert isinstance(graph.node_map()["end"], EndNode)


def test_is_end_node_true() -> None:
    """Test that `is_end_node` returns True for an EndNode."""
    nodes = [
        EndNode(
            id="end",
        )
    ]
    graph = Graph(nodes=nodes)
    end_node = graph.node_map()["end"]
    assert graph.is_end_node(end_node)


def test_is_end_node_false() -> None:
    """Test that `is_end_node` returns False for non-EndNode instances."""
    nodes = [
        StartNode(
            id="start",
            next="end",
        ),
        EndNode(
            id="end",
        ),
    ]
    graph = Graph(nodes=nodes)
    start_node = graph.node_map()["start"]
    assert not graph.is_end_node(start_node)


def test_get_node_by_id_valid() -> None:
    """Test that `get_node_by_id` returns the correct node for a valid ID."""
    nodes = [
        StartNode(
            id="start",
            next="middle",
        ),
        SingleDestNode(
            id="middle",
            next="end",
        ),
        EndNode(
            id="end",
        ),
    ]
    graph = Graph(nodes=nodes)
    middle_node = graph.get_node_by_id("middle")
    assert isinstance(middle_node, SingleDestNode)
    assert middle_node.id == "middle"


def test_get_node_by_id_invalid() -> None:
    """Test that `get_node_by_id` raises KeyError for an invalid ID."""
    nodes = [
        StartNode(
            id="start",
            next="end",
        ),
        EndNode(
            id="end",
        ),
    ]
    graph = Graph(nodes=nodes)
    with pytest.raises(KeyError):
        graph.get_node_by_id("invalid_id")


def test_validate_schemas() -> None:
    """Test JSON schema validation."""
    import jsonschema

    nodes = [
        StartNode(id="start", next="end"),
        EndNode(id="end"),
    ]

    # Valid schema should not raise
    valid_schemas = {
        "user_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
    }
    Graph(nodes=nodes, schemas=valid_schemas)  # Should not raise

    # Invalid schema should raise SchemaError
    invalid_schemas = {
        "invalid_schema": {
            "type": "invalid_type",  # Invalid type
            "properties": {"name": {"type": "string"}},
        }
    }
    with pytest.raises(jsonschema.SchemaError):
        Graph(nodes=nodes, schemas=invalid_schemas)

    # Empty schemas should not raise
    Graph(nodes=nodes, schemas={})


@pytest.fixture
def minimal_graph():
    """Fixture providing a minimal valid graph for GraphNode."""
    return Graph(
        graph_id="test",
        nodes=[
            StartNode(next="END"),
            EndNode(),
        ],
    )


@pytest.mark.parametrize(
    "yaml_mode,expected",
    [
        ("NONE", StateMode.NONE),
        ("READ", StateMode.READ),
        ("WRITE", StateMode.WRITE),
        ("READWRITE", StateMode.READWRITE),
        ("none", StateMode.NONE),
        ("read", StateMode.READ),
        ("write", StateMode.WRITE),
        ("readwrite", StateMode.READWRITE),
        ("None", StateMode.NONE),
        ("Read", StateMode.READ),
        ("0", StateMode.NONE),
        ("1", StateMode.READ),
        ("2", StateMode.WRITE),
        ("3", StateMode.READWRITE),
    ],
)
def test_valid_mode_from_yaml(minimal_graph, yaml_mode, expected):
    """Test all valid mode values from YAML."""
    yaml_str = f"""
id: test_node
next: END
mode: {yaml_mode}
"""
    data = yaml.safe_load(yaml_str)
    data["graph"] = minimal_graph
    node = GraphNode(**data)
    assert node.mode == expected


@pytest.mark.parametrize(
    "invalid_mode",
    [
        "INVALID",
        "read_write",
        "READ_WRITE",
        "rw",
        "4",
        "99",
        "-1",
        "true",
        "false",
    ],
)
def test_invalid_mode_from_yaml(minimal_graph, invalid_mode):
    """Test that invalid mode values raise ValidationError."""
    yaml_str = f"""
id: test_node
next: END
mode: {invalid_mode}
"""
    data = yaml.safe_load(yaml_str)
    data["graph"] = minimal_graph

    with pytest.raises(ValidationError) as exc_info:
        GraphNode(**data)

    assert "mode" in str(exc_info.value).lower()
