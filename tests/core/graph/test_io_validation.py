import pytest

from xagent.core.graph.error import MismatchInputOutputError
from xagent.core.graph.graph import Graph
from xagent.core.graph.node import (
    EndNode,
    InputSchema,
    MultiDestNode,
    OutputSchema,
    SingleDestNode,
    StartNode,
)


class SingleDestNodeWithIO(SingleDestNode, InputSchema, OutputSchema):
    pass


class MultiDestNodeWithIO(MultiDestNode, InputSchema, OutputSchema):
    pass


def test_validate_inputs_outputs_valid_scenario() -> None:
    """Test valid scenario: StartNode outputs data, SingleDestNodeWithIO consumes it."""
    schemas = {
        "start_out": {"type": "object", "properties": {"data": {"type": "string"}}},
        "node1_in": {"type": "object", "properties": {"data": {"type": "string"}}},
        "node1_out": {"type": "object", "properties": {"result": {"type": "string"}}},
        "end_in": {"type": "object", "properties": {"result": {"type": "string"}}},
    }

    nodes_valid = [
        StartNode(id="start", next="node1", output_schema="start_out"),
        SingleDestNodeWithIO(
            id="node1",
            next="end",
            input_schema="node1_in",
            output_schema="node1_out",
        ),
        EndNode(id="end", input_schema="end_in"),
    ]
    graph_valid = Graph(nodes=nodes_valid, schemas=schemas)
    graph_valid.validate_input_output()  # Should not raise


def test_validate_inputs_outputs_missing_input() -> None:
    """Test invalid scenario: Missing input - node1 expects missing_data."""
    schemas = {
        "start_out": {"type": "object", "properties": {"data": {"type": "string"}}},
        "node1_in": {
            "type": "object",
            "properties": {"missing_data": {"type": "string"}},
        },
    }

    nodes_missing_input = [
        StartNode(id="start", next="node1", output_schema="start_out"),
        SingleDestNodeWithIO(
            id="node1",
            next="end",
            input_schema="node1_in",
            output_schema=None,
        ),
        EndNode(id="end"),
    ]
    graph_missing_input = Graph(nodes=nodes_missing_input, schemas=schemas)
    with pytest.raises(MismatchInputOutputError):
        graph_missing_input.validate_input_output()


def test_validate_inputs_outputs_multi_dest_same_output() -> None:
    """Test multi-destination nodes with same output names."""
    schemas = {
        "node1_out": {"type": "object", "properties": {"data1": {"type": "string"}}},
        "node2_in": {"type": "object", "properties": {"data1": {"type": "string"}}},
        "node2_out": {"type": "object", "properties": {"data2": {"type": "string"}}},
        "end_in": {"type": "object", "properties": {"data2": {"type": "string"}}},
    }

    nodes_multi_dest = [
        StartNode(id="start", next="node1"),
        MultiDestNodeWithIO(
            id="node1", next=["node2", "node3"], output_schema="node1_out"
        ),
        SingleDestNodeWithIO(
            id="node2",
            next="end",
            input_schema="node2_in",
            output_schema="node2_out",
        ),
        SingleDestNodeWithIO(
            id="node3",
            next="end",
            input_schema="node2_in",
            output_schema="node2_out",
        ),
        EndNode(id="end", input_schema="end_in"),
    ]
    graph_multi_dest = Graph(nodes=nodes_multi_dest, schemas=schemas)
    graph_multi_dest.validate_input_output()  # Should not raise


def test_validate_inputs_outputs_multi_dest_different_outputs() -> None:
    """Test multi-destination nodes with different output names causing mismatch at convergence."""
    # Note: In the new system, `end` node receives inputs from both `node2` and `node3`.
    # `node2` provides `data2`. `node3` provides `data3`.
    # If `end` requires BOTH `data2` and `data3`, it will fail because it processes paths independently.
    # The graph traversal validates that for EVERY path reaching a node, the requirements are met.
    # Path 1: start -> node1 -> node2 -> end (Has data2, missing data3)
    # Path 2: start -> node1 -> node3 -> end (Has data3, missing data2)

    schemas = {
        "node1_out": {"type": "object", "properties": {"data1": {"type": "string"}}},
        "node2_in": {"type": "object", "properties": {"data1": {"type": "string"}}},
        "node2_out": {"type": "object", "properties": {"data2": {"type": "string"}}},
        "node3_out": {"type": "object", "properties": {"data3": {"type": "string"}}},
        "end_in": {
            "type": "object",
            "properties": {
                "data2": {"type": "string"},
                "data3": {"type": "string"},
            },
        },
    }

    nodes_multi_dest = [
        StartNode(id="start", next="node1"),
        MultiDestNodeWithIO(
            id="node1", next=["node2", "node3"], output_schema="node1_out"
        ),
        SingleDestNodeWithIO(
            id="node2",
            next="end",
            input_schema="node2_in",
            output_schema="node2_out",
        ),
        SingleDestNodeWithIO(
            id="node3",
            next="end",
            input_schema="node2_in",
            output_schema="node3_out",
        ),
        EndNode(id="end", input_schema="end_in"),
    ]
    graph_multi_dest = Graph(nodes=nodes_multi_dest, schemas=schemas)
    with pytest.raises(MismatchInputOutputError):
        graph_multi_dest.validate_input_output()


def test_validate_inputs_outputs_type_mismatch() -> None:
    """Test invalid scenario: Field exists but has wrong type."""
    schemas = {
        "start_out": {"type": "object", "properties": {"age": {"type": "integer"}}},
        "node1_in": {"type": "object", "properties": {"age": {"type": "string"}}},
    }

    nodes_mismatch = [
        StartNode(id="start", next="node1", output_schema="start_out"),
        SingleDestNodeWithIO(id="node1", next="end", input_schema="node1_in"),
        EndNode(id="end"),
    ]
    graph_mismatch = Graph(nodes=nodes_mismatch, schemas=schemas)
    with pytest.raises(MismatchInputOutputError) as excinfo:
        graph_mismatch.validate_input_output()
    assert "Type mismatch at path: 'age'" in str(excinfo.value)


def test_validate_inputs_outputs_nested_valid() -> None:
    """Test valid scenario: Nested object structure matches."""
    schemas = {
        "start_out": {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "id": {"type": "integer"},
                    },
                }
            },
        },
        "node1_in": {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            },
        },
    }

    nodes_nested = [
        StartNode(id="start", next="node1", output_schema="start_out"),
        SingleDestNodeWithIO(id="node1", next="end", input_schema="node1_in"),
        EndNode(id="end"),
    ]
    graph_nested = Graph(nodes=nodes_nested, schemas=schemas)
    graph_nested.validate_input_output()  # Should not raise


def test_validate_inputs_outputs_nested_missing_field() -> None:
    """Test invalid scenario: Missing field inside a nested object."""
    schemas = {
        "start_out": {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            },
        },
        "node1_in": {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},  # Missing in upstream
                    },
                }
            },
        },
    }

    nodes_nested_missing = [
        StartNode(id="start", next="node1", output_schema="start_out"),
        SingleDestNodeWithIO(id="node1", next="end", input_schema="node1_in"),
        EndNode(id="end"),
    ]
    graph_nested_missing = Graph(nodes=nodes_nested_missing, schemas=schemas)
    with pytest.raises(MismatchInputOutputError) as excinfo:
        graph_nested_missing.validate_input_output()
    assert "Missing required field, current path: 'user.email'" in str(excinfo.value)


def test_validate_inputs_outputs_nested_type_mismatch() -> None:
    """Test invalid scenario: Type mismatch inside a nested object."""
    schemas = {
        "start_out": {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {"timeout": {"type": "string"}},
                }
            },
        },
        "node1_in": {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {"timeout": {"type": "integer"}},
                }
            },
        },
    }

    nodes_nested_type = [
        StartNode(id="start", next="node1", output_schema="start_out"),
        SingleDestNodeWithIO(id="node1", next="end", input_schema="node1_in"),
        EndNode(id="end"),
    ]
    graph_nested_type = Graph(nodes=nodes_nested_type, schemas=schemas)
    with pytest.raises(MismatchInputOutputError) as excinfo:
        graph_nested_type.validate_input_output()
    assert "Type mismatch at path: 'config.timeout'" in str(excinfo.value)


def test_validate_inputs_outputs_deep_merge() -> None:
    """Test that properties are deeply merged across nodes."""
    # Start produces user.name
    # Node1 produces user.email (merging into user object)
    # Node2 requires both user.name and user.email
    schemas = {
        "start_out": {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            },
        },
        "node1_out": {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"email": {"type": "string"}},
                }
            },
        },
        "node2_in": {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                }
            },
        },
    }

    nodes_merge = [
        StartNode(id="start", next="node1", output_schema="start_out"),
        SingleDestNodeWithIO(id="node1", next="node2", output_schema="node1_out"),
        SingleDestNodeWithIO(id="node2", next="end", input_schema="node2_in"),
        EndNode(id="end"),
    ]
    graph_merge = Graph(nodes=nodes_merge, schemas=schemas)
    graph_merge.validate_input_output()  # Should not raise


def test_validate_inputs_outputs_with_const_field() -> None:
    """
    Test that fields marked as 'const' in the input schema are ignored
    during validation even if missing from upstream output.
    """
    schemas = {
        "start_out": {"type": "object", "properties": {"data": {"type": "string"}}},
        "node1_in": {
            "type": "object",
            "properties": {
                "data": {"type": "string"},
                "version": {"type": "string", "const": "v1"},  # Should be ignored
                "nested": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "const": "debug",
                        }  # Should be ignored
                    },
                },
            },
        },
    }

    nodes = [
        StartNode(id="start", next="node1", output_schema="start_out"),
        SingleDestNodeWithIO(id="node1", next="end", input_schema="node1_in"),
        EndNode(id="end"),
    ]

    graph = Graph(nodes=nodes, schemas=schemas)
    # Should not raise MismatchInputOutputError despite 'version' and 'nested.mode'
    # being missing from 'start_out'
    graph.validate_input_output()
