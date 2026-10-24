"""Generated MCP argument classes are shared between adapters of the same tool."""

import copy
import json
import logging
import socket
import threading
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any

import pytest
from jsonschema import exceptions as jsonschema_exceptions
from pydantic import create_model as real_create_model

from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module
from xagent.core.tools.adapters.vibe.mcp_adapter import EmptyArgsModel, MCPToolAdapter

BASE_BOUND = ("scope_id",)
BASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scope_id": {"type": "string"},
        "query": {"type": "string", "enum": ["a", "b"]},
        "tags": {"type": "array", "items": {"type": "string"}},
        "opts": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1}},
            "required": ["limit"],
            "default": {"limit": 5},
        },
    },
    "required": ["query"],
}


@pytest.fixture(autouse=True)
def class_cache(monkeypatch):
    cache: OrderedDict = OrderedDict()
    monkeypatch.setattr(mcp_adapter_module, "_MCP_CLASS_CACHE", cache)
    return cache


def make_adapter(
    schema: Any = BASE_SCHEMA, name: str = "search", bound: tuple[str, ...] = BASE_BOUND
) -> MCPToolAdapter:
    connection: dict[str, Any] = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
    }
    if bound:
        connection["runtime_bindings"] = [
            {
                "source": {"input_type": "context", "key": key},
                "target": {"target_type": "tool_arguments", "key": key},
            }
            for key in bound
        ]
        connection["connector_runtime"] = {
            "context": {key: "value" for key in bound},
            "secrets": {},
            "auth_selector": {},
        }
    tool = SimpleNamespace(
        name=name, description="A tool", inputSchema=copy.deepcopy(schema)
    )
    return MCPToolAdapter(mcp_tool=tool, connection=connection)


def outcome(adapter: MCPToolAdapter, payload: dict[str, Any]):
    error = adapter._nested_argument_error(payload)
    return None if error is None else (error.validator, tuple(error.absolute_path))


def standalone(class_cache, **kwargs) -> MCPToolAdapter:
    """An adapter built with no cache entry to reuse, leaving none behind."""
    class_cache.clear()
    adapter = make_adapter(**kwargs)
    class_cache.clear()
    return adapter


class CreateModelSpy:
    def __init__(
        self, fail_with: type[BaseException] | None = None, fail_times: int = 0
    ):
        self.calls = 0
        self._fail_with = fail_with
        self._fail_times = fail_times

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self._fail_with is not None and self.calls <= self._fail_times:
            raise self._fail_with("simulated")
        return real_create_model(*args, **kwargs)


def test_identical_tools_share_generated_classes(monkeypatch, class_cache):
    spy = CreateModelSpy()
    monkeypatch.setattr(mcp_adapter_module, "create_model", spy)

    adapters = [make_adapter() for _ in range(50)]

    first = adapters[0]
    assert first._args_validator is not None
    for adapter in adapters:
        assert adapter.args_type() is first.args_type()
        assert adapter._args_validator is first._args_validator
        assert adapter.return_type() is first.return_type()
    assert spy.calls == 1
    assert len(class_cache) == 1


def _reordered(schema: dict[str, Any]) -> dict[str, Any]:
    schema["properties"] = dict(reversed(list(schema["properties"].items())))
    return schema


def _changed(mutate) -> dict[str, Any]:
    schema = copy.deepcopy(BASE_SCHEMA)
    mutate(schema)
    return schema


DELTAS: dict[str, dict[str, Any]] = {
    "name": {"name": "search_other"},
    "property_added": {
        "schema": _changed(lambda s: s["properties"].update(extra={"type": "string"}))
    },
    "property_type": {
        "schema": _changed(lambda s: s["properties"]["query"].update(type="integer"))
    },
    "required_changed": {
        "schema": _changed(lambda s: s.update(required=["query", "opts"]))
    },
    "nested_items_changed": {
        "schema": _changed(
            lambda s: s["properties"]["tags"]["items"].update(type="integer")
        )
    },
    "enum_changed": {
        "schema": _changed(lambda s: s["properties"]["query"]["enum"].append("c"))
    },
    "bound_arg_added": {"bound": ("scope_id", "query")},
    "bound_arg_removed": {"bound": ()},
    "key_order_only": {"schema": _reordered(copy.deepcopy(BASE_SCHEMA))},
}


@pytest.mark.parametrize("delta", list(DELTAS))
def test_any_input_change_yields_distinct_class(delta):
    base = make_adapter()
    changed = make_adapter(**DELTAS[delta])

    assert changed.args_type() is not base.args_type()
    assert changed._args_validator is not base._args_validator


def test_adapter_subclass_does_not_share_with_base_class():
    class SubAdapter(MCPToolAdapter):
        pass

    base = make_adapter()
    sub = SubAdapter(mcp_tool=base.mcp_tool, connection=base.connection)

    assert sub.args_type() is not base.args_type()


def test_property_order_keeps_its_own_nested_validation(class_cache):
    big = {
        "type": "object",
        "properties": {
            f"k{i}": {
                "type": "object",
                "properties": {f"j{j}": {"type": "string"} for j in range(40)},
            }
            for i in range(40)
        },
    }
    small = {
        "type": "object",
        "properties": {"limit": {"type": "integer", "minimum": 1}},
        "required": ["limit"],
    }
    schemas = {
        "big_first": {
            "type": "object",
            "properties": {"big": big, "small": small},
            "required": ["big", "small"],
        },
        "small_first": {
            "type": "object",
            "properties": {"small": small, "big": big},
            "required": ["big", "small"],
        },
    }
    payloads = [
        {"big": {}, "small": {"limit": 0}},
        {"big": {}, "small": {"limit": 1}},
        {"big": {}, "small": {}},
    ]
    alone = {
        label: standalone(class_cache, schema=schema, name="t")
        for label, schema in schemas.items()
    }
    expected = {
        label: [outcome(adapter, payload) for payload in payloads]
        for label, adapter in alone.items()
    }
    assert expected["big_first"] != expected["small_first"]

    shared = {
        label: [make_adapter(schema=schema, name="t") for _ in range(2)]
        for label, schema in schemas.items()
    }

    assert len(class_cache) == 2
    for label, adapters in shared.items():
        assert adapters[0].args_type() is adapters[1].args_type()
        for adapter in adapters:
            assert [outcome(adapter, p) for p in payloads] == expected[label]


def _single_property_schema(extra_field_key: bool = False) -> dict[str, Any]:
    field: dict[str, Any] = {"type": "string"}
    if extra_field_key:
        field["x-unread-key"] = 1
    return {"type": "object", "properties": {"v": field}}


def _make_failing_build(monkeypatch):
    monkeypatch.setattr(
        mcp_adapter_module, "create_model", CreateModelSpy(RuntimeError, fail_times=3)
    )
    return {"schema": _single_property_schema()}


def _make_failing_validator_compile(monkeypatch):
    class RejectingValidator:
        @staticmethod
        def check_schema(schema):
            raise jsonschema_exceptions.SchemaError("rejected")

    monkeypatch.setattr(mcp_adapter_module, "Draft202012Validator", RejectingValidator)
    return {"schema": _single_property_schema()}


def _make_metadata_dropping_schema(monkeypatch):
    return {"schema": _single_property_schema(extra_field_key=True)}


DIAGNOSTIC_KINDS = {
    "args_model_failure_error": (
        _make_failing_build,
        logging.ERROR,
        "Failed to build args model for MCP tool diag_tool: simulated",
    ),
    "validator_compile_warning": (
        _make_failing_validator_compile,
        logging.WARNING,
        "MCP tool diag_tool argument schema could not be compiled for nested "
        "validation: SchemaError",
    ),
    "metadata_dropped_debug": (
        _make_metadata_dropping_schema,
        logging.DEBUG,
        "MCP tool diag_tool dropped 1 field schema metadata keys "
        "and 0 unreadable field schemas",
    ),
}


@pytest.mark.parametrize("kind", list(DIAGNOSTIC_KINDS))
def test_build_diagnostics_replayed_per_adapter(kind, caplog, monkeypatch):
    prepare, level, message = DIAGNOSTIC_KINDS[kind]
    kwargs = prepare(monkeypatch)
    caplog.set_level(logging.DEBUG, logger=mcp_adapter_module.logger.name)

    for _ in range(3):
        make_adapter(name="diag_tool", **kwargs)

    records = [
        r
        for r in caplog.records
        if r.name == mcp_adapter_module.logger.name and r.getMessage() == message
    ]
    assert [r.levelno for r in records] == [level] * 3


def test_class_cache_is_bounded(class_cache):
    limit = mcp_adapter_module._MCP_CLASS_CACHE_MAX_ENTRIES
    schema = _single_property_schema()
    adapters = [make_adapter(schema=schema, name=f"tool_{i}") for i in range(limit + 1)]

    assert len(class_cache) == limit
    evicted = adapters[0]
    assert outcome(evicted, {"v": 1}) == ("type", ("v",))
    assert outcome(evicted, {"v": "ok"}) is None
    assert (
        make_adapter(schema=schema, name=f"tool_{limit}").args_type()
        is adapters[limit].args_type()
    )
    assert (
        make_adapter(schema=schema, name="tool_0").args_type()
        is not evicted.args_type()
    )


class _ExhaustsRecursionWhenSerialized(dict):
    def items(self):
        raise RecursionError("simulated")


def _too_deep_to_serialize() -> dict[str, Any]:
    return {
        **_single_property_schema(),
        "x-nested": _ExhaustsRecursionWhenSerialized(a=1),
    }


def _oversized() -> dict[str, Any]:
    cap = mcp_adapter_module._MCP_CLASS_CACHE_MAX_SCHEMA_CHARS
    return {
        "type": "object",
        "properties": {"v": {"type": "string", "description": "x" * (cap + 1)}},
    }


UNCACHEABLE_SCHEMAS = {
    "unserializable": lambda: {**_single_property_schema(), "x-opaque": object()},
    "too_deep_to_serialize": _too_deep_to_serialize,
    "oversized": _oversized,
}


@pytest.mark.parametrize("kind", list(UNCACHEABLE_SCHEMAS))
def test_unserializable_or_oversized_schema_bypasses_cache(kind, class_cache):
    schema = UNCACHEABLE_SCHEMAS[kind]()
    tool = SimpleNamespace(name="tool", description="d", inputSchema=schema)
    connection = {"transport": "streamable_http", "url": "https://mcp.example.test"}

    first = MCPToolAdapter(mcp_tool=tool, connection=connection)
    second = MCPToolAdapter(mcp_tool=tool, connection=connection)

    assert list(first.args_type().model_fields) == ["v"]
    assert first.args_type() is not second.args_type()
    assert first._args_validator is not None
    assert len(class_cache) == 0


def test_shared_validator_is_not_mutated_by_calls(class_cache):
    payloads = [
        {"query": "a", "opts": {"limit": 3}},
        {"query": "zzz"},
        {"query": "b", "opts": {"limit": 0}},
        {"query": "a", "tags": [1]},
    ]
    alone = standalone(class_cache)
    expected = [outcome(alone, payload) for payload in payloads]
    first, second = make_adapter(), make_adapter()
    assert first._args_validator is second._args_validator
    assert first._args_validator is not alone._args_validator

    for i in range(1000):
        adapter = first if i % 2 == 0 else second
        assert outcome(adapter, payloads[i % len(payloads)]) == expected[i % 4]


def test_mutating_source_schema_does_not_change_cached_class(class_cache):
    source = copy.deepcopy(BASE_SCHEMA)
    tool = SimpleNamespace(name="search", description="d", inputSchema=source)
    connection = {"transport": "streamable_http", "url": "https://mcp.example.test"}
    adapter = MCPToolAdapter(mcp_tool=tool, connection=connection)
    schema_before = json.dumps(adapter.args_type().model_json_schema())
    default_before = adapter.args_type()(query="a").opts

    source["properties"]["query"]["enum"].append("INJECTED")
    source["properties"]["opts"]["default"]["limit"] = 999

    assert json.dumps(adapter.args_type().model_json_schema()) == schema_before
    assert adapter.args_type()(query="a").opts == default_before == {"limit": 5}
    clean = MCPToolAdapter(
        mcp_tool=SimpleNamespace(
            name="search", description="d", inputSchema=copy.deepcopy(BASE_SCHEMA)
        ),
        connection=connection,
    )
    assert clean.args_type() is adapter.args_type()


def test_failed_build_is_not_cached(monkeypatch, class_cache):
    spy = CreateModelSpy(RecursionError, fail_times=1)
    monkeypatch.setattr(mcp_adapter_module, "create_model", spy)

    failed = make_adapter()
    assert failed.args_type() is EmptyArgsModel
    assert len(class_cache) == 0

    retried = make_adapter()
    assert retried.args_type() is not EmptyArgsModel
    assert "query" in retried.args_type().model_fields
    assert len(class_cache) == 1
    assert make_adapter().args_type() is retried.args_type()
    assert spy.calls == 2


def test_empty_schemas_are_cached():
    for schema in ({}, {"type": "object", "properties": {}}):
        first, second = make_adapter(schema=schema), make_adapter(schema=schema)
        assert first.args_type() is EmptyArgsModel
        assert first._args_validator is second._args_validator


def test_concurrent_builds_keep_the_first_inserted_classes(monkeypatch, class_cache):
    barrier = threading.Barrier(2, timeout=10)
    calls = []

    def create_model_after_both_started(*args, **kwargs):
        calls.append(threading.current_thread().name)
        barrier.wait()
        return real_create_model(*args, **kwargs)

    monkeypatch.setattr(
        mcp_adapter_module, "create_model", create_model_after_both_started
    )
    adapters: list[MCPToolAdapter] = []
    errors: list[BaseException] = []

    def build():
        try:
            adapters.append(make_adapter())
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=build) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert not errors
    assert len(calls) == 2
    assert len(class_cache) == 1
    assert adapters[0].args_type() is adapters[1].args_type()
    assert adapters[0]._args_validator is adapters[1]._args_validator


def test_replayed_diagnostics_exclude_other_threads(caplog):
    logger = mcp_adapter_module.logger
    caplog.set_level(logging.DEBUG, logger=logger.name)
    stop = threading.Event()

    def noise():
        while not stop.is_set():
            logger.debug("noise from another thread")

    thread = threading.Thread(target=noise)
    thread.start()
    try:
        schema = _single_property_schema(extra_field_key=True)
        for _ in range(3):
            make_adapter(schema=schema, name="diag_tool")
    finally:
        stop.set()
        thread.join(timeout=10)

    (entry,) = next(iter(mcp_adapter_module._MCP_CLASS_CACHE.values())).diagnostics
    assert entry[1].startswith("MCP tool %s dropped")
    own = [r for r in caplog.records if r.getMessage().startswith("MCP tool diag_tool")]
    assert len(own) == 3
    assert {r.threadName for r in own} == {threading.current_thread().name}


def test_validator_schema_has_no_ref(monkeypatch):
    schema = {
        "type": "object",
        "properties": {
            "a": {"$ref": "http://remote.example.invalid/schema.json"},
            "b": {
                "type": "array",
                "items": {"$ref": "http://remote.example.invalid/item.json"},
            },
        },
    }
    adapter = make_adapter(schema=schema, bound=())
    connects = []
    monkeypatch.setattr(
        socket.socket, "connect", lambda *a, **k: connects.append(a) or 1 / 0
    )

    assert adapter._args_validator is not None
    assert "$ref" not in json.dumps(adapter._args_validator.schema)
    assert adapter._nested_argument_error({"a": 1, "b": [1, "x"]}) is None
    assert connects == []
