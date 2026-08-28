from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from xagent.core.tools.adapters.vibe import current_time_tool as module
from xagent.core.tools.adapters.vibe.base import ToolCategory
from xagent.core.tools.adapters.vibe.current_time_tool import (
    CurrentTimeTool,
    current_time,
)
from xagent.core.tools.adapters.vibe.factory import ToolFactory, ToolRegistry
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

# The instant behind the reported incident: already the next day in Melbourne.
FROZEN = datetime(2026, 8, 24, 22, 3, 37, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_now", lambda: FROZEN)


def test_reports_utc_when_no_zone_is_supplied() -> None:
    result = current_time()

    assert result.utc == "2026-08-24 22:03:37"
    assert result.local == result.utc
    assert result.timezone == "UTC"
    assert result.utc_offset == "+00:00"


def test_reports_local_time_for_a_supplied_zone() -> None:
    result = current_time("Australia/Melbourne")

    assert result.utc == "2026-08-24 22:03:37"
    assert result.local == "2026-08-25 08:03:37"
    assert result.timezone == "Australia/Melbourne"
    assert result.utc_offset == "+10:00"


def test_reports_a_half_hour_offset() -> None:
    result = current_time("Asia/Kolkata")

    assert result.local == "2026-08-25 03:33:37"
    assert result.utc_offset == "+05:30"


def test_reports_a_negative_offset() -> None:
    result = current_time("America/New_York")

    assert result.local == "2026-08-24 18:03:37"
    assert result.utc_offset == "-04:00"


def test_follows_daylight_saving_for_the_same_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module, "_now", lambda: datetime(2026, 12, 24, 22, 3, 37, tzinfo=timezone.utc)
    )

    result = current_time("Australia/Melbourne")

    assert result.local == "2026-12-25 09:03:37"
    assert result.utc_offset == "+11:00"


@pytest.mark.parametrize(
    "supplied",
    [
        "Not/AZone",
        "",
        "   ",
        "Australia/Melbourne\x00",
        "A" * 5000,
        "../../../etc/passwd",
        "/etc/passwd",
        None,
    ],
)
def test_unusable_zone_degrades_to_utc(supplied: object) -> None:
    result = current_time(supplied)  # type: ignore[arg-type]

    assert result.timezone == "UTC"
    assert result.local == result.utc == "2026-08-24 22:03:37"


def test_resolves_a_differently_cased_zone_name() -> None:
    # ZoneInfo is case-sensitive against tzdata filenames on Linux, so a valid
    # but lowercased name must still resolve rather than degrade to UTC.
    result = current_time("australia/melbourne")

    assert result.timezone == "Australia/Melbourne"
    assert result.local == "2026-08-25 08:03:37"
    assert result.utc_offset == "+10:00"


def test_tool_runs_through_the_json_surface() -> None:
    tool = CurrentTimeTool()

    assert tool.run_json_sync({"timezone": "Australia/Melbourne"}) == {
        "utc": "2026-08-24 22:03:37",
        "local": "2026-08-25 08:03:37",
        "timezone": "Australia/Melbourne",
        "utc_offset": "+10:00",
    }
    assert asyncio.run(tool.run_json_async({})) == {
        "utc": "2026-08-24 22:03:37",
        "local": "2026-08-24 22:03:37",
        "timezone": "UTC",
        "utc_offset": "+00:00",
    }


def test_json_surface_rejects_a_non_string_timezone() -> None:
    # A non-string timezone fails pydantic validation at the JSON boundary,
    # before _resolve_zone runs. The tool must not raise; the caller catches
    # the validation error and reports a normal failed tool call.
    tool = CurrentTimeTool()

    with pytest.raises(ValidationError):
        tool.run_json_sync({"timezone": 42})


def test_intrinsic_category_is_not_user_assignable() -> None:
    # D1: an always-on tool must not sit in a togglable builder category, or
    # unchecking that category would falsely imply it disables the tool.
    from xagent.core.tools.adapters.vibe.base import (
        AGENT_CONFIG_UNASSIGNABLE_CATEGORIES,
    )

    assert (
        CurrentTimeTool().metadata.category.value
        in AGENT_CONFIG_UNASSIGNABLE_CATEGORIES
    )


def test_tool_declares_a_read_only_intrinsic_identity() -> None:
    metadata = CurrentTimeTool().metadata

    assert metadata.name == "get_current_time"
    # OTHER, not BASIC: it is intrinsic (never category-gated), so it must not
    # appear as a togglable category in the builder picker.
    assert metadata.category == ToolCategory.OTHER
    # Reading a clock touches nothing, so the ReAct scheduler may run it
    # alongside other concurrency-safe tools.
    assert metadata.read_only is True
    assert metadata.concurrency_safe is True


def test_description_tells_the_model_the_prompt_clock_is_stale() -> None:
    description = CurrentTimeTool().description

    assert "does not advance" in description
    assert "timezone" in description


class _FakeConfig:
    """Minimal config carrying only what the factory pipeline reads."""

    def __init__(self, spec: ToolSelectionSpec) -> None:
        self._spec = spec

    def get_tool_selection_spec(self) -> ToolSelectionSpec:
        return self._spec

    def get_sandbox(self) -> None:
        return None

    def get_workspace_config(self) -> None:
        return None

    def get_max_output_length(self) -> None:
        return None

    def get_max_field_count(self) -> None:
        return None

    def get_max_recursion_depth(self) -> None:
        return None


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (ToolSelectionSpec.from_raw(tool_categories=None), 1),  # ALL
        (ToolSelectionSpec.from_raw(tool_categories=["web_search"]), 1),  # non-basic
        (ToolSelectionSpec.from_raw(tool_categories=[]), 0),  # explicit NONE
    ],
)
async def test_intrinsic_tool_is_assembled_for_non_none_specs(
    spec: ToolSelectionSpec, expected: int
) -> None:
    """The full factory pipeline assembles exactly one usable
    get_current_time for any non-NONE selection -- including one that
    never picks the basic category -- and none for an explicit NONE."""
    tools = await ToolFactory.create_all_tools(
        _FakeConfig(spec), apply_user_override_filter=False
    )

    assert [t.name for t in tools].count("get_current_time") == expected


async def test_intrinsic_creator_is_skipped_for_explicit_none() -> None:
    """The registry gate must not even build the intrinsic tool for an
    explicit zero-tools agent: the NONE contract wins over always-on.
    Pinned at the registry level because the post-build name filter is a
    second guard that would otherwise mask a broken gate."""
    spec = ToolSelectionSpec.from_raw(tool_categories=[])

    tools = await ToolRegistry.create_registered_tools(_FakeConfig(spec))

    assert "get_current_time" not in [getattr(t, "name", None) for t in tools]


async def test_registry_import_path_registers_the_tool() -> None:
    """The tool must reach the registry through the production import list
    in ``ToolRegistry._import_tool_modules`` -- not through this test's own
    top-level import. Drop that import and its registration, then drive the
    production import path from a clean state and assert it comes back."""
    import importlib

    saved_creators = list(ToolRegistry._tool_creators)
    saved_imported = ToolRegistry._modules_imported
    module_name = module.__name__
    pkg_name, attr = module_name.rsplit(".", 1)
    pkg = importlib.import_module(pkg_name)
    saved_module = sys.modules.get(module_name)
    saved_attr = getattr(pkg, attr, None)
    ToolRegistry._tool_creators = [
        entry
        for entry in ToolRegistry._tool_creators
        if getattr(entry[0], "__module__", None) != module_name
    ]
    ToolRegistry._modules_imported = False
    # Force a genuine re-import through the production list: drop both the
    # cached module and the parent-package attribute, or ``from . import
    # current_time_tool`` would resolve the stale attribute and never re-run
    # the decorator.
    sys.modules.pop(module_name, None)
    if hasattr(pkg, attr):
        delattr(pkg, attr)
    try:
        tools = await ToolFactory.create_all_tools(
            _FakeConfig(ToolSelectionSpec.from_raw(tool_categories=["web_search"])),
            apply_user_override_filter=False,
        )
        assert [t.name for t in tools].count("get_current_time") == 1
    finally:
        ToolRegistry._tool_creators = saved_creators
        ToolRegistry._modules_imported = saved_imported
        if saved_module is not None:
            sys.modules[module_name] = saved_module
        if saved_attr is not None:
            setattr(pkg, attr, saved_attr)
