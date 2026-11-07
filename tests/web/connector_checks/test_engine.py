"""The smoke V2 engine, run through a fake identity adapter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from mcp.types import Tool

from xagent.core.tools.adapters.vibe.mcp_adapter import MCPToolAdapter, MCPWriteHint
from xagent.core.tools.adapters.vibe.mcp_approval_gate import gate_mcp_tools
from xagent.web.connector_checks import engine
from xagent.web.connector_checks.declarations import (
    ConnectorChecks,
    Equals,
    HasKeys,
    L3Envelope,
    ReadCheck,
    RunScopedName,
    StatusSuccess,
    TenantPin,
    ToolStep,
    UnprobeableCheck,
    WriteCheck,
)
from xagent.web.connector_checks.engine import run_checks
from xagent.web.connector_checks.fixtures import FixtureRef, FixtureValue
from xagent.web.connector_checks.identity import (
    ConnectorTarget,
    IdentitySetupError,
    SessionPurpose,
)
from xagent.web.connector_checks.results import RunResult, load_result_schema
from xagent.web.services.connector_probe import find_tool
from xagent.web.services.mcp_connector_session import (
    ConnectorToolsOutcome,
    ConnectorToolsResult,
)

from .fakes import (
    SUCCESS,
    FakeAdapter,
    FakeTool,
    declared,
    failed,
    listed,
    rows,
    targets,
)


def read(check_id: str, tool: str, **fields: Any) -> ReadCheck:
    fields.setdefault("assertions", (StatusSuccess(),))
    return ReadCheck(check_id, tool, **fields)


def write(check_id: str) -> WriteCheck:
    app = check_id.partition(".")[0]
    name = RunScopedName("draft")
    return WriteCheck(
        check_id,
        create=ToolStep(f"{app}_create", {"name": name}, assertions=(StatusSuccess(),)),
        verify=ToolStep(f"{app}_get", {"name": name}, assertions=(StatusSuccess(),)),
        cleanup=ToolStep(
            f"{app}_delete", {"name": name}, assertions=(StatusSuccess(),)
        ),
        tenant_pin=TenantPin(
            ToolStep(f"{app}_whoami", assertions=(StatusSuccess(),)),
            ("tenant",),
            FixtureRef(f"{app}.tenant_id"),
        ),
        expected_verdicts={f"{app}_create": "allow", f"{app}_delete": "allow"},
    )


def assert_valid_document(result: RunResult) -> None:
    Draft202012Validator(load_result_schema()).validate(result.to_document())


# Sessions and checks


async def test_a_passing_session_runs_each_check_once_in_declaration_order() -> None:
    search = FakeTool("gmail_search")
    profile = FakeTool("gmail_profile")
    adapter = FakeAdapter(targets("gmail"), {"gmail": listed(search, profile)})
    checks = ConnectorChecks(
        "gmail",
        reads=(
            read("gmail.search", "gmail_search", arguments={"max_results": 1}),
            read("gmail.profile", "gmail_profile"),
        ),
    )

    result = await run_checks(adapter, declarations=declared(checks))

    assert rows(result) == [
        ("gmail", None, "pass", None, None),
        ("gmail", "gmail.search", "pass", None, None),
        ("gmail", "gmail.profile", "pass", None, None),
    ]
    assert adapter.opened == [("gmail", SessionPurpose.CALL_TOOLS)]
    assert search.calls == [{"max_results": 1}]
    assert profile.calls == [{}]
    assert result.complete is True
    assert result.failure is None
    assert result.identity == "user"
    assert_valid_document(result)


async def test_the_environment_label_reaches_the_document() -> None:
    adapter = FakeAdapter(targets("gmail"), {"gmail": listed(FakeTool("gmail_x"))})
    checks = ConnectorChecks("gmail", reads=(read("gmail.x", "gmail_x"),))

    result = await run_checks(
        adapter, environment="stage", declarations=declared(checks)
    )

    assert result.to_document()["environment"] == "stage"


async def test_a_failed_session_skips_its_checks_and_keeps_static_ones() -> None:
    adapter = FakeAdapter(
        targets("jira"),
        {"jira": failed(ConnectorToolsOutcome.NOT_CONNECTED, "oauth_token_required")},
    )
    checks = ConnectorChecks(
        "jira",
        reads=(
            read("jira.sites", "jira_sites"),
            UnprobeableCheck("jira.boards", "needs_fixture"),
        ),
        writes=(write("jira.issue_round_trip"),),
    )

    result = await run_checks(adapter, declarations=declared(checks))

    assert rows(result) == [
        ("jira", None, "fail", "not_connected", "oauth_token_required"),
        ("jira", "jira.sites", "skip", "session_unavailable", None),
        ("jira", "jira.boards", "unprobeable", None, "needs_fixture"),
        ("jira", "jira.issue_round_trip", "skip", "writes_disabled", None),
    ]
    assert_valid_document(result)


async def test_a_rejected_client_credential_is_its_own_session_class() -> None:
    adapter = FakeAdapter(
        targets("jira"),
        {
            "jira": failed(
                ConnectorToolsOutcome.NOT_CONNECTED, "client_rejected:atlassian"
            )
        },
    )
    checks = ConnectorChecks("jira", reads=(read("jira.sites", "jira_sites"),))

    result = await run_checks(adapter, declarations=declared(checks))

    assert rows(result)[0] == (
        "jira",
        None,
        "fail",
        "client_rejected",
        "client_rejected:atlassian",
    )


async def test_a_target_the_identity_skips_is_never_started() -> None:
    adapter = FakeAdapter([ConnectorTarget("slack", skip_reason="slack_bot_grant")])
    checks = ConnectorChecks(
        "slack",
        reads=(read("slack.channels", "slack_channels"),),
        writes=(write("slack.message_round_trip"),),
    )

    result = await run_checks(adapter, declarations=declared(checks))

    assert rows(result) == [
        ("slack", None, "skip", "skipped_by_identity", "slack_bot_grant"),
        ("slack", "slack.channels", "skip", "session_unavailable", None),
        ("slack", "slack.message_round_trip", "skip", "writes_disabled", None),
    ]
    assert adapter.opened == []


async def test_an_app_without_a_declaration_lists_and_reports_not_declared() -> None:
    adapter = FakeAdapter(
        targets("Acme CRM"), {"Acme CRM": listed(FakeTool("acme_list"))}
    )

    result = await run_checks(adapter, declarations=declared())

    label = ConnectorTarget("Acme CRM").connector
    assert rows(result) == [
        (label, None, "pass", None, None),
        (label, f"{label}.l3_probe", "skip", "not_declared", None),
    ]
    assert adapter.opened == [("Acme CRM", SessionPurpose.LIST_ONLY)]
    assert "Acme CRM" not in json.dumps(result.to_document())
    assert_valid_document(result)


async def test_an_undeclared_app_keeps_not_declared_under_a_failed_session() -> None:
    adapter = FakeAdapter(
        targets("acme"),
        {"acme": failed(ConnectorToolsOutcome.FAILED_TO_START, "initialize")},
    )

    result = await run_checks(adapter, declarations=declared())

    assert rows(result) == [
        ("acme", None, "fail", "failed_to_start", "initialize"),
        ("acme", "acme.l3_probe", "skip", "not_declared", None),
    ]


async def test_a_connector_whose_checks_are_all_unprobeable_lists_only() -> None:
    adapter = FakeAdapter(
        targets("google-maps"), {"google-maps": listed(FakeTool("maps_search"))}
    )
    checks = ConnectorChecks(
        "google-maps",
        reads=(UnprobeableCheck("google-maps.l3_probe", "sandboxed"),),
    )

    result = await run_checks(adapter, declarations=declared(checks))

    assert adapter.opened == [("google-maps", SessionPurpose.LIST_ONLY)]
    assert rows(result) == [
        ("google-maps", None, "pass", None, None),
        ("google-maps", "google-maps.l3_probe", "unprobeable", None, "sandboxed"),
    ]


async def test_a_write_check_starts_the_session_for_tool_calls() -> None:
    # The purpose follows the declarations, so the session starts the same
    # way once the write switch is on.
    adapter = FakeAdapter(targets("jira"), {"jira": listed(FakeTool("jira_x"))})
    checks = ConnectorChecks("jira", writes=(write("jira.issue_round_trip"),))

    await run_checks(adapter, declarations=declared(checks))

    assert adapter.opened == [("jira", SessionPurpose.CALL_TOOLS)]


async def test_no_write_check_calls_a_tool() -> None:
    tools = [FakeTool(f"jira_{name}") for name in ("create", "get", "delete")]
    whoami = FakeTool("jira_whoami")
    adapter = FakeAdapter(targets("jira"), {"jira": listed(*tools, whoami)})
    checks = ConnectorChecks("jira", writes=(write("jira.issue_round_trip"),))

    result = await run_checks(adapter, declarations=declared(checks))

    assert rows(result)[1] == (
        "jira",
        "jira.issue_round_trip",
        "skip",
        "writes_disabled",
        None,
    )
    assert result.to_document()["rows"][1]["tool"] == "jira_create"
    assert all(tool.calls == [] for tool in [*tools, whoami])


async def test_an_exception_opening_a_session_fails_only_that_connector() -> None:
    adapter = FakeAdapter(
        targets("jira", "gmail"),
        {
            "jira": RuntimeError("secret-token in a message"),
            "gmail": listed(FakeTool("gmail_x")),
        },
    )
    checks = (
        ConnectorChecks("jira", reads=(read("jira.sites", "jira_sites"),)),
        ConnectorChecks("gmail", reads=(read("gmail.x", "gmail_x"),)),
    )

    result = await run_checks(adapter, declarations=declared(*checks))

    assert rows(result) == [
        ("jira", None, "fail", "failed_to_start", "exception"),
        ("jira", "jira.sites", "skip", "session_unavailable", None),
        ("gmail", None, "pass", None, None),
        ("gmail", "gmail.x", "pass", None, None),
    ]
    assert "secret-token" not in json.dumps(result.to_document())


# One check


async def test_a_tool_the_server_does_not_list_fails_as_tool_missing() -> None:
    adapter = FakeAdapter(targets("gmail"), {"gmail": listed(FakeTool("other"))})
    checks = ConnectorChecks("gmail", reads=(read("gmail.x", "gmail_x"),))

    result = await run_checks(adapter, declarations=declared(checks))

    assert rows(result)[1] == ("gmail", "gmail.x", "fail", "tool_missing", None)


async def test_a_raising_call_fails_without_its_message() -> None:
    tool = FakeTool("gmail_x", error=RuntimeError("Bearer abc123 rejected"))
    adapter = FakeAdapter(targets("gmail"), {"gmail": listed(tool)})
    checks = ConnectorChecks("gmail", reads=(read("gmail.x", "gmail_x"),))

    result = await run_checks(adapter, declarations=declared(checks))

    assert rows(result)[1] == ("gmail", "gmail.x", "fail", "call_failed", "exception")
    assert "abc123" not in json.dumps(result.to_document())


async def test_an_assertion_failure_names_its_kind_and_never_the_body() -> None:
    body = {"status": "success", "email": "person@example.com"}
    tool = FakeTool(
        "gmail_x",
        result={"content": [{"type": "text", "text": json.dumps(body)}]},
    )
    adapter = FakeAdapter(targets("gmail"), {"gmail": listed(tool)})
    checks = ConnectorChecks(
        "gmail",
        reads=(read("gmail.x", "gmail_x", assertions=(HasKeys((), ("id",)),)),),
    )

    result = await run_checks(adapter, declarations=declared(checks))

    assert rows(result)[1] == (
        "gmail",
        "gmail.x",
        "fail",
        "assertion_failed",
        "has_keys",
    )
    assert "person@example.com" not in json.dumps(result.to_document())


async def test_a_slow_call_times_out_and_the_next_check_still_runs() -> None:
    slow = FakeTool("gmail_slow", delay=1.0)
    fast = FakeTool("gmail_fast")
    adapter = FakeAdapter(targets("gmail"), {"gmail": listed(slow, fast)})
    checks = ConnectorChecks(
        "gmail",
        reads=(read("gmail.slow", "gmail_slow"), read("gmail.fast", "gmail_fast")),
    )

    result = await run_checks(
        adapter, declarations=declared(checks), check_timeout=0.05
    )

    assert rows(result)[1:] == [
        ("gmail", "gmail.slow", "fail", "timed_out", "check_call"),
        ("gmail", "gmail.fast", "pass", None, None),
    ]
    assert result.complete is True


async def test_each_call_gets_its_own_copy_of_the_declared_arguments() -> None:
    class Mutating(FakeTool):
        async def run_json_async(self, args: Mapping[str, Any]) -> Any:
            cast(dict[str, Any], args)["max_results"] = 99
            return await super().run_json_async(args)

    tool = Mutating("gmail_x")
    arguments = {"max_results": 1}
    adapter = FakeAdapter(targets("gmail"), {"gmail": listed(tool)})
    checks = ConnectorChecks(
        "gmail", reads=(read("gmail.x", "gmail_x", arguments=arguments),)
    )

    await run_checks(adapter, declarations=declared(checks))

    assert arguments == {"max_results": 1}


# Fixtures


async def test_fixture_keys_resolve_into_arguments_and_expected_values() -> None:
    document = {"status": "success", "document_id": "doc-1", "title": "Seed"}
    tool = FakeTool(
        "google_docs_get_document",
        result={"content": [{"type": "text", "text": json.dumps(document)}]},
    )
    adapter = FakeAdapter(
        targets("google-docs"),
        {"google-docs": listed(tool)},
        fixtures={"google-docs.document_id": "doc-1", "google-docs.title": "Seed"},
    )
    checks = ConnectorChecks(
        "google-docs",
        reads=(
            read(
                "google-docs.read_fixture",
                "google_docs_get_document",
                arguments={"document_id": FixtureRef("google-docs.document_id")},
                assertions=(
                    StatusSuccess(),
                    Equals(("title",), FixtureRef("google-docs.title")),
                ),
            ),
        ),
    )

    result = await run_checks(adapter, declarations=declared(checks))

    assert tool.calls == [{"document_id": "doc-1"}]
    assert rows(result)[1] == (
        "google-docs",
        "google-docs.read_fixture",
        "pass",
        None,
        None,
    )
    assert "doc-1" not in json.dumps(result.to_document())


@pytest.mark.parametrize("missing", ["google-docs.document_id", "google-docs.title"])
async def test_a_missing_fixture_skips_the_check_without_calling(
    missing: str,
) -> None:
    tool = FakeTool("google_docs_get_document")
    fixtures = {"google-docs.document_id": "doc-1", "google-docs.title": "Seed"}
    del fixtures[missing]
    adapter = FakeAdapter(
        targets("google-docs"), {"google-docs": listed(tool)}, fixtures=fixtures
    )
    checks = ConnectorChecks(
        "google-docs",
        reads=(
            read(
                "google-docs.read_fixture",
                "google_docs_get_document",
                arguments={"document_id": FixtureRef("google-docs.document_id")},
                assertions=(Equals(("title",), FixtureRef("google-docs.title")),),
            ),
        ),
    )

    result = await run_checks(adapter, declarations=declared(checks))

    assert rows(result) == [
        ("google-docs", None, "pass", None, None),
        ("google-docs", "google-docs.read_fixture", "skip", "fixture_missing", missing),
    ]
    assert tool.calls == []
    # The purpose follows the declarations, never the fixtures: setting the
    # fixture must not change how the connector starts.
    assert adapter.opened == [("google-docs", SessionPurpose.CALL_TOOLS)]


def fixture_checks() -> tuple[ConnectorChecks, ConnectorChecks]:
    return (
        ConnectorChecks("gmail", reads=(read("gmail.x", "gmail_x"),)),
        ConnectorChecks(
            "google-docs",
            reads=(
                read(
                    "google-docs.read_fixture",
                    "google_docs_get_document",
                    arguments={"document_id": FixtureRef("google-docs.document_id")},
                ),
            ),
        ),
    )


async def test_a_fixture_that_is_not_a_scalar_fails_before_anything_starts() -> None:
    adapter = FakeAdapter(
        targets("gmail", "google-docs"),
        {"gmail": listed(FakeTool("gmail_x")), "google-docs": listed()},
        fixtures=cast(Any, {"google-docs.document_id": {"id": "doc-1"}}),
    )

    with pytest.raises(TypeError):
        await run_checks(adapter, declarations=declared(*fixture_checks()))
    assert adapter.opened == []


async def test_a_resolver_that_raises_fails_before_anything_starts() -> None:
    class Raising(FakeAdapter):
        def resolve_fixture(self, key: str) -> FixtureValue | None:
            raise RuntimeError("config store unavailable")

    adapter = Raising(
        targets("gmail", "google-docs"),
        {"gmail": listed(FakeTool("gmail_x")), "google-docs": listed()},
    )

    with pytest.raises(RuntimeError):
        await run_checks(adapter, declarations=declared(*fixture_checks()))
    assert adapter.opened == []


async def test_a_failed_session_outranks_a_missing_fixture() -> None:
    adapter = FakeAdapter(
        targets("google-docs"),
        {"google-docs": failed(ConnectorToolsOutcome.NOT_CONNECTED, None)},
    )
    checks = ConnectorChecks(
        "google-docs",
        reads=(
            read(
                "google-docs.read_fixture",
                "google_docs_get_document",
                arguments={"document_id": FixtureRef("google-docs.document_id")},
            ),
        ),
    )

    result = await run_checks(adapter, declarations=declared(checks))

    assert rows(result)[1][2:] == ("skip", "session_unavailable", None)


# readOnlyHint


def hinted_check() -> ConnectorChecks:
    return ConnectorChecks(
        "notion",
        reads=(
            read(
                "notion.current_user",
                "notion-get-self",
                require_read_only_hint=True,
            ),
        ),
    )


async def test_a_tool_advertised_read_only_is_called() -> None:
    tool = FakeTool("notion-get-self", write_hint=MCPWriteHint.READ_ONLY)
    adapter = FakeAdapter(targets("notion"), {"notion": listed(tool)})

    result = await run_checks(adapter, declarations=declared(hinted_check()))

    assert rows(result)[1] == ("notion", "notion.current_user", "pass", None, None)
    assert tool.calls == [{}]


@pytest.mark.parametrize(
    "hint", [MCPWriteHint.UNDECLARED, MCPWriteHint.DESTRUCTIVE, None]
)
async def test_a_tool_not_advertised_read_only_is_skipped_uncalled(
    hint: MCPWriteHint | None,
) -> None:
    tool = FakeTool("notion-get-self", write_hint=cast(MCPWriteHint, hint))
    adapter = FakeAdapter(targets("notion"), {"notion": listed(tool)})

    result = await run_checks(adapter, declarations=declared(hinted_check()))

    assert rows(result)[1] == (
        "notion",
        "notion.current_user",
        "skip",
        "read_only_hint_missing",
        None,
    )
    assert tool.calls == []


@pytest.mark.parametrize(
    ("annotations", "advertised"),
    [
        ({"readOnlyHint": True}, True),
        ({"readOnlyHint": True, "destructiveHint": True}, False),
        ({"readOnlyHint": 1}, False),
        (None, False),
    ],
)
def test_the_hint_is_read_through_the_approval_gate(
    annotations: dict[str, Any] | None, advertised: bool
) -> None:
    # The loader hands back MCPToolAdapters wrapped in the approval gate.
    connection: Any = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
    }
    mcp_tool = Tool(
        name="notion-get-self",
        description="Who am I",
        inputSchema={"type": "object", "properties": {}},
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool, connection=connection, raw_annotations=annotations
    )
    (gated,) = gate_mcp_tools([adapter], connection=connection)

    assert find_tool([gated], "notion-get-self") is gated
    assert engine._advertises_read_only(gated) is advertised


async def test_a_missing_tool_under_the_hint_is_skipped_not_missing() -> None:
    adapter = FakeAdapter(targets("notion"), {"notion": listed(FakeTool("other"))})

    result = await run_checks(adapter, declarations=declared(hinted_check()))

    assert rows(result)[1][2:] == ("skip", "read_only_hint_missing", None)


# The run's total timeout


async def test_the_total_timeout_cuts_the_run_and_marks_what_it_kept_out() -> None:
    slow = FakeTool("gmail_slow", delay=1.0)
    adapter = FakeAdapter(
        targets("gmail", "jira"),
        {
            "gmail": listed(slow, FakeTool("gmail_fast")),
            "jira": listed(FakeTool("jira_sites")),
        },
    )
    checks = (
        ConnectorChecks(
            "gmail",
            reads=(
                read("gmail.slow", "gmail_slow"),
                read("gmail.fast", "gmail_fast"),
                UnprobeableCheck("gmail.labels", "needs_fixture"),
            ),
            writes=(write("gmail.draft_round_trip"),),
        ),
        ConnectorChecks(
            "jira",
            reads=(
                read("jira.sites", "jira_sites"),
                UnprobeableCheck("jira.boards", "needs_fixture"),
            ),
        ),
    )

    result = await run_checks(
        adapter, declarations=declared(*checks), check_timeout=5, total_timeout=0.1
    )

    assert rows(result) == [
        ("gmail", None, "pass", None, None),
        ("gmail", "gmail.slow", "fail", "timed_out", "run_deadline"),
        ("gmail", "gmail.fast", "fail", "not_reached", None),
        ("gmail", "gmail.labels", "unprobeable", None, "needs_fixture"),
        ("gmail", "gmail.draft_round_trip", "skip", "writes_disabled", None),
        ("jira", None, "fail", "not_reached", None),
        ("jira", "jira.sites", "skip", "session_unavailable", None),
        ("jira", "jira.boards", "unprobeable", None, "needs_fixture"),
    ]
    assert result.complete is False
    assert adapter.opened == [("gmail", SessionPurpose.CALL_TOOLS)]
    assert_valid_document(result)


async def test_the_total_timeout_during_a_session_fails_that_session() -> None:
    async def hang() -> ConnectorToolsResult:
        await asyncio.sleep(1.0)
        return listed()

    adapter = FakeAdapter(targets("gmail"), {"gmail": hang})
    checks = ConnectorChecks("gmail", reads=(read("gmail.x", "gmail_x"),))

    result = await run_checks(
        adapter, declarations=declared(checks), total_timeout=0.05
    )

    assert rows(result) == [
        ("gmail", None, "fail", "timed_out", "run_deadline"),
        ("gmail", "gmail.x", "skip", "session_unavailable", None),
    ]
    assert result.complete is False


async def test_a_timeout_the_adapter_raises_itself_does_not_cut_the_run() -> None:
    adapter = FakeAdapter(
        targets("jira", "gmail"),
        {"jira": TimeoutError(), "gmail": listed(FakeTool("gmail_x"))},
    )
    checks = (
        ConnectorChecks("jira", reads=(read("jira.sites", "jira_sites"),)),
        ConnectorChecks("gmail", reads=(read("gmail.x", "gmail_x"),)),
    )

    result = await run_checks(adapter, declarations=declared(*checks))

    assert rows(result) == [
        ("jira", None, "fail", "failed_to_start", "exception"),
        ("jira", "jira.sites", "skip", "session_unavailable", None),
        ("gmail", None, "pass", None, None),
        ("gmail", "gmail.x", "pass", None, None),
    ]
    assert result.complete is True


async def test_a_timeout_the_tool_raises_itself_does_not_cut_the_run() -> None:
    # The run's deadline is the tighter bound, but it has not expired.
    tool = FakeTool("gmail_x", error=TimeoutError())
    adapter = FakeAdapter(
        targets("gmail"), {"gmail": listed(tool, FakeTool("gmail_y"))}
    )
    checks = ConnectorChecks(
        "gmail", reads=(read("gmail.x", "gmail_x"), read("gmail.y", "gmail_y"))
    )

    result = await run_checks(
        adapter, declarations=declared(checks), check_timeout=60, total_timeout=30
    )

    assert rows(result)[1:] == [
        ("gmail", "gmail.x", "fail", "timed_out", "check_call"),
        ("gmail", "gmail.y", "pass", None, None),
    ]
    assert result.complete is True


async def test_a_call_that_swallows_its_timeout_still_timed_out() -> None:
    class Swallowing(FakeTool):
        async def run_json_async(self, args: Mapping[str, Any]) -> Any:
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                return {"content": [], "is_error": True}
            return SUCCESS

    adapter = FakeAdapter(
        targets("gmail"),
        {"gmail": listed(Swallowing("gmail_x"), FakeTool("gmail_y"))},
    )
    checks = ConnectorChecks(
        "gmail", reads=(read("gmail.x", "gmail_x"), read("gmail.y", "gmail_y"))
    )

    by_call = await run_checks(
        adapter, declarations=declared(checks), check_timeout=0.05
    )
    by_deadline = await run_checks(
        adapter, declarations=declared(checks), check_timeout=5, total_timeout=0.05
    )

    assert rows(by_call)[1:] == [
        ("gmail", "gmail.x", "fail", "timed_out", "check_call"),
        ("gmail", "gmail.y", "pass", None, None),
    ]
    assert by_call.complete is True
    assert rows(by_deadline)[1:] == [
        ("gmail", "gmail.x", "fail", "timed_out", "run_deadline"),
        ("gmail", "gmail.y", "fail", "not_reached", None),
    ]
    assert by_deadline.complete is False


async def test_a_session_that_swallows_the_deadline_still_timed_out() -> None:
    async def swallow() -> ConnectorToolsResult:
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        return listed(FakeTool("gmail_x"))

    adapter = FakeAdapter(targets("gmail"), {"gmail": swallow})
    checks = ConnectorChecks("gmail", reads=(read("gmail.x", "gmail_x"),))

    result = await run_checks(
        adapter, declarations=declared(checks), total_timeout=0.05
    )

    assert rows(result) == [
        ("gmail", None, "fail", "timed_out", "run_deadline"),
        ("gmail", "gmail.x", "skip", "session_unavailable", None),
    ]
    assert result.complete is False


async def test_a_wall_clock_step_back_keeps_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    moments = iter(
        [
            datetime(2026, 11, 8, 10, 0, tzinfo=timezone.utc),
            datetime(2026, 11, 8, 9, 0, tzinfo=timezone.utc),
        ]
    )

    class SteppingClock:
        @staticmethod
        def now(tz: Any = None) -> datetime:
            return next(moments)

    monkeypatch.setattr(engine, "datetime", SteppingClock)
    adapter = FakeAdapter(targets("gmail"), {"gmail": listed(FakeTool("gmail_x"))})
    checks = ConnectorChecks("gmail", reads=(read("gmail.x", "gmail_x"),))

    result = await run_checks(adapter, declarations=declared(checks))

    assert result.started_at == datetime(2026, 11, 8, 10, 0, tzinfo=timezone.utc)
    assert result.finished_at >= result.started_at
    assert rows(result)[1][2] == "pass"


# Setup and selection


@pytest.mark.parametrize("selector", ["connectors", "checks"])
async def test_a_selector_must_not_be_one_string(selector: str) -> None:
    adapter, checks = two_connectors()

    with pytest.raises(TypeError):
        selection: dict[str, Any] = {selector: "gmail"}
        await run_checks(adapter, declarations=checks, **selection)
    assert adapter.opened == []


async def test_an_identity_setup_error_is_the_run_failure() -> None:
    adapter = FakeAdapter(
        [], setup_error=IdentitySetupError("identity", "not_a_probe_user")
    )

    result = await run_checks(adapter)

    assert result.failure == "identity:not_a_probe_user"
    assert result.rows == ()
    assert result.complete is False
    assert_valid_document(result)


async def test_an_unexpected_enumeration_error_is_a_failure_code() -> None:
    adapter = FakeAdapter([], setup_error=RuntimeError("db host 10.0.0.1"))

    result = await run_checks(adapter)

    assert result.failure == "enumeration:exception"
    assert "10.0.0.1" not in json.dumps(result.to_document())


async def test_an_identity_without_connectors_is_a_failure() -> None:
    result = await run_checks(FakeAdapter([]))

    assert result.failure == "enumeration:no_connectors"
    assert result.complete is False


async def test_two_targets_with_one_label_fail_before_anything_starts() -> None:
    adapter = FakeAdapter(
        [ConnectorTarget("gmail"), ConnectorTarget("gmail")],
        {"gmail": listed()},
    )

    result = await run_checks(adapter, declarations=declared())

    assert result.failure == "enumeration:duplicate_connector"
    assert adapter.opened == []


def two_connectors() -> tuple[FakeAdapter, Callable[[str], ConnectorChecks | None]]:
    adapter = FakeAdapter(
        targets("gmail", "jira"),
        {
            "gmail": listed(FakeTool("gmail_a"), FakeTool("gmail_b")),
            "jira": listed(FakeTool("jira_sites")),
        },
    )
    checks = declared(
        ConnectorChecks(
            "gmail",
            reads=(read("gmail.a", "gmail_a"), read("gmail.b", "gmail_b")),
        ),
        ConnectorChecks("jira", reads=(read("jira.sites", "jira_sites"),)),
    )
    return adapter, checks


async def test_a_connector_selector_runs_only_those_connectors() -> None:
    adapter, checks = two_connectors()

    result = await run_checks(adapter, declarations=checks, connectors=["jira"])

    assert [row[:2] for row in rows(result)] == [
        ("jira", None),
        ("jira", "jira.sites"),
    ]
    assert [app for app, _ in adapter.opened] == ["jira"]


async def test_a_check_selector_runs_only_those_checks() -> None:
    adapter, checks = two_connectors()

    result = await run_checks(adapter, declarations=checks, checks=["gmail.b"])

    assert [row[:2] for row in rows(result)] == [
        ("gmail", None),
        ("gmail", "gmail.b"),
    ]
    assert [app for app, _ in adapter.opened] == ["gmail"]


async def test_a_check_selector_can_name_an_undeclared_apps_row() -> None:
    adapter = FakeAdapter(targets("acme"), {"acme": listed()})

    result = await run_checks(
        adapter, declarations=declared(), checks=["acme.l3_probe"]
    )

    assert [row[:2] for row in rows(result)] == [
        ("acme", None),
        ("acme", "acme.l3_probe"),
    ]


async def test_a_selector_never_changes_how_a_connector_starts() -> None:
    adapter = FakeAdapter(targets("gmail"), {"gmail": listed(FakeTool("gmail_x"))})
    checks = ConnectorChecks(
        "gmail",
        reads=(
            read("gmail.x", "gmail_x"),
            UnprobeableCheck("gmail.labels", "needs_fixture"),
        ),
    )

    result = await run_checks(
        adapter, declarations=declared(checks), checks=["gmail.labels"]
    )

    # Selected alone, the unprobeable check still gets the full run's session.
    assert adapter.opened == [("gmail", SessionPurpose.CALL_TOOLS)]
    assert [row[1] for row in rows(result)] == [None, "gmail.labels"]


async def test_a_selector_that_matches_nothing_is_a_failure() -> None:
    adapter, checks = two_connectors()

    result = await run_checks(adapter, declarations=checks, checks=["gmail.absent"])

    assert result.failure == "selection:no_targets"
    assert adapter.opened == []


async def test_an_invalid_declaration_is_refused_before_anything_starts() -> None:
    adapter = FakeAdapter(targets("gmail"), {"gmail": listed(FakeTool("gmail_x"))})
    broken = ConnectorChecks(
        "gmail", reads=(ReadCheck("gmail.x", "gmail_x", assertions=()),)
    )

    with pytest.raises(ValueError):
        await run_checks(adapter, declarations=declared(broken))
    assert adapter.opened == []


async def test_a_declaration_for_another_app_is_refused() -> None:
    adapter = FakeAdapter(targets("gmail"), {"gmail": listed(FakeTool("jira_x"))})
    other = ConnectorChecks("jira", reads=(read("jira.x", "jira_x"),))

    with pytest.raises(ValueError):
        await run_checks(adapter, declarations=lambda app_id: other)
    assert adapter.opened == []


async def test_l3_envelope_checks_run_through_the_probe_judgement() -> None:
    tool = FakeTool(
        "get_profile",
        result={"content": [{"type": "text", "text": json.dumps({"sub": "x"})}]},
    )
    adapter = FakeAdapter(targets("linkedin"), {"linkedin": listed(tool)})
    checks = ConnectorChecks(
        "linkedin",
        reads=(
            ReadCheck(
                "linkedin.l3_probe", "get_profile", assertions=(L3Envelope(("sub",)),)
            ),
        ),
    )

    result = await run_checks(adapter, declarations=declared(checks))

    assert rows(result)[1] == ("linkedin", "linkedin.l3_probe", "pass", None, None)
