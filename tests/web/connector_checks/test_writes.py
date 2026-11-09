"""Running smoke V2 write checks (#3564), through a fake Jira-like tenant."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from xagent.web.connector_checks.declarations import (
    ConnectorChecks,
    CreatedRef,
    Equals,
    ExpectedVerdict,
    ItemRef,
    Length,
    ReadCheck,
    RunScopedName,
    StatusSuccess,
    Sweep,
    TenantPin,
    ToolStep,
    WriteCheck,
)
from xagent.web.connector_checks.engine import run_checks
from xagent.web.connector_checks.fixtures import FixtureRef
from xagent.web.connector_checks.results import RunResult, load_result_schema
from xagent.web.connector_checks.writes import new_run_id
from xagent.web.services.mcp_connector_session import ConnectorToolsOutcome

from .fakes import (
    FakeAdapter,
    FakePolicyAdapter,
    FakeTool,
    declared,
    failed,
    listed,
    rows,
    targets,
)

TENANT = FixtureRef("jira.tenant_id")
PROJECT = FixtureRef("jira.write_project_key")
NAME = RunScopedName("issue")
KEY = CreatedRef(("issue", "key"))
FIXTURES = {
    "jira.tenant_id": "site-1",
    "jira.write_project_key": "SMOKE",
    "jira.write_sweep_jql": "project = SMOKE",
}
VERDICTS: dict[str, ExpectedVerdict] = {
    "jira_create_issue": "deny",
    "jira_delete_issue": "deny",
}
NAMED = re.compile(r"xagent-smoke-[0-9]{12}-[0-9a-f]{6}-issue")
PIN = TenantPin(
    ToolStep("jira_list_accessible_sites", assertions=(Length(("sites",), 1, 1),)),
    path=("sites", 0, "id"),
    expected=TENANT,
)
SWEEP = Sweep(
    tenant_pin=PIN,
    find=ToolStep(
        "jira_search_issues",
        {"jql": FixtureRef("jira.write_sweep_jql")},
        assertions=(StatusSuccess(),),
    ),
    items=("issues",),
    name=("summary",),
    delete=ToolStep(
        "jira_delete_issue",
        {"issue_key": ItemRef(("key",))},
        assertions=(StatusSuccess(),),
    ),
)


def result(payload: Any, *, is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "structured_content": None,
        "is_error": is_error,
    }


class Tenant:
    """A Jira-like test tenant: issues by key, each with a summary."""

    def __init__(self, site: str = "site-1") -> None:
        self.site = site
        self.issues: dict[str, str] = {}
        self.created = 0

    def sites(self, args: dict[str, Any]) -> Any:
        return result({"status": "success", "sites": [{"id": self.site}]})

    def create(self, args: dict[str, Any]) -> Any:
        self.created += 1
        key = f"SMOKE-{self.created}"
        self.issues[key] = args["summary"]
        return result({"status": "success", "issue": {"key": key}})

    def get(self, args: dict[str, Any]) -> Any:
        summary = self.issues.get(args["issue_key"])
        if summary is None:
            return result({"status": "error", "message": "not found"})
        return result(
            {
                "status": "success",
                "issue": {"key": args["issue_key"], "summary": summary},
            }
        )

    def delete(self, args: dict[str, Any]) -> Any:
        if self.issues.pop(args["issue_key"], None) is None:
            return result({"status": "error", "message": "not found"})
        return result({"status": "success"})

    def delete_named(self, args: dict[str, Any]) -> Any:
        keys = [
            key for key, summary in self.issues.items() if summary == args["summary"]
        ]
        for key in keys:
            del self.issues[key]
        return result({"status": "success", "deleted": len(keys)})

    def tools(self, **overrides: FakeTool) -> dict[str, FakeTool]:
        tools = {
            "jira_list_accessible_sites": FakeTool(
                "jira_list_accessible_sites", respond=self.sites
            ),
            "jira_create_issue": FakeTool("jira_create_issue", respond=self.create),
            "jira_get_issue": FakeTool("jira_get_issue", respond=self.get),
            "jira_delete_issue": FakeTool("jira_delete_issue", respond=self.delete),
        }
        tools.update(overrides)
        return tools


def write(**changes: Any) -> WriteCheck:
    by_key = {"issue_key": KEY}
    check = WriteCheck(
        "jira.issue_round_trip",
        create=ToolStep(
            "jira_create_issue",
            {"project_key": PROJECT, "summary": NAME},
            assertions=(StatusSuccess(),),
        ),
        verify=ToolStep(
            "jira_get_issue", by_key, assertions=(Equals(("issue", "summary"), NAME),)
        ),
        cleanup=ToolStep("jira_delete_issue", by_key, assertions=(StatusSuccess(),)),
        cleanup_guard=ToolStep(
            "jira_get_issue", by_key, assertions=(Equals(("issue", "summary"), NAME),)
        ),
        tenant_pin=PIN,
        expected_verdicts=VERDICTS,
    )
    return replace(check, **changes)


def jira(*writes: WriteCheck) -> ConnectorChecks:
    return ConnectorChecks("jira", writes=writes or (write(),), sweep=SWEEP)


def adapter(
    tools: dict[str, FakeTool],
    *,
    verdicts: Mapping[str, str | None] = VERDICTS,
    fixtures: dict[str, Any] = FIXTURES,
    **kwargs: Any,
) -> FakePolicyAdapter:
    return FakePolicyAdapter(
        targets("jira"),
        {"jira": listed(*tools.values())},
        verdicts=verdicts,
        fixtures=fixtures,
        **kwargs,
    )


async def run(
    adapter: FakeAdapter, checks: ConnectorChecks | None = None, **kwargs: Any
) -> RunResult:
    kwargs.setdefault("check_timeout", 1.0)
    kwargs.setdefault("total_timeout", 60.0)
    return await run_checks(adapter, declarations=declared(checks or jira()), **kwargs)


def write_row(result: RunResult) -> tuple[Any, ...]:
    (row,) = [row for row in rows(result) if row[1] == "jira.issue_round_trip"]
    return row[2:]


def assert_valid_document(result: RunResult) -> None:
    Draft202012Validator(load_result_schema()).validate(result.to_document())


@pytest.fixture()
def writes_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_CONNECTOR_SMOKE_WRITE_APPS", "jira")


# The round trip


@pytest.mark.usefixtures("writes_on")
async def test_a_write_check_creates_verifies_and_cleans_up() -> None:
    tenant = Tenant()
    tools = tenant.tools()

    result = await run(adapter(tools))

    assert write_row(result) == ("pass", None, None)
    assert tenant.created == 1 and tenant.issues == {}
    (created,) = tools["jira_create_issue"].calls
    assert created["project_key"] == "SMOKE"
    assert NAMED.fullmatch(created["summary"])
    # verify, then the guard, read the issue create returned; cleanup deletes it.
    assert tools["jira_get_issue"].calls == [{"issue_key": "SMOKE-1"}] * 2
    assert tools["jira_delete_issue"].calls == [{"issue_key": "SMOKE-1"}]
    assert result.complete
    assert_valid_document(result)


@pytest.mark.usefixtures("writes_on")
async def test_a_cleanup_by_name_deletes_by_the_filled_in_name() -> None:
    tenant = Tenant()
    tools = tenant.tools(
        jira_delete_issue=FakeTool("jira_delete_issue", respond=tenant.delete_named)
    )
    check = write(
        cleanup=ToolStep(
            "jira_delete_issue", {"summary": NAME}, assertions=(StatusSuccess(),)
        ),
        cleanup_guard=None,
    )

    result = await run(adapter(tools), jira(check))

    assert write_row(result) == ("pass", None, None)
    assert tenant.issues == {}
    (deleted,) = tools["jira_delete_issue"].calls
    assert deleted["summary"] == tools["jira_create_issue"].calls[0]["summary"]


@pytest.mark.usefixtures("writes_on")
async def test_the_write_checks_of_a_run_share_one_run_id() -> None:
    tenant = Tenant()
    tools = tenant.tools()
    second = write(
        id="jira.second_round_trip",
        create=ToolStep(
            "jira_create_issue",
            {"project_key": PROJECT, "summary": RunScopedName("second")},
            assertions=(StatusSuccess(),),
        ),
        verify=ToolStep(
            "jira_get_issue", {"issue_key": KEY}, assertions=(StatusSuccess(),)
        ),
        cleanup_guard=ToolStep(
            "jira_get_issue",
            {"issue_key": KEY},
            assertions=(Equals(("issue", "summary"), RunScopedName("second")),),
        ),
    )

    await run(adapter(tools), jira(write(), second))

    first, other = (call["summary"] for call in tools["jira_create_issue"].calls)
    assert first.removesuffix("-issue") == other.removesuffix("-second")
    assert tenant.issues == {}


# The write switch and the session


async def test_writes_stay_off_until_the_app_is_switched_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for switch in (None, "slack,google-drive"):
        if switch is not None:
            monkeypatch.setenv("XAGENT_CONNECTOR_SMOKE_WRITE_APPS", switch)
        tenant = Tenant()
        fake = adapter(tenant.tools())

        result = await run(fake)

        assert write_row(result) == ("skip", "writes_disabled", None)
        assert tenant.created == 0
        assert fake.asked == []


@pytest.mark.usefixtures("writes_on")
async def test_a_switched_on_write_check_under_a_failed_session_is_skipped() -> None:
    fake = FakePolicyAdapter(
        targets("jira"),
        {"jira": failed(ConnectorToolsOutcome.FAILED_TO_START, "exception")},
        verdicts=VERDICTS,
        fixtures=FIXTURES,
    )

    result = await run(fake)

    assert write_row(result) == ("skip", "session_unavailable", None)
    assert_valid_document(result)


# Refusals before any write


@pytest.mark.usefixtures("writes_on")
async def test_an_identity_without_a_write_policy_writes_nothing() -> None:
    tenant = Tenant()
    fake = FakeAdapter(
        targets("jira"), {"jira": listed(*tenant.tools().values())}, fixtures=FIXTURES
    )

    result = await run(fake)

    assert write_row(result) == ("fail", "verdict_mismatch", "no_write_policy")
    assert tenant.created == 0


@pytest.mark.usefixtures("writes_on")
@pytest.mark.parametrize(
    ("verdicts", "reason"),
    [
        ({"jira_create_issue": "allow", "jira_delete_issue": "deny"}, "create"),
        ({"jira_create_issue": "deny", "jira_delete_issue": "allow"}, "cleanup"),
        ({"jira_create_issue": "deny"}, "no_write_policy"),
    ],
)
async def test_a_verdict_the_policy_does_not_give_writes_nothing(
    verdicts: dict[str, str | None], reason: str
) -> None:
    tenant = Tenant()
    tools = tenant.tools()

    result = await run(adapter(tools, verdicts=verdicts))

    assert write_row(result) == ("fail", "verdict_mismatch", reason)
    assert all(not tool.calls for tool in tools.values())


@pytest.mark.usefixtures("writes_on")
@pytest.mark.parametrize("key", ["jira.write_project_key", "jira.tenant_id"])
async def test_a_fixture_with_no_value_refuses_the_write(key: str) -> None:
    tenant = Tenant()
    tools = tenant.tools()
    fixtures = {**FIXTURES}
    del fixtures[key]

    result = await run(adapter(tools, fixtures=fixtures))

    assert write_row(result) == ("fail", "tenant_mismatch", key)
    assert all(not tool.calls for tool in tools.values())


@pytest.mark.usefixtures("writes_on")
@pytest.mark.parametrize(
    ("absent", "step"),
    [
        ("jira_list_accessible_sites", "pin"),
        ("jira_create_issue", "create"),
        ("jira_get_issue", "verify"),
        ("jira_delete_issue", "cleanup"),
        ("jira_issue_details", "guard"),
    ],
)
async def test_a_tool_the_server_does_not_list_refuses_the_write(
    absent: str, step: str
) -> None:
    tenant = Tenant()
    tools = tenant.tools(jira_issue_details=FakeTool("jira_issue_details"))
    del tools[absent]
    guard = ToolStep(
        "jira_issue_details",
        {"issue_key": KEY},
        assertions=(Equals(("issue", "summary"), NAME),),
    )

    result = await run(adapter(tools), jira(write(cleanup_guard=guard)))

    assert write_row(result) == ("fail", "tool_missing", step)
    assert all(not tool.calls for tool in tools.values())


@pytest.mark.usefixtures("writes_on")
async def test_a_write_check_starts_only_with_time_for_every_step() -> None:
    tenant = Tenant()
    tools = tenant.tools()
    sites = FakeTool("confluence_sites")
    fake = FakePolicyAdapter(
        targets("jira", "confluence"),
        {"jira": listed(*tools.values()), "confluence": listed(sites)},
        verdicts=VERDICTS,
        fixtures=FIXTURES,
    )
    confluence = ConnectorChecks(
        "confluence",
        reads=(
            ReadCheck(
                "confluence.sites", "confluence_sites", assertions=(StatusSuccess(),)
            ),
        ),
    )

    # Five steps (pin, create, verify, guard, cleanup) of a second each: 4.5
    # seconds would do for four, so the guard is counted.
    result = await run_checks(
        fake,
        declarations=declared(jira(), confluence),
        check_timeout=1.0,
        total_timeout=4.5,
    )

    assert write_row(result) == ("fail", "not_reached", "time_budget")
    assert all(not tool.calls for tool in tools.values())
    # The refusal is the write's own; the run goes on, and is complete.
    assert ("confluence", "confluence.sites", "pass", None, None) in rows(result)
    assert result.complete


@pytest.mark.usefixtures("writes_on")
async def test_another_tenant_refuses_the_write() -> None:
    tenant = Tenant(site="site-2")
    tools = tenant.tools()

    result = await run(adapter(tools))

    assert write_row(result) == ("fail", "tenant_mismatch", "mismatch")
    assert tenant.created == 0


@pytest.mark.usefixtures("writes_on")
async def test_a_pin_that_cannot_read_the_tenant_refuses_the_write() -> None:
    tenant = Tenant()
    tools = tenant.tools(
        jira_list_accessible_sites=FakeTool(
            "jira_list_accessible_sites", error=RuntimeError("boom")
        )
    )

    result = await run(adapter(tools))

    assert write_row(result) == ("fail", "tenant_mismatch", "pin_failed")
    assert tenant.created == 0


@pytest.mark.usefixtures("writes_on")
async def test_a_write_policy_that_raises_fails_the_run_before_any_connector() -> None:
    tenant = Tenant()
    fake = adapter(tenant.tools(), policy_error=RuntimeError("db down"))

    result = await run(fake)

    assert result.failure == "write_policy:exception"
    assert result.rows == ()
    assert fake.opened == []


@pytest.mark.usefixtures("writes_on")
async def test_a_resolver_that_raises_for_a_write_fails_the_run() -> None:
    class Broken(FakePolicyAdapter):
        def resolve_fixture(self, key: str) -> Any:
            raise RuntimeError("store down")

    tenant = Tenant()
    fake = Broken(
        targets("jira"), {"jira": listed(*tenant.tools().values())}, verdicts=VERDICTS
    )

    result = await run(fake)

    assert result.failure == "fixtures:exception"
    assert fake.opened == []


# Create, verify and cleanup


@pytest.mark.usefixtures("writes_on")
@pytest.mark.parametrize(
    ("create", "expected"),
    [
        (
            FakeTool("jira_create_issue", error=RuntimeError("boom")),
            ("fail", "call_failed", "create:exception"),
        ),
        (
            FakeTool(
                "jira_create_issue", result=result({"status": "error"}, is_error=True)
            ),
            ("fail", "call_failed", "create:tool_error"),
        ),
        (
            FakeTool("jira_create_issue", delay=5.0),
            ("fail", "timed_out", "create:check_call"),
        ),
    ],
)
async def test_a_create_that_did_not_run_leaves_cleanup_to_the_sweeper(
    create: FakeTool, expected: tuple[str, str, str]
) -> None:
    tenant = Tenant()
    tools = tenant.tools(jira_create_issue=create)

    result = await run(adapter(tools))

    assert write_row(result) == expected
    assert tools["jira_get_issue"].calls == []
    assert tools["jira_delete_issue"].calls == []


@pytest.mark.usefixtures("writes_on")
async def test_a_create_whose_assertion_fails_is_still_cleaned_up() -> None:
    tenant = Tenant()

    def partial(args: dict[str, Any]) -> Any:
        tenant.create(args)
        return result({"status": "partial", "issue": {"key": "SMOKE-1"}})

    tools = tenant.tools(
        jira_create_issue=FakeTool("jira_create_issue", respond=partial)
    )

    result_ = await run(adapter(tools))

    assert write_row(result_) == ("fail", "assertion_failed", "create:status_success")
    assert tools["jira_get_issue"].calls == [{"issue_key": "SMOKE-1"}]  # the guard
    assert tenant.issues == {}


@pytest.mark.usefixtures("writes_on")
@pytest.mark.parametrize("issue", [{}, {"key": True}, {"key": ""}, {"key": {"id": 1}}])
async def test_a_create_that_returns_no_id_leaves_cleanup_to_the_sweeper(
    issue: dict[str, Any],
) -> None:
    tenant = Tenant()

    def no_key(args: dict[str, Any]) -> Any:
        tenant.create(args)
        return result({"status": "success", "issue": issue})

    tools = tenant.tools(
        jira_create_issue=FakeTool("jira_create_issue", respond=no_key)
    )

    result_ = await run(adapter(tools))

    assert write_row(result_) == (
        "fail",
        "assertion_failed",
        "create:created_ref_missing",
    )
    assert tools["jira_get_issue"].calls == []
    assert tools["jira_delete_issue"].calls == []


@pytest.mark.usefixtures("writes_on")
async def test_a_failed_verify_still_cleans_up() -> None:
    tenant = Tenant()
    verify = ToolStep(
        "jira_get_issue",
        {"issue_key": KEY},
        assertions=(Equals(("issue", "summary"), "another summary"),),
    )

    result = await run(adapter(tenant.tools()), jira(write(verify=verify)))

    assert write_row(result) == ("fail", "assertion_failed", "verify:equals")
    assert tenant.issues == {}


@pytest.mark.usefixtures("writes_on")
async def test_a_timed_out_verify_still_cleans_up() -> None:
    tenant = Tenant()
    tools = tenant.tools(jira_search=FakeTool("jira_search", delay=5.0))
    verify = ToolStep("jira_search", {"jql": NAME}, assertions=(StatusSuccess(),))

    result = await run(adapter(tools), jira(write(verify=verify)))

    assert write_row(result) == ("fail", "timed_out", "verify:check_call")
    assert tenant.issues == {}


@pytest.mark.usefixtures("writes_on")
async def test_a_guard_that_does_not_confirm_the_name_deletes_nothing() -> None:
    tenant = Tenant()
    tools = tenant.tools(
        jira_issue_details=FakeTool(
            "jira_issue_details",
            result=result({"status": "success", "issue": {"summary": "seeded"}}),
        )
    )
    guard = ToolStep(
        "jira_issue_details",
        {"issue_key": KEY},
        assertions=(Equals(("issue", "summary"), NAME),),
    )

    result_ = await run(adapter(tools), jira(write(cleanup_guard=guard)))

    assert write_row(result_) == ("fail", "cleanup_failed", "ownership")
    assert tools["jira_delete_issue"].calls == []
    assert len(tenant.issues) == 1


@pytest.mark.usefixtures("writes_on")
async def test_a_guard_whose_call_fails_deletes_nothing() -> None:
    tenant = Tenant()
    tools = tenant.tools(
        jira_issue_details=FakeTool("jira_issue_details", error=RuntimeError("boom"))
    )
    guard = ToolStep(
        "jira_issue_details",
        {"issue_key": KEY},
        assertions=(Equals(("issue", "summary"), NAME),),
    )

    result = await run(adapter(tools), jira(write(cleanup_guard=guard)))

    assert write_row(result) == ("fail", "cleanup_failed", "guard:exception")
    assert tools["jira_delete_issue"].calls == []


@pytest.mark.usefixtures("writes_on")
async def test_a_failed_cleanup_outranks_a_failed_verify() -> None:
    tenant = Tenant()
    tools = tenant.tools(
        jira_delete_issue=FakeTool(
            "jira_delete_issue", result=result({"status": "error"}, is_error=True)
        )
    )
    verify = ToolStep(
        "jira_get_issue",
        {"issue_key": KEY},
        assertions=(Equals(("issue", "summary"), "another summary"),),
    )

    result_ = await run(adapter(tools), jira(write(verify=verify)))

    assert write_row(result_) == ("fail", "cleanup_failed", "cleanup:tool_error")


class _Blocking(FakeTool):
    """A tool whose call blocks until the run is cancelled."""

    started: asyncio.Event

    def __post_init__(self) -> None:
        super().__post_init__()
        self.started = asyncio.Event()

    async def run_json_async(self, args: Any) -> Any:
        self.calls.append(dict(args))
        self.started.set()
        await asyncio.sleep(3600)


@pytest.mark.usefixtures("writes_on")
async def test_cancelling_the_run_after_create_still_cleans_up() -> None:
    tenant = Tenant()
    blocking = _Blocking("jira_search")
    tools = tenant.tools(jira_search=blocking)
    verify = ToolStep("jira_search", {"jql": NAME}, assertions=(StatusSuccess(),))
    task = asyncio.create_task(
        run(
            adapter(tools),
            jira(write(verify=verify)),
            check_timeout=30.0,
            total_timeout=600.0,
        )
    )

    await asyncio.wait_for(blocking.started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert tools["jira_delete_issue"].calls == [{"issue_key": "SMOKE-1"}]
    assert tenant.issues == {}


class _SlowDelete(FakeTool):
    """Deletes after a pause, in which the run is cancelled."""

    started: asyncio.Event

    def __post_init__(self) -> None:
        super().__post_init__()
        self.started = asyncio.Event()

    async def run_json_async(self, args: Any) -> Any:
        self.calls.append(dict(args))
        self.started.set()
        await asyncio.sleep(0.2)
        assert self.respond is not None
        return self.respond(dict(args))


@pytest.mark.usefixtures("writes_on")
async def test_cancelling_the_run_during_cleanup_lets_the_delete_finish() -> None:
    tenant = Tenant()
    delete = _SlowDelete("jira_delete_issue", respond=tenant.delete)
    tools = tenant.tools(jira_delete_issue=delete)
    task = asyncio.create_task(
        run(adapter(tools), check_timeout=30.0, total_timeout=600.0)
    )

    await asyncio.wait_for(delete.started.wait(), timeout=5)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert tenant.issues == {}


class _Swallowing(FakeTool):
    """Blocks until cancelled, then swallows the cancellation and returns."""

    started: asyncio.Event

    def __post_init__(self) -> None:
        super().__post_init__()
        self.started = asyncio.Event()

    async def run_json_async(self, args: Any) -> Any:
        self.calls.append(dict(args))
        self.started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        assert self.respond is not None
        return self.respond(dict(args))


@pytest.mark.usefixtures("writes_on")
@pytest.mark.parametrize("step", ["jira_create_issue", "jira_search"])
async def test_a_cancellation_a_tool_swallows_still_cancels_the_run(step: str) -> None:
    tenant = Tenant()
    swallowing = _Swallowing(
        step,
        respond=tenant.create
        if step == "jira_create_issue"
        else lambda args: result({"status": "success"}),
    )
    tools = tenant.tools(**{step: swallowing})
    verify = ToolStep("jira_search", {"jql": NAME}, assertions=(StatusSuccess(),))
    tools.setdefault("jira_search", FakeTool("jira_search"))
    task = asyncio.create_task(
        run(
            adapter(tools),
            jira(write(verify=verify)),
            check_timeout=30.0,
            total_timeout=600.0,
        )
    )

    await asyncio.wait_for(swallowing.started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Create ran, so its object is cleaned up before the cancellation goes on.
    assert tenant.issues == {}


@pytest.mark.usefixtures("writes_on")
async def test_rows_and_logs_carry_no_value_name_or_error_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tenant = Tenant()
    tools = tenant.tools(
        jira_delete_issue=FakeTool(
            "jira_delete_issue", error=RuntimeError("secret-error-text")
        )
    )

    with caplog.at_level(logging.DEBUG):
        result_ = await run(adapter(tools))

    assert write_row(result_) == ("fail", "cleanup_failed", "cleanup:exception")
    document = json.dumps(result_.to_document())
    (created,) = tools["jira_create_issue"].calls
    for secret in ("site-1", "SMOKE", created["summary"], "secret-error-text"):
        assert secret not in document
        assert secret not in caplog.text
    assert_valid_document(result_)


def test_the_run_id_is_the_start_minute_in_utc() -> None:
    started = datetime(2026, 11, 9, 8, 30, 59, tzinfo=timezone(timedelta(hours=8)))

    run_id = new_run_id(started)

    assert re.fullmatch(r"202611090030-[0-9a-f]{6}", run_id)
    assert new_run_id(started) != run_id


@pytest.mark.usefixtures("writes_on")
async def test_a_failed_create_deletes_nothing_by_name() -> None:
    tenant = Tenant()
    tools = tenant.tools(
        jira_create_issue=FakeTool("jira_create_issue", error=RuntimeError("boom")),
        jira_delete_issue=FakeTool("jira_delete_issue", respond=tenant.delete_named),
    )
    check = write(
        cleanup=ToolStep(
            "jira_delete_issue", {"summary": NAME}, assertions=(StatusSuccess(),)
        ),
        cleanup_guard=None,
    )

    result = await run(adapter(tools), jira(check))

    assert write_row(result) == ("fail", "call_failed", "create:exception")
    assert tools["jira_delete_issue"].calls == []


@pytest.mark.usefixtures("writes_on")
async def test_a_guard_whose_id_is_missing_leaves_the_delete_to_the_sweeper() -> None:
    # Verify finds its id; the guard also needs one create did not return.
    tenant = Tenant()
    tools = tenant.tools()
    guard = ToolStep(
        "jira_get_issue",
        {"issue_key": KEY, "id": CreatedRef(("issue", "id"))},
        assertions=(Equals(("issue", "summary"), NAME),),
    )

    result = await run(adapter(tools), jira(write(cleanup_guard=guard)))

    assert write_row(result) == (
        "fail",
        "assertion_failed",
        "create:created_ref_missing",
    )
    assert tools["jira_get_issue"].calls == [{"issue_key": "SMOKE-1"}]  # verify
    assert tools["jira_delete_issue"].calls == []
    assert len(tenant.issues) == 1


class _Converting(FakeTool):
    """Turns its bound's cancellation into an error of its own."""

    async def run_json_async(self, args: Any) -> Any:
        self.calls.append(dict(args))
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise RuntimeError("interrupted") from None


@pytest.mark.usefixtures("writes_on")
async def test_a_call_that_hides_its_timeout_still_timed_out() -> None:
    tenant = Tenant()
    tools = tenant.tools(jira_search=_Converting("jira_search"))
    verify = ToolStep("jira_search", {"jql": NAME}, assertions=(StatusSuccess(),))

    result = await run(adapter(tools), jira(write(verify=verify)), check_timeout=0.3)

    assert write_row(result) == ("fail", "timed_out", "verify:check_call")
    assert tenant.issues == {}


@pytest.mark.usefixtures("writes_on")
@pytest.mark.parametrize("step", ["jira_list_accessible_sites", "jira_create_issue"])
async def test_cancelling_the_run_before_create_returns_cleans_up_nothing(
    step: str,
) -> None:
    tenant = Tenant()
    blocking = _Blocking(step)
    tools = tenant.tools(
        jira_delete_issue=FakeTool("jira_delete_issue", respond=tenant.delete_named),
        **{step: blocking},
    )
    # By name, the delete needs nothing from create: only the create's own
    # result can let it run.
    check = write(
        cleanup=ToolStep(
            "jira_delete_issue", {"summary": NAME}, assertions=(StatusSuccess(),)
        ),
        cleanup_guard=None,
    )
    task = asyncio.create_task(
        run(adapter(tools), jira(check), check_timeout=30.0, total_timeout=600.0)
    )

    await asyncio.wait_for(blocking.started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Nothing returned a created object: nothing is read back or deleted.
    assert tools["jira_get_issue"].calls == []
    assert tools["jira_delete_issue"].calls == []


@pytest.mark.usefixtures("writes_on")
async def test_a_cleanup_that_fails_on_a_cancelled_run_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tenant = Tenant()
    blocking = _Blocking("jira_search")
    tools = tenant.tools(
        jira_search=blocking,
        jira_delete_issue=FakeTool(
            "jira_delete_issue", result=result({"status": "error"}, is_error=True)
        ),
    )
    verify = ToolStep("jira_search", {"jql": NAME}, assertions=(StatusSuccess(),))
    task = asyncio.create_task(
        run(
            adapter(tools),
            jira(write(verify=verify)),
            check_timeout=30.0,
            total_timeout=600.0,
        )
    )

    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(blocking.started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert (
        "jira.issue_round_trip was cancelled and would have reported "
        "cleanup_failed/cleanup:tool_error"
    ) in caplog.text


@pytest.mark.usefixtures("writes_on")
async def test_an_object_left_to_the_sweeper_on_a_cancelled_run_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Create returns no "id", so the cleanup by it is left to the sweeper.
    tenant = Tenant()
    blocking = _Blocking("jira_search")
    tools = tenant.tools(jira_search=blocking)
    by_id = {"issue_key": CreatedRef(("issue", "id"))}
    check = write(
        verify=ToolStep("jira_search", {"jql": NAME}, assertions=(StatusSuccess(),)),
        cleanup=ToolStep("jira_delete_issue", by_id, assertions=(StatusSuccess(),)),
        cleanup_guard=ToolStep(
            "jira_get_issue", by_id, assertions=(Equals(("issue", "summary"), NAME),)
        ),
    )
    task = asyncio.create_task(
        run(adapter(tools), jira(check), check_timeout=30.0, total_timeout=600.0)
    )

    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(blocking.started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert (
        "jira.issue_round_trip was cancelled and would have reported "
        "assertion_failed/create:created_ref_missing"
    ) in caplog.text
    assert tools["jira_delete_issue"].calls == []


@pytest.mark.usefixtures("writes_on")
async def test_the_run_id_reaches_stderr_at_the_cli_s_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tenant = Tenant()
    tools = tenant.tools()

    with caplog.at_level(logging.WARNING):
        await run(adapter(tools))

    created = tools["jira_create_issue"].calls[0]["summary"]
    run_id = created.removeprefix("xagent-smoke-").removesuffix("-issue")
    assert f"Write checks run under run id {run_id}" in caplog.text
