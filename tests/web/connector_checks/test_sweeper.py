"""The smoke V2 sweeper (#3564), through a fake Jira-like test tenant."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
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
    RunScopedName,
    StatusSuccess,
    Sweep,
    TenantPin,
    ToolStep,
    WriteCheck,
)
from xagent.web.connector_checks.fixtures import FixtureRef
from xagent.web.connector_checks.identity import (
    ConnectorTarget,
    IdentitySetupError,
    WritePolicy,
)
from xagent.web.connector_checks.results import (
    FAIL_CLASSES,
    SKIP_CLASSES,
    FailureClass,
    Outcome,
)
from xagent.web.connector_checks.sweeper import (
    MAX_DELETES,
    SWEEP_AFTER,
    SweepResult,
    load_sweep_schema,
    run_sweep,
)
from xagent.web.connector_checks.user_identity import UserIdentityAdapter
from xagent.web.connector_checks.writes import new_run_id
from xagent.web.services.mcp_connector_session import ConnectorToolsOutcome

from .fakes import FakeAdapter, FakePolicyAdapter, FakeTool, declared, failed, listed

NOW = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)
TENANT = FixtureRef("jira.tenant_id")
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
PIN = TenantPin(
    ToolStep("jira_list_accessible_sites", assertions=(Length(("sites",), 1, 1),)),
    path=("sites", 0, "id"),
    expected=TENANT,
)
WRITE = WriteCheck(
    "jira.issue_round_trip",
    create=ToolStep(
        "jira_create_issue",
        {
            "project_key": FixtureRef("jira.write_project_key"),
            "summary": RunScopedName("issue"),
        },
        assertions=(StatusSuccess(),),
    ),
    verify=ToolStep(
        "jira_get_issue", {"issue_key": KEY}, assertions=(StatusSuccess(),)
    ),
    cleanup=ToolStep(
        "jira_delete_issue", {"issue_key": KEY}, assertions=(StatusSuccess(),)
    ),
    cleanup_guard=ToolStep(
        "jira_get_issue",
        {"issue_key": KEY},
        assertions=(Equals(("issue", "summary"), RunScopedName("issue")),),
    ),
    tenant_pin=PIN,
    expected_verdicts=VERDICTS,
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
JIRA = ConnectorChecks("jira", writes=(WRITE,), sweep=SWEEP)


def result(payload: Any, *, is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "structured_content": None,
        "is_error": is_error,
    }


def leftover(age: timedelta, suffix: str = "issue") -> str:
    minute = (NOW - age).strftime("%Y%m%d%H%M")
    return f"xagent-smoke-{minute}-a1b2c3-{suffix}"


OLD = timedelta(hours=3)
YOUNG = timedelta(minutes=30)


class Tenant:
    """A Jira-like write area: issues by key, each with a summary."""

    def __init__(self, *summaries: str, site: str = "site-1") -> None:
        self.site = site
        self.issues = {
            f"SMOKE-{index}": name for index, name in enumerate(summaries, 1)
        }

    def sites(self, args: dict[str, Any]) -> Any:
        return result({"status": "success", "sites": [{"id": self.site}]})

    def search(self, args: dict[str, Any]) -> Any:
        issues = [{"key": key, "summary": name} for key, name in self.issues.items()]
        return result({"status": "success", "issues": issues})

    def delete(self, args: dict[str, Any]) -> Any:
        if self.issues.pop(args["issue_key"], None) is None:
            return result({"status": "error", "message": "not found"})
        return result({"status": "success"})

    def tools(self, **overrides: FakeTool) -> dict[str, FakeTool]:
        tools = {
            "jira_list_accessible_sites": FakeTool(
                "jira_list_accessible_sites", respond=self.sites
            ),
            "jira_search_issues": FakeTool("jira_search_issues", respond=self.search),
            "jira_delete_issue": FakeTool("jira_delete_issue", respond=self.delete),
        }
        tools.update(overrides)
        return tools


def adapter(
    tools: Mapping[str, FakeTool],
    *,
    verdicts: Mapping[str, str | None] = VERDICTS,
    fixtures: Mapping[str, Any] = FIXTURES,
    **kwargs: Any,
) -> FakePolicyAdapter:
    return FakePolicyAdapter(
        [ConnectorTarget("jira")],
        {"jira": listed(*tools.values())},
        verdicts=verdicts,
        fixtures=fixtures,
        **kwargs,
    )


async def sweep(
    adapter: FakeAdapter, *declarations: ConnectorChecks, **kwargs: Any
) -> SweepResult:
    kwargs.setdefault("call_timeout", 1.0)
    kwargs.setdefault("total_timeout", 60.0)
    return await run_sweep(
        adapter,
        declarations=declared(*(declarations or (JIRA,))),
        now=lambda: NOW,
        **kwargs,
    )


def row(result: SweepResult, connector: str = "jira") -> dict[str, Any]:
    (found,) = [
        item for item in result.to_document()["rows"] if item["connector"] == connector
    ]
    return dict(found)


def verdict(item: dict[str, Any]) -> tuple[Any, ...]:
    return item["outcome"], item["failure_class"], item["reason"]


def counts(item: dict[str, Any]) -> tuple[int, ...]:
    return tuple(
        item[name] for name in ("found", "deleted", "failed", "kept", "unparsed")
    )


def assert_valid_document(result: SweepResult) -> None:
    Draft202012Validator(load_sweep_schema()).validate(result.to_document())


@pytest.fixture()
def writes_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_CONNECTOR_SMOKE_WRITE_APPS", "jira")


# What the sweep deletes


@pytest.mark.usefixtures("writes_on")
async def test_the_sweep_deletes_only_leftovers_older_than_two_hours() -> None:
    tenant = Tenant(
        leftover(OLD),
        leftover(timedelta(days=2), "other"),
        leftover(YOUNG),
        "xagent-smoke-209913451234-a1b2c3-issue",  # month 13: no time
        "xagent-smoke-garbage",
        "seeded issue",
    )
    tools = tenant.tools()

    result = await sweep(adapter(tools))

    item = row(result)
    assert verdict(item) == ("pass", None, None)
    assert counts(item) == (5, 2, 0, 1, 2)
    assert tools["jira_delete_issue"].calls == [
        {"issue_key": "SMOKE-1"},
        {"issue_key": "SMOKE-2"},
    ]
    assert sorted(tenant.issues) == ["SMOKE-3", "SMOKE-4", "SMOKE-5", "SMOKE-6"]
    assert result.complete
    assert_valid_document(result)


@pytest.mark.usefixtures("writes_on")
async def test_a_sweep_deletes_at_most_its_cap_per_connector() -> None:
    tenant = Tenant(*(leftover(OLD, f"n{index}") for index in range(MAX_DELETES + 5)))

    result = await sweep(adapter(tenant.tools()))

    item = row(result)
    assert verdict(item) == ("fail", "cleanup_failed", "cap_reached")
    assert counts(item) == (MAX_DELETES + 5, MAX_DELETES, 0, 5, 0)
    assert len(tenant.issues) == 5


@pytest.mark.usefixtures("writes_on")
async def test_a_delete_that_finds_its_item_gone_counts_as_done() -> None:
    # Another environment's sweeper took it first; a young leftover keeps
    # the second listing non-empty.
    tenant = Tenant(leftover(OLD), leftover(YOUNG, "young"))

    def raced(args: dict[str, Any]) -> Any:
        tenant.issues.pop(args["issue_key"], None)
        return result({"status": "error", "message": "not found"})

    tools = tenant.tools(jira_delete_issue=FakeTool("jira_delete_issue", respond=raced))

    item = row(await sweep(adapter(tools)))

    assert verdict(item) == ("pass", None, None)
    assert counts(item) == (2, 1, 0, 1, 0)
    assert len(tools["jira_search_issues"].calls) == 2  # listed again


@pytest.mark.usefixtures("writes_on")
async def test_a_delete_that_fails_and_leaves_the_item_fails_the_sweep() -> None:
    tenant = Tenant(leftover(OLD), leftover(OLD, "other"))
    tools = tenant.tools(
        jira_delete_issue=FakeTool(
            "jira_delete_issue", result=result({"status": "error"}, is_error=True)
        )
    )

    item = row(await sweep(adapter(tools)))

    assert verdict(item) == ("fail", "cleanup_failed", "delete:tool_error")
    assert counts(item) == (2, 0, 2, 0, 0)


@pytest.mark.usefixtures("writes_on")
async def test_an_item_with_no_id_is_never_deleted() -> None:
    tenant = Tenant()

    def no_key(args: dict[str, Any]) -> Any:
        return result({"status": "success", "issues": [{"summary": leftover(OLD)}]})

    tools = tenant.tools(
        jira_search_issues=FakeTool("jira_search_issues", respond=no_key)
    )

    item = row(await sweep(adapter(tools)))

    assert verdict(item) == ("fail", "cleanup_failed", "delete:item_id_missing")
    assert counts(item) == (1, 0, 1, 0, 0)
    assert tools["jira_delete_issue"].calls == []


@pytest.mark.usefixtures("writes_on")
async def test_the_sweep_stops_deleting_when_the_run_runs_out_of_time() -> None:
    tenant = Tenant(*(leftover(OLD, f"n{index}") for index in range(10)))
    tools = tenant.tools(
        jira_delete_issue=FakeTool(
            "jira_delete_issue", respond=tenant.delete, delay=0.2
        )
    )

    # Each delete takes 0.2 s of a 1.5 s sweep that keeps 0.4 s per call.
    result = await sweep(adapter(tools), call_timeout=0.4, total_timeout=1.5)

    item = row(result)
    assert verdict(item) == ("fail", "not_reached", "time_budget")
    found, deleted, failed_, kept, unparsed = counts(item)
    assert (found, failed_, unparsed) == (10, 0, 0)
    assert 0 < deleted < 10 and deleted + kept == 10
    assert not result.complete


# What it refuses before touching the tenant


async def test_a_connector_whose_writes_are_off_is_not_swept() -> None:
    tenant = Tenant(leftover(OLD))
    fake = adapter(tenant.tools())

    result = await sweep(fake)

    assert verdict(row(result)) == ("skip", "writes_disabled", None)
    assert fake.opened == [] and fake.asked == []
    assert len(tenant.issues) == 1


@pytest.mark.usefixtures("writes_on")
async def test_a_connector_without_a_sweep_has_no_row() -> None:
    fake = FakePolicyAdapter(
        [ConnectorTarget("jira"), ConnectorTarget("notion")],
        {"jira": listed(*Tenant().tools().values())},
        verdicts=VERDICTS,
        fixtures=FIXTURES,
    )

    result = await sweep(fake, JIRA, ConnectorChecks("notion"))

    assert [item["connector"] for item in result.to_document()["rows"]] == ["jira"]


@pytest.mark.usefixtures("writes_on")
async def test_an_identity_without_a_write_policy_sweeps_nothing() -> None:
    tenant = Tenant(leftover(OLD))
    fake = FakeAdapter(
        [ConnectorTarget("jira")],
        {"jira": listed(*tenant.tools().values())},
        fixtures=FIXTURES,
    )

    item = row(await sweep(fake))

    assert verdict(item) == ("fail", "verdict_mismatch", "no_write_policy")
    assert fake.opened == []


@pytest.mark.usefixtures("writes_on")
async def test_a_delete_verdict_the_policy_does_not_give_sweeps_nothing() -> None:
    tenant = Tenant(leftover(OLD))
    fake = adapter(
        tenant.tools(),
        verdicts={"jira_create_issue": "deny", "jira_delete_issue": "allow"},
    )

    item = row(await sweep(fake))

    assert verdict(item) == ("fail", "verdict_mismatch", "delete")
    assert fake.opened == []


@pytest.mark.usefixtures("writes_on")
async def test_a_sweep_fixture_with_no_value_sweeps_nothing() -> None:
    fixtures = {**FIXTURES}
    del fixtures["jira.write_sweep_jql"]
    fake = adapter(Tenant(leftover(OLD)).tools(), fixtures=fixtures)

    item = row(await sweep(fake))

    assert verdict(item) == ("fail", "tenant_mismatch", "jira.write_sweep_jql")
    assert fake.opened == []


@pytest.mark.usefixtures("writes_on")
@pytest.mark.parametrize(
    ("absent", "step"),
    [
        ("jira_list_accessible_sites", "pin"),
        ("jira_search_issues", "find"),
        ("jira_delete_issue", "delete"),
    ],
)
async def test_a_tool_the_server_does_not_list_sweeps_nothing(
    absent: str, step: str
) -> None:
    tenant = Tenant(leftover(OLD))
    tools = tenant.tools()
    del tools[absent]

    item = row(await sweep(adapter(tools)))

    assert verdict(item) == ("fail", "tool_missing", step)
    assert all(not tool.calls for tool in tools.values())


@pytest.mark.usefixtures("writes_on")
async def test_another_tenant_is_never_swept() -> None:
    tenant = Tenant(leftover(OLD), site="site-2")
    tools = tenant.tools()

    item = row(await sweep(adapter(tools)))

    assert verdict(item) == ("fail", "tenant_mismatch", "mismatch")
    assert tools["jira_search_issues"].calls == []
    assert len(tenant.issues) == 1


@pytest.mark.usefixtures("writes_on")
async def test_a_failed_session_sweeps_nothing() -> None:
    fake = FakePolicyAdapter(
        [ConnectorTarget("jira")],
        {"jira": failed(ConnectorToolsOutcome.FAILED_TO_START, "exception")},
        verdicts=VERDICTS,
        fixtures=FIXTURES,
    )

    result = await sweep(fake)

    assert verdict(row(result)) == ("fail", "failed_to_start", "exception")
    assert_valid_document(result)


@pytest.mark.usefixtures("writes_on")
async def test_a_target_the_identity_skips_is_not_swept() -> None:
    fake = FakePolicyAdapter(
        [ConnectorTarget("jira", skip_reason="slack_bot_grant")],
        {},
        verdicts=VERDICTS,
        fixtures=FIXTURES,
    )

    item = row(await sweep(fake))

    assert verdict(item) == ("skip", "skipped_by_identity", "slack_bot_grant")
    assert fake.opened == []


@pytest.mark.usefixtures("writes_on")
@pytest.mark.parametrize(
    ("search", "expected"),
    [
        (
            FakeTool("jira_search_issues", error=RuntimeError("boom")),
            ("fail", "call_failed", "find:exception"),
        ),
        (
            FakeTool("jira_search_issues", result=result({"status": "success"})),
            ("fail", "assertion_failed", "find:items"),
        ),
    ],
)
async def test_a_find_that_lists_nothing_usable_deletes_nothing(
    search: FakeTool, expected: tuple[str, str, str]
) -> None:
    tenant = Tenant(leftover(OLD))
    tools = tenant.tools(jira_search_issues=search)

    item = row(await sweep(adapter(tools)))

    assert verdict(item) == expected
    assert tools["jira_delete_issue"].calls == []


# Run failures


@pytest.mark.usefixtures("writes_on")
async def test_a_write_policy_that_raises_fails_the_sweep() -> None:
    fake = adapter(Tenant().tools(), policy_error=RuntimeError("db down"))

    result = await sweep(fake)

    assert result.failure == "write_policy:exception"
    assert result.rows == () and not result.complete
    assert fake.opened == []
    assert_valid_document(result)


@pytest.mark.usefixtures("writes_on")
async def test_a_resolver_that_raises_fails_the_sweep() -> None:
    class Broken(FakePolicyAdapter):
        def resolve_fixture(self, key: str) -> Any:
            raise RuntimeError("store down")

    fake = Broken([ConnectorTarget("jira")], {}, verdicts=VERDICTS)

    result = await sweep(fake)

    assert result.failure == "fixtures:exception"


@pytest.mark.parametrize(
    ("fake", "failure"),
    [
        (
            FakeAdapter([], setup_error=IdentitySetupError("identity", "not_allowed")),
            "identity:not_allowed",
        ),
        (FakeAdapter([]), "enumeration:no_connectors"),
    ],
)
async def test_an_identity_that_cannot_be_set_up_fails_the_sweep(
    fake: FakeAdapter, failure: str
) -> None:
    result = await sweep(fake)

    assert result.failure == failure
    assert_valid_document(result)


@pytest.mark.usefixtures("writes_on")
async def test_a_connector_selector_that_matches_nothing_fails_the_sweep() -> None:
    result = await sweep(adapter(Tenant().tools()), connectors=["notion"])

    assert result.failure == "selection:no_targets"


# The document


@pytest.mark.usefixtures("writes_on")
async def test_the_document_and_logs_carry_no_name_value_or_error_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    name = leftover(OLD)
    tenant = Tenant(name)
    tools = tenant.tools(
        jira_delete_issue=FakeTool(
            "jira_delete_issue", error=RuntimeError("secret-error-text")
        )
    )

    with caplog.at_level(logging.DEBUG):
        result = await sweep(adapter(tools))

    document = json.dumps(result.to_document())
    for secret in (name, "SMOKE-1", "site-1", "project = SMOKE", "secret-error-text"):
        assert secret not in document
        assert secret not in caplog.text
    assert_valid_document(result)


def test_the_schema_lists_every_outcome_and_class() -> None:
    row_schema = load_sweep_schema()["$defs"]["row"]
    classes = row_schema["properties"]["failure_class"]["enum"]
    by_outcome = {
        rule["if"]["properties"]["outcome"]["const"]: rule["then"]["properties"]
        for rule in row_schema["allOf"]
    }

    assert set(classes) - {None} == {item.value for item in FailureClass}
    assert set(row_schema["properties"]["outcome"]["enum"]) == {
        Outcome.PASS.value,
        Outcome.FAIL.value,
        Outcome.SKIP.value,
    }
    assert set(by_outcome["fail"]["failure_class"]["enum"]) == {
        item.value for item in FAIL_CLASSES
    }
    assert set(by_outcome["skip"]["failure_class"]["enum"]) == {
        item.value for item in SKIP_CLASSES
    }


def test_the_sweep_waits_two_hours_and_deletes_at_most_fifty() -> None:
    assert SWEEP_AFTER == timedelta(hours=2)
    assert MAX_DELETES == 50


@pytest.mark.usefixtures("writes_on")
async def test_a_leftover_is_deleted_only_once_more_than_two_hours_old() -> None:
    tenant = Tenant(
        leftover(timedelta(hours=1, minutes=59), "a"),
        leftover(timedelta(hours=2), "b"),
        leftover(timedelta(hours=2, minutes=1), "c"),
        leftover(timedelta(minutes=-10), "future"),
    )
    tools = tenant.tools()

    item = row(await sweep(adapter(tools)))

    assert counts(item) == (4, 1, 0, 3, 0)
    assert tools["jira_delete_issue"].calls == [{"issue_key": "SMOKE-3"}]


@pytest.mark.usefixtures("writes_on")
async def test_a_name_a_write_check_gave_is_one_the_sweeper_reads() -> None:
    tenant = Tenant(f"xagent-smoke-{new_run_id(NOW - OLD)}-issue")

    item = row(await sweep(adapter(tenant.tools())))

    assert counts(item) == (1, 1, 0, 0, 0)
    assert tenant.issues == {}


@pytest.mark.usefixtures("writes_on")
async def test_an_item_listed_twice_is_swept_once() -> None:
    tenant = Tenant(leftover(OLD))

    def twice(args: dict[str, Any]) -> Any:
        issue = {"key": "SMOKE-1", "summary": leftover(OLD)}
        return result({"status": "success", "issues": [issue, issue]})

    tools = tenant.tools(
        jira_search_issues=FakeTool("jira_search_issues", respond=twice)
    )

    item = row(await sweep(adapter(tools)))

    assert counts(item) == (1, 1, 0, 0, 0)
    assert tools["jira_delete_issue"].calls == [{"issue_key": "SMOKE-1"}]


@pytest.mark.usefixtures("writes_on")
@pytest.mark.parametrize(
    "second",
    [
        result({"status": "success", "issues": []}),
        result({"status": "error"}, is_error=True),
        # The same item, its id now a number: still listed.
        result({"status": "success", "issues": [{"key": 1, "summary": "seeded"}]}),
        result({"status": "success", "issues": [{"key": 1.0, "summary": "seeded"}]}),
        # No item has an id at the ItemRef path.
        result({"status": "success", "issues": [{"summary": "seeded"}]}),
    ],
)
async def test_a_second_listing_that_proves_nothing_leaves_the_delete_failed(
    second: Any,
) -> None:
    tenant = Tenant()
    first = result(
        {"status": "success", "issues": [{"key": "1", "summary": leftover(OLD)}]}
    )
    listings = iter([first, second])
    tools = tenant.tools(
        jira_search_issues=FakeTool(
            "jira_search_issues", respond=lambda args: next(listings)
        ),
        jira_delete_issue=FakeTool(
            "jira_delete_issue", result=result({"status": "error"}, is_error=True)
        ),
    )

    item = row(await sweep(adapter(tools)))

    assert verdict(item) == ("fail", "cleanup_failed", "delete:tool_error")
    assert counts(item) == (1, 0, 1, 0, 0)


@pytest.mark.usefixtures("writes_on")
async def test_a_pin_that_cannot_read_the_tenant_sweeps_nothing() -> None:
    tenant = Tenant(leftover(OLD))
    tools = tenant.tools(
        jira_list_accessible_sites=FakeTool(
            "jira_list_accessible_sites", error=RuntimeError("boom")
        )
    )

    item = row(await sweep(adapter(tools)))

    assert verdict(item) == ("fail", "tenant_mismatch", "pin_failed")
    assert tools["jira_search_issues"].calls == []


@pytest.mark.usefixtures("writes_on")
async def test_a_connector_reached_without_time_for_a_call_is_not_started() -> None:
    fake = adapter(Tenant(leftover(OLD)).tools())

    result_ = await sweep(fake, call_timeout=5.0, total_timeout=1.0)

    assert verdict(row(result_)) == ("fail", "not_reached", "time_budget")
    assert fake.opened == []
    assert not result_.complete


@pytest.mark.usefixtures("writes_on")
async def test_a_slow_session_leaves_no_time_for_the_pin() -> None:
    tenant = Tenant(leftover(OLD))
    tools = tenant.tools()

    async def slow() -> Any:
        await asyncio.sleep(0.6)
        return listed(*tools.values())

    fake = FakePolicyAdapter(
        [ConnectorTarget("jira")], {"jira": slow}, verdicts=VERDICTS, fixtures=FIXTURES
    )

    result_ = await sweep(fake, call_timeout=0.5, total_timeout=1.0)

    assert verdict(row(result_)) == ("fail", "not_reached", "time_budget")
    assert tools["jira_list_accessible_sites"].calls == []
    assert not result_.complete


@pytest.mark.usefixtures("writes_on")
async def test_a_session_the_deadline_cuts_is_timed_out() -> None:
    async def hangs() -> Any:
        await asyncio.sleep(30)

    fake = FakePolicyAdapter(
        [ConnectorTarget("jira")], {"jira": hangs}, verdicts=VERDICTS, fixtures=FIXTURES
    )

    result_ = await sweep(fake, call_timeout=0.1, total_timeout=0.5)

    assert verdict(row(result_)) == ("fail", "timed_out", "run_deadline")
    assert not result_.complete


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"total_timeout": 0}, ValueError),
        ({"call_timeout": float("nan")}, ValueError),
        ({"environment": "Not A Label"}, ValueError),
        ({"connectors": "jira"}, TypeError),
    ],
)
async def test_invalid_arguments_raise_before_any_connector(
    kwargs: dict[str, Any], error: type[Exception]
) -> None:
    fake = adapter(Tenant().tools())

    with pytest.raises(error):
        await run_sweep(fake, declarations=declared(JIRA), **kwargs)
    assert fake.opened == []


def test_the_user_identity_has_no_write_policy() -> None:
    # So the CLI's sweep refuses every switched-on connector it is given.
    assert not isinstance(UserIdentityAdapter(None, "monitor"), WritePolicy)


@pytest.mark.usefixtures("writes_on")
async def test_no_time_to_list_again_leaves_the_failures_and_cuts_the_sweep() -> None:
    tenant = Tenant(leftover(OLD), leftover(OLD, "other"))
    tools = tenant.tools(
        jira_delete_issue=FakeTool(
            "jira_delete_issue",
            result=result({"status": "error"}, is_error=True),
            delay=0.7,
        )
    )

    # Two failing deletes of 0.7 s leave 0.6 s of 2 s: too little for 0.8.
    result_ = await sweep(adapter(tools), call_timeout=0.8, total_timeout=2.0)

    item = row(result_)
    assert verdict(item) == ("fail", "cleanup_failed", "delete:tool_error")
    assert counts(item) == (2, 0, 2, 0, 0)
    assert len(tools["jira_search_issues"].calls) == 1
    assert not result_.complete


class _Blocking(FakeTool):
    """Blocks until the sweep is cancelled; with ``swallow``, it then
    returns success as if nothing happened."""

    swallow: bool = False
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
            if not self.swallow:
                raise
        return result({"status": "success"})


@pytest.mark.usefixtures("writes_on")
@pytest.mark.parametrize("swallow", [False, True])
async def test_cancelling_the_sweep_stops_it_at_the_delete_in_flight(
    swallow: bool,
) -> None:
    tenant = Tenant(leftover(OLD), leftover(OLD, "other"))
    blocking = _Blocking("jira_delete_issue")
    blocking.swallow = swallow
    tools = tenant.tools(jira_delete_issue=blocking)
    task = asyncio.create_task(
        sweep(adapter(tools), call_timeout=30.0, total_timeout=600.0)
    )

    await asyncio.wait_for(blocking.started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Even a tool that swallowed the cancellation gets no second delete.
    assert blocking.calls == [{"issue_key": "SMOKE-1"}]
