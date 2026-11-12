"""Jira's smoke V2 Tier 1 checks (#3570), run end to end.

The declaration in ``registry.py`` runs through the real engine and sweeper,
and every step calls the real tool function in ``web/tools/mcp/jira.py``.
Only Jira's HTTP API is replaced, by an in-memory site laid out like the test
tenant's seed: a read project with a known issue and comment, and a write
project for the write check.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import unquote

import pytest

from xagent.web.connector_checks.engine import run_checks
from xagent.web.connector_checks.registry import CONNECTOR_CHECKS
from xagent.web.connector_checks.results import RunResult
from xagent.web.connector_checks.sweeper import run_sweep
from xagent.web.tools.mcp import jira

from .fakes import FakePolicyAdapter, FakeTool, listed, rows, targets, tool_for

SITE = "site-1"
READ_PROJECT = "SMOKER"
WRITE_PROJECT = "SMOKEW"
SEED_ISSUE = "SMOKER-1"
SEED_COMMENT = "10001"
FIXTURES = {
    "jira.tenant_id": SITE,
    "jira.seed.search_jql": f"project = {READ_PROJECT} ORDER BY key ASC",
    "jira.seed.issue_key": SEED_ISSUE,
    "jira.seed.comment_id": SEED_COMMENT,
    "jira.write_project_key": WRITE_PROJECT,
    "jira.write_sweep_jql": f"project = {WRITE_PROJECT} ORDER BY created ASC",
}
# Toby withholds both from a member who did not install the workspace.
VERDICTS = {"jira_create_issue": "deny", "jira_delete_issue": "deny"}
NAMED = re.compile(r"xagent-smoke-[0-9]{12}-[0-9a-f]{6}-issue")
_JQL = re.compile(r"project = (?P<project>[A-Z]+) ORDER BY (key|created) ASC")
_ISSUE = re.compile(r"/rest/api/2/issue/(?P<key>[^/]+)(?P<comments>/comment)?")


class Site:
    """An in-memory Jira Cloud site answering the requests jira.py sends."""

    def __init__(self) -> None:
        self.revoked = False
        self.other_site = False
        self.me: dict[str, Any] = {
            "account_id": "acc-1",
            "email": "qa@example.invalid",
            "name": "QA",
        }
        self.issues: dict[str, dict[str, Any]] = {}
        self.comments: dict[str, list[dict[str, Any]]] = {}
        self.counters: dict[str, int] = {}
        self.requests: list[tuple[str, str]] = []
        self.add(READ_PROJECT, "Seeded issue", "Seeded description")
        self.add(READ_PROJECT, "Second seeded issue", "")
        self.comments[SEED_ISSUE] = [
            {
                "id": SEED_COMMENT,
                "author": {"accountId": "acc-1", "displayName": "QA"},
                # REST API v2 returns rich text as a wiki-markup string.
                "body": "Seeded comment",
                "created": "2026-11-11T00:00:00.000+0000",
                "updated": "2026-11-11T00:00:00.000+0000",
            }
        ]

    def add(self, project: str, summary: str, description: str) -> str:
        self.counters[project] = self.counters.get(project, 0) + 1
        key = f"{project}-{self.counters[project]}"
        self.issues[key] = {
            "id": str(10000 + len(self.issues)),
            "key": key,
            "fields": {
                "summary": summary,
                "description": description or None,
                "status": {"name": "To Do", "statusCategory": {"name": "To Do"}},
                "issuetype": {"name": "Task"},
                "project": {"key": project},
                "priority": {"name": "Medium"},
                "labels": [],
                "created": "2026-11-11T00:00:00.000+0000",
                "updated": "2026-11-11T00:00:00.000+0000",
            },
        }
        return key

    def project_issues(self, project: str) -> list[str]:
        return [key for key in self.issues if key.startswith(f"{project}-")]

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        if self.revoked:
            raise RuntimeError("401 Client Error: Unauthorized")
        if url == jira.ACCESSIBLE_RESOURCES_URL:
            sites = [{"id": SITE, "name": "xinference-smoke", "scopes": []}]
            if self.other_site:
                sites.append({"id": "site-2", "name": "another", "scopes": []})
            return sites
        if url == jira.ME_URL:
            return self.me
        prefix = f"{jira.JIRA_API_BASE}/{SITE}"
        if not url.startswith(prefix):
            raise RuntimeError("404 Client Error: Not Found")
        path = unquote(url[len(prefix) :])
        self.requests.append((method, path))
        params = kwargs.get("params") or {}
        body = kwargs.get("json_data") or {}
        if (method, path) == ("GET", "/rest/api/3/search/jql"):
            match = _JQL.fullmatch(params["jql"])
            assert match, params["jql"]
            keys = self.project_issues(match["project"])
            return {"issues": [self.issues[key] for key in keys]}
        if (method, path) == ("POST", "/rest/api/2/issue"):
            fields = body["fields"]
            key = self.add(fields["project"]["key"], fields["summary"], "")
            return {"id": self.issues[key]["id"], "key": key, "self": url}
        match = _ISSUE.fullmatch(path)
        if match is None or match["key"] not in self.issues:
            raise RuntimeError("404 Client Error: Not Found")
        key = match["key"]
        if match["comments"]:
            comments = self.comments.get(key, [])
            return {"comments": comments, "total": len(comments), "startAt": 0}
        if method == "GET":
            return self.issues[key]
        if method == "DELETE":
            del self.issues[key]
            return {}
        raise AssertionError((method, path))


# Every tool the server registers, so the session lists what production does.
_TOOLS = (
    "jira_list_accessible_sites",
    "jira_get_current_user",
    "jira_list_projects",
    "jira_search_issues",
    "jira_get_issue",
    "jira_create_issue",
    "jira_update_issue",
    "jira_list_transitions",
    "jira_transition_issue",
    "jira_list_comments",
    "jira_add_comment",
    "jira_add_attachment",
    "jira_delete_issue",
    "jira_search_users",
)


@pytest.fixture()
def site(monkeypatch: pytest.MonkeyPatch) -> Site:
    site = Site()
    monkeypatch.setattr(jira, "_request_absolute", site.request)
    return site


@pytest.fixture()
def writes_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_CONNECTOR_SMOKE_WRITE_APPS", "jira")


def _tools() -> dict[str, FakeTool]:
    return {name: tool_for(jira, name) for name in _TOOLS}


def _adapter(
    fixtures: dict[str, Any] = FIXTURES, tools: dict[str, FakeTool] | None = None
) -> FakePolicyAdapter:
    tools = tools if tools is not None else _tools()
    return FakePolicyAdapter(
        targets("jira"),
        {"jira": listed(*tools.values())},
        verdicts=VERDICTS,
        fixtures=fixtures,
    )


# The tools that take no site: the pin's and L3's site listing, and /me.
_SITELESS = {"jira_list_accessible_sites", "jira_get_current_user"}


def _assert_every_issue_call_names_the_site(tools: dict[str, FakeTool]) -> None:
    called = {name: tool.calls for name, tool in tools.items() if tool.calls}
    assert set(called) - _SITELESS
    for name, calls in called.items():
        if name not in _SITELESS:
            assert all(call.get("cloud_id") == SITE for call in calls), name


async def _run(adapter: FakePolicyAdapter) -> RunResult:
    return await run_checks(adapter, check_timeout=5.0, total_timeout=60.0)


def _check_rows(result: RunResult) -> dict[str, tuple[Any, ...]]:
    # The session row carries no check id.
    return {row[1] or "session": row[2:] for row in rows(result) if row[0] == "jira"}


_CHECKS = (
    "jira.l3_probe",
    "jira.search_issues",
    "jira.list_comments",
    "jira.get_issue",
    "jira.current_user",
    "jira.issue_round_trip",
)


# The declaration


def test_jira_checks_the_tools_that_carry_its_calls() -> None:
    checks = CONNECTOR_CHECKS["jira"]

    assert [check.id for check in (*checks.reads, *checks.writes)] == list(_CHECKS)
    assert checks.checked_tools() == {
        "jira_list_accessible_sites",
        "jira_search_issues",
        "jira_list_comments",
        "jira_get_issue",
        "jira_get_current_user",
        "jira_create_issue",
        "jira_delete_issue",
    }
    assert sorted(exemption.tool for exemption in checks.exemptions) == [
        "jira_add_attachment",
        "jira_add_comment",
        "jira_list_projects",
        "jira_list_transitions",
        "jira_search_users",
        "jira_transition_issue",
        "jira_update_issue",
    ]


# A run against the seeded site


@pytest.mark.usefixtures("writes_on")
async def test_every_check_passes_on_the_seeded_site_and_leaves_nothing(
    site: Site,
) -> None:
    seeded = dict(site.issues)
    tools = _tools()

    result = await _run(_adapter(tools=tools))

    _assert_every_issue_call_names_the_site(tools)
    assert set(tools["jira_create_issue"].calls[0]) == {
        "cloud_id",
        "project_key",
        "summary",
    }
    assert _check_rows(result) == {
        "session": ("pass", None, None),
        **{check: ("pass", None, None) for check in _CHECKS},
    }
    assert result.complete
    # The write check created one run-scoped issue in the write project and
    # deleted it again; the read project is untouched.
    assert [path for method, path in site.requests if method == "POST"] == [
        "/rest/api/2/issue"
    ]
    deleted = [path for method, path in site.requests if method == "DELETE"]
    assert deleted == [f"/rest/api/2/issue/{WRITE_PROJECT}-1"]
    assert site.issues == seeded


@pytest.mark.usefixtures("writes_on")
async def test_the_write_check_names_its_issue_for_its_run(site: Site) -> None:
    created: list[dict[str, Any]] = []
    original = site.add

    def add(project: str, summary: str, description: str) -> str:
        created.append({"project": project, "summary": summary})
        return original(project, summary, description)

    site.add = add  # type: ignore[method-assign]

    await _run(_adapter())

    (issue,) = created
    assert issue["project"] == WRITE_PROJECT
    assert NAMED.fullmatch(issue["summary"])


async def test_with_writes_off_the_reads_still_run(site: Site) -> None:
    result = await _run(_adapter())

    checks = _check_rows(result)
    assert checks.pop("jira.issue_round_trip") == ("skip", "writes_disabled", None)
    assert set(checks.values()) == {("pass", None, None)}
    assert not [method for method, _ in site.requests if method != "GET"]


@pytest.mark.usefixtures("writes_on")
async def test_a_revoked_grant_fails_every_check_and_writes_nothing(
    site: Site,
) -> None:
    seeded = dict(site.issues)
    site.revoked = True

    result = await _run(_adapter())

    checks = _check_rows(result)
    assert checks.pop("session") == ("pass", None, None)
    assert set(checks) == set(_CHECKS)
    assert {outcome for outcome, _, _ in checks.values()} == {"fail"}
    assert site.issues == seeded


@pytest.mark.parametrize(
    ("key", "value", "check"),
    [
        ("jira.seed.issue_key", "SMOKER-2", "jira.search_issues"),
        ("jira.seed.issue_key", "SMOKER-2", "jira.list_comments"),
        ("jira.seed.comment_id", "10002", "jira.list_comments"),
    ],
)
async def test_a_seeded_object_that_drifted_fails_its_read(
    site: Site, key: str, value: str, check: str
) -> None:
    result = await _run(_adapter({**FIXTURES, key: value}))

    outcome, failure_class, _ = _check_rows(result)[check]
    assert (outcome, failure_class) == ("fail", "assertion_failed")


async def test_a_seeded_issue_whose_description_is_gone_fails_its_read(
    site: Site,
) -> None:
    site.issues[SEED_ISSUE]["fields"]["description"] = None

    result = await _run(_adapter())

    outcome, failure_class, _ = _check_rows(result)["jira.get_issue"]
    assert (outcome, failure_class) == ("fail", "assertion_failed")


async def test_an_issue_that_moved_away_fails_its_read(site: Site) -> None:
    # Jira answers a moved issue's old key with the issue under its new key.
    site.issues[SEED_ISSUE] = {**site.issues[SEED_ISSUE], "key": "SMOKEX-1"}

    result = await _run(_adapter())

    outcome, failure_class, _ = _check_rows(result)["jira.get_issue"]
    assert (outcome, failure_class) == ("fail", "assertion_failed")


async def test_a_comment_whose_body_is_gone_fails_its_read(site: Site) -> None:
    site.comments[SEED_ISSUE][0]["body"] = None

    result = await _run(_adapter())

    outcome, failure_class, _ = _check_rows(result)["jira.list_comments"]
    assert (outcome, failure_class) == ("fail", "assertion_failed")


async def test_a_profile_without_an_account_id_fails_its_read(site: Site) -> None:
    site.me = {"email": "qa@example.invalid", "name": "QA"}

    result = await _run(_adapter())

    outcome, failure_class, _ = _check_rows(result)["jira.current_user"]
    assert (outcome, failure_class) == ("fail", "assertion_failed")


@pytest.mark.usefixtures("writes_on")
async def test_a_grant_that_reaches_another_site_still_reads_the_test_site(
    site: Site,
) -> None:
    site.other_site = True

    result = await _run(_adapter())

    checks = _check_rows(result)
    write = checks.pop("jira.issue_round_trip")
    assert {checks[check] for check in _CHECKS if check in checks} == {
        ("pass", None, None)
    }
    # The tenant pin wants exactly the test site, so nothing is written.
    assert write == ("fail", "tenant_mismatch", "pin_failed")
    assert not [method for method, _ in site.requests if method != "GET"]


# The sweep


@pytest.mark.usefixtures("writes_on")
async def test_the_sweep_deletes_only_old_leftovers_in_the_write_project(
    site: Site,
) -> None:
    now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)

    def run_name(age: timedelta) -> str:
        return f"xagent-smoke-{(now - age).strftime('%Y%m%d%H%M')}-a1b2c3-issue"

    old = site.add(WRITE_PROJECT, run_name(timedelta(hours=3)), "")
    young = site.add(WRITE_PROJECT, run_name(timedelta(minutes=30)), "")
    seeded = site.project_issues(READ_PROJECT)
    tools = _tools()

    result = await run_sweep(
        _adapter(tools=tools), call_timeout=5.0, total_timeout=60.0, now=lambda: now
    )

    _assert_every_issue_call_names_the_site(tools)
    assert len(tools["jira_delete_issue"].calls) == 1
    (row,) = result.to_document()["rows"]
    assert (row["connector"], row["outcome"]) == ("jira", "pass")
    assert old not in site.issues
    assert young in site.issues
    assert site.project_issues(READ_PROJECT) == seeded
