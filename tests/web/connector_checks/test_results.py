"""The run result: session classification, row rules and the JSON document."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from xagent.web.connector_checks.codes import (
    APP_ID_PATTERN,
    CODE_PATTERN,
    TOOL_NAME_PATTERN,
)
from xagent.web.connector_checks.fixtures import FixtureRef
from xagent.web.connector_checks.results import (
    ENVIRONMENT_PATTERN,
    FAIL_CLASSES,
    IDENTITY_PATTERN,
    RESULT_SCHEMA_VERSION,
    SKIP_CLASSES,
    CheckRow,
    FailureClass,
    Outcome,
    RunResult,
    SessionRow,
    classify_session,
    load_result_schema,
)
from xagent.web.connector_probes import UNPROBEABLE_REASONS
from xagent.web.services.mcp_connector_session import (
    ConnectorToolsOutcome,
    ConnectorToolsResult,
)

_START = datetime(2026, 11, 6, 4, 0, tzinfo=timezone.utc)


def _listing(
    outcome: ConnectorToolsOutcome, reason: str | None = None
) -> ConnectorToolsResult:
    return ConnectorToolsResult(outcome, 7, "gmail", reason=reason)


@pytest.mark.parametrize(
    ("listing", "expected"),
    [
        (_listing(ConnectorToolsOutcome.OK), (Outcome.PASS, None, None)),
        (
            _listing(ConnectorToolsOutcome.NOT_CONNECTED, "oauth_token_required"),
            (Outcome.FAIL, FailureClass.NOT_CONNECTED, "oauth_token_required"),
        ),
        (
            _listing(ConnectorToolsOutcome.NOT_CONNECTED, "auth_rejected"),
            (Outcome.FAIL, FailureClass.NOT_CONNECTED, "auth_rejected"),
        ),
        (
            _listing(ConnectorToolsOutcome.NOT_CONNECTED, "client_rejected"),
            (Outcome.FAIL, FailureClass.CLIENT_REJECTED, "client_rejected"),
        ),
        (
            _listing(ConnectorToolsOutcome.FAILED_TO_START, "client_rejected"),
            (Outcome.FAIL, FailureClass.CLIENT_REJECTED, "client_rejected"),
        ),
        (
            _listing(ConnectorToolsOutcome.NOT_CONNECTED, "client_rejected:google"),
            (Outcome.FAIL, FailureClass.CLIENT_REJECTED, "client_rejected:google"),
        ),
        (
            _listing(ConnectorToolsOutcome.FAILED_TO_START, "process_start"),
            (Outcome.FAIL, FailureClass.FAILED_TO_START, "process_start"),
        ),
        (
            _listing(ConnectorToolsOutcome.UNSUPPORTED, "sandboxed"),
            (Outcome.FAIL, FailureClass.UNSUPPORTED, "sandboxed"),
        ),
        (
            _listing(ConnectorToolsOutcome.TIMED_OUT, "list_tools"),
            (Outcome.FAIL, FailureClass.TIMED_OUT, "list_tools"),
        ),
        (
            _listing(ConnectorToolsOutcome.NO_TOOLS, "no_tools_returned"),
            (Outcome.FAIL, FailureClass.NO_TOOLS, "no_tools_returned"),
        ),
        (
            _listing(ConnectorToolsOutcome.NOT_FOUND),
            (Outcome.FAIL, FailureClass.NOT_FOUND, None),
        ),
        (
            _listing(
                ConnectorToolsOutcome.FAILED_TO_START, "Error: token=abc rejected"
            ),
            (Outcome.FAIL, FailureClass.FAILED_TO_START, None),
        ),
    ],
)
def test_a_listing_is_classified_with_a_stable_class(
    listing: ConnectorToolsResult,
    expected: tuple[Outcome, FailureClass | None, str | None],
) -> None:
    assert classify_session(listing) == expected


def test_every_listing_outcome_is_classified() -> None:
    for outcome in ConnectorToolsOutcome:
        classify_session(_listing(outcome))


def _run(*rows: SessionRow | CheckRow, **changes: Any) -> RunResult:
    fields: dict[str, Any] = {
        "identity": "toby",
        "environment": "production-au",
        "started_at": _START,
        "finished_at": _START + timedelta(seconds=95),
        "complete": True,
        "rows": rows,
    }
    fields.update(changes)
    return RunResult(**fields)


def _sample_run() -> RunResult:
    return _run(
        SessionRow("gmail", Outcome.PASS, duration_ms=1800),
        CheckRow(
            "gmail",
            "gmail.l3_probe",
            "read",
            "gmail_search_messages",
            Outcome.PASS,
            duration_ms=900,
        ),
        CheckRow(
            "gmail",
            "gmail.search_seeded",
            "read",
            "gmail_search_messages",
            Outcome.FAIL,
            FailureClass.ASSERTION_FAILED,
            "missing_keys",
            duration_ms=700,
        ),
        CheckRow(
            "gmail",
            "gmail.draft_round_trip",
            "write",
            "gmail_create_draft",
            Outcome.SKIP,
            FailureClass.WRITES_DISABLED,
        ),
        SessionRow("google-docs", Outcome.PASS, duration_ms=1500),
        CheckRow(
            "google-docs",
            "google-docs.l3_probe",
            "read",
            None,
            Outcome.UNPROBEABLE,
            reason="needs_fixture",
        ),
        SessionRow(
            "jira", Outcome.FAIL, FailureClass.CLIENT_REJECTED, "client_rejected", 400
        ),
        CheckRow(
            "jira",
            "jira.l3_probe",
            "read",
            "jira_list_accessible_sites",
            Outcome.SKIP,
            FailureClass.SESSION_UNAVAILABLE,
        ),
        # A write check keeps its own skip class under a failed session.
        CheckRow(
            "jira",
            "jira.issue_round_trip",
            "write",
            "jira_create_issue",
            Outcome.SKIP,
            FailureClass.WRITES_DISABLED,
        ),
        SessionRow(
            "slack", Outcome.SKIP, FailureClass.SKIPPED_BY_IDENTITY, "slack_bot_grant"
        ),
        CheckRow(
            "slack",
            "slack.l3_probe",
            "read",
            "slack_list_channels",
            Outcome.SKIP,
            FailureClass.SESSION_UNAVAILABLE,
        ),
        SessionRow("acme-custom", Outcome.PASS, duration_ms=1100),
        CheckRow(
            "acme-custom",
            "acme-custom.l3_probe",
            "read",
            None,
            Outcome.SKIP,
            FailureClass.NOT_DECLARED,
        ),
    )


def test_the_document_carries_versions_and_rows() -> None:
    document = _sample_run().to_document()

    assert document["schema"] == "xagent.connector_smoke.result"
    assert document["schema_version"] == 1
    assert document["declaration_format_version"] == 1
    assert document["identity"] == "toby"
    assert document["environment"] == "production-au"
    assert document["started_at"] == "2026-11-06T04:00:00Z"
    assert document["finished_at"] == "2026-11-06T04:01:35Z"
    assert document["complete"] is True
    assert document["failure"] is None
    assert document["rows"][0] == {
        "row": "session",
        "connector": "gmail",
        "check": None,
        "kind": None,
        "tool": None,
        "outcome": "pass",
        "failure_class": None,
        "reason": None,
        "duration_ms": 1800,
    }
    assert document["rows"][2] == {
        "row": "check",
        "connector": "gmail",
        "check": "gmail.search_seeded",
        "kind": "read",
        "tool": "gmail_search_messages",
        "outcome": "fail",
        "failure_class": "assertion_failed",
        "reason": "missing_keys",
        "duration_ms": 700,
    }
    assert json.loads(json.dumps(document)) == document


def test_the_document_validates_against_the_packaged_schema() -> None:
    schema = load_result_schema()
    Draft202012Validator.check_schema(schema)

    Draft202012Validator(schema).validate(_sample_run().to_document())
    Draft202012Validator(schema).validate(
        _run(complete=False, environment=None).to_document()
    )
    Draft202012Validator(schema).validate(
        _run(complete=False, failure="enumeration:no_connectors").to_document()
    )


def test_the_schema_pins_the_version_and_the_python_vocabularies() -> None:
    schema = load_result_schema()
    row = schema["$defs"]["row"]["properties"]

    assert schema["properties"]["schema_version"]["const"] == RESULT_SCHEMA_VERSION
    assert set(row["outcome"]["enum"]) == {outcome.value for outcome in Outcome}
    assert set(row["failure_class"]["enum"]) - {None} == {
        failure.value for failure in FailureClass
    }
    pairing = {
        rule["if"]["properties"]["outcome"]["const"]: rule["then"]["properties"]
        for rule in schema["$defs"]["row"]["allOf"]
        if "outcome" in rule["if"]["properties"]
    }
    assert set(pairing["fail"]["failure_class"]["enum"]) == {
        failure.value for failure in FAIL_CLASSES
    }
    assert set(pairing["skip"]["failure_class"]["enum"]) == {
        failure.value for failure in SKIP_CLASSES
    }
    assert set(pairing["unprobeable"]["reason"]["enum"]) == set(UNPROBEABLE_REASONS)


def test_the_schema_patterns_are_the_python_patterns() -> None:
    schema = load_result_schema()
    row = schema["$defs"]["row"]["properties"]

    def anchored(pattern: re.Pattern[str]) -> str:
        return f"^{pattern.pattern}$"

    assert schema["$defs"]["code"]["pattern"] == anchored(CODE_PATTERN)
    assert row["check"]["pattern"] == anchored(CODE_PATTERN)
    assert row["connector"]["pattern"] == anchored(APP_ID_PATTERN)
    assert row["tool"]["pattern"] == anchored(TOOL_NAME_PATTERN)
    assert schema["properties"]["identity"]["pattern"] == anchored(IDENTITY_PATTERN)
    assert schema["properties"]["environment"]["pattern"] == anchored(
        ENVIRONMENT_PATTERN
    )


def test_a_fixture_key_fits_a_fixture_missing_row() -> None:
    key = FixtureRef("google-docs." + "a" * 68).key

    row = CheckRow(
        "google-docs",
        "google-docs.read_seeded",
        "read",
        "google_docs_get_document",
        Outcome.SKIP,
        FailureClass.FIXTURE_MISSING,
        key,
    )

    assert row.to_json()["reason"] == key


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("rows", 0, "reason"), "Bearer abc.def"),
        (("rows", 0, "outcome"), "ok"),
        (("rows", 1, "failure_class"), "vendor said no"),
        (("identity",), "toby:T0123:U0456"),
        (("rows", 0, "body"), {"content": "x"}),
        (("token",), "abc"),
        # Pairings the rows can never produce.
        (("rows", 2, "failure_class"), "writes_disabled"),
        (("rows", 3, "failure_class"), "client_rejected"),
        (("rows", 5, "reason"), None),
        (("failure",), "enumeration:no_connectors"),
        (("rows", 0, "outcome"), "unprobeable"),
    ],
)
def test_the_schema_refuses_free_text_and_unknown_fields(
    path: tuple[str | int, ...], value: Any
) -> None:
    document = _sample_run().to_document()
    target: Any = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    assert not Draft202012Validator(load_result_schema()).is_valid(document)


@pytest.mark.parametrize(
    "build",
    [
        lambda: SessionRow(
            "gmail", Outcome.FAIL, FailureClass.NOT_CONNECTED, "Bearer abc"
        ),
        lambda: SessionRow("gmail", Outcome.PASS, FailureClass.NOT_CONNECTED),
        lambda: SessionRow("gmail", Outcome.FAIL),
        lambda: SessionRow("gmail", Outcome.SKIP, FailureClass.ASSERTION_FAILED),
        lambda: SessionRow("gmail", Outcome.FAIL, FailureClass.WRITES_DISABLED),
        lambda: SessionRow("Gmail", Outcome.PASS),
        lambda: SessionRow("gmail", Outcome.PASS, duration_ms=-1),
        lambda: CheckRow("gmail", "slack.l3_probe", "read", None, Outcome.PASS),
        lambda: CheckRow("gmail", "gmail.l3_probe", "read", "bad tool", Outcome.PASS),
        lambda: CheckRow("gmail", "gmail.l3_probe", "read", None, Outcome.UNPROBEABLE),
        lambda: CheckRow(
            "gmail", "gmail.l3_probe", "read", None, Outcome.UNPROBEABLE, reason="flaky"
        ),
        lambda: CheckRow("gmail", "gmail.l3_probe", "lookup", None, Outcome.PASS),  # type: ignore[arg-type]
        lambda: CheckRow("gmail", "gmail.", "read", None, Outcome.PASS),
    ],
)
def test_a_row_breaking_a_rule_is_refused(build: Any) -> None:
    with pytest.raises(ValueError):
        build()


@pytest.mark.parametrize(
    ("rows", "changes"),
    [
        # A check row without its connector's session row.
        ((CheckRow("gmail", "gmail.l3_probe", "read", None, Outcome.PASS),), {}),
        # Two session rows for one connector.
        ((SessionRow("gmail", Outcome.PASS), SessionRow("gmail", Outcome.PASS)), {}),
        # The same check twice.
        (
            (
                SessionRow("gmail", Outcome.PASS),
                CheckRow("gmail", "gmail.l3_probe", "read", None, Outcome.PASS),
                CheckRow("gmail", "gmail.l3_probe", "read", None, Outcome.PASS),
            ),
            {},
        ),
        # A check that ran although its session failed.
        (
            (
                SessionRow("gmail", Outcome.FAIL, FailureClass.TIMED_OUT),
                CheckRow("gmail", "gmail.l3_probe", "read", None, Outcome.PASS),
            ),
            {},
        ),
        ((), {"identity": "Toby"}),
        # A run that failed to set up is incomplete, empty and names a code.
        ((), {"failure": "enumeration:no_connectors"}),
        (
            (SessionRow("gmail", Outcome.PASS),),
            {"complete": False, "failure": "enumeration:no_connectors"},
        ),
        ((), {"complete": False, "failure": "Enumeration failed"}),
        ((), {"started_at": datetime(2026, 11, 6, 4, 0)}),
        ((), {"finished_at": _START - timedelta(seconds=1)}),
    ],
)
def test_a_run_breaking_a_rule_is_refused(
    rows: tuple[SessionRow | CheckRow, ...], changes: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        _run(*rows, **changes)
