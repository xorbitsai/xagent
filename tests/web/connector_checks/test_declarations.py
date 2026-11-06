"""The declaration format's rules, checked by ``validate_connector_checks``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, get_args

import pytest

from xagent.core.tools.adapters.vibe.mcp_approval_gate import GateDecisionValue
from xagent.web.connector_checks.declarations import (
    ConnectorChecks,
    DeclarationError,
    EachHasKeys,
    Equals,
    ExpectedVerdict,
    HasKeys,
    IsType,
    L3Envelope,
    Length,
    ReadCheck,
    RunScopedName,
    StatusSuccess,
    TenantPin,
    ToolExemption,
    ToolStep,
    UnprobeableCheck,
    WriteCheck,
    validate_connector_checks,
)
from xagent.web.connector_checks.fixtures import FixtureRef


def _write(**changes: Any) -> WriteCheck:
    check = WriteCheck(
        id="jira.issue_round_trip",
        create=ToolStep(
            "jira_create_issue",
            {
                "project": FixtureRef("jira.project_key"),
                "summary": RunScopedName("issue"),
            },
            assertions=(StatusSuccess(),),
        ),
        verify=ToolStep(
            "jira_search_issues",
            {"jql": RunScopedName("issue")},
            assertions=(Length(("issues",), min=1, max=1),),
        ),
        cleanup=ToolStep("jira_delete_issue", {"summary": RunScopedName("issue")}),
        tenant_pin=TenantPin(
            ToolStep("jira_list_accessible_sites"),
            path=("sites", 0, "id"),
            expected=FixtureRef("jira.site_id"),
        ),
        expected_verdicts={
            "jira_create_issue": "require_approval",
            "jira_delete_issue": "require_approval",
        },
    )
    return replace(check, **changes)


def _checks(**changes: Any) -> ConnectorChecks:
    checks = ConnectorChecks(
        app_id="jira",
        reads=(
            ReadCheck(
                id="jira.list_sites",
                tool="jira_list_accessible_sites",
                assertions=(
                    StatusSuccess(),
                    IsType(("sites",), "array"),
                    EachHasKeys(("sites",), ("id", "name")),
                    Equals(("sites", 0, "id"), FixtureRef("jira.site_id")),
                ),
            ),
            ReadCheck(
                id="jira.search",
                tool="jira_search_issues",
                arguments={"jql": "project = SMOKE", "max_results": 1},
                payload="structured_content",
                assertions=(HasKeys(("issues",)),),
                require_read_only_hint=True,
            ),
            UnprobeableCheck(id="jira.l3_probe", reason="needs_fixture"),
        ),
        writes=(_write(),),
        exemptions=(ToolExemption("jira_transition_issue", "moves a real issue"),),
    )
    return replace(checks, **changes)


def test_a_complete_declaration_is_valid() -> None:
    validate_connector_checks(_checks())


def test_a_connector_with_only_exemptions_is_valid() -> None:
    validate_connector_checks(
        ConnectorChecks(
            "granola", exemptions=(ToolExemption("query_meetings", "no test tenant"),)
        )
    )


def test_the_expected_verdicts_are_the_approval_gate_decisions() -> None:
    assert set(get_args(ExpectedVerdict)) == set(get_args(GateDecisionValue))


def test_the_l3_envelope_alone_is_a_valid_read_check() -> None:
    validate_connector_checks(
        ConnectorChecks(
            "linkedin",
            reads=(
                ReadCheck(
                    "linkedin.l3_probe",
                    "get_profile",
                    assertions=(L3Envelope(("sub",)),),
                ),
            ),
        )
    )


def _read(**changes: Any) -> ReadCheck:
    return replace(ReadCheck("jira.extra", "jira_get_issue"), **changes)


@pytest.mark.parametrize(
    ("checks", "message"),
    [
        (_checks(app_id="Jira"), "app id"),
        (_checks(reads=(_read(id="gmail.extra"),)), "must start with 'jira.'"),
        (_checks(reads=(_read(id="jira.Extra"),)), "check id"),
        (_checks(reads=(_read(), _read())), "duplicate check id"),
        (_checks(writes=(_write(id="jira.list_sites"),)), "duplicate check id"),
        (_checks(reads=(_read(tool=""),)), "tool name"),
        (_checks(reads=(_read(tool="jira get"),)), "tool name"),
        (_checks(reads=(_read(id="jira." + "a" * 76),)), "check id"),
        (
            _checks(
                reads=(_read(assertions=(Equals(("name",), RunScopedName("x")),)),)
            ),
            "run-scoped",
        ),
        (_checks(reads=(_read(arguments={"q": object()}),)), "argument"),
        (_checks(reads=(_read(arguments={1: "x"}),)), "argument"),
        (_checks(reads=(_read(arguments={"q": RunScopedName("x")}),)), "run-scoped"),
        (_checks(reads=(_read(assertions=(Length((), min=2, max=1),)),)), "length"),
        (_checks(reads=(_read(assertions=(Length((), min=-1),)),)), "length"),
        (_checks(reads=(_read(assertions=(HasKeys(()),)),)), "keys"),
        (_checks(reads=(_read(assertions=(EachHasKeys(("a",), ()),)),)), "keys"),
        (_checks(reads=(_read(assertions=(IsType((), "list"),)),)), "type"),  # type: ignore[arg-type]
        (_checks(reads=(_read(assertions=(HasKeys(("a",), (-1,)),)),)), "path"),
        (_checks(reads=(_read(assertions=(HasKeys(("a",), (1.5,)),)),)), "path"),  # type: ignore[arg-type]
        (
            _checks(reads=(_read(assertions=(L3Envelope(), StatusSuccess())),)),
            "L3Envelope",
        ),
        (_checks(reads=(_read(payload="text"),)), "payload"),
        (
            _checks(reads=(UnprobeableCheck("jira.l3_probe", "flaky"),)),  # type: ignore[arg-type]
            "unprobeable reason",
        ),
        (
            _checks(exemptions=(ToolExemption("jira_list_accessible_sites", "dup"),)),
            "both checked and exempted",
        ),
        (
            _checks(exemptions=(ToolExemption("jira_delete_issue", "only cleanup"),)),
            "both checked and exempted",
        ),
        (
            _checks(
                exemptions=(ToolExemption("a_tool", "x"), ToolExemption("a_tool", "y"))
            ),
            "exempted twice",
        ),
        (_checks(exemptions=(ToolExemption("a_tool", "  "),)), "reason"),
        (
            _checks(
                writes=(
                    _write(create=ToolStep("jira_create_issue", {"summary": "fixed"})),
                )
            ),
            "run-scoped name",
        ),
        (
            _checks(writes=(_write(expected_verdicts={"jira_create_issue": "allow"}),)),
            "expected verdict",
        ),
        (
            _checks(
                writes=(
                    _write(
                        expected_verdicts={
                            "jira_create_issue": "allow",
                            "jira_delete_issue": "allow",
                            "jira_other": "allow",
                        }
                    ),
                )
            ),
            "expected verdict",
        ),
        (
            _checks(
                writes=(
                    _write(
                        expected_verdicts={
                            "jira_create_issue": "maybe",
                            "jira_delete_issue": "allow",
                        }
                    ),
                )
            ),
            "expected verdict",
        ),
        (
            _checks(
                reads=(_read(payload="structured_content", assertions=(L3Envelope(),)),)
            ),
            "json_text",
        ),
        (
            _checks(reads=(_read(assertions=(Equals(("a",), {"b": 1}),)),)),  # type: ignore[arg-type]
            "scalar",
        ),
        (
            _checks(reads=(_read(assertions=(Equals(("a",), float("nan")),)),)),
            "finite",
        ),
        (ConnectorChecks("custom-foo"), "result label"),
        (ConnectorChecks("a" * 65), "result label"),
    ],
)
def test_a_declaration_breaking_a_rule_is_refused(
    checks: ConnectorChecks, message: str
) -> None:
    with pytest.raises(DeclarationError, match=message):
        validate_connector_checks(checks)


def test_run_scoped_name_suffixes_are_kebab_case() -> None:
    assert RunScopedName("issue-1").suffix == "issue-1"
    with pytest.raises(ValueError):
        RunScopedName("Issue 1")
