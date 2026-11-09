"""The declaration format's rules, checked by ``validate_connector_checks``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, get_args

import pytest

from xagent.core.tools.adapters.vibe.mcp_approval_gate import GateDecisionValue
from xagent.web.connector_checks.declarations import (
    ConnectorChecks,
    CreatedRef,
    DeclarationError,
    EachHasKeys,
    Equals,
    ExpectedVerdict,
    HasKeys,
    IsType,
    ItemRef,
    L3Envelope,
    Length,
    ReadCheck,
    RunScopedName,
    StatusSuccess,
    Sweep,
    TenantPin,
    ToolExemption,
    ToolStep,
    UnprobeableCheck,
    WriteCheck,
    validate_connector_checks,
)
from xagent.web.connector_checks.fixtures import FixtureRef

_TENANT = FixtureRef("jira.tenant_id")
_ISSUE_KEY = CreatedRef(("issue", "key"))
_ISSUE_NAME = RunScopedName("issue")


def _pin(**changes: Any) -> TenantPin:
    pin = TenantPin(
        ToolStep(
            "jira_list_accessible_sites",
            assertions=(Length(("sites",), min=1, max=1),),
        ),
        path=("sites", 0, "id"),
        expected=_TENANT,
    )
    return replace(pin, **changes)


def _by_key(tool: str, *assertions: Any) -> ToolStep:
    return ToolStep(
        tool, {"cloud_id": _TENANT, "issue_key": _ISSUE_KEY}, assertions=assertions
    )


def _write(**changes: Any) -> WriteCheck:
    check = WriteCheck(
        id="jira.issue_round_trip",
        create=ToolStep(
            "jira_create_issue",
            {
                "cloud_id": _TENANT,
                "project_key": FixtureRef("jira.write_project_key"),
                "summary": _ISSUE_NAME,
            },
            assertions=(StatusSuccess(), IsType(("issue", "key"), "string")),
        ),
        verify=_by_key("jira_get_issue", Equals(("issue", "summary"), _ISSUE_NAME)),
        cleanup=_by_key("jira_delete_issue", StatusSuccess()),
        cleanup_guard=_by_key(
            "jira_get_issue", Equals(("issue", "summary"), _ISSUE_NAME)
        ),
        tenant_pin=_pin(),
        expected_verdicts={"jira_create_issue": "deny", "jira_delete_issue": "deny"},
    )
    return replace(check, **changes)


def _sweep(**changes: Any) -> Sweep:
    sweep = Sweep(
        tenant_pin=_pin(),
        find=ToolStep(
            "jira_search_issues",
            {
                "cloud_id": _TENANT,
                "jql": FixtureRef("jira.write_sweep_jql"),
                "raw_fields": False,
            },
            assertions=(IsType(("issues",), "array"),),
        ),
        items=("issues",),
        name=("summary",),
        delete=ToolStep(
            "jira_delete_issue",
            {"cloud_id": _TENANT, "issue_key": ItemRef(("key",))},
            assertions=(StatusSuccess(),),
        ),
    )
    return replace(sweep, **changes)


def _create(**arguments: Any) -> ToolStep:
    return ToolStep(
        "jira_create_issue",
        {
            "project_key": FixtureRef("jira.write_project_key"),
            "summary": _ISSUE_NAME,
            **arguments,
        },
        assertions=(StatusSuccess(),),
    )


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
                    Equals(("sites", 0, "id"), _TENANT),
                ),
            ),
            ReadCheck(
                id="jira.search",
                tool="jira_search_issues",
                arguments={"jql": "project = SMOKE", "max_results": 1},
                payload="structured_content",
                assertions=(HasKeys((), ("issues",)),),
                require_read_only_hint=True,
            ),
            UnprobeableCheck(id="jira.l3_probe", reason="needs_fixture"),
        ),
        writes=(_write(),),
        sweep=_sweep(),
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
    check = ReadCheck("jira.extra", "jira_get_issue", assertions=(StatusSuccess(),))
    return replace(check, **changes)


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
        (_checks(reads=(_read(assertions=(HasKeys((), ()),)),)), "keys"),
        (_checks(reads=(_read(assertions=(EachHasKeys(("a",), ()),)),)), "keys"),
        (_checks(reads=(_read(assertions=(IsType((), "list"),)),)), "type"),  # type: ignore[arg-type]
        (_checks(reads=(_read(assertions=(HasKeys((-1,), ("a",)),)),)), "path"),
        (_checks(reads=(_read(assertions=(HasKeys((1.5,), ("a",)),)),)), "path"),  # type: ignore[arg-type]
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
        (_checks(writes=(_write(create=_create(summary="fixed")),)), "run-scoped name"),
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
        (
            # A cleanup cannot target a seeded object.
            _checks(
                writes=(
                    _write(
                        cleanup=ToolStep(
                            "jira_delete_issue",
                            {"issue_key": FixtureRef("jira.seed.issue")},
                            assertions=(StatusSuccess(),),
                        ),
                        cleanup_guard=None,
                    ),
                )
            ),
            "cleanup",
        ),
        (_checks(reads=(_read(assertions=()),)), "assertion"),
        (
            _checks(writes=(_write(cleanup=_by_key("jira_delete_issue")),)),
            "assertion",
        ),
        (_checks(reads=(_write(),), writes=()), "reads"),
        (_checks(writes=(_read(),)), "writes"),
        (
            _checks(writes=(_write(tenant_pin=_pin(expected="site-1")),)),
            "tenant pin",
        ),
        (
            _checks(
                writes=(
                    _write(
                        tenant_pin=_pin(
                            step=ToolStep(
                                "jira_list_accessible_sites", {"q": RunScopedName("x")}
                            )
                        )
                    ),
                )
            ),
            "run-scoped",
        ),
        (_checks(writes=(_write(tenant_pin=_pin(path=("sites", -1))),)), "path"),
        # Created-object and listed-item references
        (
            _checks(writes=(_write(create=_create(parent=_ISSUE_KEY)),)),
            "created-object reference",
        ),
        (
            _checks(reads=(_read(arguments={"issue_key": _ISSUE_KEY}),)),
            "created-object reference",
        ),
        (
            _checks(reads=(_read(assertions=(Equals(("key",), _ISSUE_KEY),)),)),
            "created-object reference",
        ),
        (
            _checks(
                writes=(
                    _write(
                        tenant_pin=_pin(
                            step=ToolStep(
                                "jira_list_accessible_sites", {"q": _ISSUE_KEY}
                            )
                        )
                    ),
                )
            ),
            "created-object reference",
        ),
        (
            _checks(
                writes=(
                    _write(
                        cleanup=ToolStep(
                            "jira_delete_issue",
                            {"issue_key": ItemRef(("key",))},
                            assertions=(StatusSuccess(),),
                        ),
                        cleanup_guard=None,
                    ),
                )
            ),
            "listed-item reference",
        ),
        # The cleanup guard
        (_checks(writes=(_write(cleanup_guard=None),)), "guard"),
        (
            # Another write check's cleanup tool is no read either.
            _checks(
                writes=(
                    _write(),
                    _write(
                        id="jira.archive_round_trip",
                        cleanup=_by_key("jira_archive_issue", StatusSuccess()),
                        cleanup_guard=_by_key(
                            "jira_delete_issue",
                            Equals(("issue", "summary"), _ISSUE_NAME),
                        ),
                        expected_verdicts={
                            "jira_create_issue": "deny",
                            "jira_archive_issue": "deny",
                        },
                    ),
                )
            ),
            "create or cleanup tool",
        ),
        (
            _checks(
                writes=(
                    _write(
                        cleanup=ToolStep(
                            "jira_delete_issue",
                            {"summary": RunScopedName("other")},
                            assertions=(StatusSuccess(),),
                        ),
                        cleanup_guard=None,
                    ),
                )
            ),
            "object create named",
        ),
        (
            _checks(
                writes=(
                    _write(
                        expected_verdicts=["jira_create_issue", "jira_delete_issue"]
                    ),
                )
            ),
            "mapping",
        ),
        (_checks(reads=(_read(id=1),)), "check id"),
        (_checks(reads=(_read(assertions=(HasKeys("issues", ("a",)),)),)), "tuple"),  # type: ignore[arg-type]
        (_checks(writes=(_write(tenant_pin=_pin(path="id")),)), "tuple"),
        (
            _checks(
                writes=(
                    _write(
                        create=_create(project_key=FixtureRef("slack.write_channel_id"))
                    ),
                )
            ),
            "write area",
        ),
        (
            _checks(
                writes=(
                    _write(
                        cleanup_guard=_by_key(
                            "jira_delete_issue",
                            Equals(("issue", "summary"), _ISSUE_NAME),
                        )
                    ),
                )
            ),
            "create or cleanup tool",
        ),
        (
            _checks(
                writes=(
                    _write(
                        cleanup=ToolStep(
                            "jira_delete_issue",
                            {"summary": _ISSUE_NAME},
                            assertions=(StatusSuccess(),),
                        )
                    ),
                )
            ),
            "guard",
        ),
        (
            _checks(
                writes=(
                    _write(
                        cleanup_guard=ToolStep(
                            "jira_get_issue",
                            {"issue_key": CreatedRef(("issue", "id"))},
                            assertions=(Equals(("issue", "summary"), _ISSUE_NAME),),
                        )
                    ),
                )
            ),
            "guard",
        ),
        (
            _checks(
                writes=(
                    _write(cleanup_guard=_by_key("jira_get_issue", StatusSuccess())),
                )
            ),
            "guard",
        ),
        (
            _checks(
                writes=(
                    _write(
                        cleanup_guard=_by_key(
                            "jira_get_issue",
                            Equals(("issue", "summary"), RunScopedName("other")),
                        )
                    ),
                )
            ),
            "guard",
        ),
        # The tenant's write area and literal addresses
        (
            _checks(
                writes=(
                    _write(create=_create(project_key=FixtureRef("other.project_key"))),
                )
            ),
            "write area",
        ),
        (
            _checks(writes=(_write(create=_create(project_key="SMOKE")),)),
            "write area",
        ),
        (
            # The pin's own key names the tenant, not the write area.
            _checks(
                writes=(_write(create=_create(project_key="PROD", cloud_id=_TENANT)),)
            ),
            "write area",
        ),
        (
            _checks(
                writes=(
                    _write(
                        create=_create(project_key=FixtureRef("jira.seed.project_key"))
                    ),
                )
            ),
            "write area",
        ),
        (
            _checks(writes=(_write(create=_create(to={"qa@example.com": True})),)),
            "email",
        ),
        (
            _checks(writes=(_write(create=_create(assignee="qa@example.com")),)),
            "email",
        ),
        (
            _checks(
                writes=(
                    _write(create=_create(watchers=["a", {"user": "qa@example.com"}])),
                )
            ),
            "email",
        ),
        # The sweep
        (_checks(sweep=None), "sweep"),
        (_checks(writes=()), "sweep"),
        (_checks(sweep=_write()), "Sweep"),
        (
            _checks(
                sweep=_sweep(
                    delete=ToolStep(
                        "jira_delete_issue",
                        {"issue_key": "SMOKE-1"},
                        assertions=(StatusSuccess(),),
                    )
                )
            ),
            "listed-item reference",
        ),
        (
            _checks(
                sweep=_sweep(
                    delete=ToolStep(
                        "jira_transition_issue",
                        {"issue_key": ItemRef(("key",))},
                        assertions=(StatusSuccess(),),
                    )
                )
            ),
            "cleanup tool",
        ),
        (
            _checks(
                sweep=_sweep(
                    delete=ToolStep(
                        "jira_delete_issue",
                        {"issue_key": ItemRef(("key",))},
                    )
                )
            ),
            "assertion",
        ),
        (
            _checks(
                sweep=_sweep(
                    delete=ToolStep(
                        "jira_delete_issue",
                        {"issue_key": _ISSUE_KEY},
                        assertions=(StatusSuccess(),),
                    )
                )
            ),
            "created-object reference",
        ),
        (
            _checks(
                sweep=_sweep(
                    find=ToolStep(
                        "jira_search_issues",
                        {"cloud_id": _TENANT, "jql": _ISSUE_NAME},
                        assertions=(IsType(("issues",), "array"),),
                    )
                )
            ),
            "run-scoped",
        ),
        (
            _checks(
                sweep=_sweep(
                    find=ToolStep(
                        "jira_search_issues",
                        {"cloud_id": _TENANT, "start": ItemRef(("key",))},
                        assertions=(IsType(("issues",), "array"),),
                    )
                )
            ),
            "listed-item reference",
        ),
        (
            _checks(
                sweep=_sweep(
                    find=ToolStep(
                        "jira_search_issues",
                        {"jql": "project = SMOKE"},
                        assertions=(IsType(("issues",), "array"),),
                    )
                )
            ),
            "write area",
        ),
        (
            _checks(
                sweep=_sweep(
                    find=ToolStep(
                        "jira_search_issues",
                        {"cloud_id": _TENANT, "jql": "reporter = qa@example.com"},
                        assertions=(IsType(("issues",), "array"),),
                    )
                )
            ),
            "email",
        ),
        (
            _checks(
                sweep=_sweep(find=ToolStep("jira_search_issues", {"cloud_id": _TENANT}))
            ),
            "assertion",
        ),
        (
            _checks(
                sweep=_sweep(tenant_pin=_pin(expected=FixtureRef("jira.other_id")))
            ),
            "same fixture key",
        ),
        (_checks(sweep=_sweep(items=("issues", -1))), "path"),
        (_checks(sweep=_sweep(items="issues")), "path"),
        (_checks(sweep=_sweep(name=())), "name path"),
        (_checks(sweep=_sweep(name="summary")), "name path"),
        (_checks(sweep=_sweep(name=("summary", -1))), "name path"),
        (ConnectorChecks("custom-foo"), "result label"),
        (ConnectorChecks("a" * 65), "result label"),
    ],
)
def test_a_declaration_breaking_a_rule_is_refused(
    checks: ConnectorChecks, message: str
) -> None:
    with pytest.raises(DeclarationError, match=message):
        validate_connector_checks(checks)


def test_a_cleanup_by_run_scoped_name_needs_no_guard() -> None:
    cleanup = ToolStep(
        "jira_delete_issue",
        {"cloud_id": _TENANT, "summary": _ISSUE_NAME},
        assertions=(StatusSuccess(),),
    )
    validate_connector_checks(
        _checks(writes=(_write(cleanup=cleanup, cleanup_guard=None),))
    )


def test_checked_tools_leave_out_the_tenant_pin_the_guard_and_the_sweep() -> None:
    checks = ConnectorChecks("jira", writes=(_write(),), sweep=_sweep())
    # The guard's tool is not verify's here.
    verify = ToolStep(
        "jira_search_issues",
        {"cloud_id": _TENANT, "jql": _ISSUE_NAME},
        assertions=(Length(("issues",), min=1, max=1),),
    )
    guarded = ConnectorChecks("jira", writes=(_write(verify=verify),), sweep=_sweep())

    # The pin lists sites and the sweep searches: neither is a subject.
    assert checks.checked_tools() == {
        "jira_create_issue",
        "jira_get_issue",
        "jira_delete_issue",
    }
    # The guard reads with jira_get_issue: not a subject either.
    assert guarded.checked_tools() == {
        "jira_create_issue",
        "jira_search_issues",
        "jira_delete_issue",
    }


@pytest.mark.parametrize("reference", [CreatedRef, ItemRef])
@pytest.mark.parametrize("path", [(), ("a", -1), ("a", True), ["issue", "key"]])
def test_a_reference_takes_a_non_empty_payload_path(
    reference: type[CreatedRef] | type[ItemRef], path: Any
) -> None:
    with pytest.raises(ValueError, match="path"):
        reference(path)


def test_a_write_area_key_may_be_in_the_writing_connector_s_namespace() -> None:
    # The registry names a tenant family's write area per connector
    # (google-drive.write_folder_id) while the pin is the tenant's.
    name = RunScopedName("doc")
    pin = TenantPin(
        ToolStep("drive_about", assertions=(StatusSuccess(),)),
        path=("user", "email"),
        expected=FixtureRef("google.tenant_id"),
    )
    folder = FixtureRef("google-drive.write_folder_id")
    validate_connector_checks(
        ConnectorChecks(
            "google-drive",
            writes=(
                WriteCheck(
                    "google-drive.file_round_trip",
                    create=ToolStep(
                        "drive_create_file",
                        {"parent": folder, "name": name},
                        assertions=(StatusSuccess(),),
                    ),
                    verify=ToolStep(
                        "drive_find", {"name": name}, assertions=(StatusSuccess(),)
                    ),
                    cleanup=ToolStep(
                        "drive_delete_file",
                        {"name": name},
                        assertions=(StatusSuccess(),),
                    ),
                    tenant_pin=pin,
                    expected_verdicts={
                        "drive_create_file": "allow",
                        "drive_delete_file": "allow",
                    },
                ),
            ),
            sweep=Sweep(
                tenant_pin=pin,
                find=ToolStep(
                    "drive_list", {"parent": folder}, assertions=(StatusSuccess(),)
                ),
                items=("files",),
                name=("name",),
                delete=ToolStep(
                    "drive_delete_file",
                    {"id": ItemRef(("id",))},
                    assertions=(StatusSuccess(),),
                ),
            ),
        )
    )


def test_run_scoped_name_suffixes_are_kebab_case() -> None:
    assert RunScopedName("issue-1").suffix == "issue-1"
    with pytest.raises(ValueError):
        RunScopedName("Issue 1")
