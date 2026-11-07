"""Every connector's V2 declaration: declared, or derived from its L3 probe."""

from __future__ import annotations

import pytest

from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows
from xagent.web.connector_checks.declarations import (
    ConnectorChecks,
    HasKeys,
    L3Envelope,
    ReadCheck,
    UnprobeableCheck,
    validate_connector_checks,
)
from xagent.web.connector_checks.registry import (
    CONNECTOR_CHECKS,
    connector_checks_for,
)


def test_a_probed_connector_gets_its_l3_probe_as_one_read_check() -> None:
    assert connector_checks_for("gmail") == ConnectorChecks(
        "gmail",
        reads=(
            ReadCheck(
                "gmail.l3_probe",
                "gmail_search_messages",
                {"max_results": 1},
                assertions=(L3Envelope(),),
            ),
        ),
    )


def test_the_l3_expect_keys_carry_over() -> None:
    checks = connector_checks_for("linkedin")

    assert checks is not None
    assert checks.reads == (
        ReadCheck(
            "linkedin.l3_probe", "get_profile", assertions=(L3Envelope(("sub",)),)
        ),
    )


@pytest.mark.parametrize(
    ("app_id", "reason"),
    [
        ("google-docs", "needs_fixture"),
        ("hubspot", "covered_by_l2"),
        ("google-maps", "sandboxed"),
        ("chrome-devtools", "execution_scoped"),
    ],
)
def test_an_unprobeable_connector_gets_an_unprobeable_slot(
    app_id: str, reason: str
) -> None:
    assert connector_checks_for(app_id) == ConnectorChecks(
        app_id,
        reads=(UnprobeableCheck(f"{app_id}.l3_probe", reason),),  # type: ignore[arg-type]
    )


def test_an_unknown_app_has_no_declaration() -> None:
    assert connector_checks_for("not-a-catalog-app") is None


@pytest.mark.parametrize(
    ("app_id", "check"),
    [
        (
            "notion",
            ReadCheck(
                "notion.current_user",
                "notion-get-self",
                payload="json_text",
                assertions=(
                    HasKeys((), ("workspace", "user")),
                    HasKeys(("user",), ("id",)),
                ),
                require_read_only_hint=True,
            ),
        ),
        (
            "miro",
            ReadCheck(
                "miro.current_user",
                "user_who_am_i",
                payload="structured_content",
                assertions=(HasKeys((), ("user_id",)),),
                require_read_only_hint=True,
            ),
        ),
        (
            "atlassian",
            ReadCheck(
                "atlassian.current_user",
                "atlassianUserInfo",
                payload="json_text",
                assertions=(HasKeys(("data",), ("accountId",)),),
                require_read_only_hint=True,
            ),
        ),
    ],
)
def test_a_vendor_hosted_read_check_is_declared_exactly(
    app_id: str, check: ReadCheck
) -> None:
    assert connector_checks_for(app_id) == ConnectorChecks(app_id, reads=(check,))


def test_every_remote_mcp_read_check_requires_the_read_only_hint() -> None:
    remote_app_ids = {
        row["app_id"]
        for row in get_builtin_public_mcp_app_rows()
        if "url" in row["launch_config"]
    }
    assert {"notion", "miro", "atlassian"} <= remote_app_ids
    for app_id in remote_app_ids & set(CONNECTOR_CHECKS):
        for check in CONNECTOR_CHECKS[app_id].reads:
            if isinstance(check, ReadCheck):
                assert check.require_read_only_hint, (app_id, check.id)


def test_every_builtin_catalog_app_has_a_valid_declaration() -> None:
    app_ids = sorted(row["app_id"] for row in get_builtin_public_mcp_app_rows())

    assert app_ids
    for app_id in app_ids:
        checks = connector_checks_for(app_id)
        assert checks is not None, app_id
        assert checks.app_id == app_id
        validate_connector_checks(checks)


def test_declared_v2_checks_are_valid_and_keyed_by_their_app() -> None:
    app_ids = {row["app_id"] for row in get_builtin_public_mcp_app_rows()}

    assert set(CONNECTOR_CHECKS) <= app_ids
    for app_id, checks in CONNECTOR_CHECKS.items():
        assert checks.app_id == app_id
        validate_connector_checks(checks)
