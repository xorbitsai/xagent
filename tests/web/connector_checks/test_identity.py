"""A connector target's name in the result, for builtin and custom apps."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from xagent.web.connector_checks.identity import ConnectorTarget, IdentitySetupError
from xagent.web.connector_checks.results import (
    CheckRow,
    FailureClass,
    Outcome,
    RunResult,
    SessionRow,
)


@pytest.mark.parametrize(
    ("app_id", "connector"),
    [
        ("gmail", "gmail"),
        ("google-docs", "google-docs"),
        ("acme_crm", "custom-9ac3cd146cf9290a"),
        ("Acme CRM", "custom-31365de30849a645"),
        ("a" * 65, "custom-635361c48bb9eab1"),
        ("custom-9ac3cd146cf9290a", "custom-99365d0083e712b0"),
        ("custom-foo", "custom-56c0afbf51db36ac"),
    ],
)
def test_a_target_is_named_by_its_app_id_or_a_stable_custom_label(
    app_id: str, connector: str
) -> None:
    target = ConnectorTarget(app_id, ref=7)

    assert target.connector == connector
    assert ConnectorTarget(app_id).connector == connector


def test_a_generated_label_never_names_another_app() -> None:
    generated = ConnectorTarget("acme_crm")
    literal = ConnectorTarget("custom-9ac3cd146cf9290a")
    assert generated.connector != literal.connector

    started = datetime(2026, 11, 6, 10, 0, tzinfo=timezone.utc)
    result = RunResult(
        identity="toby",
        environment="production-au",
        started_at=started,
        finished_at=started + timedelta(seconds=5),
        complete=True,
        rows=(
            SessionRow(generated.connector, Outcome.PASS),
            SessionRow(literal.connector, Outcome.PASS),
        ),
    )

    rows = result.to_document()["rows"]
    assert [row["row"] for row in rows] == ["session", "session"]
    assert [row["connector"] for row in rows] == [
        generated.connector,
        literal.connector,
    ]


@pytest.mark.parametrize("app_id", ["acme_crm", "Acme CRM", "a" * 100, "a" * 64])
def test_any_catalog_app_id_fits_the_rows_of_an_undeclared_app(app_id: str) -> None:
    connector = ConnectorTarget(app_id).connector

    row = CheckRow(
        connector,
        f"{connector}.l3_probe",
        "read",
        None,
        Outcome.SKIP,
        FailureClass.NOT_DECLARED,
    )

    assert row.connector == connector


@pytest.mark.parametrize(
    ("app_id", "skip_reason"),
    [("", None), ("gmail", "slack bot grant only"), ("gmail", "")],
)
def test_a_target_needs_an_app_id_and_a_code_shaped_skip_reason(
    app_id: str, skip_reason: str | None
) -> None:
    with pytest.raises(ValueError):
        ConnectorTarget(app_id, skip_reason=skip_reason)


def test_an_identity_setup_error_carries_a_stage_reason_code() -> None:
    error = IdentitySetupError("identity", "not_allowed")

    assert error.code == "identity:not_allowed"
    assert str(error) == "identity:not_allowed"


@pytest.mark.parametrize(
    ("stage", "reason"),
    [("Identity", "x"), ("a:b", "c"), ("identity", "not allowed")],
)
def test_an_identity_setup_error_refuses_a_stage_or_reason_that_is_no_code(
    stage: str, reason: str
) -> None:
    with pytest.raises(ValueError):
        IdentitySetupError(stage, reason)
