"""A connector target's name in the result, for builtin and custom apps."""

from __future__ import annotations

import pytest

from xagent.web.connector_checks.identity import ConnectorTarget
from xagent.web.connector_checks.results import CheckRow, FailureClass, Outcome


@pytest.mark.parametrize(
    ("app_id", "connector"),
    [
        ("gmail", "gmail"),
        ("google-docs", "google-docs"),
        ("acme_crm", "custom-9ac3cd146cf9290a"),
        ("Acme CRM", "custom-31365de30849a645"),
        ("a" * 65, "custom-635361c48bb9eab1"),
    ],
)
def test_a_target_is_named_by_its_app_id_or_a_stable_custom_label(
    app_id: str, connector: str
) -> None:
    target = ConnectorTarget(app_id, ref=7)

    assert target.connector == connector
    assert ConnectorTarget(app_id).connector == connector


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
