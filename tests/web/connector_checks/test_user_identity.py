"""The ``user`` identity: one local probe user's connected catalog connectors.

The tests run the real enumeration and connector session against a seeded
catalog, with the MCP loader replaced by a fake whose tools are scripted.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.web.services.test_connector_probe import (
    _chartmogul_ok,
    _connect_catalog_chartmogul,
    _connect_catalog_stripe,
    _custom_row,
    _github_ok,
    _Loader,
    _stripe_ok,
)
from tests.web.services.test_mcp_connector_session import (
    LOADER,
    _connect_github,
    _link,
)
from xagent.web.connector_checks import user_identity
from xagent.web.connector_checks.engine import run_checks
from xagent.web.connector_checks.identity import (
    ConnectorTarget,
    IdentitySetupError,
    SessionPurpose,
)
from xagent.web.connector_checks.user_identity import UserIdentityAdapter
from xagent.web.models.mcp import MCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.services.mcp_connector_session import (
    ConnectorToolsOutcome,
    ConnectorToolsResult,
)

from .fakes import rows

pytestmark = pytest.mark.usefixtures("probe_users")


# --- who may probe ---------------------------------------------------------


async def test_the_users_connected_catalog_connectors_run_their_checks(
    db_users, monkeypatch
):
    db, user, _ = db_users
    _connect_catalog_chartmogul(db, user)
    _connect_catalog_stripe(db, user)
    chartmogul, stripe = _chartmogul_ok(), _stripe_ok()
    loader = _Loader(chartmogul, stripe)
    monkeypatch.setattr(LOADER, loader)

    result = await run_checks(UserIdentityAdapter(db, "monitor"))

    assert result.identity == "user"
    assert result.failure is None
    assert rows(result) == [
        ("chartmogul", None, "pass", None, None),
        ("chartmogul", "chartmogul.l3_probe", "pass", None, None),
        ("stripe", None, "pass", None, None),
        ("stripe", "stripe.l3_probe", "pass", None, None),
    ]
    # One listing per connector, then one call per check.
    assert len(loader.calls) == 2
    assert chartmogul.calls == [{"per_page": 1}]
    assert len(stripe.calls) == 1


async def test_a_user_outside_the_probe_users_is_refused_before_anything_starts(
    db_users, monkeypatch
):
    db, _, other_user = db_users
    # An admin is no exception: the allow-list alone decides.
    _connect_github(db, other_user)
    loader = _Loader(_github_ok())
    monkeypatch.setattr(LOADER, loader)

    result = await run_checks(UserIdentityAdapter(db, "other"))

    assert result.failure == "identity:user_not_allowed"
    assert result.rows == ()
    assert loader.calls == []


async def test_nobody_may_probe_when_the_probe_users_are_unset(db_users, monkeypatch):
    db, user, _ = db_users
    monkeypatch.delenv("XAGENT_CONNECTOR_PROBE_USERS")
    _connect_github(db, user)
    loader = _Loader(_github_ok())
    monkeypatch.setattr(LOADER, loader)

    result = await run_checks(UserIdentityAdapter(db, "monitor"))

    assert result.failure == "identity:user_not_allowed"
    assert loader.calls == []


async def test_an_unknown_user_is_refused(db_users):
    db, _, _ = db_users

    result = await run_checks(UserIdentityAdapter(db, "nobody"))

    assert result.failure == "identity:user_not_found"
    assert result.rows == ()


@pytest.mark.parametrize("name", ["monitor@example.com", " MONITOR@Example.com "])
async def test_a_user_is_found_by_email_case_insensitively(
    db_users, monkeypatch, name: str
):
    db, user, _ = db_users
    _connect_github(db, user)
    monkeypatch.setattr(LOADER, _Loader(_github_ok()))

    result = await run_checks(UserIdentityAdapter(db, name))

    assert rows(result)[0] == ("github", None, "pass", None, None)


async def test_a_github_grant_is_still_judged_where_the_checks_need_fixtures(
    db_users, monkeypatch
):
    """The user identity resolves no fixture, so the five fixture-bound GitHub
    checks skip without a call; the probe-derived row reads none, so a revoked
    grant still fails a row instead of the connector reading green."""
    db, user, _ = db_users
    _connect_github(db, user)
    monkeypatch.setattr(LOADER, _Loader(_github_ok()))

    result = await run_checks(UserIdentityAdapter(db, "monitor"))

    fixture_missing = ("skip", "fixture_missing")
    assert rows(result) == [
        ("github", None, "pass", None, None),
        ("github", "github.l3_probe", "pass", None, None),
        ("github", "github.current_user", *fixture_missing, "github.login"),
        ("github", "github.read_repository", *fixture_missing, "github.read_repo"),
        ("github", "github.read_issue", *fixture_missing, "github.read_repo"),
        ("github", "github.read_commits", *fixture_missing, "github.read_repo"),
        ("github", "github.read_pull_request", *fixture_missing, "github.read_repo"),
    ]


async def test_the_allow_list_is_matched_on_the_user_found(db_users, monkeypatch):
    """Named by username, allowed by email: one account, one rule
    (``is_connector_probe_user``)."""
    db, user, _ = db_users
    monkeypatch.setenv("XAGENT_CONNECTOR_PROBE_USERS", "Monitor@Example.com")
    _connect_github(db, user)
    monkeypatch.setattr(LOADER, _Loader(_github_ok()))

    result = await run_checks(UserIdentityAdapter(db, "monitor"))

    assert result.failure is None


# --- which connectors ------------------------------------------------------


async def test_no_connected_connector_is_a_run_failure(db_users):
    db, _, _ = db_users

    result = await run_checks(UserIdentityAdapter(db, "monitor"))

    assert result.failure == "enumeration:no_connectors"


async def test_a_custom_server_is_not_a_target(db_users, monkeypatch):
    db, user, _ = db_users
    custom = MCPServer(name="mine", managed="external", transport="stdio")
    custom.command = "python"
    db.add(custom)
    db.commit()
    _link(db, user, custom)
    _connect_catalog_stripe(db, user)
    monkeypatch.setattr(LOADER, _Loader(_stripe_ok()))

    result = await run_checks(UserIdentityAdapter(db, "monitor"))

    assert [row[0] for row in rows(result)] == ["stripe", "stripe"]


def _lookalike_rows() -> list[tuple[Any, ...]]:
    return [
        ("stripe", None, "skip", "skipped_by_identity", "not_catalog_identity"),
        ("stripe", "stripe.l3_probe", "skip", "session_unavailable", None),
    ]


async def test_a_custom_row_stamped_with_a_catalog_app_id_is_not_a_target(
    db_users, monkeypatch
):
    """Its ``auth`` is caller-authored, so the Connectors page does not show
    the app as connected through it."""
    db, user, _ = db_users
    server = _custom_row(
        "my-stripe", transport="streamable_http", url="https://evil.example.com/mcp"
    )
    db.add(server)
    db.commit()
    _link(db, user, server)
    loader = _Loader(_stripe_ok())
    monkeypatch.setattr(LOADER, loader)

    result = await run_checks(UserIdentityAdapter(db, "monitor"))

    assert result.failure == "enumeration:no_connectors"
    assert loader.calls == []


def test_a_connected_app_hidden_from_the_catalog_is_not_a_target(db_users):
    db, user, _ = db_users
    _connect_github(db, user)
    _connect_catalog_stripe(db, user)
    adapter = UserIdentityAdapter(db, "monitor")
    assert [t.connector for t in adapter.enumerate_targets()] == ["github", "stripe"]
    db.query(PublicMCPApp).filter_by(app_id="stripe").update(
        {"is_visible_in_connector": False}
    )
    db.commit()

    assert [t.connector for t in adapter.enumerate_targets()] == ["github"]


@pytest.mark.parametrize("command", ["node", "npx"])
async def test_a_catalog_row_whose_command_drifted_is_skipped(
    db_users, monkeypatch, command: str
):
    """The page shows the app connected through the shared row, but the row
    no longer launches what the catalog launches: the probe route's
    catalog-identity rule refuses it, and nothing starts."""
    db, user, _ = db_users
    server = _connect_catalog_stripe(db, user)
    server.command = command
    server.args = ["-y", "some-pkg"]
    db.commit()
    loader = _Loader(_stripe_ok())
    monkeypatch.setattr(LOADER, loader)

    result = await run_checks(UserIdentityAdapter(db, "monitor"))

    assert rows(result) == _lookalike_rows()
    assert loader.calls == []


async def test_a_user_owned_row_under_a_catalog_id_is_skipped(db_users, monkeypatch):
    """Named exactly like the app, but owned by a user: a squat, not the
    shared catalog row."""
    db, user, _ = db_users
    server = MCPServer(
        name="stripe", managed="external", transport="stdio", command="python"
    )
    db.add(server)
    db.commit()
    _link(db, user, server)
    loader = _Loader(_stripe_ok())
    monkeypatch.setattr(LOADER, loader)

    result = await run_checks(UserIdentityAdapter(db, "monitor"))

    assert rows(result) == _lookalike_rows()
    assert loader.calls == []


# --- the session -----------------------------------------------------------


@pytest.mark.parametrize(
    ("purpose", "refused"),
    [(SessionPurpose.CALL_TOOLS, True), (SessionPurpose.LIST_ONLY, False)],
)
async def test_a_sandboxed_connection_is_refused_only_when_tools_will_be_called(
    db_users, monkeypatch, purpose: SessionPurpose, refused: bool
):
    db, user, _ = db_users
    server = _connect_github(db, user)
    calls: list[dict[str, Any]] = []

    async def listing(db: Any, **kwargs: Any) -> ConnectorToolsResult:
        calls.append(kwargs)
        return ConnectorToolsResult(ConnectorToolsOutcome.OK, kwargs["server_id"])

    monkeypatch.setattr(user_identity, "list_connector_tools", listing)
    adapter = UserIdentityAdapter(db, "monitor")
    (target,) = adapter.enumerate_targets()

    await adapter.open_session(target, purpose=purpose)

    (call,) = calls
    assert call["user"].id == user.id
    assert call["server_id"] == server.id
    assert call["refuse_sandboxed"] is refused
    # The user's own connectors: no team scope and no actor policy.
    assert "connector_team_id" not in call
    assert "mcp_runtime_authorization_policy" not in call


async def test_a_session_needs_the_enumeration_first(db_users):
    db, _, _ = db_users

    with pytest.raises(RuntimeError):
        await UserIdentityAdapter(db, "monitor").open_session(
            ConnectorTarget("github", ref=1), purpose=SessionPurpose.CALL_TOOLS
        )


def test_the_enumeration_raises_the_setup_error_itself(db_users):
    db, _, _ = db_users

    with pytest.raises(IdentitySetupError) as raised:
        UserIdentityAdapter(db, "other").enumerate_targets()

    assert raised.value.code == "identity:user_not_allowed"


def test_no_fixture_has_a_value_yet(db_users):
    db, _, _ = db_users

    assert UserIdentityAdapter(db, "monitor").resolve_fixture("jira.project") is None
