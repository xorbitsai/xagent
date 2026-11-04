"""``GET /api/mcp/servers/{server_id}`` for a caller with no personal row.

A caller with a personal association row is answered from that row and the
connector access hook is never asked. A caller without one reaches the
definition only when the installed access hook reports that the caller's team
links it, and then sees the same projection the list endpoint returns for a
team-owned row. A failing hook surfaces its declared status rather than a 404
or a generic 500.

Every test installs hooks through ``snapshot_connector_team_hooks`` so no hook
state leaks between tests or into suites that run after this one.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.core.tools.core.mcp.model import MASKED_SECRET_VALUE, SENSITIVE_AUTH_FIELDS
from xagent.web.api.mcp import get_mcp_server, get_mcp_servers
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.user import User
from xagent.web.services.connector_team_scope import (
    ConnectorAccess,
    set_connector_team_hooks,
    snapshot_connector_team_hooks,
)

ENV_SEED = {"API_TOKEN": "env-secret-1", "REGION_KEY": "env-secret-2"}


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def _make_user(db, user_id: int, *, is_admin: bool = False) -> User:
    user = User(
        id=user_id, username=f"user-{user_id}", password_hash="x", is_admin=is_admin
    )
    db.add(user)
    db.commit()
    return user


def _make_shared_server(db) -> MCPServer:
    """An external SSE definition carrying secrets in ``env`` and ``auth``, with
    no personal association row for anyone."""
    auth = {"type": "bearer"}
    for index, field in enumerate(SENSITIVE_AUTH_FIELDS):
        auth[field] = f"auth-secret-{index}"
    server = MCPServer.from_config(
        {
            "name": "team-shared",
            "description": "shared by a team",
            "managed": "external",
            "transport": "sse",
            "url": "https://mcp.example.test/sse",
            "env": dict(ENV_SEED),
            "auth": auth,
        }
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    return server


def _link_personally(
    db, user: User, server: MCPServer, *, is_owner: bool, is_active: bool
) -> None:
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=is_owner,
            is_active=is_active,
        )
    )
    db.commit()


def _visibility_listing(*mcp_ids: int):
    def hook(db, user_id):
        return {"mcp": set(mcp_ids), "custom_api": set()}

    return hook


def _recording_access_hook(*, can_edit: bool = False):
    """An access hook that grants every asked ref and counts its calls."""
    calls: list[object] = []

    def hook(db, user_id, refs):
        calls.append(refs)
        return {
            ref: ConnectorAccess(team_owned=True, can_edit=can_edit) for ref in refs
        }

    hook.calls = calls
    return hook


def _detail(server_id, current_user, db):
    return get_mcp_server(server_id, current_user=current_user, db=db)


def _list(current_user, db):
    return get_mcp_servers(user_id=None, current_user=current_user, db=db)


@pytest.mark.parametrize("is_admin", [False, True])
@pytest.mark.parametrize("verdict_can_edit", [False, True])
def test_a_team_member_without_a_personal_row_reads_the_same_projection_the_list_returns(
    db, verdict_can_edit, is_admin
):
    server = _make_shared_server(db)
    member = _make_user(db, 2, is_admin=is_admin)

    with snapshot_connector_team_hooks():
        set_connector_team_hooks(
            visibility=_visibility_listing(server.id),
            access=_recording_access_hook(can_edit=verdict_can_edit),
        )
        listed = [
            row
            for row in _list(member, db)
            if row.id == server.id and row.transport != "custom_api"
        ]
        detail = _detail(server.id, member, db)

    assert len(listed) == 1
    assert detail.model_dump() == listed[0].model_dump()
    assert detail.user_id == member.id
    if not is_admin:
        assert detail.can_edit_global is False
    assert detail.config["env"] == {key: MASKED_SECRET_VALUE for key in ENV_SEED}
    for field in SENSITIVE_AUTH_FIELDS:
        assert detail.config["auth"][field] == MASKED_SECRET_VALUE


def _answers_nothing(db, user_id, refs):
    return {}


def _must_not_be_asked(db, user_id, refs):
    raise AssertionError("the access hook must not be asked")


@pytest.mark.parametrize(
    ("access", "visibility_lists_it", "id_offset"),
    [
        pytest.param(_answers_nothing, False, 0, id="hook_answers_empty"),
        pytest.param(None, False, 0, id="no_hook_installed"),
        pytest.param(_must_not_be_asked, False, 1000, id="unknown_id"),
        pytest.param(_answers_nothing, True, 0, id="listed_by_visibility_only"),
    ],
)
def test_detail_404s_without_a_personal_row_or_a_team_verdict(
    db, access, visibility_lists_it, id_offset
):
    server = _make_shared_server(db)
    member = _make_user(db, 2)
    visibility = _visibility_listing(server.id) if visibility_lists_it else None

    with snapshot_connector_team_hooks():
        set_connector_team_hooks(visibility=visibility, access=access)
        with pytest.raises(HTTPException) as exc:
            _detail(server.id + id_offset, member, db)

    assert exc.value.status_code == 404
    assert exc.value.detail == "MCP server not found"


def test_detail_is_decided_by_the_access_hook_not_the_visibility_hook(db):
    server = _make_shared_server(db)
    member = _make_user(db, 2)

    with snapshot_connector_team_hooks():
        set_connector_team_hooks(
            visibility=_visibility_listing(),
            access=_recording_access_hook(),
        )
        detail = _detail(server.id, member, db)

    assert detail.id == server.id
    assert detail.user_id == member.id


@pytest.mark.parametrize(
    ("is_owner", "is_active"), [(True, True), (False, True), (False, False)]
)
def test_a_personal_row_decides_the_detail_without_asking_the_hook(
    db, is_owner, is_active
):
    server = _make_shared_server(db)
    caller = _make_user(db, 2)
    _link_personally(db, caller, server, is_owner=is_owner, is_active=is_active)
    calls: list[object] = []

    def exploding_access_hook(db, user_id, refs):
        calls.append(refs)
        raise RuntimeError("the access hook must not be asked")

    with snapshot_connector_team_hooks():
        if (is_owner, is_active) == (False, False):
            set_connector_team_hooks(
                visibility=_visibility_listing(server.id),
                access=exploding_access_hook,
            )
        else:
            set_connector_team_hooks(access=exploding_access_hook)
        detail = _detail(server.id, caller, db)

    assert detail.id == server.id
    assert detail.user_id == caller.id
    assert detail.is_active is is_active
    assert calls == []


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_detail"),
    [
        ("generic_exception", 503, "Connector access is unavailable."),
        ("connector_runtime_error", 418, "short and stout"),
        ("malformed_answer", 503, "Connector access is unavailable."),
        ("verdict_for_another_server", 503, "Connector access is unavailable."),
    ],
)
def test_a_failing_access_hook_surfaces_its_status_not_a_500_or_a_404(
    db, failure, expected_status, expected_detail
):
    server = _make_shared_server(db)
    member = _make_user(db, 2)

    def failing_access_hook(db, user_id, refs):
        if failure == "generic_exception":
            raise RuntimeError("team service down")
        if failure == "connector_runtime_error":
            raise ConnectorRuntimeError("teapot", "short and stout", status_code=418)
        if failure == "verdict_for_another_server":
            grant = ConnectorAccess(team_owned=True, can_edit=False)
            return {("mcp", server.id + 1): grant}
        return {ref: ConnectorAccess(team_owned=False, can_edit=False) for ref in refs}

    with snapshot_connector_team_hooks():
        set_connector_team_hooks(access=failing_access_hook)
        with pytest.raises(HTTPException) as exc:
            _detail(server.id, member, db)

    assert exc.value.status_code == expected_status
    assert exc.value.detail == expected_detail
