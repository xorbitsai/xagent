"""Static request header values of custom MCP connectors are secrets.

Every MCP read surface returns them masked (names stay visible), create rejects
a mask because nothing is stored to retain, and update resolves a mask only to
the stored value of the same header name.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import xagent.web.api.mcp as mcp_module
from xagent.core.tools.core.mcp.model import MASKED_SECRET_VALUE
from xagent.web.api.mcp import (
    MCPAppConnectRequest,
    MCPServerCreate,
    MCPServerUpdate,
    connect_mcp_app,
    create_mcp_server,
    get_mcp_server,
    get_mcp_servers,
    toggle_mcp_server,
    update_mcp_server,
)
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.services.connector_team_scope import (
    set_connector_team_hooks,
    snapshot_connector_team_hooks,
)

URL = "https://mcp.example.test/mcp"
ID_KEY = "X-Proxy-Client-Id"
SECRET_KEY = "X-Proxy-Client-Secret"
ID_VALUE = "proxy-id-1"
SECRET_VALUE = "proxy-secret-1"
STORED = {ID_KEY: ID_VALUE, SECRET_KEY: SECRET_VALUE}
EMPTY_KEY = "X-Proxy-Trace"
# Read surfaces also carry an empty value, which has nothing to hide.
SEED = {**STORED, EMPTY_KEY: ""}
PLAINTEXTS = (ID_VALUE, SECRET_VALUE)


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _make_user(db, user_id: int) -> User:
    user = User(id=user_id, username=f"user-{user_id}", password_hash="x")
    db.add(user)
    db.commit()
    return user


def _make_server(db, headers: dict[str, Any] | None = None) -> MCPServer:
    server = MCPServer.from_config(
        {
            "name": "proxy-connector",
            "managed": "external",
            "transport": "streamable_http",
            "url": URL,
            "headers": dict(STORED if headers is None else headers),
        }
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    return server


def _link(db, user: User, server: MCPServer, *, is_owner: bool) -> None:
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=is_owner,
            is_active=True,
        )
    )
    db.commit()


def _owned_server(db, headers: dict[str, Any] | None = None) -> tuple[User, MCPServer]:
    owner = _make_user(db, 1)
    server = _make_server(db, headers)
    _link(db, owner, server, is_owner=True)
    return owner, server


def _assert_masked(
    response: Any, expected_names: set[str] | None = None, *, empty: bool = False
) -> None:
    headers = dict(response.config["headers"])
    if empty:
        assert headers.pop(EMPTY_KEY) == ""
    assert set(headers) == (expected_names or set(STORED))
    assert all(value == MASKED_SECRET_VALUE for value in headers.values())
    dumped = json.dumps(response.model_dump(mode="json"), default=str)
    for plaintext in PLAINTEXTS:
        assert plaintext not in dumped


def _row(rows, server_id: int):
    matches = [r for r in rows if r.id == server_id and r.transport != "custom_api"]
    assert len(matches) == 1
    return matches[0]


def _read_via(surface: str, db) -> Any:
    if surface == "list_personal":
        owner, server = _owned_server(db, SEED)
        return _row(get_mcp_servers(user_id=None, current_user=owner, db=db), server.id)
    if surface == "list_team_overlay":
        server = _make_server(db, SEED)
        member = _make_user(db, 2)
        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                visibility=lambda _db, _uid: {"mcp": {server.id}, "custom_api": set()}
            )
            return _row(
                get_mcp_servers(user_id=None, current_user=member, db=db), server.id
            )
    if surface == "detail_owner":
        owner, server = _owned_server(db, SEED)
        return get_mcp_server(server.id, current_user=owner, db=db)
    if surface == "connect_mcp_app":
        db.add(
            PublicMCPApp(
                app_id="proxy-app",
                name="proxy-app",
                description="Proxy app",
                transport="stdio",
                launch_config={"command": "npx", "args": ["-y", "proxy-app"]},
            )
        )
        user = _make_user(db, 1)
        db.commit()
        body = MCPAppConnectRequest()
        connect_mcp_app("proxy-app", body, current_user=user, db=db)
        server = db.query(MCPServer).filter(MCPServer.name == "proxy-app").one()
        server.headers = dict(SEED)
        db.commit()
        return connect_mcp_app("proxy-app", body, current_user=user, db=db)
    if surface == "create":
        owner = _make_user(db, 1)
        return create_mcp_server(
            MCPServerCreate(
                name="created-proxy",
                transport="streamable_http",
                config={"url": URL, "headers": dict(SEED)},
            ),
            current_user=owner,
            db=db,
        )
    if surface == "update":
        owner, server = _owned_server(db, SEED)
        return update_mcp_server(
            server.id,
            MCPServerUpdate(description="x"),
            current_user=owner,
            db=db,
        )
    if surface == "toggle":
        owner, server = _owned_server(db, SEED)
        return asyncio.run(toggle_mcp_server(server.id, owner, db))
    raise AssertionError(surface)


@pytest.mark.parametrize(
    "surface",
    [
        "list_personal",
        "list_team_overlay",
        "detail_owner",
        "connect_mcp_app",
        "create",
        "update",
        "toggle",
    ],
    ids=[
        "list_personal",
        "list_team_overlay",
        "detail_owner",
        "connect_mcp_app",
        "create",
        "update",
        "toggle",
    ],
)
def test_mcp_header_values_masked_on_every_read_surface(db, surface):
    response = _read_via(surface, db)
    _assert_masked(response, empty=True)


def test_db_server_to_response_call_sites_are_the_known_seven():
    tree = ast.parse(Path(mcp_module.__file__).read_text())
    callers: list[str] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_db_server_to_response"
            ):
                callers.append(func.name)
    # Nested defs would be double counted by ast.walk on the outer function.
    assert len(callers) == 7
    assert set(callers) == {
        "get_mcp_servers",
        "get_mcp_server",
        "connect_mcp_app",
        "create_mcp_server",
        "update_mcp_server",
        "toggle_mcp_server",
    }


def test_mcp_header_mask_rejected_on_create(db):
    owner = _make_user(db, 1)
    servers_before = db.query(MCPServer).count()
    links_before = db.query(UserMCPServer).count()

    with pytest.raises(HTTPException) as exc:
        create_mcp_server(
            MCPServerCreate(
                name="created-proxy",
                transport="streamable_http",
                config={"url": URL, "headers": {SECRET_KEY: MASKED_SECRET_VALUE}},
            ),
            current_user=owner,
            db=db,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == (
        "Invalid configuration: Masked secret 'X-Proxy-Client-Secret' "
        "has no stored value; provide a new value"
    )
    assert db.query(MCPServer).count() == servers_before
    assert db.query(UserMCPServer).count() == links_before


_ABSENT = object()


@pytest.mark.parametrize(
    "case, incoming, stored_after, response_names",
    [
        (
            "mask_same_key",
            {ID_KEY: MASKED_SECRET_VALUE, SECRET_KEY: MASKED_SECRET_VALUE},
            STORED,
            {ID_KEY, SECRET_KEY},
        ),
        (
            "new_value",
            {ID_KEY: MASKED_SECRET_VALUE, SECRET_KEY: "proxy-secret-2"},
            {ID_KEY: ID_VALUE, SECRET_KEY: "proxy-secret-2"},
            {ID_KEY, SECRET_KEY},
        ),
        (
            "omitted_key",
            {ID_KEY: MASKED_SECRET_VALUE},
            {ID_KEY: ID_VALUE},
            {ID_KEY},
        ),
        ("mask_unknown_key", {"X-Other": MASKED_SECRET_VALUE}, None, None),
        (
            "mask_case_variant",
            {ID_KEY.lower(): MASKED_SECRET_VALUE},
            None,
            None,
        ),
        ("headers_absent", _ABSENT, STORED, {ID_KEY, SECRET_KEY}),
        ("headers_empty", {}, {}, set()),
    ],
    ids=[
        "mask_same_key",
        "new_value",
        "omitted_key",
        "mask_unknown_key",
        "mask_case_variant",
        "headers_absent",
        "headers_empty",
    ],
)
def test_mcp_header_mask_merge_on_update(
    db, case, incoming, stored_after, response_names
):
    owner, server = _owned_server(db)
    config: dict[str, Any] = {"url": URL}
    if incoming is not _ABSENT:
        config["headers"] = incoming
    payload = MCPServerUpdate(config=config)

    if stored_after is None:
        with pytest.raises(HTTPException) as exc:
            update_mcp_server(server.id, payload, current_user=owner, db=db)
        assert exc.value.status_code == 400
        assert "has no stored value" in str(exc.value.detail)
        db.rollback()
        db.refresh(server)
        assert server.headers == STORED
        return

    response = update_mcp_server(server.id, payload, current_user=owner, db=db)

    db.refresh(server)
    assert (server.headers or {}) == stored_after
    if response_names:
        _assert_masked(response, response_names)
    else:
        assert not response.config.get("headers")
