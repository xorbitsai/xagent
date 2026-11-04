"""Creating or renaming an MCP server or Custom API is rejected when the name
folds (``normalize_mcp_server_name``) to the same key as any other MCP server
or Custom API name.

Stored rows always use un-folded spellings, so a database-side fold that
diverges from the Python normalizer cannot hide behind inputs that are already
folded. Rows are inserted directly; the route functions are called directly.
"""

from __future__ import annotations

import ast
import asyncio
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event, literal, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import xagent.web
from xagent.core.tools.adapters.vibe.selection_spec import normalize_mcp_server_name
from xagent.web.api.custom_api import (
    CustomApiCreate,
    CustomApiUpdate,
    create_custom_api,
    update_custom_api,
)
from xagent.web.api.mcp import (
    MCPServerCreate,
    MCPServerUpdate,
    _ensure_catalog_app_server,
    create_mcp_server,
    update_mcp_server,
)
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.services import connector_team_scope
from xagent.web.services.connector_name_policy import (
    _sql_fold,
    catalog_app_name_detail,
    folded_name_conflict_detail,
    folds_to_catalog_app_name,
    has_folded_connector_name_conflict,
)

_MCP = "mcp"
_API = "custom_api"

_MCP_NAME_OK = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")

# (stored spelling, requested spelling): both un-folded, different from each
# other, and equal after folding. Each pair is sensitive to a different step
# of the fold.
_NAME_PAIRS = [
    pytest.param("Foo-Bar", "FOO_bar", id="hyphen-vs-underscore"),
    pytest.param("ACME", "Acme", id="case"),
    pytest.param("Google Drive", "google_Drive", id="space-vs-underscore"),
    pytest.param(" Padded ", "PADDED", id="surrounding-spaces"),
    pytest.param("foo_bar", "Foo Bar", id="requested-with-space"),
]


@pytest.fixture()
def db() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture(autouse=True)
def _reset_connector_team_hooks() -> Iterator[None]:
    with connector_team_scope.snapshot_connector_team_hooks():
        connector_team_scope.set_connector_team_hooks()
        yield


@pytest.fixture()
def user(db: Session) -> User:
    row = User(username="admin", password_hash="x", is_admin=True)
    db.add(row)
    db.commit()
    return row


def _insert_mcp(
    db: Session, name: str, owner: User | None = None, *, row_id: int | None = None
) -> MCPServer:
    kwargs: dict[str, Any] = {} if row_id is None else {"id": row_id}
    server = MCPServer(
        name=name,
        description="d",
        managed="external",
        transport="streamable_http",
        url="https://example.com/mcp",
        **kwargs,
    )
    db.add(server)
    db.flush()
    if owner is not None:
        db.add(
            UserMCPServer(
                user_id=owner.id,
                mcpserver_id=server.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=True,
            )
        )
    db.commit()
    return server


def _insert_api(
    db: Session, name: str, owner: User | None = None, *, row_id: int | None = None
) -> CustomApi:
    kwargs: dict[str, Any] = {} if row_id is None else {"id": row_id}
    api = CustomApi(name=name, url="https://example.com/api", method="GET", **kwargs)
    db.add(api)
    db.flush()
    if owner is not None:
        db.add(
            UserCustomApi(
                user_id=owner.id,
                custom_api_id=api.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_active=True,
            )
        )
    db.commit()
    return api


def _insert(db: Session, table: str, name: str, owner: User | None = None) -> Any:
    return (_insert_mcp if table == _MCP else _insert_api)(db, name, owner)


def _counts(db: Session) -> tuple[list[str], list[str]]:
    db.rollback()
    return (
        sorted(r[0] for r in db.query(MCPServer.name).all()),
        sorted(r[0] for r in db.query(CustomApi.name).all()),
    )


def _create(db: Session, user: User, entry: str, name: str) -> Any:
    if entry == _MCP:
        return create_mcp_server(
            MCPServerCreate(
                name=name,
                transport="streamable_http",
                config={"url": "https://example.com/new/mcp"},
            ),
            current_user=user,
            db=db,
        )
    return asyncio.run(
        create_custom_api(
            CustomApiCreate(name=name, url="https://example.com/new/api"),
            current_user=user,
            db=db,
        )
    )


def _rename(db: Session, user: User, entry: str, row_id: int, name: str) -> Any:
    if entry == _MCP:
        return update_mcp_server(
            row_id, MCPServerUpdate(name=name), current_user=user, db=db
        )
    return update_custom_api(
        row_id, CustomApiUpdate(name=name), current_user=user, db=db
    )


def _skip_if_mcp_cannot_take(entry: str, name: str) -> None:
    if entry == _MCP and not set(name) <= _MCP_NAME_OK:
        pytest.skip("MCP server names only allow [A-Za-z0-9_-]")


@pytest.mark.parametrize("stored,requested", _NAME_PAIRS)
@pytest.mark.parametrize("stored_in", [_MCP, _API])
@pytest.mark.parametrize("entry", [_MCP, _API])
def test_create_rejects_folded_collision(
    db: Session, user: User, entry: str, stored_in: str, stored: str, requested: str
) -> None:
    _skip_if_mcp_cannot_take(entry, requested)
    _insert(db, stored_in, stored)
    before = _counts(db)

    with pytest.raises(HTTPException) as exc:
        _create(db, user, entry, requested)

    assert exc.value.status_code == 400
    assert exc.value.detail == folded_name_conflict_detail(requested)
    assert not db.new and not db.dirty
    assert _counts(db) == before


@pytest.mark.parametrize("action", ["create", "rename"])
@pytest.mark.parametrize("entry", [_MCP, _API])
def test_exact_duplicate_keeps_existing_message(
    db: Session, user: User, entry: str, action: str
) -> None:
    _insert(db, entry, "jira-same")
    target = _insert(db, entry, "renamed_row", owner=user)
    target_id = target.id

    with pytest.raises(HTTPException) as exc:
        if action == "create":
            _create(db, user, entry, "jira-same")
        else:
            _rename(db, user, entry, target_id, "jira-same")

    assert exc.value.status_code == 400
    expected = (
        "MCP server 'jira-same' already exists"
        if entry == _MCP
        else "Custom API with name 'jira-same' already exists"
    )
    assert exc.value.detail == expected


@pytest.mark.parametrize("stored_in,entry", [(_MCP, _API), (_API, _MCP)])
def test_create_exact_name_in_other_table_is_a_folded_conflict(
    db: Session, user: User, stored_in: str, entry: str
) -> None:
    _insert(db, stored_in, "jira")

    with pytest.raises(HTTPException) as exc:
        _create(db, user, entry, "jira")

    assert exc.value.status_code == 400
    assert exc.value.detail == (
        "'jira' conflicts with an existing connector name. Connector names are "
        "compared case-insensitively, treating spaces, hyphens and underscores "
        "as the same character, and must be unique across MCP servers and "
        "custom APIs."
    )


@pytest.mark.parametrize("stored,requested", [("Foo-Bar", "FOO_bar")])
@pytest.mark.parametrize("stored_in", [_MCP, _API])
@pytest.mark.parametrize("entry", [_MCP, _API])
def test_rename_rejects_folded_collision(
    db: Session, user: User, entry: str, stored_in: str, stored: str, requested: str
) -> None:
    _skip_if_mcp_cannot_take(entry, requested)
    target = _insert(db, entry, "renamed_row", owner=user)
    target_id = target.id
    _insert(db, stored_in, stored)
    before = _counts(db)

    with patch.object(connector_team_scope, "rename_team_connector") as hook:
        with pytest.raises(HTTPException) as exc:
            _rename(db, user, entry, target_id, requested)

    assert exc.value.status_code == 400
    assert exc.value.detail == folded_name_conflict_detail(requested)
    assert not db.new and not db.dirty
    assert _counts(db) == before
    hook.assert_not_called()


@pytest.mark.parametrize(
    "entry,stored,requested",
    [(_MCP, "foo_bar", "Foo-Bar"), (_API, "foo bar", "Foo_Bar")],
)
def test_rename_to_own_folded_variant_is_allowed(
    db: Session, user: User, entry: str, stored: str, requested: str
) -> None:
    row = _insert(db, entry, stored, owner=user)

    result = _rename(db, user, entry, row.id, requested)

    assert result.name == requested


@pytest.mark.parametrize("entry", [_MCP, _API])
def test_rename_excludes_only_same_type_same_id(
    db: Session, user: User, entry: str
) -> None:
    # An MCP server and a Custom API may share a numeric id. Only the row
    # being renamed is exempt, not every row in the other table with that id.
    if entry == _MCP:
        target = _insert_mcp(db, "plain_mcp", owner=user, row_id=5)
        _insert_api(db, "foo-bar", row_id=5)
    else:
        target = _insert_api(db, "plain_api", owner=user, row_id=5)
        _insert_mcp(db, "foo-bar", row_id=5)
    assert target.id == 5

    with pytest.raises(HTTPException) as exc:
        _rename(db, user, entry, 5, "foo_bar")

    assert exc.value.status_code == 400


@pytest.mark.parametrize(
    "name", ["\tGmail", "Gmail\n", "\u00a0Gmail", "Gmail\r\n", " \tGmail", "Gmail\n "]
)
@pytest.mark.parametrize("action", ["create", "rename"])
def test_custom_api_rejects_edge_whitespace_other_than_spaces(
    db: Session, user: User, action: str, name: str
) -> None:
    target = _insert(db, _API, "renamed_row", owner=user)
    target_id = target.id
    before = _counts(db)

    with pytest.raises(HTTPException) as exc:
        if action == "create":
            _create(db, user, _API, name)
        else:
            _rename(db, user, _API, target_id, name)

    assert exc.value.status_code == 400
    assert exc.value.detail == (
        "Connector names cannot start or end with tabs, line breaks or other "
        "whitespace besides spaces."
    )
    assert not db.new and not db.dirty
    assert _counts(db) == before


@pytest.mark.parametrize("name", [" Gmail ", "Google Maps", "a\tb"])
@pytest.mark.parametrize("action", ["create", "rename"])
def test_custom_api_allows_space_padding_and_inner_whitespace(
    db: Session, user: User, action: str, name: str
) -> None:
    target = _insert(db, _API, "renamed_row", owner=user)

    if action == "create":
        result = _create(db, user, _API, name)
    else:
        result = _rename(db, user, _API, target.id, name)

    assert result.name == name


@pytest.mark.parametrize("sent_name", [None, "\tGmail"])
def test_existing_edge_whitespace_name_stays_editable(
    db: Session, user: User, sent_name: str | None
) -> None:
    row = _insert(db, _API, "\tGmail", owner=user)

    result = update_custom_api(
        row.id,
        CustomApiUpdate(name=sent_name, description="edited"),
        current_user=user,
        db=db,
    )

    assert result.name == "\tGmail"
    assert result.description == "edited"


@pytest.mark.parametrize("entry", [_MCP, _API])
def test_existing_twins_stay_editable(db: Session, user: User, entry: str) -> None:
    _insert(db, entry, "records-mcp", owner=user)
    twin = _insert(db, entry, "records_mcp", owner=user)

    if entry == _MCP:
        result = update_mcp_server(
            twin.id,
            MCPServerUpdate(name="records_mcp", description="edited"),
            current_user=user,
            db=db,
        )
    else:
        result = update_custom_api(
            twin.id,
            CustomApiUpdate(name="records_mcp", description="edited"),
            current_user=user,
            db=db,
        )

    assert result.description == "edited"
    assert result.name == "records_mcp"


@pytest.mark.parametrize("entry", [_MCP, _API])
@pytest.mark.parametrize("action", ["create", "rename"])
def test_name_scan_failure_propagates_and_writes_nothing(
    db: Session, user: User, entry: str, action: str
) -> None:
    target = _insert(db, entry, "renamed_row", owner=user)
    target_id = target.id
    before = _counts(db)
    engine = db.get_bind()

    def fail_on_fold(conn, cursor, statement, parameters, context, executemany):
        if "replace(" in statement.lower():
            raise RuntimeError("fold scan failed")

    event.listen(engine, "before_cursor_execute", fail_on_fold)
    try:
        if entry == _MCP:
            # The MCP routes wrap everything in a generic handler.
            with pytest.raises(HTTPException) as exc:
                if action == "create":
                    _create(db, user, entry, "brand_new")
                else:
                    _rename(db, user, entry, target_id, "brand_new")
            assert exc.value.status_code == 500
            assert "fold scan failed" in exc.value.detail
        else:
            # The Custom API routes have no handler: the original error
            # reaches the caller unchanged.
            with pytest.raises(RuntimeError, match="fold scan failed"):
                if action == "create":
                    _create(db, user, entry, "brand_new")
                else:
                    _rename(db, user, entry, target_id, "brand_new")
    finally:
        event.remove(engine, "before_cursor_execute", fail_on_fold)

    assert _counts(db) == before


def _insert_catalog_app(
    db: Session, app_id: str, name: str, *, visible: bool = True
) -> None:
    db.add(
        PublicMCPApp(
            app_id=app_id,
            name=name,
            description="A catalog app",
            icon="",
            category="Productivity",
            transport="stdio",
            launch_config={
                "command": "npx",
                "args": ["-y", f"{app_id}-mcp"],
                "required_env": ["API_KEY"],
            },
            is_visible_in_connector=visible,
        )
    )
    db.commit()


def test_catalog_provisioning_is_not_checked(db: Session) -> None:
    _insert_catalog_app(db, "google-maps", "Google Maps")
    _insert_api(db, "google_maps")

    server, _ = _ensure_catalog_app_server(db, "google-maps")

    assert server.name == "google-maps"


@pytest.mark.parametrize(
    "name",
    [
        "Foo-Bar",
        "foo bar",
        "FOO_BAR",
        " Acme ",
        "Google Drive",
        "hub-spot",
        "a - b",
        "X",
    ],
)
def test_sql_fold_matches_python_normalizer(db: Session, name: str) -> None:
    folded = db.execute(select(_sql_fold(literal(name)))).scalar_one()

    assert folded == normalize_mcp_server_name(name)


def _web_functions() -> Iterator[tuple[Path, ast.AST, str]]:
    root = Path(xagent.web.__file__).parent
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield path, node, node.name


def _called_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def test_name_check_call_sites() -> None:
    root = Path(xagent.web.__file__).parent
    checked = {
        (root / "api" / "mcp.py", "create_mcp_server"),
        (root / "api" / "mcp.py", "update_mcp_server"),
        (root / "api" / "custom_api.py", "create_custom_api"),
        (root / "api" / "custom_api.py", "update_custom_api"),
    }
    not_checked = {
        (root / "api" / "mcp.py", "_ensure_catalog_app_server"),
        (root / "api" / "mcp.py", "_ensure_catalog_mcp_oauth_server"),
        (root / "api" / "auth.py", "_ensure_user_mcp_server"),
        (root / "mcp_apps.py", "ensure_builtin_oauth_server_definition"),
    }
    defined = {(path, name) for path, _, name in _web_functions()}
    assert checked <= defined
    assert not_checked <= defined

    for check in ("has_folded_connector_name_conflict", "folds_to_catalog_app_name"):
        callers: set[tuple[Path, str]] = set()
        for path, func, name in _web_functions():
            for node in ast.walk(func):
                if isinstance(node, ast.Call) and _called_name(node) == check:
                    callers.add((path, name))

        assert callers == checked, check


def test_conflict_is_checked_in_sql(db: Session) -> None:
    _insert_mcp(db, "Foo-Bar")
    _insert_api(db, "other name")
    engine = db.get_bind()
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        has_folded_connector_name_conflict(db, "nothing_like_it")
    finally:
        event.remove(engine, "before_cursor_execute", record)

    scans = [s for s in statements if "mcp_servers" in s or "custom_apis" in s]
    assert len(scans) == 2
    for statement in scans:
        assert re.search(r"\bWHERE\b", statement, re.IGNORECASE)


# (requested name, catalog app id, catalog display name). The MCP routes also
# run an older catalog check under a different fold, which already refuses
# "GitHub" and "Google Maps"; "google_maps" is the variant only this check
# catches there.
_CATALOG_NAMES = [
    pytest.param("GitHub", "github", "GitHub", id="display-name"),
    pytest.param("github", "github", "GitHub", id="app-id"),
    pytest.param("google_maps", "google-maps", "Google Maps", id="underscore"),
    pytest.param("Google Maps", "google-maps", "Google Maps", id="space"),
]


@pytest.mark.parametrize("provisioned", [False, True])
@pytest.mark.parametrize("requested,app_id,app_name", _CATALOG_NAMES)
@pytest.mark.parametrize("action", ["create", "rename"])
def test_custom_api_rejects_catalog_app_name(
    db: Session,
    user: User,
    action: str,
    requested: str,
    app_id: str,
    app_name: str,
    provisioned: bool,
) -> None:
    _insert_catalog_app(db, app_id, app_name)
    if provisioned:
        # The row provisioning names after the app id. Inserted directly:
        # some ids here belong to built-in OAuth apps, which connect through
        # a different flow.
        _insert_mcp(db, app_id)
    target = _insert_api(db, "renamed_row", owner=user)
    target_id = target.id
    before = _counts(db)

    with patch.object(connector_team_scope, "rename_team_connector") as hook:
        with pytest.raises(HTTPException) as exc:
            if action == "create":
                _create(db, user, _API, requested)
            else:
                _rename(db, user, _API, target_id, requested)

    assert exc.value.status_code == 400
    assert exc.value.detail == catalog_app_name_detail(requested)
    assert not db.new and not db.dirty
    assert _counts(db) == before
    hook.assert_not_called()


@pytest.mark.parametrize("action", ["create", "rename"])
def test_mcp_rejects_catalog_app_name_under_the_selection_fold(
    db: Session, user: User, action: str
) -> None:
    _insert_catalog_app(db, "google-maps", "Google Maps")
    target = _insert_mcp(db, "renamed_row", owner=user)
    target_id = target.id
    before = _counts(db)

    with pytest.raises(HTTPException) as exc:
        if action == "create":
            _create(db, user, _MCP, "google_maps")
        else:
            _rename(db, user, _MCP, target_id, "google_maps")

    assert exc.value.status_code == 400
    assert exc.value.detail == catalog_app_name_detail("google_maps")
    assert _counts(db) == before


@pytest.mark.parametrize("entry", [_MCP, _API])
def test_hidden_catalog_app_still_reserves_its_names(
    db: Session, user: User, entry: str
) -> None:
    _insert_catalog_app(db, "google-maps", "Google Maps", visible=False)

    with pytest.raises(HTTPException) as exc:
        _create(db, user, entry, "google_maps")

    assert exc.value.status_code == 400
    assert exc.value.detail == catalog_app_name_detail("google_maps")


@pytest.mark.parametrize("entry", [_MCP, _API])
@pytest.mark.parametrize("action", ["create", "rename"])
def test_name_that_only_resembles_a_catalog_app_is_accepted(
    db: Session, user: User, entry: str, action: str
) -> None:
    _insert_catalog_app(db, "hubspot", "HubSpot")
    target = _insert(db, entry, "renamed_row", owner=user)
    target_id = target.id

    if action == "create":
        result = _create(db, user, entry, "hub-spot")
    else:
        result = _rename(db, user, entry, target_id, "hub-spot")

    assert result.name == "hub-spot"


def test_catalog_check_ignores_unrelated_names(db: Session) -> None:
    _insert_catalog_app(db, "google-maps", "Google Maps")

    assert folds_to_catalog_app_name(db, "Google-Maps")
    assert not folds_to_catalog_app_name(db, "google_maps_extra")
    assert not folds_to_catalog_app_name(db, "")


@pytest.mark.parametrize("name_in_update", [True, False])
@pytest.mark.parametrize(
    "entry,stored",
    [
        pytest.param(_MCP, "google-maps", id="provisioned-mcp-row"),
        pytest.param(_API, "google_maps", id="custom-api-named-before-the-check"),
    ],
)
def test_row_named_after_catalog_app_stays_editable(
    db: Session, user: User, entry: str, stored: str, name_in_update: bool
) -> None:
    _insert_catalog_app(db, "google-maps", "Google Maps")
    row = _insert(db, entry, stored, owner=user)
    row_id = row.id
    name = stored if name_in_update else None

    if entry == _MCP:
        result = update_mcp_server(
            row_id,
            MCPServerUpdate(name=name, description="edited"),
            current_user=user,
            db=db,
        )
    else:
        result = update_custom_api(
            row_id,
            CustomApiUpdate(name=name, description="edited"),
            current_user=user,
            db=db,
        )

    assert result.name == stored
    assert result.description == "edited"


@pytest.mark.parametrize("action", ["create", "rename"])
def test_mcp_catalog_rejections_share_one_message(
    db: Session, user: User, action: str
) -> None:
    # "Google-Maps" is caught by the older catalog-key check and "google_maps"
    # only by the selection-fold check; both answer with the same text.
    _insert_catalog_app(db, "google-maps", "Google Maps")
    target = _insert_mcp(db, "renamed_row", owner=user)
    target_id = target.id

    for requested in ("Google-Maps", "google_maps"):
        with pytest.raises(HTTPException) as exc:
            if action == "create":
                _create(db, user, _MCP, requested)
            else:
                _rename(db, user, _MCP, target_id, requested)
        assert exc.value.status_code == 400
        assert exc.value.detail == catalog_app_name_detail(requested)
