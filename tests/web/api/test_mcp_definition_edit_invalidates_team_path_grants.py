"""Editing a connector's definition deletes the OAuth state of team members.

A team member who authorized a shared connector holds a grant whose access
token is sent to the connector's URL and whose refresh token is sent to the
token endpoint on the OAuth client row the grant was issued through. The
owner's next connect rewrites that row from the authorization server that the
auth keys ``issuer`` and ``resource_metadata_url`` select. All three are
controlled by the connector's owner. When the owner (or a platform admin)
changes the URL text or either key, ``update_mcp_server`` deletes, in the same
transaction, every OAuth flow and grant on that connector that belongs to a
user without an association row and that was stored under the user's own
default owner key. Other auth edits, such as a client secret rotation, delete
nothing. The owner's own grants are never touched.

Two scans keep the rule complete: every auth key the OAuth code reads must be
classified as bound or not bound, with a reason, and every place under
``src/xagent`` that can write a connector's URL or auth must be listed, with a
reason.
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import xagent.web.api.mcp as mcp_module
from xagent.core.tools.core.mcp.model import (
    MASKED_SECRET_VALUE,
    SENSITIVE_AUTH_FIELDS,
)
from xagent.core.utils.encryption import decrypt_value, encrypt_value
from xagent.web.api.mcp import MCPServerUpdate, update_mcp_server
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import (
    MCPOAuthClient,
    MCPOAuthFlowState,
    MCPOAuthGrant,
)
from xagent.web.models.user import User
from xagent.web.services.mcp_oauth import _validate_issuer, select_mcp_oauth_grants

URL = "https://mcp.edit.test/mcp"
ISSUER = "https://auth.edit.test"
SECRET = "client-secret-1"


def _stored_auth(**overrides: Any) -> dict[str, Any]:
    auth: dict[str, Any] = {
        "type": "mcp_oauth",
        "client_id": "client-1",
        "client_secret": encrypt_value(SECRET),
        "issuer": ISSUER,
        "resource": URL,
        "resource_metadata_url": f"{URL}/.well-known/oauth-protected-resource",
        "token_endpoint_auth_method": "client_secret_post",
        "redirect_uri": "https://app.edit.test/api/mcp/oauth/callback",
        "scope": "records.read",
    }
    auth.update(overrides)
    return auth


def _submitted_auth(**overrides: Any) -> dict[str, Any]:
    """What an edit form sends: the stored auth with the secret masked."""
    auth = _stored_auth(client_secret=MASKED_SECRET_VALUE)
    auth.update(overrides)
    return auth


@pytest.fixture()
def db(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'definition-edit.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _user(db, name: str, *, is_admin: bool = False) -> User:
    user = User(username=name, password_hash="x", is_admin=is_admin)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


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


def _grant(
    db,
    *,
    oauth_client: MCPOAuthClient,
    server: MCPServer,
    user: User,
    scope: str,
    status: str,
) -> MCPOAuthGrant:
    grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        mcp_oauth_client_id=oauth_client.id,
        resource_owner_key=f"xagent:user:{user.id}",
        issuer=ISSUER,
        resource=URL,
        scope=scope,
        access_token=encrypt_value(f"token-{user.id}-{scope}-{status}"),
        status=status,
    )
    db.add(grant)
    db.commit()
    db.refresh(grant)
    return grant


class Fixture:
    def __init__(self, db) -> None:
        self.db = db
        self.owner = _user(db, "owner")
        self.member = _user(db, "member")
        self.admin = _user(db, "platform-admin", is_admin=True)
        self.other_member = _user(db, "associated-member")
        server = MCPServer.from_config(
            {
                "name": "shared-records",
                "managed": "external",
                "transport": "streamable_http",
                "url": URL,
                "auth": {"type": "mcp_oauth"},
                "headers": {"X-Proxy-Key": "proxy-1"},
            }
        )
        server.auth = _stored_auth()
        server.description = "records"
        db.add(server)
        db.commit()
        db.refresh(server)
        self.server = server
        _link(db, self.owner, server, is_owner=True)
        _link(db, self.admin, server, is_owner=False)
        _link(db, self.other_member, server, is_owner=False)
        oauth_client = MCPOAuthClient(
            mcp_server_id=server.id,
            issuer=ISSUER,
            authorization_endpoint=f"{ISSUER}/authorize",
            token_endpoint=f"{ISSUER}/token",
            client_id="client-1",
            token_endpoint_auth_method="client_secret_post",
            redirect_uri="https://app.edit.test/api/mcp/oauth/callback",
        )
        db.add(oauth_client)
        db.commit()
        db.refresh(oauth_client)
        self.member_active = _grant(
            db,
            oauth_client=oauth_client,
            server=server,
            user=self.member,
            scope="records.read",
            status="active",
        )
        self.member_revoked = _grant(
            db,
            oauth_client=oauth_client,
            server=server,
            user=self.member,
            scope="records.write",
            status="revoked",
        )
        self.member_flow = MCPOAuthFlowState(
            state="member-flow-state",
            mcp_server_id=server.id,
            user_id=self.member.id,
            mcp_oauth_client_id=oauth_client.id,
            resource_owner_key=f"xagent:user:{self.member.id}",
            issuer=ISSUER,
            resource=URL,
            scope="records.read",
            code_verifier="verifier",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        )
        db.add(self.member_flow)
        db.commit()
        self.owner_grant = _grant(
            db,
            oauth_client=oauth_client,
            server=server,
            user=self.owner,
            scope="records.read",
            status="active",
        )
        # A grant stored under a non-default owner key is not team-path state
        # and survives every edit.
        non_default = _grant(
            db,
            oauth_client=oauth_client,
            server=server,
            user=self.member,
            scope="records.admin",
            status="active",
        )
        non_default.resource_owner_key = "external:account:1"
        db.commit()
        self.non_default_key_grant = non_default
        # A member who holds an association keeps the grant: the rule is about
        # users the connector's owner did not attach to the connector.
        self.associated_grant = _grant(
            db,
            oauth_client=oauth_client,
            server=server,
            user=self.other_member,
            scope="records.read",
            status="active",
        )
        # A second connector on which the same member holds a grant and an
        # association row. An edit of the first connector never touches its
        # grant, and the association on it does not shelter the member's rows
        # on the first connector.
        other_server = MCPServer.from_config(
            {
                "name": "other-records",
                "managed": "external",
                "transport": "streamable_http",
                "url": "https://mcp.edit.test/second",
                "auth": {"type": "mcp_oauth"},
            }
        )
        db.add(other_server)
        db.commit()
        db.refresh(other_server)
        self.other_server = other_server
        _link(db, self.owner, other_server, is_owner=True)
        _link(db, self.member, other_server, is_owner=False)
        other_oauth_client = MCPOAuthClient(
            mcp_server_id=other_server.id,
            issuer=ISSUER,
            authorization_endpoint=f"{ISSUER}/authorize",
            token_endpoint=f"{ISSUER}/token",
            client_id="client-other",
            token_endpoint_auth_method="client_secret_post",
            redirect_uri="https://app.edit.test/api/mcp/oauth/callback",
        )
        db.add(other_oauth_client)
        db.commit()
        db.refresh(other_oauth_client)
        self.other_server_member_grant = _grant(
            db,
            oauth_client=other_oauth_client,
            server=other_server,
            user=self.member,
            scope="records.read",
            status="active",
        )
        self.member_row_ids = (
            self.member_active.id,
            self.member_revoked.id,
            self.member_flow.id,
        )

    def put(self, actor: User, **payload: Any) -> None:
        update_mcp_server(
            self.server.id, MCPServerUpdate(**payload), current_user=actor, db=self.db
        )
        self.db.expire_all()

    def member_rows_present(self) -> tuple[bool, bool, bool]:
        grant_ids = {g.id for g in self.db.query(MCPOAuthGrant.id).all()}
        flow_ids = {f.id for f in self.db.query(MCPOAuthFlowState.id).all()}
        active_id, revoked_id, flow_id = self.member_row_ids
        return (
            active_id in grant_ids,
            revoked_id in grant_ids,
            flow_id in flow_ids,
        )

    def break_stored_secret(self) -> None:
        wrong_key_token = Fernet(Fernet.generate_key()).encrypt(b"x").decode()
        assert decrypt_value(wrong_key_token) == wrong_key_token
        self.server.auth = _stored_auth(client_secret=wrong_key_token)
        self.db.commit()


# (actor, payload kwargs, deleted)
CASES: list[Any] = [
    pytest.param(
        "owner",
        {"config": {"url": "https://mcp.edit.test/other"}},
        True,
        id="url",
    ),
    pytest.param(
        "owner", {"config": {"url": f"{URL}?tenant=2"}}, True, id="url_query_only"
    ),
    pytest.param(
        "owner", {"config": {"url": f"{URL}/"}}, True, id="url_trailing_slash_only"
    ),
    pytest.param(
        "owner", {"config": {"url": f"  {URL}  "}}, False, id="url_whitespace_only"
    ),
    pytest.param(
        "owner",
        {"config": {"auth": _submitted_auth(client_id="client-2")}},
        False,
        id="auth_client_id",
    ),
    pytest.param(
        "owner",
        {"config": {"auth": _submitted_auth(issuer="https://auth2.edit.test")}},
        True,
        id="auth_issuer",
    ),
    pytest.param(
        "owner",
        {
            "config": {
                "auth": {k: v for k, v in _submitted_auth().items() if k != "issuer"}
            }
        },
        True,
        id="auth_issuer_removed",
    ),
    pytest.param(
        "owner",
        {"config": {"auth": _submitted_auth(resource="https://mcp.edit.test/v2")}},
        False,
        id="auth_resource",
    ),
    pytest.param(
        "owner",
        {
            "config": {
                "auth": _submitted_auth(
                    resource_metadata_url="https://meta.edit.test/.well-known/x"
                )
            }
        },
        True,
        id="auth_resource_metadata_url",
    ),
    pytest.param(
        "owner",
        {
            "config": {
                "auth": _submitted_auth(
                    token_endpoint_auth_method="client_secret_basic"
                )
            }
        },
        False,
        id="auth_token_endpoint_auth_method",
    ),
    pytest.param(
        "owner",
        {"config": {"auth": _submitted_auth(redirect_uri="https://app2.edit.test/cb")}},
        False,
        id="auth_redirect_uri",
    ),
    pytest.param(
        "owner",
        {"config": {"auth": _submitted_auth(client_secret="client-secret-2")}},
        False,
        id="auth_new_client_secret",
    ),
    pytest.param(
        "owner",
        {"config": {"auth": _submitted_auth(client_secret=SECRET)}},
        False,
        id="auth_same_client_secret_resubmitted",
    ),
    pytest.param(
        "owner",
        {"config": {"auth": _submitted_auth()}},
        False,
        id="auth_masked_secret",
    ),
    pytest.param(
        "owner",
        {"config": {"auth": _submitted_auth(scope="records.read records.write")}},
        False,
        id="auth_scope_only",
    ),
    pytest.param(
        "owner",
        {"config": {"headers": {"X-Proxy-Key": "proxy-2"}}},
        False,
        id="headers_value",
    ),
    pytest.param(
        "owner", {"description": "renamed records"}, False, id="description_only"
    ),
    pytest.param("owner", {"is_active": False}, False, id="is_active_only"),
    pytest.param(
        "admin",
        {"config": {"url": "https://mcp.edit.test/admin-moved"}},
        True,
        id="admin_edits_url",
    ),
    pytest.param(
        "owner",
        {"transport": "sse"},
        False,
        id="transport_only",
    ),
]


def _assert_team_path_state(fx: Fixture, deleted: bool) -> None:
    assert fx.member_rows_present() == ((not deleted),) * 3
    remaining = {g.id for g in fx.db.query(MCPOAuthGrant.id).all()}
    assert fx.owner_grant.id in remaining
    assert fx.associated_grant.id in remaining
    assert fx.non_default_key_grant.id in remaining
    assert fx.other_server_member_grant.id in remaining


@pytest.mark.parametrize(("actor", "payload", "deleted"), CASES)
def test_definition_edit_deletes_team_path_state_only_when_binding_changes(
    db, actor: str, payload: dict[str, Any], deleted: bool
) -> None:
    fx = Fixture(db)

    fx.put(fx.admin if actor == "admin" else fx.owner, **payload)

    _assert_team_path_state(fx, deleted)


@pytest.mark.parametrize("stored_secret", ["readable", "undecryptable"])
def test_non_owner_update_with_unchanged_definition_deletes_nothing(
    db, stored_secret: str
) -> None:
    fx = Fixture(db)
    if stored_secret == "undecryptable":
        fx.break_stored_secret()

    fx.put(fx.other_member, config={"url": URL})

    assert fx.member_rows_present() == (True, True, True)


def test_resource_metadata_url_edit_leaves_no_member_grant_on_the_rewritten_client(
    db,
) -> None:
    """The edit deletes the grants an owner's connect could redirect.

    The owner points ``resource_metadata_url`` at metadata naming another
    authorization server, which claims the configured issuer. Issuer validation
    accepts it, and the owner's connect then finds the client row the members'
    grants were issued through (same issuer and client id) and rewrites its
    token endpoint. A member grant left on that row would send its refresh
    token there.
    """
    fx = Fixture(db)
    redirected_token_endpoint = "https://other-auth.example/token"

    fx.put(
        fx.owner,
        config={
            "auth": _submitted_auth(
                resource_metadata_url=(
                    "https://other-auth.example/.well-known/oauth-protected-resource"
                )
            )
        },
    )
    _validate_issuer(
        selected_authorization_server="https://other-auth.example/as",
        metadata=SimpleNamespace(issuer=ISSUER),
        configured_issuer=ISSUER,
    )
    mcp_module._upsert_mcp_oauth_client(
        db,
        server_id=fx.server.id,
        discovery=SimpleNamespace(
            authorization_server=SimpleNamespace(
                issuer=ISSUER,
                authorization_endpoint="https://other-auth.example/authorize",
                token_endpoint=redirected_token_endpoint,
                raw={},
            )
        ),
        client_id="client-1",
        client_secret=SECRET,
        token_endpoint_auth_method="client_secret_post",
        redirect_uri="https://app.edit.test/api/mcp/oauth/callback",
    )
    db.commit()
    db.expire_all()

    rewritten = (
        db.query(MCPOAuthClient)
        .filter(
            MCPOAuthClient.mcp_server_id == fx.server.id,
            MCPOAuthClient.client_id == "client-1",
        )
        .one()
    )
    assert rewritten.token_endpoint == redirected_token_endpoint
    assert fx.member_rows_present() == (False, False, False)
    assert (
        select_mcp_oauth_grants(
            db,
            server_id=fx.server.id,
            user_id=fx.member.id,
            auth_config={"type": "mcp_oauth"},
            resource_owner_key=f"xagent:user:{fx.member.id}",
            allow_team_member_grants=True,
        )
        == []
    )


def test_deletion_failure_rolls_back_the_definition_write(db, monkeypatch) -> None:
    fx = Fixture(db)
    real_delete = mcp_module._delete_team_path_mcp_oauth_state

    def delete_then_fail(session: Any, *, server_id: int) -> None:
        # The URL update and the deletions reach the database before the
        # failure, so only the route's rollback can undo them.
        real_delete(session, server_id=server_id)
        session.flush()
        raise RuntimeError("delete failed")

    monkeypatch.setattr(
        mcp_module, "_delete_team_path_mcp_oauth_state", delete_then_fail
    )

    with pytest.raises(Exception) as exc:
        fx.put(fx.owner, config={"url": "https://mcp.edit.test/other"})

    assert getattr(exc.value, "status_code", None) == 500
    db.expire_all()
    assert db.get(MCPServer, fx.server.id).url == URL
    assert fx.member_rows_present() == (True, True, True)


# --- Scan of every place that can write a connector's URL or auth -----------

SRC_ROOT = Path(mcp_module.__file__).resolve().parents[2]

# (path under src/xagent, enclosing function, written target) -> why the write
# cannot change where a team member's tokens go without the binding comparison
WRITER_ALLOWLIST: dict[tuple[str, str, str], str] = {
    (
        "web/api/mcp.py",
        "_update_server_from_config",
        "setattr(server, db_field, value)",
    ): (
        "The generic definition writer; its only caller, update_mcp_server, "
        "compares the binding before and after it."
    ),
    (
        "web/api/mcp.py",
        "_ensure_catalog_mcp_oauth_server",
        "cast(Any, server).auth",
    ): (
        "Writes a catalog connector row from the deployment's MCP app catalog, "
        "not from a user's edit."
    ),
    (
        "web/api/auth.py",
        "_ensure_user_mcp_server._ensure_server_matches_oauth_app",
        "cast(Any, server).auth",
    ): (
        "Writes built-in OAuth app metadata on a row whose transport is "
        "'oauth', which never uses MCP OAuth grants."
    ),
    ("web/mcp_apps.py", "_adopt_builtin_auth", "server.auth"): (
        "Repairs a built-in catalog row's auth from the catalog definition, not "
        "from a user's edit."
    ),
    (
        "web/services/google_oauth_grants.py",
        "_copy_shared_google_credentials",
        "setattr(target, field, value)",
    ): "Writes Google OAuth account rows, not MCPServer rows.",
    (
        "web/tools/config.py",
        "WebToolConfig._resolve_factory_model_field",
        "setattr(self, cache_name, cached)",
    ): "Writes a cache attribute on the tool configuration object.",
    (
        "web/tools/config.py",
        "WebToolConfig._finish_legacy_oauth_access_token_resolution",
        "setattr(oauth_account, field, value)",
    ): "Writes UserOAuth rows, not MCPServer rows.",
    ("web/api/custom_api.py", "update_custom_api", "api.url"): (
        "Writes a Custom API row, not an MCP connector."
    ),
    ("core/model/rerank/dashscope.py", "DashscopeRerank.__init__", "self.url"): (
        "Sets the endpoint of a rerank client object."
    ),
    (
        "core/tools/adapters/vibe/a2a_agent_tool.py",
        "_PinnedA2ATransport.handle_async_request",
        "request.url",
    ): "Rewrites an outgoing HTTP request object.",
    (
        "web/services/mcp_oauth.py",
        "SafeOAuthAsyncHTTPTransport.handle_async_request",
        "request.url",
    ): "Rewrites an outgoing HTTP request object.",
}


def _call_chain(node: ast.AST) -> list[ast.Call]:
    """Every call on the receiver chain of ``node``, outermost first."""
    calls: list[ast.Call] = []
    while True:
        if isinstance(node, ast.Call):
            calls.append(node)
            node = node.func
        elif isinstance(node, ast.Attribute):
            node = node.value
        else:
            return calls


def _callee_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _is_model_arg_call(call: ast.Call, callee: str) -> bool:
    return (
        _callee_name(call) == callee
        and bool(call.args)
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "MCPServer"
    )


def _imports_mcp_server(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and any(alias.name == "MCPServer" for alias in node.names)
        for node in ast.walk(tree)
    )


class _WriterScanner(ast.NodeVisitor):
    def __init__(self, tree: ast.Module) -> None:
        self.imports_mcp_server = _imports_mcp_server(tree)
        self.scope: list[str] = []
        self.hits: set[tuple[str, str]] = set()
        self.consumed: set[int] = set()

    def _where(self) -> str:
        return ".".join(self.scope) or "<module>"

    def _scoped(self, node: Any) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = _scoped
    visit_AsyncFunctionDef = _scoped
    visit_ClassDef = _scoped

    def _targets(self, targets: list[ast.expr]) -> None:
        for target in targets:
            if isinstance(target, (ast.Tuple, ast.List)):
                self._targets(list(target.elts))
            elif isinstance(target, ast.Starred):
                self._targets([target.value])
            elif isinstance(target, ast.Attribute) and target.attr in {"url", "auth"}:
                self.hits.add((self._where(), ast.unparse(target)))

    def visit_Assign(self, node: ast.Assign) -> None:
        self._targets(node.targets)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._targets([node.target])
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._targets([node.target])
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _callee_name(node)
        if (
            name == "setattr"
            and isinstance(node.func, ast.Name)
            and len(node.args) >= 2
        ):
            field = node.args[1]
            literal_field = isinstance(field, ast.Constant) and field.value in {
                "url",
                "auth",
            }
            non_literal = not isinstance(field, ast.Constant)
            if literal_field or (non_literal and self.imports_mcp_server):
                self.hits.add((self._where(), ast.unparse(node)))
        if isinstance(node.func, ast.Attribute):
            receiver_chain = _call_chain(node.func.value)
            if node.func.attr == "update" and any(
                _is_model_arg_call(call, "query") for call in receiver_chain
            ):
                self.hits.add((self._where(), ast.unparse(node)))
            if node.func.attr == "values":
                inner = [c for c in receiver_chain if _is_model_arg_call(c, "update")]
                if inner:
                    self.consumed.add(id(inner[0]))
                    self.hits.add((self._where(), ast.unparse(node)))
        if _is_model_arg_call(node, "update") and id(node) not in self.consumed:
            self.hits.add((self._where(), ast.unparse(node)))
        self.generic_visit(node)


def _scan_writers(path: Path) -> set[tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scanner = _WriterScanner(tree)
    scanner.visit(tree)
    return scanner.hits


def _scan_tree(root: Path) -> set[tuple[str, str, str]]:
    found: set[tuple[str, str, str]] = set()
    for path in sorted(root.rglob("*.py")):
        for where, target in _scan_writers(path):
            found.add((path.relative_to(root).as_posix(), where, target))
    return found


def test_mcp_server_url_and_auth_writers_are_enumerated() -> None:
    found = _scan_tree(SRC_ROOT)
    listed = set(WRITER_ALLOWLIST)

    assert found - listed == set(), (
        "A new write to a connector URL or auth is not accounted for. Route it "
        "through update_mcp_server's binding comparison, or add it to the "
        "allowlist with the reason it cannot affect team-member grants."
    )
    assert listed - found == set(), "Allowlist entry no longer matches code"
    assert all(reason.strip() for reason in WRITER_ALLOWLIST.values())


_SYNTHETIC_WRITERS = {
    "query_update": (
        "from m import MCPServer\n"
        "def f(db, x):\n"
        '    db.query(MCPServer).filter(MCPServer.id == 1).update({"url": x})\n'
    ),
    "core_update_values": (
        "from m import MCPServer\n"
        "from sqlalchemy import update\n"
        "def f(db, x):\n"
        "    db.execute(update(MCPServer).values(auth=x))\n"
    ),
    "dynamic_setattr": (
        "from m import MCPServer\n"
        "def f(definition, field, v):\n"
        "    setattr(definition, field, v)\n"
    ),
    "literal_setattr": ('def f(row, v):\n    setattr(row, "url", v)\n'),
    "attribute_assign": "def f(x, v):\n    x.url = v\n",
}


@pytest.mark.parametrize("name", sorted(_SYNTHETIC_WRITERS))
def test_writer_scanner_catches_each_node_class(tmp_path, name: str) -> None:
    sample = tmp_path / f"{name}.py"
    sample.write_text(_SYNTHETIC_WRITERS[name], encoding="utf-8")

    hits = _scan_writers(sample)

    assert len(hits) == 1, hits


# --- Every auth key the OAuth code reads is classified ------------------------

_OAUTH_SOURCES = (
    SRC_ROOT / "web/api/mcp.py",
    SRC_ROOT / "web/services/mcp_oauth.py",
    SRC_ROOT / "web/services/mcp_runtime.py",
)
# auth key -> (bound, reason). Bound keys are _MEMBER_GRANT_BINDING_AUTH_KEYS:
# changing one deletes team-path state.
AUTH_KEY_CLASSIFICATION: dict[str, tuple[bool, str]] = {
    "issuer": (
        True,
        "Decides which authorization server metadata the owner's connect "
        "accepts and writes into the client row members' grants refresh "
        "through, including its token endpoint.",
    ),
    "resource_metadata_url": (
        True,
        "Decides where discovery reads the protected resource metadata that "
        "names the authorization server, whose token endpoint the owner's "
        "connect writes into the client row members' grants refresh through.",
    ),
    "type": (
        False,
        "Only says whether the connector uses MCP OAuth; tokens still go to "
        "the URL and refreshes to the client row's token endpoint.",
    ),
    "resource": (
        False,
        "Only checked against the discovered protected resource and used to "
        "filter grants; it does not choose where any token is sent.",
    ),
    "client_id": (
        False,
        "Names the OAuth client; the token endpoint a connect writes into any "
        "client row comes from the discovery the URL, issuer and "
        "resource_metadata_url select.",
    ),
    "client_secret": (
        False,
        "The owner's client credential; it changes neither where access tokens "
        "nor where refresh tokens are sent.",
    ),
    "token_endpoint_auth_method": (
        False,
        "Only says how the client authenticates within the same token request.",
    ),
    "redirect_uri": (
        False,
        "Only says where a new authorization delivers its code; refresh "
        "requests carry no redirect URI.",
    ),
    "scope": (
        False,
        "Grants missing a required scope are already excluded by grant selection.",
    ),
    "required_scope": (
        False,
        "Grants missing a required scope are already excluded by grant selection.",
    ),
}


def _literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _auth_keys_read(path: Path) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "get"
                and isinstance(func.value, ast.Name)
                and func.value.id == "auth_config"
                and node.args
            ):
                key = _literal(node.args[0])
                if key:
                    keys.add(key)
            if (
                isinstance(func, ast.Name)
                and func.id in {"_configured_mcp_oauth_value", "_runtime_config_value"}
                and len(node.args) >= 3
                and isinstance(node.args[1], ast.Name)
                and node.args[1].id == "auth_config"
            ):
                key = _literal(node.args[2])
                if key:
                    keys.add(key)
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "auth_config"
        ):
            key = _literal(node.slice)
            if key:
                keys.add(key)
    return keys


def test_every_oauth_auth_key_read_is_classified() -> None:
    read: set[str] = set()
    for path in _OAUTH_SOURCES:
        read |= _auth_keys_read(path)
    classified = set(AUTH_KEY_CLASSIFICATION)
    bound = {key for key, (is_bound, _) in AUTH_KEY_CLASSIFICATION.items() if is_bound}

    assert read - classified == set(), (
        "The OAuth code reads an auth key that is not classified. Add it to "
        "AUTH_KEY_CLASSIFICATION as bound or not bound, with the reason."
    )
    assert classified - read == set(), "Classified key is no longer read"
    assert all(reason.strip() for _, reason in AUTH_KEY_CLASSIFICATION.values())
    assert bound == set(mcp_module._MEMBER_GRANT_BINDING_AUTH_KEYS)
    # The binding compares stored values, which needs them unencrypted.
    assert bound.isdisjoint(SENSITIVE_AUTH_FIELDS)
