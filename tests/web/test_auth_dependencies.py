"""Authentication dependency contract tests."""

import ast
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from tests.web.auth_token_cases import (
    REJECTED_ACCESS_TOKEN_CASES,
    RejectedAccessTokenCase,
)
from tests.web.auth_token_cases import build_access_token as _access_token
from xagent.web import auth_dependencies
from xagent.web.app import global_exception_handler
from xagent.web.auth_dependencies import (
    get_current_user,
    get_current_user_optional,
    get_user_from_token,
    get_user_from_websocket_token,
)
from xagent.web.models.database import Base
from xagent.web.models.user import User


@pytest.fixture
def db_session() -> Session:
    """Provide a SQLite session with one matching user."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(
        User(
            id=1,
            username="existing-user",
            email="existing@example.com",
            password_hash="not-used-by-auth-dependencies",
        )
    )
    session.commit()
    session.info["user_query_count"] = 0

    def count_user_queries(*_args: object) -> None:
        session.info["user_query_count"] += 1

    event.listen(engine, "before_cursor_execute", count_user_queries)
    try:
        yield session
    finally:
        event.remove(engine, "before_cursor_execute", count_user_queries)
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.mark.parametrize("case", REJECTED_ACCESS_TOKEN_CASES, ids=lambda case: case.id)
def test_access_token_rejection_matrix_preserves_required_http_contract(
    db_session: Session, case: RejectedAccessTokenCase
) -> None:
    """Required HTTP auth exposes its established reason-specific rejection."""
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=case.build_token()
    )

    with pytest.raises(HTTPException) as raised:
        get_current_user(credentials, db_session)

    assert raised.value.status_code == 401
    assert raised.value.detail == case.expected_detail
    assert raised.value.headers == case.expected_headers


@pytest.mark.parametrize("case", REJECTED_ACCESS_TOKEN_CASES, ids=lambda case: case.id)
def test_optional_auth_rejects_every_credential_reason(
    db_session: Session, case: RejectedAccessTokenCase
) -> None:
    """Optional authentication suppresses every typed credential rejection."""
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=case.build_token()
    )

    assert get_current_user_optional(credentials, db_session) is None


@pytest.mark.parametrize(
    ("claims", "remove_claims"),
    (
        ({}, ("sub",)),
        ({}, ("user_id",)),
        ({"user_id": None}, ()),
        ({"user_id": "1"}, ()),
        ({"user_id": True}, ()),
    ),
    ids=(
        "absent-sub",
        "absent-user-id",
        "null-user-id",
        "wrong-type-user-id",
        "bool-user-id",
    ),
)
def test_malformed_identity_claim_shapes_are_rejected_before_any_user_query(
    db_session: Session,
    claims: dict[str, object],
    remove_claims: tuple[str, ...],
) -> None:
    """Identity claims with non-bindable types stay credential rejections."""
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials=_access_token(remove_claims=remove_claims, **claims),
    )

    with pytest.raises(HTTPException) as raised:
        get_current_user(credentials, db_session)

    assert raised.value.status_code == 401
    assert raised.value.detail == "Invalid token payload"
    assert raised.value.headers == {"WWW-Authenticate": "Bearer"}
    assert db_session.info["user_query_count"] == 0


@pytest.mark.parametrize("sub", (None, 123, True), ids=("null", "integer", "boolean"))
def test_present_non_string_subject_preserves_invalid_token_http_contract(
    db_session: Session, sub: object
) -> None:
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=_access_token(sub=sub)
    )

    with pytest.raises(HTTPException) as raised:
        get_current_user(credentials, db_session)

    assert raised.value.status_code == 401
    assert raised.value.detail == "Invalid token"
    assert raised.value.headers == {
        "WWW-Authenticate": "Bearer",
        "Error-Type": "InvalidToken",
    }
    assert db_session.info["user_query_count"] == 0


@pytest.mark.parametrize("claim", ("exp", "nbf", "iat"))
@pytest.mark.parametrize(
    "value",
    ([], {}, None, float("inf"), float("-inf")),
    ids=("list", "dict", "null", "positive-infinity", "negative-infinity"),
)
def test_malformed_temporal_claims_are_rejected_before_any_user_query(
    db_session: Session, claim: str, value: object
) -> None:
    """Dependency conversion failures in signed temporal claims are credentials."""
    token = _access_token(**{claim: value})
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with pytest.raises(HTTPException) as raised:
        get_current_user(credentials, db_session)

    assert raised.value.status_code == 401
    assert raised.value.detail == "Invalid token payload"
    assert raised.value.headers == {"WWW-Authenticate": "Bearer"}
    assert db_session.info["user_query_count"] == 0


@pytest.mark.parametrize(
    "claims",
    (
        {"iat": [], "exp": float("inf")},
        {"iat": float("inf"), "exp": []},
    ),
    ids=("type-error-before-overflow-error", "overflow-error-before-type-error"),
)
def test_mixed_malformed_temporal_claims_are_rejected_before_any_user_query(
    db_session: Session, claims: dict[str, object]
) -> None:
    """A later mixed temporal claim cannot defeat the original error proof."""
    token = _access_token(**claims)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with pytest.raises(HTTPException) as raised:
        get_current_user(credentials, db_session)

    assert raised.value.status_code == 401
    assert raised.value.detail == "Invalid token payload"
    assert raised.value.headers == {"WWW-Authenticate": "Bearer"}
    assert db_session.info["user_query_count"] == 0


@pytest.mark.parametrize(
    "claims",
    (
        {"iat": [], "exp": "not-a-number"},
        {"iat": float("inf"), "exp": "not-a-number"},
    ),
    ids=("type-error-before-value-error", "overflow-error-before-value-error"),
)
def test_mixed_temporal_claims_preserve_original_error_proof_after_value_error(
    db_session: Session, claims: dict[str, object]
) -> None:
    """A dependency-handled temporal ValueError cannot replace the original error."""
    token = _access_token(**claims)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with pytest.raises(HTTPException) as raised:
        get_current_user(credentials, db_session)

    assert raised.value.status_code == 401
    assert raised.value.detail == "Invalid token payload"
    assert raised.value.headers == {"WWW-Authenticate": "Bearer"}
    assert db_session.info["user_query_count"] == 0


@pytest.mark.parametrize("claim", ("exp", "nbf", "iat"))
def test_numeric_temporal_claims_keep_the_library_accepted_behavior(
    db_session: Session, claim: str
) -> None:
    """Integer NumericDate values continue through verified token validation."""
    now = int(datetime.now(timezone.utc).timestamp())
    value = now + 300 if claim == "exp" else now - 300
    token = _access_token(**{claim: value})
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    user = get_current_user(credentials, db_session)

    assert user.username == "existing-user"
    assert db_session.info["user_query_count"] == 1


@pytest.mark.parametrize("exception_type", (TypeError, OverflowError))
def test_unrelated_decode_type_errors_propagate_by_identity(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[Exception],
) -> None:
    """Only proven temporal-claim conversion failures are credential rejections."""
    original = exception_type("unrelated decode failure")

    def raise_original(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise original

    monkeypatch.setattr(auth_dependencies.jwt, "decode", raise_original)
    monkeypatch.setattr(
        auth_dependencies.jwt, "get_unverified_claims", lambda _token: {"aud": "x"}
    )
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")

    with pytest.raises(exception_type) as raised:
        get_current_user(credentials, db_session)

    assert raised.value is original
    assert db_session.info["user_query_count"] == 0


@pytest.mark.parametrize(
    ("username", "user_id"),
    (
        ("", 0),
        ("long-" * 20, -1),
        ("sqlite\x00nul", 2**31),
        ("中文-😀", 2**63 - 1),
        ("signed-64-minimum", -(2**63)),
    ),
    ids=("empty", "long", "nul", "unicode-and-signed-64-max", "signed-64-min"),
)
def test_sqlite_bindable_claims_reach_matching_persisted_users(
    db_session: Session, username: str, user_id: int
) -> None:
    """SQLite-compatible claims preserve real persisted user identities."""
    db_session.add(
        User(
            id=user_id,
            username=username,
            email=f"sqlite-{user_id}@example.com",
            password_hash="not-used-by-auth-dependencies",
        )
    )
    db_session.commit()
    db_session.info["user_query_count"] = 0
    token = _access_token(sub=username, user_id=user_id)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    user = get_current_user(credentials, db_session)

    assert user.id == user_id
    assert user.username == username
    assert db_session.info["user_query_count"] == 1


@pytest.mark.parametrize(
    "claims",
    (
        {"user_id": 2**63},
        {"user_id": -(2**63) - 1},
        {"sub": "\ud800"},
    ),
    ids=("above-signed-64", "below-signed-64", "lone-surrogate"),
)
def test_sqlite_unbindable_claims_are_rejected_without_a_user_query(
    db_session: Session, claims: dict[str, object]
) -> None:
    """SQLite claim values outside the driver's bindability contract are rejected."""
    token = _access_token(**claims)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with pytest.raises(HTTPException) as raised:
        get_current_user(credentials, db_session)

    assert raised.value.detail == "Invalid token payload"
    assert db_session.info["user_query_count"] == 0


class _DialectSession:
    """A query-counting Session shape with an already-bound dialect."""

    def __init__(self, dialect_name: str, user: User | None = None) -> None:
        self._bind = SimpleNamespace(dialect=SimpleNamespace(name=dialect_name))
        self._user = user
        self.bind_count = 0
        self.query_count = 0

    def get_bind(self) -> SimpleNamespace:
        self.bind_count += 1
        return self._bind

    def query(self, _model: type[User]) -> "_DialectSession":
        self.query_count += 1
        return self

    def filter(self, *_conditions: object) -> "_DialectSession":
        return self

    def first(self) -> User | None:
        return self._user


def test_wrong_type_user_id_is_rejected_before_reading_database_binding() -> None:
    db = _DialectSession("sqlite")
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=_access_token(user_id="1")
    )

    with pytest.raises(HTTPException):
        get_current_user(credentials, db)  # type: ignore[arg-type]

    assert db.bind_count == 0
    assert db.query_count == 0


@pytest.mark.parametrize(
    "claims",
    (
        {"user_id": 2**31},
        {"user_id": -(2**31) - 1},
        {"sub": "postgresql\x00nul"},
        {"sub": "\udfff"},
    ),
    ids=("above-signed-32", "below-signed-32", "nul", "lone-surrogate"),
)
def test_postgresql_unbindable_claims_are_rejected_without_a_user_query(
    claims: dict[str, object],
) -> None:
    """PostgreSQL-specific unbindable claims do not enter the User query."""
    db = _DialectSession("postgresql")
    token = _access_token(**claims)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with pytest.raises(HTTPException) as raised:
        get_current_user(credentials, db)  # type: ignore[arg-type]

    assert raised.value.detail == "Invalid token payload"
    assert db.query_count == 0


@pytest.mark.parametrize("username", ("", "long-" * 20))
def test_postgresql_empty_and_long_claims_remain_queryable(username: str) -> None:
    """PostgreSQL does not impose unsupported empty or length claim rules."""
    expected = User(
        id=2**31 - 1,
        username=username,
        email="postgresql@example.com",
        password_hash="not-used-by-auth-dependencies",
    )
    db = _DialectSession("postgresql", expected)
    token = _access_token(sub=username, user_id=2**31 - 1)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    assert get_current_user(credentials, db) is expected  # type: ignore[arg-type]
    assert db.query_count == 1


def test_unrecognized_dialect_runs_the_query_and_propagates_backend_failure() -> None:
    """Unknown dialects retain operational behavior instead of invented limits."""
    original = RuntimeError("backend-specific failure")

    class FailingDialectSession(_DialectSession):
        def query(self, _model: type[User]) -> "_DialectSession":
            self.query_count += 1
            raise original

    db = FailingDialectSession("unrecognized")
    token = _access_token(sub="\x00still-queryable", user_id=2**63)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with pytest.raises(RuntimeError) as raised:
        get_current_user(credentials, db)  # type: ignore[arg-type]

    assert raised.value is original
    assert db.query_count == 1


@pytest.mark.parametrize(
    "adapter",
    (
        lambda credentials, db: get_current_user(credentials, db),
        lambda credentials, db: get_current_user_optional(credentials, db),
        lambda credentials, db: get_user_from_token(credentials.credentials, db),
        lambda credentials, db: get_user_from_websocket_token(
            credentials.credentials, db
        ),
    ),
    ids=("required", "optional", "token", "websocket-token-alias"),
)
def test_auth_adapters_propagate_database_pool_timeout_by_identity(
    adapter: object,
) -> None:
    """Operational database failures never become an authentication absence."""
    original = SQLAlchemyTimeoutError("auth pool timeout", None, None)

    class TimeoutSession(_DialectSession):
        def query(self, _model: type[User]) -> "_DialectSession":
            self.query_count += 1
            raise original

    db = TimeoutSession("sqlite")
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=_access_token()
    )

    with pytest.raises(SQLAlchemyTimeoutError) as raised:
        adapter(credentials, db)  # type: ignore[operator,arg-type]

    assert raised.value is original
    assert db.query_count == 1


def test_token_adapter_propagates_real_queue_pool_checkout_timeout() -> None:
    """A real exhausted pool stays on the operational exception channel."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.01,
    )
    held_connection = engine.connect()
    session = sessionmaker(bind=engine)()
    try:
        with pytest.raises(SQLAlchemyTimeoutError):
            get_user_from_token(_access_token(), session)
    finally:
        session.close()
        held_connection.close()
        engine.dispose()


def test_optional_direct_none_returns_none(db_session: Session) -> None:
    """The optional helper preserves its direct no-credential contract."""
    assert get_current_user_optional(None, db_session) is None


@pytest.mark.parametrize(
    ("dependency_kind", "credential_kind", "expected_status"),
    (
        ("required", "missing", (401, 403)),
        ("required", "empty-bearer", (401, 403)),
        ("required", "basic", (401, 403)),
        ("optional", "missing", (200,)),
        ("optional", "empty-bearer", (200,)),
        ("optional", "basic", (200,)),
        ("required", "valid", (500,)),
        ("optional", "valid", (500,)),
    ),
)
def test_fastapi_auth_dependencies_preserve_framework_rejection_and_safe_pool_failure(
    dependency_kind: str,
    credential_kind: str,
    expected_status: tuple[int, ...],
) -> None:
    driver_message = "database pool driver detail"

    class TimeoutSession(_DialectSession):
        def query(self, _model: type[User]) -> "_DialectSession":
            self.query_count += 1
            raise SQLAlchemyTimeoutError(driver_message, None, None)

    app = FastAPI()
    app.add_exception_handler(Exception, global_exception_handler)

    @app.get("/optional")
    def optional_endpoint(
        user: User | None = Depends(get_current_user_optional),
    ) -> dict[str, int | None]:
        return {"user_id": user.id if user is not None else None}

    @app.get("/required")
    def required_endpoint(user: User = Depends(get_current_user)) -> dict[str, int]:
        return {"user_id": user.id}

    app.dependency_overrides[auth_dependencies.get_db] = lambda: TimeoutSession(
        "sqlite"
    )

    headers = {
        "missing": {},
        "empty-bearer": {"Authorization": "Bearer "},
        "basic": {"Authorization": "Basic abc"},
        "valid": {"Authorization": f"Bearer {_access_token()}"},
    }[credential_kind]
    path = f"/{dependency_kind}"
    response = TestClient(app, raise_server_exceptions=False).get(path, headers=headers)

    assert response.status_code in expected_status
    if dependency_kind == "optional" and credential_kind != "valid":
        assert response.json() == {"user_id": None}
    if credential_kind == "valid":
        assert driver_message not in response.text
        assert "Traceback" not in response.text


@pytest.mark.parametrize("helper", (get_user_from_token, get_user_from_websocket_token))
@pytest.mark.parametrize("token", ("", "not-a-jwt"), ids=("empty", "garbage"))
def test_direct_token_helpers_reject_malformed_input_without_query(
    db_session: Session, helper: object, token: str
) -> None:
    assert helper(token, db_session) is None  # type: ignore[operator]
    assert db_session.info["user_query_count"] == 0


@pytest.mark.parametrize(
    "adapter",
    (
        lambda credentials, db: get_current_user(credentials, db),
        lambda credentials, db: get_current_user_optional(credentials, db),
        lambda credentials, db: get_user_from_token(
            f"Bearer {credentials.credentials}", db
        ),
        lambda credentials, db: get_user_from_websocket_token(
            credentials.credentials, db
        ),
    ),
    ids=("required", "optional", "token-bearer-prefix", "websocket-token-alias"),
)
def test_valid_access_token_returns_matching_user_across_adapters(
    db_session: Session, adapter: object
) -> None:
    """All public access-token adapters share the same successful resolution."""
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=_access_token()
    )

    user = adapter(credentials, db_session)  # type: ignore[operator]

    assert user is not None
    assert user.id == 1
    assert user.username == "existing-user"


def test_invalid_claims_read_the_bound_dialect_without_pool_checkout() -> None:
    """Pre-query rejection inspects Session binding without acquiring a connection."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.01,
    )
    held_connection = engine.connect()
    session = sessionmaker(bind=engine)()
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=_access_token(user_id=2**63)
    )
    try:
        with pytest.raises(HTTPException) as raised:
            get_current_user(credentials, session)
    finally:
        session.close()
        held_connection.close()
        engine.dispose()

    assert raised.value.detail == "Invalid token payload"


def test_auth_consumer_topology_is_closed_across_source_tree() -> None:
    """Authentication consumers stay within the reviewed ownership topology."""
    tracked = {
        "_resolve_access_token_user",
        "get_current_user",
        "get_current_user_optional",
        "get_user_from_token",
        "get_user_from_websocket_token",
        "get_authenticated_user",
    }
    auth_file = "web/auth_dependencies.py"
    websocket_file = "web/api/websocket.py"
    rejected = ("_AccessTokenRejected",)
    terminated = ("_WebSocketAuthenticationTerminated",)
    expected = (
        (auth_file, "get_current_user", "_resolve_access_token_user", rejected),
        (
            auth_file,
            "get_current_user_optional",
            "_resolve_access_token_user",
            rejected,
        ),
        (auth_file, "get_user_from_token", "_resolve_access_token_user", rejected),
        (auth_file, "get_user_from_websocket_token", "get_user_from_token", ()),
        (
            "web/api/websocket_auth.py",
            "_load_websocket_principal_sync",
            "get_user_from_websocket_token",
            (),
        ),
        (
            websocket_file,
            "websocket_chat_endpoint",
            "get_authenticated_user",
            terminated,
        ),
        (
            websocket_file,
            "websocket_builder_chat_endpoint",
            "get_authenticated_user",
            terminated,
        ),
        (
            websocket_file,
            "websocket_build_preview_endpoint",
            "get_authenticated_user",
            terminated,
        ),
        (
            "web/api/progress_ws.py",
            "progress_websocket_endpoint",
            "get_authenticated_user",
            terminated,
        ),
        (
            "web/api/files.py",
            "_user_from_bearer_or_stream_ticket",
            "get_current_user",
            (),
        ),
    )
    modules = {"xagent.web.auth_dependencies", "xagent.web.api.websocket_auth"}
    source_root = Path(auth_dependencies.__file__).parents[1]
    actual: Counter[tuple[str, str, str, tuple[str, ...]]] = Counter()
    lines: dict[tuple[str, str, str, tuple[str, ...]], list[int]] = {}
    escapes: list[tuple[tuple[str, str, str, tuple[str, ...]], int]] = []

    def handler_names(node: ast.expr | None) -> tuple[str, ...]:
        if node is None:
            return ("<bare>",)
        if isinstance(node, ast.Tuple):
            return tuple(name for item in node.elts for name in handler_names(item))
        return (node.id if isinstance(node, ast.Name) else node.attr,)

    def module_path(node: ast.expr, bases: dict[str, str]) -> str | None:
        if isinstance(node, ast.Name):
            return bases.get(node.id)
        if isinstance(node, ast.Attribute):
            base = module_path(node.value, bases)
            return f"{base}.{node.attr}" if base else None
        return None

    for source_path in source_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        relative_path = source_path.relative_to(source_root).as_posix()
        names = {
            node.name
            for node in tree.body
            if relative_path == "web/auth_dependencies.py"
            and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in tracked
        }
        bases: dict[str, str] = {}
        for node in ast.walk(tree):
            if node in tree.body:
                continue
            if isinstance(node, ast.ImportFrom):
                leaf = (node.module or "").rsplit(".", 1)[-1]
                for alias in node.names:
                    tracked_symbol = (
                        leaf in {"auth_dependencies", "websocket_auth"}
                        and alias.name in tracked
                    )
                    tracked_module = alias.name in {
                        "auth_dependencies",
                        "websocket_auth",
                    } and (
                        node.level > 0
                        or node.module in {"xagent.web", "xagent.web.api"}
                    )
                    if tracked_symbol or tracked_module:
                        escapes.append(
                            (
                                (
                                    relative_path,
                                    "<local import>",
                                    alias.name,
                                    (),
                                ),
                                alias.lineno,
                            )
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in modules:
                        escapes.append(
                            (
                                (
                                    relative_path,
                                    "<local import>",
                                    alias.name,
                                    (),
                                ),
                                alias.lineno,
                            )
                        )
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                leaf = (node.module or "").rsplit(".", 1)[-1]
                for alias in node.names:
                    if (
                        leaf in {"auth_dependencies", "websocket_auth"}
                        and alias.name in tracked
                    ):
                        if alias.asname:
                            escapes.append(
                                (
                                    (relative_path, "<module>", alias.name, ()),
                                    alias.lineno,
                                )
                            )
                        else:
                            names.add(alias.name)
                    if node.level > 0 and alias.name in {
                        "auth_dependencies",
                        "websocket_auth",
                    }:
                        if alias.asname:
                            escapes.append(
                                (
                                    (relative_path, "<module>", alias.name, ()),
                                    alias.lineno,
                                )
                            )
                        else:
                            bases[alias.name] = (
                                "xagent.web.auth_dependencies"
                                if alias.name == "auth_dependencies"
                                else "xagent.web.api.websocket_auth"
                            )
                    elif node.module in {
                        "xagent.web",
                        "xagent.web.api",
                    } and alias.name in {"auth_dependencies", "websocket_auth"}:
                        bases[alias.asname or alias.name] = (
                            f"{node.module}.{alias.name}"
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in modules:
                        bases[alias.asname or alias.name.split(".")[0]] = (
                            alias.name if alias.asname else alias.name.split(".")[0]
                        )

        for node in ast.walk(tree):
            callee = (
                node.id if isinstance(node, ast.Name) and node.id in names else None
            )
            if (
                isinstance(node, ast.Attribute)
                and module_path(node.value, bases) in modules
                and node.attr in tracked
            ):
                callee = node.attr
            if callee is None:
                continue
            chain = [node]
            while parent := parents.get(chain[-1]):
                chain.append(parent)
            scope = (
                ".".join(
                    ancestor.name
                    for ancestor in reversed(chain)
                    if isinstance(
                        ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    )
                )
                or "<module>"
            )
            catch_policy = ()
            for child, ancestor in zip(chain, chain[1:]):
                if (
                    isinstance(ancestor, (ast.Try, ast.TryStar))
                    and child in ancestor.body
                ):
                    catch_policy = tuple(
                        name
                        for handler in ancestor.handlers
                        for name in handler_names(handler.type)
                    )
                    break
            fact = (relative_path, scope, callee, catch_policy)
            parent = parents.get(node)
            safe_depends = (
                callee == "get_current_user"
                and isinstance(parent, ast.Call)
                and isinstance(parent.func, ast.Name)
                and parent.func.id == "Depends"
                and bool(parent.args)
                and parent.args[0] is node
            )
            if safe_depends:
                continue
            if isinstance(parent, ast.Call) and parent.func is node:
                actual[fact] += 1
                lines.setdefault(fact, []).append(node.lineno)
            else:
                escapes.append((fact, node.lineno))

    assert not escapes, f"unreviewed first-class callable escapes: {escapes}"
    assert actual == Counter(expected), (
        "authentication consumer facts differ: "
        f"actual={actual}; expected={Counter(expected)}; lines={lines}"
    )
