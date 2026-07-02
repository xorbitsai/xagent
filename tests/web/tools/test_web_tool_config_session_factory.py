import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.tools.config import WebToolConfig


def _factory():
    engine = create_engine("sqlite://")  # in-memory, fresh
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


class _Chain:
    """Minimal chainable query stub: filter/join return self, terminals empty."""

    def filter(self, *a, **k):
        return self

    def join(self, *a, **k):
        return self

    def all(self):
        return []

    def first(self):
        return None


class _TrackingSession:
    """Records whether ``.query`` was driven (i.e. the session was used)."""

    def __init__(self):
        self.query_calls = 0
        self.closed = False

    def query(self, *a, **k):
        self.query_calls += 1
        return _Chain()

    def close(self):
        self.closed = True


class _FailingQuerySession:
    def __init__(self):
        self.query_calls = 0

    def query(self, *a, **k):
        self.query_calls += 1
        raise RuntimeError("db down")


def _mcp_server(**overrides):
    data = {
        "id": 1,
        "name": "local",
        "transport": "stdio",
        "description": "Local MCP",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-everything"],
        "env": None,
        "cwd": None,
        "url": None,
        "headers": None,
        "auth": None,
        "managed": "external",
        "concurrency_safe": False,
        "concurrent_tools": [],
        "_decrypt_auth_config": lambda auth: auth,
        "_merge_auth_headers": lambda headers, auth: headers,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _mcp_db(servers):
    db = MagicMock()
    db.query.return_value.join.return_value.filter.return_value.all.return_value = (
        servers
    )
    return db


def test_get_session_factory_prefers_injected_factory():
    factory = _factory()
    cfg = WebToolConfig(db=None, request=None, db_factory=factory)
    assert cfg.get_session_factory() is factory


def test_factory_built_get_db_is_lazy_and_closed_by_close():
    factory = _factory()
    cfg = WebToolConfig(db=None, request=None, db_factory=factory)
    db1 = cfg.get_db()
    db2 = cfg.get_db()
    assert db1 is db2  # cached, single construction-time session
    cfg.close()
    # closing twice is safe
    cfg.close()


def test_live_db_path_unchanged():
    sentinel = object()
    cfg = WebToolConfig(db=sentinel, request=None)
    assert cfg.get_db() is sentinel
    cfg.close()  # must not raise; caller owns the request session


def test_custom_api_loader_uses_factory_session():
    # Factory-only (nested child) config: the loader must mint/reuse the lazy
    # factory session via get_db(), not read the None live ``self.db`` and
    # silently swallow ``None.query`` into an empty tool list.
    sess = _TrackingSession()
    cfg = WebToolConfig(db=None, request=None, db_factory=lambda: sess, user_id=1)
    cfg.get_custom_api_configs()
    assert sess.query_calls >= 1


def test_mcp_loader_uses_factory_session():
    sess = _TrackingSession()
    cfg = WebToolConfig(db=None, request=None, db_factory=lambda: sess, user_id=1)
    asyncio.run(cfg._load_mcp_server_configs())
    assert sess.query_calls >= 1


def test_mcp_loader_db_failure_returns_empty_without_cache():
    sess = _FailingQuerySession()
    cfg = WebToolConfig(db=sess, request=None, user_id=1)

    assert asyncio.run(cfg.get_mcp_server_configs()) == []
    assert asyncio.run(cfg.get_mcp_server_configs()) == []
    assert sess.query_calls == 2
    assert cfg._cached_mcp_configs is None


def test_mcp_loader_skips_bad_server_and_does_not_cache_partial_result():
    def bad_decrypt(_auth):
        raise RuntimeError("bad auth")

    bad = _mcp_server(
        id=1,
        name="bad",
        transport="streamable_http",
        url="https://bad.example/mcp",
        auth={"type": "bearer", "bearer_token": "x"},
        _decrypt_auth_config=bad_decrypt,
    )
    good = _mcp_server(id=2, name="good")
    db = _mcp_db([bad, good])
    cfg = WebToolConfig(db=db, request=None, user_id=1)

    configs = asyncio.run(cfg.get_mcp_server_configs())

    assert [config["name"] for config in configs] == ["good"]
    # The bad server is never cached (so it's retried every call); the good
    # server IS cached, since a single permanently-broken server must not
    # block caching for the rest. See
    # test_mcp_loader_caches_good_server_and_retries_only_bad_server below
    # for the assertion that the good server's expensive per-server work
    # (decrypt/header-merge) is not repeated on the next call.
    assert cfg._cached_mcp_configs is not None
    assert set(cfg._cached_mcp_configs.keys()) == {2}
    assert cfg._mcp_load_errored_server_ids == {1}
    asyncio.run(cfg.get_mcp_server_configs())
    assert db.query.call_count == 2


def test_mcp_loader_caches_good_server_and_retries_only_bad_server():
    """A permanently-broken server must not block caching for the rest.

    The DB query itself always reruns (to detect add/remove/changes), but
    the expensive per-server work -- here observed via calls to the
    server's own ``_decrypt_auth_config``/``_merge_auth_headers`` mocks --
    must not be repeated for a server that already succeeded, while a
    server that keeps failing must be retried on every call.
    """

    def bad_decrypt(_auth):
        raise RuntimeError("bad auth")

    good_decrypt_calls = []
    bad_decrypt_calls = []

    def good_decrypt(auth):
        good_decrypt_calls.append(auth)
        return auth

    def tracking_bad_decrypt(auth):
        bad_decrypt_calls.append(auth)
        return bad_decrypt(auth)

    bad = _mcp_server(
        id=1,
        name="bad",
        transport="streamable_http",
        url="https://bad.example/mcp",
        auth={"type": "bearer", "bearer_token": "x"},
        _decrypt_auth_config=tracking_bad_decrypt,
    )
    good = _mcp_server(
        id=2,
        name="good",
        transport="streamable_http",
        url="https://good.example/mcp",
        auth=None,
        _decrypt_auth_config=good_decrypt,
    )
    db = _mcp_db([bad, good])
    cfg = WebToolConfig(db=db, request=None, user_id=1)

    configs1 = asyncio.run(cfg.get_mcp_server_configs())
    assert [c["name"] for c in configs1] == ["good"]
    assert len(good_decrypt_calls) == 1
    assert len(bad_decrypt_calls) == 1

    configs2 = asyncio.run(cfg.get_mcp_server_configs())
    assert [c["name"] for c in configs2] == ["good"]
    # Good server's expensive per-server work is NOT repeated.
    assert len(good_decrypt_calls) == 1
    # Bad server is retried every call.
    assert len(bad_decrypt_calls) == 2
    # The DB query itself still reruns every call (to detect add/remove).
    assert db.query.call_count == 2


def test_mcp_loader_drops_server_removed_between_calls():
    """A server cached as healthy must disappear once removed/unlinked."""
    good = _mcp_server(id=2, name="good")
    db = _mcp_db([good])
    cfg = WebToolConfig(db=db, request=None, user_id=1)

    configs1 = asyncio.run(cfg.get_mcp_server_configs())
    assert [c["name"] for c in configs1] == ["good"]
    assert set(cfg._cached_mcp_configs.keys()) == {2}

    # Server removed/unlinked: the query now returns nothing for this user.
    db.query.return_value.join.return_value.filter.return_value.all.return_value = []

    configs2 = asyncio.run(cfg.get_mcp_server_configs())
    assert configs2 == []
    assert cfg._cached_mcp_configs == {}


def test_mcp_loader_skips_oauth_mcp_server_when_user_id_is_none():
    """An oauth_mcp-flagged server must be skipped, not appended unauthenticated,
    when there's no user context to bind a token store to."""
    server = _mcp_server(
        id=3,
        name="oauth-server",
        transport="streamable_http",
        url="https://oauth.example/mcp",
        auth={"type": "oauth_mcp"},
        _decrypt_auth_config=lambda auth: auth,
        _merge_auth_headers=lambda headers, auth: headers,
    )
    db = _mcp_db([server])
    cfg = WebToolConfig(db=db, request=None, user_id=1)
    # Force the "no resolvable user" state directly, matching the pattern
    # used in test_web_tool_config_reauth_hook.py -- the constructor always
    # falls back to a default int user id, so this is the only way to
    # exercise this guard.
    cfg._user_id = None

    configs = asyncio.run(cfg.get_mcp_server_configs())

    assert configs == []


def test_mcp_loader_does_not_attach_oauth_provider_to_websocket():
    server = _mcp_server(
        name="ws",
        transport="websocket",
        url="ws://mcp.example/ws",
        auth={"type": "oauth_mcp"},
        _decrypt_auth_config=lambda auth: auth,
        _merge_auth_headers=lambda headers, auth: headers,
    )
    db = _mcp_db([server])
    cfg = WebToolConfig(db=db, request=None, user_id=1)

    configs = asyncio.run(cfg.get_mcp_server_configs())

    connection = configs[0]["config"]
    assert "oauth_mcp" not in connection
    assert "auth" not in connection
