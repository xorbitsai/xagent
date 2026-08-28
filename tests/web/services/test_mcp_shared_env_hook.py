"""Tests for the shared (app-injected) MCP env hook and layered merge."""

import asyncio

import pytest

from xagent.web.services import mcp_runtime


def test_merge_stdio_env_layers_later_wins():
    """global -> shared -> user, each layer overriding the previous."""
    merged = mcp_runtime.merge_stdio_env(
        {"A": "g", "B": "g", "C": "g"},
        {"B": "s", "C": "s"},
        {"C": "u"},
    )
    assert merged == {"A": "g", "B": "s", "C": "u"}


def test_merge_stdio_env_two_arg_unchanged():
    """Existing two-arg callers keep working (user over global)."""
    assert mcp_runtime.merge_stdio_env({"A": "g"}, {"A": "u"}) == {"A": "u"}
    assert mcp_runtime.merge_stdio_env({"A": "g"}, None) == {"A": "g"}


def test_resolve_stdio_env_by_source():
    """env_source selects which layer to apply over the global env."""
    g = {"A": "g", "B": "g"}
    shared = {"B": "team"}
    user = {"B": "own"}
    assert mcp_runtime.resolve_stdio_env("platform", g, shared, user) == g
    assert mcp_runtime.resolve_stdio_env("shared", g, shared, user) == {
        "A": "g",
        "B": "team",
    }
    assert mcp_runtime.resolve_stdio_env("own", g, shared, user) == {
        "A": "g",
        "B": "own",
    }
    # None keeps the legacy fallback: global < shared < user (most specific wins).
    assert mcp_runtime.resolve_stdio_env(None, g, shared, user) == {
        "A": "g",
        "B": "own",
    }


def test_resolve_stdio_env_missing_chosen_layer_falls_to_global():
    """A pick whose layer is empty falls back to global, never a blank key."""
    g = {"A": "g"}
    assert mcp_runtime.resolve_stdio_env("shared", g, None, {"A": "own"}) == g
    assert mcp_runtime.resolve_stdio_env("own", g, {"A": "team"}, None) == g


def test_shared_env_hook_default_is_noop():
    mcp_runtime.set_mcp_shared_env_hook(None)
    assert mcp_runtime.load_shared_env_overrides(object(), 1) == {}


def test_shared_env_hook_is_invoked():
    calls = {}

    def hook(db, user_id):
        calls["args"] = (db, user_id)
        return {5: {"KEY": "shared-value"}}

    mcp_runtime.set_mcp_shared_env_hook(hook)
    try:
        assert mcp_runtime.load_shared_env_overrides("db", 7) == {
            5: {"KEY": "shared-value"}
        }
        assert calls["args"] == ("db", 7)
    finally:
        mcp_runtime.set_mcp_shared_env_hook(None)


def test_shared_env_hook_none_user_returns_empty():
    mcp_runtime.set_mcp_shared_env_hook(lambda db, uid: {1: {"K": "v"}})
    try:
        assert mcp_runtime.load_shared_env_overrides("db", None) == {}
    finally:
        mcp_runtime.set_mcp_shared_env_hook(None)


def test_shared_env_hook_failure_degrades_to_empty():
    def boom(db, user_id):
        raise RuntimeError("hook exploded")

    mcp_runtime.set_mcp_shared_env_hook(boom)
    try:
        assert mcp_runtime.load_shared_env_overrides("db", 7) == {}
    finally:
        mcp_runtime.set_mcp_shared_env_hook(None)


def test_caller_id_env_returns_user_id_keyed_var():
    assert mcp_runtime.caller_id_env(42) == {"XAGENT_MCP_CALLER_ID": "42"}


def test_caller_id_env_empty_without_user_id():
    assert mcp_runtime.caller_id_env(None) == {}


def test_caller_id_env_var_name_matches_aws_connector_literal():
    """CALLER_ID_ENV_VAR is duplicated as an independent literal in
    xagent.web.tools.mcp.aws (a standalone subprocess entrypoint that can't
    import this module), with no other check tying the two together. A
    rename on either side would silently degrade attribution back to the
    un-attributed default without failing any other test."""
    from xagent.web.tools.mcp import aws

    assert aws.CALLER_ID_ENV_VAR == mcp_runtime.CALLER_ID_ENV_VAR


class _FakeStdioServer:
    """Minimal stand-in for MCPServer: just enough for the stdio branch of
    build_mcp_runtime_connection (transport check + auth decrypt call)."""

    def __init__(self, connection, server_id=5):
        self._connection = connection
        self.id = server_id
        self.transport = connection.get("transport")

    def to_connection_dict(self):
        return dict(self._connection)

    def _decrypt_auth_config(self, auth):
        return {}


def test_build_mcp_runtime_connection_injects_caller_id_for_stdio():
    """Every stdio connection gets the caller's user id, even with no other
    env overrides present — this is what lets a connector like the AWS MCP
    module attribute its own external calls to a specific xagent user."""
    server = _FakeStdioServer({"transport": "stdio", "env": {"FOO": "bar"}})

    build = asyncio.run(
        mcp_runtime.build_mcp_runtime_connection(db=None, server=server, user_id=42)
    )

    assert build.connection["env"] == {"FOO": "bar", "XAGENT_MCP_CALLER_ID": "42"}


def test_build_mcp_runtime_connection_omits_caller_id_without_user():
    server = _FakeStdioServer({"transport": "stdio", "env": {"FOO": "bar"}})

    build = asyncio.run(
        mcp_runtime.build_mcp_runtime_connection(db=None, server=server, user_id=None)
    )

    assert build.connection["env"] == {"FOO": "bar"}


def test_build_mcp_runtime_connection_caller_id_wins_over_user_overrides():
    """When a real user_id is present, overrides still apply, but the
    caller-id layer is merged in last so it can never be shadowed by a
    stale/forged XAGENT_MCP_CALLER_ID override. This guarantee is
    conditional on user_id being set — see the user_id=None test below,
    where caller_id_env contributes nothing and a forged override passes
    through unchanged."""
    server = _FakeStdioServer({"transport": "stdio", "env": {"FOO": "global"}})

    build = asyncio.run(
        mcp_runtime.build_mcp_runtime_connection(
            db=None,
            server=server,
            user_id=42,
            user_env_overrides={5: {"XAGENT_MCP_CALLER_ID": "spoofed"}},
        )
    )

    assert build.connection["env"]["XAGENT_MCP_CALLER_ID"] == "42"


def test_build_mcp_runtime_connection_without_user_id_does_not_scrub_forged_override():
    """Without a user_id, caller_id_env returns {} and contributes nothing —
    a pre-existing or forged XAGENT_MCP_CALLER_ID override passes through to
    the subprocess env unchanged. The "can never be shadowed" guarantee in
    the test above only holds once user_id is set."""
    server = _FakeStdioServer({"transport": "stdio", "env": {"FOO": "global"}})

    build = asyncio.run(
        mcp_runtime.build_mcp_runtime_connection(
            db=None,
            server=server,
            user_id=None,
            user_env_overrides={5: {"XAGENT_MCP_CALLER_ID": "spoofed"}},
        )
    )

    assert build.connection["env"]["XAGENT_MCP_CALLER_ID"] == "spoofed"


class _FakeHttpOAuthServer:
    """Stand-in for an mcp_oauth-authorized HTTP MCPServer row."""

    def __init__(self, transport, server_id=7):
        self.id = server_id
        self.name = "remote-notes"
        self.transport = transport
        self.url = "https://mcp.example.com/mcp"

    def to_connection_dict(self):
        return {
            "transport": self.transport,
            "url": self.url,
            "auth": {"type": "mcp_oauth"},
        }

    def _decrypt_auth_config(self, auth):
        return {"type": "mcp_oauth"}


def test_mcp_oauth_http_classification_is_case_insensitive():
    """A row stored before write-time transport normalization can hold
    "Streamable_HTTP". A False here is not a rejection: the caller returns
    the connection as-is, skipping the per-user OAuth token exchange, so the
    server would be handed to the runtime without a grant."""
    server = _FakeHttpOAuthServer("Streamable_HTTP")

    assert (
        mcp_runtime._is_mcp_oauth_http_server(server, server._decrypt_auth_config(None))
        is True
    )


@pytest.mark.parametrize(
    "transport",
    ["stdio", "oauth", "oauth2", "http", "streamable", "custom_api", ""],
    ids=["stdio", "oauth", "oauth2", "http", "prefix", "custom-api", "empty"],
)
def test_mcp_oauth_http_classification_excludes_non_http_transports(transport):
    """Normalizing case must not widen the set. `stdio` alone does not
    discriminate -- it is excluded with or without normalization. These
    include values that a sloppy normalizer (substring/prefix matching, or
    treating "oauth" as HTTP) would wrongly admit."""
    server = _FakeHttpOAuthServer(transport)

    assert (
        mcp_runtime._is_mcp_oauth_http_server(server, server._decrypt_auth_config(None))
        is False
    )


def test_mixed_case_mcp_oauth_server_is_not_served_without_a_grant():
    """The end-to-end consequence: build_mcp_runtime_connection must refuse a
    mixed-case mcp_oauth row (no db/user context to resolve a grant) rather
    than fall through and return it unauthorized."""
    server = _FakeHttpOAuthServer("Streamable_HTTP")

    build = asyncio.run(
        mcp_runtime.build_mcp_runtime_connection(db=None, server=server, user_id=42)
    )

    assert build.connection is None
    assert build.diagnostic is not None
    assert build.diagnostic["code"] == "authorization_required"


def test_normalize_transport_canonicalizes_case_and_padding():
    assert mcp_runtime.normalize_transport("  Streamable_HTTP ") == "streamable_http"
    assert mcp_runtime.normalize_transport("SSE") == "sse"
    assert mcp_runtime.normalize_transport(None) == ""
