"""Unit tests for the single source of truth for catalog app auth classification.

`classify_app_auth` is the one place backend + both frontend dialogs read from,
so its edge cases matter: oauth wins over any launch_config, key-based needs
BOTH required_env and command, a stdio command with no required_env is keyless,
everything else is unconnectable.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from xagent.web import mcp_apps
from xagent.web.mcp_apps import classify_app_auth, get_app_by_id
from xagent.web.models.public_mcp import PublicMCPApp


def test_oauth_transport_is_builtin_oauth():
    assert classify_app_auth("oauth", None) == "builtin_oauth"
    assert classify_app_auth("OAuth", {}) == "builtin_oauth"
    # oauth transport wins even if a stray launch_config is present
    assert (
        classify_app_auth("oauth", {"command": "npx", "required_env": ["X"]})
        == "builtin_oauth"
    )


def test_key_based_needs_command_and_required_env():
    assert (
        classify_app_auth("stdio", {"command": "npx", "required_env": ["KEY"]})
        == "api_key"
    )


def test_remote_mcp_oauth_shape_is_mcp_oauth():
    for transport in ("sse", "websocket", "streamable_http"):
        assert (
            classify_app_auth(
                transport,
                {"url": "https://mcp.example.com/mcp", "auth": {"type": "mcp_oauth"}},
            )
            == "mcp_oauth"
        )


def test_mcp_oauth_requires_both_url_and_auth_type():
    # url without auth.type=mcp_oauth -> not launchable via OAuth
    assert (
        classify_app_auth("streamable_http", {"url": "https://mcp.example.com/mcp"})
        == "unconnectable"
    )
    # auth.type=mcp_oauth without url -> nothing to connect to
    assert (
        classify_app_auth("streamable_http", {"auth": {"type": "mcp_oauth"}})
        == "unconnectable"
    )
    # wrong auth.type is not mcp_oauth
    assert (
        classify_app_auth(
            "streamable_http",
            {"url": "https://mcp.example.com/mcp", "auth": {"type": "bearer"}},
        )
        == "unconnectable"
    )


def test_remote_mcp_oauth_transport_check_is_case_insensitive():
    # N1: the builtin_oauth check above lowercases ("OAuth" -> "oauth"); the
    # remote-transport check must be equally forgiving, or an admin PATCH that
    # stores a mixed-case transport puts the two halves of this feature in
    # disagreement about the same row.
    assert (
        classify_app_auth(
            "Streamable_HTTP",
            {"url": "https://mcp.example.com/mcp", "auth": {"type": "mcp_oauth"}},
        )
        == "mcp_oauth"
    )
    assert (
        classify_app_auth(
            "SSE",
            {"url": "https://mcp.example.com/mcp", "auth": {"type": "mcp_oauth"}},
        )
        == "mcp_oauth"
    )


def test_mcp_oauth_shape_requires_a_remote_transport():
    # Same launch_config shape on a stdio transport must not be misrouted —
    # only sse/websocket/streamable_http describe a genuine remote MCP server.
    assert (
        classify_app_auth(
            "stdio",
            {"url": "https://mcp.example.com/mcp", "auth": {"type": "mcp_oauth"}},
        )
        == "unconnectable"
    )


def test_keyless_is_a_stdio_command_with_no_required_env():
    # e.g. the Chrome connector: a launchable local module with no secrets.
    assert classify_app_auth("stdio", {"command": "npx"}) == "keyless"
    assert (
        classify_app_auth(
            "stdio", {"command": "npx", "args": ["-y", "chrome-devtools-mcp@latest"]}
        )
        == "keyless"
    )
    # empty required_env is the same as absent, not a key-based app
    assert (
        classify_app_auth("stdio", {"command": "npx", "required_env": []}) == "keyless"
    )
    # transport check is case-insensitive, like the other classifications
    assert classify_app_auth("STDIO", {"command": "npx"}) == "keyless"
    # keyless is stdio-only: a command on a remote transport is mis-authored,
    # not connectable
    assert classify_app_auth("streamable_http", {"command": "npx"}) == "unconnectable"
    assert classify_app_auth("sse", {"command": "npx"}) == "unconnectable"
    assert classify_app_auth(None, {"command": "npx"}) == "unconnectable"


def test_keyless_excludes_env_mapping_shapes():
    # env_mapping means the launcher expects an injected token (the builtin
    # OAuth apps' pattern, e.g. env_mapping={"SLACK_ACCESS_TOKEN":
    # "access_token"}) -- not required_env, but not secret-free either. A
    # custom app authored with this shape and no required_env must not
    # classify keyless: that would offer a no-secrets Connect button for a
    # server that fails at tool-call time for a missing token.
    assert (
        classify_app_auth(
            "stdio",
            {"command": "uv", "env_mapping": {"TOKEN": "access_token"}},
        )
        == "unconnectable"
    )
    # Still keyless with an empty env_mapping -- same as absent.
    assert (
        classify_app_auth("stdio", {"command": "npx", "env_mapping": {}}) == "keyless"
    )


def test_inconsistent_entries_are_unconnectable_not_misrouted():
    # required_env but no command -> not launchable
    assert classify_app_auth("stdio", {"required_env": ["KEY"]}) == "unconnectable"
    assert classify_app_auth("stdio", None) == "unconnectable"
    assert classify_app_auth(None, None) == "unconnectable"
    # malformed launch_config (non-dict) must not raise AttributeError
    assert classify_app_auth("stdio", ["not", "a", "dict"]) == "unconnectable"
    assert classify_app_auth("stdio", "garbage") == "unconnectable"


def test_oauth_landing_rejects_non_oauth_app():
    # Symmetric guard: a key-based app must not be connectable via the OAuth
    # flow (it would get a token, never its required_env API key). The guard
    # runs before any DB access, so db=None is fine. It raises the dedicated
    # AppNotOAuthError so the batch loop can skip only this case.
    from xagent.web.api.auth import AppNotOAuthError, _ensure_user_mcp_server

    with pytest.raises(AppNotOAuthError, match="not an OAuth app"):
        _ensure_user_mcp_server(
            None,
            "1",
            {"id": "google-maps", "name": "Google Maps", "auth_type": "api_key"},
        )


@pytest.fixture()
def catalog_db():
    engine = create_engine("sqlite:///:memory:")
    PublicMCPApp.__table__.create(engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


def test_builtin_runtime_view_overlays_stale_execution_fields(catalog_db):
    catalog_db.add(
        PublicMCPApp(
            app_id="gmail",
            name="Stale Gmail Name",
            description="Environment-specific description",
            icon="https://example.com/gmail.png",
            transport="oauth",
            provider_name="wrong-provider",
            category="Environment Category",
            oauth_scopes=["wrong-scope"],
            is_visible_in_connector=False,
            launch_config={
                "command": "uv",
                "args": ["run", "python", "-m", "xagent.web.tools.mcp.gmail"],
                "env_mapping": {"WRONG_TOKEN": "access_token"},
            },
        )
    )
    catalog_db.commit()

    app = get_app_by_id(catalog_db, "gmail")

    assert app is not None
    assert app["name"] == "Gmail"
    assert app["transport"] == "oauth"
    assert app["provider"] == "google"
    assert app["oauth_scopes"] == ["https://www.googleapis.com/auth/gmail.modify"]
    assert app["launch_config"] == {
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.gmail"],
        "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"},
    }
    assert app["description"] == "Environment-specific description"
    assert app["icon"] == "https://example.com/gmail.png"
    assert app["category"] == "Environment Category"
    assert app["is_visible_in_connector"] is False


def test_custom_runtime_view_preserves_database_execution_fields(catalog_db):
    launch_config = {
        "command": "uv",
        "args": ["run", "custom_server.py"],
        "required_env": ["CUSTOM_TOKEN"],
    }
    catalog_db.add(
        PublicMCPApp(
            app_id="custom-gmail",
            name="Gmail",
            transport="stdio",
            launch_config=launch_config,
        )
    )
    catalog_db.commit()

    app = get_app_by_id(catalog_db, "custom-gmail")

    assert app is not None
    assert app["name"] == "Gmail"
    assert app["transport"] == "stdio"
    assert app["launch_config"] == launch_config
    assert app["auth_type"] == "api_key"


def test_mcp_server_catalog_lookup_prefers_stable_app_id(catalog_db):
    from types import SimpleNamespace

    catalog_db.add_all(
        [
            PublicMCPApp(
                app_id="gmail",
                name="Gmail",
                transport="oauth",
                provider_name="google",
                launch_config={"command": "uv"},
            ),
            PublicMCPApp(
                app_id="renamed-custom",
                name="Renamed Server",
                transport="stdio",
                launch_config={
                    "command": "uv",
                    "required_env": ["CUSTOM_TOKEN"],
                },
            ),
        ]
    )
    catalog_db.commit()
    server = SimpleNamespace(name="Renamed Server", auth={"app_id": "gmail"})

    app = mcp_apps.get_app_for_mcp_server(catalog_db, server)

    assert app is not None
    assert app["id"] == "gmail"
    assert app["name"] == "Gmail"


def test_mcp_server_catalog_lookup_falls_back_to_name_only_for_legacy_row(catalog_db):
    from types import SimpleNamespace

    catalog_db.add(
        PublicMCPApp(
            app_id="legacy-custom",
            name="Legacy Server",
            transport="stdio",
            launch_config={"command": "uv", "required_env": ["TOKEN"]},
        )
    )
    catalog_db.commit()

    app = mcp_apps.get_app_for_mcp_server(
        catalog_db, SimpleNamespace(name="Legacy Server", auth=None)
    )

    assert app is not None
    assert app["id"] == "legacy-custom"


@pytest.mark.parametrize(
    "invalid_app_id",
    ["missing-stable-identity", "", 123],
    ids=["unknown-string", "empty-string", "non-string"],
)
def test_invalid_stable_app_id_does_not_fall_back_to_name(
    catalog_db, invalid_app_id: object
):
    from types import SimpleNamespace

    catalog_db.add(
        PublicMCPApp(
            app_id="legacy-custom",
            name="Legacy Server",
            transport="stdio",
            launch_config={"command": "uv", "required_env": ["TOKEN"]},
        )
    )
    catalog_db.commit()

    app = mcp_apps.get_app_for_mcp_server(
        catalog_db,
        SimpleNamespace(name="Legacy Server", auth={"app_id": invalid_app_id}),
    )

    assert app is None
