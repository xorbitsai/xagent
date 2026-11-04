"""Regression coverage for connector-specific OAuth policy membership in
APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT.

That set is hand-maintained (src/xagent/web/mcp_apps.py) with no mechanical
link to the builtin registry it protects: an app whose oauth_scopes need
something the provider's own default_scopes don't grant, but that's missing
from the set, fails silently -- a bare provider-level OAuth grant is treated
as sufficient, the app reports "connected", and every scope-gated tool call
then fails. This pins every connector currently documented by the policy,
including whatsapp, Planner, SharePoint, PowerPoint, Excel, and Word, so a
future edit cannot silently drop one.

The Microsoft provider is held to a mechanical invariant (see
test_every_microsoft_app_beyond_provider_defaults_requires_app_scoped_grant):
every builtin Microsoft app whose scopes exceed the provider's identity-only
default must be listed, so a new connector cannot silently omit itself.

That invariant is deliberately not extended to the other providers. A fully
general "every builtin oauth app whose scopes exceed its provider's
default_scopes must be listed here" rule does not hold across the registry
today: several unlisted apps under other providers (e.g. most google apps,
zoom, linear, xero, employment-hero) also request scopes beyond their
provider's default_scopes, and instagram is excluded on purpose so its
existing bare "meta" grants keep working. Closing that gap is a separate,
cross-connector decision.
"""

from xagent.web.builtin_mcp_registry import (
    get_builtin_oauth_provider_rows,
    get_builtin_public_mcp_app_rows,
)
from xagent.web.mcp_apps import requires_app_scoped_oauth_grant

# Apps already known (from mcp_apps.py's own comment) to need this guard,
# pinned here so a regression in any of them -- not just the newest connectors
# -- is caught the same way.
_EXPECTED_APP_SCOPED_APPS = frozenset(
    {
        "excel",
        "facebook",
        "github",
        "myob",
        "meta-ads",
        "onedrive",
        "outlook",
        "planner",
        "powerpoint",
        "sharepoint",
        "teams",
        "whatsapp",
        "word",
    }
)


def test_expected_apps_require_app_scoped_oauth_grant():
    for app_id in _EXPECTED_APP_SCOPED_APPS:
        assert requires_app_scoped_oauth_grant(app_id), (
            f"{app_id!r} is expected to require an app-scoped OAuth grant "
            "(see the rationale in mcp_apps.py) but requires_app_scoped_"
            "oauth_grant now returns False for it"
        )


def _app_scopes_beyond_provider_defaults(app_id: str) -> set[str]:
    provider_default_scopes = {
        row["provider_name"]: set(row.get("default_scopes") or [])
        for row in get_builtin_oauth_provider_rows()
    }
    app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == app_id
    )
    default_scopes = provider_default_scopes.get(app["provider_name"], set())
    app_scopes = set(app.get("oauth_scopes") or [])
    return app_scopes - default_scopes


def test_whatsapp_scopes_actually_exceed_the_meta_providers_default_scopes():
    """The reason whatsapp needs to be in the set at all: confirms its
    premise (scopes beyond the bare provider grant) instead of just
    asserting the set's membership in isolation."""
    assert _app_scopes_beyond_provider_defaults("whatsapp"), (
        "whatsapp's oauth_scopes are now fully covered by the meta "
        "provider's default_scopes -- if that's genuinely true, it no "
        "longer needs to be in APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT and "
        "this test (and the set) should be updated together, not left to "
        "silently drift."
    )


def test_planner_scopes_exceed_the_microsoft_providers_default_scopes():
    assert _app_scopes_beyond_provider_defaults("planner") == {
        "Tasks.ReadWrite",
        "Team.ReadBasic.All",
        "offline_access",
    }


def test_sharepoint_scopes_actually_exceed_the_microsoft_providers_default_scopes():
    """Same premise check as whatsapp's, for sharepoint's Sites.ReadWrite.All
    against the microsoft provider's default_scopes (["User.Read"])."""
    assert _app_scopes_beyond_provider_defaults("sharepoint"), (
        "sharepoint's oauth_scopes are now fully covered by the microsoft "
        "provider's default_scopes -- if that's genuinely true, it no "
        "longer needs to be in APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT and "
        "this test (and the set) should be updated together, not left to "
        "silently drift."
    )


def test_word_scopes_actually_exceed_the_microsoft_providers_default_scopes():
    """Word requires Files.ReadWrite.All, which a bare User.Read grant lacks."""
    word_app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "word"
    )
    assert word_app["oauth_scopes"] == ["Files.ReadWrite.All", "offline_access"]
    assert word_app["launch_config"]["static_env"] == {
        "XAGENT_TOOL_MAX_OUTPUT_LENGTH": "XAGENT_TOOL_MAX_OUTPUT_LENGTH"
    }
    assert "top-level main-body paragraphs" in word_app["description"]
    assert "Tables, headers, footers" in word_app["description"]
    assert _app_scopes_beyond_provider_defaults("word"), (
        "word's oauth_scopes are now fully covered by the microsoft "
        "provider's default_scopes -- if that's genuinely true, it no "
        "longer needs to be in APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT and "
        "this test (and the set) should be updated together, not left to "
        "silently drift."
    )


def test_excel_scopes_actually_exceed_the_microsoft_providers_default_scopes():
    provider_default_scopes = {
        row["provider_name"]: set(row.get("default_scopes") or [])
        for row in get_builtin_oauth_provider_rows()
    }
    excel = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "excel"
    )

    assert set(excel["oauth_scopes"]) - provider_default_scopes["microsoft"] == {
        "Files.ReadWrite",
        "offline_access",
    }


def test_every_microsoft_app_beyond_provider_defaults_requires_app_scoped_grant():
    """A bare Microsoft login only ever requests the provider's default scopes,
    so no builtin Microsoft app that needs more can be satisfied by it."""
    microsoft_app_ids = [
        row["app_id"]
        for row in get_builtin_public_mcp_app_rows()
        if row["provider_name"] == "microsoft"
    ]
    # Guard against the filter silently matching nothing.
    assert {"onedrive", "outlook", "teams"} <= set(microsoft_app_ids)

    unguarded = sorted(
        app_id
        for app_id in microsoft_app_ids
        if _app_scopes_beyond_provider_defaults(app_id)
        and not requires_app_scoped_oauth_grant(app_id)
    )

    assert not unguarded, (
        f"Microsoft apps {unguarded} request scopes beyond the microsoft "
        "provider's default_scopes but are not in "
        "APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT, so a bare microsoft grant "
        "would report them connected while every Graph call fails for "
        "insufficient scope. Add them to the set in src/xagent/web/mcp_apps.py."
    )
