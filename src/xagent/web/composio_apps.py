"""The Composio per-app switch lookup, free of the web stack.

``composio_auth`` carries the whole Composio hosted-auth flow and imports
``cryptography``, ``fastapi`` and the HTTP wrapper. The built-in connector
registry and the connector smoke job (which runs with only ``httpx`` and
``sqlalchemy``) need nothing but the answer to "does this app's new connects
go through Composio?", so that answer lives here, importing the standard
library only. ``composio_auth`` re-exports every name below, so existing
importers (and tests that read or patch these names there, or here) keep
working. Keep this module dependency-free:
``tests/web/test_composio_apps_light_import.py`` fails if it is not.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple
from urllib.parse import urlsplit

# Logged under ``composio_auth``'s name, as it was before the move.
logger = logging.getLogger("xagent.web.composio_auth")

FACEBOOK_AUTH_BACKEND_ENV = "XAGENT_FACEBOOK_AUTH_BACKEND"
COMPOSIO_FACEBOOK_AUTH_CONFIG_ID_ENV = "COMPOSIO_FACEBOOK_AUTH_CONFIG_ID"

GRAPH_ME_URL = "https://graph.facebook.com/v25.0/me"
FACEBOOK_APP_ID = "facebook"
FACEBOOK_TOOLKIT_SLUG = "facebook"
INSTAGRAM_AUTH_BACKEND_ENV = "XAGENT_INSTAGRAM_AUTH_BACKEND"
COMPOSIO_INSTAGRAM_AUTH_CONFIG_ID_ENV = "COMPOSIO_INSTAGRAM_AUTH_CONFIG_ID"
INSTAGRAM_GRAPH_ME_URL = "https://graph.instagram.com/v25.0/me"
INSTAGRAM_APP_ID = "instagram"
INSTAGRAM_TOOLKIT_SLUG = "instagram"
META_PROVIDER = "meta"
GOOGLE_PROVIDER = "google"
GMAIL_APP_ID = "gmail"
GMAIL_AUTH_BACKEND_ENV = "XAGENT_GMAIL_AUTH_BACKEND"
COMPOSIO_GMAIL_AUTH_CONFIG_ID_ENV = "COMPOSIO_GMAIL_AUTH_CONFIG_ID"
GOOGLE_DRIVE_APP_ID = "google-drive"
GOOGLE_DRIVE_AUTH_BACKEND_ENV = "XAGENT_GOOGLE_DRIVE_AUTH_BACKEND"
COMPOSIO_GOOGLE_DRIVE_AUTH_CONFIG_ID_ENV = "COMPOSIO_GOOGLE_DRIVE_AUTH_CONFIG_ID"
# Google's OIDC userinfo endpoint (its discovery document's
# ``userinfo_endpoint``); ``sub`` never changes and is never reused.
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
SALESFORCE_APP_ID = "salesforce"
SALESFORCE_AUTH_BACKEND_ENV = "XAGENT_SALESFORCE_AUTH_BACKEND"
COMPOSIO_SALESFORCE_AUTH_CONFIG_ID_ENV = "COMPOSIO_SALESFORCE_AUTH_CONFIG_ID"
SALESFORCE_TOOLKIT_SLUG = "salesforce"
# On the org's own host: login.salesforce.com's userinfo does not serve
# sandbox orgs.
SALESFORCE_USERINFO_PATH = "/services/oauth2/userinfo"


class ComposioApp(NamedTuple):
    """What differs between the apps that can connect through Composio."""

    app_id: str
    # The OAuth provider the app's connector is started under; its state is
    # minted for this provider and is refused under any other.
    provider: str
    label: str
    backend_env: str
    auth_config_env: str
    toolkit_slug: str
    # GET this through the proxy with ``identity_params`` as its query;
    # ``identity_field`` of the reply, which must fully match
    # ``identity_id_pattern``, is stored as the grant's provider_user_id.
    # Either a fixed URL, or, for an app whose identity lives on the
    # account's own host, a function of the connected account that
    # ``verify_account`` read, returning None when the account names no
    # usable host.
    identity_url: str | Callable[[Mapping[str, Any]], str | None]
    identity_field: str
    identity_params: tuple[tuple[str, str], ...]
    identity_id_pattern: re.Pattern[str]
    # Shown by the connect UI while the app goes through Composio:
    # ``before_connect`` before the sign-in window opens, ``not_connected``
    # when it closes without a connection. None for an app that needs none.
    connect_guidance: Mapping[str, str] | None = None
    # For an app whose identity URL is a function of the account: what to
    # change on Composio's page when the account names no usable host. Ends
    # the sentence "Close this window, select Connect again, and ...".
    identity_host_help: str | None = None
    # Whether ordinary (web) connects also go through Composio; an actor
    # start follows the app's switch alone.
    web_start: bool = False


# Graph ids, Facebook user ids and Instagram professional account ids alike.
_GRAPH_ID_RE = re.compile(r"[0-9]{1,32}")
# Google subjects are numeric in practice; the OIDC limit is 255 characters.
_GOOGLE_SUBJECT_RE = re.compile(r"[0-9]{1,255}")
# A Salesforce User id in its 18-character form (key prefix 005), as OIDC
# userinfo's ``user_id`` gives it.
_SALESFORCE_USER_ID_RE = re.compile(r"005[0-9A-Za-z]{15}")
# DNS labels of letters, digits and inner hyphens under salesforce.com: the
# Salesforce runtime's pattern for the same value (#3708), so every host
# accepted here is one the runtime calls.
_SALESFORCE_HOST_RE = re.compile(
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+salesforce\.com"
)
# Printable ASCII without the space: what a base_url may consist of, checked
# before urlsplit, which would silently drop tabs and newlines.
_VISIBLE_ASCII_RE = re.compile(r"[!-~]+")


def _salesforce_userinfo_url(account: Mapping[str, Any]) -> str | None:
    """The userinfo URL on the org host of the account's ``data.base_url``.

    Composio finds the org at sign-in and keeps its host plus
    ``/services/data/vNN.0`` there (#3646 live test). The value comes from
    what the member typed on Composio's page, so it is never logged, and is
    refused unless it is made only of printable ASCII without spaces (checked
    before parsing, since ``urlsplit`` drops tabs and newlines), is https,
    has no credentials and no port (an empty one included), and its host,
    lowercased by ``urlsplit`` and with at most one trailing dot removed, is
    a subdomain of salesforce.com whose labels are letters, digits and inner
    hyphens, at most 63 characters each. Only the host is kept. This is at
    least as strict as the Salesforce runtime's check on the same value
    (#3708), so every host accepted here is accepted there.
    """
    data = account.get("data")
    base_url = data.get("base_url") if isinstance(data, dict) else None
    if not isinstance(base_url, str) or not _VISIBLE_ASCII_RE.fullmatch(base_url):
        return None
    try:
        parts = urlsplit(base_url)
        hostname = (parts.hostname or "").removesuffix(".")
        port = parts.port
    except ValueError:
        # Unparsable (an unclosed "[" bracket) or an invalid port.
        return None
    if (
        parts.scheme != "https"
        or parts.username is not None
        or parts.password is not None
        or port is not None
        or parts.netloc.endswith(":")
        or not _SALESFORCE_HOST_RE.fullmatch(hostname)
    ):
        return None
    return f"https://{hostname}{SALESFORCE_USERINFO_PATH}"


_SALESFORCE_INSTALL = (
    "A Salesforce admin has to install Composio's app in your org first: in "
    "Setup, open Connected Apps OAuth Usage, click Install next to Composio's "
    'app, then set Permitted Users to "All users may self-authorize". If '
    "Composio's app isn't listed there yet, connect once as an admin first, "
    "then install it."
)
_SALESFORCE_LOGIN_POLICY = (
    "change the Salesforce subdomain on Composio's page to your My Domain "
    "prefix (for example acme.my, or acme.develop.my for a Developer Edition "
    'org), or choose "Use Custom Domain" on the Salesforce login page.'
)
# A sandbox org signs in through test.salesforce.com, not the default
# subdomain "login". Not tested live (no sandbox at hand, #3646); this
# follows Salesforce's own login rules.
_SALESFORCE_SANDBOX = (
    "For a sandbox org, change the subdomain on Composio's page to test, or "
    "to your sandbox's My Domain prefix (for example acme--uat.sandbox.my)."
)
# Shown when the connected account names no usable org host: connecting
# again the same way can't fix that.
_SALESFORCE_IDENTITY_HOST_HELP = (
    "on Composio's page change the Salesforce subdomain to your My Domain "
    "prefix (for example acme.my, or acme.develop.my for a Developer Edition "
    f"org). {_SALESFORCE_SANDBOX}"
)
# Salesforce stops on its own page when the app is not installed, the
# subdomain is mistyped, or the org refuses logins from login.salesforce.com;
# nothing comes back to Composio or to us, so the only help is this text.
SALESFORCE_CONNECT_GUIDANCE: Mapping[str, str] = {
    "before_connect": (
        f"{_SALESFORCE_INSTALL} Admins can connect without installing it, but "
        "nobody else in the org can until it is installed, so admins should "
        "install it too. If your org doesn't allow logins from "
        f"login.salesforce.com, {_SALESFORCE_LOGIN_POLICY} {_SALESFORCE_SANDBOX}"
    ),
    "not_connected": (
        "Salesforce wasn't connected. If Salesforce showed "
        'OAUTH_APPROVAL_ERROR_GENERIC or "app must be installed into org": '
        f"{_SALESFORCE_INSTALL} If the page couldn't be reached, the subdomain "
        "on Composio's page may be mistyped. If Salesforce said \"Please check "
        'your username and password" although they are right, your org may '
        f"not allow logins from login.salesforce.com: {_SALESFORCE_LOGIN_POLICY} "
        f"{_SALESFORCE_SANDBOX}"
    ),
}


COMPOSIO_APPS: dict[str, ComposioApp] = {
    FACEBOOK_APP_ID: ComposioApp(
        app_id=FACEBOOK_APP_ID,
        provider=META_PROVIDER,
        label="Facebook",
        backend_env=FACEBOOK_AUTH_BACKEND_ENV,
        auth_config_env=COMPOSIO_FACEBOOK_AUTH_CONFIG_ID_ENV,
        toolkit_slug=FACEBOOK_TOOLKIT_SLUG,
        identity_url=GRAPH_ME_URL,
        identity_field="id",
        identity_params=(("fields", "id"),),
        identity_id_pattern=_GRAPH_ID_RE,
    ),
    INSTAGRAM_APP_ID: ComposioApp(
        app_id=INSTAGRAM_APP_ID,
        provider=META_PROVIDER,
        label="Instagram",
        backend_env=INSTAGRAM_AUTH_BACKEND_ENV,
        auth_config_env=COMPOSIO_INSTAGRAM_AUTH_CONFIG_ID_ENV,
        toolkit_slug=INSTAGRAM_TOOLKIT_SLUG,
        identity_url=INSTAGRAM_GRAPH_ME_URL,
        # The professional account id the Instagram tools address.
        identity_field="user_id",
        identity_params=(("fields", "user_id"),),
        identity_id_pattern=_GRAPH_ID_RE,
    ),
    # Composio's Google app holds Google's restricted-scope approval, so these
    # connect with full Gmail and Drive access while our own review gate
    # (XAGENT_GOOGLE_RESTRICTED_SCOPES_ENABLED) stays closed (#3649).
    GMAIL_APP_ID: ComposioApp(
        app_id=GMAIL_APP_ID,
        provider=GOOGLE_PROVIDER,
        label="Gmail",
        backend_env=GMAIL_AUTH_BACKEND_ENV,
        auth_config_env=COMPOSIO_GMAIL_AUTH_CONFIG_ID_ENV,
        toolkit_slug="gmail",
        identity_url=GOOGLE_USERINFO_URL,
        identity_field="sub",
        identity_params=(),
        identity_id_pattern=_GOOGLE_SUBJECT_RE,
    ),
    GOOGLE_DRIVE_APP_ID: ComposioApp(
        app_id=GOOGLE_DRIVE_APP_ID,
        provider=GOOGLE_PROVIDER,
        label="Google Drive",
        backend_env=GOOGLE_DRIVE_AUTH_BACKEND_ENV,
        auth_config_env=COMPOSIO_GOOGLE_DRIVE_AUTH_CONFIG_ID_ENV,
        toolkit_slug="googledrive",
        identity_url=GOOGLE_USERINFO_URL,
        identity_field="sub",
        identity_params=(),
        identity_id_pattern=_GOOGLE_SUBJECT_RE,
    ),
    # Composio's connected app, so an org admin has to install it before the
    # org's other users can connect (#3646).
    SALESFORCE_APP_ID: ComposioApp(
        app_id=SALESFORCE_APP_ID,
        provider=SALESFORCE_APP_ID,
        label="Salesforce",
        backend_env=SALESFORCE_AUTH_BACKEND_ENV,
        auth_config_env=COMPOSIO_SALESFORCE_AUTH_CONFIG_ID_ENV,
        toolkit_slug=SALESFORCE_TOOLKIT_SLUG,
        identity_url=_salesforce_userinfo_url,
        identity_field="user_id",
        identity_params=(),
        identity_id_pattern=_SALESFORCE_USER_ID_RE,
        connect_guidance=SALESFORCE_CONNECT_GUIDANCE,
        identity_host_help=_SALESFORCE_IDENTITY_HOST_HELP,
        web_start=True,
    ),
}

AUTH_BACKEND = "composio"


def normalize_oauth_grant_key(value: object) -> str | None:
    """Case/whitespace-insensitive key, matching ``normalize_catalog_key``.

    The one copy: ``mcp_apps`` (which ``mcp.py`` imports, so it cannot import
    back from there without a cycle) and ``composio_auth`` both take it from
    this dependency-free module.
    """
    if value is None:
        return None
    normalized = "-".join(str(value).strip().lower().split())
    return normalized or None


def _env(name: str) -> str | None:
    """Read an environment variable, treating blank as unset."""
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip() or None


def composio_app(provider: str, app_id: str | None) -> ComposioApp | None:
    """The app whose new connects go through Composio, if this one does.

    Each app follows only its own switch, so turning Facebook over never
    moves Instagram, and Meta Ads and WhatsApp have no switch at all. An app
    is only ever started under its own provider.
    """
    if not app_id:
        return None
    app = COMPOSIO_APPS.get(normalize_oauth_grant_key(app_id) or "")
    if app is None or app.provider != provider.lower() or not _uses_composio(app):
        return None
    return app


def composio_web_app(provider: str, app_id: str | None) -> ComposioApp | None:
    """The app whose ordinary web connects go through Composio, if this one does."""
    app = composio_app(provider, app_id)
    if app is None or not app.web_start:
        return None
    return app


def _uses_composio(app: ComposioApp) -> bool:
    backend = (_env(app.backend_env) or "native").lower()
    if (
        backend not in ("native", AUTH_BACKEND)
        and (
            app.backend_env,
            backend,
        )
        not in _WARNED_BACKENDS
    ):
        _WARNED_BACKENDS.add((app.backend_env, backend))
        logger.warning(
            "%s=%r is neither 'native' nor %r; %s connects stay native",
            app.backend_env,
            backend,
            AUTH_BACKEND,
            app.label,
        )
    return backend == AUTH_BACKEND


_WARNED_BACKENDS: set[tuple[str, str]] = set()
