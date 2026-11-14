"""An ordinary (web) connect completes through Composio-hosted auth.

These tests mint the state by hand, matching what a Composio web start emits:
the Composio auth-backend claim, no ``resource_owner_key``, the ordinary
browser proof cookie named by the state's ``browser_nonce_hash`` claim, and a
single-use flow row the verify or denied trip spends.

Composio is faked by the Facebook connect tests; no request leaves the
process.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from xagent.web import composio_auth
from xagent.web.api import auth as auth_api
from xagent.web.api.mcp import list_mcp_apps
from xagent.web.models.actor_oauth_flow import ActorOAuthFlowState
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.services.user_oauth import (
    delete_scoped_user_oauth_accounts,
    normalize_user_oauth_resource_owner_key,
)

from . import test_composio_facebook_connect as facebook_connect
from .test_composio_facebook_connect import (
    CALLBACK_PATH,
    CALLBACK_URL,
    FACEBOOK_USER_ID,
)
from .test_composio_facebook_connect import NOW as RECONCILE_NOW
from .test_composio_facebook_connect import (
    OTHER_OWNER,
    OWNER,
    Browser,
    FakeComposio,
    _denied_url,
    _flow_cookies,
    _grants,
    _listed,
    _store_composio_grant,
    _verify_url,
)

composio_env = facebook_connect.composio_env
db_session = facebook_connect.db_session
composio = facebook_connect.composio

MISSING_ACCOUNT = facebook_connect.MISSING_ACCOUNT


@pytest.fixture
def ordinary(db_session, composio_env, composio):
    """A user with no actor link at all: an ordinary connect needs none."""
    db, user = db_session
    db.query(UserMCPServer).delete()
    db.commit()
    return db, user, composio


DEDICATED_URL = "https://app.example.com/api/auth/composio/callback"
DEDICATED_PATH = "/api/auth/composio/callback"


class RoutedBrowser(Browser):
    """A browser whose requests reach whichever callback route they name."""

    def get(self, db, url):
        parts = urlsplit(url)
        if parts.path == CALLBACK_PATH:
            return super().get(db, url)
        assert parts.path == DEDICATED_PATH, parts.path
        query = {name: values[0] for name, values in parse_qs(parts.query).items()}
        request = SimpleNamespace(query_params=query, cookies=self.for_path(parts.path))
        response = auth_api.composio_oauth_callback(request, db)
        self.receive(response)
        return response


@pytest.fixture
def dedicated_ordinary(ordinary, monkeypatch):
    monkeypatch.setenv("XAGENT_COMPOSIO_CALLBACK_URL", DEDICATED_URL)
    return ordinary


def _mint_ordinary_state(
    db,
    user,
    browser: Browser,
    *,
    expired: bool = False,
    with_browser_nonce_claim: bool = True,
    app_id: str = "facebook",
    provider: str = "meta",
    cookie_path: str = CALLBACK_PATH,
    actor_flow_nonce: str | None = None,
) -> str:
    """The state the later Composio start will mint for a web connect."""
    browser_nonce = secrets.token_urlsafe(32)
    nonce = hashlib.sha256(browser_nonce.encode()).hexdigest()
    data = {
        "type": "oauth_state",
        "user_id": int(user.id),
        "provider": provider,
        "app_id": app_id,
        "redirect": None,
        "session_epoch": user.session_epoch,
        composio_auth.AUTH_BACKEND_CLAIM: composio_auth.AUTH_BACKEND,
        "actor_flow_nonce": actor_flow_nonce or nonce,
    }
    if with_browser_nonce_claim:
        data["browser_nonce_hash"] = nonce
    state = auth_api.create_access_token(data=data, expires_delta=timedelta(minutes=10))
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
    if expired:
        expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    db.add(ActorOAuthFlowState(nonce=nonce, expires_at=expires_at))
    db.commit()
    browser.cookies[auth_api._ordinary_oauth_cookie_name(nonce)] = (
        browser_nonce,
        cookie_path,
    )
    return state


def _begin(db, browser: Browser, state: str):
    return browser.get(db, composio_auth.begin_location(CALLBACK_URL, state))


def _connect_ordinary(db, user, composio: FakeComposio) -> tuple[str, Browser]:
    browser = Browser()
    state = _mint_ordinary_state(db, user, browser)
    assert _begin(db, browser, state).status_code == 302
    response = browser.get(db, _verify_url(composio))
    assert response.status_code == 200, response.body
    return composio.created[-1], browser


def _ordinary_grants(db):
    db.expire_all()
    return [grant for grant in _grants(db) if grant.resource_owner_key is None]


def _ordinary_composio_user(user) -> str:
    config = composio_auth.load_config("facebook")
    assert config is not None
    return composio_auth.composio_user_id(config, user_id=int(user.id), owner_key="")


# --- The flow -------------------------------------------------------------------


def test_an_ordinary_flow_begins_verifies_and_stores_an_ordinary_grant(ordinary):
    db, user, composio = ordinary
    _store_composio_grant(db, user, account_id="ca_actor", owner=OWNER)
    browser = Browser()
    state = _mint_ordinary_state(db, user, browser)

    begin = _begin(db, browser, state)

    assert begin.status_code == 302, begin.body
    link = [body for _, path, body in composio.calls if path.endswith("/link")]
    ordinary_user = _ordinary_composio_user(user)
    assert link == [
        {
            "auth_config_id": facebook_connect.AUTH_CONFIG_ID,
            "user_id": ordinary_user,
            "callback_url": CALLBACK_URL,
        }
    ]

    response = browser.get(db, _verify_url(composio))

    assert response.status_code == 200, response.body
    account = composio.created[-1]
    (stored,) = _ordinary_grants(db)
    assert stored.provider == "facebook"
    assert stored.token_type == "composio"
    assert stored.access_token == account
    assert stored.provider_user_id == FACEBOOK_USER_ID
    assert stored.instance_url == ordinary_user
    assert stored.refresh_token is None
    # The user's actor grant is untouched.
    actor = [grant for grant in _grants(db) if grant.resource_owner_key == OWNER]
    assert [grant.access_token for grant in actor] == ["ca_actor"]
    assert db.query(ActorOAuthFlowState).count() == 0
    assert _flow_cookies(browser) == []
    assert composio.paths("DELETE") == []


def test_a_second_ordinary_connect_replaces_the_first_and_deletes_its_account(
    ordinary,
):
    db, user, composio = ordinary
    _store_composio_grant(db, user, account_id="ca_actor", owner=OWNER)
    first, _ = _connect_ordinary(db, user, composio)

    second, _ = _connect_ordinary(db, user, composio)

    (stored,) = _ordinary_grants(db)
    assert second != first
    assert stored.access_token == second
    assert composio.paths("DELETE") == [f"/connected_accounts/{first}"]
    actor = [grant for grant in _grants(db) if grant.resource_owner_key == OWNER]
    assert [grant.access_token for grant in actor] == ["ca_actor"]


def test_the_ordinary_composio_user_cannot_collide_with_an_actor_of_the_same_user(
    composio_env,
):
    config = composio_auth.load_config("facebook")
    assert config is not None
    ordinary = composio_auth.composio_user_id(config, user_id=7, owner_key="")
    actors = {
        composio_auth.composio_user_id(config, user_id=7, owner_key=owner)
        for owner in (OWNER, OTHER_OWNER)
    }
    assert ordinary not in actors
    assert ordinary == composio_auth.composio_user_id(config, user_id=7, owner_key="")
    # An actor owner key can never be empty, so the two forms cannot coincide.
    for blank in ("", "   "):
        with pytest.raises(ValueError):
            normalize_user_oauth_resource_owner_key(blank)


# --- The connector association ---------------------------------------------------


def _listed_app(db, user, app_id: str = "facebook") -> dict:
    apps = list_mcp_apps(location="remote", current_user=user, db=db)
    return {app["id"]: app for app in apps}[app_id]


def test_an_ordinary_connect_associates_the_connector(ordinary):
    """A first-time member's listing says connected only with a grant and the
    user's server association, so the ordinary store writes both."""
    db, user, composio = ordinary
    db.query(UserMCPServer).delete()
    db.query(MCPServer).delete()
    db.commit()
    assert _listed_app(db, user)["is_connected"] is False

    _connect_ordinary(db, user, composio)

    (association,) = db.query(UserMCPServer).all()
    assert int(association.user_id) == int(user.id)
    assert association.is_owner
    assert association.is_active
    assert _listed_app(db, user)["is_connected"] is True


def test_an_actor_connect_associates_no_connector(
    db_session, composio_env, composio, monkeypatch
):
    """An actor flow's link must already exist, so assert the store never asks
    for one: an ensure here would be a no-op and hide the regression."""
    db, user = db_session
    ensured = []
    real = auth_api._ensure_user_mcp_server

    def spy(*args, **kwargs):
        ensured.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(auth_api, "_ensure_user_mcp_server", spy)
    browser, _ = facebook_connect._connect_until_verification(db, user, OWNER)

    assert browser.get(db, _verify_url(composio)).status_code == 200

    assert [grant.resource_owner_key for grant in _grants(db)] == [OWNER]
    assert ensured == []
    assert db.query(UserMCPServer).count() == 1


def test_a_hidden_app_stores_nothing_and_its_account_is_deleted(ordinary):
    db, user, composio = ordinary
    db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "facebook").update(
        {PublicMCPApp.is_visible_in_connector: False}
    )
    db.commit()
    browser = Browser()
    state = _mint_ordinary_state(db, user, browser)
    assert _begin(db, browser, state).status_code == 302
    account = composio.created[-1]

    response = browser.get(db, _verify_url(composio, account))

    assert response.status_code == 404
    assert "not currently available" in response.body.decode()
    assert _ordinary_grants(db) == []
    assert db.query(UserMCPServer).count() == 0
    assert db.query(ActorOAuthFlowState).count() == 0
    assert composio.paths("DELETE") == [f"/connected_accounts/{account}"]


def test_a_failed_association_stores_nothing_and_its_account_is_deleted(
    ordinary, monkeypatch
):
    db, user, composio = ordinary
    db.query(UserMCPServer).delete()
    db.commit()

    def conflict(*_args, **_kwargs):
        raise ValueError("catalog metadata conflict")

    monkeypatch.setattr(auth_api, "_ensure_user_mcp_server", conflict)
    browser = Browser()
    state = _mint_ordinary_state(db, user, browser)
    assert _begin(db, browser, state).status_code == 302
    account = composio.created[-1]

    response = browser.get(db, _verify_url(composio, account))

    assert response.status_code == 500
    assert _ordinary_grants(db) == []
    assert db.query(UserMCPServer).count() == 0
    assert composio.paths("DELETE") == [f"/connected_accounts/{account}"]


# --- Proof cookie and one-time use ----------------------------------------------


def test_replaying_a_spent_ordinary_flow_creates_no_link(ordinary):
    db, user, composio = ordinary
    browser = Browser()
    state = _mint_ordinary_state(db, user, browser)
    assert _begin(db, browser, state).status_code == 302
    spent_cookies = dict(browser.cookies)
    assert browser.get(db, _verify_url(composio)).status_code == 200

    browser.cookies = dict(spent_cookies)
    replay_verify = browser.get(db, _verify_url(composio))
    browser.cookies = dict(spent_cookies)
    replay_begin = _begin(db, browser, state)

    assert replay_verify.status_code == 400
    assert replay_begin.status_code == 400
    assert "already been used" in replay_begin.body.decode()
    assert composio.paths().count("/connected_accounts/complete_auth") == 1
    assert composio.paths().count("/connected_accounts/link") == 1
    assert len(_ordinary_grants(db)) == 1


def test_an_ordinary_begin_without_its_browser_proof_creates_no_link(ordinary):
    db, user, composio = ordinary
    browser = Browser()
    state = _mint_ordinary_state(db, user, browser)
    browser.cookies = {}

    response = _begin(db, browser, state)

    assert response.status_code == 400
    assert "same browser" in response.body.decode()
    assert composio.calls == []
    assert db.query(ActorOAuthFlowState).count() == 1


def test_an_ordinary_begin_with_a_foreign_browser_proof_creates_no_link(ordinary):
    db, user, composio = ordinary
    browser = Browser()
    state = _mint_ordinary_state(db, user, browser)
    (name,) = browser.cookies
    browser.cookies[name] = ("someone-elses-proof", CALLBACK_PATH)

    response = _begin(db, browser, state)

    assert response.status_code == 400
    assert composio.calls == []
    assert db.query(ActorOAuthFlowState).count() == 1


def test_an_ordinary_state_without_its_browser_nonce_claim_creates_no_link(ordinary):
    db, user, composio = ordinary
    browser = Browser()
    state = _mint_ordinary_state(db, user, browser, with_browser_nonce_claim=False)

    response = _begin(db, browser, state)

    assert response.status_code == 400
    assert composio.calls == []


def test_an_ordinary_state_whose_two_nonce_claims_disagree_is_refused(ordinary):
    """The proof cookie matches ``browser_nonce_hash``, so only the inequality
    with ``actor_flow_nonce`` can refuse this."""
    db, user, composio = ordinary
    browser = Browser()
    state = _mint_ordinary_state(
        db,
        user,
        browser,
        actor_flow_nonce=hashlib.sha256(secrets.token_urlsafe(32).encode()).hexdigest(),
    )

    response = _begin(db, browser, state)

    assert response.status_code == 400
    assert "could not be verified" in response.body.decode()
    assert composio.calls == []
    assert db.query(ActorOAuthFlowState).count() == 1


def test_an_expired_ordinary_flow_creates_no_link(ordinary):
    db, user, composio = ordinary
    browser = Browser()
    state = _mint_ordinary_state(db, user, browser, expired=True)

    response = _begin(db, browser, state)

    assert response.status_code == 400
    assert "already been used" in response.body.decode()
    assert composio.calls == []
    assert _ordinary_grants(db) == []


# --- Session fence ----------------------------------------------------------------


def test_a_session_epoch_rotated_during_the_flow_stores_nothing(ordinary, caplog):
    """A password change/reset during the sign-in wins: no row is written, and
    the flow is already spent so a retry cannot quietly store it."""
    db, user, composio = ordinary
    browser = Browser()
    state = _mint_ordinary_state(db, user, browser)
    assert _begin(db, browser, state).status_code == 302
    account = composio.created[-1]
    db.query(User).filter(User.id == int(user.id)).update(
        {User.session_epoch: "rotated"}
    )
    db.commit()
    caplog.set_level(logging.WARNING)

    response = browser.get(db, _verify_url(composio, account))

    assert response.status_code == 400
    assert "reason=session_epoch_changed_after_exchange" in caplog.text
    assert "actor_flow=false" in caplog.text
    assert _ordinary_grants(db) == []
    assert db.query(ActorOAuthFlowState).count() == 0
    retry = browser.get(db, _verify_url(composio, account))
    assert retry.status_code == 400
    assert _ordinary_grants(db) == []
    # The account this request activated is deleted, not left unbound.
    assert composio.paths("DELETE") == [f"/connected_accounts/{account}"]


# --- One shape is never the other ------------------------------------------------


def test_an_ordinary_state_is_not_accepted_on_the_actor_path(ordinary):
    """The actor proof cookie (whose digest matches the nonce too) may not
    stand in for the ordinary browser proof."""
    db, user, composio = ordinary
    browser = Browser()
    state = _mint_ordinary_state(db, user, browser)
    payload = auth_api.verify_token(state)
    assert payload is not None
    (proof_name,) = browser.cookies
    proof_value = browser.cookies.pop(proof_name)[0]
    browser.cookies[auth_api._actor_oauth_cookie_name(payload["actor_flow_nonce"])] = (
        proof_value,
        CALLBACK_PATH,
    )

    response = _begin(db, browser, state)

    assert response.status_code == 400
    assert composio.calls == []
    assert _ordinary_grants(db) == []


def test_an_actor_state_is_not_accepted_on_the_ordinary_path(
    db_session, composio_env, composio
):
    db, user = db_session
    browser = Browser()
    start = facebook_connect._start(db, user, browser, OWNER)
    actor_cookie = next(
        name
        for name in browser.cookies
        if name.startswith(auth_api._ACTOR_OAUTH_COOKIE_PREFIX)
    )
    browser.cookies.pop(actor_cookie)
    browser_nonce = secrets.token_urlsafe(32)
    browser.cookies[
        auth_api._ordinary_oauth_cookie_name(
            hashlib.sha256(browser_nonce.encode()).hexdigest()
        )
    ] = (browser_nonce, CALLBACK_PATH)

    response = browser.get(db, start.headers["location"])

    assert response.status_code == 400
    assert composio.calls == []
    assert _grants(db) == []


def test_a_browser_holding_both_shapes_completes_neither(
    db_session, composio_env, composio
):
    db, user = db_session
    browser = Browser()
    start = facebook_connect._start(db, user, browser, OWNER)
    assert browser.get(db, start.headers["location"]).status_code == 302
    (actor_flow_cookie,) = _flow_cookies(browser)
    actor_flow = browser.cookies[actor_flow_cookie]
    ordinary_state = _mint_ordinary_state(db, user, browser)
    assert _begin(db, browser, ordinary_state).status_code == 302
    ordinary_account = composio.created[-1]
    # Both flow cookies open at once: one browser, two shapes.
    browser.cookies[actor_flow_cookie] = actor_flow

    response = browser.get(db, _verify_url(composio, ordinary_account))

    assert response.status_code == 400
    assert composio.paths().count("/connected_accounts/complete_auth") == 0
    assert _grants(db) == []
    assert db.query(ActorOAuthFlowState).count() == 2


def test_an_ordinary_begin_drops_the_actor_flow_cookie_it_finds(
    db_session, composio_env, composio
):
    db, user = db_session
    browser = Browser()
    start = facebook_connect._start(db, user, browser, OWNER)
    assert browser.get(db, start.headers["location"]).status_code == 302
    (actor_flow_cookie,) = _flow_cookies(browser)

    begin = _begin(db, browser, _mint_ordinary_state(db, user, browser))

    assert begin.status_code == 302
    assert actor_flow_cookie not in browser.cookies
    assert len(_flow_cookies(browser)) == 1


def test_an_actor_begin_drops_the_ordinary_flow_cookie_it_finds(
    db_session, composio_env, composio
):
    db, user = db_session
    browser = Browser()
    assert (
        _begin(db, browser, _mint_ordinary_state(db, user, browser)).status_code == 302
    )
    (ordinary_flow_cookie,) = _flow_cookies(browser)

    start = facebook_connect._start(db, user, browser, OWNER)
    begin = browser.get(db, start.headers["location"])

    assert begin.status_code == 302
    assert ordinary_flow_cookie not in browser.cookies
    assert len(_flow_cookies(browser)) == 1


# --- The dedicated callback -----------------------------------------------------


def test_an_ordinary_flow_completes_on_the_dedicated_callback(dedicated_ordinary):
    """The proof cookie sits on the path an ordinary start sets: the shared
    callback, here the dedicated route rather than the Meta one."""
    db, user, composio = dedicated_ordinary
    browser = RoutedBrowser()
    state = _mint_ordinary_state(db, user, browser, cookie_path=DEDICATED_PATH)

    begin = browser.get(db, f"{DEDICATED_URL}?state={state}")

    assert begin.status_code == 302, begin.body
    (link,) = [body for _, path, body in composio.calls if path.endswith("/link")]
    assert link["callback_url"] == DEDICATED_URL
    assert link["user_id"] == _ordinary_composio_user(user)

    verify = browser.get(
        db, f"{DEDICATED_URL}?session_uri={_verify_url(composio).split('=', 1)[1]}"
    )

    assert verify.status_code == 200, verify.body
    (stored,) = _ordinary_grants(db)
    assert stored.resource_owner_key is None
    assert stored.access_token == composio.created[-1]
    assert _flow_cookies(browser) == []


# --- Denial logs ----------------------------------------------------------------


def test_a_denied_ordinary_flow_is_logged_as_an_ordinary_flow(ordinary, caplog):
    db, user, composio = ordinary
    browser = Browser()
    state = _mint_ordinary_state(db, user, browser)
    assert _begin(db, browser, state).status_code == 302
    caplog.set_level(logging.WARNING)

    response = browser.get(db, _denied_url(composio.created[-1]))

    assert response.status_code == 400
    assert "Permission was not granted" in response.body.decode()
    assert "provider=meta app_id=facebook actor_flow=false" in caplog.text
    assert "reason=provider_error" in caplog.text
    assert db.query(ActorOAuthFlowState).count() == 0


# --- The background passes and disconnect ---------------------------------------


def test_the_missing_account_pass_tombstones_an_ordinary_row(ordinary, monkeypatch):
    db, user, composio = ordinary
    monkeypatch.setattr(composio_auth, "_missing_account_cursor", 0)
    _store_composio_grant(db, user, account_id="ca_gone", owner=None)
    composio.account_lookup["ca_gone"] = (404, MISSING_ACCOUNT)

    assert composio_auth.tombstone_missing_composio_accounts(db) == 1

    (stored,) = _ordinary_grants(db)
    assert stored.resource_owner_key is None
    assert stored.access_token == ""
    assert stored.refresh_token is None


def test_the_orphan_pass_counts_an_ordinary_row_as_a_reference(ordinary):
    db, user, composio = ordinary
    _store_composio_grant(db, user, account_id="ca_bound", owner=None)
    composio.listing[""] = {"items": [_listed("ca_bound"), _listed("ca_orphan")]}

    assert composio_auth.reconcile_orphan_accounts(db, now=RECONCILE_NOW) == 1

    assert composio.paths("DELETE") == ["/connected_accounts/ca_orphan"]
    assert [grant.access_token for grant in _ordinary_grants(db)] == ["ca_bound"]


def test_the_web_disconnect_schedules_deletion_of_an_ordinary_account(ordinary):
    """The web disconnect calls this with ``resource_owner_key=None``
    (api/mcp.py); it must delete the Composio account too."""
    db, user, composio = ordinary
    _store_composio_grant(db, user, account_id="ca_web", owner=None)
    _store_composio_grant(db, user, account_id="ca_actor", owner=OWNER)

    deleted = delete_scoped_user_oauth_accounts(
        db,
        user_id=int(user.id),
        resource_owner_key=None,
        providers=["facebook"],
    )
    assert deleted == 1
    assert composio.calls == []

    db.commit()

    assert composio.paths("DELETE") == ["/connected_accounts/ca_web"]
    grants = _grants(db)
    assert [(grant.resource_owner_key, grant.access_token) for grant in grants] == [
        (OWNER, "ca_actor")
    ]
