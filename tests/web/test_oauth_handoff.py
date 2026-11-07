import html
import json
import re
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from starlette.requests import Request
from starlette.responses import RedirectResponse

from xagent.web.api import auth
from xagent.web.models.database import Base, get_db
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.user import User
from xagent.web.oauth_handoff import OAuthHandoffContext, issue_handoff, router

OLD = "https://cloud.xagent.co"
NEW = "https://tobybot.ai"


class Adapter:
    def context(self, request):
        return OAuthHandoffContext("xagent", "prod", "au", OLD)

    def set_route_cookie(self, response, *, state, kind, provider):
        response.set_cookie("scoped-" + kind, "au")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_APP_BASE_URL", OLD)
    monkeypatch.setenv("XAGENT_DUAL_ENTRY_OAUTH_ENABLED", "true")
    monkeypatch.setenv(
        "XAGENT_PUBLIC_ENTRIES_JSON",
        json.dumps(
            [
                {
                    "origin": origin,
                    "region_cookie_domain": None,
                    "google_oidc_redirect_uri": origin
                    + "/api/auth/oidc/google/callback",
                }
                for origin in [OLD, NEW]
            ]
        ),
    )
    engine = create_engine(
        f"sqlite:///{tmp_path / 'handoff.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user = User(username="handoff", password_hash="unused", session_epoch="epoch")
        db.add(user)
        db.add(
            OAuthProvider(
                name="Sample",
                redirect_uri=OLD + "/api/auth/sample/callback",
                provider_name="sample",
                client_id="test",
                client_secret="test",
                auth_url="https://provider.test",
                token_url="https://provider.test/token",
            )
        )
        db.commit()
        app = FastAPI()
        app.state.oauth_handoff_adapter = Adapter()
        app.include_router(router)
        app.dependency_overrides[get_db] = lambda: db
        req = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/",
                "headers": [(b"host", b"tobybot.ai")],
                "app": app,
            }
        )
        launch = issue_handoff(
            req,
            user,
            kind="builtin",
            resource={"provider": "sample", "options": {"redirect": NEW}},
        )
        monkeypatch.setattr(
            auth,
            "_generic_oauth_login",
            lambda *a, **kw: RedirectResponse(
                "https://provider.test/authorize?state=signed-state"
            ),
        )
        yield TestClient(app, base_url=OLD), db, launch, app
    engine.dispose()


def _confirm_href(page) -> str:
    """The Continue link of the step-one confirmation page."""
    return html.unescape(re.search(r'href="([^"]+)"', page.text)[1])


def _confirmed_url(launch, page) -> str:
    return launch["authorization_url"].split("?", 1)[0] + _confirm_href(page)


def test_confirmation_page_names_account_and_consumes_nothing(setup):
    client, _, launch, _ = setup
    page = client.get(launch["authorization_url"], follow_redirects=False)
    assert page.status_code == 200
    assert "Connect Sample?" in page.text
    assert "<strong>handoff</strong>" in page.text
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert page.headers["x-frame-options"] == "DENY"
    assert page.headers["cache-control"] == "no-store"
    assert page.headers["referrer-policy"] == "no-referrer"
    cookie = page.headers["set-cookie"]
    assert cookie.startswith("oauth_launch_confirm_")
    assert "samesite=strict" in cookie.lower() and "httponly" in cookie.lower()
    assert "Path=/api/oauth/launch/au" in cookie
    assert "Secure" in cookie
    assert "scoped-builtin" not in cookie
    # Rendering again is harmless: the ticket was not consumed.
    assert client.get(launch["authorization_url"]).status_code == 200


def test_bootstrap_single_use_and_scoped_only(setup):
    client, _, launch, _ = setup
    client.cookies.set("selected_region", "sg")
    page = client.get(launch["authorization_url"], follow_redirects=False)
    response = client.get(_confirmed_url(launch, page), follow_redirects=False)
    assert response.status_code == 303
    assert "scoped-builtin" in response.headers["set-cookie"]
    assert "selected_region" not in response.headers["set-cookie"]
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert (
        client.get(_confirmed_url(launch, page), follow_redirects=False).status_code
        == 401
    )


@pytest.mark.parametrize("variant", ["no-cookie", "wrong-nonce", "foreign-cookie"])
def test_confirmation_requires_this_browsers_nonce(setup, variant):
    """A link straight to step two (e.g. from someone else's ticket) is refused."""
    client, _, launch, _ = setup
    page = client.get(launch["authorization_url"], follow_redirects=False)
    url = _confirmed_url(launch, page)
    if variant == "no-cookie":
        client.cookies.clear()
    if variant == "wrong-nonce":
        url = url.rsplit("confirm=", 1)[0] + "confirm=forged"
    if variant == "foreign-cookie":
        name = page.headers["set-cookie"].split("=", 1)[0]
        client.cookies.clear()
        client.cookies.set(name, "attacker-nonce", path="/api/oauth/launch/au")
    response = client.get(url, follow_redirects=False)
    assert response.status_code == 401
    assert "set-cookie" not in response.headers
    # The refusal did not consume the ticket: the real browser can still finish.
    page = client.get(launch["authorization_url"], follow_redirects=False)
    finished = client.get(_confirmed_url(launch, page), follow_redirects=False)
    assert finished.status_code == 303


def test_consume_storage_failure_is_unavailable(setup, monkeypatch):
    from sqlalchemy.exc import OperationalError

    from xagent.web.services import oauth_launch

    def unavailable(*args, **kwargs):
        raise OperationalError("consume", {}, Exception("down"))

    client, _, launch, _ = setup
    page = client.get(launch["authorization_url"], follow_redirects=False)
    monkeypatch.setattr(oauth_launch, "consume_oauth_launch_ticket", unavailable)
    response = client.get(_confirmed_url(launch, page), follow_redirects=False)
    assert response.status_code == 503


@pytest.mark.parametrize(
    "field,value",
    [
        ("product", "toby"),
        ("environment", "dev"),
        ("region", "sg"),
        ("target_origin", NEW),
        ("purpose", "toby"),
        ("type", "oauth_launch"),
        ("session_epoch", "old"),
        ("exp", 1),
        ("user_id", 2**31),
        ("jti", None),
        ("return_origin", "https://evil.test"),
    ],
)
def test_bound_claims_rejected_before_proof(setup, field, value):
    client, _, launch, _ = setup
    claims = auth.verify_token(
        parse_qs(urlsplit(launch["authorization_url"]).query)["ticket"][0]
    )
    claims[field] = value
    # Encode directly so the token helper cannot overwrite the expiry under test.
    jwt = auth.jwt

    ticket = jwt.encode(claims, auth.JWT_SECRET_KEY, algorithm=auth.JWT_ALGORITHM)
    response = client.get(
        "/api/oauth/launch/au", params={"ticket": ticket}, follow_redirects=False
    )
    assert response.status_code == 401
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize(
    "variant", ["region", "host", "duplicate", "head", "adapter", "disabled"]
)
def test_request_context_fail_closed(setup, monkeypatch, variant):
    client, _, launch, app = setup
    url = launch["authorization_url"]
    if variant == "region":
        url = url.replace("/au?", "/sg?")
    if variant == "host":
        url = url.replace(OLD, NEW)
    if variant == "duplicate":
        url += "&ticket=second"
    if variant == "adapter":
        del app.state.oauth_handoff_adapter
    if variant == "disabled":
        monkeypatch.setenv("XAGENT_DUAL_ENTRY_OAUTH_ENABLED", "false")
    response = client.request(
        "HEAD" if variant == "head" else "GET", url, follow_redirects=False
    )
    assert response.status_code in {401, 405, 503}
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize("enabled,version", [(False, "2"), (True, None), (True, "1")])
def test_old_frontend_keeps_legacy_response(setup, monkeypatch, enabled, version):
    client, db, _, app = setup
    monkeypatch.setenv("XAGENT_DUAL_ENTRY_OAUTH_ENABLED", str(enabled).lower())
    user = db.query(User).filter_by(username="handoff").one()
    headers = [(b"host", b"tobybot.ai")]
    if version:
        headers.append((b"x-xagent-oauth-launch-version", version.encode()))
    req = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "query_string": b"",
            "headers": headers,
            "app": app,
        }
    )
    response = auth.start_oauth_authorization(
        "sample", req, auth.OAuthLaunchRequest(), db, user
    )
    data = json.loads(response.body)
    assert "launch_mode" not in data
    assert data["authorization_url"].startswith("/api/auth/sample/login?ticket=")
