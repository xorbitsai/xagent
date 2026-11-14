"""End-to-end L1 runs against an in-process fake deployment (#3245).

The fake answers the routes the check calls with the shapes the real ones
return (``/api/auth/login``, ``/api/mcp/apps``, and the
``/api/auth/{provider}/login`` launch start and redeem); no test opens a socket.
"""

import importlib
import io
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode

import httpx
import pytest

# Importable under both pytest and `unittest discover -s tests/scripts`.
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
cli = importlib.import_module("connector_smoke.cli")

BASE = "https://cloud.example.com"
ACCESS_TOKEN = "access-token-SECRET"
STATE = "state-jwt-SECRET"

REGISTRY_ROWS = [
    {"app_id": "slack", "is_visible_in_connector": True},
    {"app_id": "google-calendar", "is_visible_in_connector": True},
    {"app_id": "notion", "is_visible_in_connector": True},
    {"app_id": "zendesk", "is_visible_in_connector": False},
]


def _catalog_app(
    app_id: str, provider: str | None, auth_type: str, **extra: Any
) -> dict[str, Any]:
    app: dict[str, Any] = {
        "id": app_id,
        "name": app_id,
        "provider": provider,
        "auth_type": auth_type,
        "is_visible_in_connector": True,
        "launch_config": {"env_mapping": {"TOKEN": "access_token"}},
    }
    if provider == "zendesk":
        app["oauth_account_input"] = {"suffix": ".zendesk.com"}
    app.update(extra)
    return app


class FakeDeployment:
    def __init__(self) -> None:
        self.catalog = [
            _catalog_app("slack", "slack", "builtin_oauth"),
            _catalog_app("google-calendar", "google", "builtin_oauth"),
            _catalog_app("notion", None, "mcp_oauth"),
        ]
        self.client_ids = {
            "slack": "slack-client-SECRET",
            "google": "google-client-SECRET",
            "zendesk": "zendesk-client-SECRET",
        }
        self.callback_origin = BASE
        self.shared_callback_origin = "https://global.example.net"
        self.composio_apps: set[str] = set()
        self.requests: list[httpx.Request] = []
        self.launch_bodies: dict[str, dict[str, Any]] = {}
        self.require_login_route = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/auth/login/start":
            return httpx.Response(
                204, headers={"set-cookie": "login_route=route-SECRET; Path=/"}
            )
        if path == "/api/auth/login":
            if self.require_login_route and "login_route=route-SECRET" not in (
                request.headers.get("cookie", "")
            ):
                return httpx.Response(404, json={"detail": "route_token_not_found"})
            body = json.loads(request.content)
            if body.get("password") != "pw":
                return httpx.Response(401, json={"detail": "Invalid credentials"})
            return httpx.Response(200, json={"access_token": ACCESS_TOKEN})
        if request.headers.get("authorization") != f"Bearer {ACCESS_TOKEN}":
            return httpx.Response(401, json={"detail": "Not authenticated"})
        if path == "/api/mcp/apps":
            return httpx.Response(200, json=self.catalog)
        parts = path.split("/")
        if len(parts) == 5 and parts[1:3] == ["api", "auth"] and parts[4] == "login":
            provider = parts[3]
            if request.method == "POST":
                return self._start(provider, request)
            return self._redeem(provider, request)
        return httpx.Response(404)

    def _start(self, provider: str, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.launch_bodies[provider] = body
        ticket = f"ticket-{provider}-SECRET"
        origin = (
            self.shared_callback_origin
            if body.get("app_id") in self.composio_apps
            else self.callback_origin
        )
        return httpx.Response(
            200,
            json={
                "authorization_url": f"/api/auth/{provider}/login?"
                + urlencode({"ticket": ticket}),
                "callback_origin": origin,
            },
        )

    def _redeem(self, provider: str, request: httpx.Request) -> httpx.Response:
        assert parse_qs(request.url.query.decode())["ticket"] == [
            f"ticket-{provider}-SECRET"
        ]
        app_id = self.launch_bodies.get(provider, {}).get("app_id")
        if app_id in self.composio_apps:
            query = urlencode({"state": STATE})
            return httpx.Response(
                307,
                headers={
                    "location": f"{self.shared_callback_origin}"
                    f"/api/auth/meta/callback?{query}"
                },
            )
        client_id = self.client_ids.get(provider, "")
        if not client_id:
            env_name = f"{provider.upper()}_CLIENT_ID"
            return httpx.Response(
                500,
                html="<h1>Error: OAuth provider not configured</h1>"
                f"<p>Missing {env_name} for provider {provider}.</p>",
            )
        query = urlencode(
            {
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": f"{self.callback_origin}/api/auth/{provider}/callback",
                "state": STATE,
            }
        )
        return httpx.Response(
            307,
            headers={
                "location": f"https://{provider}.example.net/authorize?{query}",
                "set-cookie": "xagent_oauth_state_abc=nonce-SECRET; Path=/",
            },
        )


ENV = {
    "XAGENT_CONNECTOR_SMOKE_BASE_URL": BASE,
    "XAGENT_CONNECTOR_SMOKE_LOGIN": "monitor",
    "XAGENT_CONNECTOR_SMOKE_PASSWORD": "pw",
}


def _run(
    fake: FakeDeployment,
    *args: str,
    env: dict[str, str] | None = None,
    registry_rows: list[dict[str, Any]] | None = None,
) -> tuple[int, list[str]]:
    out = io.StringIO()
    code = cli.main(
        ["l1", *args],
        env={**ENV, **(env or {})},
        registry_rows=REGISTRY_ROWS if registry_rows is None else registry_rows,
        transport=httpx.MockTransport(fake.handler),
        out=out,
    )
    return code, out.getvalue().splitlines()


def _failures(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("FAIL")]


def _failed_checks(lines: list[str]) -> set[str]:
    return {line.split(" ")[1].rstrip(":") for line in _failures(lines)}


def test_healthy_deployment_passes_every_check() -> None:
    code, lines = _run(FakeDeployment())

    assert code == 0, lines
    assert _failures(lines) == []
    assert "PASS catalog:slack" in lines
    assert "PASS oauth:slack:callback" in lines
    assert "PASS oauth:slack:authorize" in lines
    assert "PASS oauth:google:authorize" in lines
    assert lines[-1] == "7 checks, 0 failed"


def test_unset_client_id_fails_only_that_provider() -> None:
    fake = FakeDeployment()
    fake.client_ids["slack"] = ""

    code, lines = _run(fake)

    assert code == 1
    assert _failures(lines) == [
        "FAIL oauth:slack:authorize: no provider redirect: HTTP 500 "
        "(Error: OAuth provider not configured - "
        "Missing SLACK_CLIENT_ID for provider slack.)"
    ]
    assert "PASS oauth:slack:callback" in lines
    assert "PASS oauth:google:authorize" in lines


def test_removed_expected_app_fails_the_catalog_check_by_name() -> None:
    fake = FakeDeployment()
    fake.catalog = [app for app in fake.catalog if app["id"] != "notion"]

    code, lines = _run(fake)

    assert code == 1
    assert _failures(lines) == [
        "FAIL catalog:notion: missing from the catalog (absent or hidden)"
    ]


def test_excluded_app_is_not_expected() -> None:
    fake = FakeDeployment()
    fake.catalog = [app for app in fake.catalog if app["id"] != "notion"]

    code, lines = _run(fake, "--exclude", "notion")

    assert code == 0, lines


def test_redirect_uri_fallback_to_localhost_fails_both_oauth_checks() -> None:
    fake = FakeDeployment()
    fake.callback_origin = "http://localhost:8000"

    code, lines = _run(fake)

    assert code == 1
    assert _failed_checks(lines) == {
        "oauth:slack:callback",
        "oauth:slack:authorize",
        "oauth:google:callback",
        "oauth:google:authorize",
    }


def test_output_never_contains_tokens_tickets_or_redirect_values() -> None:
    fake = FakeDeployment()
    fake.callback_origin = "http://localhost:8000"
    fake.client_ids["google"] = ""

    _, lines = _run(fake)

    output = "\n".join(lines)
    assert "SECRET" not in output
    assert "example.net" not in output


COMPOSIO_REGISTRY_ROWS = [
    {"app_id": "slack", "is_visible_in_connector": True},
    {"app_id": "salesforce", "is_visible_in_connector": True},
]


def _composio_deployment() -> FakeDeployment:
    fake = FakeDeployment()
    fake.catalog = [
        _catalog_app("slack", "slack", "builtin_oauth"),
        _catalog_app(
            "salesforce", "salesforce", "builtin_oauth", auth_backend="composio"
        ),
    ]
    fake.composio_apps = {"salesforce"}
    fake.client_ids.pop("salesforce", None)
    return fake


def test_a_composio_only_provider_needs_no_native_client() -> None:
    code, lines = _run(_composio_deployment(), registry_rows=COMPOSIO_REGISTRY_ROWS)

    assert code == 0, lines
    assert _failures(lines) == []
    assert "PASS oauth:salesforce:composio" in lines
    assert "PASS oauth:slack:callback" in lines
    assert "oauth:salesforce:callback" not in "\n".join(lines)
    assert "state-jwt-SECRET" not in "\n".join(lines)


def _launch_events(fake: FakeDeployment) -> list[tuple[str, str, str | None]]:
    events: list[tuple[str, str, str | None]] = []
    for request in fake.requests:
        parts = request.url.path.split("/")
        if len(parts) != 5 or parts[1:3] != ["api", "auth"] or parts[4] != "login":
            continue
        app_id = None
        if request.method == "POST":
            app_id = json.loads(request.content).get("app_id")
        events.append((request.method, parts[3], app_id))
    return events


def test_each_launch_is_redeemed_before_the_next_one_starts() -> None:
    fake = _composio_deployment()
    code, lines = _run(fake, registry_rows=COMPOSIO_REGISTRY_ROWS)

    assert code == 0, lines
    events = _launch_events(fake)
    redeems = [index for index, event in enumerate(events) if event[0] == "GET"]
    assert len(redeems) == len(events) // 2
    for index in redeems:
        method, provider, app_id = events[index - 1]
        assert (method, provider) == ("POST", events[index][1])
        assert app_id is not None


def test_the_composio_redeem_is_checked_against_its_own_start_origin() -> None:
    fake = _composio_deployment()
    fake.shared_callback_origin = "https://connect.example.net"
    assert fake.shared_callback_origin != BASE

    code, lines = _run(fake, registry_rows=COMPOSIO_REGISTRY_ROWS)

    assert code == 0, lines
    assert "PASS oauth:salesforce:composio" in lines


def test_a_composio_redeem_on_another_origin_than_its_start_fails() -> None:
    fake = _composio_deployment()
    original_redeem = fake._redeem

    def redeem(provider: str, request: httpx.Request) -> httpx.Response:
        if provider != "salesforce":
            return original_redeem(provider, request)
        return httpx.Response(
            307,
            headers={
                "location": "https://elsewhere.example.net"
                f"/api/auth/composio/callback?{urlencode({'state': STATE})}"
            },
        )

    fake._redeem = redeem  # type: ignore[method-assign]

    code, lines = _run(fake, registry_rows=COMPOSIO_REGISTRY_ROWS)

    assert code == 1
    assert _failures(lines) == [
        "FAIL oauth:salesforce:composio: callback origin is "
        "https://elsewhere.example.net, expected https://global.example.net"
    ]


@pytest.mark.parametrize("origin", [None, "", "/relative", "ftp://host"])
def test_a_composio_start_without_a_valid_callback_origin_fails(
    origin: str | None,
) -> None:
    fake = _composio_deployment()
    original_start = fake._start

    def start(provider: str, request: httpx.Request) -> httpx.Response:
        response = original_start(provider, request)
        if provider != "salesforce":
            return response
        payload = json.loads(response.content)
        if origin is None:
            del payload["callback_origin"]
        else:
            payload["callback_origin"] = origin
        return httpx.Response(200, json=payload)

    fake._start = start  # type: ignore[method-assign]

    code, lines = _run(fake, registry_rows=COMPOSIO_REGISTRY_ROWS)

    assert code == 1
    assert _failures(lines) == [
        "FAIL oauth:salesforce:composio: launch start returned no usable "
        "callback origin"
    ]
    assert ("GET", "salesforce", None) not in _launch_events(fake)


def _real_catalog_app(app_id: str) -> dict[str, Any]:
    """The entry the server's own catalog code builds for a builtin app."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app
    from xagent.web.mcp_apps import _app_to_dict
    from xagent.web.models.public_mcp import PublicMCPApp

    row = get_builtin_public_mcp_app(app_id)
    assert row is not None
    return _app_to_dict(PublicMCPApp(**row))


def test_the_facebook_switch_leaves_meta_native_and_adds_no_composio_check(
    monkeypatch,
) -> None:
    monkeypatch.setenv("XAGENT_FACEBOOK_AUTH_BACKEND", "composio")
    monkeypatch.setenv("COMPOSIO_FACEBOOK_AUTH_CONFIG_ID", "ac_facebook")
    monkeypatch.setenv("XAGENT_SALESFORCE_AUTH_BACKEND", "composio")
    monkeypatch.setenv("COMPOSIO_SALESFORCE_AUTH_CONFIG_ID", "ac_salesforce")
    monkeypatch.setenv("COMPOSIO_API_KEY", "key")
    monkeypatch.setenv("COMPOSIO_USER_ID_SECRET", "secret")
    fake = FakeDeployment()
    fake.shared_callback_origin = BASE
    fake.catalog = [_real_catalog_app("facebook"), _real_catalog_app("salesforce")]
    fake.composio_apps = {"salesforce"}
    fake.client_ids["meta"] = "meta-client-SECRET"
    fake.client_ids.pop("salesforce", None)

    code, lines = _run(
        fake,
        registry_rows=[
            {"app_id": "facebook", "is_visible_in_connector": True},
            {"app_id": "salesforce", "is_visible_in_connector": True},
        ],
    )

    assert code == 0, lines
    assert "PASS oauth:meta:callback" in lines
    assert "PASS oauth:meta:authorize" in lines
    assert "PASS oauth:salesforce:composio" in lines
    assert not [line for line in lines if "oauth:meta:composio" in line]
    events = _launch_events(fake)
    assert events == [
        ("POST", "meta", "facebook"),
        ("GET", "meta", None),
        ("POST", "salesforce", "salesforce"),
        ("GET", "salesforce", None),
    ]
    assert [event[1] for event in events if event[0] == "POST"] == [
        event[1] for event in events if event[0] == "GET"
    ]


def test_launch_names_the_catalog_app_and_never_follows_the_redirect() -> None:
    fake = FakeDeployment()

    _run(fake)

    assert fake.launch_bodies == {
        "google": {"app_id": "google-calendar"},
        "slack": {"app_id": "slack"},
    }
    assert all(r.url.host == "cloud.example.com" for r in fake.requests)


def test_saas_login_start_routes_with_the_given_cookies() -> None:
    fake = FakeDeployment()
    fake.require_login_route = True

    code, lines = _run(
        fake,
        "--login-start",
        "--login-field",
        "email",
        env={"XAGENT_CONNECTOR_SMOKE_COOKIES": "selected_region=sg"},
    )

    assert code == 0, lines
    start, login = fake.requests[0], fake.requests[1]
    assert start.url.path == "/api/auth/login/start"
    assert json.loads(start.content) == {"email": "monitor"}
    assert json.loads(login.content) == {"email": "monitor", "password": "pw"}
    for request in fake.requests:
        assert "selected_region=sg" in request.headers["cookie"]
    # The launch's OAuth state cookie is never sent back.
    assert all("nonce-SECRET" not in r.headers["cookie"] for r in fake.requests)
    assert all("login_route" not in r.headers["cookie"] for r in fake.requests[2:])


def test_login_failure_stops_the_run() -> None:
    code, lines = _run(
        FakeDeployment(), env={"XAGENT_CONNECTOR_SMOKE_PASSWORD": "wrong"}
    )

    assert code == 1
    assert lines == ["FAIL login: login failed: HTTP 401", "1 checks, 1 failed"]


def test_account_hosted_provider_needs_an_account_label() -> None:
    fake = FakeDeployment()
    fake.catalog.append(_catalog_app("zendesk", "zendesk", "builtin_oauth"))

    code, lines = _run(fake, "--include", "zendesk")

    assert code == 1
    assert _failures(lines) == [
        "FAIL oauth:zendesk:launch: needs an account label: "
        "pass --oauth-account zendesk=<label>"
    ]
    assert "zendesk" not in fake.launch_bodies


def test_account_label_is_sent_with_the_launch() -> None:
    fake = FakeDeployment()
    fake.catalog.append(_catalog_app("zendesk", "zendesk", "builtin_oauth"))

    code, lines = _run(fake, "--include", "zendesk", "--oauth-account", "zendesk=acme")

    assert code == 0, lines
    assert fake.launch_bodies["zendesk"] == {"app_id": "zendesk", "account": "acme"}


def test_foreign_authorization_url_is_never_requested() -> None:
    fake = FakeDeployment()
    fake.catalog = [_catalog_app("slack", "slack", "builtin_oauth")]
    original_start = fake._start

    def hostile_start(provider: str, request: httpx.Request) -> httpx.Response:
        original_start(provider, request)
        return httpx.Response(
            200,
            json={
                "authorization_url": "https://evil.example.org/api/auth/slack/login?x",
                "callback_origin": BASE,
            },
        )

    fake._start = hostile_start  # type: ignore[method-assign]

    _, lines = _run(fake, "--exclude", "google-calendar", "--exclude", "notion")

    assert _failed_checks(lines) == {"oauth:slack:authorize"}
    assert "unexpected authorization URL" in _failures(lines)[0]
    assert all(r.url.host == "cloud.example.com" for r in fake.requests)


def test_command_line_list_replaces_the_environment_value() -> None:
    fake = FakeDeployment()
    fake.catalog = [app for app in fake.catalog if app["id"] != "notion"]

    code, lines = _run(
        fake,
        "--exclude",
        "notion",
        env={"XAGENT_CONNECTOR_SMOKE_EXCLUDE": "slack,google-calendar"},
    )

    assert code == 0, lines
    assert "PASS catalog:slack" in lines


def test_connection_error_reports_the_class_only() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {request.url}", request=request)

    out = io.StringIO()
    code = cli.main(
        ["l1"],
        env=ENV,
        registry_rows=REGISTRY_ROWS,
        transport=httpx.MockTransport(refuse),
        out=out,
    )

    assert code == 1
    assert out.getvalue().splitlines()[0] == (
        "FAIL login: POST request failed: ConnectError"
    )


def test_unsendable_request_reports_the_class_only() -> None:
    # httpx refuses a non-ASCII header before sending; that is not an
    # httpx.HTTPError, and its traceback must not reach the output.
    code, lines = _run(FakeDeployment(), "--cookie", "selected_region=s\u00e9")

    assert code == 1
    assert lines[0] == "FAIL login: POST request failed: UnicodeEncodeError"


@pytest.mark.parametrize(
    ("env", "args"),
    [
        ({"XAGENT_CONNECTOR_SMOKE_PASSWORD": ""}, []),
        ({"XAGENT_CONNECTOR_SMOKE_BASE_URL": ""}, []),
        ({}, ["--exclude", "gmail"]),
        ({}, ["--cookie", "no-value"]),
    ],
)
def test_bad_configuration_is_a_usage_error(
    env: dict[str, str], args: list[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        _run(FakeDeployment(), *args, env=env)

    assert exc.value.code == 2
