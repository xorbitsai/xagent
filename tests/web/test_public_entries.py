import asyncio
import json

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from xagent.config import (
    get_dual_entry_oauth_enabled,
    get_primary_public_site_url,
    get_public_api_base_url,
    get_public_entries,
    normalize_public_origin,
)
from xagent.web.public_entries import (
    google_state_entry,
    request_public_entry,
    validate_exchange_entry,
)

OLD = "https://cloud.xagent.co"
NEW = "https://tobybot.ai"


@pytest.fixture
def entries(monkeypatch):
    monkeypatch.setenv("XAGENT_APP_BASE_URL", OLD)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", OLD)
    records = [
        {
            "origin": origin,
            "region_cookie_domain": domain,
            "google_oidc_redirect_uri": origin + "/api/auth/oidc/google/callback",
        }
        for origin, domain in [(OLD, ".cloud.xagent.co"), (NEW, None)]
    ]
    monkeypatch.setenv("XAGENT_PUBLIC_ENTRIES_JSON", json.dumps(records))
    return records


def request(host, query=b"", headers=None):
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "query_string": query,
            "headers": headers or [(b"host", host.encode())],
            "session": {},
        }
    )


def test_null_cookie_and_promotion_do_not_move_protocols(entries, monkeypatch):
    assert get_public_entries()[1].region_cookie_domain is None
    monkeypatch.setenv("XAGENT_PRIMARY_PUBLIC_SITE_URL", NEW)
    assert get_primary_public_site_url() == NEW
    assert get_public_api_base_url() == OLD
    assert not get_dual_entry_oauth_enabled()
    assert request_public_entry(request("tobybot.ai:443")).origin == NEW


@pytest.mark.parametrize(
    "raw",
    [
        "https://evil.test/path",
        "https://u@evil.test",
        "https://evil.test?",
        "https://evil.test#",
        "https://evil.test\\foo",
        "https://evil.test,other",
        "javascript:bad",
    ],
)
def test_strict_origins(raw):
    with pytest.raises(ValueError):
        normalize_public_origin(raw)


def test_default_port_normalization():
    assert normalize_public_origin("https://TOBYBOT.AI:443/") == NEW


@pytest.mark.parametrize(
    "headers",
    [
        [(b"host", b"evil.test")],
        [(b"x-forwarded-host", b"tobybot.ai,cloud.xagent.co")],
        [(b"x-forwarded-host", b"tobybot.ai"), (b"x-forwarded-host", b"tobybot.ai")],
    ],
)
def test_ambiguous_or_unknown_forwarding_rejected(entries, headers):
    with pytest.raises(HTTPException):
        request_public_entry(request("unused", headers=headers))


def test_bad_entry_configuration(entries, monkeypatch):
    for records in [
        entries + entries[:1],
        [entries[1]],
        [dict(entries[0], region_cookie_domain=".evil.test")],
        [
            dict(
                entries[0],
                google_oidc_redirect_uri=NEW + "/api/auth/oidc/google/callback",
            )
        ],
    ]:
        monkeypatch.setenv("XAGENT_PUBLIC_ENTRIES_JSON", json.dumps(records))
        with pytest.raises(ValueError):
            get_public_entries()


def test_authlib_framework_contract_and_cross_entry_rejection(entries):
    from authlib.integrations.starlette_client import OAuth

    google = OAuth().register(name="google", client_id="test")
    req = request("tobybot.ai", b"state=bound")
    asyncio.run(
        google.save_authorize_data(
            req, state="bound", redirect_uri=NEW + "/api/auth/oidc/google/callback"
        )
    )
    assert asyncio.run(google_state_entry(req, google)).origin == NEW
    req.scope["headers"] = [(b"host", b"cloud.xagent.co")]
    # Build a new Request to avoid Starlette's cached headers.
    with pytest.raises(ValueError):
        asyncio.run(google_state_entry(Request(req.scope), google))


def test_exchange_old_codes_are_legacy_only(entries):
    assert validate_exchange_entry(request("cloud.xagent.co"), None)
    assert not validate_exchange_entry(request("tobybot.ai"), None)
    assert validate_exchange_entry(request("tobybot.ai"), NEW)
    assert not validate_exchange_entry(request("cloud.xagent.co"), NEW)


def test_legacy_frontend_path_prefix_is_not_reinterpreted(monkeypatch):
    monkeypatch.delenv("XAGENT_PUBLIC_ENTRIES_JSON", raising=False)
    monkeypatch.setenv("XAGENT_APP_BASE_URL", OLD + "/base")
    assert get_public_entries()[0].origin == OLD
    assert get_primary_public_site_url() == OLD + "/base"


def test_explicit_callback_preserves_actual_redirect_uri_normalization(
    entries, monkeypatch
):
    entries[1]["origin"] = "https://TOBYBOT.AI:443"
    entries[1]["google_oidc_redirect_uri"] = (
        "https://TOBYBOT.AI:443/api/auth/oidc/google/callback"
    )
    monkeypatch.setenv("XAGENT_PUBLIC_ENTRIES_JSON", json.dumps(entries))
    assert get_public_entries()[1].origin == NEW
    assert (
        get_public_entries()[1].google_oidc_redirect_uri
        == entries[1]["google_oidc_redirect_uri"]
    )


def test_unconfigured_entry_keeps_raw_legacy_origin(monkeypatch):
    monkeypatch.delenv("XAGENT_PUBLIC_ENTRIES_JSON", raising=False)
    monkeypatch.delenv("XAGENT_APP_BASE_URL", raising=False)
    monkeypatch.setenv("XAGENT_FRONTEND_URL", "http://xagent_web:3000")
    entry = get_public_entries()[0]
    assert entry.origin == "http://xagent_web:3000"
    assert entry.region_cookie_domain is None
    assert request_public_entry(request("anything.example")) == entry
    assert validate_exchange_entry(request("anything.example"), None)
