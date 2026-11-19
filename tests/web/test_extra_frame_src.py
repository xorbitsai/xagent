"""Host-supplied frame-src origins in the packaged SPA policy."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from xagent import config
from xagent.web.browser_security import page_security_headers
from xagent.web.extra_frame_src import extra_frame_src_origins

BASE_FRAME_SRC = "frame-src 'self' blob: https://docs.google.com"

VALID = [
    ("https://widget.example.com", ["https://widget.example.com"]),
    ("https://*.widget.example.com", ["https://*.widget.example.com"]),
    ("https://intranet", ["https://intranet"]),
    ("https://xn--bcher-kva.example", ["https://xn--bcher-kva.example"]),
    (
        " https://a.example.com\thttps://*.b.example.com\nhttps://a.example.com ",
        ["https://a.example.com", "https://*.b.example.com"],
    ),
    (
        "https://a.example.com https://evil.example.com;script-src "
        "https://*.b.example.com",
        ["https://a.example.com", "https://*.b.example.com"],
    ),
]

INVALID = [
    "widget.example.com",
    "http://widget.example.com",
    "wss://widget.example.com",
    "https://widget.example.com:443",
    "https://widget.example.com/",
    "https://widget.example.com/chat",
    "https://widget.example.com?x=1",
    "https://widget.example.com#x",
    "https://user@widget.example.com",
    "https://widget%2eexample.com",
    "https://widget.example.com\\evil",
    "https://Widget.example.com",
    "https://wídget.example.com",
    "https://widget.example.com.",
    "https://-widget.example.com",
    "https://widget-.example.com",
    f"https://{'a' * 64}.example.com",
    "https://10.0.0.1",
    "https://[::1]",
    "https://*",
    "*",
    "https://*.com",
    "https://*.*.example.com",
    "https://a.*.example.com",
    "https://widget.example.com;script-src",
    "https://widget.example.com,https://evil.example.com",
    "https://a.example.com\u00a0https://b.example.com",
    "'self'",
    "'unsafe-inline'",
    "data:",
]


def _frame_src(policy: str) -> str:
    return next(part for part in policy.split("; ") if part.startswith("frame-src "))


@pytest.mark.parametrize("configured, expected", VALID)
def test_valid_extra_frame_src_origins_are_appended(monkeypatch, configured, expected):
    monkeypatch.delenv(config.HTML_PREVIEW_ORIGIN, raising=False)
    monkeypatch.setenv(config.CSP_EXTRA_FRAME_SRC, configured)
    for embeddable in [False, True]:
        policy = page_security_headers(embeddable=embeddable)["Content-Security-Policy"]
        assert _frame_src(policy) == " ".join([BASE_FRAME_SRC, *expected])


@pytest.mark.parametrize("configured", INVALID)
def test_invalid_extra_frame_src_entries_are_refused(configured):
    assert extra_frame_src_origins(configured) == []


@pytest.mark.parametrize("embeddable", [False, True])
def test_unset_extra_frame_src_leaves_the_policy_unchanged(monkeypatch, embeddable):
    monkeypatch.delenv(config.HTML_PREVIEW_ORIGIN, raising=False)
    monkeypatch.delenv(config.CSP_EXTRA_FRAME_SRC, raising=False)
    assert config.get_csp_extra_frame_src() == ""
    baseline = page_security_headers(embeddable=embeddable)
    assert f"; {BASE_FRAME_SRC}; worker-src" in baseline["Content-Security-Policy"]
    for configured in ["", " \t\n", *INVALID]:
        monkeypatch.setenv(config.CSP_EXTRA_FRAME_SRC, configured)
        assert page_security_headers(embeddable=embeddable) == baseline


@pytest.mark.parametrize("embeddable", [False, True])
def test_packaged_and_next_policies_agree_on_extra_frame_src(monkeypatch, embeddable):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for cross-runtime policy parity")
    configured = " ".join([*INVALID, *(value for value, _ in VALID)])
    monkeypatch.delenv(config.HTML_PREVIEW_ORIGIN, raising=False)
    monkeypatch.setenv(config.CSP_EXTRA_FRAME_SRC, configured)
    result = subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            "import {browserSecurityHeaders} from './browser-security.mjs';"
            f"console.log(JSON.stringify(Object.fromEntries(browserSecurityHeaders({{embeddable: {str(embeddable).lower()}}}).map(x=>[x.key,x.value]))))",
        ],
        cwd=Path(__file__).resolve().parents[2] / "frontend",
        check=True,
        capture_output=True,
        text=True,
    )
    headers = page_security_headers(embeddable=embeddable)
    assert json.loads(result.stdout) == headers
    assert _frame_src(headers["Content-Security-Policy"]) == " ".join(
        [
            BASE_FRAME_SRC,
            "https://widget.example.com",
            "https://*.widget.example.com",
            "https://intranet",
            "https://xn--bcher-kva.example",
            "https://a.example.com",
            "https://*.b.example.com",
        ]
    )
