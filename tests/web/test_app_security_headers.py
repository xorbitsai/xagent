"""Regression tests for the single-process web security boundaries."""

import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.web.api.files import _inline_preview_response
from xagent.web.app import (
    SafeUploadsStaticFiles,
    _blocked_upload_extension,
    _is_public_upload_path,
    _security_header_values,
    app,
)
from xagent.web.browser_security import page_security_headers
from xagent.web.frontend_static import mount_frontend


@pytest.mark.parametrize(
    "filename",
    [
        "page.html",
        "page.HTML",
        "page.xhtml",
        "page.xht",
        "style.xsl",
        "feed.rdf",
        "page.html.gz",
        "page.mathml",
    ],
)
def test_active_upload_document_extensions_are_blocked(filename: str) -> None:
    assert _blocked_upload_extension(filename)


def test_passive_upload_image_extension_is_not_blocked() -> None:
    assert not _blocked_upload_extension("logo.PNG")


def test_public_upload_mount_only_allows_agent_logos() -> None:
    assert _is_public_upload_path("agent_logos/agent_1.png")
    assert not _is_public_upload_path("agent_logos/")
    assert not _is_public_upload_path("user_1/private/report.pdf")
    assert not _is_public_upload_path("../user_1/private/report.pdf")
    assert not _is_public_upload_path("agent_logos/../user_1/report.pdf")
    assert not _is_public_upload_path("agent_logos/..")


def test_static_upload_mount_enforces_allowlist(tmp_path):
    logo = tmp_path / "agent_logos/a.png"
    logo.parent.mkdir()
    logo.write_bytes(b"logo")
    private = tmp_path / "user_1/x.pdf"
    private.parent.mkdir()
    private.write_bytes(b"private")
    app = FastAPI()
    app.mount("/uploads", SafeUploadsStaticFiles(directory=str(tmp_path)))
    with TestClient(app) as client:
        assert client.get("/uploads/agent_logos/a.png").content == b"logo"
        assert client.get("/uploads/user_1/x.pdf").status_code == 404
        assert client.get("/uploads/agent_logos/%2e%2e/user_1/x.pdf").status_code == 404


def test_security_headers_keep_widget_and_preview_embeddable() -> None:
    widget = _security_header_values("/widget/chat/session")
    preview = _security_header_values("/api/files/public/preview/file-id")

    assert "X-Frame-Options" not in widget
    assert "Content-Security-Policy" not in widget
    assert "X-Frame-Options" not in preview
    assert "frame-ancestors 'self'" in preview["Content-Security-Policy"]


def test_security_headers_harden_api_and_regular_pages() -> None:
    api = _security_header_values("/api/me")
    page = _security_header_values("/settings")

    assert api["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in api["Content-Security-Policy"]
    assert "script-src 'none'" in api["Content-Security-Policy"]
    assert "unsafe-eval" not in api["Content-Security-Policy"]
    assert page["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" not in page


def test_widget_traversal_does_not_exempt_regular_page_headers():
    assert _security_header_values("/widget/../tools.html")["X-Frame-Options"] == "DENY"


def test_security_headers_preserve_inline_oauth_popup_feedback_without_eval():
    policy = _security_header_values("/api/auth/google/callback")[
        "Content-Security-Policy"
    ]
    assert "script-src 'self' 'unsafe-inline'" in policy
    assert "unsafe-eval" not in policy


def test_security_headers_only_exempt_file_preview_routes() -> None:
    agent_preview = _security_header_values("/api/agents/preview")
    workforce_preview = _security_header_values("/api/workforces/preview/runs")

    assert agent_preview["X-Frame-Options"] == "DENY"
    assert workforce_preview["X-Frame-Options"] == "DENY"


def test_public_html_preview_is_downloaded_instead_of_executed(tmp_path: Path) -> None:
    html = tmp_path / "report.html"
    html.write_text("<script>alert(1)</script>", encoding="utf-8")

    response = asyncio.run(
        _inline_preview_response(
            html,
            filename=html.name,
            media_type="text/html",
            public_validation=True,
        )
    )

    content_disposition = response.headers["content-disposition"]
    assert content_disposition.startswith("attachment;")
    assert 'filename="report.html"' in content_disposition


@pytest.mark.parametrize(
    "preview_origin", ["", "https://preview.example.test/path", "auto"]
)
def test_production_middleware_and_frontend_mount_share_one_page_policy(
    tmp_path, monkeypatch, preview_origin
):
    # Exercise the actual middleware stack without starting background workers
    # or external services. Restore the production route table after this test.
    monkeypatch.setattr(app.router, "routes", list(app.router.routes))
    monkeypatch.setenv("NEXT_PUBLIC_XAGENT_HTML_PREVIEW_ORIGIN", preview_origin)
    monkeypatch.setattr(app, "middleware_stack", None)
    monkeypatch.setattr(
        app.state, "file_storage_startup_sync_task", None, raising=False
    )
    monkeypatch.setattr(
        app.state, "file_storage_startup_sync_error", None, raising=False
    )
    (tmp_path / "index.html").write_text("<html>home</html>")
    (tmp_path / "widget").mkdir()
    (tmp_path / "widget" / "__shell__.html").write_text("<html>widget</html>")
    assert mount_frontend(app, tmp_path)
    client = TestClient(app)
    for path, embeddable in [("/", False), ("/widget/capability", True)]:
        response = client.get(path)
        assert response.status_code == 200
        policies = response.headers.get_list("content-security-policy")
        assert policies == [
            page_security_headers(embeddable=embeddable)["Content-Security-Policy"]
        ]
        assert response.headers.get("x-frame-options") == (
            None if embeddable else "DENY"
        )
        frame_policy = policies[0].split("frame-src ")[1].split(";")[0]
        if preview_origin == "auto":
            assert "https://preview.xagent.co" in frame_policy
        elif preview_origin:
            assert "https://preview.example.test" in frame_policy


@pytest.mark.parametrize(
    "configured, expected",
    [
        ("", []),
        ("invalid", []),
        ("http://preview.example.test", []),
        ("https://user:pass@preview.example.test", []),
        ("https://preview.example.test;evil", []),
        ("https://[fe80::1%25x * y]", []),
        ("https://[fe80::1%25a;script-src *]", []),
        ("https://[v1.evil.example]", []),
        ("https://preview.example.test:invalid", []),
        ("https://preview.example.test/", ["https://preview.example.test"]),
        (
            " https://preview.example.test:443/path?q=1 ",
            ["https://preview.example.test"],
        ),
        ("http://localhost:8787", ["http://localhost:8787"]),
        ("http://127.0.0.1:8787/", ["http://127.0.0.1:8787"]),
        ("http://[::1]:8787/path", ["http://[::1]:8787"]),
        (
            "auto",
            [
                "https://preview.xagent.co",
                "https://preview-stage.xagent.co",
                "https://preview-dev.xagent.co",
            ],
        ),
    ],
)
def test_configured_runner_csp_is_a_narrow_origin_allowlist(
    monkeypatch, configured, expected
):
    monkeypatch.setenv("NEXT_PUBLIC_XAGENT_HTML_PREVIEW_ORIGIN", configured)
    for embeddable in [False, True]:
        policy = page_security_headers(embeddable=embeddable)["Content-Security-Policy"]
        frame_policy = next(
            part for part in policy.split("; ") if part.startswith("frame-src ")
        )
        assert frame_policy == " ".join(
            ["frame-src 'self' blob: https://docs.google.com", *expected]
        )
