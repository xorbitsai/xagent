"""Regression tests for the single-process web security boundaries."""

import asyncio
from pathlib import Path

import pytest

from xagent.web.api.files import _inline_preview_response
from xagent.web.app import _blocked_upload_extension, _security_header_values


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
    assert page["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" not in page


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
