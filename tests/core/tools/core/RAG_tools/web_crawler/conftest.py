"""Shared fixtures for web crawler tests."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest


@pytest.fixture(autouse=True)
def _no_real_robots_fetch(monkeypatch):
    """Explicit robots loading uses a mocked client; config/constructor stay I/O-free.

    404 keeps the behaviour these tests were already getting from the live
    request, minus the network. Tests that care about robots.txt override this
    by patching httpx.Client themselves.
    """
    client = MagicMock()
    client.return_value.__enter__.return_value.get.return_value = MagicMock(
        status_code=404, text=""
    )
    monkeypatch.setattr(httpx, "Client", client)
    # Unit tests mock HTTP sessions, including target DNS validation. Real DNS
    # policy and real clients are exercised outside this fixture's subtree.
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.web_crawler.crawler.validate_public_http_url",
        AsyncMock(return_value=["93.184.216.34"]),
    )
