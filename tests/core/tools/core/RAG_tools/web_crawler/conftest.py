"""Shared fixtures for web crawler tests."""

from unittest.mock import MagicMock

import httpx
import pytest


@pytest.fixture(autouse=True)
def _no_real_robots_fetch(monkeypatch):
    """URLFilter fetches robots.txt synchronously in __init__, so any test that
    builds one with the default respect_robots_txt fires a real request at the
    start URL's host. The fetch swallows its own exception, so an offline run
    would quietly exercise a different code path instead of failing.

    404 keeps the behaviour these tests were already getting from the live
    request, minus the network. Tests that care about robots.txt override this
    by patching httpx.Client themselves.
    """
    client = MagicMock()
    client.return_value.__enter__.return_value.get.return_value = MagicMock(
        status_code=404, text=""
    )
    monkeypatch.setattr(httpx, "Client", client)
