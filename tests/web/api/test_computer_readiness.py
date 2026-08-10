from __future__ import annotations

from types import SimpleNamespace

import pytest
from starlette.responses import Response

from xagent.core.computer.native_browser_readiness import LocalBrowserReadiness
from xagent.web.api import computer


@pytest.mark.asyncio
async def test_local_browser_readiness_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XAGENT_NATIVE_BROWSER_ENABLED", raising=False)
    response = Response()

    result = await computer.get_local_browser_readiness_endpoint(
        response,
        user=SimpleNamespace(is_admin=True),  # type: ignore[arg-type]
    )

    assert result.ready is False
    assert [issue.code for issue in result.issues] == ["disabled"]
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.asyncio
async def test_local_browser_readiness_does_not_probe_for_non_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_NATIVE_BROWSER_ENABLED", "true")

    async def unexpected_probe() -> LocalBrowserReadiness:
        raise AssertionError("driver probe must not run")

    monkeypatch.setattr(
        computer,
        "probe_local_browser_readiness",
        unexpected_probe,
    )

    result = await computer.get_local_browser_readiness_endpoint(
        Response(),
        user=SimpleNamespace(is_admin=False),  # type: ignore[arg-type]
    )

    assert result.ready is False
    assert [issue.code for issue in result.issues] == ["not_authorized"]


@pytest.mark.asyncio
async def test_local_browser_readiness_probes_for_enabled_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_NATIVE_BROWSER_ENABLED", "true")
    expected = LocalBrowserReadiness(
        ready=True,
        connected=True,
        attached=True,
        application="Google Chrome",
        title="Inbox",
    )

    async def probe() -> LocalBrowserReadiness:
        return expected

    monkeypatch.setattr(computer, "probe_local_browser_readiness", probe)

    result = await computer.get_local_browser_readiness_endpoint(
        Response(),
        user=SimpleNamespace(is_admin=True),  # type: ignore[arg-type]
    )

    assert result == expected


@pytest.mark.asyncio
async def test_local_browser_readiness_rejects_non_browser_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_NATIVE_BROWSER_ENABLED", "true")
    monkeypatch.setenv("XAGENT_NATIVE_BROWSER_APP_NAME", "Terminal")

    result = await computer.get_local_browser_readiness_endpoint(
        Response(),
        user=SimpleNamespace(is_admin=True),  # type: ignore[arg-type]
    )

    assert result.ready is False
    assert [issue.code for issue in result.issues] == ["invalid_configuration"]
    assert "supported browser" in result.message
