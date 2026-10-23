"""Sensitive tool inventory and usage endpoints require an authenticated user."""

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from xagent.web.api.monitor import monitor_router
from xagent.web.api.tools import tools_router
from xagent.web.auth_dependencies import get_current_user
from xagent.web.models.database import get_db


def _route_dependency_calls(router, path: str) -> set[object]:
    route = next(
        route
        for route in router.routes
        if isinstance(route, APIRoute) and route.path == path
    )
    dependency_graph = getattr(route, "depend" + "ant")
    return {dependency.call for dependency in dependency_graph.dependencies}


def test_tool_usage_requires_current_user() -> None:
    assert get_current_user in _route_dependency_calls(tools_router, "/api/tools/usage")


def test_monitor_tools_requires_current_user() -> None:
    assert get_current_user in _route_dependency_calls(
        monitor_router, "/api/monitor/tools"
    )


@pytest.mark.parametrize("path", ["/api/tools/usage", "/api/monitor/tools"])
def test_sensitive_tool_endpoints_reject_anonymous_requests(path: str) -> None:
    app = FastAPI()
    app.include_router(tools_router)
    app.include_router(monitor_router)
    app.dependency_overrides[get_db] = lambda: None

    response = TestClient(app).get(path)

    assert response.status_code == 401
