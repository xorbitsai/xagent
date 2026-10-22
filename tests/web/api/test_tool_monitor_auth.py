"""Sensitive tool inventory and usage endpoints require an authenticated user."""

from fastapi.routing import APIRoute

from xagent.web.api.monitor import monitor_router
from xagent.web.api.tools import tools_router
from xagent.web.auth_dependencies import get_current_user


def _route_dependency_calls(router, path: str) -> set[object]:
    route = next(
        route
        for route in router.routes
        if isinstance(route, APIRoute) and route.path == path
    )
    return {dependency.call for dependency in route.dependant.dependencies}


def test_tool_usage_requires_current_user() -> None:
    assert get_current_user in _route_dependency_calls(tools_router, "/api/tools/usage")


def test_monitor_tools_requires_current_user() -> None:
    assert get_current_user in _route_dependency_calls(
        monitor_router, "/api/monitor/tools"
    )
