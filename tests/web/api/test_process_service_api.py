"""Tests for process service API routing."""

from xagent.web.api.services import router


def test_process_service_router_uses_explicit_prefix():
    """Process service routes must not collide with app-level health routes."""
    paths = {route.path for route in router.routes}

    assert "/api/process-service/status" in paths
    assert "/api/process-service/health" in paths
    assert "/health" not in paths
