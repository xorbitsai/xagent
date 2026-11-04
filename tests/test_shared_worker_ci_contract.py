"""The shared multiprocess tests must receive Redis in the fast CI job."""

from pathlib import Path

import yaml


def test_pytest_fast_keeps_redis_for_shared_multiprocess_tests():
    workflow = yaml.safe_load(
        (Path(__file__).parents[1] / ".github/workflows/ci.yml").read_text()
    )
    job = workflow["jobs"]["pytest-fast"]
    assert job["env"]["XAGENT_TEST_REDIS_URL"] == "redis://localhost:6379/0"
    service = job["services"]["redis"]
    assert service["image"].startswith("redis:")
    assert "6379:6379" in service["ports"]
    assert "redis-cli ping" in service["options"]
