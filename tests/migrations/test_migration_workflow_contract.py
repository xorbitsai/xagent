from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "test-migrations.yml"
GENERATION_MODEL_PATHS = (
    "src/xagent/web/models/generation.py",
    "src/xagent/web/models/mcp.py",
    "src/xagent/web/models/public_mcp.py",
)


def _workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _push_paths(text: str) -> set[str]:
    block = re.search(r"(?ms)^  push:\n.*?^  pull_request:", text)
    assert block is not None
    return set(re.findall(r"^      - '([^']+)'$", block.group(0), re.MULTILINE))


def _detector_paths(text: str) -> set[str]:
    block = re.search(r"(?ms)^          RELEVANT_PATHS=\(\n(.*?)^          \)", text)
    assert block is not None
    return set(re.findall(r"^            (\S+)$", block.group(1), re.MULTILINE))


@pytest.mark.parametrize("model_path", GENERATION_MODEL_PATHS)
def test_generation_model_changes_run_real_postgresql_migration_step(
    model_path: str,
) -> None:
    text = _workflow_text()
    assert re.search(r"(?m)^  pull_request:\n    branches: \[main\]$", text)
    assert model_path in _push_paths(text)
    assert model_path in _detector_paths(text)

    workflow = yaml.safe_load(text)
    job = workflow["jobs"]["test-postgresql-migrations"]
    assert job["needs"] == "detect-migration-changes"
    steps = job["steps"]
    sentinel = next(step for step in steps if step["name"] == "Skip migration tests")
    regression = next(
        step
        for step in steps
        if step["name"] == "Test MCP lifecycle generation migration (Postgres-only)"
    )
    output = "needs.detect-migration-changes.outputs.should-test"
    assert sentinel["if"] == f"{output} != 'true'"
    assert regression["if"] == f"{output} == 'true'"
    assert (
        "pytest tests/migrations/test_20260902_add_mcp_lifecycle_generations.py "
        "-m postgresql -q"
    ) in regression["run"]
