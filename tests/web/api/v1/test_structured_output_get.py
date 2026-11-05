"""GET /v1/chat/tasks/{id}: the structured output record column and the fields
projected from it."""

from __future__ import annotations

import pytest

from xagent.web.models.task import Task
from xagent.web.schemas.v1 import TaskInfoResponse

from ..conftest import client
from .structured_output_shared import (
    COLUMN,
    bearer,
    create_agent_with_key,
    insert_task,
    recorded_statements,
    task_selects,
)

pytestmark = pytest.mark.usefixtures("_test_db")


def test_the_column_is_a_plain_mapped_column_and_the_response_allows_extra_keys():
    """The record is read with the rest of the row, never lazily afterwards."""
    column = Task.__table__.c[COLUMN]
    prop = Task.__mapper__.get_property(COLUMN)

    assert not prop.deferred
    assert column.nullable
    assert TaskInfoResponse.model_config.get("extra") != "forbid"


def test_get_reads_the_record_column_in_the_same_select_as_the_row():
    agent_id, key = create_agent_with_key()
    task_id = insert_task(
        agent_id,
        record={"run_id": "run-1", "status": "ready", "output": {}, "reason": None},
    )

    with recorded_statements() as statements:
        resp = client.get(f"/v1/chat/tasks/{task_id}", headers=bearer(key))

    assert resp.status_code == 200, resp.text
    selects = task_selects(statements)
    assert selects, statements
    assert all(f"tasks.{COLUMN}" in s for s in selects)
