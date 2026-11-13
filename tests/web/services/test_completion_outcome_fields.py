"""The ``completion_outcome`` key of a task response: when it is present and
what it holds."""

from __future__ import annotations

from typing import Any

import pytest

from xagent.core.agent.result import COMPLETION_OUTCOMES
from xagent.web.models.task import TaskStatus
from xagent.web.services import execution_result_projection as projection
from xagent.web.services import structured_output_projection as format_projection

STORED_VALUES: list[Any] = [*sorted(COMPLETION_OUTCOMES), None, "other", ["list"]]


def _is_rating(value: Any) -> bool:
    return type(value) is str and value in COMPLETION_OUTCOMES


@pytest.mark.parametrize("accept_partial", [True, False])
@pytest.mark.parametrize("stored", STORED_VALUES, ids=repr)
@pytest.mark.parametrize("status", list(TaskStatus))
def test_completion_outcome_fields(status, stored, accept_partial):
    fields = projection.completion_outcome_fields(
        status, stored, accept_partial=accept_partial
    )

    if not accept_partial:
        assert fields == {}
        return
    known = status is TaskStatus.COMPLETED and _is_rating(stored)
    assert fields == {"completion_outcome": stored if known else None}


@pytest.mark.parametrize(
    ("agent_config", "expected"),
    [
        (None, False),
        ("response_format", False),
        ({}, False),
        ({"response_format": None}, False),
        ({"response_format": {"type": "json_schema"}}, False),
        ({"response_format": {"accept_partial": True}}, True),
        ({"response_format": {"accept_partial": False}}, False),
        ({"response_format": {"accept_partial": 1}}, False),
        ({"response_format": {"accept_partial": "true"}}, False),
        ({"response_format": "accept_partial"}, False),
    ],
)
def test_the_switch_is_on_only_when_stored_as_exactly_true(agent_config, expected):
    assert format_projection.response_format_accepts_partial(agent_config) is expected


@pytest.mark.parametrize("status", list(TaskStatus))
@pytest.mark.parametrize("outcome", STORED_VALUES, ids=repr)
def test_completion_outcomes_single_source(status, outcome):
    result = {"completion_outcome": outcome}

    expected = (
        outcome if status is TaskStatus.COMPLETED and _is_rating(outcome) else None
    )
    assert projection.completion_outcome_for_status(result, status) == expected


@pytest.mark.parametrize(
    "read",
    [
        lambda: projection.completion_outcome_for_status(
            {"completion_outcome": "extra"}, TaskStatus.COMPLETED
        ),
        lambda: projection.reported_completion_outcome(TaskStatus.COMPLETED, "extra"),
    ],
    ids=["for_status", "reported"],
)
def test_the_ratings_follow_the_shared_set(monkeypatch, read):
    monkeypatch.setattr(
        projection, "COMPLETION_OUTCOMES", COMPLETION_OUTCOMES | {"extra"}
    )

    assert read() == "extra"
