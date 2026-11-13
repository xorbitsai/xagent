"""Tests for host memory provenance validation."""

from types import MappingProxyType
from typing import Any

import pytest

from xagent.core.memory.provenance import validate_memory_provenance


def test_validate_memory_provenance_returns_frozen_copy() -> None:
    source = {"channel_kind": "dm", "channel_id": "D123ABC", "task_id": "42"}

    validated = validate_memory_provenance(source)

    assert isinstance(validated, MappingProxyType)
    assert dict(validated) == source
    # A later host-side mutation cannot reach notes stored from the copy.
    source["channel_id"] = "C999"
    assert validated["channel_id"] == "D123ABC"


def test_validate_memory_provenance_accepts_non_slack_identifiers() -> None:
    teams_id = "19:abcDEF0123@thread.tacv2"
    assert validate_memory_provenance({"channel_id": teams_id})["channel_id"] == (
        teams_id
    )


@pytest.mark.parametrize(
    "provenance",
    [
        # Conversation text can never pass the identifier grammar (#2753).
        {"channel_id": "please remember my salary is 100k"},
        {"note": "line\nbreak"},
        {"channel_id": ""},
        {"channel_id": "x" * 257},
        {"bad key": "C1"},
        {"": "C1"},
        {"channel:id": "C1"},
    ],
)
def test_validate_memory_provenance_rejects_text_and_bad_keys(
    provenance: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        validate_memory_provenance(provenance)


@pytest.mark.parametrize(
    "provenance",
    [{}, {"task_id": 42}, {"channel_id": None}, ["channel_id", "C1"]],
)
def test_validate_memory_provenance_rejects_wrong_shapes(provenance: Any) -> None:
    with pytest.raises(TypeError):
        validate_memory_provenance(provenance)
