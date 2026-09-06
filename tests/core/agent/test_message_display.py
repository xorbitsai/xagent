from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from xagent.core.agent.message_display import resolve_message_display


CONTRACT_CASES: list[dict[str, Any]] = json.loads(
    (
        Path(__file__).parents[2] / "fixtures" / "message_surface_contract.json"
    ).read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case["name"])
def test_message_surface_contract(case: dict[str, Any]) -> None:
    data = case["data"]
    metadata = data.get("metadata")
    metadata_display = metadata.get("display") if isinstance(metadata, dict) else None
    display = (
        data.get("display") if data.get("display") is not None else metadata_display
    )

    assert (
        resolve_message_display(
            display=display,
            event_type=case["event_type"],
            message_type=data.get("message_type"),
            expect_response=data.get("expect_response") is True,
            visible=data.get("visible") is not False,
        )
        == case["surface"]
    )
