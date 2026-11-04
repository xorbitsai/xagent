from __future__ import annotations

import json
from typing import Any

import pytest

from xagent.core.agent.context import CompactConfig, ExecutionContext
from xagent.core.agent.context.enrichment import (
    TOP_LEVEL_USER_REQUEST_METADATA_KEY,
    TopLevelUserRequest,
    display_message_override,
    top_level_user_request,
)
from xagent.core.agent.pattern.dag.dag import DAGPattern

EXECUTION_REQUEST = "Summarize the email.\n[Connector context: contacto@example.es]"
CLEAN_REQUEST = "Summarize the email."


@pytest.mark.parametrize(
    ("metadata", "language_text", "display_state"),
    [
        pytest.param({}, EXECUTION_REQUEST, "missing", id="missing"),
        pytest.param(
            {"display_message": CLEAN_REQUEST}, CLEAN_REQUEST, "text", id="text"
        ),
        pytest.param({"display_message": ""}, "", "empty", id="blank"),
        pytest.param({"display_message": "  \n\t"}, "", "empty", id="whitespace"),
    ],
)
def test_top_level_request_preserves_display_tri_state(
    metadata: dict[str, Any], language_text: str, display_state: str
) -> None:
    context = ExecutionContext(metadata={"task": EXECUTION_REQUEST})
    context.add_user_message(EXECUTION_REQUEST, metadata=metadata)

    request = top_level_user_request(context)

    assert request == TopLevelUserRequest(
        execution_text=EXECUTION_REQUEST,
        language_text=language_text,
        display_state=display_state,
    )
    assert context.metadata[TOP_LEVEL_USER_REQUEST_METADATA_KEY] == {
        "execution_text": EXECUTION_REQUEST,
        "language_text": language_text,
        "display_state": display_state,
    }


@pytest.mark.parametrize("value", [None, 17, [], {}])
def test_direct_non_string_display_falls_back_to_execution_text(value: Any) -> None:
    context = ExecutionContext()
    context.add_user_message(
        EXECUTION_REQUEST,
        metadata={"display_message": value},
    )

    request = top_level_user_request(context)

    assert display_message_override({"display_message": value}) is None
    assert request.language_text == EXECUTION_REQUEST
    assert request.display_state == "missing"


def test_new_independent_request_replaces_stored_snapshot() -> None:
    context = ExecutionContext()
    context.add_user_message(
        EXECUTION_REQUEST,
        metadata={"display_message": CLEAN_REQUEST},
    )
    assert top_level_user_request(context).language_text == CLEAN_REQUEST

    follow_up = "Switch to Spanish now."
    context.add_user_message(
        f"{follow_up}\n[Connector context in English]",
        metadata={"display_message": follow_up},
    )

    request = top_level_user_request(context)
    assert request.execution_text == f"{follow_up}\n[Connector context in English]"
    assert request.language_text == follow_up
    assert request.display_state == "text"


@pytest.mark.parametrize(
    ("display_message", "language_text", "display_state"),
    [
        pytest.param(CLEAN_REQUEST, CLEAN_REQUEST, "text", id="text"),
        pytest.param("", "", "empty", id="blank"),
        pytest.param("  \n\t", "", "empty", id="whitespace"),
    ],
)
@pytest.mark.parametrize("compaction", ["summary", "truncate"])
@pytest.mark.parametrize("cold_restore", [False, True], ids=["live", "restored"])
def test_request_provenance_survives_compaction_and_restore(
    display_message: str,
    language_text: str,
    display_state: str,
    compaction: str,
    cold_restore: bool,
) -> None:
    context = ExecutionContext(metadata={"task": EXECUTION_REQUEST})
    context.add_user_message(
        EXECUTION_REQUEST,
        metadata={"display_message": display_message},
    )
    context.add_user_message(
        "DAG step instruction",
        metadata={"dag_step_id": "draft", "kind": "dag_step_instruction"},
    )

    if compaction == "summary":
        assert context.compact_with_llm_response({"content": "Work summary"}).compacted
    else:
        context.compact_config = CompactConfig(
            enabled=True,
            threshold=1,
            max_messages=1,
        )
        assert context.compact_if_needed().compacted
    if cold_restore:
        context = ExecutionContext.from_dict(context.to_dict())

    request = top_level_user_request(context)
    assert request.execution_text == EXECUTION_REQUEST
    assert request.language_text == language_text
    assert request.display_state == display_state


def test_llm_compaction_request_persists_provenance_before_history_changes() -> None:
    context = ExecutionContext(
        metadata={"task": EXECUTION_REQUEST},
        compact_config=CompactConfig(enabled=True, threshold=1, max_messages=1),
    )
    context.add_user_message(
        EXECUTION_REQUEST,
        metadata={"display_message": CLEAN_REQUEST},
    )

    assert context.build_llm_compact_request_if_needed() is not None
    context.messages = []

    assert top_level_user_request(context) == TopLevelUserRequest(
        execution_text=EXECUTION_REQUEST,
        language_text=CLEAN_REQUEST,
        display_state="text",
    )


def test_child_creation_snapshots_request_before_metadata_clone() -> None:
    root = ExecutionContext(metadata={"task": EXECUTION_REQUEST})
    root.add_user_message(
        EXECUTION_REQUEST,
        metadata={"display_message": CLEAN_REQUEST},
    )

    child = root.create_child_context(metadata={"dag_step_id": "draft"})
    child.messages = []

    assert top_level_user_request(child).language_text == CLEAN_REQUEST


def test_provenance_roundtrip_is_checkpoint_compatible() -> None:
    context = ExecutionContext(metadata={"task": EXECUTION_REQUEST})
    context.add_user_message(
        EXECUTION_REQUEST,
        metadata={"display_message": CLEAN_REQUEST},
    )
    expected = top_level_user_request(context)

    restored = ExecutionContext.from_dict(json.loads(json.dumps(context.to_dict())))

    assert top_level_user_request(restored) == expected


@pytest.mark.parametrize(
    ("display_message", "language_text", "display_state"),
    [
        pytest.param(CLEAN_REQUEST, CLEAN_REQUEST, "text", id="text"),
        pytest.param("", "", "empty", id="blank"),
        pytest.param("  \n\t", "", "empty", id="whitespace"),
    ],
)
def test_legacy_restored_child_hydrates_provenance_from_root(
    display_message: str,
    language_text: str,
    display_state: str,
) -> None:
    root = ExecutionContext(metadata={"task": EXECUTION_REQUEST})
    root.add_user_message(
        EXECUTION_REQUEST,
        metadata={"display_message": display_message},
    )
    root = ExecutionContext.from_dict(root.to_dict())
    child = ExecutionContext(
        metadata={"task": EXECUTION_REQUEST, "dag_step_id": "draft"}
    )
    child.add_user_message(
        "DAG step instruction",
        metadata={"dag_step_id": "draft", "kind": "dag_step_instruction"},
    )

    DAGPattern._refresh_restored_step_runtime_metadata(child, root)

    request = top_level_user_request(child)
    assert request.execution_text == EXECUTION_REQUEST
    assert request.language_text == language_text
    assert request.display_state == display_state


@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        {},
        {"execution_text": 1, "language_text": "x", "display_state": "text"},
        {
            "execution_text": "x",
            "language_text": "x",
            "display_state": "unknown",
        },
    ],
)
def test_invalid_legacy_child_snapshot_is_hydrated(snapshot: Any) -> None:
    root = ExecutionContext()
    root.add_user_message(
        EXECUTION_REQUEST, metadata={"display_message": CLEAN_REQUEST}
    )
    child = ExecutionContext(
        metadata={
            "dag_step_id": "draft",
            TOP_LEVEL_USER_REQUEST_METADATA_KEY: snapshot,
        }
    )

    DAGPattern._refresh_restored_step_runtime_metadata(child, root)

    assert top_level_user_request(child).language_text == CLEAN_REQUEST


def test_valid_child_snapshot_wins_over_root_hydration() -> None:
    root = ExecutionContext()
    root.add_user_message(
        EXECUTION_REQUEST, metadata={"display_message": CLEAN_REQUEST}
    )
    child = ExecutionContext(
        metadata={
            "dag_step_id": "draft",
            TOP_LEVEL_USER_REQUEST_METADATA_KEY: {
                "execution_text": "Child execution request",
                "language_text": "Child language request",
                "display_state": "text",
            },
        }
    )

    DAGPattern._refresh_restored_step_runtime_metadata(child, root)

    assert top_level_user_request(child).language_text == "Child language request"


def test_legacy_waiting_child_uses_shared_restore_hydration_seam() -> None:
    root = ExecutionContext(execution_id="waiting-root")
    root.add_user_message(
        EXECUTION_REQUEST, metadata={"display_message": CLEAN_REQUEST}
    )
    child = root.create_child_context(
        execution_id="waiting-child", metadata={"dag_step_id": "confirm"}
    )
    child.messages = []
    child.metadata.pop(TOP_LEVEL_USER_REQUEST_METADATA_KEY)
    pattern = DAGPattern(lambda **_: None)
    pattern.status = "waiting_for_user"
    pattern.active_step_id = "confirm"
    pattern.active_step_ids = ["confirm"]
    pattern.active_step_contexts = {"confirm": child.to_dict()}
    pattern.active_step_pattern_states = {
        "confirm": {
            "status": "waiting_for_user",
            "waiting_for_user_request": {"message": "Which date?"},
        }
    }
    pattern.planned_user_message_count = 1
    root.add_user_message("Friday")

    assert pattern._forward_user_response_to_waiting_step(root)

    restored_child = ExecutionContext.from_dict(pattern.active_step_contexts["confirm"])
    request = top_level_user_request(restored_child)
    assert request.execution_text == EXECUTION_REQUEST
    assert request.language_text == CLEAN_REQUEST
    assert request.display_state == "text"


def test_persisting_provenance_does_not_change_rendered_prompt() -> None:
    context = ExecutionContext()
    context.add_user_message(EXECUTION_REQUEST, metadata={"display_message": ""})
    before = context.get_messages_for_llm()

    top_level_user_request(context)

    assert context.get_messages_for_llm() == before
