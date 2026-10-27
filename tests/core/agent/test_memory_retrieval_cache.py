from __future__ import annotations

import json

import pytest

from xagent.core.agent import DAGPattern, ExecutionContext, ExecutionPlan, PlanStep
from xagent.core.agent.context.memory_retrieval import (
    MEMORY_CONTEXT_METADATA_KEY,
    MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY,
    RETRIEVED_MEMORIES_METADATA_KEY,
    reconcile_execution_memory_retrieval_cache,
    reconcile_memory_retrieval_metadata,
)


def _memory_metadata(scope: str | None = "scope-a") -> dict[str, object]:
    metadata: dict[str, object] = {
        RETRIEVED_MEMORIES_METADATA_KEY: {
            "react": [{"content": "SCOPE_A_PRIVATE_NOTE"}]
        },
        MEMORY_CONTEXT_METADATA_KEY: "SCOPE_A_PRIVATE_NOTE",
        "selected_skill_context": "Keep the selected skill",
        "output_language": "English",
        "execution_budget": {"remaining": 7},
    }
    if scope is not None:
        metadata[MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY] = scope
    return metadata


def test_same_scope_preserves_automatic_retrieval_cache() -> None:
    metadata = _memory_metadata()

    changed = reconcile_memory_retrieval_metadata(
        metadata,
        trusted_scope_fingerprint="scope-a",
    )

    assert not changed
    assert metadata[MEMORY_CONTEXT_METADATA_KEY] == "SCOPE_A_PRIVATE_NOTE"
    assert RETRIEVED_MEMORIES_METADATA_KEY in metadata


@pytest.mark.parametrize(
    ("stored_scope", "trusted_scope"),
    [
        ("scope-a", "scope-b"),
        ("scope-a", None),
        (None, "scope-b"),
        (None, None),
    ],
)
def test_changed_denied_and_legacy_scopes_clear_only_automatic_retrieval(
    stored_scope: str | None,
    trusted_scope: str | None,
) -> None:
    metadata = _memory_metadata(stored_scope)

    assert reconcile_memory_retrieval_metadata(
        metadata,
        trusted_scope_fingerprint=trusted_scope,
    )

    assert RETRIEVED_MEMORIES_METADATA_KEY not in metadata
    assert MEMORY_CONTEXT_METADATA_KEY not in metadata
    assert metadata["selected_skill_context"] == "Keep the selected skill"
    assert metadata["output_language"] == "English"
    assert metadata["execution_budget"] == {"remaining": 7}
    if trusted_scope is None:
        assert MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY not in metadata
    else:
        assert (
            metadata[MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY] == trusted_scope
        )


def test_empty_trusted_scope_fingerprint_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty or None"):
        reconcile_memory_retrieval_metadata(
            {},
            trusted_scope_fingerprint="",
        )


@pytest.mark.parametrize("trusted_scope", ["scope-b", None])
def test_root_and_restored_dag_children_drop_stale_memory_without_other_state_loss(
    trusted_scope: str | None,
) -> None:
    root = ExecutionContext(
        execution_id="root",
        system_prompt="Base prompt.",
        metadata=_memory_metadata(),
        execution_budget={"token_limit": 1000},
    )
    root.add_user_message("Use my current memory")
    root.add_assistant_message("I will check it")

    pattern = DAGPattern(lambda **_: None)
    pattern.plan = ExecutionPlan(
        [
            PlanStep(id="first", task="First step"),
            PlanStep(id="second", task="Second step", dependencies=["first"]),
        ]
    )
    pattern.status = "waiting_for_user"
    pattern.active_step_ids = ["first", "second"]
    for step_id in pattern.active_step_ids:
        child = root.create_child_context(
            metadata={
                "dag_step_id": step_id,
                "dag_step_name": f"{step_id} name",
            }
        )
        child.add_user_message(f"Run {step_id}")
        pattern.active_step_contexts[step_id] = child.to_dict()
    pattern.step_results = {"completed": {"success": True}}

    checkpoint = json.loads(
        json.dumps(
            {
                "context": root.to_dict(),
                "pattern_state": pattern.get_state(),
            }
        )
    )
    resumed_root = ExecutionContext.from_dict(checkpoint["context"])
    resumed_pattern = DAGPattern(lambda **_: None)
    resumed_pattern.load_state(checkpoint["pattern_state"])
    original_messages = [message.to_dict() for message in resumed_root.messages]
    original_plan = resumed_pattern.plan.to_dict() if resumed_pattern.plan else None
    original_step_results = dict(resumed_pattern.step_results)

    changed_count = reconcile_execution_memory_retrieval_cache(
        resumed_root,
        resumed_pattern.active_step_contexts,
        trusted_scope_fingerprint=trusted_scope,
    )

    assert changed_count == 3
    for context in [
        resumed_root,
        *(
            ExecutionContext.from_dict(payload)
            for payload in resumed_pattern.active_step_contexts.values()
        ),
    ]:
        assert RETRIEVED_MEMORIES_METADATA_KEY not in context.metadata
        assert MEMORY_CONTEXT_METADATA_KEY not in context.metadata
        assert "SCOPE_A_PRIVATE_NOTE" not in context._system_context()
        assert context.metadata["selected_skill_context"] == "Keep the selected skill"
        assert context.metadata["output_language"] == "English"
        if trusted_scope is None:
            assert (
                MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY not in context.metadata
            )
        else:
            assert (
                context.metadata[MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY]
                == trusted_scope
            )
    assert [message.to_dict() for message in resumed_root.messages] == original_messages
    assert resumed_root.execution_budget == {"token_limit": 1000}
    assert resumed_pattern.plan is not None
    assert resumed_pattern.plan.to_dict() == original_plan
    assert resumed_pattern.step_results == original_step_results
    assert resumed_pattern.status == "waiting_for_user"

    assert (
        reconcile_execution_memory_retrieval_cache(
            resumed_root,
            resumed_pattern.active_step_contexts,
            trusted_scope_fingerprint=trusted_scope,
        )
        == 0
    )


def test_same_scope_root_and_children_keep_only_current_scope_memory() -> None:
    root = ExecutionContext(metadata=_memory_metadata("scope-b"))
    root.metadata[MEMORY_CONTEXT_METADATA_KEY] = "SCOPE_B_NOTE"
    root.metadata[RETRIEVED_MEMORIES_METADATA_KEY] = {
        "react": [{"content": "SCOPE_B_NOTE"}]
    }
    current_child = root.create_child_context(metadata={"dag_step_id": "current"})
    stale_child = root.create_child_context(metadata={"dag_step_id": "stale"})
    stale_child.metadata[MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY] = "scope-a"
    stale_child.metadata[MEMORY_CONTEXT_METADATA_KEY] = "SCOPE_A_PRIVATE_NOTE"
    legacy_child = root.create_child_context(metadata={"dag_step_id": "legacy"})
    legacy_child.metadata.pop(MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY)
    children = {
        "current": current_child.to_dict(),
        "stale": stale_child.to_dict(),
        "legacy": legacy_child.to_dict(),
    }

    changed_count = reconcile_execution_memory_retrieval_cache(
        root,
        children,
        trusted_scope_fingerprint="scope-b",
    )

    assert changed_count == 2
    assert root.metadata[MEMORY_CONTEXT_METADATA_KEY] == "SCOPE_B_NOTE"
    restored_current = ExecutionContext.from_dict(children["current"])
    assert restored_current.metadata[MEMORY_CONTEXT_METADATA_KEY] == "SCOPE_B_NOTE"
    for step_id in ("stale", "legacy"):
        restored = ExecutionContext.from_dict(children[step_id])
        assert MEMORY_CONTEXT_METADATA_KEY not in restored.metadata
        assert RETRIEVED_MEMORIES_METADATA_KEY not in restored.metadata
        assert (
            restored.metadata[MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY]
            == "scope-b"
        )

    new_child = root.create_child_context(metadata={"dag_step_id": "new"})
    assert new_child.metadata[MEMORY_CONTEXT_METADATA_KEY] == "SCOPE_B_NOTE"
