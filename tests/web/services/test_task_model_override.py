"""A caller-selected model replaces every model the Agent overlay resolved."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.agent.service import AgentService
from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.basic.call_boundary import UnavailableVisionModel
from xagent.web.models.task import TaskStatus
from xagent.web.models.user import User
from xagent.web.services.agent_service_manager import (
    AgentServiceManager,
    AgentServiceMemoryPolicy,
)
from xagent.web.services.llm_utils import AgentRuntimeFields
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskModelOverride,
    TaskModelOverrideError,
    TaskSetupSnapshot,
    _TaskFields,
    apply_task_model_override,
)


class NamedLLM(BaseLLM):
    def __init__(self, name: str, abilities: list[str] | None = None) -> None:
        self._name = name
        self._abilities = abilities or ["chat", "tool_calling"]

    @property
    def abilities(self) -> List[str]:
        return self._abilities

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(self, messages, **kwargs):  # type: ignore[override]
        raise AssertionError(f"{self._name} must not be called")


# The models the Agent overlay resolved -- an explicit platform model saved
# on the agent row.
OVERLAY_GENERAL = NamedLLM("platform-general")
OVERLAY_FAST = NamedLLM("platform-fast")
OVERLAY_VISION = NamedLLM("platform-vision", ["chat", "vision"])
OVERLAY_COMPACT = NamedLLM("platform-compact")


def _snapshot(tool_categories: list[str] | None) -> TaskSetupSnapshot:
    agent_config: dict[str, Any] = {
        "llms": [OVERLAY_GENERAL, OVERLAY_FAST, OVERLAY_VISION, OVERLAY_COMPACT],
        "saved_model_ids": {"general": "platform/x"},
        "saved_model_descriptors": {},
        "execution_mode": "balanced",
        "instructions": "Be helpful.",
        "skills": [],
        "knowledge_bases": [],
        "tool_categories": tool_categories,
    }
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=42,
            user_id=1,
            status=TaskStatus.PENDING,
            source="external",
            agent_id=7,
            agent_config={},
            model_name=None,
            compact_model_name=None,
            execution_mode="balanced",
            agent_type="standard",
        ),
        runtime_user=RuntimeUserFields(id=1, is_admin=False),
        has_reconstructable_history=False,
        task_pattern="react",
        task_llm=OVERLAY_GENERAL,
        task_fast_llm=OVERLAY_FAST,
        task_vision_llm=OVERLAY_VISION,
        task_compact_llm=OVERLAY_COMPACT,
        agent=AgentRuntimeFields(
            id=7,
            name="Assistant",
            status="published",
            instructions="Be helpful.",
            agent_creator_user_id=1,
        ),
        agent_config=agent_config,
        excluded_agent_id=None,
    )


class TestApply:
    def test_every_model_slot_takes_the_override(self):
        selected = NamedLLM("selected", ["chat", "tool_calling", "vision"])
        coordinate = {"scope_id": "root", "event_id": "history", "sequence": 4}
        snapshot = replace(
            _snapshot(["basic", "image"]), conversation_event_watermark=coordinate
        )
        applied = apply_task_model_override(
            snapshot,
            TaskModelOverride(llm=selected, vision_llm=selected),
        )
        assert applied.conversation_event_watermark == coordinate
        assert applied.model_override is not None
        assert applied.task_llm is selected
        assert applied.task_fast_llm is selected
        assert applied.task_compact_llm is selected
        assert applied.task_vision_llm is selected
        assert applied.agent_config["tool_categories"] == ["basic", "image"]

    def test_no_vision_model_means_vision_is_unavailable(self):
        selected = NamedLLM("selected")
        applied = apply_task_model_override(
            _snapshot(["basic"]), TaskModelOverride(llm=selected)
        )
        assert isinstance(applied.task_vision_llm, UnavailableVisionModel)
        assert applied.task_vision_llm is not OVERLAY_VISION

    def test_excluded_categories_leave_the_selection(self):
        selected = NamedLLM("selected")
        original = _snapshot(["web_search", "image", "video", "basic"])
        applied = apply_task_model_override(
            original,
            TaskModelOverride(
                llm=selected, excluded_tool_categories=frozenset({"image", "video"})
            ),
        )
        assert applied.agent_config["tool_categories"] == ["web_search", "basic"]
        # The caller's snapshot is not mutated.
        assert original.agent_config["tool_categories"] == [
            "web_search",
            "image",
            "video",
            "basic",
        ]

    def test_a_missing_vision_model_takes_vision_tools_out_of_the_selection(self):
        applied = apply_task_model_override(
            _snapshot(["basic", "vision", "image"]),
            TaskModelOverride(llm=NamedLLM("selected")),
        )
        assert applied.agent_config["tool_categories"] == ["basic", "image"]
        assert isinstance(applied.task_vision_llm, UnavailableVisionModel)

    def test_a_vision_model_keeps_the_vision_tools(self):
        selected = NamedLLM("selected", ["chat", "vision"])
        applied = apply_task_model_override(
            _snapshot(["basic", "vision"]),
            TaskModelOverride(llm=selected, vision_llm=selected),
        )
        assert applied.agent_config["tool_categories"] == ["basic", "vision"]

    def test_an_unrestricted_selection_keeps_the_stand_in(self):
        # Nothing to remove the vision category from: its tools are built on
        # the stand-in, which refuses, and nothing falls back to another model.
        original = _snapshot(None)
        applied = apply_task_model_override(
            original, TaskModelOverride(llm=NamedLLM("selected"))
        )
        assert applied.agent_config["tool_categories"] is None
        assert isinstance(applied.task_vision_llm, UnavailableVisionModel)
        assert applied.task_vision_llm  # truthy: the tool config's fallback stays shut

    def test_the_snapshot_records_the_override(self):
        override = TaskModelOverride(llm=NamedLLM("selected"))
        assert _snapshot(["basic"]).model_override is None
        assert (
            apply_task_model_override(_snapshot(["basic"]), override).model_override
            is override
        )

    def test_exclusion_needs_an_explicit_selection(self):
        with pytest.raises(TaskModelOverrideError):
            apply_task_model_override(
                _snapshot(None),
                TaskModelOverride(
                    llm=NamedLLM("selected"),
                    excluded_tool_categories=frozenset({"image"}),
                ),
            )


class TestExcludedCategories:
    def test_any_collection_becomes_a_frozenset(self):
        override = TaskModelOverride(
            llm=NamedLLM("selected"),
            excluded_tool_categories=["image", "video"],  # type: ignore[arg-type]
        )
        assert override.excluded_tool_categories == frozenset({"image", "video"})

    @pytest.mark.parametrize("value", ["image", b"image"])
    def test_a_single_string_is_refused(self, value):
        # A string would match by substring: "video" in "videos" and "a" in
        # "image" would both be true.
        with pytest.raises(TypeError):
            TaskModelOverride(llm=NamedLLM("selected"), excluded_tool_categories=value)  # type: ignore[arg-type]

    def test_the_models_must_be_models(self):
        with pytest.raises(TypeError):
            TaskModelOverride(llm=None)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            TaskModelOverride(llm=NamedLLM("selected"), vision_llm="vision")  # type: ignore[arg-type]

    def test_names_must_be_strings(self):
        with pytest.raises(TypeError):
            TaskModelOverride(llm=NamedLLM("selected"), excluded_tool_categories={1})  # type: ignore[arg-type]


def test_the_stand_in_is_not_reported_as_a_configured_vision_model() -> None:
    def status(vision_llm: Any) -> dict[str, Any]:
        service = SimpleNamespace(
            name="svc",
            llm=NamedLLM("selected"),
            fast_llm=None,
            vision_llm=vision_llm,
            compact_llm=None,
            tools=[],
            memory=object(),
            memory_enabled=False,
            memory_available=True,
            memory_availability_reason=None,
            _execution_type=lambda: "react",
        )
        return AgentService.get_status(service)  # type: ignore[arg-type]

    assert status(UnavailableVisionModel())["vision_llm_configured"] is False
    assert status(NamedLLM("seer", ["vision"]))["vision_llm_configured"] is True
    assert status(None)["vision_llm_configured"] is False


class _Builds:
    """Drives ``get_agent_for_task`` with the build stubbed; records each build."""

    def __init__(self) -> None:
        self.manager = AgentServiceManager()
        self.services: list[Any] = []
        self.tool_calls: list[dict[str, Any]] = []

    async def get(
        self,
        snapshot: TaskSetupSnapshot | None,
        *,
        loaded: TaskSetupSnapshot | None = None,
    ) -> Any:
        """Look the task up with ``snapshot`` (``None``: no snapshot, like a
        control path); ``loaded`` is what a snapshot load would read."""
        builds = self

        class RecordingService(MagicMock):
            def _get_child_mock(self, **kw: Any) -> MagicMock:
                return MagicMock(**kw)

            def get_execution_status(self, execution_id: str) -> dict[str, Any]:
                return dict(self.execution_status)

            def revoke_memory(self, **kwargs: Any) -> None:
                AgentService.revoke_memory(self, **kwargs)

            def __init__(self, **kwargs: Any) -> None:
                super().__init__()
                self.execution_status = {"is_running": False, "is_resumable": False}
                self.built = kwargs
                self.llm = kwargs["llm"]
                self.fast_llm = kwargs["fast_llm"]
                self.vision_llm = kwargs["vision_llm"]
                self.compact_llm = kwargs["compact_llm"]
                self.memory = kwargs["memory"]
                self.memory_enabled = kwargs["memory_enabled"]
                self.memory_available = kwargs["memory_available"]
                self.memory_availability_reason = kwargs["memory_availability_reason"]
                self.execution_metadata = dict(kwargs["execution_metadata"])
                self.agent = SimpleNamespace(memory_store=self.memory)
                self._execution_adapter = None
                self.workspace = None
                builds.services.append(self)

        async def create_tools(*args: Any, **kwargs: Any) -> Any:
            builds.tool_calls.append(kwargs)
            return [], MagicMock()

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = MagicMock(
            status=TaskStatus.PENDING
        )
        with ExitStack() as stack:
            for target in (
                patch(
                    "xagent.web.services.agent_service_manager.load_task_setup_snapshot_sync",
                    return_value=loaded or snapshot or _snapshot(["basic"]),
                ),
                patch.object(self.manager, "_load_persisted_conversation_history"),
                patch.object(
                    self.manager, "_load_persisted_execution_context", new=AsyncMock()
                ),
                patch(
                    "xagent.web.services.agent_service_manager.create_task_tracer",
                    return_value=MagicMock(),
                ),
                patch(
                    "xagent.web.services.agent_service_manager.create_default_tools",
                    new=create_tools,
                ),
                patch(
                    "xagent.web.sandbox_manager.get_sandbox_manager", return_value=None
                ),
                patch(
                    "xagent.web.services.agent_service_manager.AgentService",
                    new=RecordingService,
                ),
            ):
                stack.enter_context(target)
            return await self.manager.get_agent_for_task(
                task_id=42,
                db=db,
                user=User(id=1, username="u", password_hash="h", is_admin=False),
                task_setup_snapshot=snapshot,
            )


def _override(name: str) -> TaskModelOverride:
    return TaskModelOverride(llm=NamedLLM(name))


class TestACachedTask:
    """The override holds on every turn, not only on the turn that built it."""

    async def test_a_new_override_rebuilds_the_cached_service(self):
        builds = _Builds()
        first = _override("first-turn")
        second = _override("second-turn")
        one = await builds.get(apply_task_model_override(_snapshot(["basic"]), first))
        two = await builds.get(apply_task_model_override(_snapshot(["basic"]), second))

        assert two is not one
        assert two.llm is second.llm and two.compact_llm is second.llm
        assert two.fast_llm is second.llm
        assert isinstance(two.vision_llm, UnavailableVisionModel)
        # The tools were rebuilt with the new model too.
        assert [call["llm"] for call in builds.tool_calls] == [first.llm, second.llm]

    async def test_the_same_override_reuses_the_cached_service(self):
        builds = _Builds()
        override = _override("selected")
        snapshot = apply_task_model_override(_snapshot(["basic"]), override)
        one = await builds.get(snapshot)
        two = await builds.get(snapshot)
        assert two is one
        assert len(builds.services) == 1

    async def test_the_same_override_still_reconciles_revoked_memory(self):
        builds = _Builds()
        override = _override("selected")
        snapshot = apply_task_model_override(_snapshot(["basic"]), override)
        ready = AgentServiceMemoryPolicy(
            memory=InMemoryMemoryStore(), memory_enabled=True
        )
        blocked = AgentServiceMemoryPolicy(
            memory=InMemoryMemoryStore(),
            memory_enabled=False,
            memory_available=False,
            memory_availability_reason="restart_required",
        )
        with patch(
            "xagent.web.services.agent_service_manager.resolve_agent_service_memory_policy_async",
            new=AsyncMock(side_effect=[ready, blocked]),
        ) as resolve_memory:
            one = await builds.get(snapshot)
            two = await builds.get(snapshot)

        assert two is one
        assert len(builds.services) == len(builds.tool_calls) == 1
        assert two.llm is two.fast_llm is two.compact_llm is override.llm
        assert isinstance(two.vision_llm, UnavailableVisionModel)
        assert two.memory is blocked.memory
        assert two.agent.memory_store is blocked.memory
        assert two.memory_enabled is False and two.memory_available is False
        assert (
            two.execution_metadata["memory_availability_reason"] == "restart_required"
        )
        assert builds.manager._cached_model_override(42) is override
        assert resolve_memory.await_count == 2

    async def test_a_new_override_builds_with_the_current_memory_policy(self):
        builds = _Builds()
        first, second = _override("first-turn"), _override("second-turn")
        ready = AgentServiceMemoryPolicy(
            memory=InMemoryMemoryStore(), memory_enabled=True
        )
        blocked = AgentServiceMemoryPolicy(
            memory=InMemoryMemoryStore(),
            memory_enabled=False,
            memory_available=False,
            memory_availability_reason="restart_required",
        )
        with patch(
            "xagent.web.services.agent_service_manager.resolve_agent_service_memory_policy_async",
            new=AsyncMock(side_effect=[ready, blocked]),
        ) as resolve_memory:
            one = await builds.get(
                apply_task_model_override(_snapshot(["basic"]), first)
            )
            two = await builds.get(
                apply_task_model_override(_snapshot(["basic"]), second)
            )

        assert two is not one
        assert two.llm is two.fast_llm is two.compact_llm is second.llm
        assert isinstance(two.vision_llm, UnavailableVisionModel)
        assert [call["llm"] for call in builds.tool_calls] == [first.llm, second.llm]
        assert two.memory is blocked.memory
        assert two.memory_enabled is False and two.memory_available is False
        assert (
            two.execution_metadata["memory_availability_reason"] == "restart_required"
        )
        assert one.memory is ready.memory and one.memory_enabled is True
        assert builds.manager._cached_model_override(42) is second
        assert resolve_memory.await_count == 2

    async def test_a_request_without_the_override_rebuilds_on_the_task_models(self):
        builds = _Builds()
        override = _override("selected")
        await builds.get(apply_task_model_override(_snapshot(["basic"]), override))
        plain = await builds.get(_snapshot(["basic"]))
        assert plain.llm is OVERLAY_GENERAL
        assert plain.compact_llm is OVERLAY_COMPACT
        assert len(builds.services) == 2

    async def test_an_override_on_a_service_built_without_one_rebuilds_it(self):
        builds = _Builds()
        await builds.get(_snapshot(["basic"]))
        override = _override("selected")
        applied = await builds.get(
            apply_task_model_override(_snapshot(["basic"]), override)
        )
        assert applied.llm is override.llm
        assert len(builds.services) == 2

    async def test_a_waiting_task_is_reconstructed_on_the_new_override(self):
        # A continuation of a waiting task rebuilds from its history rather
        # than from scratch; that path must take the new override too.
        builds = _Builds()
        reconstructed: list[Any] = []

        async def reconstruct(task_id, db, *, task_setup_snapshot, **kwargs):
            service = MagicMock()
            service.llm = task_setup_snapshot.task_llm
            service.compact_llm = task_setup_snapshot.task_compact_llm
            builds.manager._agents[task_id] = service
            reconstructed.append(service)

        def waiting(override: TaskModelOverride) -> TaskSetupSnapshot:
            base = _snapshot(["basic"])
            return apply_task_model_override(
                replace(
                    base,
                    task=replace(base.task, status=TaskStatus.WAITING_FOR_USER),
                    has_reconstructable_history=True,
                ),
                override,
            )

        first, second = _override("first-turn"), _override("second-turn")
        with patch.object(
            builds.manager, "_reconstruct_agent_from_history", new=reconstruct
        ):
            one = await builds.get(waiting(first))
            two = await builds.get(waiting(second))
            again = waiting(second)
            three = await builds.get(again)

        assert one.llm is first.llm
        assert two is not one and two.llm is second.llm
        assert two.compact_llm is second.llm
        # A snapshot carrying the same override object reuses the rebuilt one.
        assert three is two
        assert len(reconstructed) == 2

    @pytest.mark.parametrize(
        "status",
        [
            {"is_running": True, "is_resumable": False},
            {"is_running": False, "is_resumable": True},
        ],
    )
    async def test_a_lookup_without_an_override_reaches_the_run_in_progress(
        self, status
    ):
        # Pause, resume and a reply injected into a waiting run look the task
        # up without the override; they must reach the run it is executing.
        builds = _Builds()
        override = _override("selected")
        service = await builds.get(
            apply_task_model_override(_snapshot(["basic"]), override)
        )
        service.execution_status = status
        assert await builds.get(_snapshot(["basic"])) is service
        assert await builds.get(None) is service
        assert len(builds.services) == 1
        # Still known as an override service: its models stay in place.
        with patch(
            "xagent.web.services.agent_service_manager.resolve_llms_from_names",
            side_effect=AssertionError("must not resolve the task's models"),
        ):
            builds.manager.set_task_llms(42, ["platform/x"], db=MagicMock())
        assert service.llm is override.llm

    async def test_a_new_override_rebuilds_even_while_a_run_is_waiting(self):
        builds = _Builds()
        first, second = _override("first-turn"), _override("second-turn")
        service = await builds.get(
            apply_task_model_override(_snapshot(["basic"]), first)
        )
        service.execution_status = {"is_running": False, "is_resumable": True}
        rebuilt = await builds.get(
            apply_task_model_override(_snapshot(["basic"]), second)
        )
        assert rebuilt is not service and rebuilt.llm is second.llm

    async def test_an_unknown_execution_status_keeps_control_lookup_reachable(self):
        builds = _Builds()
        first, second = _override("first-turn"), _override("second-turn")
        service = await builds.get(
            apply_task_model_override(_snapshot(["basic"]), first)
        )
        with patch.object(
            service, "get_execution_status", side_effect=RuntimeError("unavailable")
        ):
            assert await builds.get(None) is service
            assert builds.manager._cached_model_override(42) is first
            # An explicit new run must still bind its own model.
            rebuilt = await builds.get(
                apply_task_model_override(_snapshot(["basic"]), second)
            )
        assert rebuilt is not service and rebuilt.llm is second.llm
        assert rebuilt.compact_llm is second.llm

    @pytest.mark.parametrize("status", [None, {}])
    async def test_an_empty_execution_status_rebuilds_on_the_task_models(self, status):
        builds = _Builds()
        service = await builds.get(
            apply_task_model_override(_snapshot(["basic"]), _override("selected"))
        )
        with patch.object(service, "get_execution_status", return_value=status):
            rebuilt = await builds.get(None)
        assert rebuilt is not service
        assert rebuilt.llm is OVERLAY_GENERAL
        assert rebuilt.compact_llm is OVERLAY_COMPACT
        assert builds.manager._cached_model_override(42) is None

    async def test_set_task_llms_leaves_an_override_service_alone(self):
        builds = _Builds()
        override = _override("selected")
        service = await builds.get(
            apply_task_model_override(_snapshot(["basic"]), override)
        )
        with patch(
            "xagent.web.services.agent_service_manager.resolve_llms_from_names",
            side_effect=AssertionError("must not resolve the task's models"),
        ):
            builds.manager.set_task_llms(42, ["platform/x"], db=MagicMock())
        assert service.llm is override.llm
        assert service.compact_llm is override.llm

    async def test_set_task_llms_still_updates_a_plain_service(self):
        builds = _Builds()
        service = await builds.get(_snapshot(["basic"]))
        replacement = NamedLLM("replacement")
        with patch(
            "xagent.web.services.agent_service_manager.resolve_llms_from_names",
            return_value=(replacement, None, None, replacement),
        ):
            builds.manager.set_task_llms(42, ["platform/y"], db=MagicMock())
        assert service.llm is replacement


@pytest.mark.asyncio
async def test_the_built_service_and_its_tools_use_only_the_override() -> None:
    selected = NamedLLM("selected")
    snapshot = apply_task_model_override(
        _snapshot(["basic", "image", "vision"]),
        TaskModelOverride(llm=selected, excluded_tool_categories=frozenset({"image"})),
    )
    manager = AgentServiceManager()
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        status=TaskStatus.PENDING
    )
    create_tools = AsyncMock(return_value=([], MagicMock()))
    built: dict[str, Any] = {}

    class RecordingService(MagicMock):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            built.update(kwargs)
            self.workspace = None

    with (
        patch(
            "xagent.web.services.agent_service_manager.load_task_setup_snapshot_sync",
            return_value=snapshot,
        ),
        patch.object(manager, "_load_persisted_conversation_history"),
        patch.object(manager, "_load_persisted_execution_context", new=AsyncMock()),
        patch(
            "xagent.web.services.agent_service_manager.create_task_tracer",
            return_value=MagicMock(),
        ),
        patch(
            "xagent.web.services.agent_service_manager.create_default_tools",
            new=create_tools,
        ),
        patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
        patch(
            "xagent.web.services.agent_service_manager.AgentService",
            new=RecordingService,
        ),
    ):
        await manager.get_agent_for_task(
            task_id=42,
            db=db,
            user=User(id=1, username="u", password_hash="h", is_admin=False),
            task_setup_snapshot=snapshot,
        )

    tool_kwargs = create_tools.await_args.kwargs
    assert tool_kwargs["llm"] is selected
    assert isinstance(tool_kwargs["vision_model"], UnavailableVisionModel)
    spec = tool_kwargs["tool_selection_spec"]
    assert "image" not in spec.categories
    # No vision model: no vision tools either.
    assert "vision" not in spec.categories
    assert "basic" in spec.categories
    assert built["llm"] is selected
    assert built["fast_llm"] is selected
    assert built["compact_llm"] is selected
    assert isinstance(built["vision_llm"], UnavailableVisionModel)
    overlay = {OVERLAY_GENERAL, OVERLAY_FAST, OVERLAY_VISION, OVERLAY_COMPACT}
    assert not overlay & {
        built["llm"],
        built["fast_llm"],
        built["compact_llm"],
        built["vision_llm"],
        tool_kwargs["llm"],
        tool_kwargs["vision_model"],
    }
