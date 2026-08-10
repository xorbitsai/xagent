"""Deletion dispatches runtime-extension cleanup by per-task binding only."""

from __future__ import annotations

import logging

import pytest
from fastapi import HTTPException

from xagent.core.execution_scope import (
    EXECUTION_SCOPE_AGENT_CONFIG_KEY,
    execution_scope_from_agent_config,
)
from xagent.core.task_runtime import TaskRuntimeContribution
from xagent.web.api.admin_users import delete_user
from xagent.web.api.chat import create_task, delete_task
from xagent.web.models.task import Task
from xagent.web.models.user import User
from xagent.web.schemas.chat import TaskCreateRequest
from xagent.web.services.task_runtime import (
    SELECTED_FILE_IDS_AGENT_CONFIG_KEY,
    TASK_RUNTIME_BINDINGS_AGENT_CONFIG_KEY,
    agent_config_with_task_extension_bindings,
    register_task_extension,
    task_extension_bindings_from_agent_config,
    unregister_task_extension,
)

from .conftest import _admin_headers, _direct_db_session, _register_second_user

pytestmark = pytest.mark.usefixtures("_test_db")


def _bound_to(*extensions: str) -> dict:
    return agent_config_with_task_extension_bindings({}, extensions)


class _Provider:
    """Minimal provider that records dispatch and can fail on delete."""

    def __init__(self, *, fail_delete: bool = False) -> None:
        self.fail_delete = fail_delete
        self.deleted_task_ids: list[int] = []

    async def on_task_created(self, context, configuration) -> None:
        return None

    async def build_runtime(self, context) -> TaskRuntimeContribution:
        return TaskRuntimeContribution()

    async def public_metadata(self, context) -> dict:
        return {}

    async def on_task_deleted(self, context) -> None:
        self.deleted_task_ids.append(int(context.task_id))
        if self.fail_delete:
            raise RuntimeError("provider is down")


@pytest.fixture
def registered() -> list[str]:
    names: list[str] = []
    yield names
    for name in names:
        unregister_task_extension(name)


def _register(name: str, provider: _Provider, registered: list[str]) -> _Provider:
    register_task_extension(name, provider)
    registered.append(name)
    return provider


@pytest.mark.asyncio
async def test_task_create_persists_inline_preview_agent_without_top_level_agent_id() -> (
    None
):
    _admin_headers()
    _register_second_user("preview-owner", "previewpass1")
    db = _direct_db_session()
    try:
        owner = db.query(User).filter(User.username == "preview-owner").one()
        created = await create_task(
            TaskCreateRequest(
                title="inline preview",
                description="inline preview",
                agent_id=None,
                agent_config={
                    "instructions": "preview",
                    "knowledge_bases": [],
                    "skills": [],
                    "tool_categories": ["ssh"],
                    "is_preview": True,
                    "preview_agent_id": 41,
                },
                is_visible=False,
            ),
            db=db,
            user=owner,
        )

        db.expire_all()
        task = db.query(Task).filter(Task.id == int(created.task_id)).one()
        assert task.agent_id is None
        assert task.agent_config["preview_agent_id"] == 41
        assert task.agent_config["tool_categories"] == ["ssh"]
    finally:
        db.close()


# ===== reserved agent_config keys are server-owned, never client-supplied =====


@pytest.mark.asyncio
async def test_task_create_drops_a_client_forged_binding_record(
    registered: list[str],
) -> None:
    """``agent_config`` is a free-form client dict; the binding record is not.

    A user who posts ``runtime_extension_bindings`` in the request body while
    requesting *no* runtime extensions would otherwise persist a binding the
    task never made. Deletion dispatches by that record, so a forged entry
    naming a broken provider makes the task permanently undeletable by its
    owner -- the record is persisted, so every retry replays it.
    """

    _admin_headers()
    _register_second_user("forging-owner", "forgepass1")
    victim = _register("victim_ext", _Provider(fail_delete=True), registered)
    db = _direct_db_session()
    try:
        owner = db.query(User).filter(User.username == "forging-owner").one()
        created = await create_task(
            TaskCreateRequest(
                title="forged binding",
                description="forged binding",
                agent_config={
                    TASK_RUNTIME_BINDINGS_AGENT_CONFIG_KEY: ["victim_ext"],
                    "keep_me": "client value",
                },
            ),
            db=db,
            user=owner,
        )
        task_id = int(created.task_id)

        db.expire_all()
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task_extension_bindings_from_agent_config(task.agent_config) == ()
        # Only the reserved key is stripped; ordinary client config survives.
        assert task.agent_config.get("keep_me") == "client value"
        # The task genuinely never bound: no provider hook ever ran.
        assert victim.deleted_task_ids == []

        result = await delete_task(task_id, db=db, user=owner)

        assert result["success"] is True
        assert victim.deleted_task_ids == []
        assert db.query(Task).filter(Task.id == task_id).count() == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_task_create_binding_record_ignores_forged_entries(
    registered: list[str],
) -> None:
    """Override-drift guard for the legitimate case.

    A request that really does bind ``victim_ext`` while smuggling a forged
    record naming ``other_ext`` must record exactly what the server bound.
    """

    _admin_headers()
    _register_second_user("mixed-owner", "mixedpass1")
    _register("victim_ext", _Provider(), registered)
    other = _register("other_ext", _Provider(fail_delete=True), registered)
    db = _direct_db_session()
    try:
        owner = db.query(User).filter(User.username == "mixed-owner").one()
        created = await create_task(
            TaskCreateRequest(
                title="mixed binding",
                description="mixed binding",
                agent_config={
                    TASK_RUNTIME_BINDINGS_AGENT_CONFIG_KEY: ["other_ext"],
                },
                runtime_extensions={"victim_ext": {}},
            ),
            db=db,
            user=owner,
        )
        task_id = int(created.task_id)

        db.expire_all()
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task_extension_bindings_from_agent_config(task.agent_config) == (
            "victim_ext",
        )
        assert other.deleted_task_ids == []
    finally:
        db.close()


@pytest.mark.asyncio
async def test_task_create_drops_a_client_forged_execution_scope() -> None:
    """A request cannot pre-seed the scope snapshot that governs where a
    task's bytes land -- sandbox mount, storage prefix, workspace directory,
    memory dimensions -- or name a file id for the bound file list, by
    putting either in the request body's ``agent_config``. This boundary
    substitutes a server-validated file list rather than only dropping the
    client's, so a request that carries no ``files`` of its own persists no
    ``selected_file_ids`` key at all; either way, a forged id never survives.
    """

    _admin_headers()
    _register_second_user("scope-owner", "scopepass1")
    db = _direct_db_session()
    try:
        owner = db.query(User).filter(User.username == "scope-owner").one()
        created = await create_task(
            TaskCreateRequest(
                title="forged scope",
                description="forged scope",
                agent_config={
                    EXECUTION_SCOPE_AGENT_CONFIG_KEY: {
                        "sandbox_key_suffix": "victim",
                        "workspace_segments": ["victim"],
                        "memory_dimensions": {"tenant": "victim"},
                    },
                    SELECTED_FILE_IDS_AGENT_CONFIG_KEY: ["victim-file-id"],
                    "keep_me": "client value",
                },
            ),
            db=db,
            user=owner,
        )
        task_id = int(created.task_id)

        db.expire_all()
        task = db.query(Task).filter(Task.id == task_id).one()
        assert EXECUTION_SCOPE_AGENT_CONFIG_KEY not in task.agent_config
        assert execution_scope_from_agent_config(task.agent_config) is None
        assert "victim-file-id" not in (
            task.agent_config.get(SELECTED_FILE_IDS_AGENT_CONFIG_KEY) or []
        )
        assert task.agent_config.get("keep_me") == "client value"
    finally:
        db.close()


# ===== single-task endpoint =====


@pytest.mark.asyncio
async def test_delete_task_succeeds_when_a_failing_provider_owns_nothing(
    registered: list[str],
) -> None:
    """The damaging case: one broken extension must not lock every task.

    The task never bound to ``broken_ext``, so ``broken_ext`` gets no say in
    whether it can be deleted.
    """

    _admin_headers()
    _register_second_user("unbound-owner", "unboundpass1")
    broken = _register("broken_ext", _Provider(fail_delete=True), registered)
    db = _direct_db_session()
    try:
        owner = db.query(User).filter(User.username == "unbound-owner").one()
        task = Task(user_id=int(owner.id), title="unbound", description="")
        db.add(task)
        db.commit()
        task_id = int(task.id)

        result = await delete_task(task_id, db=db, user=owner)

        assert result["success"] is True
        assert broken.deleted_task_ids == []
        assert db.query(Task).filter(Task.id == task_id).count() == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_delete_task_is_fail_closed_for_a_provider_that_owns_the_task(
    registered: list[str],
) -> None:
    """M3 regression guard: an owning provider's failure preserves the task."""

    _admin_headers()
    _register_second_user("bound-owner", "boundpass1")
    owning = _register("owning_ext", _Provider(fail_delete=True), registered)
    db = _direct_db_session()
    try:
        owner = db.query(User).filter(User.username == "bound-owner").one()
        task = Task(
            user_id=int(owner.id),
            title="bound",
            description="",
            agent_config=_bound_to("owning_ext"),
        )
        db.add(task)
        db.commit()
        task_id = int(task.id)

        with pytest.raises(HTTPException) as exc_info:
            await delete_task(task_id, db=db, user=owner)

        assert exc_info.value.status_code == 503
        assert owning.deleted_task_ids == [task_id]
        assert db.query(Task).filter(Task.id == task_id).count() == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_admin_force_delete_removes_core_rows_despite_failing_provider(
    registered: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The escape hatch an operator uses to unblock a chronically broken provider."""

    _admin_headers()
    _register_second_user("force-owner", "forcepass1")
    owning = _register("owning_ext", _Provider(fail_delete=True), registered)
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        owner = db.query(User).filter(User.username == "force-owner").one()
        task = Task(
            user_id=int(owner.id),
            title="force me",
            description="",
            agent_config=_bound_to("owning_ext"),
        )
        db.add(task)
        db.commit()
        task_id = int(task.id)

        with caplog.at_level(logging.ERROR):
            result = await delete_task(task_id, force=True, db=db, user=admin)

        assert result["success"] is True
        assert owning.deleted_task_ids == [task_id]
        assert db.query(Task).filter(Task.id == task_id).count() == 0
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "owning_ext" in messages
        assert "force" in messages.lower()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_force_delete_requires_admin(registered: list[str]) -> None:
    _admin_headers()
    _register_second_user("nonadmin-force", "forcepass1")
    _register("owning_ext", _Provider(fail_delete=True), registered)
    db = _direct_db_session()
    try:
        owner = db.query(User).filter(User.username == "nonadmin-force").one()
        task = Task(
            user_id=int(owner.id),
            title="no force for you",
            description="",
            agent_config=_bound_to("owning_ext"),
        )
        db.add(task)
        db.commit()
        task_id = int(task.id)

        with pytest.raises(HTTPException) as exc_info:
            await delete_task(task_id, force=True, db=db, user=owner)

        assert exc_info.value.status_code == 403
        assert db.query(Task).filter(Task.id == task_id).count() == 1
    finally:
        db.close()


# ===== admin user deletion =====


@pytest.mark.asyncio
async def test_admin_user_delete_ignores_providers_no_task_bound_to(
    registered: list[str],
) -> None:
    _admin_headers()
    _register_second_user("unbound-account", "unboundpass1")
    broken = _register("broken_ext", _Provider(fail_delete=True), registered)
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        target = db.query(User).filter(User.username == "unbound-account").one()
        db.add_all(
            [
                Task(user_id=int(target.id), title=f"t{index}", description="")
                for index in range(3)
            ]
        )
        db.commit()
        target_id = int(target.id)

        response = await delete_user(target_id, admin, db)

        assert response == {"message": "User deleted successfully"}
        assert broken.deleted_task_ids == []
        assert db.query(User).filter(User.id == target_id).count() == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_admin_user_delete_marks_released_bindings_before_aborting(
    registered: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later page's failure must not silently strand earlier pages.

    Pages 1 and 2 release cleanly and their binding records are cleared in the
    DB, so the state on disk matches reality and a retry does not re-dispatch
    providers that already released.
    """

    import xagent.web.api.admin_users as admin_users_module

    _admin_headers()
    _register_second_user("paged-account", "pagedpass1")

    class _FailOnThirdTask(_Provider):
        def __init__(self) -> None:
            super().__init__()
            self.fail_task_ids: set[int] = set()

        async def on_task_deleted(self, context) -> None:
            self.deleted_task_ids.append(int(context.task_id))
            if int(context.task_id) in self.fail_task_ids:
                raise RuntimeError("provider is down for this task")

    provider = _FailOnThirdTask()
    _register("paged_ext", provider, registered)
    monkeypatch.setattr(admin_users_module, "_TASK_RUNTIME_DELETE_PAGE_SIZE", 1)

    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        target = db.query(User).filter(User.username == "paged-account").one()
        tasks = [
            Task(
                user_id=int(target.id),
                title=f"paged {index}",
                description="",
                agent_config=_bound_to("paged_ext"),
            )
            for index in range(3)
        ]
        db.add_all(tasks)
        db.commit()
        task_ids = [int(task.id) for task in tasks]
        provider.fail_task_ids = {task_ids[2]}
        target_id = int(target.id)

        with pytest.raises(HTTPException) as exc_info:
            await delete_user(target_id, admin, db)

        assert exc_info.value.status_code == 503
        assert provider.deleted_task_ids == task_ids

        db.expire_all()
        remaining = {
            int(task.id): task_extension_bindings_from_agent_config(task.agent_config)
            for task in db.query(Task).filter(Task.user_id == target_id).all()
        }
        # Everything is preserved for retry ...
        assert set(remaining) == set(task_ids)
        # ... but the released pages no longer claim a live provider binding,
        # while the task whose provider failed still does.
        assert remaining[task_ids[0]] == ()
        assert remaining[task_ids[1]] == ()
        assert remaining[task_ids[2]] == ("paged_ext",)
        assert db.query(User).filter(User.id == target_id).count() == 1
    finally:
        db.close()
