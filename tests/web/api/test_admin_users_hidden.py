from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from xagent.core.task_runtime import TaskRuntimeContribution
from xagent.web.api.admin_users import delete_user, get_users
from xagent.web.models.task import DAGExecution, DAGExecutionPhase, Task
from xagent.web.models.user import User
from xagent.web.services.task_runtime import (
    agent_config_with_task_extension_bindings,
    register_task_extension,
    unregister_task_extension,
)
from xagent.web.services.user_admin_scope import set_hidden_user_filter

from .conftest import _admin_headers, _direct_db_session, _register_second_user

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def _reset_hidden_filter():
    yield
    set_hidden_user_filter(None)


def _bound_to(*extensions: str) -> dict:
    """Task ``agent_config`` recording the runtime extensions the task bound to."""

    return agent_config_with_task_extension_bindings({}, extensions)


class _DeleteObserverProvider:
    def __init__(self) -> None:
        self.task_existed_on_delete: list[bool] = []

    async def on_task_created(self, context, configuration) -> None:
        return None

    async def build_runtime(self, context) -> TaskRuntimeContribution:
        return TaskRuntimeContribution()

    async def public_metadata(self, context) -> dict:
        return {}

    async def on_task_deleted(self, context) -> None:
        db = context.session_factory()
        try:
            self.task_existed_on_delete.append(
                db.query(Task).filter(Task.id == context.task_id).count() == 1
            )
        finally:
            db.close()


@pytest.mark.asyncio
async def test_hidden_users_excluded_from_admin_list_and_delete():
    _admin_headers()  # ensures the admin account exists
    _register_second_user("ghost", "ghostpass1")
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        ghost_id = int(db.query(User).filter(User.username == "ghost").one().id)

        # Baseline: no filter (standalone default) -> user is listed.
        res = await get_users(1, 100, "", admin, db)
        assert any(u.id == ghost_id for u in res.users)

        set_hidden_user_filter(lambda _db: [ghost_id])

        # Excluded from the list (and total) ...
        res = await get_users(1, 100, "", admin, db)
        assert all(u.id != ghost_id for u in res.users)
        assert res.total == db.query(User).count() - 1
        # ... from search ...
        res = await get_users(1, 100, "ghost", admin, db)
        assert all(u.id != ghost_id for u in res.users)
        # ... and cannot be deleted (would orphan the data it backs).
        with pytest.raises(HTTPException) as exc:
            await delete_user(ghost_id, admin, db)
        assert exc.value.status_code == 404
        assert db.query(User).filter(User.id == ghost_id).count() == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_admin_users_include_email_and_support_email_search():
    _admin_headers()
    opaque_username = "acct_0123456789abcdef0123456789abcdef"
    _register_second_user(opaque_username, "opaque-password")
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        target = db.query(User).filter(User.username == opaque_username).one()
        target.email = "display@example.com"
        db.commit()

        result = await get_users(1, 100, "display@example.com", admin, db)

        assert len(result.users) == 1
        assert result.users[0].username == opaque_username
        assert result.users[0].email == "display@example.com"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_admin_user_search_still_matches_username():
    _admin_headers()
    username = "searchable-user"
    _register_second_user(username, "search-password")
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()

        result = await get_users(1, 100, username, admin, db)

        assert [user.username for user in result.users] == [username]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_admin_user_list_serializes_null_email():
    _admin_headers()
    username = "email-less-user"
    _register_second_user(username, "search-password")
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        target = db.query(User).filter(User.username == username).one()
        target.email = None
        db.commit()

        result = await get_users(1, 100, username, admin, db)

        assert len(result.users) == 1
        assert result.users[0].email is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_admin_user_email_search_is_case_insensitive():
    _admin_headers()
    opaque_username = "acct_0123456789abcdef0123456789abcdef"
    _register_second_user(opaque_username, "opaque-password")
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        target = db.query(User).filter(User.username == opaque_username).one()
        target.email = "display@example.com"
        db.commit()
        db.execute(text("PRAGMA case_sensitive_like = ON"))

        result = await get_users(1, 100, "DISPLAY@EXAMPLE.COM", admin, db)

        assert [user.id for user in result.users] == [target.id]
    finally:
        db.execute(text("PRAGMA case_sensitive_like = OFF"))
        db.close()


@pytest.mark.asyncio
async def test_admin_user_search_treats_wildcards_as_literal_text():
    _admin_headers()
    _register_second_user("search-user", "search-password")
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()

        result = await get_users(1, 100, "%_", admin, db)

        assert result.users == []
        assert result.total == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_admin_user_delete_runs_runtime_cleanup_before_task_delete():
    _admin_headers()
    _register_second_user("runtime-user", "runtimepass1")
    provider = _DeleteObserverProvider()
    register_task_extension("delete_observer", provider)
    db = _direct_db_session()
    try:
        db.execute(text("PRAGMA foreign_keys = ON"))
        admin = db.query(User).filter(User.username == "admin").one()
        target = db.query(User).filter(User.username == "runtime-user").one()
        task = Task(
            user_id=int(target.id),
            title="runtime task",
            description="runtime task",
            agent_config=_bound_to("delete_observer"),
        )
        db.add(task)
        db.flush()
        db.add(
            DAGExecution(
                task_id=int(task.id),
                phase=DAGExecutionPhase.COMPLETED,
            )
        )
        db.commit()
        task_id = int(task.id)
        target_id = int(target.id)

        response = await delete_user(target_id, admin, db)

        assert response == {"message": "User deleted successfully"}
        assert provider.task_existed_on_delete == [True]
        assert db.query(Task).filter(Task.id == task_id).count() == 0
        assert (
            db.query(DAGExecution).filter(DAGExecution.task_id == task_id).count() == 0
        )
        assert db.query(User).filter(User.id == target_id).count() == 0
    finally:
        unregister_task_extension("delete_observer")
        db.close()


@pytest.mark.asyncio
async def test_admin_user_delete_preserves_user_when_runtime_cleanup_fails():
    _admin_headers()
    _register_second_user("failing-runtime-user", "runtimepass1")

    class _FailingDeleteProvider(_DeleteObserverProvider):
        async def on_task_deleted(self, context) -> None:
            await super().on_task_deleted(context)
            raise RuntimeError("provider cleanup failed")

    provider = _FailingDeleteProvider()
    register_task_extension("failing_delete_observer", provider)
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        target = db.query(User).filter(User.username == "failing-runtime-user").one()
        task = Task(
            user_id=int(target.id),
            title="preserve on cleanup failure",
            description="",
            agent_config=_bound_to("failing_delete_observer"),
        )
        db.add(task)
        db.commit()
        task_id = int(task.id)
        target_id = int(target.id)

        with pytest.raises(HTTPException) as exc_info:
            await delete_user(target_id, admin, db)

        assert exc_info.value.status_code == 503
        assert provider.task_existed_on_delete == [True]
        assert db.query(Task).filter(Task.id == task_id).count() == 1
        assert db.query(User).filter(User.id == target_id).count() == 1
    finally:
        unregister_task_extension("failing_delete_observer")
        db.close()


@pytest.mark.asyncio
async def test_admin_user_delete_skips_task_runtime_scan_without_providers(
    monkeypatch,
):
    import xagent.web.api.admin_users as admin_users_module

    _admin_headers()
    _register_second_user("no-runtime-user", "runtimepass1")
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        target = db.query(User).filter(User.username == "no-runtime-user").one()
        db.add(
            Task(
                user_id=int(target.id),
                title="ordinary task",
                description="ordinary task",
            )
        )
        db.commit()
        target_id = int(target.id)

        monkeypatch.setattr(
            admin_users_module,
            "registered_task_extensions",
            lambda: (),
        )

        async def unexpected_cleanup(context):
            raise AssertionError("runtime cleanup must be skipped")

        monkeypatch.setattr(
            admin_users_module,
            "delete_task_extensions",
            unexpected_cleanup,
        )

        response = await delete_user(target_id, admin, db)

        assert response == {"message": "User deleted successfully"}
        assert db.query(User).filter(User.id == target_id).count() == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_admin_user_delete_cleans_task_created_during_runtime_cleanup():
    _admin_headers()
    _register_second_user("racy-runtime-user", "runtimepass1")

    class _CreateTaskDuringCleanupProvider(_DeleteObserverProvider):
        def __init__(self) -> None:
            super().__init__()
            self.created_replacement = False
            self.cleaned_task_ids: list[int] = []

        async def on_task_deleted(self, context) -> None:
            self.cleaned_task_ids.append(context.task_id)
            if not self.created_replacement:
                self.created_replacement = True
                replacement_db = context.session_factory()
                try:
                    replacement_db.add(
                        Task(
                            user_id=context.user_id,
                            title="replacement during cleanup",
                            description="",
                            agent_config=_bound_to("racy_delete_observer"),
                        )
                    )
                    replacement_db.commit()
                finally:
                    replacement_db.close()

    provider = _CreateTaskDuringCleanupProvider()
    register_task_extension("racy_delete_observer", provider)
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        target = db.query(User).filter(User.username == "racy-runtime-user").one()
        task = Task(
            user_id=int(target.id),
            title="initial runtime task",
            description="",
            agent_config=_bound_to("racy_delete_observer"),
        )
        db.add(task)
        db.commit()
        initial_task_id = int(task.id)
        target_id = int(target.id)

        response = await delete_user(target_id, admin, db)

        assert response == {"message": "User deleted successfully"}
        assert initial_task_id in provider.cleaned_task_ids
        assert len(provider.cleaned_task_ids) == 2
        assert db.query(Task).filter(Task.user_id == target_id).count() == 0
    finally:
        unregister_task_extension("racy_delete_observer")
        db.close()


@pytest.mark.asyncio
async def test_admin_user_delete_pages_runtime_cleanup(monkeypatch):
    import xagent.web.api.admin_users as admin_users_module

    _admin_headers()
    _register_second_user("paged-runtime-user", "runtimepass1")

    class _PagedDeleteObserver(_DeleteObserverProvider):
        def __init__(self) -> None:
            super().__init__()
            self.cleaned_task_ids: list[int] = []

        async def on_task_deleted(self, context) -> None:
            self.cleaned_task_ids.append(context.task_id)

    provider = _PagedDeleteObserver()
    register_task_extension("paged_delete_observer", provider)
    monkeypatch.setattr(admin_users_module, "_TASK_RUNTIME_DELETE_PAGE_SIZE", 2)
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        target = db.query(User).filter(User.username == "paged-runtime-user").one()
        tasks = [
            Task(
                user_id=int(target.id),
                title=f"runtime task {index}",
                description="",
                agent_config=_bound_to("paged_delete_observer"),
            )
            for index in range(5)
        ]
        db.add_all(tasks)
        db.commit()
        task_ids = {int(task.id) for task in tasks}
        target_id = int(target.id)

        response = await delete_user(target_id, admin, db)

        assert response == {"message": "User deleted successfully"}
        assert set(provider.cleaned_task_ids) == task_ids
        assert db.query(Task).filter(Task.user_id == target_id).count() == 0
    finally:
        unregister_task_extension("paged_delete_observer")
        db.close()
