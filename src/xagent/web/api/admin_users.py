import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session

from ...core.task_runtime import TaskRuntimeContext
from ..auth_dependencies import get_current_user
from ..models.chat_message import TaskChatMessage
from ..models.database import (
    get_db,
    get_session_local,
    release_db_connection_if_clean,
)
from ..models.task import (
    DAGExecution,
    Task,
    TraceCheckpointBlob,
    TraceEvent,
    TraceMessageBlob,
)
from ..models.task_interaction import TaskInteractionRequest
from ..models.uploaded_file import UploadedFile
from ..models.user import User
from ..schemas.user import UserListResponse, UserResponse
from ..services.model_store import ModelStore
from ..services.task_interaction_schema import interaction_requests_table_exists
from ..services.task_runtime import (
    TaskRuntimeExtensionError,
    delete_task_extensions,
    registered_task_extensions,
    store_task_extension_bindings,
    task_extension_bindings_from_agent_config,
)
from ..services.task_workspace_cleanup import (
    WorkspaceCleanupTarget,
    capture_workspace_cleanup_target_best_effort,
    remove_task_workspace,
    unscoped_workspace_cleanup_target,
)
from ..services.user_admin_scope import hidden_user_ids

router = APIRouter(prefix="/api/admin/users", tags=["admin-users"])
logger = logging.getLogger(__name__)
_TASK_RUNTIME_DELETE_CONCURRENCY = 4
_TASK_RUNTIME_DELETE_PAGE_SIZE = 100


def _delete_legacy_text2sql_rows(db: Session, user_id: int) -> None:
    """Delete legacy Text2SQL rows without importing removed ORM models."""
    inspector = inspect(db.get_bind())
    if not inspector.has_table("text2sql_databases"):
        return
    db.execute(
        text("DELETE FROM text2sql_databases WHERE user_id = :user_id"),
        {"user_id": user_id},
    )


def _purge_user_task_rows(db: Session, *, user_id: int) -> None:
    """Delete every task-owned row for one user with one statement per table.

    The single-task endpoint purges row-by-row through ``purge_task_rows``.
    Doing that in a loop costs a SELECT plus five statements per task, and the
    ORM ``db.delete(task)`` additionally loads and per-row-deletes every
    ``TaskChatMessage`` and per-row-NULLs every ``UploadedFile``. Admin user
    deletion has no bound on task count, so it uses set-based statements keyed
    off a single ``tasks`` sub-select instead.

    Ordering matches ``purge_task_rows``: every child row referencing
    ``tasks.id`` goes before the ``tasks`` delete, so the purge stays valid
    under enforced foreign keys regardless of whether the deployment's schema
    carries ``ON DELETE`` clauses. ``tasks`` itself is also referenced --
    ``last_checkpoint_trace_event_id`` FKs to ``trace_events.id`` -- so the
    pointer columns are NULLed first, before the ``trace_events`` delete
    below. ``task_interaction_requests`` rows sit on both ordering
    obligations at once: they reference ``tasks.id`` directly, and their
    ``resume_trace_event_id`` references ``trace_events.id`` through an
    ``ON DELETE SET NULL`` that a still-active row's CHECK forbids -- so
    they must go after the pointer NULL-first and before the
    ``trace_events`` delete.

    This path holds no lease and takes no row fence, matching how it
    already treats every other child table; the interaction rows inherit
    that shape. If a concurrent writer turns a row active between the
    interaction delete and the ``trace_events`` delete, the failure mode
    is one loud ``IntegrityError`` (the active-anchor CHECK) rolling back
    the whole purge transaction with no partial state -- retrying the
    deletion is the complete remedy.
    """

    task_ids = select(Task.id).where(Task.user_id == user_id).scalar_subquery()

    # NULL the checkpoint pointer columns before the trace_events delete
    # below: a task still pointing at a row would block (or, without
    # DB-level enforcement, orphan) that delete. A bulk statement, not an
    # ORM attribute assignment, because this session has autoflush
    # disabled -- an attribute assignment would not reach the database
    # until a later flush, by which point the trace_events delete has
    # already run.
    db.query(Task).filter(Task.user_id == user_id).update(
        {
            Task.last_checkpoint_event_id: None,
            Task.last_checkpoint_trace_event_id: None,
        },
        synchronize_session=False,
    )

    # Interaction rows go before the trace_events delete below, for the same
    # reason and with the same consequence as in purge_task_rows: the
    # resume anchor's ON DELETE SET NULL would otherwise violate
    # ck_task_interaction_requests_active_anchor on every active row and
    # fail the whole user deletion.
    #
    # A separate statement rather than an entry in the loop below: the loop
    # is unconditional, and this delete is gated on the table existing.
    if interaction_requests_table_exists(db):
        db.query(TaskInteractionRequest).filter(
            TaskInteractionRequest.task_id.in_(task_ids)
        ).delete(synchronize_session=False)

    # Children without a DB-level ``ON DELETE`` clause -- these are the rows a
    # bare ``DELETE FROM tasks`` would strand or fail on under strict FKs.
    for model in (
        TraceCheckpointBlob,
        TraceMessageBlob,
        TraceEvent,
        DAGExecution,
    ):
        db.query(model).filter(model.task_id.in_(task_ids)).delete(
            synchronize_session=False
        )

    # ``Task.chat_messages`` relies on an ORM ``delete-orphan`` cascade that a
    # bulk ``tasks`` delete bypasses. Delete explicitly so the behaviour does
    # not depend on the DB honouring the FK's ``ON DELETE CASCADE``.
    db.query(TaskChatMessage).filter(TaskChatMessage.task_id.in_(task_ids)).delete(
        synchronize_session=False
    )

    # ``Task.uploaded_files`` has no cascade, so ORM deletion detaches the rows
    # rather than removing them. Preserve that behaviour: the files are removed
    # by the ``uploaded_files.user_id`` cascade when the user row goes away.
    db.query(UploadedFile).filter(UploadedFile.task_id.in_(task_ids)).update(
        {UploadedFile.task_id: None},
        synchronize_session=False,
    )

    db.query(Task).filter(Task.user_id == user_id).delete(synchronize_session=False)


def _load_task_page_sync(
    *,
    user_id: int,
    last_seen_task_id: int,
    page_size: int,
) -> list[tuple[int, int, str | None, object]]:
    """Read one keyset page of a user's tasks in an operation-local session.

    The page count grows with the user's task count and is unbounded, so this
    read cannot run on the event loop. It selects scalar columns and returns
    plain tuples, so nothing session-bound crosses the thread boundary.
    """

    session_factory = get_session_local()
    page_db = session_factory()
    try:
        return [
            (int(task_id), int(task_user_id), source, agent_config)
            for task_id, task_user_id, source, agent_config in page_db.query(
                Task.id, Task.user_id, Task.source, Task.agent_config
            )
            .filter(
                Task.user_id == user_id,
                Task.id > last_seen_task_id,
            )
            .order_by(Task.id)
            .limit(page_size)
            .all()
        ]
    finally:
        page_db.close()


def _record_settled_bindings_sync(
    *,
    settled: list[tuple[int, tuple[str, ...]]],
) -> None:
    """Persist the post-cleanup binding record for each task on a page.

    This is the DB-visible marker that provider state was released. It makes a
    partially-completed multi-page user deletion honest: tasks whose providers
    already released are no longer bound to them, so retrying the deletion does
    not re-dispatch cleanup for state that is already gone.
    """

    if not settled:
        return
    session_factory = get_session_local()
    record_db = session_factory()
    try:
        for task_id, remaining in settled:
            store_task_extension_bindings(
                record_db,
                task_id=task_id,
                extensions=remaining,
            )
        record_db.commit()
    except Exception:
        record_db.rollback()
        # A lost marker only costs an idempotent re-dispatch on retry; it must
        # not mask the provider failure the caller is about to report.
        logger.exception("Failed to persist runtime extension binding markers")
    finally:
        record_db.close()


def _capture_page_workspace_targets_sync(
    task_rows: list[tuple[int, int, str | None, object]],
) -> tuple[list[WorkspaceCleanupTarget], set[int]]:
    """Capture one page's cleanup targets, and count the ones that failed.

    A task whose scope will not resolve loses only its own directory; it must
    not cost the account's other workspaces or the deletion itself. The count
    is returned rather than swallowed because a dropped target is a directory
    nothing downstream can find: reporting the account as fully cleaned would
    make the response silent in exactly the case it is meant to surface.

    One resolution per task, which is a read per task on a path that is already
    O(tasks) -- and the reason this runs in a worker thread rather than on the
    event loop. Reusing the ``agent_config`` this page already read would avoid
    the per-task read; see #2598.
    """

    targets: list[WorkspaceCleanupTarget] = []
    dropped: set[int] = set()
    for task_id, task_user_id, _source, _agent_config in task_rows:
        target = capture_workspace_cleanup_target_best_effort(
            int(task_id),
            int(task_user_id),
            # Many tasks, so no single activated scope can be theirs.
            prefer_active_scope=False,
        )
        if target is None:
            # Degrade to the unscoped candidates rather than abandoning the
            # task: that is the set the task-level path falls back to when it
            # re-captures after the row is gone, so both paths reclaim the same
            # directories. Still counted as dropped -- what these candidates
            # cannot name is a workspace under a scope segment, which is
            # precisely the leak the failed capture predicts.
            dropped.add(int(task_id))
            targets.append(
                unscoped_workspace_cleanup_target(int(task_id), int(task_user_id))
            )
        else:
            targets.append(target)
    return targets, dropped


def _remove_workspaces_sync(
    *, targets: list[WorkspaceCleanupTarget], user_id: int
) -> set[int]:
    """Remove the captured workspaces, reporting which were left behind.

    Task ids rather than a count, because a task whose capture failed is
    already pending and is in ``targets`` as well: summing the two would
    report one task as two leaked directories.

    Runs after the rows are committed as deleted, so nothing here can fail the
    request: a raised error would tell the admin the account still exists.
    """

    pending: set[int] = set()
    for target in targets:
        try:
            remove_task_workspace(target)
        except Exception:
            pending.add(target.task_id)
            logger.error(
                "User %s was deleted but the workspace of task %s could not be "
                "removed; the directory is leaked and needs manual reconciliation",
                user_id,
                target.task_id,
                exc_info=True,
            )
    return pending


def _delete_user_rows_sync(*, user_id: int) -> bool:
    """Delete one user and every row it owns in an operation-local session."""

    from ..models.auto_model import AutoModelCandidate, AutoModelConfig
    from ..models.mcp import UserMCPServer
    from ..models.model import Model as DBModel
    from ..models.user import UserModel
    from ..services.model_service import _is_model_visible_to_user

    session_factory = get_session_local()
    delete_db = session_factory()
    try:
        user = delete_db.query(User).filter(User.id == user_id).first()
        if user is None:
            return False

        # Existing deployments may still have the removed Text2SQL table. Clean
        # it up by table name so user deletion keeps working under strict FKs.
        _delete_legacy_text2sql_rows(delete_db, user_id)
        _purge_user_task_rows(delete_db, user_id=user_id)

        # Delete user's MCP server associations (not the servers themselves)
        delete_db.query(UserMCPServer).filter(UserMCPServer.user_id == user_id).delete()

        # Auto configs own a virtual model row. Removing the model first also
        # cascades its config and prevents an orphaned router model.
        auto_model_ids = [
            int(model_id)
            for (model_id,) in delete_db.query(AutoModelConfig.router_model_id)
            .filter(AutoModelConfig.user_id == user_id)
            .all()
        ]
        if auto_model_ids:
            delete_db.query(DBModel).filter(DBModel.id.in_(auto_model_ids)).delete(
                synchronize_session=False
            )

        affected_candidates = (
            delete_db.query(AutoModelCandidate)
            .join(AutoModelConfig)
            .filter(
                AutoModelCandidate.target_model_id.in_(
                    select(UserModel.model_id).where(UserModel.user_id == user_id)
                ),
                AutoModelConfig.user_id != user_id,
            )
            .all()
        )

        # Delete the user, then evaluate the surviving grants. Other owners or
        # shared grants can still make a target available to a bound Auto.
        delete_db.delete(user)
        delete_db.flush()
        changed_configs = set()
        for candidate in affected_candidates:
            config = candidate.config
            if _is_model_visible_to_user(
                delete_db, candidate.target_model_id, int(config.user_id)
            ):
                continue
            if config.fallback_model_id == candidate.target_model_id:
                config.fallback_model_id = None
            changed_configs.add(int(config.id))
            delete_db.delete(candidate)
        ModelStore(delete_db).refresh_auto_model_abilities(list(changed_configs))
        delete_db.commit()
        ModelStore(delete_db).invalidate_after_user_delete()
        return True
    except Exception:
        delete_db.rollback()
        raise
    finally:
        delete_db.close()


@router.get("", response_model=UserListResponse)
async def get_users(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    search: str = Query("", description="Search username or email"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserListResponse:
    """Get an admin-only user list with email for human-facing labels."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    # Build query
    query = db.query(User)

    # Hide app-managed service users (e.g. SaaS team storage principals) that
    # own data but are not real accounts.
    hidden = hidden_user_ids(db)
    if hidden:
        query = query.filter(User.id.notin_(hidden))

    # Match account identifiers without exposing SQL LIKE wildcards from user input.
    if search:
        query = query.filter(
            User.username.icontains(search, autoescape=True)
            | User.email.icontains(search, autoescape=True)
        )

    # Get total count
    total = query.count()

    # Apply pagination
    offset = (page - 1) * size
    users = query.offset(offset).limit(size).all()

    return UserListResponse(
        users=[UserResponse.model_validate(user) for user in users],
        total=total,
        page=page,
        size=size,
        pages=(total + size - 1) // size,
    )


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Delete a user (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    # Cannot delete yourself
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    # App-managed service users (e.g. SaaS team storage principals) are hidden
    # from the directory and must not be deletable — doing so would orphan the
    # data they back.
    if user_id in set(hidden_user_ids(db)):
        raise HTTPException(status_code=404, detail="User not found")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Captured while the rows still exist, because a workspace's base
    # directory comes from its task's execution scope and that scope is only
    # readable from the row. Removal happens after the deletion commits, at
    # the bottom of this function: every exit between here and there leaves
    # the account alive, and a workspace removed on one of those paths would
    # be destroying a live user's files.
    #
    # One walk serves both. Where a provider is registered this is the walk
    # that cleanup already ran, and a second pass would double an unbounded
    # admin path's page SELECTs. Where none is registered there was no walk at
    # all before, and this is a new cost: enumerating the account's tasks is
    # the only way to name the directories they own. #2598 removes the
    # per-task read that cost currently carries.
    #
    # This does weaken the walk's own "keyset pages bound memory" property: the
    # targets outlive their page, so the list is O(tasks). It is the narrowest
    # thing that can be held -- an id, an owner and the base dirs, no row --
    # and it is held because removal cannot happen until the deletion commits,
    # by which point no page can be re-read.
    workspace_targets: list[WorkspaceCleanupTarget] = []
    workspace_capture_failures: set[int] = set()
    extension_cleanup_required = bool(registered_task_extensions())
    session_factory = get_session_local()
    cleanup_semaphore = asyncio.Semaphore(_TASK_RUNTIME_DELETE_CONCURRENCY)

    async def _cleanup_context(
        context: TaskRuntimeContext,
        bound_extensions: tuple[str, ...],
    ) -> tuple[str, ...]:
        async with cleanup_semaphore:
            return await delete_task_extensions(
                context,
                bound_extensions=bound_extensions,
            )

    # Keyset pages bound memory and gather fan-out without growing a NOT IN
    # parameter list. New tasks receive higher ids and are picked up by the
    # next page after provider cleanup awaits external systems.
    last_seen_task_id = 0
    while True:
        release_db_connection_if_clean(db)
        task_rows = await asyncio.to_thread(
            _load_task_page_sync,
            user_id=user_id,
            last_seen_task_id=last_seen_task_id,
            page_size=_TASK_RUNTIME_DELETE_PAGE_SIZE,
        )
        if not task_rows:
            break
        last_seen_task_id = max(int(row[0]) for row in task_rows)
        page_targets, page_dropped = await asyncio.to_thread(
            _capture_page_workspace_targets_sync, task_rows
        )
        workspace_targets.extend(page_targets)
        workspace_capture_failures |= page_dropped
        if not extension_cleanup_required:
            continue
        page = [
            (
                TaskRuntimeContext(
                    task_id=int(task_id),
                    user_id=int(task_user_id),
                    source=str(source) if source is not None else None,
                    session_factory=session_factory,
                ),
                task_extension_bindings_from_agent_config(agent_config),
            )
            for task_id, task_user_id, source, agent_config in task_rows
        ]
        # Tasks with no binding record own nothing in any provider; skip
        # them entirely so a broken extension cannot block the account.
        bound_page = [(context, bindings) for context, bindings in page if bindings]
        if not bound_page:
            continue
        release_db_connection_if_clean(db)
        cleanup_results = await asyncio.gather(
            *(_cleanup_context(context, bindings) for context, bindings in bound_page),
            return_exceptions=True,
        )
        for result in cleanup_results:
            if isinstance(result, asyncio.CancelledError):
                raise result

        # Narrow every task's binding record to what is still held before
        # deciding whether to abort. Without this, a failure on a later page
        # would leave earlier pages' tasks with released provider bindings,
        # no DB-visible marker, and a retry that re-dispatches them.
        settled: list[tuple[int, tuple[str, ...]]] = []
        cleanup_failures: list[tuple[int, BaseException]] = []
        for (context, _bindings), result in zip(
            bound_page, cleanup_results, strict=True
        ):
            if isinstance(result, TaskRuntimeExtensionError):
                settled.append((context.task_id, result.unreleased_extensions))
                cleanup_failures.append((context.task_id, result))
            elif isinstance(result, BaseException):
                # Unknown failure: assume nothing was released.
                cleanup_failures.append((context.task_id, result))
            else:
                settled.append((context.task_id, tuple(result)))
        await asyncio.to_thread(_record_settled_bindings_sync, settled=settled)

        if cleanup_failures:
            for failed_task_id, failure in cleanup_failures:
                logger.error(
                    "Runtime extension cleanup failed; preserving user %s "
                    "and task %s for retry: %s",
                    user_id,
                    failed_task_id,
                    failure,
                )
            raise HTTPException(
                status_code=503,
                detail=("Runtime extension cleanup failed; the user was not deleted"),
            )

    # The rest of user deletion is synchronous ORM/DBAPI work whose cost scales
    # with the user's task count. Run it in a worker thread and in its own
    # session so an admin deleting a large account cannot stall the event loop.
    release_db_connection_if_clean(db)
    deleted = await asyncio.to_thread(_delete_user_rows_sync, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")

    pending = workspace_capture_failures | await asyncio.to_thread(
        _remove_workspaces_sync, targets=workspace_targets, user_id=user_id
    )

    if pending:
        logger.error(
            "User %s was deleted with %d of %d task workspace(s) left on disk",
            user_id,
            len(pending),
            len(workspace_targets),
        )

    return {
        "message": "User deleted successfully",
        # Always present, for the same reason as on the task-level delete: the
        # account really was deleted, so failing the request would be a lie,
        # and an admin who is told only "deleted" has no way to learn that
        # directories are still on disk. A bool, not the count, so the field
        # means the same thing on both endpoints; the count is in the log line
        # above, which is where an operator reconciling them is looking.
        "workspace_cleanup_pending": bool(pending),
    }
