"""Chat API route handlers"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from ...config import (
    get_default_task_execution_mode,
)
from ...core.model.chat.token_context import (
    aggregate_media_usage_by_model,
    aggregate_token_usage_by_model,
)
from ...core.task_runtime import (
    TaskRuntimeClientError,
)
from ...core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from ...core.tools.core.knowledge_base_scope import KnowledgeBaseScopeError
from ..auth_dependencies import get_current_user
from ..models.agent import Agent, is_workforce_generated_manager_agent
from ..models.chat_message import TaskChatMessage
from ..models.database import (
    get_db,
    get_session_local,
    release_db_connection_if_clean,
)
from ..models.model import Model as DBModel
from ..models.task import AgentType, Task, TaskStatus, TraceEvent
from ..models.user import User
from ..models.user_channel import UserChannel
from ..schemas.chat import TaskCreateRequest, TaskCreateResponse
from ..schemas.connector_runtime import (
    ConnectorRuntimeRequirementsModel,
    ConnectorRuntimeValuesRequest,
)
from ..services import agent_service_manager as agent_runtime_service
from ..services.agent_access import list_accessible_published_agents
from ..services.agent_team_scope import (
    get_agent_team_scope,
    owned_agent_clause,
)
from ..services.assistant_history_safety import ASSISTANT_RESPONSE_MESSAGE_TYPE
from ..services.chat_history_service import (
    persist_assistant_message_no_commit,
)
from ..services.client_error_messages import ClientErrorCode, client_error_message
from ..services.connector_runtime import (
    apply_task_connector_runtime_context_values,
    bind_connector_runtime_selection_snapshot,
    build_task_runtime_requirements,
    resolve_agent_runtime_requirements,
)
from ..services.hot_path_cache import (
    cache_get,
    cache_set,
    cache_version_token,
    invalidate_task_cache,
    task_cache_ttl_seconds,
    web_task_detail_key,
    web_task_status_key,
)
from ..services.llm_utils import AutoModelUnavailableError, resolve_llms_from_names
from ..services.managed_file_ref import ensure_uploaded_file_local_path
from ..services.model_service import _get_visible_user_ids
from ..services.public_trace_events import public_task_trace_filter
from ..services.task_deletion import purge_task_rows
from ..services.task_interaction_read import get_pending_interaction_question
from ..services.task_runtime import (
    SELECTED_FILE_IDS_AGENT_CONFIG_KEY,
    TaskRuntimeExtensionError,
    agent_config_with_task_extension_bindings,
    create_task_extensions,
    delete_task_extensions,
    get_task_runtime_public_metadata,
    sanitize_client_agent_config,
    task_extension_bindings_from_agent_config,
    validate_task_extension_requests,
)
from ..services.workforce_runtime import resolve_workforce_task_runtime
from ..utils.db_timezone import format_datetime_for_api, safe_timestamp_to_unix

logger = logging.getLogger(__name__)


# Create router
chat_router = APIRouter(prefix="/api/chat", tags=["chat"])

_TERMINAL_CACHE_STATUSES = {TaskStatus.COMPLETED, TaskStatus.FAILED}


def _build_task_agent_config(
    request_agent_config: Optional[Dict[str, Any]],
    selected_file_ids: list[str],
) -> Optional[Dict[str, Any]]:
    """Build task agent_config with server-owned selected file ids."""
    task_agent_config: Dict[str, Any] = sanitize_client_agent_config(
        request_agent_config
    )
    task_agent_config.pop(SELECTED_FILE_IDS_AGENT_CONFIG_KEY, None)
    if selected_file_ids:
        task_agent_config[SELECTED_FILE_IDS_AGENT_CONFIG_KEY] = selected_file_ids
    return task_agent_config or None


def _load_agent_for_task_create(
    db: Session,
    user: User,
    agent_id: int,
) -> Agent | None:
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if agent is None:
        return None
    if is_workforce_generated_manager_agent(agent):
        return None
    # Team-scoped ownership: teammates may run a team-visible agent, and an
    # admins-only agent stays hidden from non-admins. Falls back to the
    # published-visibility path below when the caller does not own it.
    owned = (
        db.query(Agent)
        .filter(
            Agent.id == agent_id,
            owned_agent_clause(int(user.id), get_agent_team_scope(db, int(user.id))),
        )
        .first()
    )
    if owned is not None:
        return owned
    if not agent_runtime_service._is_published_agent(agent):
        return None
    visible_agent_ids = {
        int(item.id)
        for item in list_accessible_published_agents(
            db,
            user,
            purpose="agent_list",
        )
    }
    return agent if int(agent.id) in visible_agent_ids else None


def _get_task_activity_ids(db: Session, task_id: int) -> tuple[int, int]:
    max_trace_event_id = (
        db.query(func.max(TraceEvent.id))
        .filter(
            TraceEvent.task_id == task_id,
            public_task_trace_filter(TraceEvent),
        )
        .scalar()
        or 0
    )
    max_chat_message_id = (
        db.query(func.max(TaskChatMessage.id))
        .filter(TaskChatMessage.task_id == task_id)
        .scalar()
        or 0
    )
    return int(max_trace_event_id), int(max_chat_message_id)


def _compensate_failed_task_extension_create(
    db: Session,
    *,
    task_id: int,
) -> None:
    """Remove a just-created task after provider binding setup failed."""

    db.rollback()
    deleted = purge_task_rows(db, task_id=task_id)
    db.commit()
    if deleted:
        invalidate_task_cache(task_id)


def _load_task_delete_snapshot_sync(
    *,
    task_id: int,
    requester_user_id: int,
    is_admin: bool,
) -> tuple[str, int, Any, tuple[str, ...]] | None:
    """Load detached delete inputs without sharing the request session.

    The fourth element is the task's runtime-extension binding record, so
    provider cleanup dispatches only to providers this task actually bound to.
    """

    session_factory = get_session_local()
    delete_db = session_factory()
    try:
        query = delete_db.query(Task).filter(Task.id == task_id)
        if not is_admin:
            query = query.filter(Task.user_id == requester_user_id)
        task = query.first()
        if task is None:
            return None
        return (
            str(task.title),
            int(task.user_id),
            task.source,
            task_extension_bindings_from_agent_config(task.agent_config),
        )
    finally:
        delete_db.close()


def _delete_task_sync(*, task_id: int) -> bool:
    """Delete one task in an operation-local session."""

    session_factory = get_session_local()
    delete_db = session_factory()
    try:
        deleted = purge_task_rows(delete_db, task_id=task_id)
        if not deleted:
            delete_db.rollback()
            return False
        delete_db.commit()
        return True
    except Exception:
        delete_db.rollback()
        raise
    finally:
        delete_db.close()


def _build_unique_workspace_target(base_dir: Path, filename: str) -> Path:
    candidate = base_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        next_candidate = base_dir / f"{stem}_{index}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        index += 1


@chat_router.post("/task/create", response_model=TaskCreateResponse)
async def create_task(
    request: TaskCreateRequest,
    # FastAPI always injects the real Request for HTTP calls regardless of the
    # default; the None default keeps direct (test) callers working. Must stay
    # a bare `Request` annotation, not `Optional[Request]` -- FastAPI only
    # recognizes the special-cased injected-Request parameter with the exact
    # bare type; wrapping it in Optional makes FastAPI try to build a Pydantic
    # field from it instead, which fails at route-registration time since
    # Request isn't a valid Pydantic field type (verified: this reproduces a
    # collection-time FastAPIError in every test that imports this module).
    http_request: Request = None,  # type: ignore[assignment]
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> TaskCreateResponse:
    """Create new chat task"""
    try:
        try:
            # Pre-flight only. ``create_task_extensions`` validates again below,
            # and both calls are needed:
            #  * here, so an unregistered extension or an oversized
            #    configuration is a 400 *before* the ``Task`` row is committed
            #    and has to be compensated away again;
            #  * there, because the service layer is the SSOT: SDK and internal
            #    callers reach ``create_task_extensions`` without ever passing
            #    through this endpoint, and it re-reads the registry immediately
            #    before dispatching, so an extension unregistered between the
            #    two points is rejected instead of dispatched.
            # Do not delete either call as a "duplicate".
            runtime_extension_requests = validate_task_extension_requests(
                request.runtime_extensions
            )
        except (TypeError, ValueError) as exc:
            logger.info("Rejected invalid task runtime extension request: %s", exc)
            raise HTTPException(
                status_code=400,
                detail="Invalid task runtime extension request",
            ) from exc

        # Build task description with file information
        task_description = request.description or ""

        selected_file_ids: list[str] = []

        # Add file information to description if files are specified
        if request.files:
            from ..models.uploaded_file import UploadedFile

            file_info_list = []
            file_paths = []

            for file_id in request.files:
                uploaded_file = (
                    db.query(UploadedFile)
                    .filter(
                        UploadedFile.file_id == file_id,
                        UploadedFile.user_id == int(user.id),
                        UploadedFile.task_id.is_(None),
                        UploadedFile.storage_status != "compensating",
                    )
                    .first()
                )
                if uploaded_file is None:
                    file_info_list.append(f"File ID: {file_id} (File does not exist)")
                    continue

                selected_file_ids.append(str(file_id))

                file_path = ensure_uploaded_file_local_path(uploaded_file)
                file_paths.append(str(file_path))

                if file_path.exists():
                    file_info_list.append(
                        f"File: {uploaded_file.filename} (Path: {file_path})"
                    )
                else:
                    file_info_list.append(
                        f"File: {uploaded_file.filename} (File does not exist)"
                    )

            if file_info_list:
                if task_description:
                    task_description += "\n\nUploaded files:\n" + "\n".join(
                        file_info_list
                    )
                else:
                    task_description = "File processing task:\n" + "\n".join(
                        file_info_list
                    )

        # Set LLM configuration for this task first to get model info.
        # Prefer internal model identifiers (llm_ids).
        # If neither is provided but agent_id is, fetch from agent config.
        from ..models.user import UserDefaultModel, UserModel
        from ..services.llm_utils import CoreStorage

        core_storage = CoreStorage(db, DBModel)

        def _to_internal_model_id_if_accessible(
            model_ref: Optional[Any],
        ) -> Optional[str]:
            if model_ref is None:
                return None
            if isinstance(model_ref, str):
                model_ref = model_ref.strip()
                if not model_ref:
                    return None

            db_model = core_storage.get_db_model(model_ref)
            if not db_model or not bool(db_model.is_active):
                return None

            # Two-step access check: own → shared from visible users
            own_model = (
                db.query(UserModel)
                .filter(
                    UserModel.user_id == int(user.id),
                    UserModel.model_id == db_model.id,
                    UserModel.is_owner.is_(True),
                )
                .first()
            )
            if not own_model:
                visible_ids = _get_visible_user_ids(db, int(user.id))
                own_model = (
                    db.query(UserModel)
                    .filter(
                        UserModel.model_id == db_model.id,
                        UserModel.user_id.in_(visible_ids),
                        UserModel.is_shared.is_(True),
                    )
                    .first()
                )
            has_access = own_model is not None
            if not has_access:
                return None

            return str(db_model.model_id)

        def _normalize_llm_refs(llm_refs: List[Optional[Any]]) -> List[Optional[str]]:
            return [
                _to_internal_model_id_if_accessible(model_ref) for model_ref in llm_refs
            ]

        def _get_default_internal_model_ids() -> Dict[str, Optional[str]]:
            from ..models.model import Model as DBModel

            config_types = ["general", "small_fast", "visual", "compact"]
            defaults: Dict[str, Optional[str]] = {ct: None for ct in config_types}

            # User-specific defaults (Mode A: use DBModel JOIN).
            user_defaults = (
                db.query(UserDefaultModel)
                .join(DBModel, UserDefaultModel.model_id == DBModel.id)
                .filter(
                    UserDefaultModel.user_id == int(user.id),
                    DBModel.is_active,
                    UserDefaultModel.config_type.in_(config_types),
                )
                .all()
            )
            from ..services.model_service import _is_model_visible_to_user

            for row in user_defaults:
                if row.model:
                    if _is_model_visible_to_user(db, row.model.id, int(user.id)):
                        config_type = cast(str, row.config_type)
                        defaults[config_type] = str(row.model.model_id)

            # Fill missing defaults from visible users' shared defaults.
            if any(defaults[ct] is None for ct in config_types):
                visible_ids = _get_visible_user_ids(db, int(user.id))
                shared_defaults = (
                    db.query(UserDefaultModel)
                    .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                    .join(DBModel, UserDefaultModel.model_id == DBModel.id)
                    .filter(
                        UserDefaultModel.config_type.in_(config_types),
                        DBModel.is_active,
                        UserModel.is_shared.is_(True),
                        UserDefaultModel.user_id.in_(visible_ids),
                    )
                    .all()
                )
                for row in shared_defaults:
                    config_type = row.config_type  # type: ignore
                    if row.model and defaults.get(config_type) is None:
                        defaults[config_type] = str(row.model.model_id)

            return defaults

        selected_agent: Optional[Agent] = None
        if request.agent_id:
            selected_agent = _load_agent_for_task_create(
                db,
                user,
                int(request.agent_id),
            )
            if not selected_agent:
                raise HTTPException(
                    status_code=404,
                    detail="Agent not found or access denied",
                )

        llm_ids_to_use = request.llm_ids
        if selected_agent:
            if request.llm_ids:
                logger.warning(
                    f"Ignoring caller-supplied llm_ids {request.llm_ids} because agent_id {request.agent_id} is present."
                )
            llm_ids_to_use = None
            if selected_agent.models:
                # Fetch model configuration from agent
                agent_models = selected_agent.models
                # Agent Builder stores references that may be DB PKs; normalize to internal
                # model_id only if the current user has access.
                llm_ids_to_use = _normalize_llm_refs(
                    [
                        agent_models.get("general"),
                        agent_models.get("small_fast"),
                        agent_models.get("visual"),
                        agent_models.get("compact"),
                    ]
                )
                logger.info(
                    f"Using agent {request.agent_id} model configuration (llm_ids): {llm_ids_to_use}"
                )

        # Normalize any refs (pk/model_name/model_id) to internal model_id strings,
        # but only if the current user has access to the model.
        if llm_ids_to_use:
            llm_ids_to_use = _normalize_llm_refs(llm_ids_to_use)

        default_llm, fast_llm, vision_llm, compact_llm = resolve_llms_from_names(
            llm_ids_to_use, db, int(user.id)
        )

        # Extract provider model names from resolved LLM instances for database storage
        default_model_name = default_llm.model_name if default_llm else None
        fast_model_name = fast_llm.model_name if fast_llm else None
        visual_model_name = vision_llm.model_name if vision_llm else None
        compact_model_name = compact_llm.model_name if compact_llm else None

        # Persist both:
        # - *_model_id: internal stable identifier (preferred for selection)
        # - *_model_name: provider-facing model name (useful for display/audit)
        default_model_id: Optional[str] = None
        fast_model_id: Optional[str] = None
        visual_model_id: Optional[str] = None
        compact_model_id: Optional[str] = None

        if llm_ids_to_use and len(llm_ids_to_use) == 4:
            default_model_id = llm_ids_to_use[0]
            fast_model_id = llm_ids_to_use[1]
            visual_model_id = llm_ids_to_use[2]
            compact_model_id = llm_ids_to_use[3]

        if (
            default_model_id is None
            or fast_model_id is None
            or visual_model_id is None
            or compact_model_id is None
        ):
            default_ids = _get_default_internal_model_ids()
            default_model_id = default_model_id or default_ids.get("general")
            fast_model_id = fast_model_id or default_ids.get("small_fast")
            visual_model_id = visual_model_id or default_ids.get("visual")
            compact_model_id = compact_model_id or default_ids.get("compact")

        # Convert agent_type string to enum
        agent_type_enum = AgentType.STANDARD
        if request.agent_type:
            try:
                agent_type_enum = AgentType(request.agent_type)
            except ValueError:
                logger.warning(
                    f"Unknown agent_type '{request.agent_type}', using STANDARD"
                )
                agent_type_enum = AgentType.STANDARD

        # Convert examples to list of dicts if provided
        examples_data = None
        if request.examples:
            examples_data = [
                {"input": ex.input, "output": ex.output} for ex in request.examples
            ]

        task_agent_config = _build_task_agent_config(
            request.agent_config,
            selected_file_ids,
        )
        if request.is_preview:
            task_agent_config = task_agent_config or {}
            task_agent_config["is_preview"] = True

        task_execution_mode = request.execution_mode
        if not task_execution_mode:
            task_execution_mode = get_default_task_execution_mode(
                agent_id=request.agent_id,
            )

        # Create task with PENDING status and model configuration
        task_title = request.title if request.title else task_description
        if task_title and len(task_title) > 50:
            task_title = task_title[:50] + "..."

        task = Task(
            user_id=user.id,  # Use authenticated user ID
            title=task_title,
            description=task_description,
            status=TaskStatus.PENDING,
            model_id=default_model_id,
            small_fast_model_id=fast_model_id,
            visual_model_id=visual_model_id,
            compact_model_id=compact_model_id,
            model_name=default_model_name,
            small_fast_model_name=fast_model_name,
            visual_model_name=visual_model_name,
            compact_model_name=compact_model_name,
            agent_config=task_agent_config,
            execution_mode=task_execution_mode,
            process_description=request.process_description,
            examples=examples_data,
            agent_id=request.agent_id,  # Set agent_id if provided
            is_visible=False if request.is_preview else request.is_visible,
        )
        selected_refs, connector_runtime_requirements = (
            resolve_agent_runtime_requirements(
                db=db,
                agent=selected_agent,
                connector_user_id=int(user.id),
            )
        )
        bind_connector_runtime_selection_snapshot(
            task=task, selected_refs=selected_refs
        )

        # Set agent_type using the property to avoid Column type issues
        task.agent_type_enum = agent_type_enum
        db.add(task)
        db.flush()

        # Set LLM configuration for this task in agent manager
        task_llm_ids_to_set = [
            default_model_id,
            fast_model_id,
            visual_model_id,
            compact_model_id,
        ]
        logger.info(
            f"Setting LLM configuration for task {task.id} with llm_ids: {task_llm_ids_to_set}"
        )
        # ``http_request`` (the real Starlette Request, with cookies/headers)
        # -- not ``request`` (the parsed TaskCreateRequest body, which has
        # neither) -- so WebToolConfig.get_browser_locale() can resolve the
        # account's app_locale cookie once this task's tools get built.
        agent_runtime_service.get_agent_manager(http_request).set_task_llms(
            int(task.id), task_llm_ids_to_set, db
        )

        if selected_file_ids:
            from ..models.uploaded_file import UploadedFile

            (
                db.query(UploadedFile)
                .filter(
                    UploadedFile.file_id.in_(selected_file_ids),
                    UploadedFile.user_id == int(user.id),
                    UploadedFile.task_id.is_(None),
                    UploadedFile.storage_status != "compensating",
                )
                .update(
                    {UploadedFile.task_id: int(task.id)},
                    synchronize_session=False,
                )
            )

        if runtime_extension_requests:
            # Record which providers this task binds to *before* any hook runs,
            # in the same transaction that creates the task. Deletion dispatches
            # only to this set, so over-recording (a provider whose hook never
            # completed) is safe -- ``on_task_deleted`` is required to be
            # idempotent -- while under-recording would silently leak
            # provider-owned state.
            setattr(
                task,
                "agent_config",
                agent_config_with_task_extension_bindings(
                    task.agent_config,
                    runtime_extension_requests.keys(),
                ),
            )

        if request.seed_assistant_message is not None:
            # Staged (not committed) here so the seed message lands in the
            # same transaction as task creation - a client that opens this
            # task never observes it existing with zero history.
            # `seed_interactions` (e.g. a marketplace persona's "connect your
            # apps" prompt) rides along on the same row; replay still forces
            # expect_response=False for every historical row regardless (see
            # websocket.py), so this never puts the task into
            # waiting_for_user - any interaction type attached here must be
            # able to stand on its own without that state, same as
            # "connect_apps" (a live widget, not a question-and-submit form).
            seeded_message = persist_assistant_message_no_commit(
                db,
                task_id=int(task.id),
                user_id=int(user.id),
                content=request.seed_assistant_message,
                interactions=request.seed_interactions,
                message_type=ASSISTANT_RESPONSE_MESSAGE_TYPE,
            )
            if seeded_message is None:
                # persist_assistant_message_no_commit silently drops a
                # message that normalizes to empty (e.g. an
                # all-whitespace seed) - not an error worth failing task
                # creation over, but worth a trail for whoever is
                # debugging why a "speak first" flow produced no history.
                logger.warning(
                    "seed_assistant_message for task %s normalized to "
                    "empty content and was not persisted",
                    task.id,
                )

        db.commit()
        db.refresh(task)

        runtime_context = agent_runtime_service._task_runtime_context(
            task_id=int(task.id),
            user_id=int(task.user_id),
            source=task.source,
        )
        release_db_connection_if_clean(db)
        try:
            await create_task_extensions(
                runtime_context,
                runtime_extension_requests,
            )
        except TaskRuntimeExtensionError as exc:
            task_id = int(task.id)
            try:
                _compensate_failed_task_extension_create(db, task_id=task_id)
            except Exception:
                logger.exception(
                    "Failed to compensate task %s after runtime extension "
                    "creation failure",
                    task_id,
                )
            agent_runtime_service.get_agent_manager(http_request).remove_agent(
                task_id, int(user.id)
            )
            if isinstance(exc.cause, TaskRuntimeClientError):
                status_code = exc.cause.status_code
                detail = exc.cause.detail
            else:
                status_code = 503
                detail = "Service unavailable"
                logger.exception(
                    "Task runtime extension creation failed for task %s",
                    task_id,
                )
            raise HTTPException(status_code=status_code, detail=detail) from exc

        # Public metadata is optional decoration on the create response. The
        # binding has already been persisted successfully, so creation degrades
        # to an empty mapping here; the dedicated GET endpoint remains
        # fail-closed because metadata is its primary response.
        runtime_extensions_status = "complete"
        runtime_extensions_omitted: list[str] = []
        try:
            metadata_result = await get_task_runtime_public_metadata(runtime_context)
            runtime_extensions = metadata_result.extensions
            runtime_extensions_status = metadata_result.status
            runtime_extensions_omitted = list(metadata_result.omitted_extensions)
        except TaskRuntimeExtensionError:
            logger.warning(
                "Failed to load public runtime metadata for task %s",
                task.id,
                exc_info=True,
            )
            runtime_extensions = {}
            runtime_extensions_status = "failed"

        return TaskCreateResponse(
            task_id=task.id,
            title=task.title,
            status=task.status.value,
            created_at=format_datetime_for_api(task.created_at)
            if task.created_at
            else None,
            model_id=task.model_id,
            small_fast_model_id=task.small_fast_model_id,
            visual_model_id=task.visual_model_id,
            compact_model_id=task.compact_model_id,
            model_name=task.model_name,
            small_fast_model_name=task.small_fast_model_name,
            visual_model_name=task.visual_model_name,
            compact_model_name=task.compact_model_name,
            execution_mode=task.execution_mode,
            channel_id=task.channel_id,
            channel_name=task.channel_name,
            agent_id=task.agent_id,
            agent_name=task.agent.name if task.agent else None,
            agent_logo_url=task.agent.logo_url if task.agent else None,
            run_id=task.run_id,
            state_version=int(task.state_version or 0),
            control_state=str(task.control_state or "idle"),
            runtime_extensions=runtime_extensions,
            runtime_extensions_status=runtime_extensions_status,
            runtime_extensions_omitted=runtime_extensions_omitted,
            connector_runtime_requirements=connector_runtime_requirements,
        )

    except HTTPException:
        raise
    except AutoModelUnavailableError as exc:
        raise HTTPException(
            status_code=409,
            detail=client_error_message(ClientErrorCode.AUTO_MODEL_UNAVAILABLE),
        ) from exc
    except ConnectorRuntimeError as exc:
        raise HTTPException(
            status_code=exc.status_code, detail=exc.safe_message
        ) from exc
    except KnowledgeBaseScopeError as exc:
        # Symmetric with the ConnectorRuntimeError arm above, and the only
        # place anything reads this error's status_code/safe_message: the
        # typed knowledge-base scope error already carries the status it
        # wants (503, "resolution failed", retryable) and a message that is
        # safe to hand a caller, so map both through rather than let the
        # blanket handler below flatten it into a 500 built from str(exc).
        # ``exc.code``/``exc.details`` are diagnostic and go to the log only.
        #
        # No step inside this endpoint resolves the team knowledge-base
        # layer today (resolution happens per search call, on the run path),
        # so this arm mirrors the two typed re-raises on the tool-build path
        # in ``factory.py`` and ``knowledge_tools.py``: it exists so that a
        # future in-request resolution surfaces the seam's own 503 instead
        # of being silently reclassified as an internal error.
        #
        # The asymmetry with the ConnectorRuntimeError arm above is real
        # and deliberate: that arm has a live producer inside this endpoint
        # (``resolve_agent_runtime_requirements``), this one has
        # none. It is kept because what the two arms share is the
        # failure-path contract, not the producer: both errors carry their
        # own status and a caller-safe message, and the blanket handler
        # below turns anything it does not name into a 500 built from
        # ``str(exc)``. Dropping this arm would make the first in-request
        # producer -- the run-path resolution moving earlier, or a
        # save-time validation added here -- answer 500 with a raw
        # exception string, silently. The test beside it injects the raise
        # for the same reason: what is pinned is this funnel's
        # classification, not any particular producer.
        logger.warning(
            "Knowledge base scope unavailable during task creation "
            "(code=%s, details=%s)",
            exc.code,
            exc.details,
        )
        raise HTTPException(
            status_code=exc.status_code, detail=exc.safe_message
        ) from exc
    except Exception as e:
        logger.error(f"Create task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.get("/tasks")
async def get_tasks(
    page: int = 1,
    per_page: int = 10,
    search: Optional[str] = None,
    agent_type: Optional[str] = None,
    exclude_agent_type: Optional[str] = None,
    execution_mode: Optional[str] = None,
    exclude_execution_mode: Optional[str] = None,
    include_hidden: bool = False,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get tasks list with pagination"""
    try:
        # Run synchronous database queries in thread pool to avoid blocking event loop
        def _get_tasks_sync() -> Dict[str, Any]:
            # Build base query - filter by current user, unless admin
            if user.is_admin:
                # Admin can see all tasks - include user relationship for admin
                from sqlalchemy.orm import joinedload

                query = db.query(Task).options(joinedload(Task.user))
            else:
                # Regular users can only see their own tasks
                query = db.query(Task).filter(Task.user_id == user.id)

            if not include_hidden:
                query = query.filter(Task.is_visible.is_(True))

            # Apply search filter if provided
            if search:
                query = query.filter(Task.title.ilike(f"%{search}%"))

            # Apply agent type filter if provided
            if agent_type:
                from ..models.task import AgentType

                try:
                    agent_type_enum = AgentType(agent_type)
                    if agent_type_enum.value == AgentType.STANDARD.value:
                        # For STANDARD agent type, include both 'standard' and NULL values
                        query = query.filter(
                            (Task.agent_type == agent_type_enum.value)
                            | (Task.agent_type.is_(None))
                        )
                    else:
                        # For other agent types, filter by exact value
                        query = query.filter(Task.agent_type == agent_type_enum.value)
                except ValueError:
                    # Invalid agent type, ignore filter
                    pass

            # Apply agent type exclusion filter if provided
            if exclude_agent_type:
                from ..models.task import AgentType

                try:
                    exclude_type_enum = AgentType(exclude_agent_type)
                    if exclude_type_enum.value == AgentType.STANDARD.value:
                        # Exclude STANDARD agent type (both 'standard' and NULL)
                        query = query.filter(
                            (Task.agent_type != exclude_type_enum.value)
                            & (Task.agent_type.isnot(None))
                        )
                    else:
                        # Exclude specific agent type
                        query = query.filter(Task.agent_type != exclude_type_enum.value)
                except ValueError:
                    # Invalid agent type, ignore filter
                    pass

            # Apply execution mode filter if provided
            if execution_mode:
                query = query.filter(Task.execution_mode == execution_mode)
            elif exclude_execution_mode:
                query = query.filter(Task.execution_mode != exclude_execution_mode)

            # Get total count
            total = query.count()

            # Apply pagination
            offset = (page - 1) * per_page
            query = (
                query.order_by(Task.created_at.desc()).offset(offset).limit(per_page)
            )
            tasks_query = query.all()

            # Batch fetch agents for tasks with agent_id
            agent_ids = {task.agent_id for task in tasks_query if task.agent_id}
            agents_map = {}
            if agent_ids:
                agents = db.query(Agent).filter(Agent.id.in_(agent_ids)).all()
                agents_map = {agent.id: agent for agent in agents}

            # Channel names are user-defined, so clients need the persisted type
            # to render a reliable platform indicator without guessing from text.
            channel_ids = {
                task.channel_id for task in tasks_query if task.channel_id is not None
            }
            channels_map = {}
            if channel_ids:
                channels = (
                    db.query(UserChannel.id, UserChannel.channel_type)
                    .filter(UserChannel.id.in_(channel_ids))
                    .all()
                )
                channels_map = {
                    channel_id: channel_type for channel_id, channel_type in channels
                }

            # Convert Task objects to dictionaries for JSON serialization
            tasks = []
            for task in tasks_query:
                try:
                    # Get the raw status value from the database
                    if hasattr(task, "status") and task.status is not None:
                        if hasattr(task.status, "value"):
                            status_value = task.status.value
                        else:
                            status_value = str(task.status)
                    else:
                        status_value = "unknown"

                    task_data = {
                        "task_id": task.id,
                        "title": task.title,
                        "status": status_value,
                        "run_id": task.run_id,
                        "state_version": int(task.state_version or 0),
                        "control_state": str(task.control_state or "idle"),
                        "created_at": format_datetime_for_api(task.created_at),
                        "updated_at": format_datetime_for_api(task.updated_at),
                        "model_id": task.model_id,
                        "small_fast_model_id": task.small_fast_model_id,
                        "visual_model_id": task.visual_model_id,
                        "compact_model_id": task.compact_model_id,
                        "model_name": task.model_name,
                        "small_fast_model_name": task.small_fast_model_name,
                        "visual_model_name": task.visual_model_name,
                        "execution_mode": task.execution_mode,
                        "input_tokens": task.input_tokens or 0,
                        "output_tokens": task.output_tokens or 0,
                        "total_tokens": task.total_tokens or 0,
                        "llm_calls": task.llm_calls or 0,
                        "agent_id": task.agent_id,
                        "channel_id": task.channel_id,
                        "channel_name": task.channel_name,
                        "channel_type": channels_map.get(task.channel_id),
                    }

                    if task.agent_id and task.agent_id in agents_map:
                        task_data["agent_logo_url"] = agents_map[task.agent_id].logo_url
                        task_data["agent_name"] = agents_map[task.agent_id].name

                    # Include user information for admin users
                    if user.is_admin:
                        task_data["user_id"] = task.user_id
                        task_data["username"] = (
                            task.user.username if task.user else "Unknown"
                        )

                    tasks.append(task_data)
                except Exception as e:
                    logger.warning(f"Error processing task {task.id}: {e}")
                    continue

            # Calculate pagination metadata
            total_pages = (total + per_page - 1) // per_page

            return {
                "tasks": tasks,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_count": total,
                    "total_pages": total_pages,
                    "has_next": page < total_pages,
                    "has_prev": page > 1,
                },
            }

        # Execute in thread pool to avoid blocking
        result = await asyncio.to_thread(_get_tasks_sync)

        return result
    except Exception as e:
        logger.error(f"Get tasks failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.get("/task/{task_id}")
async def get_task(
    task_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get task details"""
    try:
        # Run synchronous database queries in thread pool to avoid blocking event loop
        def _get_task_sync() -> Dict[str, Any]:
            # Admin can see any task, regular users can only see their own
            if user.is_admin:
                task = db.query(Task).filter(Task.id == task_id).first()
            else:
                task = (
                    db.query(Task)
                    .filter(Task.id == task_id, Task.user_id == user.id)
                    .first()
                )
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")

            cache_key = web_task_detail_key(task_id)
            task_updated_at = cache_version_token(task.updated_at)

            # Get the raw status value safely
            if hasattr(task, "status") and task.status is not None:
                if hasattr(task.status, "value"):
                    status_value = task.status.value
                else:
                    status_value = str(task.status)
            else:
                status_value = "unknown"

            # Get DAG execution data
            dag_data = None
            from ..models.task import DAGExecution

            dag_execution = (
                db.query(DAGExecution).filter(DAGExecution.task_id == task_id).first()
            )
            dag_updated_at = (
                cache_version_token(dag_execution.updated_at) if dag_execution else None
            )
            activity_ids: tuple[int, int] | None = None
            cached = cache_get(cache_key)
            if (
                isinstance(cached, dict)
                and cached.get("updated_at") == task_updated_at
                and cached.get("dag_updated_at") == dag_updated_at
            ):
                if task.status in _TERMINAL_CACHE_STATUSES:
                    return cast(Dict[str, Any], cached["response"])
                activity_ids = _get_task_activity_ids(db, task_id)
                if cached.get("max_trace_event_id") == int(
                    activity_ids[0]
                ) and cached.get("max_chat_message_id") == int(activity_ids[1]):
                    return cast(Dict[str, Any], cached["response"])

            if task.status not in _TERMINAL_CACHE_STATUSES and activity_ids is None:
                activity_ids = _get_task_activity_ids(db, task_id)

            if dag_execution:
                dag_data = {
                    "phase": dag_execution.phase.value if dag_execution.phase else None,
                    "current_plan": dag_execution.current_plan,
                    "created_at": safe_timestamp_to_unix(dag_execution.created_at)
                    if dag_execution.created_at
                    else None,
                    "updated_at": safe_timestamp_to_unix(dag_execution.updated_at)
                    if dag_execution.updated_at
                    else None,
                }

            # If model_id columns are not populated (legacy rows), best-effort resolve them
            # from stored provider-facing model_name values.
            llm_ids = agent_runtime_service.get_agent_manager()._get_task_llm_ids(
                task, db
            )
            model_id, small_fast_model_id, visual_model_id, compact_model_id = llm_ids
            waiting_question = None
            waiting_interactions = None
            if task.status == TaskStatus.WAITING_FOR_USER:
                waiting_question, waiting_interactions = (
                    get_pending_interaction_question(db, task)
                )

            # Fetch agent info if agent relationship is available
            agent_name = task.agent.name if task.agent else None
            agent_logo_url = task.agent.logo_url if task.agent else None

            model_usage = aggregate_token_usage_by_model(task.token_usage_details)
            media_usage = aggregate_media_usage_by_model(task.token_usage_details)
            response = {
                "task_id": task.id,
                "title": task.title,
                "description": task.description,
                "status": status_value,
                "run_id": task.run_id,
                "state_version": int(task.state_version or 0),
                "control_state": str(task.control_state or "idle"),
                "created_at": format_datetime_for_api(task.created_at),
                "updated_at": format_datetime_for_api(task.updated_at),
                "model_id": model_id,
                "small_fast_model_id": small_fast_model_id,
                "visual_model_id": visual_model_id,
                "compact_model_id": compact_model_id,
                "model_name": task.model_name,
                "small_fast_model_name": task.small_fast_model_name,
                "visual_model_name": task.visual_model_name,
                "compact_model_name": task.compact_model_name,
                "dag_data": dag_data,
                "input_tokens": task.input_tokens or 0,
                "output_tokens": task.output_tokens or 0,
                "total_tokens": task.total_tokens or 0,
                "llm_calls": task.llm_calls or 0,
                "cached_input_tokens": sum(
                    entry["cached_input_tokens"] for entry in model_usage
                ),
                "cache_write_input_tokens": sum(
                    entry["cache_write_input_tokens"] for entry in model_usage
                ),
                "model_usage": model_usage,
                # No media_calls companion: the client derives its own count
                # from these rows, and a second server-side reduction would be
                # a duplicate that can drift. Deliberately no cross-unit
                # quantity total either — summing images + seconds + characters
                # produces a number with no meaning.
                "media_usage": media_usage,
                "agent_id": task.agent_id,
                "agent_name": agent_name,
                "agent_logo_url": agent_logo_url,
                "channel_id": task.channel_id,
                "channel_name": task.channel_name,
                "waiting_question": waiting_question,
                "waiting_interactions": waiting_interactions,
            }
            cache_set(
                cache_key,
                {
                    "updated_at": task_updated_at,
                    "dag_updated_at": dag_updated_at,
                    "max_trace_event_id": (
                        activity_ids[0] if activity_ids is not None else None
                    ),
                    "max_chat_message_id": (
                        activity_ids[1] if activity_ids is not None else None
                    ),
                    "response": response,
                },
                ttl_seconds=task_cache_ttl_seconds(),
            )
            return response

        # Execute in thread pool to avoid blocking
        return await asyncio.to_thread(_get_task_sync)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.get("/task/{task_id}/status")
async def get_task_status(
    task_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get task status"""
    try:
        # Run synchronous database queries in thread pool to avoid blocking event loop
        def _get_task_status_sync() -> Dict[str, Any]:
            # Admin can see any task, regular users can only see their own
            if user.is_admin:
                task = db.query(Task).filter(Task.id == task_id).first()
            else:
                task = (
                    db.query(Task)
                    .filter(Task.id == task_id, Task.user_id == user.id)
                    .first()
                )
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")

            cache_key = web_task_status_key(task_id)
            task_updated_at = cache_version_token(task.updated_at)
            activity_ids: tuple[int, int] | None = None
            cached = cache_get(cache_key)
            if isinstance(cached, dict) and cached.get("updated_at") == task_updated_at:
                if task.status in _TERMINAL_CACHE_STATUSES:
                    return cast(Dict[str, Any], cached["response"])
                activity_ids = _get_task_activity_ids(db, task_id)
                if cached.get("max_trace_event_id") == int(
                    activity_ids[0]
                ) and cached.get("max_chat_message_id") == int(activity_ids[1]):
                    return cast(Dict[str, Any], cached["response"])

            if task.status not in _TERMINAL_CACHE_STATUSES and activity_ids is None:
                activity_ids = _get_task_activity_ids(db, task_id)

            # Get the raw status value safely
            if hasattr(task, "status") and task.status is not None:
                if hasattr(task.status, "value"):
                    status_value = task.status.value
                else:
                    status_value = str(task.status)
            else:
                status_value = "unknown"

            llm_ids = agent_runtime_service.get_agent_manager()._get_task_llm_ids(
                task, db
            )
            model_id, small_fast_model_id, visual_model_id, compact_model_id = llm_ids
            waiting_question = None
            waiting_interactions = None
            if task.status == TaskStatus.WAITING_FOR_USER:
                waiting_question, waiting_interactions = (
                    get_pending_interaction_question(db, task)
                )

            # Fetch agent info if agent relationship is available
            agent_name = task.agent.name if task.agent else None
            agent_logo_url = task.agent.logo_url if task.agent else None

            response = {
                "task_id": task.id,
                "title": task.title,
                "status": status_value,
                "run_id": task.run_id,
                "state_version": int(task.state_version or 0),
                "control_state": str(task.control_state or "idle"),
                "created_at": format_datetime_for_api(task.created_at),
                "updated_at": format_datetime_for_api(task.updated_at),
                "model_id": model_id,
                "small_fast_model_id": small_fast_model_id,
                "visual_model_id": visual_model_id,
                "compact_model_id": compact_model_id,
                "model_name": task.model_name,
                "small_fast_model_name": task.small_fast_model_name,
                "visual_model_name": task.visual_model_name,
                "compact_model_name": task.compact_model_name,
                "input_tokens": task.input_tokens or 0,
                "output_tokens": task.output_tokens or 0,
                "total_tokens": task.total_tokens or 0,
                "llm_calls": task.llm_calls or 0,
                "agent_id": task.agent_id,
                "agent_name": agent_name,
                "agent_logo_url": agent_logo_url,
                "channel_id": task.channel_id,
                "channel_name": task.channel_name,
                "waiting_question": waiting_question,
                "waiting_interactions": waiting_interactions,
            }
            cache_set(
                cache_key,
                {
                    "updated_at": task_updated_at,
                    "max_trace_event_id": (
                        activity_ids[0] if activity_ids is not None else None
                    ),
                    "max_chat_message_id": (
                        activity_ids[1] if activity_ids is not None else None
                    ),
                    "response": response,
                },
                ttl_seconds=task_cache_ttl_seconds(),
            )
            return response

        # Execute in thread pool to avoid blocking
        return await asyncio.to_thread(_get_task_status_sync)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get task status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.put("/task/{task_id}")
async def update_task(
    task_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Update task details."""
    try:
        data = await request.json()
        title = data.get("title")

        if not title:
            raise HTTPException(status_code=400, detail="Title is required")

        # Verify task exists and belongs to user
        if user.is_admin:
            task = db.query(Task).filter(Task.id == task_id).first()
        else:
            task = (
                db.query(Task)
                .filter(Task.id == task_id, Task.user_id == user.id)
                .first()
            )

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        task.title = title
        db.commit()
        invalidate_task_cache(task_id)

        return {"status": "success", "message": "Task updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.get("/task/{task_id}/runtime-extensions")
async def get_task_runtime_extensions(
    task_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return live, provider-approved runtime metadata for one task.

    Metadata is re-read from every registered runtime extension on each call
    rather than served from the task row, so it reflects provider state now
    instead of at creation time. Only fields a provider explicitly publishes
    are returned; provider-internal state and secrets are never exposed.

    Access follows normal task ownership: an admin may read any task, other
    users only their own, and an unreadable or missing task is a 404.

    Unlike ``POST /task/create``, where metadata is optional decoration, this
    endpoint is fail-closed: a provider error is surfaced as its approved
    client error (400/403) or a generic 500, never as partial data.

    Response fields:
        ``task_id``: the task the metadata belongs to.
        ``runtime_extensions``: extension name to that provider's public
            metadata object.
        ``runtime_extensions_status``: ``complete`` when every registered
            provider's metadata is included, ``truncated`` when some was
            dropped to keep the response under its aggregate size cap.
        ``runtime_extensions_omitted``: names dropped for that size cap.
    """

    query = db.query(Task).filter(Task.id == task_id)
    if not user.is_admin:
        query = query.filter(Task.user_id == user.id)
    task = query.first()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    context = agent_runtime_service._task_runtime_context(
        task_id=int(task.id),
        user_id=int(task.user_id),
        source=task.source,
    )
    release_db_connection_if_clean(db)
    try:
        metadata_result = await get_task_runtime_public_metadata(context)
    except TaskRuntimeExtensionError as exc:
        if isinstance(exc.cause, TaskRuntimeClientError):
            status_code = exc.cause.status_code
            detail = exc.cause.detail
        else:
            status_code = 500
            detail = "Internal server error"
            logger.exception(
                "Failed to load public runtime metadata for task %s",
                task_id,
            )
        raise HTTPException(status_code=status_code, detail=detail) from exc
    return {
        "task_id": task_id,
        "runtime_extensions": metadata_result.extensions,
        "runtime_extensions_status": metadata_result.status,
        "runtime_extensions_omitted": list(metadata_result.omitted_extensions),
    }


def _connector_runtime_error_response(exc: ConnectorRuntimeError) -> JSONResponse:
    """Render a connector-runtime failure in this endpoint's error envelope.

    Mirrors the ``{"error": {"code", "message", "details"}}`` shape
    ``_raise_v1_connector_runtime_error`` uses for the /v1 surface
    (``api/v1/tasks.py``) -- only the envelope is shared, not ``V1ErrorCode``,
    which is a separate SDK-facing contract this endpoint does not
    participate in. The status code always comes from ``exc.status_code``,
    never recomputed from ``exc.code`` via ``_status_for_code``: that helper
    cannot produce 503, and both this endpoint's own conditional-update
    failure and a team-scope resolution failure construct their
    ``ConnectorRuntimeError`` with an explicit ``status_code=503``.
    """

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.safe_message,
                "details": exc.to_public_error()["details"],
            }
        },
    )


@chat_router.get(
    "/agent/{agent_id}/connector-runtime-requirements",
    response_model=ConnectorRuntimeRequirementsModel,
)
async def get_agent_connector_runtime_requirements(
    agent_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ConnectorRuntimeRequirementsModel:
    """Report the runtime inputs a prospective task would need, before one
    exists.

    Reuses the same predicate ``POST /task/create`` applies to an agent id
    (``_load_agent_for_task_create``) rather than a second authorization
    path for the same resource, so a caller who could create a task with
    this agent sees exactly the same "not found" boundary here that they
    would hit on that call. Lives on the chat router, next to the
    task-keyed sibling below and the existing task-keyed
    ``/task/{task_id}/runtime-extensions``, rather than under
    ``/api/agents`` -- this endpoint has no consumer outside chat.

    There is no task yet, so every reported input is unsatisfied and the
    connector team scope is whatever ``resolve_agent_selected_connectors``
    derives from the agent's own team, never a value this endpoint passes
    in itself.
    """

    agent = _load_agent_for_task_create(db, user, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found or access denied")
    try:
        _refs, requirements = resolve_agent_runtime_requirements(
            db=db, agent=agent, connector_user_id=int(user.id)
        )
    except ConnectorRuntimeError as exc:
        raise HTTPException(
            status_code=exc.status_code, detail=exc.safe_message
        ) from exc
    return requirements


@chat_router.get(
    "/task/{task_id}/connector-runtime-requirements",
    response_model=ConnectorRuntimeRequirementsModel,
)
async def get_task_connector_runtime_requirements(
    task_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ConnectorRuntimeRequirementsModel:
    """Report which of a task's declared connector runtime inputs already
    have a value.

    Access is plain task ownership -- ``Task.user_id == current_user.id`` in
    the same query that loads the task, unlike
    ``/task/{task_id}/runtime-extensions`` above, which additionally lets an
    admin read any task; this endpoint does not extend that exception.
    A task that does not exist or is not the caller's own is a uniform 404.

    Pure read: never writes, and never asserts that a required value is
    present -- that assertion belongs to the per-turn gate that runs later,
    not to this report.
    """

    task = db.query(Task).filter(Task.id == task_id, Task.user_id == user.id).first()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    # The connector scope this report is built from must be the scope a
    # turn would actually run under, so the agent is resolved by the same
    # two calls the per-turn tool build makes for this task, in the same
    # order. Reading the agent row directly would key team-shared
    # connectors on the raw row's team even where the runtime resolves the
    # agent to None, and resolving with no workforce runtime would drop
    # the team of a workforce manager agent whose run the runtime does
    # find. A report that says what a turn will need can afford neither
    # the over-report nor the under-report.
    workforce_runtime = resolve_workforce_task_runtime(db, task)
    agent = agent_runtime_service._load_agent_for_task_runtime(
        db, task, workforce_runtime
    )
    try:
        return build_task_runtime_requirements(db=db, task=task, agent=agent)
    except ConnectorRuntimeError as exc:
        raise HTTPException(
            status_code=exc.status_code, detail=exc.safe_message
        ) from exc


@chat_router.post(
    "/task/{task_id}/connector-runtime-values",
    response_model=ConnectorRuntimeRequirementsModel,
)
def post_task_connector_runtime_values(
    task_id: int,
    request: ConnectorRuntimeValuesRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ConnectorRuntimeRequirementsModel | JSONResponse:
    """Accept ``context`` values for a task's connectors, merged key by key.

    A key not yet stored is written; one already stored with the identical
    value is a no-op; one already stored with a different value fails the
    whole request with no partial write (``runtime_context_immutable``,
    409). There is no override switch: a stored value is never replaced.
    ``secrets``/``auth_selector`` are out of scope this phase and rejected
    at the request-shape level (``extra="forbid"``), not accepted and
    ignored.

    Access is the same plain task ownership the task-keyed read endpoint
    applies -- ``Task.user_id == current_user.id`` in the query that loads
    the task, with no admin exception -- so a task that does not exist and
    one that is not the caller's own answer the same 404.

    On success, the response is the same requirements report the read
    endpoints return, reflecting exactly what was just written -- not the
    request's own echo, and not whatever the read endpoints would have
    said before this call. A 200 here means only that the submitted keys
    were merged in; it says nothing about whether every required input is
    now present, which is what the response's own ``satisfied`` fields
    answer.

    Every ``ConnectorRuntimeError`` raised anywhere on this request path --
    validation, a stored-value conflict, a concurrent write racing this one
    to the same row, and the agent/workforce resolution that runs before
    the write -- is rendered through ``_connector_runtime_error_response``
    rather than the plain-``detail`` ``HTTPException`` the two read
    endpoints above use, because a caller needs the structured
    ``code``/``details.reason`` to decide how to recover (retry, refresh
    and drop already-satisfied keys, or give up), not just a status code.

    Anything else -- a resolver raising on a malformed workforce snapshot,
    a driver error mid-write, a failing ``db.commit()`` -- is rolled back
    and re-raised unchanged, so it ends as a bare 500. Such a failure is
    deliberately not dressed up in this envelope: the envelope's ``code``
    is a contract about a condition the caller can act on, and an
    unclassified fault is not one. What the rollback guarantees either way
    is that no partial batch survives the request.
    """

    task = db.query(Task).filter(Task.id == task_id, Task.user_id == user.id).first()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    # The connector scope this endpoint writes into, and reports on, must
    # be the scope a turn would actually run under, so the agent is
    # resolved by the same two calls the per-turn tool build makes for
    # this task, in the same order -- the identical pair the task-keyed
    # read endpoint above uses, so neither endpoint can answer with a
    # connector set the other would not. Reading the agent row directly
    # would key team-shared connectors on the raw row's team even where
    # the runtime resolves the agent to None, and resolving with no
    # workforce runtime would drop the team of a workforce manager agent
    # whose run the runtime does find. Here the scope decides not only
    # what the response lists but which connectors the caller may write
    # to at all, so neither the over-report nor the under-report is
    # acceptable.
    try:
        workforce_runtime = resolve_workforce_task_runtime(db, task)
        agent = agent_runtime_service._load_agent_for_task_runtime(
            db, task, workforce_runtime
        )
        requirements = apply_task_connector_runtime_context_values(
            db=db, task=task, agent=agent, payload_items=request.items
        )
        db.commit()
    except ConnectorRuntimeError as exc:
        db.rollback()
        return _connector_runtime_error_response(exc)
    except Exception:
        # Not a fallback: the exception keeps travelling and this request
        # still ends as a bare 500. The rollback is the whole point --
        # a commit that raises leaves the session's transaction open with
        # this batch's rows already flushed into it, and every other
        # failure past the flush leaves the same thing behind.
        db.rollback()
        raise
    return requirements


@chat_router.delete("/task/{task_id}")
async def delete_task(
    task_id: int,
    request: Any = None,
    # Admin escape hatch: delete the core task rows even when a runtime
    # extension that owns state for this task fails to release it. A plain
    # default (not ``Query(...)``) keeps this callable directly from internal
    # code and tests without picking up a truthy ``Query`` sentinel.
    force: bool = False,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Delete a task and all related data"""
    try:
        requester_user_id = int(user.id)
        is_admin = bool(user.is_admin)
        if force and not is_admin:
            raise HTTPException(
                status_code=403,
                detail="Force delete requires admin access",
            )
        release_db_connection_if_clean(db)
        task_snapshot = await asyncio.to_thread(
            _load_task_delete_snapshot_sync,
            task_id=task_id,
            requester_user_id=requester_user_id,
            is_admin=is_admin,
        )
        if task_snapshot is None:
            raise HTTPException(status_code=404, detail="Task not found")
        task_title, task_user_id, task_source, bound_extensions = task_snapshot
        runtime_context = agent_runtime_service._task_runtime_context(
            task_id=task_id,
            user_id=task_user_id,
            source=task_source,
        )

        try:
            # Only the providers this task actually bound to are dispatched, so
            # an unrelated broken extension cannot block deletion.
            unreleased = await delete_task_extensions(
                runtime_context,
                bound_extensions=bound_extensions,
                force=force,
            )
        except TaskRuntimeExtensionError as exc:
            logger.error(
                "Runtime extension cleanup failed; preserving task %s for retry",
                task_id,
                exc_info=True,
            )
            raise HTTPException(
                status_code=503,
                detail=("Runtime extension cleanup failed; the task was not deleted"),
            ) from exc
        if unreleased:
            logger.error(
                "Deleting task %s with unreleased runtime extension state for %s",
                task_id,
                ", ".join(unreleased),
            )

        deleted = await asyncio.to_thread(_delete_task_sync, task_id=task_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Task no longer exists")
        invalidate_task_cache(task_id)

        # Remove agent from manager if it exists
        agent_runtime_service.get_agent_manager(request).remove_agent(
            task_id, requester_user_id
        )

        from ..services.task_execution import background_task_manager
        from .websocket import manager

        connections = manager.detach_task_connections(task_id)

        async def _cleanup_runtime_state() -> None:
            await background_task_manager.cancel_task(task_id, timeout_seconds=0.05)
            for connection in list(connections):
                try:
                    await connection.close()
                except Exception as e:
                    logger.warning(f"Failed to close WebSocket connection: {e}")

        asyncio.create_task(_cleanup_runtime_state())

        logger.info(f"Task {task_id} deleted successfully")

        return {
            "success": True,
            "message": f"Task '{task_title}' deleted successfully",
            "task_id": task_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete task failed: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@chat_router.get("/workspace/{task_id}/files")
async def get_task_workspace_files(
    task_id: int,
    request: Any = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get all workspace files for a task"""
    try:
        # Run synchronous database queries in thread pool to avoid blocking event loop
        def _verify_task_sync() -> Task:
            # Verify task ownership - admin can access any task
            if user.is_admin:
                task = db.query(Task).filter(Task.id == task_id).first()
            else:
                task = (
                    db.query(Task)
                    .filter(Task.id == task_id, Task.user_id == user.id)
                    .first()
                )
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            return task

        # Execute database operations in thread pool to avoid blocking
        await asyncio.to_thread(_verify_task_sync)

        workspace_files = agent_runtime_service.get_agent_manager(
            request
        ).get_agent_workspace_files(task_id)
        return {
            "success": True,
            "task_id": task_id,
            "workspace_files": workspace_files,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Get workspace files failed for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.get("/workspace/{task_id}/output")
async def get_task_output_files(
    task_id: int,
    request: Any = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get output files for a task"""
    try:
        # Run synchronous database queries in thread pool to avoid blocking event loop
        def _verify_task_sync() -> Task:
            # Verify task ownership - admin can access any task
            if user.is_admin:
                task = db.query(Task).filter(Task.id == task_id).first()
            else:
                task = (
                    db.query(Task)
                    .filter(Task.id == task_id, Task.user_id == user.id)
                    .first()
                )
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            return task

        # Execute database operations in thread pool to avoid blocking
        await asyncio.to_thread(_verify_task_sync)

        agent_service = agent_runtime_service.get_agent_manager(request)
        output_files = agent_service.get_agent_output_files(task_id)
        return {
            "success": True,
            "task_id": task_id,
            "output_files": output_files,
            "file_count": len(output_files),
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Get output files failed for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
