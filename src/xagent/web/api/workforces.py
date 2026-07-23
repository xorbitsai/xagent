import logging
from datetime import timezone
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, or_
from sqlalchemy.orm import Session, selectinload

from ..auth_dependencies import get_current_user
from ..models.agent import Agent, is_workforce_generated_manager_agent
from ..models.database import get_db
from ..models.task import TaskStatus, TraceEvent
from ..models.user import User
from ..models.workforce import (
    Workforce,
    WorkforceAgent,
    WorkforceRun,
)
from ..services.agent_access import (
    AccessibleAgent,
    accessible_agent_permissions,
    list_accessible_published_agent_items,
)
from ..services.agent_team_scope import (
    AgentTeamScope,
    get_agent_team_scope,
    owns_agent,
)
from ..services.trace_message_storage import decode_trace_events_data
from ..services.workforce_access import (
    can_create_workforce,
    ensure_agent_access,
    ensure_workforce_access,
    filter_visible_workforces,
    resolve_create_scope,
)
from ..services.workforce_creator import create_workforce_from_prompt
from ..services.workforce_lifecycle import (
    acquire_workforce_lifecycle_fence,
    discard_draft_workforce,
)
from ..services.workforce_names import workforce_name_exists
from ..services.workforce_runs import create_workforce_run as start_workforce_run
from ..services.workforce_runtime import (
    cancel_active_workforce_runs,
    pause_workforce_tasks_after_archive,
)
from ..services.workforce_snapshot import (
    normalize_text,
    normalize_workforce_status,
    validate_workforce_for_run,
)
from ..services.workforce_workers import create_workforce_worker
from .public_trace_events import (
    DELEGATED_AGENT_TRACE_SOURCE,
    is_audit_only_trace_data,
    normalize_public_trace_event,
)

router = APIRouter(prefix="/api/workforces", tags=["workforces"])
logger = logging.getLogger(__name__)


class WorkforceWorkerInput(BaseModel):
    source_type: str = Field(default="existing")
    agent_id: int | None = None
    alias: str | None = Field(None, max_length=200)
    assignment_instructions: str = Field(..., min_length=1)
    enabled: bool = True
    sort_order: int | None = None
    canvas_position: dict[str, Any] | None = None


class WorkforceCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    manager_agent_id: int
    canvas_layout: dict[str, Any] | None = None
    workers: list[WorkforceWorkerInput] = Field(default_factory=list)


class WorkforcePromptCreateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class WorkforceUpdateRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = None
    manager_agent_id: int | None = None
    canvas_layout: dict[str, Any] | None = None


class WorkforceWorkerUpdateRequest(BaseModel):
    alias: str | None = Field(None, max_length=200)
    assignment_instructions: str | None = Field(None, min_length=1)
    enabled: bool | None = None
    sort_order: int | None = None
    canvas_position: dict[str, Any] | None = None


class WorkforceRunRequest(BaseModel):
    message: str = Field(..., min_length=1)
    files: list[str] = Field(default_factory=list)
    execution_mode: str | None = None
    is_preview: bool = False
    is_visible: bool = True


def _field_supplied(model: BaseModel, field_name: str) -> bool:
    return field_name in model.model_fields_set


def _load_workforce(db: Session, workforce_id: int) -> Workforce | None:
    return (
        db.query(Workforce)
        .options(
            selectinload(Workforce.manager_agent),
            selectinload(Workforce.workers).selectinload(WorkforceAgent.agent),
        )
        .filter(Workforce.id == workforce_id)
        .first()
    )


def _reload_workforce(db: Session, workforce: Workforce) -> Workforce:
    workforce_id = int(workforce.id)
    loaded = _load_workforce(db, workforce_id)
    if loaded is None:
        raise HTTPException(status_code=404, detail="Workforce not found")
    return loaded


def _agent_status_value(agent: Agent) -> str:
    value = getattr(agent.status, "value", None)
    if isinstance(value, str):
        return value
    return str(agent.status or "")


def _serialize_datetime(value: Any) -> str | None:
    return value.isoformat() if value else None


def _serialize_agent(
    agent: Agent,
    user: User | None = None,
    scope: AgentTeamScope | None = None,
) -> dict[str, Any]:
    item = {
        "id": agent.id,
        "name": agent.name,
        "description": agent.description,
        "logo_url": agent.logo_url,
        "status": _agent_status_value(agent),
    }
    if user is None:
        return item

    is_owner = owns_agent(agent, int(user.id), scope)
    is_generated_manager = is_workforce_generated_manager_agent(agent)
    can_edit = is_owner and not is_generated_manager
    item.update(
        {
            "access": "owner" if is_owner else "policy",
            "readonly": not can_edit,
            "can_edit": can_edit,
            "can_publish": can_edit,
            "can_delete": can_edit,
        }
    )
    return item


def _serialize_accessible_agent_option(
    accessible_agent: AccessibleAgent,
) -> dict[str, Any]:
    item = _serialize_agent(accessible_agent.agent)
    item.update(accessible_agent_permissions(accessible_agent))
    return item


def _sorted_workers(workforce: Workforce) -> list[WorkforceAgent]:
    return sorted(
        workforce.workers,
        key=lambda item: (item.sort_order or 0, item.id or 0),
    )


def _serialize_worker(
    worker: WorkforceAgent,
    user: User | None = None,
    scope: AgentTeamScope | None = None,
) -> dict[str, Any]:
    return {
        "id": worker.id,
        "agent": _serialize_agent(worker.agent, user, scope),
        "alias": worker.alias,
        "assignment_instructions": worker.assignment_instructions,
        "source_type": worker.source_type,
        "template_id": worker.template_id,
        "enabled": worker.enabled,
        "sort_order": worker.sort_order,
        "canvas_position": worker.canvas_position,
        "created_at": _serialize_datetime(worker.created_at),
        "updated_at": _serialize_datetime(worker.updated_at),
    }


def _serialize_workforce_detail(
    workforce: Workforce,
    user: User | None = None,
    scope: AgentTeamScope | None = None,
) -> dict[str, Any]:
    return {
        "id": workforce.id,
        "name": workforce.name,
        "description": workforce.description,
        "status": workforce.status,
        "manager": _serialize_agent(workforce.manager_agent, user, scope),
        "workers": [
            _serialize_worker(worker, user, scope)
            for worker in _sorted_workers(workforce)
        ],
        "canvas_layout": workforce.canvas_layout,
        "scope_type": workforce.scope_type,
        "scope_id": workforce.scope_id,
        "owner_user_id": workforce.owner_user_id,
        "created_at": _serialize_datetime(workforce.created_at),
        "updated_at": _serialize_datetime(workforce.updated_at),
    }


def _serialize_workforce_list_item(
    workforce: Workforce,
    last_run: WorkforceRun | None,
) -> dict[str, Any]:
    return {
        "id": workforce.id,
        "name": workforce.name,
        "description": workforce.description,
        "status": workforce.status,
        "manager": {
            "id": workforce.manager_agent.id,
            "name": workforce.manager_agent.name,
            "logo_url": workforce.manager_agent.logo_url,
        },
        "worker_count": len(workforce.workers),
        "last_run": (
            {
                "id": last_run.id,
                "task_id": last_run.task_id,
                "status": last_run.status,
                "created_at": _serialize_datetime(last_run.created_at),
            }
            if last_run
            else None
        ),
        "created_at": _serialize_datetime(workforce.created_at),
        "updated_at": _serialize_datetime(workforce.updated_at),
    }


def _trace_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return float(value.timestamp())


_AGENT_EXECUTION_METADATA_FIELDS = (
    "agent_id",
    "agent_name",
    "worker_member_id",
    "worker_alias",
)


def _merge_agent_execution_metadata(
    metadata: dict[str, Any], data: dict[str, Any]
) -> None:
    """Keep the first immutable delegation identity observed in trace order."""

    for key in _AGENT_EXECUTION_METADATA_FIELDS:
        if key in data and key not in metadata:
            metadata[key] = data[key]


def _derive_agent_execution_status(
    trace_events: list[dict[str, Any]],
) -> str | None:
    """Infer a terminal worker status when its parent summary is missing."""

    status_aliases = {
        "completed": "completed",
        "failed": "failed",
        "interrupted": "interrupted",
        "waiting_for_user": "waiting_for_user",
    }
    for event in reversed(trace_events):
        event_type = str(event.get("event_type") or "")
        if event_type == "trace_error":
            return "failed"
        if event_type not in {
            "react_task_end",
            "task_completion",
            "dag_execute_end",
        }:
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            data = {}
        result = data.get("result")
        if not isinstance(result, dict):
            result = {}
        for candidate in (result.get("status"), data.get("status")):
            if isinstance(candidate, str):
                normalized = status_aliases.get(candidate.strip().lower())
                if normalized is not None:
                    return normalized
        success = result.get("success", data.get("success"))
        if success is False:
            return "failed"
        if success is True or event_type == "task_completion":
            return "completed"
    return None


def _serialize_agent_execution_traces(
    db: Session,
    *,
    task_id: int,
    worker_task_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = (
        db.query(TraceEvent)
        .filter(
            TraceEvent.task_id == task_id,
            TraceEvent.build_id == worker_task_id,
            TraceEvent.data["source"].as_string() == DELEGATED_AGENT_TRACE_SOURCE,
            TraceEvent.event_type != "agent_checkpoint",
        )
        .order_by(TraceEvent.timestamp, TraceEvent.id)
        .all()
    )
    if not events:
        raise HTTPException(status_code=404, detail="Agent execution not found")

    decoded = decode_trace_events_data(
        db,
        task_id=task_id,
        data_items=[event.data for event in events],
        strict=False,
    )
    public_events: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"worker_task_id": worker_task_id}
    for event, data in zip(events, decoded):
        if is_audit_only_trace_data(data):
            continue
        if isinstance(data, dict):
            _merge_agent_execution_metadata(metadata, data)
        event_type, public_data = normalize_public_trace_event(
            str(event.event_type), data
        )
        public_events.append(
            {
                "event_id": event.event_id,
                "event_type": event_type,
                "step_id": event.step_id,
                "timestamp": _trace_timestamp(event.timestamp),
                "data": public_data,
                "parent_event_id": event.parent_event_id,
            }
        )
    return public_events, metadata


def _load_latest_runs_by_workforce(
    db: Session,
    workforce_ids: list[int],
) -> dict[int, WorkforceRun]:
    if not workforce_ids:
        return {}

    ranked_runs = (
        db.query(
            WorkforceRun.id.label("id"),
            func.row_number()
            .over(
                partition_by=WorkforceRun.workforce_id,
                order_by=(WorkforceRun.created_at.desc(), WorkforceRun.id.desc()),
            )
            .label("rank"),
        )
        .filter(
            WorkforceRun.workforce_id.in_(workforce_ids),
            WorkforceRun.is_preview.is_(False),
        )
        .subquery()
    )
    latest_runs = (
        db.query(WorkforceRun)
        .join(ranked_runs, WorkforceRun.id == ranked_runs.c.id)
        .filter(ranked_runs.c.rank == 1)
        .all()
    )
    return {int(run.workforce_id): run for run in latest_runs}


def _ensure_unique_workforce_name(
    db: Session,
    workforce: Workforce | None,
    *,
    scope_type: str,
    scope_id: str,
    name: str,
) -> str:
    normalized_name = normalize_text(name, "name", required=True)
    if workforce_name_exists(
        db,
        scope_type=scope_type,
        scope_id=scope_id,
        name=normalized_name,
        exclude_workforce_id=int(workforce.id) if workforce is not None else None,
    ):
        raise HTTPException(status_code=409, detail="Workforce name already exists")
    return normalized_name


def _ensure_publish_state_mutable(workforce: Workforce) -> None:
    if workforce.status == "archived":
        raise HTTPException(
            status_code=409,
            detail="Archived workforce cannot change publish state",
        )


def _validate_if_active(db: Session, user: User, workforce: Workforce) -> None:
    if workforce.status != "active":
        return
    db.flush()
    db.expire(workforce, ["manager_agent", "workers"])
    validate_workforce_for_run(db, user, workforce)


def _load_worker(db: Session, workforce: Workforce, member_id: int) -> WorkforceAgent:
    worker = (
        db.query(WorkforceAgent)
        .options(selectinload(WorkforceAgent.agent))
        .filter(
            WorkforceAgent.id == member_id,
            WorkforceAgent.workforce_id == workforce.id,
        )
        .first()
    )
    if worker is None:
        raise HTTPException(status_code=404, detail="Workforce worker not found")
    return worker


@router.get("")
async def list_workforces(
    search: str = "",
    page: int = 1,
    size: int = 20,
    status: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    if page < 1 or size < 1 or size > 100:
        raise HTTPException(status_code=400, detail="Invalid pagination parameters")

    query = db.query(Workforce)
    normalized_search = search.strip()
    if normalized_search:
        query = query.filter(
            or_(
                Workforce.name.ilike(f"%{normalized_search}%"),
                Workforce.description.ilike(f"%{normalized_search}%"),
            )
        )
    if status:
        query = query.filter(Workforce.status == normalize_workforce_status(status))
    query = filter_visible_workforces(db, user, query)

    total = query.count()
    offset = (page - 1) * size
    paged_workforces = (
        query.options(
            selectinload(Workforce.manager_agent),
            selectinload(Workforce.workers).selectinload(WorkforceAgent.agent),
        )
        .order_by(Workforce.updated_at.desc(), Workforce.id.desc())
        .offset(offset)
        .limit(size)
        .all()
    )
    latest_runs = _load_latest_runs_by_workforce(
        db,
        [int(workforce.id) for workforce in paged_workforces],
    )
    return {
        "items": [
            _serialize_workforce_list_item(
                workforce,
                latest_runs.get(int(workforce.id)),
            )
            for workforce in paged_workforces
        ],
        "total": total,
        "page": page,
        "size": size,
        "pages": (total + size - 1) // size,
    }


@router.post("")
async def create_workforce(
    request: WorkforceCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    scope_type, scope_id = resolve_create_scope(db, user)
    if not can_create_workforce(db, user, scope_type, scope_id):
        raise HTTPException(status_code=403, detail="Access denied")

    name = _ensure_unique_workforce_name(
        db,
        None,
        scope_type=scope_type,
        scope_id=scope_id,
        name=request.name,
    )
    manager_agent = ensure_agent_access(
        db.query(Agent).filter(Agent.id == request.manager_agent_id).first(),
        user,
        db,
        require_published=True,
    )

    try:
        workforce = Workforce(
            owner_user_id=int(user.id),
            scope_type=scope_type,
            scope_id=scope_id,
            name=name,
            description=normalize_text(request.description, "description"),
            manager_agent_id=int(manager_agent.id),
            status="draft",
            canvas_layout=request.canvas_layout,
        )
        db.add(workforce)
        db.flush()

        for worker_input in request.workers:
            create_workforce_worker(
                db,
                workforce,
                user,
                source_type=worker_input.source_type,
                assignment_instructions=worker_input.assignment_instructions,
                alias=worker_input.alias,
                agent_id=worker_input.agent_id,
                enabled=worker_input.enabled,
                sort_order=worker_input.sort_order,
                canvas_position=worker_input.canvas_position,
            )

        db.commit()
    except Exception:
        db.rollback()
        raise

    return _serialize_workforce_detail(
        _reload_workforce(db, workforce), user, get_agent_team_scope(db, int(user.id))
    )


@router.post("/from-prompt")
async def create_workforce_from_prompt_endpoint(
    request: WorkforcePromptCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    result = await create_workforce_from_prompt(db, user, prompt=request.prompt)
    workforce = _reload_workforce(db, result.workforce)
    return _serialize_workforce_detail(
        workforce, user, get_agent_team_scope(db, int(user.id))
    )


@router.get("/agent-options")
async def list_workforce_agent_options(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[dict[str, Any]]:
    return [
        _serialize_accessible_agent_option(agent)
        for agent in list_accessible_published_agent_items(
            db,
            user,
            purpose="workforce_select",
        )
    ]


@router.get("/{workforce_id}")
async def get_workforce(
    workforce_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = ensure_workforce_access(
        db,
        user,
        _load_workforce(db, workforce_id),
        action="view",
    )
    return _serialize_workforce_detail(
        workforce, user, get_agent_team_scope(db, int(user.id))
    )


@router.get("/{workforce_id}/runs/{task_id}/agent-executions/{worker_task_id}")
async def get_workforce_agent_execution(
    workforce_id: int,
    task_id: int,
    worker_task_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = ensure_workforce_access(
        db,
        user,
        _load_workforce(db, workforce_id),
        action="view",
    )
    run_query = db.query(WorkforceRun).filter(
        WorkforceRun.workforce_id == int(workforce.id),
        WorkforceRun.task_id == task_id,
    )
    if not user.is_admin:
        run_query = run_query.filter(WorkforceRun.user_id == int(user.id))
    run = run_query.options(selectinload(WorkforceRun.task)).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Workforce run not found")

    trace_events, metadata = _serialize_agent_execution_traces(
        db,
        task_id=task_id,
        worker_task_id=worker_task_id,
    )

    status = _derive_agent_execution_status(trace_events) or "running"
    if (
        status == "running"
        and run.task is not None
        and run.task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}
    ):
        status = "interrupted"
    summary_events = (
        db.query(TraceEvent)
        .filter(
            TraceEvent.task_id == task_id,
            TraceEvent.build_id.is_(None),
            TraceEvent.data["worker_task_id"].as_string() == worker_task_id,
        )
        .order_by(TraceEvent.id)
        .all()
    )
    for event in summary_events:
        data: dict[str, Any] = event.data if isinstance(event.data, dict) else {}
        summary_type = data.get("event_type")
        _merge_agent_execution_metadata(metadata, data)
        if summary_type == "workforce_delegation_end":
            status = "completed"
        elif summary_type == "workforce_delegation_error":
            status = "failed"

    return {
        **metadata,
        "task_id": task_id,
        "status": status,
        "trace_events": trace_events,
    }


@router.patch("/{workforce_id}")
async def update_workforce(
    workforce_id: int,
    request: WorkforceUpdateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = ensure_workforce_access(
        db,
        user,
        _load_workforce(db, workforce_id),
        action="edit",
    )
    workforce_row = cast(Any, workforce)

    if _field_supplied(request, "name"):
        workforce_row.name = _ensure_unique_workforce_name(
            db,
            workforce,
            scope_type=str(workforce.scope_type),
            scope_id=str(workforce.scope_id),
            name=cast(str, request.name),
        )
    if _field_supplied(request, "description"):
        workforce_row.description = normalize_text(request.description, "description")
    if _field_supplied(request, "canvas_layout"):
        workforce_row.canvas_layout = request.canvas_layout
    if _field_supplied(request, "manager_agent_id"):
        if request.manager_agent_id is None:
            raise HTTPException(status_code=400, detail="manager_agent_id is required")
        if int(request.manager_agent_id) != int(workforce.manager_agent_id):
            manager_agent = ensure_agent_access(
                db.query(Agent).filter(Agent.id == request.manager_agent_id).first(),
                user,
                db,
                require_published=True,
            )
            if any(
                int(worker.agent_id) == int(manager_agent.id)
                for worker in workforce.workers
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Manager agent cannot also be a worker",
                )
            workforce_row.manager_agent_id = int(manager_agent.id)

    try:
        _validate_if_active(db, user, workforce)
        db.commit()
    except Exception:
        db.rollback()
        raise

    return _serialize_workforce_detail(
        _reload_workforce(db, workforce), user, get_agent_team_scope(db, int(user.id))
    )


@router.delete("/{workforce_id}")
async def archive_workforce(
    workforce_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = ensure_workforce_access(
        db,
        user,
        _load_workforce(db, workforce_id),
        action="edit",
    )
    workforce_id_value = int(workforce.id)
    # Serialize against create_workforce_run, which takes the same lifecycle
    # fence: without this row lock, archive's cancellation sweep can run
    # while a concurrent create's uncommitted run is invisible to it (the
    # session is autoflush=False, so even the status flip below stays in
    # memory), letting that run permanently evade cancellation. The fence's
    # no-op UPDATE locks on SQLite too, where SELECT FOR UPDATE is ignored.
    if acquire_workforce_lifecycle_fence(db, workforce_id_value) is None:
        raise HTTPException(status_code=404, detail="Workforce not found")
    cast(Any, workforce).status = "archived"
    # Archive must also stop what is already running: flipping the status
    # alone leaves in-flight runs executing (turn resolution never re-checks
    # live workforce state) and external sessions open. The status flip and
    # every run cancellation commit atomically; PAUSE dispatch for still
    # RUNNING tasks happens after the commit (best-effort, own sessions).
    pause_targets = cancel_active_workforce_runs(db, workforce_id_value)
    db.commit()
    await pause_workforce_tasks_after_archive(
        pause_targets,
        workforce_id=workforce_id_value,
        actor_user_id=int(user.id),
    )
    return {"id": workforce.id, "status": workforce.status}


@router.post("/{workforce_id}/discard", status_code=204)
def discard_workforce(
    workforce_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Response:
    try:
        discard_draft_workforce(db, user, _load_workforce(db, workforce_id))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to discard workforce %s", workforce_id)
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail={
                "code": "workforce_discard_failed",
                "message": "Failed to discard workforce",
            },
        ) from None
    return Response(status_code=204)


@router.post("/{workforce_id}/publish")
async def publish_workforce(
    workforce_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = ensure_workforce_access(
        db,
        user,
        _load_workforce(db, workforce_id),
        action="edit",
    )
    _ensure_publish_state_mutable(workforce)
    cast(Any, workforce).status = "active"

    try:
        _validate_if_active(db, user, workforce)
        db.commit()
    except Exception:
        db.rollback()
        raise

    return _serialize_workforce_detail(
        _reload_workforce(db, workforce), user, get_agent_team_scope(db, int(user.id))
    )


@router.post("/{workforce_id}/unpublish")
async def unpublish_workforce(
    workforce_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = ensure_workforce_access(
        db,
        user,
        _load_workforce(db, workforce_id),
        action="edit",
    )
    _ensure_publish_state_mutable(workforce)
    cast(Any, workforce).status = "draft"
    db.commit()
    return _serialize_workforce_detail(
        _reload_workforce(db, workforce), user, get_agent_team_scope(db, int(user.id))
    )


@router.post("/{workforce_id}/agents")
async def add_workforce_agent(
    workforce_id: int,
    request: WorkforceWorkerInput,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = ensure_workforce_access(
        db,
        user,
        _load_workforce(db, workforce_id),
        action="edit",
    )
    try:
        worker = create_workforce_worker(
            db,
            workforce,
            user,
            source_type=request.source_type,
            assignment_instructions=request.assignment_instructions,
            alias=request.alias,
            agent_id=request.agent_id,
            enabled=request.enabled,
            sort_order=request.sort_order,
            canvas_position=request.canvas_position,
        )
        _validate_if_active(db, user, workforce)
        db.commit()
    except Exception:
        db.rollback()
        raise

    db.refresh(worker)
    return _serialize_worker(worker, user, get_agent_team_scope(db, int(user.id)))


@router.patch("/{workforce_id}/agents/{member_id}")
async def update_workforce_agent(
    workforce_id: int,
    member_id: int,
    request: WorkforceWorkerUpdateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = ensure_workforce_access(
        db,
        user,
        _load_workforce(db, workforce_id),
        action="edit",
    )
    worker = _load_worker(db, workforce, member_id)
    worker_row = cast(Any, worker)

    if _field_supplied(request, "alias"):
        worker_row.alias = normalize_text(request.alias, "alias")
    if _field_supplied(request, "assignment_instructions"):
        worker_row.assignment_instructions = normalize_text(
            request.assignment_instructions,
            "assignment_instructions",
            required=True,
        )
    if _field_supplied(request, "enabled"):
        if request.enabled is None:
            raise HTTPException(status_code=400, detail="enabled is required")
        worker_row.enabled = bool(request.enabled)
    if _field_supplied(request, "sort_order"):
        if request.sort_order is None:
            raise HTTPException(status_code=400, detail="sort_order is required")
        worker_row.sort_order = request.sort_order
    if _field_supplied(request, "canvas_position"):
        worker_row.canvas_position = request.canvas_position

    try:
        _validate_if_active(db, user, workforce)
        db.commit()
    except Exception:
        db.rollback()
        raise

    db.refresh(worker)
    return _serialize_worker(worker, user, get_agent_team_scope(db, int(user.id)))


@router.delete("/{workforce_id}/agents/{member_id}")
async def remove_workforce_agent(
    workforce_id: int,
    member_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, str]:
    workforce = ensure_workforce_access(
        db,
        user,
        _load_workforce(db, workforce_id),
        action="edit",
    )
    worker = _load_worker(db, workforce, member_id)

    try:
        db.delete(worker)
        db.flush()
        db.expire(workforce, ["workers"])
        _validate_if_active(db, user, workforce)
        db.commit()
    except Exception:
        db.rollback()
        raise

    return {"status": "deleted"}


@router.post("/{workforce_id}/runs")
async def create_workforce_run(
    workforce_id: int,
    request: WorkforceRunRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    result = await start_workforce_run(
        db,
        user,
        _load_workforce(db, workforce_id),
        message=request.message,
        selected_file_ids=request.files,
        execution_mode=request.execution_mode,
        is_preview=request.is_preview,
        is_visible=request.is_visible,
    )
    return {
        "workforce_run_id": result.workforce_run.id,
        "task_id": result.task.id,
        "status": result.workforce_run.status,
        "redirect_url": f"/task/{result.task.id}",
    }


_RUN_MESSAGE_PREVIEW_LIMIT = 200


def _serialize_run_list_item(run: WorkforceRun) -> dict[str, Any]:
    task = run.task
    message = cast(str | None, task.description) if task is not None else None
    if message and len(message) > _RUN_MESSAGE_PREVIEW_LIMIT:
        message = message[:_RUN_MESSAGE_PREVIEW_LIMIT] + "..."
    return {
        "id": run.id,
        "task_id": run.task_id,
        "status": run.status,
        "is_preview": bool(run.is_preview),
        "source": task.source if task is not None else None,
        "task_title": task.title if task is not None else None,
        "message": message,
        "created_at": _serialize_datetime(run.created_at),
        "completed_at": _serialize_datetime(run.completed_at),
    }


@router.get("/{workforce_id}/runs")
async def list_workforce_runs(
    workforce_id: int,
    page: int = 1,
    size: int = 20,
    include_preview: bool = False,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = ensure_workforce_access(
        db,
        user,
        _load_workforce(db, workforce_id),
        action="view",
    )
    if page < 1 or size < 1 or size > 100:
        raise HTTPException(status_code=400, detail="Invalid pagination parameters")

    query = db.query(WorkforceRun).filter(
        WorkforceRun.workforce_id == int(workforce.id)
    )
    if not include_preview:
        query = query.filter(WorkforceRun.is_preview.is_(False))

    total = query.count()
    runs = (
        query.options(selectinload(WorkforceRun.task))
        .order_by(WorkforceRun.created_at.desc(), WorkforceRun.id.desc())
        .offset((page - 1) * size)
        .limit(size)
        .all()
    )
    return {
        "items": [_serialize_run_list_item(run) for run in runs],
        "total": total,
        "page": page,
        "size": size,
        "pages": (total + size - 1) // size,
    }


@router.get("/{workforce_id}/runs/{run_id}")
async def get_workforce_run(
    workforce_id: int,
    run_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = ensure_workforce_access(
        db,
        user,
        _load_workforce(db, workforce_id),
        action="view",
    )
    run = (
        db.query(WorkforceRun)
        .options(selectinload(WorkforceRun.task))
        .filter(
            WorkforceRun.id == run_id,
            WorkforceRun.workforce_id == int(workforce.id),
        )
        .first()
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Workforce run not found")
    return {
        **_serialize_run_list_item(run),
        "snapshot": run.snapshot,
    }


@router.get("/{workforce_id}/canvas")
async def get_workforce_canvas(
    workforce_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    workforce = ensure_workforce_access(
        db,
        user,
        _load_workforce(db, workforce_id),
        action="view",
    )
    manager_node_id = f"manager-{workforce.manager_agent.id}"
    nodes: list[dict[str, Any]] = [
        {"id": "human", "type": "human", "label": "Human"},
        {
            "id": manager_node_id,
            "type": "manager",
            "agent_id": workforce.manager_agent.id,
            "label": workforce.manager_agent.name,
        },
    ]
    edges: list[dict[str, Any]] = [
        {"id": "human-manager", "source": "human", "target": manager_node_id}
    ]

    for worker in _sorted_workers(workforce):
        worker_node_id = f"worker-{worker.id}"
        nodes.append(
            {
                "id": worker_node_id,
                "type": "worker",
                "agent_id": worker.agent_id,
                "label": worker.alias or worker.agent.name,
                "position": worker.canvas_position,
                "enabled": worker.enabled,
            }
        )
        edges.append(
            {
                "id": f"manager-worker-{worker.id}",
                "source": manager_node_id,
                "target": worker_node_id,
            }
        )

    return {"nodes": nodes, "edges": edges, "layout": workforce.canvas_layout or {}}
