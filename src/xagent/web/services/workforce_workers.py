from typing import Any, cast

from fastapi import HTTPException
from sqlalchemy.orm import Session

from xagent.web.models.agent import Agent
from xagent.web.models.user import User

from ..models.workforce import Workforce, WorkforceAgent
from .workforce_access import ensure_agent_access, ensure_workforce_access
from .workforce_snapshot import normalize_text


def ensure_supported_source_type(source_type: str) -> None:
    if source_type != "existing":
        raise HTTPException(
            status_code=400,
            detail="source_type must be existing; publish an agent before adding it to a workforce",
        )


def next_worker_sort_order(db: Session, workforce_id: int) -> int:
    max_sort_order = (
        db.query(WorkforceAgent.sort_order)
        .filter(WorkforceAgent.workforce_id == workforce_id)
        .order_by(WorkforceAgent.sort_order.desc(), WorkforceAgent.id.desc())
        .first()
    )
    return (
        int(max_sort_order[0]) + 1
        if max_sort_order and max_sort_order[0] is not None
        else 1
    )


def create_workforce_worker(
    db: Session,
    workforce: Workforce,
    user: User,
    *,
    source_type: str,
    assignment_instructions: str,
    alias: str | None = None,
    agent_id: int | None = None,
    enabled: bool = True,
    sort_order: int | None = None,
    canvas_position: dict[str, Any] | None = None,
    template_id: str | None = None,
) -> WorkforceAgent:
    workforce = ensure_workforce_access(db, user, workforce, action="edit")
    ensure_supported_source_type(source_type)

    normalized_assignment = normalize_text(
        assignment_instructions,
        "assignment_instructions",
        required=True,
    )
    if normalized_assignment is None:
        raise HTTPException(
            status_code=400, detail="assignment_instructions is required"
        )

    if agent_id is None:
        raise HTTPException(status_code=400, detail="agent_id is required")

    agent = ensure_agent_access(
        db.query(Agent).filter(Agent.id == agent_id).first(),
        user,
        db,
        require_published=True,
    )
    existing = (
        db.query(WorkforceAgent)
        .filter(
            WorkforceAgent.workforce_id == workforce.id,
            WorkforceAgent.agent_id == agent.id,
        )
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail="Agent already added to workforce")

    agent_id_value = cast(int, agent.id)
    workforce_manager_id = cast(int, workforce.manager_agent_id)
    if agent_id_value == workforce_manager_id:
        raise HTTPException(
            status_code=400, detail="Manager agent cannot also be a worker"
        )

    workforce_id = cast(int, workforce.id)
    worker = WorkforceAgent(
        workforce_id=workforce_id,
        agent_id=agent_id_value,
        alias=normalize_text(alias, "alias"),
        assignment_instructions=normalized_assignment,
        source_type=source_type,
        enabled=bool(enabled),
        sort_order=sort_order
        if sort_order is not None
        else next_worker_sort_order(db, workforce_id),
        canvas_position=canvas_position,
        template_id=template_id,
    )
    db.add(worker)
    db.flush()
    return worker
