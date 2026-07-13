from typing import Any, cast

from sqlalchemy import func
from sqlalchemy.orm import Session

from xagent.web.models.agent import Agent

from ..models.workforce import Workforce
from .agent_team_scope import get_agent_team_scope, owned_agent_clause
from .workforce_snapshot import normalize_text


def workforce_name_exists(
    db: Session,
    *,
    scope_type: str,
    scope_id: str,
    name: str,
    exclude_workforce_id: int | None = None,
) -> bool:
    normalized_name = normalize_text(name, "name", required=True)
    query = db.query(Workforce.id).filter(
        Workforce.scope_type == scope_type,
        Workforce.scope_id == scope_id,
        func.lower(cast(Any, Workforce.name)) == normalized_name.lower(),
    )
    if exclude_workforce_id is not None:
        query = query.filter(Workforce.id != exclude_workforce_id)
    return query.first() is not None


def resolve_unique_workforce_name(
    db: Session,
    *,
    scope_type: str,
    scope_id: str,
    name: str,
) -> str:
    normalized_name = normalize_text(name, "name", required=True)
    if not workforce_name_exists(
        db,
        scope_type=scope_type,
        scope_id=scope_id,
        name=normalized_name,
    ):
        return normalized_name

    base_name = normalized_name
    suffix = 2
    while True:
        suffix_text = f" {suffix}"
        candidate_base = base_name[: max(1, 200 - len(suffix_text))].rstrip()
        candidate = f"{candidate_base}{suffix_text}"
        if not workforce_name_exists(
            db,
            scope_type=scope_type,
            scope_id=scope_id,
            name=candidate,
        ):
            return candidate
        suffix += 1


def agent_name_exists(
    db: Session,
    *,
    user_id: int,
    name: str,
    exclude_agent_id: int | None = None,
) -> bool:
    normalized_name = normalize_text(name, "name", required=True)
    query = db.query(Agent.id).filter(
        owned_agent_clause(user_id, get_agent_team_scope(db, user_id)),
        func.lower(cast(Any, Agent.name)) == normalized_name.lower(),
    )
    if exclude_agent_id is not None:
        query = query.filter(Agent.id != exclude_agent_id)
    return query.first() is not None


def resolve_unique_agent_name(
    db: Session,
    *,
    user_id: int,
    name: str,
) -> str:
    normalized_name = normalize_text(name, "name", required=True)
    if not agent_name_exists(db, user_id=user_id, name=normalized_name):
        return normalized_name

    base_name = normalized_name
    suffix = 2
    while True:
        suffix_text = f" {suffix}"
        candidate_base = base_name[: max(1, 200 - len(suffix_text))].rstrip()
        candidate = f"{candidate_base}{suffix_text}"
        if not agent_name_exists(db, user_id=user_id, name=candidate):
            return candidate
        suffix += 1
