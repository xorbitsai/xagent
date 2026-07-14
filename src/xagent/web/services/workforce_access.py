from collections.abc import Iterable
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Query, Session

from xagent.web.models.agent import (
    Agent,
    AgentStatus,
    is_workforce_generated_manager_agent,
)
from xagent.web.models.user import User

from ..models.workforce import Workforce
from .agent_team_scope import get_agent_team_scope, owns_agent


class WorkforcePolicy:
    def resolve_create_scope(self, db: Session, user: User) -> tuple[str, str]:
        del db
        return ("user", str(user.id))

    def can_create_workforce(
        self,
        db: Session,
        user: User,
        scope_type: str,
        scope_id: str,
    ) -> bool:
        del db, scope_type, scope_id
        return bool(user.id)

    def can_view_workforce(self, db: Session, user: User, workforce: Workforce) -> bool:
        del db
        return bool(user.is_admin or int(workforce.owner_user_id) == int(user.id))

    def filter_visible_workforces(
        self,
        db: Session,
        user: User,
        query: Query[Workforce],
    ) -> Query[Workforce]:
        del db
        if user.is_admin:
            return query
        return query.filter(Workforce.owner_user_id == int(user.id))

    def can_run_workforce(self, db: Session, user: User, workforce: Workforce) -> bool:
        return self.can_view_workforce(db, user, workforce)

    def can_edit_workforce(self, db: Session, user: User, workforce: Workforce) -> bool:
        del db
        return bool(user.is_admin or int(workforce.owner_user_id) == int(user.id))

    def get_visible_agent_ids(
        self,
        db: Session,
        user: User,
        purpose: str,
    ) -> set[int] | None:
        del db, user, purpose
        return None

    def before_workforce_run(
        self, db: Session, user: User, workforce: Workforce
    ) -> None:
        del db, user, workforce

    def is_agent_in_workforce_run_scope(
        self,
        db: Session,
        user: User,
        workforce: Workforce,
        agent: Agent,
    ) -> bool:
        scope = get_agent_team_scope(db, int(user.id))
        return bool(
            int(workforce.owner_user_id) == int(user.id)
            and owns_agent(agent, int(user.id), scope)
        )

    def after_workforce_run_created(
        self,
        db: Session,
        user: User,
        workforce: Workforce,
        run: Any,
        task: Any,
    ) -> None:
        del db, user, workforce, run, task

    def record_workforce_event(
        self,
        db: Session,
        user: User,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        del db, user, event_type, payload


_workforce_policy: WorkforcePolicy = WorkforcePolicy()


def set_workforce_policy(policy: WorkforcePolicy) -> None:
    global _workforce_policy
    _workforce_policy = policy


def get_workforce_policy() -> WorkforcePolicy:
    return _workforce_policy


def resolve_create_scope(db: Session, user: User) -> tuple[str, str]:
    return get_workforce_policy().resolve_create_scope(db, user)


def can_create_workforce(
    db: Session,
    user: User,
    scope_type: str,
    scope_id: str,
) -> bool:
    return get_workforce_policy().can_create_workforce(db, user, scope_type, scope_id)


def _normalize_visible_agent_ids(values: Iterable[int] | None) -> set[int] | None:
    if values is None:
        return None
    normalized: set[int] = set()
    for value in values:
        if isinstance(value, int):
            normalized.add(value)
    return normalized


def get_visible_agent_ids(db: Session, user: User, purpose: str) -> set[int] | None:
    visible_agent_ids = get_workforce_policy().get_visible_agent_ids(db, user, purpose)
    return _normalize_visible_agent_ids(visible_agent_ids)


def can_view_workforce(db: Session, user: User, workforce: Workforce) -> bool:
    return get_workforce_policy().can_view_workforce(db, user, workforce)


def filter_visible_workforces(
    db: Session,
    user: User,
    query: Query[Workforce],
) -> Query[Workforce]:
    return get_workforce_policy().filter_visible_workforces(db, user, query)


def can_run_workforce(db: Session, user: User, workforce: Workforce) -> bool:
    return get_workforce_policy().can_run_workforce(db, user, workforce)


def can_edit_workforce(db: Session, user: User, workforce: Workforce) -> bool:
    return get_workforce_policy().can_edit_workforce(db, user, workforce)


def ensure_workforce_access(
    db: Session,
    user: User,
    workforce: Workforce | None,
    action: str = "view",
) -> Workforce:
    if workforce is None:
        raise HTTPException(status_code=404, detail="Workforce not found")

    allowed = False
    if action == "view":
        allowed = can_view_workforce(db, user, workforce)
    elif action == "run":
        allowed = can_run_workforce(db, user, workforce)
    elif action == "edit":
        allowed = can_edit_workforce(db, user, workforce)
    else:
        raise ValueError(f"Unsupported workforce access action: {action}")

    if not allowed:
        raise HTTPException(status_code=403, detail="Access denied")
    if action == "edit" and workforce.status == "archived":
        raise HTTPException(
            status_code=409,
            detail="Archived workforce cannot be edited",
        )
    return workforce


def ensure_agent_access(
    agent: Agent | None,
    user: User,
    db: Session,
    purpose: str = "workforce_select",
    require_published: bool = False,
) -> Agent:
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    if is_workforce_generated_manager_agent(agent):
        raise HTTPException(status_code=404, detail="Agent not found")
    scope = get_agent_team_scope(db, int(user.id))
    if user.is_admin or owns_agent(agent, int(user.id), scope):
        if require_published and agent.status != AgentStatus.PUBLISHED:
            raise HTTPException(
                status_code=400,
                detail="Workforce agents must be published",
            )
        return agent
    visible_agent_ids = get_visible_agent_ids(db, user, purpose)
    if visible_agent_ids is not None and int(agent.id) in visible_agent_ids:
        if require_published and agent.status != AgentStatus.PUBLISHED:
            raise HTTPException(
                status_code=400,
                detail="Workforce agents must be published",
            )
        return agent
    raise HTTPException(status_code=403, detail="Access denied to agent")


def list_accessible_published_agents(
    db: Session,
    user: User,
    *,
    purpose: str = "workforce_select",
    exclude_agent_ids: Iterable[int] | None = None,
) -> list[Agent]:
    from .agent_access import list_accessible_published_agents as list_agents

    return list_agents(
        db,
        user,
        purpose=purpose,
        exclude_agent_ids=exclude_agent_ids,
    )


def ensure_workforce_agent_run_access(
    agent: Agent | None,
    user: User,
    db: Session,
    workforce: Workforce,
) -> Agent:
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent.status != AgentStatus.PUBLISHED:
        raise HTTPException(
            status_code=400, detail="Workforce agents must be published"
        )
    if get_workforce_policy().is_agent_in_workforce_run_scope(
        db, user, workforce, agent
    ):
        return agent
    raise HTTPException(status_code=403, detail="Access denied to agent")
