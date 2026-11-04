"""Execution-budget settings, separate from monthly usage and billing."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user
from ..models.database import get_db, release_db_connection_if_clean
from ..models.system_setting import SystemSetting
from ..models.user import User
from ..services.execution_budget import (
    BUDGET_SETTINGS_KEY,
    BudgetPolicyRequest,
    ExecutionBudgetDefaults,
    load_budget_defaults,
    resolve_execution_budget_policy,
)

router = APIRouter(prefix="/api/execution-budget", tags=["execution-budget"])


@router.get("/me")
async def get_my_execution_budget(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict[str, Any]:
    user_id = int(user.id)
    preferences: dict[str, Any] = dict(user.preferences or {})
    if not release_db_connection_if_clean(db):
        raise HTTPException(status_code=503, detail="Could not load execution budget")
    policy = await resolve_execution_budget_policy(BudgetPolicyRequest(user_id=user_id))
    return {
        "effective": policy.model_dump(),
        "preferences": {
            "execution_budget_tokens": preferences.get("execution_budget_tokens"),
            "execution_budget_soft_percent": preferences.get(
                "execution_budget_soft_percent"
            ),
        },
    }


@router.get("/defaults", response_model=ExecutionBudgetDefaults)
def get_execution_budget_defaults(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> ExecutionBudgetDefaults:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return load_budget_defaults(db)


@router.put("/defaults", response_model=ExecutionBudgetDefaults)
def update_execution_budget_defaults(
    settings: ExecutionBudgetDefaults,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ExecutionBudgetDefaults:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    row = (
        db.query(SystemSetting).filter(SystemSetting.key == BUDGET_SETTINGS_KEY).first()
    )
    if row is None:
        row = SystemSetting(key=BUDGET_SETTINGS_KEY, value=settings.model_dump_json())
        db.add(row)
    else:
        setattr(row, "value", settings.model_dump_json())
    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(
            status_code=409, detail="Settings changed concurrently; retry"
        ) from error
    return settings
