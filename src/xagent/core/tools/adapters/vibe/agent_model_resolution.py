"""Model resolution helpers for delegated agent tools."""

from collections.abc import Mapping
from typing import Any

AGENT_MODEL_KEYS = ("general", "small_fast", "visual", "compact")


def _coerce_model_db_id(raw_model_id: Any) -> int | None:
    if raw_model_id in (None, "") or isinstance(raw_model_id, bool):
        return None
    try:
        return int(raw_model_id)
    except (TypeError, ValueError):
        return None


def resolve_agent_model_llms(
    db: Any,
    storage: Any,
    agent_models: Mapping[str, Any] | None,
    user_id: int,
) -> tuple[Any | None, Any | None, Any | None, Any | None]:
    """Resolve delegated-agent model slots with one database lookup."""
    if not agent_models or not isinstance(agent_models, Mapping):
        return None, None, None, None

    from .....web.models.model import Model as DBModel

    model_ids_by_key = {
        model_key: coerced_model_id
        for model_key in AGENT_MODEL_KEYS
        if (coerced_model_id := _coerce_model_db_id(agent_models.get(model_key)))
        is not None
    }
    model_ids = list(model_ids_by_key.values())
    if not model_ids:
        return None, None, None, None

    models_by_id: dict[Any, Any] = {}
    for model in db.query(DBModel).filter(DBModel.id.in_(model_ids)).all():
        models_by_id[model.id] = model

    resolved_llms = {}
    for model_key, model_id in model_ids_by_key.items():
        model = models_by_id.get(model_id)
        if model:
            resolved_llms[model_key] = storage.get_llm_by_name_with_access(
                str(model.model_id), user_id
            )

    return (
        resolved_llms.get("general"),
        resolved_llms.get("small_fast"),
        resolved_llms.get("visual"),
        resolved_llms.get("compact"),
    )
