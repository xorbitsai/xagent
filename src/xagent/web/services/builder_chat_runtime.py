"""Detached runtime preparation for the Builder Chat WebSocket flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ...core.model.chat.basic.base import BaseLLM
from ..models.database import get_session_local
from ..models.uploaded_file import UploadedFile
from .db_runtime import run_db_io_cancellation_safe
from .llm_utils import UserAwareModelStorage


@dataclass(frozen=True)
class BuilderChatRuntimeInputs:
    """Database-derived inputs safe to retain across asynchronous execution."""

    authorized_file_ids: tuple[str, ...]
    llm: BaseLLM | None
    compact_llm: BaseLLM | None


def _load_builder_chat_runtime_inputs_sync(
    *,
    user_id: int,
    requested_file_ids: Iterable[str],
    model_name: Any,
    compact_model_name: Any,
) -> BuilderChatRuntimeInputs:
    """Resolve one Builder Chat turn inside a worker-owned short Session."""

    ordered_file_ids = tuple(str(file_id) for file_id in requested_file_ids)
    unique_file_ids = tuple(dict.fromkeys(ordered_file_ids))

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        authorized_ids: set[str] = set()
        if unique_file_ids:
            rows = (
                db.query(UploadedFile)
                .filter(
                    UploadedFile.file_id.in_(unique_file_ids),
                    UploadedFile.user_id == user_id,
                )
                .all()
            )
            authorized_ids = {str(row.file_id) for row in rows}

        resolver = UserAwareModelStorage(db)
        llm = (
            resolver.get_llm_by_name_with_access(model_name, user_id=user_id)
            if model_name
            else None
        )
        compact_llm = (
            resolver.get_llm_by_name_with_access(
                compact_model_name,
                user_id=user_id,
            )
            if compact_model_name
            else None
        )

        if not llm or compact_llm is None:
            default_llm, _fast_llm, _vision_llm, default_compact_llm = (
                resolver.get_configured_defaults(
                    user_id=user_id,
                    config_types=tuple(
                        kind
                        for kind, missing in (
                            ("general", llm is None),
                            ("compact", compact_llm is None),
                        )
                        if missing
                    ),
                    fallback_llm=llm,
                )
            )
            if not llm:
                llm = default_llm
            if compact_llm is None:
                compact_llm = default_compact_llm

        return BuilderChatRuntimeInputs(
            authorized_file_ids=tuple(
                file_id for file_id in ordered_file_ids if file_id in authorized_ids
            ),
            llm=llm,
            compact_llm=compact_llm,
        )


async def load_builder_chat_runtime_inputs(
    *,
    user_id: int,
    requested_file_ids: Iterable[str],
    model_name: Any,
    compact_model_name: Any,
) -> BuilderChatRuntimeInputs:
    """Load Builder Chat inputs without blocking the asyncio event loop."""

    return await run_db_io_cancellation_safe(
        lambda: _load_builder_chat_runtime_inputs_sync(
            user_id=user_id,
            requested_file_ids=requested_file_ids,
            model_name=model_name,
            compact_model_name=compact_model_name,
        )
    )
