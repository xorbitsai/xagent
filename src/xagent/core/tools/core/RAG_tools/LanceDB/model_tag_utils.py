from __future__ import annotations

import re
from typing import Final, Optional

__all__ = [
    "to_model_tag",
    "embeddings_table_name",
    "model_tag_candidates",
    "written_embeddings_table_name",
    "embeddings_table_candidates",
    "legacy_embeddings_table_name",
    "searchable_embeddings_table_names",
]

_INVALID_CHARS: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


def _normalize_slug(text: str) -> str:
    lowered = text.strip().lower()
    slug = _INVALID_CHARS.sub("_", lowered).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug


def to_model_tag(model_name: str) -> str:
    raw = model_name.strip()
    if "/" in raw:
        prefix, rest = raw.split("/", 1)
        vendor = prefix.strip().upper()
        slug = _normalize_slug(rest)
        return f"{vendor}_{slug}" if slug else vendor
    return _normalize_slug(raw)


def embeddings_table_name(model_name: str) -> str:
    return f"embeddings_{to_model_tag(model_name)}"


def model_tag_candidates(model_name: str) -> list[str]:
    """Tags for a model: the writer's twice-applied tag, then the once-applied one."""
    writer = to_model_tag(to_model_tag(model_name))
    once = to_model_tag(model_name)
    return list(dict.fromkeys([writer, once]))


def written_embeddings_table_name(model_name: str) -> str:
    """Table the current write path creates: ``to_model_tag`` applied twice."""
    return f"embeddings_{to_model_tag(to_model_tag(model_name))}"


def legacy_embeddings_table_name(model_tag: str) -> Optional[str]:
    """The pre-candidate name for a model (``embeddings_{model_name_tag}``).

    Both search and the migration resolve it here, so the two cannot drift.
    ``None`` when the model cannot be resolved to a name.
    """
    from ..utils.model_resolver import resolve_embedding_adapter

    try:
        config, _ = resolve_embedding_adapter(model_tag)
    except Exception:
        return None
    name = getattr(config, "model_name", None)
    return f"embeddings_{to_model_tag(name)}" if name else None


def embeddings_table_candidates(model_name: str) -> list[str]:
    """Candidate tables: the writer's spelling first, de-duplicated."""
    return [f"embeddings_{tag}" for tag in model_tag_candidates(model_name)]


def searchable_embeddings_table_names(model_tag: str) -> list[str]:
    """Every table search may open for a model, in the order it tries them."""
    names = embeddings_table_candidates(model_tag)
    legacy = legacy_embeddings_table_name(model_tag)
    return list(dict.fromkeys(names + ([legacy] if legacy else [])))
