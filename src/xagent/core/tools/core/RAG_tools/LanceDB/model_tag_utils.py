from __future__ import annotations

import re
from typing import Final

__all__ = ["to_model_tag", "embeddings_table_name"]

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
