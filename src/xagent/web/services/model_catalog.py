"""Static ability catalog for known LLM models.

Loads ``src/xagent/core/model/abilities_catalog.yaml`` once at first use and
provides a single ``lookup(provider, model_name)`` entry point that returns the
recommended ability list for a given (provider, model_name) pair.

Resolution order
----------------
1. Exact provider AND specific model pattern (``model_pattern != "*"``)
2. Wildcard provider (``*``) AND specific model pattern
3. Exact provider AND ``model_pattern == "*"`` (provider fallback)
4. Otherwise: ``source="none"`` and an empty ability list

Putting tier 2 above tier 3 is important: it lets ``*/deepseek-chat*`` win
against ``openai/*`` when DeepSeek is added through an OpenAI-compatible
endpoint. A pure provider fallback should only fire when no model-name rule
matches at all.

The catalog is loaded lazily and cached. Call :func:`reload_catalog` in tests
to force a re-read after editing the YAML.
"""

from __future__ import annotations

import fnmatch
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)

# Path is resolved at import time so it is independent of CWD.
_CATALOG_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "core"
    / "model"
    / "abilities_catalog.yaml"
)

# Allowed abilities for category=llm. Mirrors the frontend list and the
# server-side validator in ``schemas/model.py``.
_ALLOWED_LLM_ABILITIES = {"chat", "vision", "tool_calling", "thinking_mode"}


@dataclass(frozen=True)
class CatalogRule:
    """A single rule loaded from the catalog YAML."""

    provider_pattern: str  # e.g. "openai" or "*"
    model_pattern: str  # e.g. "gpt-4o*"
    abilities: tuple[str, ...]


@dataclass(frozen=True)
class AbilitySuggestion:
    """Result of a catalog lookup."""

    abilities: List[str]
    matched_pattern: Optional[str]
    source: str  # "exact" | "wildcard_provider" | "none"


@dataclass(frozen=True)
class _RulesIndex:
    """Rules pre-partitioned into the three matching tiers used by
    :func:`lookup`, so each call iterates only the relevant slice instead
    of scanning the entire catalog three times."""

    tier1_exact_specific: tuple[CatalogRule, ...]
    tier2_wildcard_specific: tuple[CatalogRule, ...]
    tier3_provider_fallback: tuple[CatalogRule, ...]

    @property
    def all_rules(self) -> List[CatalogRule]:
        return [
            *self.tier1_exact_specific,
            *self.tier2_wildcard_specific,
            *self.tier3_provider_fallback,
        ]


_lock = threading.Lock()
_index_cache: Optional[_RulesIndex] = None


def _parse_pattern(pattern: str) -> tuple[str, str]:
    """Split a ``provider/model`` pattern. Defaults to ``*/<pattern>`` if no
    slash is present."""
    if "/" not in pattern:
        return "*", pattern
    provider, model = pattern.split("/", 1)
    return provider.strip() or "*", model.strip() or "*"


def _load_rules() -> List[CatalogRule]:
    """Load and parse the YAML catalog. Returns an empty list (and logs) if
    the file is missing or malformed, so the API stays online."""
    if not _CATALOG_PATH.exists():
        logger.warning(
            "Ability catalog not found at %s; suggestions disabled.",
            _CATALOG_PATH,
        )
        return []
    try:
        with _CATALOG_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as exc:
        logger.error("Failed to read ability catalog %s: %s", _CATALOG_PATH, exc)
        return []

    raw_rules = data.get("rules") or []
    if not isinstance(raw_rules, list):
        logger.error("Ability catalog 'rules' must be a list, got %s", type(raw_rules))
        return []

    parsed: List[CatalogRule] = []
    for idx, item in enumerate(raw_rules):
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict catalog rule at index %d", idx)
            continue
        pattern = str(item.get("pattern", "")).strip()
        abilities_raw = item.get("abilities") or []
        if not pattern or not isinstance(abilities_raw, list):
            logger.warning("Skipping malformed catalog rule at index %d", idx)
            continue

        # Filter to allowed abilities; drop unknowns with a warning rather than
        # silently letting bad data leak to the UI.
        cleaned: List[str] = []
        for a in abilities_raw:
            if not isinstance(a, str):
                continue
            a_norm = a.strip().lower()
            if a_norm in _ALLOWED_LLM_ABILITIES:
                cleaned.append(a_norm)
            else:
                logger.warning(
                    "Catalog rule '%s' references unknown ability '%s'; ignoring.",
                    pattern,
                    a,
                )
        if not cleaned:
            continue

        provider_pat, model_pat = _parse_pattern(pattern)
        parsed.append(
            CatalogRule(
                provider_pattern=provider_pat.lower(),
                model_pattern=model_pat.lower(),
                abilities=tuple(cleaned),
            )
        )

    logger.info("Loaded %d ability catalog rules from %s", len(parsed), _CATALOG_PATH)
    return parsed


def _partition_rules(rules: List[CatalogRule]) -> _RulesIndex:
    """Bucket the flat rule list into the three resolution tiers so that
    :func:`lookup` doesn't have to scan rules belonging to other tiers."""
    t1: List[CatalogRule] = []
    t2: List[CatalogRule] = []
    t3: List[CatalogRule] = []
    for rule in rules:
        provider_wild = rule.provider_pattern == "*"
        model_wild = rule.model_pattern == "*"
        if not provider_wild and not model_wild:
            t1.append(rule)
        elif provider_wild and not model_wild:
            t2.append(rule)
        elif not provider_wild and model_wild:
            t3.append(rule)
        # "*/*" is meaningless — silently drop.
    return _RulesIndex(
        tier1_exact_specific=tuple(t1),
        tier2_wildcard_specific=tuple(t2),
        tier3_provider_fallback=tuple(t3),
    )


def _get_index() -> _RulesIndex:
    global _index_cache
    if _index_cache is None:
        with _lock:
            if _index_cache is None:
                _index_cache = _partition_rules(_load_rules())
    return _index_cache


def _get_rules() -> List[CatalogRule]:
    """Backwards-compatible flat view of all rules across tiers."""
    return _get_index().all_rules


def reload_catalog() -> int:
    """Force a re-read of the catalog YAML. Returns the number of rules
    loaded. Intended for tests / admin reload endpoints."""
    global _index_cache
    with _lock:
        _index_cache = _partition_rules(_load_rules())
    return len(_index_cache.all_rules)


def lookup(provider: str, model_name: str) -> AbilitySuggestion:
    """Return the recommended abilities for a given (provider, model) pair.

    Both inputs are matched case-insensitively. Whitespace is stripped.
    """
    provider_norm = (provider or "").strip().lower()
    model_norm = (model_name or "").strip().lower()
    if not model_norm:
        return AbilitySuggestion(abilities=[], matched_pattern=None, source="none")

    index = _get_index()

    def _matches(rule: CatalogRule) -> bool:
        return fnmatch.fnmatchcase(model_norm, rule.model_pattern)

    # Tier 1: exact provider + specific model
    for rule in index.tier1_exact_specific:
        if rule.provider_pattern == provider_norm and _matches(rule):
            return AbilitySuggestion(
                abilities=list(rule.abilities),
                matched_pattern=f"{rule.provider_pattern}/{rule.model_pattern}",
                source="exact",
            )

    # Tier 2: wildcard provider + specific model
    for rule in index.tier2_wildcard_specific:
        if _matches(rule):
            return AbilitySuggestion(
                abilities=list(rule.abilities),
                matched_pattern=f"*/{rule.model_pattern}",
                source="wildcard_provider",
            )

    # Tier 3: provider-only fallback
    for rule in index.tier3_provider_fallback:
        if rule.provider_pattern == provider_norm:
            return AbilitySuggestion(
                abilities=list(rule.abilities),
                matched_pattern=f"{rule.provider_pattern}/*",
                source="exact",
            )

    return AbilitySuggestion(abilities=[], matched_pattern=None, source="none")
