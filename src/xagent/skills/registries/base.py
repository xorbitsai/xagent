"""
Abstract base class for skill registry providers.

Contributors: subclass ``SkillRegistry``, implement the abstract
methods, and export an instance named ``<id>_registry`` from your
module. Then add your provider to the ``_REGISTRY_PROVIDERS`` tuple
in ``__init__.py``.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import requests
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Shared across all providers.
MAX_REGISTRY_BODY = 2 * 1024 * 1024  # 2 MiB
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # 50 MiB

_HTTP = requests.Session()
_HTTP.headers.update(
    {
        "User-Agent": (
            "xagent-saas-skill-hub/0.1 (+https://github.com/xorbitsai/xagent)"
        ),
        "Accept": "application/json",
    }
)


class SkillRegistry(ABC):
    """Contracts every registry provider must fulfil.

    Each method maps to one upstream operation.
    Async routes call these via ``asyncio.to_thread``.

    Quick reference for contributors
    --------------------------------
    * ``list_skills`` / ``search_skills`` / ``get_skill``
      → return the **raw upstream dict** (route layer normalises).
    * ``download_skill`` → return ``(http_status, raw_bytes)``.
    * ``extract_scan_status`` → return ``"clean"|"suspicious"|"malicious"|None``.
    * ``search_results_field`` → override if your search response wraps
      items in a key other than ``"results"`` (e.g. ``"items"`` or
      ``"hits"``).
    """

    # ── identity ────────────────────────────────────────────────

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique machine ID, e.g. ``"clawhub"``."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable label, e.g. ``"ClawHub"``."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description for tooltips."""

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Base URL for the upstream API."""

    # ── core operations ─────────────────────────────────────────

    @abstractmethod
    def list_skills(
        self, sort: str, limit: int, cursor: Optional[str]
    ) -> Dict[str, Any]:
        """Browse / list skills."""

    @abstractmethod
    def search_skills(self, query: str, limit: int) -> Dict[str, Any]:
        """Full-text search."""

    @abstractmethod
    def get_skill(self, slug: str) -> Dict[str, Any]:
        """Get a single skill's detail."""

    @abstractmethod
    def download_skill(self, slug: str, version: Optional[str]) -> Tuple[int, bytes]:
        """Download the skill ZIP.  Returns ``(http_status, raw_bytes)``."""

    @abstractmethod
    def extract_scan_status(self, raw_item: Dict[str, Any]) -> Optional[str]:
        """Pull security scan status.

        Must return one of ``"clean" | "suspicious" | "malicious" | None``.
        """

    @property
    def search_results_field(self) -> str:
        """The key that holds result items in a search response.

        ClawHub uses ``"results"``. Override if your registry uses
        ``"items"``, ``"hits"``, or another key.
        """
        return "results"

    # ── convenience HTTP helper ─────────────────────────────────

    def _http_get(self, path: str, params: Optional[dict] = None) -> Any:
        """GET ``{base_url}{path}``, parse JSON, raise on errors."""
        url = f"{self.base_url}{path}"
        try:
            with _HTTP.get(url, params=params or {}, timeout=15, stream=True) as r:
                status = r.status_code
                chunks: list[bytes] = []
                total = 0
                for chunk in r.iter_content(chunk_size=65536):
                    total += len(chunk)
                    if total > MAX_REGISTRY_BODY:
                        raise HTTPException(
                            status_code=502,
                            detail=f"{self.display_name} response too large.",
                        )
                    chunks.append(chunk)
                raw = b"".join(chunks)
        except HTTPException:
            raise
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail=f"{self.display_name} unreachable: {exc}",
            ) from exc
        if status == 404:
            raise HTTPException(
                status_code=404,
                detail=f"Skill not found on {self.display_name}.",
            )
        if status == 429:
            raise HTTPException(
                status_code=503,
                detail=f"{self.display_name} rate-limited — try again soon.",
            )
        if status >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"{self.display_name} returned HTTP {status}.",
            )
        try:
            return json.loads(raw)
        except Exception:
            raise HTTPException(
                status_code=502,
                detail=f"Invalid JSON from {self.display_name}.",
            )
