"""
ClawHub (clawhub.ai) skill registry provider.

ClawHub is the OpenClaw public skill registry.
API docs: https://docs.openclaw.ai/clawhub/http-api
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import requests
from fastapi import HTTPException

from xagent.skills.registries.base import (
    _HTTP,
    MAX_DOWNLOAD_BYTES,
    SkillRegistry,
)

logger = logging.getLogger(__name__)

CLAWHUB_BASE_URL = "https://clawhub.ai/api/v1"


class ClawHubRegistry(SkillRegistry):
    """ClawHub (clawhub.ai) — the OpenClaw public skill registry.

    * Browse / search → ``GET /api/v1/skills``
    * Detail           → ``GET /api/v1/skills/{slug}``
    * Download         → ``GET /api/v1/download?slug=...``
    """

    # ── identity ────────────────────────────────────────────────

    @property
    def id(self) -> str:
        return "clawhub"

    @property
    def display_name(self) -> str:
        return "ClawHub"

    @property
    def description(self) -> str:
        return "ClawHub public skill registry"

    @property
    def base_url(self) -> str:
        return CLAWHUB_BASE_URL

    # ── core operations ─────────────────────────────────────────

    def list_skills(
        self, sort: str, limit: int, cursor: Optional[str]
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"sort": sort, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._http_get("/skills", params)  # type: ignore[no-any-return]

    def search_skills(self, query: str, limit: int) -> Dict[str, Any]:
        return self._http_get("/search", {"q": query, "limit": limit})  # type: ignore[no-any-return]

    def get_skill(self, slug: str) -> Dict[str, Any]:
        return self._http_get(f"/skills/{slug}")  # type: ignore[no-any-return]

    def download_skill(self, slug: str, version: Optional[str]) -> Tuple[int, bytes]:
        params: Dict[str, Any] = {"slug": slug}
        if version:
            params["version"] = version
        try:
            with _HTTP.get(
                f"{self.base_url}/download",
                params=params,
                timeout=60,
                stream=True,
            ) as dl:
                chunks: list[bytes] = []
                total = 0
                for chunk in dl.iter_content(chunk_size=65536):
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise HTTPException(
                            status_code=502,
                            detail=f"{self.display_name} download too large (>{MAX_DOWNLOAD_BYTES} bytes).",
                        )
                    chunks.append(chunk)
                return dl.status_code, b"".join(chunks)
        except HTTPException:
            raise
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail=f"{self.display_name} download failed: {exc}",
            ) from exc

    def extract_scan_status(self, raw_item: Dict[str, Any]) -> Optional[str]:
        latest = raw_item.get("latestVersion") or {}
        security = latest.get("security") or {}
        return security.get("status") if isinstance(security, dict) else None


# ── Module-level instance (name matches ``{id}_registry`` convention) ──
clawhub_registry = ClawHubRegistry()
