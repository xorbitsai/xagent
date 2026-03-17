import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_package_version
from json import JSONDecodeError, loads
from typing import Any, cast

from fastapi import APIRouter
from packaging.version import InvalidVersion, Version

system_router = APIRouter(prefix="/api/system", tags=["system"])

_latest_version_cache: str | None = None
_latest_version_fetched_at = 0.0
_latest_version_ttl_seconds = 600.0


def _resolve_backend_version() -> str:
    injected_version = os.getenv("XAGENT_VERSION") or ""
    if injected_version:
        return injected_version

    try:
        package_version = get_package_version("xagent")
        if package_version:
            return package_version
    except PackageNotFoundError:
        pass

    return "unknown"


def _normalize_version(value: str) -> Version | None:
    normalized = value.strip()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    if not normalized:
        return None
    try:
        return Version(normalized)
    except InvalidVersion:
        return None


def _build_display_version(version: str, commit: str) -> str:
    raw_version = version.strip()
    if not raw_version or raw_version == "unknown":
        return "unknown"

    normalized = raw_version[1:] if raw_version.startswith("v") else raw_version
    plus_index = normalized.find("+")
    without_local = normalized[:plus_index] if plus_index >= 0 else normalized
    base_version = (
        without_local.split(".dev")[0] if ".dev" in without_local else without_local
    )

    short_hash = commit.strip()
    if not short_hash and plus_index >= 0:
        local_part = normalized[plus_index + 1 :]
        hash_match = re.search(r"g([0-9a-f]{5,40})", local_part, re.IGNORECASE)
        short_hash = hash_match.group(1) if hash_match else ""

    if not base_version:
        return "unknown"

    return f"v{base_version}{f'-{short_hash[:5]}' if short_hash else ''}"


def _resolve_latest_version() -> str | None:
    global _latest_version_cache, _latest_version_fetched_at

    now = time.time()
    if now - _latest_version_fetched_at <= _latest_version_ttl_seconds:
        return _latest_version_cache

    repo = os.getenv("XAGENT_GITHUB_REPO") or "xorbitsai/xagent"
    encoded_repo = urllib.parse.quote(repo, safe="/")
    url = f"https://api.github.com/repos/{encoded_repo}/releases/latest"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "xagent-version-check",
        },
    )

    latest: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            payload_bytes = cast(bytes, response.read())
            payload = payload_bytes.decode("utf-8")
            parsed = cast(dict[str, Any], loads(payload))
            tag_name = parsed.get("tag_name") if isinstance(parsed, dict) else None
            if isinstance(tag_name, str) and tag_name.strip():
                latest = tag_name.strip()
    except (urllib.error.URLError, TimeoutError, JSONDecodeError, UnicodeDecodeError):
        latest = None

    _latest_version_cache = latest
    _latest_version_fetched_at = now
    return latest


def _resolve_is_latest(current: str, latest: str | None) -> bool | None:
    if current == "unknown" or not latest:
        return None
    current_version = _normalize_version(current)
    latest_version = _normalize_version(latest)
    if current_version is None or latest_version is None:
        return None
    return current_version >= latest_version


@system_router.get("/version")
async def get_version() -> dict[str, str | bool | None]:
    raw_commit = os.getenv("XAGENT_GIT_COMMIT") or os.getenv("GITHUB_SHA") or ""
    commit = raw_commit[:12] if raw_commit else ""
    build_time = os.getenv("XAGENT_BUILD_TIME") or ""
    current_version = _resolve_backend_version()
    latest_version = _resolve_latest_version()
    return {
        "version": current_version,
        "display_version": _build_display_version(current_version, raw_commit),
        "commit": commit,
        "build_time": build_time,
        "latest_version": latest_version,
        "is_latest": _resolve_is_latest(current_version, latest_version),
    }
