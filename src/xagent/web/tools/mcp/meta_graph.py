import json
import os
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

import requests

GRAPH_BASE_URL = "https://graph.facebook.com/v25.0"
DEFAULT_TIMEOUT_SECONDS = 30
MAX_ERROR_RESPONSE_TEXT_CHARS = 1000


class GraphAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        details: Any = None,
        sensitive_values: set[str] | None = None,
    ):
        super().__init__(message)
        self.details = details
        self.sensitive_values = sensitive_values or set()


def success_response(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def redact_secrets(value: Any, sensitive_values: set[str] | None = None) -> Any:
    tokens = {token for token in sensitive_values or set() if token}
    env_token = os.environ.get("META_ACCESS_TOKEN")
    if env_token:
        tokens.add(env_token)

    if isinstance(value, dict):
        return {
            key: redact_secrets(item, sensitive_values=tokens)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item, sensitive_values=tokens) for item in value]
    if isinstance(value, str):
        redacted = value
        for token in tokens:
            redacted = redacted.replace(token, "[redacted]")
        return redacted
    return value


def error_response(
    message: str,
    *,
    details: Any = None,
    sensitive_values: set[str] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "status": "error",
        "message": redact_secrets(message, sensitive_values=sensitive_values),
    }
    if details is not None:
        payload["details"] = redact_secrets(details, sensitive_values=sensitive_values)
    return json.dumps(payload, ensure_ascii=False)


def graph_error_response(e: GraphAPIError) -> str:
    return error_response(
        str(e), details=e.details, sensitive_values=e.sensitive_values
    )


def user_token() -> str:
    token = os.environ.get("META_ACCESS_TOKEN")
    if not token:
        raise ValueError("META_ACCESS_TOKEN environment variable is missing")
    return token


def graph_headers(token: str, *, form: bool = False) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    if form:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    return headers


def response_error_text(response: Any) -> str:
    response_text = str(getattr(response, "text", "")).strip()
    if len(response_text) > MAX_ERROR_RESPONSE_TEXT_CHARS:
        return response_text[:MAX_ERROR_RESPONSE_TEXT_CHARS] + "... [truncated]"
    return response_text


def graph_request(
    method: str,
    path: str,
    *,
    token: str | None = None,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> Any:
    request_token = token or user_token()
    response = requests.request(
        method=method,
        url=f"{GRAPH_BASE_URL}{path}",
        headers=graph_headers(request_token, form=method.upper() != "GET"),
        params=params,
        data=data,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        details = response_json(response)
        message = str(exc)
        response_text = response_error_text(response)
        if response_text:
            message = f"{message} - {response_text}"
        sensitive_values = {request_token, os.environ.get("META_ACCESS_TOKEN", "")}
        raise GraphAPIError(
            message, details=details, sensitive_values=sensitive_values
        ) from exc

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def response_json(response: Any) -> Any:
    try:
        return response.json()
    except Exception:
        return None


def bounded_limit(limit: int, maximum: int = 100) -> int:
    return max(1, min(int(limit), maximum))


def get_permissions(token: str | None = None) -> dict[str, str]:
    """Fetch this token's /me/permissions edge as {permission: status}.

    A user can decline an individual permission on Meta's consent screen (or
    a configured Login Configuration can simply omit one an app normally
    requests), and the token is still issued either way -- unlike /me, which
    only proves the token identifies *some* user, this is what actually
    proves whether a specific permission was granted.
    """
    result = graph_request("GET", "/me/permissions", token=token)
    entries = result.get("data") if isinstance(result, dict) else None
    if not isinstance(entries, list):
        return {}
    return {
        str(entry["permission"]): str(entry.get("status", ""))
        for entry in entries
        if isinstance(entry, dict) and entry.get("permission")
    }


def missing_permissions(
    required: Iterable[str], granted: Mapping[str, str]
) -> list[str]:
    """Required permissions that are not in a "granted" state."""
    return [
        permission for permission in required if granted.get(permission) != "granted"
    ]


def is_public_image_url(image_url: str) -> bool:
    parsed = urlparse(image_url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
