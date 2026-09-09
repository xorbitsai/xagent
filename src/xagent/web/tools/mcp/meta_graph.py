import json
import os
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


def graph_headers(
    token: str, *, form: bool = False, as_json: bool = False
) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    if as_json:
        headers["Content-Type"] = "application/json"
    elif form:
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
    json_body: dict[str, Any] | None = None,
) -> Any:
    """Issue a Graph API request and return the decoded JSON body.

    ``data`` is sent form-encoded (the shape the Pages/Instagram endpoints
    take). ``json_body`` is sent as an ``application/json`` body instead --
    the WhatsApp Cloud API's ``/{phone_number_id}/messages`` endpoint takes
    nested objects (``template.components``, ``text.body``) that
    form-encoding can't express. The two are mutually exclusive. ``json=``
    is always passed to ``requests`` (as ``None`` when unused); `requests`
    only substitutes a JSON body ``if not data and json is not None``, so a
    stray ``json=None`` alongside a real ``data=`` value never changes what
    goes over the wire.
    """
    if data is not None and json_body is not None:
        raise ValueError("graph_request takes either data or json_body, not both")
    request_token = token or user_token()
    response = requests.request(
        method=method,
        url=f"{GRAPH_BASE_URL}{path}",
        headers=graph_headers(
            request_token,
            form=method.upper() != "GET",
            as_json=json_body is not None,
        ),
        params=params,
        data=data,
        json=json_body,
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


def is_public_image_url(image_url: str) -> bool:
    parsed = urlparse(image_url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
