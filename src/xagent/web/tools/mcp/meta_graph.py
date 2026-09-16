import json
import logging
import os
from typing import Any
from urllib.parse import quote, urlparse

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


def graph_headers(token: str, *, content_type: str | None = None) -> dict[str, str]:
    """Bearer + Accept headers, plus an explicit Content-Type when given.

    One parameter instead of the earlier ``form``/``as_json`` pair: there
    is exactly one body shape per call (form, JSON, or none for GET), so a
    single ``content_type`` says which without the two flags needing to
    stay mutually exclusive by convention.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    if content_type:
        headers["Content-Type"] = content_type
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
    form-encoding can't express. The two are mutually exclusive, and only
    whichever one is actually given is passed on to ``requests`` -- not
    relying on ``requests``' own ``if not data and json is not None``
    precedence rule to sort out an unused one.
    """
    if data is not None and json_body is not None:
        raise ValueError("graph_request takes either data or json_body, not both")
    request_token = token or user_token()
    if json_body is not None:
        content_type = "application/json"
    elif method.upper() != "GET":
        content_type = "application/x-www-form-urlencoded"
    else:
        content_type = None
    body_kwargs: dict[str, Any] = {}
    if data is not None:
        body_kwargs["data"] = data
    if json_body is not None:
        body_kwargs["json"] = json_body
    response = requests.request(
        method=method,
        url=f"{GRAPH_BASE_URL}{path}",
        headers=graph_headers(request_token, content_type=content_type),
        params=params,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        **body_kwargs,
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


def graph_path(object_id: str, suffix: str | None = None) -> str:
    """URL-quote a Graph node id and optionally append an edge suffix."""
    if not object_id or not str(object_id).strip():
        raise ValueError("object id is required")
    path = f"/{quote(str(object_id).strip(), safe='')}"
    if suffix:
        path = f"{path}/{suffix}"
    return path


def auth_status(logger: logging.Logger, connector_label: str) -> str:
    """Shared body for a connector's `<x>_auth_status` tool: check whether
    the injected Meta access token is usable at all (an identity read, not
    a check of any connector-specific scope)."""
    try:
        me = graph_request("GET", "/me", params={"fields": "id,name,email"})
        return success_response(
            authenticated=True,
            user={"id": me.get("id"), "name": me.get("name"), "email": me.get("email")},
        )
    except GraphAPIError as e:
        logger.error("Error checking %s auth status: %s", connector_label, e)
        return graph_error_response(e)
    except Exception as e:
        logger.error("Error checking %s auth status: %s", connector_label, e)
        return error_response(str(e))
