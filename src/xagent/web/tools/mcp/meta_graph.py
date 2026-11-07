import json
import logging
import os
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

from ....core.utils import public_requests as requests

GRAPH_BASE_URL = "https://graph.facebook.com/v25.0"
# Instagram Login (Composio's Instagram toolkit, #3533). Its tokens are not
# valid on graph.facebook.com.
INSTAGRAM_GRAPH_BASE_URL = "https://graph.instagram.com/v25.0"
# graph_request never sends a token anywhere else.
ALLOWED_GRAPH_BASE_URLS = frozenset({GRAPH_BASE_URL, INSTAGRAM_GRAPH_BASE_URL})
DEFAULT_TIMEOUT_SECONDS = 30
MAX_ERROR_RESPONSE_TEXT_CHARS = 1000

# A Composio-backed grant (#3531) launches with these instead of
# META_ACCESS_TOKEN. Composio keeps the user token and injects it itself, so
# every Graph call goes through Proxy Execute.
COMPOSIO_PROXY_URL = "https://backend.composio.dev/api/v3.1/tools/execute/proxy"
COMPOSIO_API_KEY_ENV = "COMPOSIO_API_KEY"
COMPOSIO_CONNECTED_ACCOUNT_ID_ENV = "COMPOSIO_CONNECTED_ACCOUNT_ID"
COMPOSIO_USER_ID_ENV = "COMPOSIO_USER_ID"
PROXY_FILE_UPLOAD_UNSUPPORTED = (
    "Uploading a local file is not supported for this connection. A workspace "
    "image works when it is a PNG or JPEG of at most 10 MB and this deployment "
    "has a public HTTPS base URL; otherwise use a public image_url"
)
# Proxy responses carry Page tokens in plaintext (/me/accounts). They are
# redacted from everything this process reports afterwards.
_proxy_response_secrets: set[str] = set()


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


class ComposioProxyError(GraphAPIError):
    """Composio itself refused the proxied call (bad key, unknown account,
    rate limit...). Not a Graph error: ``details`` carries no Graph error
    body, so nothing mistakes it for a Graph permission error, and nothing
    retries it."""


def success_response(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def redact_secrets(value: Any, sensitive_values: set[str] | None = None) -> Any:
    tokens = {token for token in sensitive_values or set() if token}
    env_token = os.environ.get("META_ACCESS_TOKEN")
    if env_token:
        tokens.add(env_token)
    for env_var in (
        COMPOSIO_API_KEY_ENV,
        COMPOSIO_CONNECTED_ACCOUNT_ID_ENV,
        COMPOSIO_USER_ID_ENV,
    ):
        composio_value = os.environ.get(env_var)
        if composio_value:
            tokens.add(composio_value)
    tokens.update(_proxy_response_secrets)

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
    return _truncated(str(getattr(response, "text", "")).strip())


def graph_request(
    method: str,
    path: str,
    *,
    token: str | None = None,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    base_url: str = GRAPH_BASE_URL,
) -> Any:
    """Issue a Graph API request and return the decoded JSON body.

    ``base_url`` is one of ``ALLOWED_GRAPH_BASE_URLS``; Instagram Login calls
    use ``INSTAGRAM_GRAPH_BASE_URL``.

    ``data`` is sent form-encoded (the shape the Pages/Instagram endpoints
    take). ``json_body`` is sent as an ``application/json`` body instead --
    the WhatsApp Cloud API's ``/{phone_number_id}/messages`` endpoint takes
    nested objects (``template.components``, ``text.body``) that
    form-encoding can't express. ``files`` enables multipart uploads and may
    accompany ``data``; requests must generate that body's boundary, so this
    helper deliberately omits an explicit Content-Type for multipart calls.
    JSON cannot be combined with either form data or files.
    """
    if json_body is not None and (data is not None or files is not None):
        raise ValueError(
            "graph_request takes either json_body or form/multipart data, not both"
        )
    if base_url not in ALLOWED_GRAPH_BASE_URLS:
        raise ValueError("graph_request base_url is not an allowed Graph host")
    if uses_composio_proxy():
        return _composio_proxy_request(
            method,
            path,
            base_url=base_url,
            token=token,
            params=params,
            data=data,
            json_body=json_body,
            files=files,
            timeout=timeout,
        )
    request_token = token or user_token()
    if json_body is not None:
        content_type = "application/json"
    elif files is not None:
        content_type = None
    elif method.upper() != "GET":
        content_type = "application/x-www-form-urlencoded"
    else:
        content_type = None
    body_kwargs: dict[str, Any] = {}
    if data is not None:
        body_kwargs["data"] = data
    if json_body is not None:
        body_kwargs["json"] = json_body
    if files is not None:
        body_kwargs["files"] = files
    response = requests.request(
        method=method,
        url=f"{base_url}{path}",
        headers=graph_headers(request_token, content_type=content_type),
        params=params,
        timeout=timeout,
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


def uses_composio_proxy() -> bool:
    """Whether this process serves a Composio-backed grant (#3531)."""
    return bool(os.environ.get(COMPOSIO_CONNECTED_ACCOUNT_ID_ENV))


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} environment variable is missing")
    return value


def _remember_proxy_response_secrets(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "access_token" and isinstance(item, str) and item:
                _proxy_response_secrets.add(item)
            else:
                _remember_proxy_response_secrets(item)
    elif isinstance(value, list):
        for item in value:
            _remember_proxy_response_secrets(item)


def _without_access_token(url: str) -> str:
    parsed = urlparse(url)
    query = [
        (name, value)
        for name, value in parse_qsl(parsed.query, keep_blank_values=True)
        if name != "access_token"
    ]
    return urlunparse(parsed._replace(query=urlencode(query)))


def _strip_tokens_from_paging_links(value: Any) -> None:
    """Graph echoes a query-parameter token into ``paging.next``/``previous``.

    In proxy mode the Page token travels as the ``access_token`` query
    parameter, so those links would hand it to the model. Drop it; the
    cursor (``after``/``before``) is all a caller needs.
    """
    if isinstance(value, dict):
        paging = value.get("paging")
        if isinstance(paging, dict):
            for key in ("next", "previous"):
                link = paging.get(key)
                if isinstance(link, str):
                    paging[key] = _without_access_token(link)
        for item in value.values():
            _strip_tokens_from_paging_links(item)
    elif isinstance(value, list):
        for item in value:
            _strip_tokens_from_paging_links(item)


def _truncated(text: str) -> str:
    if len(text) > MAX_ERROR_RESPONSE_TEXT_CHARS:
        return text[:MAX_ERROR_RESPONSE_TEXT_CHARS] + "... [truncated]"
    return text


def _composio_proxy_request(
    method: str,
    path: str,
    *,
    token: str | None,
    params: dict[str, Any] | None,
    data: dict[str, Any] | None,
    json_body: dict[str, Any] | None,
    files: dict[str, Any] | None,
    timeout: float,
    # Required: ``graph_request`` has already checked it against the allowlist.
    base_url: str,
) -> Any:
    """Send one Graph call through Composio's Proxy Execute endpoint.

    Composio injects the user token, so the request carries no
    Authorization header. A Page token goes in as the ``access_token`` query
    parameter, which Graph prefers over the injected one. The proxy has no
    multipart support, so ``files`` is refused.
    """
    if files is not None:
        raise ValueError(PROXY_FILE_UPLOAD_UNSUPPORTED)
    api_key = _required_env(COMPOSIO_API_KEY_ENV)
    connected_account_id = _required_env(COMPOSIO_CONNECTED_ACCOUNT_ID_ENV)
    user_id = _required_env(COMPOSIO_USER_ID_ENV)
    sensitive_values = {api_key, connected_account_id, user_id, token or ""}

    def redacted_text(text: str) -> str:
        # Redact before truncating, so a secret cut in half cannot survive.
        return _truncated(redact_secrets(text, sensitive_values=sensitive_values))

    parameters = [
        {"name": str(name), "value": str(value), "type": "query"}
        for name, value in (params or {}).items()
    ]
    if token:
        parameters.append({"name": "access_token", "value": token, "type": "query"})
    payload: dict[str, Any] = {
        "endpoint": f"{base_url}{path}",
        "method": method.upper(),
        "connected_account_id": connected_account_id,
        # Required: the Composio project enforces 2FA on private connections.
        "user_id": user_id,
    }
    if parameters:
        payload["parameters"] = parameters
    if json_body is not None:
        payload["body"] = json_body
    elif data is not None:
        payload["body"] = data

    response = requests.request(
        method="POST",
        url=COMPOSIO_PROXY_URL,
        headers={
            "x-api-key": api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    body = response_json(response)

    if not 200 <= response.status_code < 300:
        error = body.get("error") if isinstance(body, dict) else None
        error = error if isinstance(error, dict) else {}
        slug = error.get("slug")
        composio_message = error.get("message")
        message = f"Composio proxy request failed with HTTP {response.status_code}"
        if slug:
            message = f"{message} ({slug})"
        if composio_message:
            message = f"{message}: {composio_message}"
        raise ComposioProxyError(
            redacted_text(message),
            details={
                "composio_error": redact_secrets(
                    {
                        "status": response.status_code,
                        "slug": slug,
                        "message": composio_message,
                    },
                    sensitive_values=sensitive_values,
                )
            },
            sensitive_values=sensitive_values,
        )
    if not isinstance(body, dict):
        raise ComposioProxyError(
            "Composio proxy returned a response that is not a JSON object",
            sensitive_values=sensitive_values,
        )

    upstream_data = body.get("data")
    _remember_proxy_response_secrets(upstream_data)
    _strip_tokens_from_paging_links(upstream_data)
    upstream_status = body.get("status")
    if isinstance(upstream_status, int) and upstream_status >= 400:
        message = f"Graph API returned HTTP {upstream_status} through Composio"
        if upstream_data is not None:
            message = f"{message} - {json.dumps(upstream_data)}"
        raise GraphAPIError(
            redacted_text(message),
            details=upstream_data,
            sensitive_values=sensitive_values,
        )
    return upstream_data if upstream_data is not None else {}


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
