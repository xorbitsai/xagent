"""HTTP client for offloading DeepDoc PDF parsing to a remote Xinference server.

Set ``XAGENT_DEEPDOC_XINFERENCE_URL`` to route PDF parsing to a GPU host
instead of running local ONNX inference. Any failure raises
:class:`DeepDocRemoteError`, which the caller treats as "fall back to local
parsing" -- fallback is always on, which is what makes the switch transparent.

This module deliberately depends on nothing but ``httpx`` and
``xagent.config``. In particular it must not import ``deepdoc.py``: the
image-saving function is injected by the caller instead, which keeps the import
graph acyclic and lets the client be tested without loading any parser.

Server API contract
-------------------

Xinference serves DeepDoc's whole-document pipeline as ``task="parse"`` on the
OCR endpoint (xorbitsai/inference#5299). One request runs
``parse_into_bboxes()`` over the entire PDF server-side, so there is nothing to
stitch together on this side.

.. code-block:: text

    POST {base_url}/v1/images/ocr
    Authorization: Bearer <token>            # omitted when unauthenticated

    Request (multipart/form-data):
      model   str     required            launched DeepDoc model UID
      image   binary  required            the PDF itself, sent as application/pdf
      kwargs  str     required            JSON object: {"task": "parse", ...}

    Response 200 application/json:
    {
      "task": "parse",
      "elements": [
        {
          "type": "title",               // text | title | table | figure
          "text": "...",                 // complete HTML for tables
          "image_base64": "...",         // omitted when the element has no crop
          "metadata": {"x0": 70.7, "x1": 256.3, "top": 77.3, "bottom": 96.3,
                       "page_number": 1, "layout_type": "title",
                       "layoutno": "title-0", "col_id": 0,
                       "positions": [[1, 70.7, 256.3, 77.3, 96.3]]}
        }
      ]
    }

    Errors: 400 non-PDF upload, invalid kwargs, or a render exceeding the
            200 MP per-page / 1 GP per-document budget; 401 auth failure;
            500 inference failure.

A managed deployment answers with that object wrapped in a gateway envelope,
verified against Xinference Cloud::

    {"code": 0, "message": "Request successful.", "data": {"task": ..., "elements": [...]}}

Both shapes are accepted. The envelope also carries failures the gateway
returns with HTTP 200 and a non-zero ``code``, which ``raise_for_status`` cannot
see, so that case is turned into an error rather than read as an empty result.

Three details of that contract drive the code below:

* ``task="parse"`` consumes a PDF, not a page image, and rejects ``pages`` and
  ``dpi``. Only PDFs may be sent, which is why the caller gates on the
  extension before calling in.
* ``image_base64`` is *omitted* for elements without a crop rather than sent as
  ``null``, and ``col_id`` is absent on elements the pipeline never assigned to
  a column. Both are read with that in mind.
* Crops are always PNG. The server encodes with ``image.save(buffer,
  format="PNG")`` and picks it deliberately, since crops are line art, so the
  caller writes them with a ``.png`` suffix rather than sniffing the bytes.
* Coordinates in ``metadata`` accumulate across pages, matching what local
  ``parse_into_bboxes`` produces, so the translated result stays interchangeable
  with local output when fallback kicks in.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from ...config import (
    get_deepdoc_xinference_api_key,
    get_deepdoc_xinference_model_uid,
    get_deepdoc_xinference_password,
    get_deepdoc_xinference_timeout_seconds,
    get_deepdoc_xinference_url,
    get_deepdoc_xinference_username,
)

logger = logging.getLogger(__name__)

PARSE_ENDPOINT = "/v1/images/ocr"
TOKEN_ENDPOINT = "/token"

# The only format ``task="parse"`` accepts. The caller gates on this too, so a
# non-PDF never reaches the network; the constant keeps both checks honest.
SUPPORTED_EXTENSION = ".pdf"
PDF_MIME_TYPE = "application/pdf"

# Only table and figure elements carry an image downstream, so asking for the
# other crops would inflate the response for nothing. This is the server
# default as well, sent explicitly so the payload does not change under us.
IMAGE_SCOPE = "table_figure"

# The server's own cap on the render scale (``MAX_PARSE_ZOOMIN`` there). Above
# it a request is refused outright, so the value is mirrored rather than
# discovered by a 400.
MAX_ZOOMIN = 6

# Connecting to and writing headers against a reachable host is fast; only the
# parse itself is slow, so the connect/pool budgets stay short regardless of
# how long the configured read timeout is.
_CONNECT_TIMEOUT_SECONDS = 10.0
_POOL_TIMEOUT_SECONDS = 10.0
# The token exchange is a database lookup, not inference, so it gets its own
# short budget instead of the whole-document read timeout.
_TOKEN_TIMEOUT_SECONDS = 30.0

ProgressCallback = Callable[[float, str], None]
SaveImage = Callable[[bytes], str]


class DeepDocRemoteError(Exception):
    """Remote parsing failed and the caller should fall back to local parsing.

    Every failure mode -- unreachable host, timeout, 4xx/5xx, unparsable or
    malformed body, undecodable image, failed image write -- is reported as
    this single type so callers need only one ``except`` clause.
    """


def is_remote_configured() -> bool:
    """Return whether remote DeepDoc parsing is configured.

    A malformed URL makes the config getter raise. Degrading to local parsing
    with a warning is the right response: a typo in one environment variable
    must not break every document parse.
    """
    try:
        return get_deepdoc_xinference_url() is not None
    except ValueError as exc:
        logger.warning(
            "Ignoring remote DeepDoc configuration and parsing locally: %s", exc
        )
        return False


def _report(
    callback: Optional[ProgressCallback], progress: float, message: str
) -> None:
    """Emit a progress update, swallowing anything the sink raises.

    Progress reporting is never worth a parse. The sink reaches websocket and
    task-state broadcasting, and ``DeepDocProgressAdapter`` re-raises out of its
    own ``except`` handler, so a failing sink is not hypothetical. Left
    unguarded on the entry call it would bypass the caller's fallback; on the
    completion call it would discard a successful parse along with every crop
    already written to disk.
    """
    if callback is None:
        return
    try:
        callback(progress, message)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Remote DeepDoc progress callback failed: %s", exc)


def _fetch_jwt(
    base_url: str,
    username: str,
    password: str,
    transport: Optional[httpx.BaseTransport],
) -> str:
    """Exchange credentials for a bearer token at ``POST /token``.

    Raises:
        httpx.HTTPError: On any transport failure or non-2xx status.
        ValueError: If the response carries no usable ``access_token``.
    """
    with httpx.Client(
        timeout=httpx.Timeout(_TOKEN_TIMEOUT_SECONDS), transport=transport
    ) as client:
        response = client.post(
            f"{base_url}{TOKEN_ENDPOINT}",
            json={"username": username, "password": password},
        )
        response.raise_for_status()
        payload = response.json()

    token = payload.get("access_token") if isinstance(payload, dict) else None
    if not isinstance(token, str) or not token:
        raise ValueError("Remote DeepDoc token response carried no 'access_token'")
    return token


def _build_headers(
    base_url: str, transport: Optional[httpx.BaseTransport]
) -> dict[str, str]:
    """Return the auth headers for a parse request.

    A configured username/password pair is exchanged for a short-lived JWT;
    otherwise a configured API key is sent as the bearer token directly.
    Xinference accepts either, and an unauthenticated cluster needs neither, so
    an empty mapping is a valid result.
    """
    username = get_deepdoc_xinference_username()
    password = get_deepdoc_xinference_password()
    # Gate on a non-blank password but send it unstripped: whitespace can be
    # significant in a secret, yet a password of only spaces is a misconfigured
    # variable rather than a credential. Treating it as one would take this
    # branch, ignore a perfectly good API key, and 401 forever -- visible only
    # as a silent permanent fallback to local parsing.
    if username and password and password.strip():
        return {
            "Authorization": f"Bearer {_fetch_jwt(base_url, username, password, transport)}"
        }

    api_key = get_deepdoc_xinference_api_key()
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def _post_pdf(
    file_path: str | BytesIO,
    *,
    ext: str,
    zoomin: int,
    transport: Optional[httpx.BaseTransport],
) -> dict[str, Any]:
    """Upload the PDF for ``task="parse"`` and return the decoded JSON body.

    Args:
        file_path: Path to the PDF, or its bytes in memory.
        ext: Lower-cased file extension. Only ``".pdf"`` is accepted, because
            that is all ``task="parse"`` can consume.
        zoomin: PDF render scale forwarded to the server.
        transport: Optional transport override. This is the seam tests use to
            install an ``httpx.MockTransport`` instead of reaching the network.

    Raises:
        httpx.HTTPError: On any transport failure or non-2xx status.
        ValueError: If remote mode is unconfigured, the extension is not
            ``.pdf``, ``zoomin`` is out of range, or the body is not a JSON
            object.
        OSError: If a file path cannot be read.
    """
    base_url = get_deepdoc_xinference_url()
    if base_url is None:
        raise ValueError("Remote DeepDoc parsing is not configured")
    if ext.lower() != SUPPORTED_EXTENSION:
        raise ValueError(
            f"Remote DeepDoc parsing supports {SUPPORTED_EXTENSION} only, got {ext!r}"
        )
    # The server rejects anything outside this range with a 400. Checking here
    # costs nothing and saves uploading a whole PDF only to have it refused --
    # and the local parser accepts the value regardless, so the fallback still
    # honors what the caller asked for.
    if (
        not isinstance(zoomin, int)
        or isinstance(zoomin, bool)
        or not 1 <= zoomin <= MAX_ZOOMIN
    ):
        raise ValueError(
            f"Remote DeepDoc zoomin must be an integer between 1 and {MAX_ZOOMIN}, "
            f"got {zoomin!r}"
        )

    timeout_seconds = float(get_deepdoc_xinference_timeout_seconds())
    timeout = httpx.Timeout(
        connect=_CONNECT_TIMEOUT_SECONDS,
        read=timeout_seconds,
        write=timeout_seconds,
        pool=_POOL_TIMEOUT_SECONDS,
    )
    # The server parses this field as JSON, so the whole task configuration
    # travels as one string rather than as sibling form fields.
    data = {
        "model": get_deepdoc_xinference_model_uid(),
        "kwargs": json.dumps(
            {"task": "parse", "zoomin": zoomin, "image_scope": IMAGE_SCOPE}
        ),
    }
    # The token exchange happens before the upload so a rejected credential
    # costs nothing; it shares the caller's transport for testability.
    headers = _build_headers(base_url, transport)

    def _send(filename: str, fh: Any) -> dict[str, Any]:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            response = client.post(
                f"{base_url}{PARSE_ENDPOINT}",
                headers=headers,
                data=data,
                files={"image": (filename, fh, PDF_MIME_TYPE)},
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(
                f"Remote DeepDoc response body is {type(payload).__name__}, "
                "expected a JSON object"
            )
        return payload

    if isinstance(file_path, BytesIO):
        # The buffer is owned by the caller, who may still read from it after a
        # fallback, so put the cursor back where it was found.
        original_position = file_path.tell()
        try:
            file_path.seek(0)
            return _send(f"document{ext}", file_path)
        finally:
            file_path.seek(original_position)

    with open(file_path, "rb") as fh:
        return _send(Path(file_path).name, fh)


def _extract_elements(payload: dict[str, Any]) -> list[Any]:
    """Return the element list, whether or not the response is enveloped.

    A self-hosted Xinference answers with the parse result directly, while the
    managed gateway wraps it as ``{"code": 0, "message": ..., "data": {...}}``.
    Both are legitimate, so unwrap a ``data`` object when the elements are not
    where a direct response would put them.

    A non-zero ``code`` is surfaced rather than swallowed: the gateway uses it to
    report a failure it still returned with HTTP 200, which ``raise_for_status``
    cannot catch.
    """
    # Checked before unwrapping: a gateway reporting a failure this way is
    # authoritative, whatever it left in `data`. Deciding afterwards would let a
    # non-zero code accompanied by an element list read as success.
    code = payload.get("code")
    if code is not None and code != 0:
        raise ValueError(
            f"Remote DeepDoc reported code {code!r}: {payload.get('message')!r}"
        )

    for candidate in (payload, payload.get("data")):
        if not isinstance(candidate, dict):
            continue
        elements = candidate.get("elements")
        if isinstance(elements, list):
            return elements

    raise ValueError(
        "Remote DeepDoc response is missing an 'elements' list "
        f"(top-level keys: {sorted(payload)})"
    )


def _normalize_elements(
    payload: dict[str, Any], save_image: SaveImage
) -> list[dict[str, Any]]:
    """Validate the response elements and materialize their images on disk.

    Each element's ``image_base64`` is replaced by an ``image`` key holding the
    path string of the saved file, or ``None`` when the element carries no
    image. A path string is exactly what the caller's existing image handling
    already accepts, so no downstream change is needed. The field is omitted
    rather than nulled for elements without a crop, so its absence is the
    normal case and not an error.

    Args:
        payload: Decoded response body.
        save_image: Writes decoded image bytes and returns the resulting path.

    Raises:
        ValueError: If the payload does not carry a list of elements that each
            look like a parsed element.
        binascii.Error: If an element's image is not valid base64.

    Any crop already written before a failure is removed before the exception
    propagates, so a rejected response leaves nothing behind.
    """
    elements = _extract_elements(payload)

    # Crops are written as the loop goes, so a failure partway through would
    # otherwise leave the earlier ones orphaned under the artifacts directory --
    # nothing GCs that tree, and the local fallback then writes its own full set
    # alongside them under fresh names.
    written: list[str] = []
    try:
        return _normalize_elements_inner(elements, save_image, written)
    except Exception:
        for path in written:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not remove orphaned crop %s: %s", path, exc)
        raise


def _normalize_elements_inner(
    elements: list[Any], save_image: SaveImage, written: list[str]
) -> list[dict[str, Any]]:
    """Validate and normalize elements, recording every crop written to disk."""
    normalized: list[dict[str, Any]] = []
    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            raise ValueError(
                f"Remote DeepDoc element {index} is {type(element).__name__}, "
                "expected an object"
            )
        missing = [key for key in ("type", "text") if key not in element]
        if missing:
            raise ValueError(
                f"Remote DeepDoc element {index} is missing "
                f"{', '.join(repr(key) for key in missing)}"
            )
        # Reject a present-but-null or wrongly-typed value here rather than
        # letting it reach the translator, where it would surface as a pydantic
        # ValidationError -- an exception the caller does not catch, so the
        # local fallback would be skipped and the whole parse would fail.
        wrong_type = [
            key for key in ("type", "text") if not isinstance(element[key], str)
        ]
        if wrong_type:
            raise ValueError(
                f"Remote DeepDoc element {index} has non-string "
                f"{', '.join(repr(key) for key in wrong_type)}"
            )

        normalized_element = dict(element)
        image_base64 = normalized_element.pop("image_base64", None)
        if image_base64:
            # Checked explicitly so a JSON number or array names the element and
            # the field. Left to b64decode it raises TypeError, which the caller
            # reports as "argument should be a bytes-like object" -- true but
            # useless for working out which response was wrong.
            if not isinstance(image_base64, str):
                raise ValueError(
                    f"Remote DeepDoc element {index} has non-string "
                    f"'image_base64' ({type(image_base64).__name__})"
                )
            # validate=True raises on non-alphabet characters instead of
            # silently dropping them, so corruption surfaces as a
            # DeepDocRemoteError and the caller falls back, rather than a
            # truncated image being written to disk and treated as valid.
            image_bytes = base64.b64decode(image_base64, validate=True)
            image_path = save_image(image_bytes)
            written.append(image_path)
            normalized_element["image"] = image_path
        else:
            normalized_element["image"] = None
        normalized.append(normalized_element)

    return normalized


def parse_document_remote(
    file_path: str | BytesIO,
    *,
    ext: str,
    save_image: SaveImage,
    callback: Optional[ProgressCallback] = None,
    zoomin: int = 3,
    _transport: Optional[httpx.BaseTransport] = None,
) -> list[dict[str, Any]]:
    """Parse a PDF on the remote DeepDoc server in a single ``task=parse`` call.

    Args:
        file_path: Path to the PDF, or its bytes in memory.
        ext: File extension of the document. Only ``".pdf"`` is supported.
        save_image: Writes decoded image bytes and returns the resulting path.
            Injected by the caller so this module stays independent of the
            parser that owns the artifact directory layout.
        callback: Optional progress sink taking ``(fraction, message)``. The
            ``"message (1.23s)"`` shape of the completion notice matches what
            local DeepDoc emits, so the existing progress adapter strips the
            timing suffix and dedupes statuses without any change.
        zoomin: PDF render scale forwarded to the server.
        _transport: Test-only transport override.

    Returns:
        The parsed elements, each with an ``image`` path string or ``None``.

    Raises:
        DeepDocRemoteError: On any failure. Callers fall back to local parsing.
    """
    started = time.monotonic()
    _report(callback, 0.05, "Uploading document to remote DeepDoc server")

    try:
        payload = _post_pdf(file_path, ext=ext, zoomin=zoomin, transport=_transport)
        elements = _normalize_elements(payload, save_image)
    except httpx.HTTPError as exc:
        raise DeepDocRemoteError(f"Remote DeepDoc request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DeepDocRemoteError(
            f"Remote DeepDoc returned a non-JSON response: {exc}"
        ) from exc
    except binascii.Error as exc:
        raise DeepDocRemoteError(
            f"Remote DeepDoc returned an undecodable image: {exc}"
        ) from exc
    except ValueError as exc:
        raise DeepDocRemoteError(
            f"Remote DeepDoc returned an unusable response: {exc}"
        ) from exc
    except OSError as exc:
        # Covers both an unreadable source file and a failed image write.
        raise DeepDocRemoteError(f"Remote DeepDoc parse failed: {exc}") from exc
    except Exception as exc:
        # save_image is caller-supplied, so it may fail in ways this module
        # cannot enumerate. Fallback must still be the outcome.
        raise DeepDocRemoteError(f"Remote DeepDoc parse failed: {exc}") from exc

    elapsed = time.monotonic() - started
    _report(callback, 1.0, f"Remote DeepDoc parse finished ({elapsed:.2f}s)")
    logger.info(
        "Remote DeepDoc parsed %s into %d elements in %.2fs",
        ext,
        len(elements),
        elapsed,
    )
    return elements
