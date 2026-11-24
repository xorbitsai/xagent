"""Xinference video provider implementation."""

import asyncio
import base64
import json
import logging
import math
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Self
from urllib import parse
from urllib.request import url2pathname

import aiohttp
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from ....config import (
    get_media_download_max_bytes,
    get_xinference_saas_video_async_enabled,
)
from ...tools.core.media_errors import (
    _PROVIDER_UNAVAILABLE_STATUSES,
    MediaParameterError,
    redact_provider_detail,
)
from ...utils.public_download import PublicDownloadError, download_public_input
from ..chat.token_context import MediaCallType
from ..media_metering import record_provider_usage
from ..public_http import create_public_model_aiohttp_session
from .base import (
    BaseVideoModel,
    BilledVideoTaskError,
    BilledVideoTaskTimeout,
    VideoGenerationCapabilities,
)

logger = logging.getLogger(__name__)
# "No response body yet", distinct from anything a server could return.
_NOT_RECEIVED: Any = object()
REFERENCE_MEDIA_FETCH_TIMEOUT = 60.0


async def fetch_running_xinference_models(
    base_url: str, api_key: str | None, *, timeout: float = 30.0
) -> dict[str, dict[str, Any]]:
    """Read the live catalog keyed by the exact UID used for generation."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with create_public_model_aiohttp_session(
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as session:
        async with session.get(
            f"{base_url.rstrip('/')}/v1/models", headers=headers
        ) as response:
            response.raise_for_status()
            payload = await response.json()
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return {}
    return {
        item["id"]: item
        for item in entries
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
    }


def lightning_inference_steps(model_info: dict[str, Any]) -> int | None:
    """Xinference advertises a step count when the instance supports the switch."""
    steps = model_info.get("lightning_inference_steps")
    if type(steps) is int and steps > 0:
        return steps
    return None


async def _load_capability_catalog(
    base_url: str, api_key: str | None
) -> dict[str, dict[str, Any]]:
    try:
        return await fetch_running_xinference_models(base_url, api_key, timeout=10)
    except Exception as exc:
        logger.warning(
            "Could not read Xinference video capabilities: %s", type(exc).__name__
        )
        return {}


def _to_plain_data(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item) for item in value]
    if hasattr(value, "model_dump"):
        return _to_plain_data(value.model_dump())
    if hasattr(value, "dict"):
        return _to_plain_data(value.dict())
    if hasattr(value, "__dict__"):
        return {
            key: _to_plain_data(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def _first_video_item(data: Any) -> dict[str, Any]:
    data = _to_plain_data(data)
    if isinstance(data, dict):
        items = data.get("data")
        if isinstance(items, list) and items:
            first = items[0]
            return first if isinstance(first, dict) else {"raw": first}
        return data
    return {"raw": data}


def _video_url_from_item(item: dict[str, Any]) -> Optional[str]:
    url = item.get("url") or item.get("video_url")
    if url:
        return str(url)
    b64_json = item.get("b64_json")
    if b64_json:
        return f"data:video/mp4;base64,{b64_json}"
    return None


def _size_from_resolution_ratio(
    resolution: Optional[str], ratio: Optional[str]
) -> Optional[str]:
    if not resolution:
        return None
    resolution_text = str(resolution).lower().strip()
    if "x" in resolution_text:
        return resolution_text
    if not resolution_text.endswith("p"):
        return None
    try:
        short_side = int(resolution_text[:-1])
    except ValueError:
        return None
    ratio_text = (ratio or "16:9").strip()
    if ":" not in ratio_text:
        return None
    width_text, height_text = ratio_text.split(":", 1)
    try:
        width_ratio = int(width_text)
        height_ratio = int(height_text)
    except ValueError:
        return None
    if width_ratio <= 0 or height_ratio <= 0:
        return None
    if width_ratio >= height_ratio:
        width = int(round(short_side * width_ratio / height_ratio))
        height = short_side
    else:
        width = short_side
        height = int(round(short_side * height_ratio / width_ratio))
    return f"{width}x{height}"


# The xinference-saas ("Model API") prepaid video endpoint accepts a closed set
# of top-level fields instead of the open-source server's ``kwargs`` JSON string.
SAAS_VIDEO_SIZES: tuple[tuple[int, int], ...] = (
    (1344, 768),
    (768, 1344),
    (1024, 768),
    (768, 1024),
    (768, 768),
    (1792, 768),
    (768, 1792),
)
SAAS_VIDEO_MIN_SECONDS = 4
SAAS_VIDEO_MAX_SECONDS = 15
SAAS_VIDEO_MAX_SEED = 2**31 - 1

# Async job API (``XAGENT_XINFERENCE_SAAS_VIDEO_ASYNC``). The server expires a
# job that sat queued for 30 minutes, so the wait has to outlast that plus the
# render itself; a shorter configured timeout (the platform default is 30
# minutes) would give up on jobs the server is still going to run.
SAAS_VIDEO_QUEUE_EXPIRY_SECONDS = 1800.0
SAAS_VIDEO_RENDER_ALLOWANCE_SECONDS = 900.0
SAAS_VIDEO_MIN_JOB_DEADLINE_SECONDS = (
    SAAS_VIDEO_QUEUE_EXPIRY_SECONDS + SAAS_VIDEO_RENDER_ALLOWANCE_SECONDS
)
SAAS_VIDEO_JOB_POLL_SECONDS = 5.0
SAAS_VIDEO_QUEUE_FULL_RETRY_SECONDS = 30.0
SAAS_VIDEO_SUBMIT_ATTEMPTS = 3
SAAS_VIDEO_CONTENT_ATTEMPTS = 3
SAAS_VIDEO_SUBMIT_PAUSE_SECONDS = 2.0
SAAS_VIDEO_REQUEST_TIMEOUT_SECONDS = 60.0
SAAS_VIDEO_CONTENT_TIMEOUT_SECONDS = 600.0
_SAAS_ACCEPTED_STATUSES = (200, 201, 202)
# Failure codes the server documents as not charged.
_SAAS_UNBILLED_FAILURE_CODES = frozenset(
    {"video_expired", "video_cancelled", "upstream_failure"}
)


@dataclass
class _JobResponse:
    status: int
    headers: Any
    body: Any = None
    content: bytes = b""

    @property
    def error_code(self) -> Optional[str]:
        return _saas_error_code(self.body)


def _saas_error_code(body: Any) -> Optional[str]:
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    if isinstance(error, dict) and isinstance(error.get("code"), str):
        return str(error["code"])
    if isinstance(error, str):
        return error
    detail = body.get("detail")
    if isinstance(detail, dict) and isinstance(detail.get("code"), str):
        return str(detail["code"])
    return None


def _saas_error_message(body: Any) -> str:
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if error:
            return str(error)
    return ""


def _retry_after_seconds(headers: Any, default: float) -> float:
    try:
        value = float(headers.get("Retry-After"))
    except (AttributeError, TypeError, ValueError):
        return default
    return value if value > 0 else default


def _saas_allowed_sizes() -> list[str]:
    return [f"{w}x{h}" for w, h in SAAS_VIDEO_SIZES]


def _saas_duration(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        number: float = float("nan")
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
    if not (SAAS_VIDEO_MIN_SECONDS <= number <= SAAS_VIDEO_MAX_SECONDS):
        raise MediaParameterError(
            "Xinference video duration must be between "
            f"{SAAS_VIDEO_MIN_SECONDS} and {SAAS_VIDEO_MAX_SECONDS} seconds",
            field="duration",
            allowed_values={
                "duration": [SAAS_VIDEO_MIN_SECONDS, SAAS_VIDEO_MAX_SECONDS]
            },
        )
    return int(number) if number == int(number) else number


def _saas_dimensions(size: Optional[str]) -> tuple[int, int] | None:
    """Return an allowed pair; other well-formed sizes snap to the closest ratio.

    Callers (and the video tool prompt) commonly ask for sizes like 1280x720 or
    1920x1080 that the SaaS API does not list. Rejecting them would just make
    the model retry blind, so they snap to the allowed size whose aspect ratio
    is nearest (ties go to the larger area). Text that is not ``WxH`` is
    rejected with the allowed values.
    """
    if not size:
        return None
    text = str(size).lower().strip()
    try:
        width_text, height_text = text.split("x", 1)
        width, height = int(width_text), int(height_text)
        if width <= 0 or height <= 0:
            raise ValueError
    except ValueError:
        raise MediaParameterError(
            f"Unsupported video size {size!r}; use one of "
            + ", ".join(_saas_allowed_sizes()),
            field="size",
            allowed_values={"size": _saas_allowed_sizes()},
        ) from None
    if (width, height) in SAAS_VIDEO_SIZES:
        return (width, height)
    target = math.log(width / height)
    return min(
        SAAS_VIDEO_SIZES,
        key=lambda pair: (
            abs(math.log(pair[0] / pair[1]) - target),
            -pair[0] * pair[1],
        ),
    )


def _saas_seed(value: Any) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= SAAS_VIDEO_MAX_SEED
    ):
        raise MediaParameterError(
            f"seed must be an integer between 0 and {SAAS_VIDEO_MAX_SEED}",
            field="seed",
            allowed_values={"seed": [0, SAAS_VIDEO_MAX_SEED]},
        )
    return int(value)


def _raise_frame_download_error(field: str, exc: PublicDownloadError) -> None:
    """Keep a fetch that may succeed later apart from an unusable input."""
    text = str(exc)
    status_text = text.removeprefix("Remote input returned HTTP ")
    status = int(status_text) if status_text.isdigit() else None
    if "timed out" in text:
        raise TimeoutError(f"The {field} image download timed out") from None
    if "connection failed" in text or (
        status is not None
        and (status >= 500 or status in _PROVIDER_UNAVAILABLE_STATUSES)
    ):
        raise ConnectionError(
            f"The {field} image could not be downloaded: {text}"
        ) from None
    raise MediaParameterError(
        f"The {field} image could not be loaded: {text}", field=field
    ) from None


def _frame_data_url(raw: bytes) -> str:
    """The server refuses remote URLs, so frames go out as data URLs.

    The subtype is only a label; the server decides from the bytes.
    """
    if raw.startswith(b"\x89PNG"):
        subtype = "png"
    elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        subtype = "webp"
    else:
        subtype = "jpeg"
    return f"data:image/{subtype};base64," + base64.b64encode(raw).decode("ascii")


@dataclass
class _SubmitState:
    """What the submit attempts have left uncertain."""

    # An attempt may have created a job whose id we never saw.
    ambiguous: bool = False
    # The server accepted a job but its id could not be read.
    accepted_unreadable: bool = False


class _JobDeadline(Exception):
    """The job wait outlived its deadline."""


class _JobFatal(Exception):
    """Polling can no longer reach the job."""


class XinferenceVideoModel(BaseVideoModel):
    """Xinference video model client using OpenAI-compatible video endpoints.

    With ``saas_api=True`` it targets the xinference-saas prepaid video API
    (top-level fields, ``b64_json`` responses). The default
    keeps the open-source server contract (``kwargs`` JSON string, image and
    first/last-frame endpoints).
    """

    @classmethod
    async def discover_generation_capabilities(
        cls, models: Mapping[str, Self]
    ) -> dict[str, VideoGenerationCapabilities]:
        endpoints = {(model.base_url, model.api_key) for model in models.values()}
        catalogs = await asyncio.gather(
            *(_load_capability_catalog(*endpoint) for endpoint in endpoints)
        )
        catalogs_by_endpoint = dict(zip(endpoints, catalogs, strict=True))
        capabilities = {}
        for model_id, model in models.items():
            model_info = catalogs_by_endpoint[model.base_url, model.api_key].get(
                model._model_uid
            )
            if model_info is None:
                capabilities[model_id] = VideoGenerationCapabilities(
                    supports_lightning=None
                )
                continue
            steps = lightning_inference_steps(model_info)
            capabilities[model_id] = VideoGenerationCapabilities(
                steps is not None, steps
            )
        return capabilities

    def __init__(
        self,
        model_name: str,
        model_uid: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 3600.0,
        abilities: Optional[List[str]] = None,
        model_id: str = "",
        saas_api: bool = False,
    ) -> None:
        self.saas_api = saas_api
        self.model_name = model_name
        # The configured row id usage is billed under; see media_metering.
        self.model_id = model_id
        self._model_uid = model_uid or model_name
        self.base_url = (base_url or "http://localhost:9997").rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._abilities = abilities or ["generate"]

    @property
    def abilities(self) -> List[str]:
        return self._abilities

    def validate_configuration(self) -> None:
        if not self.base_url:
            raise ValueError("Xinference base_url cannot be empty")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def _request_json(
        self,
        session: aiohttp.ClientSession,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        url = f"{self.base_url}{path}"
        accepted = False
        try:
            async with session.request(
                method, url, headers=self._headers(), **kwargs
            ) as response:
                if response.status != 200:
                    try:
                        detail = await response.json()
                    except Exception:
                        detail = await response.text()
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=f"Xinference video request failed: detail: {detail}",
                        headers=response.headers,
                    )
                # The server answered 200, so the generation already ran. A
                # body that will not decode is a billed response, not a
                # retryable failure: retrying would generate (and pay for) it
                # again.
                accepted = True
                try:
                    return await response.json()
                except Exception as exc:  # noqa: BLE001 - positional classification
                    raise BilledVideoTaskError(
                        f"Xinference video response could not be decoded: {exc}"
                    ) from exc
        except asyncio.CancelledError:
            # Cancelled after the 200 but before the caller got the body: the
            # render happened and is billed, but its duration never arrives.
            # The caller only records once it holds the body, so this is the
            # row for this call.
            if accepted:
                record_provider_usage(self, MediaCallType.VIDEO, 0)
            raise

    async def _post_video_json(
        self,
        session: aiohttp.ClientSession,
        path: str,
        body: dict[str, Any],
    ) -> Any:
        return await self._request_json(session, "POST", path, json=body)

    async def _post_video_form(
        self,
        session: aiohttp.ClientSession,
        path: str,
        fields: dict[str, Any],
        files: dict[str, bytes],
    ) -> Any:
        data = aiohttp.FormData()
        for key, value in fields.items():
            if value is None:
                continue
            data.add_field(key, str(value))
        for key, content in files.items():
            data.add_field(
                key,
                content,
                filename="image",
                content_type="application/octet-stream",
            )
        return await self._request_json(session, "POST", path, data=data)

    @staticmethod
    def _resolve_allowed_local_media_path(
        path: Path, allowed_local_media_roots: Optional[list[str | Path]]
    ) -> Path:
        allowed_roots = [
            Path(root).expanduser().resolve()
            for root in allowed_local_media_roots or []
        ]
        if not allowed_roots:
            raise ValueError(
                "Xinference local reference media requires an allowed workspace root"
            )

        expanded_path = path.expanduser()
        if expanded_path.is_absolute():
            resolved_path = expanded_path.resolve()
        else:
            resolved_path = (allowed_roots[0] / expanded_path).resolve()
        if any(
            resolved_path == root or resolved_path.is_relative_to(root)
            for root in allowed_roots
        ):
            return resolved_path

        raise ValueError(
            "Access denied: local reference media path is outside the allowed workspace"
        )

    async def _read_media_bytes(
        self,
        media: str | bytes,
        allowed_local_media_roots: Optional[list[str | Path]] = None,
        timeout: Optional[float] = None,
    ) -> bytes:
        if isinstance(media, bytes):
            return media

        media_text = str(media).strip()
        if media_text.startswith("data:"):
            header, _, encoded = media_text.partition(",")
            if ";base64" not in header or not encoded:
                raise ValueError("Unsupported media data URL")
            return base64.b64decode(encoded)

        parsed = parse.urlparse(media_text)
        if parsed.scheme == "file":
            path = Path(url2pathname(parsed.path))
            allowed_path = await asyncio.to_thread(
                self._resolve_allowed_local_media_path,
                path,
                allowed_local_media_roots,
            )
            return await asyncio.to_thread(allowed_path.read_bytes)
        if parsed.scheme in {"http", "https"}:
            timeout_value = timeout if timeout is not None else self.timeout
            response = await download_public_input(
                media_text,
                max_bytes=get_media_download_max_bytes("image"),
                timeout=min(REFERENCE_MEDIA_FETCH_TIMEOUT, timeout_value),
            )
            return response.content

        if parsed.scheme:
            raise ValueError(
                "Xinference image-to-video requires an image URL, file URL, data URL, or local file path"
            )

        path = await asyncio.to_thread(
            self._resolve_allowed_local_media_path,
            Path(media_text),
            allowed_local_media_roots,
        )
        if await asyncio.to_thread(path.is_file):
            return await asyncio.to_thread(path.read_bytes)

        raise ValueError(
            "Xinference image-to-video requires an image URL, file URL, data URL, or local file path"
        )

    async def create_video_task(
        self,
        content: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt = ""
        image: Optional[str] = None
        first_frame: Optional[str] = None
        last_frame: Optional[str] = None
        for item in content:
            if item.get("type") == "text":
                prompt = str(item.get("text") or "")
            elif item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url")
                role = item.get("role")
                if role == "last_frame":
                    last_frame = url
                elif role == "first_frame":
                    first_frame = url
                elif image is None:
                    image = url

        result = await self.generate_video(
            prompt=prompt,
            input_reference=image,
            first_frame_image_url=first_frame,
            last_frame_image_url=last_frame,
            **kwargs,
        )
        return {
            "task_id": result.get("task_id"),
            "status": result.get("status"),
            "video_url": result.get("video_url"),
            "raw_response": result.get("raw_response"),
        }

    async def get_video_task(self, task_id: str) -> dict[str, Any]:
        if not task_id:
            raise ValueError("task_id cannot be empty")
        return {
            "task_id": task_id,
            "status": "succeeded",
            "raw_response": {},
        }

    def _record_response_usage(
        self, result: Any, requested_seconds: Any, resolution: Optional[str]
    ) -> tuple[Any, Optional[dict[str, Any]]]:
        """Record the one usage row of a received response; never raises.

        Returns ``(data, item)``, with ``item`` None when the body is too
        malformed to read -- recorded as an unmeasured call, since the video
        was rendered either way. Recorded before the response is validated.
        The server's own duration wins; the requested length is the fallback,
        since a server that reports none rendered what was asked.
        """
        try:
            data = _to_plain_data(result)
            item = _first_video_item(data)
            reported_seconds = item.get("duration") or item.get("seconds")
            if self.saas_api and isinstance(data, dict):
                usage = data.get("usage")
                if isinstance(usage, dict) and usage.get("video_seconds"):
                    reported_seconds = usage["video_seconds"]
            item_resolution = str(item.get("resolution") or resolution or "")
        except Exception as exc:  # noqa: BLE001 - positional classification
            logger.warning("Xinference video response was malformed: %s", exc)
            record_provider_usage(self, MediaCallType.VIDEO, 0)
            return result, None
        record_provider_usage(
            self,
            MediaCallType.VIDEO,
            reported_seconds if reported_seconds is not None else requested_seconds,
            resolution=item_resolution,
        )
        return data, item

    async def _load_saas_frame(
        self,
        field: str,
        value: Optional[str],
        allowed_local_media_roots: Optional[list[str | Path]],
        timeout: float,
    ) -> Optional[str]:
        """A frame (URL, file, FileRef-resolved path or data URL) as a data URL.

        Image validity is the server's call: it answers 400 for a bad image.
        """
        if not value:
            return None
        try:
            raw = await self._read_media_bytes(
                value,
                allowed_local_media_roots=allowed_local_media_roots,
                timeout=timeout,
            )
        except PublicDownloadError as exc:
            _raise_frame_download_error(field, exc)
        except Exception as exc:  # noqa: BLE001 - classified as bad input
            raise MediaParameterError(
                f"The {field} image could not be loaded: {redact_provider_detail(exc)}",
                field=field,
            ) from None
        return _frame_data_url(raw)

    async def _generate_saas_video(
        self,
        *,
        prompt: str,
        reference_image: Optional[str],
        last_frame_image_url: Optional[str],
        allowed_local_media_roots: Optional[list[str | Path]],
        extra_references: dict[str, Any],
        seconds: Any,
        size: Optional[str],
        resolution: Optional[str],
        ratio: Optional[str],
        seed: Any,
        timeout: float,
    ) -> dict[str, Any]:
        # Every refusal and validation happens before a request is built.
        for key in (
            "reference_image_urls",
            "reference_video_urls",
            "reference_audio_urls",
        ):
            if extra_references.get(key):
                raise MediaParameterError(
                    f"This Xinference video model does not accept {key}; use a "
                    "first frame (input_reference) and optional last frame, or "
                    "choose another video model",
                    field=key,
                )
        duration_value = _saas_duration(seconds)
        # No size is sent unless the caller named one, so with an image the
        # server picks the size that matches it.
        dimensions = _saas_dimensions(size)
        seed_value = _saas_seed(seed)
        first_frame = await self._load_saas_frame(
            "first_frame_image",
            reference_image,
            allowed_local_media_roots,
            timeout,
        )
        last_frame = await self._load_saas_frame(
            "last_frame_image",
            last_frame_image_url,
            allowed_local_media_roots,
            timeout,
        )

        if get_xinference_saas_video_async_enabled():
            return await self._run_saas_video_job(
                prompt=prompt,
                duration_value=duration_value,
                dimensions=dimensions,
                seed_value=seed_value,
                resolution=resolution,
                ratio=ratio,
                timeout=timeout,
                first_frame=first_frame,
                last_frame=last_frame,
            )

        body: dict[str, Any] = {
            "model": self._model_uid,
            "prompt": prompt,
            "n": 1,
            "response_format": "b64_json",
        }
        if duration_value is not None:
            body["duration"] = duration_value
        if dimensions is not None:
            body["width"], body["height"] = dimensions
        if seed_value is not None:
            body["seed"] = seed_value
        if first_frame is not None:
            body["first_frame_image"] = first_frame
        if last_frame is not None:
            body["last_frame_image"] = last_frame

        result: Any = _NOT_RECEIVED
        try:
            async with create_public_model_aiohttp_session(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                result = await self._post_video_json(
                    session, "/v1/video/generations", body
                )
        except asyncio.CancelledError:
            if result is not _NOT_RECEIVED:
                self._record_response_usage(result, duration_value, resolution)
            raise
        except BilledVideoTaskError:
            record_provider_usage(self, MediaCallType.VIDEO, 0)
            raise
        except Exception as exc:
            logger.error(
                "Xinference video generation failed: %s",
                redact_provider_detail(exc),
            )
            raise RuntimeError(f"Xinference video generation failed: {exc}") from exc

        data, video_item = self._record_response_usage(
            result, duration_value, resolution
        )
        if video_item is None:
            raise BilledVideoTaskError("Xinference video response was malformed")
        video_url = _video_url_from_item(video_item)
        if not video_url:
            raise BilledVideoTaskError(
                "Xinference video generation response did not include "
                "url, video_url, or b64_json"
            )
        usage = data.get("usage") if isinstance(data, dict) else None
        reported = usage.get("video_seconds") if isinstance(usage, dict) else None
        return {
            "task_id": video_item.get("id") or video_item.get("task_id"),
            "model": self.model_name,
            "status": "succeeded",
            "video_url": video_url,
            "last_frame_url": None,
            "seed": video_item.get("seed", seed_value),
            "resolution": video_item.get("resolution") or resolution,
            "ratio": video_item.get("ratio") or ratio,
            # Same precedence the usage recorder applied, so the returned
            # duration is the quantity that was billed.
            "duration": reported
            or video_item.get("duration")
            or video_item.get("seconds")
            or duration_value,
            "frames": video_item.get("frames"),
            "raw_response": data,
        }

    # ----------------------------------------------------------------------
    # xinference-saas async job API (POST/GET/DELETE /v1/videos)
    # ----------------------------------------------------------------------

    def _job_host(self) -> str:
        """Bare host: the job paths carry their own ``/v1``."""
        host = self.base_url
        return host[: -len("/v1")] if host.endswith("/v1") else host

    async def _job_call(
        self,
        session: aiohttp.ClientSession,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        read_bytes: bool = False,
        timeout: float = SAAS_VIDEO_REQUEST_TIMEOUT_SECONDS,
        trace: Optional[dict[str, Any]] = None,
    ) -> _JobResponse:
        """One request; ``trace["status"]`` is set as soon as headers arrive.

        The trace lets a caller that is cancelled while the body is still being
        read know the server had already answered.
        """
        kwargs: dict[str, Any] = {
            "headers": {**self._headers(), **(headers or {})},
            "timeout": aiohttp.ClientTimeout(total=timeout),
        }
        if json_body is not None:
            kwargs["json"] = json_body
        async with session.request(
            method, f"{self._job_host()}{path}", **kwargs
        ) as response:
            if trace is not None:
                trace["status"] = response.status
            if read_bytes and response.status == 200:
                limit = get_media_download_max_bytes("video")
                if (response.content_length or 0) > limit:
                    raise ValueError(f"video content exceeds {limit} bytes")
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.content.iter_chunked(1024 * 1024):
                    total += len(chunk)
                    if total > limit:
                        raise ValueError(f"video content exceeds {limit} bytes")
                    chunks.append(chunk)
                return _JobResponse(
                    response.status, response.headers, content=b"".join(chunks)
                )
            try:
                body = await response.json(content_type=None)
            except Exception:  # noqa: BLE001 - an unreadable body is just absent
                body = None
            return _JobResponse(response.status, response.headers, body)

    def _job_http_error(
        self, path: str, response: _JobResponse, message: str
    ) -> aiohttp.ClientResponseError:
        url = URL(f"{self._job_host()}{path}")
        info = aiohttp.RequestInfo(url, "POST", CIMultiDictProxy(CIMultiDict()), url)
        return aiohttp.ClientResponseError(
            info,
            (),
            status=response.status,
            message=message,
            headers=response.headers,
        )

    def _record_job_usage(
        self, quantity: Any, resolution: Optional[str], job_id: str, why: str
    ) -> None:
        logger.info(
            "Recording xinference-saas video usage for %s (%s): %s",
            job_id,
            why,
            quantity,
        )
        record_provider_usage(
            self, MediaCallType.VIDEO, quantity, resolution=resolution or ""
        )

    async def _submit_video_job(
        self,
        session: aiohttp.ClientSession,
        body: dict[str, Any],
        deadline: float,
    ) -> str:
        """POST the job; return its id.

        One ``Idempotency-Key`` covers every attempt, so a POST whose answer
        was lost cannot start a second render, and an accepted answer whose
        body is unreadable is recovered by replaying the POST with that key.

        A full queue (429) is retried after the server's ``Retry-After``; once
        the attempts run out it is raised as a plain 429, which the retry
        wrapper may try again with a *new* key. That is only safe while no
        earlier attempt could have created a job, so once one might have
        (``ambiguous``) a 429 is raised as a non-retryable error instead.

        A call that is accepted but never yields an id is recorded as one
        unmeasured call before the error leaves, because the job may run on
        and be billed.
        """
        key = uuid.uuid4().hex
        sub = _SubmitState()
        try:
            return await self._submit_attempts(session, body, deadline, key, sub)
        except BaseException:
            if sub.accepted_unreadable:
                self._record_job_usage(
                    0, None, "unknown", "accepted job id could not be read"
                )
            raise

    async def _submit_attempts(
        self,
        session: aiohttp.ClientSession,
        body: dict[str, Any],
        deadline: float,
        key: str,
        sub: "_SubmitState",
    ) -> str:
        for attempt in range(SAAS_VIDEO_SUBMIT_ATTEMPTS):
            last = attempt == SAAS_VIDEO_SUBMIT_ATTEMPTS - 1
            trace: dict[str, Any] = {}
            try:
                response = await self._job_call(
                    session,
                    "POST",
                    "/v1/videos",
                    json_body=body,
                    headers={"Idempotency-Key": key},
                    trace=trace,
                )
            except asyncio.CancelledError:
                if trace.get("status") in _SAAS_ACCEPTED_STATUSES:
                    sub.accepted_unreadable = True
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                # The request may have reached the server and created the job.
                sub.ambiguous = True
                if last:
                    raise RuntimeError(
                        "Xinference video job submission failed and may have "
                        f"created a job: {type(exc).__name__}"
                    ) from None
                await asyncio.sleep(SAAS_VIDEO_SUBMIT_PAUSE_SECONDS)
                continue
            if response.status in _SAAS_ACCEPTED_STATUSES:
                job_id = (
                    response.body.get("id") if isinstance(response.body, dict) else None
                )
                if job_id:
                    return str(job_id)
                sub.ambiguous = True
                sub.accepted_unreadable = True
                if last:
                    raise BilledVideoTaskError(
                        "Xinference accepted the video job but its id could not be read"
                    )
                await asyncio.sleep(SAAS_VIDEO_SUBMIT_PAUSE_SECONDS)
                continue
            if response.status == 429:
                wait = _retry_after_seconds(
                    response.headers, SAAS_VIDEO_QUEUE_FULL_RETRY_SECONDS
                )
                if last or time.monotonic() + wait >= deadline:
                    if sub.ambiguous:
                        raise RuntimeError(
                            "Xinference video queue is full after an earlier "
                            "submission that may have created a job"
                        )
                    raise self._job_http_error(
                        "/v1/videos",
                        response,
                        "Xinference video queue is full: "
                        f"{response.error_code or 'rate limited'}",
                    )
                logger.warning(
                    "Xinference video queue is full; retrying in %.0fs", wait
                )
                await asyncio.sleep(wait)
                continue
            if response.status >= 500:
                sub.ambiguous = True
                if last:
                    raise RuntimeError(
                        "Xinference video job submission failed with HTTP "
                        f"{response.status} and may have created a job"
                    )
                await asyncio.sleep(SAAS_VIDEO_SUBMIT_PAUSE_SECONDS)
                continue
            raise self._job_http_error(
                "/v1/videos",
                response,
                f"Xinference video request failed: "
                f"{response.error_code or response.body}",
            )
        raise AssertionError("unreachable: the last attempt always returns or raises")

    async def _abandon_video_job(
        self,
        session: aiohttp.ClientSession,
        job_id: str,
        requested_seconds: Any,
        resolution: Optional[str],
    ) -> str:
        """DELETE a job nobody is waiting for, then record what it may cost.

        The DELETE answer alone cannot say what was billed: the same 200
        covers a queued job cancelled for free, a failed job (no-op) and a
        completed job whose file was deleted without a refund. So the job is
        read again afterwards and classified by that final server state:

        - ``failed`` with a documented unbilled code: ``"free"``, no row;
        - ``completed``: ``"charged"``, its ``usage.video_seconds`` (the
          requested seconds if absent);
        - anything else, or a job that cannot be read: ``"unknown"``, one
          unmeasured row, unless the DELETE itself said ``409
          video_in_progress`` (documented as charged), which records the
          requested seconds.
        """
        render_running = False
        outcome = "unknown"
        billed: Any = 0
        try:
            try:
                response = await asyncio.wait_for(
                    self._job_call(
                        session, "DELETE", f"/v1/videos/{job_id}", timeout=15.0
                    ),
                    timeout=20.0,
                )
                render_running = (
                    response.status == 409
                    and response.error_code == "video_in_progress"
                )
            except Exception as exc:  # noqa: BLE001 - best effort
                logger.warning(
                    "Could not cancel xinference-saas video job %s: %s",
                    job_id,
                    redact_provider_detail(exc),
                )

            try:
                final = await asyncio.wait_for(
                    self._job_call(
                        session, "GET", f"/v1/videos/{job_id}", timeout=15.0
                    ),
                    timeout=20.0,
                )
                job = final.body if isinstance(final.body, dict) else {}
                status = job.get("status") if final.status == 200 else None
                if status == "completed":
                    outcome = "charged"
                    billed = self._job_billed_seconds(job, requested_seconds)
                elif status == "failed" and _saas_error_code(job) in (
                    _SAAS_UNBILLED_FAILURE_CODES
                ):
                    outcome = "free"
            except Exception as exc:  # noqa: BLE001 - best effort
                logger.warning(
                    "Could not read xinference-saas video job %s after cancelling: %s",
                    job_id,
                    redact_provider_detail(exc),
                )
        finally:
            # Also runs when a second cancel lands mid-cleanup: the defaults
            # above then record the job as unmeasured instead of not at all.
            if outcome == "unknown" and render_running:
                outcome = "charged"
                billed = requested_seconds or 0
            if outcome == "charged":
                self._record_job_usage(billed or 0, resolution, job_id, "job charged")
            elif outcome == "unknown":
                self._record_job_usage(0, resolution, job_id, "final state unknown")
        return outcome

    async def _run_saas_video_job(
        self,
        *,
        prompt: str,
        duration_value: int | float | None,
        dimensions: tuple[int, int] | None,
        seed_value: int | None,
        resolution: Optional[str],
        ratio: Optional[str],
        timeout: float,
        first_frame: Optional[str] = None,
        last_frame: Optional[str] = None,
    ) -> dict[str, Any]:
        seconds = math.ceil(duration_value) if duration_value is not None else None
        size = f"{dimensions[0]}x{dimensions[1]}" if dimensions else None
        body: dict[str, Any] = {"model": self._model_uid, "prompt": prompt}
        if seconds is not None:
            body["seconds"] = seconds
        if size is not None:
            body["size"] = size
        if seed_value is not None:
            body["seed"] = seed_value
        if first_frame is not None:
            body["input_reference"] = first_frame
        if last_frame is not None:
            body["last_frame_reference"] = last_frame

        deadline = time.monotonic() + max(timeout, SAAS_VIDEO_MIN_JOB_DEADLINE_SECONDS)
        async with create_public_model_aiohttp_session(
            timeout=aiohttp.ClientTimeout(total=None)
        ) as session:
            try:
                job_id = await self._submit_video_job(session, body, deadline)
            except (asyncio.CancelledError, BilledVideoTaskError):
                raise
            except Exception as exc:
                logger.error(
                    "Xinference video generation failed: %s",
                    redact_provider_detail(exc),
                )
                raise RuntimeError(
                    f"Xinference video generation failed: {exc}"
                ) from exc

            state: dict[str, Any] = {"status": None}
            try:
                completed = await self._wait_for_video_job(
                    session, job_id, deadline, state
                )
            except asyncio.CancelledError:
                await self._abandon_video_job(session, job_id, seconds, size)
                raise
            except _JobDeadline as exc:
                outcome = await self._abandon_video_job(session, job_id, seconds, size)
                message = (
                    f"Timed out waiting for video job {job_id}; "
                    f"last status: {state['status']}"
                )
                if outcome == "free":
                    raise TimeoutError(message) from exc
                raise BilledVideoTaskTimeout(message) from exc
            except _JobFatal as exc:
                outcome = await self._abandon_video_job(session, job_id, seconds, size)
                if outcome == "free":
                    raise RuntimeError(str(exc)) from None
                raise BilledVideoTaskError(str(exc)) from None

            billed = self._job_billed_seconds(completed, seconds)
            self._record_job_usage(billed, size or resolution, job_id, "completed")
            content: Optional[_JobResponse] = None
            for attempt in range(SAAS_VIDEO_CONTENT_ATTEMPTS):
                try:
                    content = await self._job_call(
                        session,
                        "GET",
                        f"/v1/videos/{job_id}/content",
                        read_bytes=True,
                        timeout=SAAS_VIDEO_CONTENT_TIMEOUT_SECONDS,
                    )
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    if attempt == SAAS_VIDEO_CONTENT_ATTEMPTS - 1:
                        raise BilledVideoTaskError(
                            f"Could not download video job {job_id}: "
                            f"{type(exc).__name__}"
                        ) from None
                    await asyncio.sleep(SAAS_VIDEO_SUBMIT_PAUSE_SECONDS)
                    continue
                except ValueError as exc:
                    raise BilledVideoTaskError(
                        f"Could not download video job {job_id}: {exc}"
                    ) from None
                break
            assert content is not None
            if content.status != 200 or not content.content:
                raise BilledVideoTaskError(
                    f"Could not download video job {job_id}: HTTP {content.status} "
                    f"{content.error_code or ''}".strip()
                )

        encoded = base64.b64encode(content.content).decode("ascii")
        return {
            "task_id": job_id,
            "model": self.model_name,
            "status": "succeeded",
            "video_url": f"data:video/mp4;base64,{encoded}",
            "last_frame_url": None,
            "seed": completed.get("seed", seed_value),
            "resolution": size or resolution,
            "ratio": ratio,
            "duration": billed,
            "frames": None,
            "raw_response": completed,
        }

    @staticmethod
    def _job_billed_seconds(job: dict[str, Any], requested: Any) -> Any:
        usage = job.get("usage")
        reported = usage.get("video_seconds") if isinstance(usage, dict) else None
        return reported if reported else requested

    async def _wait_for_video_job(
        self,
        session: aiohttp.ClientSession,
        job_id: str,
        deadline: float,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Poll to ``completed`` and return the job; raise on any other end."""
        while True:
            try:
                response = await self._job_call(session, "GET", f"/v1/videos/{job_id}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "Transient error polling video job %s: %s",
                    job_id,
                    redact_provider_detail(exc),
                )
                response = None
            except ValueError as exc:
                # Refused by the public-HTTP layer: polling can never work.
                raise _JobFatal(
                    f"Video job {job_id} could not be polled: "
                    f"{redact_provider_detail(exc)}"
                ) from None
            if response is not None and response.status == 200:
                job = response.body if isinstance(response.body, dict) else {}
                status = job.get("status")
                state["status"] = status if isinstance(status, str) else state["status"]
                if status == "completed":
                    return job
                if status == "failed":
                    code = _saas_error_code(job)
                    detail = _saas_error_message(job)
                    if code in _SAAS_UNBILLED_FAILURE_CODES:
                        raise RuntimeError(
                            f"Video job {job_id} failed ({code}): {detail}"
                        )
                    self._record_job_usage(0, None, job_id, "failed, code unknown")
                    raise BilledVideoTaskError(
                        f"Video job {job_id} failed ({code or 'unknown'}): {detail}"
                    )
                position = job.get("queue_position")
                if position is not None:
                    logger.info("Video job %s queue position %s", job_id, position)
            elif (
                response is not None
                and 400 <= response.status < 500
                and response.status not in (408, 429)
            ):
                # The job id stopped resolving or the key stopped working; the
                # job may still run, so stop waiting without a second submit.
                raise _JobFatal(
                    f"Video job {job_id} could not be polled: HTTP {response.status} "
                    f"{response.error_code or ''}".strip()
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _JobDeadline()
            await asyncio.sleep(min(SAAS_VIDEO_JOB_POLL_SECONDS, remaining))

    async def generate_video(
        self,
        prompt: str = "",
        content: Optional[list[dict[str, Any]]] = None,
        wait_for_result: bool = True,
        poll_interval: float = 30.0,
        timeout: Optional[float] = None,
        input_reference: Optional[str] = None,
        image_url: Optional[str] = None,
        image: Optional[str] = None,
        first_frame_image_url: Optional[str] = None,
        last_frame_image_url: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        n: int = 1,
        seconds: Optional[int | float | str] = None,
        duration: Optional[int | float | str] = None,
        size: Optional[str] = None,
        resolution: Optional[str] = None,
        ratio: Optional[str] = None,
        allowed_local_media_roots: Optional[list[str | Path]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        _ = wait_for_result, poll_interval
        if not self.has_ability("generate"):
            raise RuntimeError("This model doesn't support video generation")
        if n not in (None, 1, "1"):
            # Only the first video of a batch is ever returned to the caller,
            # yet the server renders all of them; reject rather than bill for
            # videos nobody receives.
            raise MediaParameterError(
                "Xinference video generation supports one video per request",
                field="n",
                allowed_values={"n": [1]},
            )

        if content:
            for item in content:
                if item.get("type") == "text" and not prompt:
                    prompt = str(item.get("text") or "")
                elif item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url")
                    role = item.get("role")
                    if role == "first_frame" and not first_frame_image_url:
                        first_frame_image_url = url
                    elif role == "last_frame" and not last_frame_image_url:
                        last_frame_image_url = url
                    elif not input_reference:
                        input_reference = url

        if not prompt:
            raise ValueError("prompt cannot be empty")

        effective_timeout = timeout if timeout is not None else self.timeout

        if self.saas_api:
            return await self._generate_saas_video(
                prompt=prompt,
                reference_image=first_frame_image_url
                or input_reference
                or image_url
                or image,
                last_frame_image_url=last_frame_image_url,
                allowed_local_media_roots=allowed_local_media_roots,
                extra_references=kwargs,
                seconds=seconds if seconds is not None else duration,
                size=size or _size_from_resolution_ratio(resolution, ratio),
                resolution=resolution,
                ratio=ratio,
                seed=kwargs.get("seed"),
                timeout=effective_timeout,
            )

        effective_seconds = seconds if seconds is not None else duration
        if effective_seconds is not None and "seconds" not in kwargs:
            kwargs["seconds"] = effective_seconds
        if not size:
            size = _size_from_resolution_ratio(resolution, ratio)
        if size and "size" not in kwargs:
            kwargs["size"] = size

        reference_image = input_reference or image_url or image or first_frame_image_url

        result: Any = _NOT_RECEIVED
        try:
            timeout_obj = aiohttp.ClientTimeout(total=effective_timeout)
            async with create_public_model_aiohttp_session(
                timeout=timeout_obj
            ) as session:
                if first_frame_image_url and last_frame_image_url:
                    first_frame = await self._read_media_bytes(
                        first_frame_image_url,
                        allowed_local_media_roots=allowed_local_media_roots,
                        timeout=effective_timeout,
                    )
                    last_frame = await self._read_media_bytes(
                        last_frame_image_url,
                        allowed_local_media_roots=allowed_local_media_roots,
                        timeout=effective_timeout,
                    )
                    result = await self._post_video_form(
                        session,
                        "/v1/video/generations/flf",
                        {
                            "model": self._model_uid,
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "n": n,
                            "kwargs": json.dumps(kwargs),
                        },
                        {"first_frame": first_frame, "last_frame": last_frame},
                    )
                elif reference_image:
                    image_bytes = await self._read_media_bytes(
                        reference_image,
                        allowed_local_media_roots=allowed_local_media_roots,
                        timeout=effective_timeout,
                    )
                    result = await self._post_video_form(
                        session,
                        "/v1/video/generations/image",
                        {
                            "model": self._model_uid,
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "n": n,
                            "kwargs": json.dumps(kwargs),
                        },
                        {"image": image_bytes},
                    )
                else:
                    result = await self._post_video_json(
                        session,
                        "/v1/video/generations",
                        {
                            "model": self._model_uid,
                            "prompt": prompt,
                            "n": n,
                            "kwargs": json.dumps(kwargs),
                        },
                    )
        except asyncio.CancelledError:
            # Cancelled while the session closed, after the body arrived: the
            # body is in hand, so this call is recorded from it.
            if result is not _NOT_RECEIVED:
                self._record_response_usage(result, effective_seconds, resolution)
            raise
        except BilledVideoTaskError:
            record_provider_usage(self, MediaCallType.VIDEO, 0)
            raise
        except Exception as exc:
            logger.error(
                "Xinference video generation failed: %s",
                redact_provider_detail(exc),
            )
            raise RuntimeError(f"Xinference video generation failed: {exc}") from exc

        data, video_item = self._record_response_usage(
            result, effective_seconds, resolution
        )
        if video_item is None:
            raise BilledVideoTaskError("Xinference video response was malformed")
        video_url = _video_url_from_item(video_item)
        if not video_url:
            raise BilledVideoTaskError(
                "Xinference video generation response did not include "
                "url, video_url, or b64_json"
            )
        return {
            "task_id": video_item.get("id") or video_item.get("task_id"),
            "model": self.model_name,
            "status": "succeeded",
            "video_url": video_url,
            "last_frame_url": video_item.get("last_frame_url"),
            "seed": video_item.get("seed"),
            "resolution": video_item.get("resolution") or resolution,
            "ratio": video_item.get("ratio") or ratio,
            "duration": video_item.get("duration")
            or video_item.get("seconds")
            or effective_seconds,
            "frames": video_item.get("frames"),
            "raw_response": data,
        }
