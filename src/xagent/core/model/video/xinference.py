"""Xinference video provider implementation."""

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any, List, Optional
from urllib import parse
from urllib.request import url2pathname

import aiohttp

from .base import BaseVideoModel

logger = logging.getLogger(__name__)
REFERENCE_MEDIA_FETCH_TIMEOUT = 60.0


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


class XinferenceVideoModel(BaseVideoModel):
    """Xinference video model client using OpenAI-compatible video endpoints."""

    def __init__(
        self,
        model_name: str,
        model_uid: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 3600.0,
        abilities: Optional[List[str]] = None,
    ) -> None:
        self.model_name = model_name
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
            return await response.json()

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
            timeout_obj = aiohttp.ClientTimeout(
                total=min(REFERENCE_MEDIA_FETCH_TIMEOUT, timeout_value)
            )
            async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                async with session.get(media_text) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"Failed to fetch reference media: HTTP {response.status}"
                        )
                    return await response.read()

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

        effective_seconds = seconds if seconds is not None else duration
        if effective_seconds is not None and "seconds" not in kwargs:
            kwargs["seconds"] = effective_seconds
        if not size:
            size = _size_from_resolution_ratio(resolution, ratio)
        if size and "size" not in kwargs:
            kwargs["size"] = size

        reference_image = input_reference or image_url or image or first_frame_image_url

        try:
            timeout_obj = aiohttp.ClientTimeout(total=effective_timeout)
            async with aiohttp.ClientSession(timeout=timeout_obj) as session:
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
        except Exception as exc:
            logger.error("Xinference video generation failed: %s", exc)
            raise RuntimeError(f"Xinference video generation failed: {exc}") from exc

        data = _to_plain_data(result)
        item = _first_video_item(data)
        video_url = _video_url_from_item(item)
        if not video_url:
            raise RuntimeError(
                "Xinference video generation response did not include "
                "url, video_url, or b64_json"
            )
        return {
            "task_id": item.get("id") or item.get("task_id"),
            "model": self.model_name,
            "status": "succeeded",
            "video_url": video_url,
            "last_frame_url": item.get("last_frame_url"),
            "seed": item.get("seed"),
            "resolution": item.get("resolution") or resolution,
            "ratio": item.get("ratio") or ratio,
            "duration": item.get("duration")
            or item.get("seconds")
            or effective_seconds,
            "frames": item.get("frames"),
            "raw_response": data,
        }
