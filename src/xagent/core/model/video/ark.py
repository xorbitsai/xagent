import asyncio
import importlib
import logging
import math
import os
import time
from typing import Any, List, Optional

from .base import BaseVideoModel

ARK_DOMESTIC_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_BYTEPLUS_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
_STANDARD_RATIOS = (
    ("16:9", 16 / 9),
    ("9:16", 9 / 16),
    ("21:9", 21 / 9),
    ("1:1", 1.0),
    ("4:3", 4 / 3),
    ("3:4", 3 / 4),
)
_RATIO_TOLERANCE = 0.02
logger = logging.getLogger(__name__)


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


def _get_nested(value: Any, *path: str) -> Any:
    current = value
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def _size_to_seedance_params(
    size: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    if not size:
        return None, None
    normalized = size.lower().replace("*", "x").strip()
    if "x" not in normalized:
        return None, None
    width_text, height_text = normalized.split("x", 1)
    try:
        width = int(width_text.strip())
        height = int(height_text.strip())
    except ValueError:
        return None, None
    if width <= 0 or height <= 0:
        return None, None
    actual_ratio = width / height
    ratio = None
    for standard_ratio, expected_ratio in _STANDARD_RATIOS:
        if math.isclose(
            actual_ratio,
            expected_ratio,
            rel_tol=_RATIO_TOLERANCE,
            abs_tol=_RATIO_TOLERANCE,
        ):
            ratio = standard_ratio
            break
    if ratio is None:
        divisor = math.gcd(width, height)
        ratio = f"{width // divisor}:{height // divisor}"
    resolution = f"{min(width, height)}p"
    return ratio, resolution


def _first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


class ArkVideoModel(BaseVideoModel):
    """Video generation client for Volcengine/BytePlus ModelArk Seedance models."""

    def __init__(
        self,
        model_name: str = "doubao-seedance-2-0-fast-260128",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 1800.0,
        abilities: Optional[List[str]] = None,
        model_provider: str = "volcengine-ark",
    ):
        self.model_name = model_name
        self.model_provider = model_provider.lower().strip().replace("_", "-")
        is_byteplus = self.model_provider in {
            "byteplus",
            "byteplus-ark",
        } or model_name.startswith("dreamina-")
        if is_byteplus:
            self.api_key = api_key or _first_env(
                "BYTEPLUS_ARK_API_KEY",
                "BYTEPLUS_API_KEY",
                "ARK_API_KEY",
            )
            self.base_url = (
                base_url
                or _first_env("BYTEPLUS_ARK_BASE_URL", "BYTEPLUS_BASE_URL")
                or ARK_BYTEPLUS_BASE_URL
            ).rstrip("/")
        else:
            self.api_key = api_key or _first_env(
                "VOLCENGINE_ARK_API_KEY", "ARK_API_KEY"
            )
            self.base_url = (
                base_url
                or _first_env("VOLCENGINE_ARK_BASE_URL", "ARK_BASE_URL")
                or os.getenv("MODELARK_BASE_URL")
                or ARK_DOMESTIC_BASE_URL
            ).rstrip("/")
        self.timeout = timeout
        self._abilities = abilities or ["generate"]
        self._client: Any = None

    @property
    def abilities(self) -> List[str]:
        return self._abilities

    @property
    def _uses_byteplus_sdk(self) -> bool:
        return (
            self.model_provider in {"byteplus", "byteplus-ark"}
            or "bytepluses.com" in self.base_url
            or self.model_name.startswith("dreamina-")
        )

    def _load_ark_class(self) -> Any:
        if self._uses_byteplus_sdk:
            try:
                return importlib.import_module("byteplussdkarkruntime").Ark
            except ImportError as exc:
                raise RuntimeError(
                    "BytePlus ModelArk SDK is required for this base_url/model. "
                    "Install byteplus-python-sdk-v2."
                ) from exc

        try:
            return importlib.import_module("volcenginesdkarkruntime").Ark
        except ImportError as exc:
            raise RuntimeError(
                "Volcengine Ark SDK is required for this base_url/model. "
                "Install volcengine-python-sdk[ark]."
            ) from exc

    def validate_configuration(self) -> None:
        if not self.api_key:
            raise ValueError("ModelArk API key cannot be empty")
        if not self.base_url:
            raise ValueError("Ark base_url cannot be empty")
        self._load_ark_class()

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        self.validate_configuration()
        Ark = self._load_ark_class()
        self._client = Ark(base_url=self.base_url, api_key=self.api_key)

    def _client_create(self, content: list[dict[str, Any]], **kwargs: Any) -> Any:
        self._ensure_client()
        return self._client.content_generation.tasks.create(
            model=self.model_name,
            content=content,
            **kwargs,
        )

    def _client_get(self, task_id: str) -> Any:
        self._ensure_client()
        return self._client.content_generation.tasks.get(
            task_id=task_id,
        )

    async def create_video_task(
        self,
        content: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self.has_ability("generate"):
            raise RuntimeError("This model doesn't support video generation")
        if not content:
            raise ValueError("content must include at least one text or media item")

        response = await asyncio.to_thread(self._client_create, content, **kwargs)
        data = _to_plain_data(response)
        return {
            "task_id": _get_nested(data, "id") or getattr(response, "id", None),
            "raw_response": data,
        }

    async def get_video_task(self, task_id: str) -> dict[str, Any]:
        if not task_id:
            raise ValueError("task_id cannot be empty")

        response = await asyncio.to_thread(self._client_get, task_id)
        data = _to_plain_data(response)
        content = _get_nested(data, "content") or {}
        error = _get_nested(data, "error")
        return {
            "task_id": _get_nested(data, "id") or task_id,
            "model": _get_nested(data, "model"),
            "status": _get_nested(data, "status"),
            "error": error,
            "video_url": _get_nested(content, "video_url"),
            "last_frame_url": _get_nested(content, "last_frame_url"),
            "seed": _get_nested(data, "seed"),
            "resolution": _get_nested(data, "resolution"),
            "ratio": _get_nested(data, "ratio"),
            "duration": _get_nested(data, "duration"),
            "frames": _get_nested(data, "frames"),
            "raw_response": data,
        }

    def _build_content(
        self,
        prompt: str,
        reference_image_urls: Optional[list[str]] = None,
        reference_video_urls: Optional[list[str]] = None,
        reference_audio_urls: Optional[list[str]] = None,
        first_frame_image_url: Optional[str] = None,
        last_frame_image_url: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if prompt:
            content.append({"type": "text", "text": prompt})

        if first_frame_image_url:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": first_frame_image_url},
                    "role": "first_frame",
                }
            )
        if last_frame_image_url:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": last_frame_image_url},
                    "role": "last_frame",
                }
            )

        for url in reference_image_urls or []:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                    "role": "reference_image",
                }
            )
        for url in reference_video_urls or []:
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": url},
                    "role": "reference_video",
                }
            )
        for url in reference_audio_urls or []:
            content.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": url},
                    "role": "reference_audio",
                }
            )
        return content

    async def generate_video(
        self,
        prompt: str = "",
        content: Optional[list[dict[str, Any]]] = None,
        wait_for_result: bool = True,
        poll_interval: float = 30.0,
        timeout: Optional[float] = None,
        reference_image_urls: Optional[list[str]] = None,
        reference_video_urls: Optional[list[str]] = None,
        reference_audio_urls: Optional[list[str]] = None,
        first_frame_image_url: Optional[str] = None,
        last_frame_image_url: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        seconds = kwargs.pop("seconds", None)
        duration = kwargs.pop("duration", None)
        size = kwargs.pop("size", None)
        input_reference = (
            kwargs.pop("input_reference", None)
            or kwargs.pop("image_url", None)
            or kwargs.pop("image", None)
        )
        if kwargs.pop("negative_prompt", None):
            logger.debug("Ark video generation does not support negative_prompt")
        n = kwargs.pop("n", 1)
        if n not in (None, 1, "1"):
            raise ValueError("Seedance video generation supports one video per request")
        if duration is not None:
            kwargs["duration"] = duration
        elif seconds is not None:
            kwargs["duration"] = seconds
        if size:
            ratio, resolution = _size_to_seedance_params(str(size))
            if ratio and "ratio" not in kwargs:
                kwargs["ratio"] = ratio
            if resolution and "resolution" not in kwargs:
                kwargs["resolution"] = resolution
        if input_reference and not first_frame_image_url:
            first_frame_image_url = str(input_reference)

        request_content = content or self._build_content(
            prompt=prompt,
            reference_image_urls=reference_image_urls,
            reference_video_urls=reference_video_urls,
            reference_audio_urls=reference_audio_urls,
            first_frame_image_url=first_frame_image_url,
            last_frame_image_url=last_frame_image_url,
        )
        if reference_audio_urls and not (
            reference_image_urls
            or reference_video_urls
            or first_frame_image_url
            or last_frame_image_url
        ):
            raise ValueError(
                "Seedance audio references require at least one image or video reference"
            )

        create_result = await self.create_video_task(request_content, **kwargs)
        task_id = create_result.get("task_id")
        if not wait_for_result:
            return {
                **create_result,
                "status": "created",
                "video_url": None,
                "last_frame_url": None,
            }
        if not task_id:
            raise RuntimeError("Ark did not return a video generation task id")

        deadline = time.monotonic() + (timeout or self.timeout)
        while True:
            try:
                task_result = await self.get_video_task(str(task_id))
            except Exception as exc:
                logger.warning(
                    "Transient error polling video task %s: %s", task_id, exc
                )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for video generation task {task_id} "
                        "due to persistent polling errors"
                    ) from exc
                await asyncio.sleep(max(1.0, poll_interval))
                continue
            status = task_result.get("status")
            if status == "succeeded":
                return task_result
            if status in {"failed", "cancelled", "expired"}:
                raise RuntimeError(
                    f"Video generation task {task_id} ended with status {status}: "
                    f"{task_result.get('error')}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for video generation task {task_id}; "
                    f"last status: {status}"
                )
            await asyncio.sleep(max(1.0, poll_interval))
