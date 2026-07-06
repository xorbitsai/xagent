"""
Video generation tool for xagent.

This module provides video generation capabilities using pre-configured video
models passed from the web layer.
"""

import asyncio
import base64
import logging
import math
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Dict, Optional, TypeAlias
from urllib import parse
from urllib.request import url2pathname

import aiohttp
from pydantic import Field

from ...file_ref import build_workspace_file_ref, guess_mime_type
from ...model.video.ark import ArkVideoModel
from ...model.video.base import BaseVideoModel
from ...model.video.xinference import XinferenceVideoModel
from ...workspace import TaskWorkspace

logger = logging.getLogger(__name__)

_CoercedOptionalString: TypeAlias = Annotated[
    Optional[str],
    Field(coerce_numbers_to_str=True),
]

_VIDEO_TOOL_ARG_KEYS = {
    "prompt",
    "seconds",
    "size",
    "input_reference",
    "image_url",
    "image",
    "negative_prompt",
    "n",
    "ratio",
    "duration",
    "resolution",
    "generate_audio",
    "watermark",
    "return_last_frame",
    "first_frame_image_url",
    "last_frame_image_url",
    "reference_image_urls",
    "reference_video_urls",
    "reference_audio_urls",
    "wait_for_result",
    "poll_interval",
    "timeout",
    "model_id",
}
_MODEL_GENERATED_TYPE_SUFFIXES = (
    "_str",
    "_int",
    "_float",
    "_number",
    "_bool",
)
_STANDARD_RATIOS = (
    ("16:9", 16 / 9),
    ("9:16", 9 / 16),
    ("21:9", 21 / 9),
    ("1:1", 1.0),
    ("4:3", 4 / 3),
    ("3:4", 3 / 4),
)
_RATIO_TOLERANCE = 0.02
_VIDEO_MIME_EXTENSIONS = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/mpeg": ".mpeg",
}


def _has_value(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def _canonical_video_tool_arg_key(key: str) -> str:
    clean_key = str(key).strip().rstrip(":").rstrip("_").strip()
    if clean_key in _VIDEO_TOOL_ARG_KEYS:
        return clean_key
    for suffix in _MODEL_GENERATED_TYPE_SUFFIXES:
        if clean_key.endswith(suffix):
            base_key = clean_key[: -len(suffix)]
            if base_key in _VIDEO_TOOL_ARG_KEYS:
                return base_key
    return clean_key


def _standard_ratio_from_dimensions(width: int, height: int) -> Optional[str]:
    if width <= 0 or height <= 0:
        return None
    actual_ratio = width / height
    for ratio, expected_ratio in _STANDARD_RATIOS:
        if math.isclose(
            actual_ratio,
            expected_ratio,
            rel_tol=_RATIO_TOLERANCE,
            abs_tol=_RATIO_TOLERANCE,
        ):
            return ratio
    return None


def _video_extension_from_data_url_header(header: str) -> str:
    mime = header.split(";", 1)[0].removeprefix("data:")
    return _VIDEO_MIME_EXTENSIONS.get(mime, ".mp4")


class VideoGenerationToolCore:
    """Video generation tool that uses pre-configured video models."""

    GENERATE_VIDEO_DESCRIPTION = """
Generate videos from text prompts and optional reference media.

When given a user request, rewrite and enrich the prompt into a concise professional video generation prompt:
- Describe the subject, action, camera motion, lighting, visual style, environment, and mood.
- Use concrete temporal language for motion and scene progression.
- Preserve user-specified text only when the user explicitly asks for visible text.
- Prefer the default model marked with ⭐[DEFAULT]. Only specify model_id if the user explicitly requests a different model.

Available models (⭐[DEFAULT] marks the configured default model):
{}

Parameters:
- prompt (required unless using raw content): optimized video prompt.
- seconds (optional): OpenAI-compatible requested video length in seconds. This is the main cost/latency driver.
- size (optional): OpenAI-compatible output size such as "1280x720", "720x1280", or "1920x1080".
- input_reference / image_url / image (optional): OpenAI/Xinference-style image reference for image-to-video.
- negative_prompt (optional): negative prompt for providers that support it, especially Xinference video models.
- n (optional): number of videos to request when supported by the provider. Defaults to 1.
- ratio (optional): aspect ratio alias such as "16:9", "9:16", "1:1", "4:3", or "3:4".
- duration (optional): legacy alias for seconds.
- resolution (optional): provider alias such as "480p", "720p", or "1080p"; converted with ratio when size is omitted.
- generate_audio (optional): whether to generate synchronized audio. Defaults to true when supported.
- watermark (optional): whether to include provider watermark. Defaults to true when supported.
- return_last_frame (optional): ask the provider to return the last frame URL when supported.
- first_frame_image_url / last_frame_image_url (optional): Seedance-style first/last-frame control.
- reference_image_urls / reference_video_urls / reference_audio_urls (optional): public URLs used as references. Audio references require at least one image or video reference.
- wait_for_result (optional): wait for task completion and download the result. Defaults to true.
- poll_interval (optional): polling interval in seconds while waiting. Defaults to 30.
- timeout (optional): maximum wait time in seconds.
- model_id (optional): model ID from the list above. Omit to use the default model marked with ⭐[DEFAULT].

Provider notes:
- For Volcengine/BytePlus Ark Seedance models, prefer resolution + ratio instead of arbitrary size. Valid ratios are "16:9", "4:3", "1:1", "3:4", "9:16", "21:9", and "adaptive".
- Ark Seedance duration accepts whole seconds only. Seedance 1.5 Pro supports 4-12 seconds or -1 for intelligent duration; Seedance 2.0 supports 4-15 seconds or -1. If the user gives a range such as 4-5 seconds, choose one valid integer seconds value before calling the tool.
- Ark resolutions include "480p", "720p", "1080p", and "4k" depending on the model. Seedance 2.0 Fast and Mini do not support 1080p, and 4k is only for Seedance 2.0.
- Ark image references may be http(s) URLs, data URLs, asset:// IDs, or workspace file references such as file:file_id. Workspace image references are converted to base64 data URLs automatically.
- Xinference/OpenAI-compatible video models can use size and seconds directly when those parameters are supported by the model.

The generated video URL is temporary on the provider side, so completed videos are automatically downloaded and saved to the workspace.
    """.strip()

    def __init__(
        self,
        video_models: Dict[str, BaseVideoModel],
        model_descriptions: Optional[Dict[str, str]] = None,
        workspace: Optional[TaskWorkspace] = None,
        default_video_model: Optional[BaseVideoModel] = None,
    ):
        self._video_models = video_models
        self._model_descriptions = model_descriptions or {}
        self._workspace = workspace
        self._default_video_model = default_video_model
        self._generate_model_info_text()

    def _generate_model_info_text(self) -> None:
        if not self._video_models:
            self._model_info_text = "No video models available"
            return

        default_video_id = None
        if self._default_video_model:
            default_video_id = getattr(
                self._default_video_model, "model_id", None
            ) or getattr(self._default_video_model, "model_name", None)

        default_model_lines = []
        other_model_lines = []
        for model_id, model in self._video_models.items():
            if hasattr(model, "has_ability") and not model.has_ability("generate"):
                continue
            description = self._model_descriptions.get(model_id, "")
            is_default = model_id == default_video_id
            default_marker = " ⭐[DEFAULT]" if is_default else ""

            if description:
                line = f"- {model_id}: {description}{default_marker}"
            else:
                line = f"- {model_id}: No description available{default_marker}"

            if is_default:
                default_model_lines.append(line)
            else:
                other_model_lines.append(line)

        model_lines = default_model_lines + other_model_lines
        self._model_info_text = (
            "\n".join(model_lines)
            if model_lines
            else "No video models with generate capabilities available"
        )

    def _get_model(self, model_id: Optional[str] = None) -> Optional[BaseVideoModel]:
        if model_id and model_id in self._video_models:
            model = self._video_models[model_id]
            if hasattr(model, "has_ability") and model.has_ability("generate"):
                return model
            logger.warning("Video model %s does not support generation", model_id)
            return None
        if model_id:
            logger.warning("Video model %s is not configured", model_id)
            return None

        if self._default_video_model:
            return self._default_video_model

        for model in self._video_models.values():
            if hasattr(model, "has_ability") and model.has_ability("generate"):
                return model

        return None

    def _available_generate_model_ids(self) -> list[str]:
        return [
            model_id
            for model_id, model in self._video_models.items()
            if not hasattr(model, "has_ability") or model.has_ability("generate")
        ]

    def _model_id_for_model(self, selected_model: BaseVideoModel) -> str:
        selected_model_id = getattr(selected_model, "model_id", None)
        if selected_model_id:
            return str(selected_model_id)

        selected_inner = getattr(selected_model, "_inner", selected_model)
        for model_id, model in self._video_models.items():
            if model is selected_model:
                return model_id
        return (
            str(getattr(selected_inner, "model_id", "") or "")
            or str(getattr(selected_inner, "model_name", "") or "")
            or "default"
        )

    def _missing_model_error(self, model_id: Optional[str]) -> str:
        if not model_id:
            return "No available video models configured"

        available = self._available_generate_model_ids()
        available_text = ", ".join(available) if available else "none"
        if model_id in self._video_models:
            return (
                f"Video model '{model_id}' does not support video generation; "
                f"available video generation models: {available_text}"
            )
        return (
            f"Video model '{model_id}' is not configured; "
            f"available video generation models: {available_text}"
        )

    @staticmethod
    def _normalize_url_list(value: Optional[str | list[str]]) -> Optional[list[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else None
        urls = [str(item).strip() for item in value if str(item).strip()]
        return urls or None

    @staticmethod
    def _normalize_seconds(
        value: Optional[int | float | str],
    ) -> Optional[int | float | str]:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                numeric = float(stripped)
            except ValueError:
                return stripped
            return int(numeric) if numeric.is_integer() else numeric
        if isinstance(value, float):
            return int(value) if value.is_integer() else value
        return int(value)

    @staticmethod
    def _normalize_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0", "no", "off"}
        return bool(value)

    @staticmethod
    def _normalize_extra_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        return VideoGenerationToolCore.normalize_raw_tool_args(kwargs)

    @staticmethod
    def normalize_raw_tool_args(args: Mapping[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in args.items():
            clean_key = _canonical_video_tool_arg_key(str(key))
            if clean_key in normalized and _has_value(normalized[clean_key]):
                continue
            normalized[clean_key] = value
        return normalized

    @staticmethod
    def _parse_size(size: Optional[str]) -> Optional[tuple[int, int]]:
        if not size:
            return None
        normalized = size.lower().replace("*", "x").strip()
        if "x" not in normalized:
            return None
        width_text, height_text = normalized.split("x", 1)
        try:
            width = int(width_text.strip())
            height = int(height_text.strip())
        except ValueError:
            return None
        if width <= 0 or height <= 0:
            return None
        return width, height

    @staticmethod
    def _ratio_from_size(size: str) -> Optional[str]:
        parsed = VideoGenerationToolCore._parse_size(size)
        if not parsed:
            return None
        width, height = parsed
        standard_ratio = _standard_ratio_from_dimensions(width, height)
        if standard_ratio:
            return standard_ratio
        divisor = math.gcd(width, height)
        return f"{width // divisor}:{height // divisor}"

    @staticmethod
    def _resolution_from_size(size: str) -> Optional[str]:
        parsed = VideoGenerationToolCore._parse_size(size)
        if not parsed:
            return None
        width, height = parsed
        return f"{min(width, height)}p"

    @staticmethod
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
        width_ratio_text, height_ratio_text = ratio_text.split(":", 1)
        try:
            width_ratio = int(width_ratio_text)
            height_ratio = int(height_ratio_text)
        except ValueError:
            return None
        if width_ratio <= 0 or height_ratio <= 0:
            return None

        if width_ratio >= height_ratio:
            height = short_side
            width = int(round(short_side * width_ratio / height_ratio))
        else:
            width = short_side
            height = int(round(short_side * height_ratio / width_ratio))
        return f"{width}x{height}"

    def _build_video_artifacts(
        self, video_path: Optional[str], file_id: Optional[str]
    ) -> list[dict[str, str]]:
        if not file_id:
            return []

        filename = Path(video_path).name if video_path else "generated_video.mp4"
        return [
            {
                "type": "video",
                "file_id": file_id,
                "filename": filename,
                "mime_type": guess_mime_type(filename),
                "display": "inline",
            }
        ]

    def _resolve_workspace_local_path(self, path: Path) -> Path:
        if not self._workspace:
            raise ValueError("No workspace available for local file access")

        workspace_root = self._workspace.workspace_dir.resolve()
        expanded_path = path.expanduser()
        if expanded_path.is_absolute():
            resolved_path = expanded_path.resolve()
        else:
            resolved_path = (workspace_root / expanded_path).resolve()
        if resolved_path == workspace_root or resolved_path.is_relative_to(
            workspace_root
        ):
            return resolved_path

        raise ValueError("Access denied: local path is outside the workspace")

    async def _ark_image_ref_to_data_url(self, image_ref: str) -> str:
        image_ref = str(image_ref).strip()
        parsed_ref = parse.urlparse(image_ref)
        if parsed_ref.scheme in {"http", "https", "data", "asset"}:
            return image_ref

        if not self._workspace:
            raise ValueError(
                "Ark image-to-video local references require a workspace so they can "
                "be converted to base64 data URLs"
            )

        if parsed_ref.scheme and not image_ref.startswith("file:"):
            raise ValueError(
                "Ark image references must be http(s), data URLs, asset:// IDs, "
                f"or workspace file references; got scheme '{parsed_ref.scheme}'"
            )

        ref_for_resolution = image_ref
        if image_ref.startswith("file://"):
            ref_for_resolution = url2pathname(parsed_ref.path)

        resolved_path = await asyncio.to_thread(
            self._workspace.resolve_path_with_search,
            ref_for_resolution,
        )
        if not await asyncio.to_thread(resolved_path.is_file):
            raise ValueError(f"Ark image reference is not a file: {image_ref}")

        mime_type = guess_mime_type(str(resolved_path))
        if not mime_type.startswith("image/"):
            raise ValueError(
                f"Ark image reference must resolve to an image file: {image_ref}"
            )

        image_bytes = await asyncio.to_thread(resolved_path.read_bytes)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    async def _prepare_ark_image_references(
        self, generate_params: dict[str, Any]
    ) -> None:
        converted_refs: dict[str, str] = {}

        async def convert_ref(value: Any) -> str:
            ref = str(value).strip()
            if ref not in converted_refs:
                converted_refs[ref] = await self._ark_image_ref_to_data_url(ref)
            return converted_refs[ref]

        for key in ("input_reference", "first_frame_image_url", "last_frame_image_url"):
            if _has_value(generate_params.get(key)):
                generate_params[key] = await convert_ref(generate_params[key])

        reference_image_urls = generate_params.get("reference_image_urls")
        if reference_image_urls:
            generate_params["reference_image_urls"] = [
                await convert_ref(url) for url in reference_image_urls
            ]

    async def _download_video(
        self, video_url: str, filename: Optional[str] = None, timeout: int = 3600
    ) -> str:
        if not self._workspace:
            raise ValueError("No workspace available for saving videos")

        try:
            if video_url.startswith("data:"):
                header, _, encoded = video_url.partition(",")
                if ";base64" not in header or not encoded:
                    raise RuntimeError("Unsupported video data URL")
                if not filename:
                    extension = _video_extension_from_data_url_header(header)
                    filename = f"generated_video_{uuid.uuid4().hex[:8]}{extension}"
                elif "." not in filename:
                    filename = (
                        f"{filename}{_video_extension_from_data_url_header(header)}"
                    )
                filename = "".join(
                    c for c in filename if c.isalnum() or c in ("-", "_", ".")
                )
                save_path = self._workspace.output_dir / filename
                await asyncio.to_thread(
                    save_path.write_bytes, base64.b64decode(encoded)
                )
                logger.info("Saved base64 video to: %s", save_path)
                return str(save_path)

            if not filename:
                url_path = parse.urlparse(video_url).path
                extension = os.path.splitext(url_path)[1] or ".mp4"
                filename = f"generated_video_{uuid.uuid4().hex[:8]}{extension}"
            filename = "".join(
                c for c in filename if c.isalnum() or c in ("-", "_", ".")
            )
            save_path = self._workspace.output_dir / filename

            local_path: Optional[Path] = None
            parsed_video_url = parse.urlparse(video_url)
            if parsed_video_url.scheme == "file":
                local_path = Path(url2pathname(parsed_video_url.path))
            elif not parsed_video_url.scheme:
                local_path = Path(video_url)

            if local_path is not None:
                resolved_local_path = await asyncio.to_thread(
                    self._resolve_workspace_local_path, local_path
                )
                if not await asyncio.to_thread(resolved_local_path.is_file):
                    raise RuntimeError(
                        f"Local video path is not a file: {resolved_local_path}"
                    )
                resolved_save_path = await asyncio.to_thread(save_path.resolve)
                if resolved_local_path != resolved_save_path:
                    await asyncio.to_thread(
                        shutil.copyfile, resolved_local_path, save_path
                    )
                logger.info("Copied local video to: %s", save_path)
                return str(save_path)

            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                async with session.get(video_url) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"Failed to download video: HTTP {response.status}"
                        )

                    with open(save_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(1024 * 1024):
                            await asyncio.to_thread(f.write, chunk)

            logger.info("Downloaded video to: %s", save_path)
            return str(save_path)

        except Exception as e:
            logger.warning("Failed to download video from %s: %s", video_url, e)
            raise

    async def generate_video(
        self,
        prompt: str,
        seconds: _CoercedOptionalString = None,
        size: Optional[str] = None,
        input_reference: Optional[str] = None,
        image_url: Optional[str] = None,
        image: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        n: int = 1,
        ratio: Optional[str] = None,
        duration: _CoercedOptionalString = None,
        resolution: Optional[str] = None,
        generate_audio: bool = True,
        watermark: bool = True,
        return_last_frame: bool = False,
        first_frame_image_url: Optional[str] = None,
        last_frame_image_url: Optional[str] = None,
        reference_image_urls: Optional[str | list[str]] = None,
        reference_video_urls: Optional[str | list[str]] = None,
        reference_audio_urls: Optional[str | list[str]] = None,
        wait_for_result: bool = True,
        poll_interval: float = 30.0,
        timeout: Optional[float] = None,
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        actual_model_id = model_id or "default"
        try:
            kwargs = self._normalize_extra_kwargs(kwargs)
            kwargs.pop("allowed_local_media_roots", None)
            if not _has_value(ratio):
                ratio = None
            explicit_ratio = ratio is not None
            if "seconds" in kwargs:
                seconds = kwargs.pop("seconds")
            if "duration" in kwargs:
                duration = kwargs.pop("duration")
            if "size" in kwargs:
                size = kwargs.pop("size")
            if "ratio" in kwargs:
                ratio_value = kwargs.pop("ratio")
                ratio = str(ratio_value) if _has_value(ratio_value) else None
                explicit_ratio = ratio is not None
            if "resolution" in kwargs:
                resolution = str(kwargs.pop("resolution"))
            if "generate_audio" in kwargs:
                generate_audio = self._normalize_bool(kwargs.pop("generate_audio"))
            if "watermark" in kwargs:
                watermark = self._normalize_bool(kwargs.pop("watermark"))
            if "return_last_frame" in kwargs:
                return_last_frame = self._normalize_bool(
                    kwargs.pop("return_last_frame")
                )
            if "input_reference" in kwargs:
                input_reference = str(kwargs.pop("input_reference"))
            if "image_url" in kwargs:
                image_url = str(kwargs.pop("image_url"))
            if "image" in kwargs:
                image = str(kwargs.pop("image"))
            if "negative_prompt" in kwargs:
                negative_prompt = str(kwargs.pop("negative_prompt"))
            if "n" in kwargs:
                n = int(kwargs.pop("n"))

            video_model = self._get_model(model_id)
            if not video_model:
                return {
                    "success": False,
                    "error": self._missing_model_error(model_id),
                    "video_path": None,
                    "model_used": actual_model_id,
                }
            actual_model_id = self._model_id_for_model(video_model)
            effective_ratio = ratio or "16:9"

            generate_params: dict[str, Any] = {
                "prompt": prompt,
                "n": n,
                "ratio": effective_ratio,
                "generate_audio": generate_audio,
                "watermark": watermark,
                "return_last_frame": return_last_frame,
                "wait_for_result": wait_for_result,
                "poll_interval": poll_interval,
            }
            effective_seconds = self._normalize_seconds(
                seconds if seconds is not None else duration
            )
            if effective_seconds is not None:
                generate_params["seconds"] = effective_seconds
                generate_params["duration"] = effective_seconds
            effective_size = size or self._size_from_resolution_ratio(
                resolution, effective_ratio
            )
            if effective_size:
                generate_params["size"] = effective_size
                derived_ratio = self._ratio_from_size(effective_size)
                derived_resolution = self._resolution_from_size(effective_size)
                if derived_ratio and not explicit_ratio:
                    generate_params["ratio"] = derived_ratio
                if resolution is None and derived_resolution:
                    generate_params["resolution"] = derived_resolution
            if resolution is not None:
                generate_params["resolution"] = resolution
            if negative_prompt is not None:
                generate_params["negative_prompt"] = negative_prompt
            if timeout is not None:
                generate_params["timeout"] = timeout
            reference_input = input_reference or image_url or image
            if reference_input:
                generate_params["input_reference"] = reference_input
                if not first_frame_image_url:
                    first_frame_image_url = reference_input
            if first_frame_image_url:
                generate_params["first_frame_image_url"] = first_frame_image_url
            if last_frame_image_url:
                generate_params["last_frame_image_url"] = last_frame_image_url

            normalized_reference_image_urls = self._normalize_url_list(
                reference_image_urls
            )
            normalized_reference_video_urls = self._normalize_url_list(
                reference_video_urls
            )
            normalized_reference_audio_urls = self._normalize_url_list(
                reference_audio_urls
            )
            if normalized_reference_image_urls:
                generate_params["reference_image_urls"] = (
                    normalized_reference_image_urls
                )
            if normalized_reference_video_urls:
                generate_params["reference_video_urls"] = (
                    normalized_reference_video_urls
                )
            if normalized_reference_audio_urls:
                generate_params["reference_audio_urls"] = (
                    normalized_reference_audio_urls
                )

            generate_params.update(kwargs)
            inner_video_model = getattr(video_model, "_inner", video_model)
            if isinstance(inner_video_model, ArkVideoModel):
                await self._prepare_ark_image_references(generate_params)
            if self._workspace and isinstance(inner_video_model, XinferenceVideoModel):
                generate_params["allowed_local_media_roots"] = [
                    self._workspace.workspace_dir
                ]

            result = await video_model.generate_video(**generate_params)

            video_url = result.get("video_url")
            video_path = None
            video_file_id: Optional[str] = None
            file_ref: Optional[dict[str, Any]] = None

            if video_url and self._workspace:
                try:
                    with self._workspace.auto_register_files():
                        video_path = await self._download_video(video_url)
                    if video_path:
                        try:
                            file_ref = build_workspace_file_ref(
                                workspace=self._workspace,
                                file_path=video_path,
                            )
                            video_file_id = file_ref["file_id"]
                        except Exception as e:
                            logger.warning(
                                "Failed to build generated video FileRef: %s", e
                            )
                            video_file_id = self._workspace.get_file_id_from_path(
                                video_path
                            )
                except Exception as e:
                    logger.warning("Failed to download video to workspace: %s", e)
            elif video_url and not self._workspace:
                logger.warning("No workspace available, video not saved locally")

            return {
                "success": True,
                "task_id": result.get("task_id"),
                "status": result.get("status"),
                "video_url": video_url,
                "last_frame_url": result.get("last_frame_url"),
                "video_path": video_path,
                "file_id": video_file_id,
                "artifacts": self._build_video_artifacts(video_path, video_file_id),
                "file_ref": file_ref,
                "request_id": result.get("request_id"),
                "seed": result.get("seed"),
                "resolution": result.get("resolution"),
                "ratio": result.get("ratio"),
                "duration": result.get("duration"),
                "frames": result.get("frames"),
                "model_used": actual_model_id,
                "saved_to_workspace": video_path is not None,
                "raw_response": result.get("raw_response"),
            }

        except Exception as e:
            logger.error("Video generation failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "video_path": None,
                "model_used": actual_model_id,
            }

    def list_available_models(self) -> Dict[str, Any]:
        try:
            models_info = []
            for model_id, model in self._video_models.items():
                models_info.append(
                    {
                        "model_id": model_id,
                        "available": True,
                        "abilities": getattr(model, "abilities", []),
                        "description": self._model_descriptions.get(model_id, ""),
                    }
                )

            return {
                "success": True,
                "models": models_info,
                "count": len(models_info),
            }

        except Exception as e:
            logger.error("Failed to list available video models: %s", e)
            return {
                "success": False,
                "error": str(e),
                "models": [],
                "count": 0,
            }
