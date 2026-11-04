"""Music generation tool core."""

from __future__ import annotations

import logging
import uuid
from inspect import isawaitable
from typing import Any, Optional

from ...file_ref import build_workspace_file_ref
from ...model.music import BaseMusicModel, MusicResult
from ...workspace import TaskWorkspace

logger = logging.getLogger(__name__)


class MusicToolCore:
    """Generate music with configured music models."""

    GENERATE_MUSIC_DESCRIPTION = """
Generate music from a natural-language prompt.

Use this tool for songs, background music, jingles, cinematic scores, loops,
and instrumental tracks. Use generate_sound_effect for non-musical Foley,
ambience, impacts, or transitions, and synthesize_speech for spoken voice.

Available music models:
{}

Parameters:
- prompt (required): musical style, mood, instrumentation, structure, and
  optional vocal/lyric direction. Do not name artists or quote copyrighted lyrics.
- music_length_seconds (optional): duration from 3 to 600 seconds. Omit to let
  the provider choose.
- force_instrumental (optional): guarantee no vocals. Default: false.
- output_format (optional): provider output format. Default: auto.
- model_id (optional): configured music model ID. Omit to use the default.

The generated audio file is saved to the workspace and returned as
file_id/file_ref.
""".strip()

    def __init__(
        self,
        models: Optional[dict[str, BaseMusicModel]] = None,
        workspace: Optional[TaskWorkspace] = None,
        default_model: Optional[BaseMusicModel] = None,
    ) -> None:
        self._models = models or {}
        self._workspace = workspace
        self._default_model = default_model
        self._last_teardown_task_id: Optional[str] = None

    def model_info_text(self) -> str:
        if not self._models:
            return "No music models available"
        default_id = self._configured_model_id(self._default_model)
        lines = []
        for model_id, model in self._models.items():
            provider = str(getattr(model, "provider_name", "unknown"))
            marker = " ⭐[DEFAULT]" if model_id == default_id else ""
            lines.append(f"- {model_id} ({provider}){marker}")
        return "\n".join(lines)

    def _configured_model_id(self, model: Optional[BaseMusicModel]) -> Optional[str]:
        if model is None:
            return None
        for model_id, configured_model in self._models.items():
            if configured_model is model:
                return model_id
        declared_model_id = getattr(model, "model_id", None)
        if isinstance(declared_model_id, str) and declared_model_id in self._models:
            return declared_model_id
        return None

    def _get_model(
        self, model_id: Optional[str] = None
    ) -> tuple[Optional[BaseMusicModel], Optional[str]]:
        if model_id:
            model = self._models.get(model_id)
            return model, model_id if model is not None else None
        if self._default_model is not None:
            return self._default_model, self._configured_model_id(self._default_model)
        if self._models:
            first_id = next(iter(self._models))
            return self._models[first_id], first_id
        return None, None

    async def aclose(self) -> None:
        models = [self._default_model, *self._models.values()]
        seen: set[int] = set()
        for model in models:
            if model is None or id(model) in seen:
                continue
            seen.add(id(model))
            close = getattr(model, "aclose", None) or getattr(model, "close", None)
            if callable(close):
                result = close()
                if isawaitable(result):
                    await result

    async def teardown(self, task_id: Optional[str] = None) -> None:
        if task_id is not None and task_id == self._last_teardown_task_id:
            return
        await self.aclose()
        if task_id is not None:
            self._last_teardown_task_id = task_id

    async def generate_music(
        self,
        prompt: str,
        music_length_seconds: Optional[float] = None,
        force_instrumental: bool = False,
        output_format: str = "auto",
        model_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Generate music and save it to the task workspace."""
        try:
            model, configured_model_id = self._get_model(model_id)
            if model is None:
                error = (
                    f"Music model '{model_id}' is not configured"
                    if model_id
                    else "No music models configured"
                )
                return {
                    "success": False,
                    "error": error,
                    "audio_path": None,
                    "file_id": None,
                    "saved_to_workspace": False,
                }

            result = await model.generate_music(
                prompt=prompt,
                music_length_seconds=music_length_seconds,
                force_instrumental=force_instrumental,
                output_format=output_format,
            )
            if not isinstance(result, MusicResult):
                raise RuntimeError(f"Unexpected music response: {type(result)}")
            if not result.audio:
                raise RuntimeError("Music model returned no audio data")

            audio_path: Optional[str] = None
            file_id: Optional[str] = None
            file_ref: Optional[dict[str, Any]] = None
            if self._workspace:
                filename = f"music_{uuid.uuid4().hex[:8]}.{result.format}"
                with self._workspace.auto_register_files():
                    save_path = self._workspace.output_dir / filename
                    save_path.write_bytes(result.audio)
                    audio_path = str(save_path)
                file_ref = build_workspace_file_ref(
                    workspace=self._workspace, file_path=audio_path
                )
                file_id = file_ref["file_id"]

            return {
                "success": True,
                "audio_path": audio_path,
                "file_id": file_id,
                "file_ref": file_ref,
                "format": result.format,
                "sample_rate": result.sample_rate,
                "music_length_seconds": music_length_seconds,
                "force_instrumental": force_instrumental,
                "model_used": configured_model_id,
                "provider_model": getattr(model, "model_name", None),
                "provider": getattr(model, "provider_name", "unknown"),
                "saved_to_workspace": audio_path is not None,
            }
        except Exception as exc:
            logger.error("Music generation failed: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "audio_path": None,
                "file_id": None,
                "model_used": model_id,
                "saved_to_workspace": False,
            }
