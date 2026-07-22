"""Download an exact web image asset into the current task workspace."""

from __future__ import annotations

import io
import mimetypes
from pathlib import Path
from typing import Any, Mapping, Type
from urllib.parse import unquote, urljoin, urlsplit

import httpx
from PIL import Image
from pydantic import BaseModel, Field

from ....file_ref import build_workspace_file_ref
from ....utils.security import validate_public_http_url
from ....workspace import TaskWorkspace
from ...core.web_content import (
    DEFAULT_MAX_CONTENT_BYTES,
    DEFAULT_USER_AGENT,
    get_proxy_url,
)
from .base import AbstractBaseTool, ToolCategory, ToolVisibility


class DownloadWebAssetArgs(BaseModel):
    url: str = Field(
        description=(
            "Exact HTTP or HTTPS image URL discovered on an authoritative page. "
            "Prefer the official brand domain for logos."
        )
    )
    filename: str | None = Field(
        default=None,
        description=(
            "Optional output basename. The remote filename and MIME type are used "
            "when omitted."
        ),
    )


class DownloadWebAssetResult(BaseModel):
    success: bool
    source_url: str
    content_type: str = ""
    size: int | None = None
    filename: str | None = None
    file_id: str | None = None
    file_ref: dict[str, Any] | None = None
    markdown_link: str | None = None
    error: str | None = None


class DownloadWebAssetTool(AbstractBaseTool):
    """Download and register an exact remote image without model reconstruction."""

    category = ToolCategory.WEB_SEARCH
    _MAX_REDIRECTS = 5

    def __init__(
        self,
        workspace: TaskWorkspace,
        *,
        max_content_bytes: int = DEFAULT_MAX_CONTENT_BYTES,
    ) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self._workspace = workspace
        self._max_content_bytes = max_content_bytes

    @property
    def name(self) -> str:
        return "download_web_asset"

    @property
    def description(self) -> str:
        return (
            "Download an exact PNG, JPEG, WebP, GIF, BMP, or SVG asset from a "
            "known URL and register it in the current workspace. Use this after "
            "fetch_web_content(include_assets=true) discovers an official logo or "
            "brand image. It returns a trusted FileRef, so do not reproduce the "
            "asset through api_call plus write_file."
        )

    @property
    def tags(self) -> list[str]:
        return ["web", "image", "asset", "download", "logo"]

    def args_type(self) -> Type[BaseModel]:
        return DownloadWebAssetArgs

    def return_type(self) -> Type[BaseModel]:
        return DownloadWebAssetResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("DownloadWebAssetTool only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        download_args = DownloadWebAssetArgs.model_validate(args)
        try:
            content, final_url, content_type = await self._download(download_args.url)
            detected_extension = self._validate_image(content, content_type, final_url)
            filename = self._resolve_filename(
                download_args.filename,
                final_url,
                content_type,
                detected_extension,
            )
            with self._workspace.auto_register_files():
                target = self._write_unique_output(filename, content)
            file_ref = build_workspace_file_ref(
                workspace=self._workspace,
                file_path=target,
                mime_type=(
                    self._media_type(content_type)
                    or mimetypes.guess_type(target.name)[0]
                    or ""
                ),
            )
            return DownloadWebAssetResult(
                success=True,
                source_url=final_url,
                content_type=content_type,
                size=len(content),
                filename=target.name,
                file_id=file_ref["file_id"],
                file_ref=file_ref,
                markdown_link=file_ref["markdown_link"],
            ).model_dump()
        except Exception as exc:
            return DownloadWebAssetResult(
                success=False,
                source_url=download_args.url,
                error=str(exc),
            ).model_dump()

    async def _download(self, url: str) -> tuple[bytes, str, str]:
        client_kwargs: dict[str, Any] = {}
        proxy_url = get_proxy_url()
        if proxy_url:
            client_kwargs["proxy"] = proxy_url

        current_url = url
        async with httpx.AsyncClient(**client_kwargs) as client:
            for redirect_count in range(self._MAX_REDIRECTS + 1):
                await validate_public_http_url(current_url)
                async with client.stream(
                    "GET",
                    current_url,
                    headers={"User-Agent": DEFAULT_USER_AGENT},
                    timeout=30,
                    follow_redirects=False,
                ) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("Remote asset redirect has no Location")
                        if redirect_count >= self._MAX_REDIRECTS:
                            raise ValueError("Remote asset exceeded redirect limit")
                        current_url = urljoin(current_url, location)
                        continue

                    response.raise_for_status()
                    declared_length = response.headers.get("content-length")
                    if declared_length is not None:
                        try:
                            length = int(declared_length)
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                f"Invalid declared content length: {declared_length}"
                            ) from exc
                        if length < 0:
                            raise ValueError(
                                f"Invalid declared content length: {declared_length}"
                            )
                        if length > self._max_content_bytes:
                            raise ValueError(
                                "Remote asset exceeds maximum size of "
                                f"{self._max_content_bytes} bytes"
                            )

                    chunks: list[bytes] = []
                    downloaded = 0
                    async for chunk in response.aiter_bytes():
                        downloaded += len(chunk)
                        if downloaded > self._max_content_bytes:
                            raise ValueError(
                                "Remote asset exceeds maximum size of "
                                f"{self._max_content_bytes} bytes"
                            )
                        chunks.append(chunk)
                    content = b"".join(chunks)
                    if not content:
                        raise ValueError("Remote asset response was empty")
                    return (
                        content,
                        str(response.url),
                        response.headers.get("content-type", ""),
                    )

        raise ValueError("Remote asset exceeded redirect limit")

    @classmethod
    def _validate_image(
        cls,
        content: bytes,
        content_type: str,
        url: str,
    ) -> str:
        media_type = cls._media_type(content_type)
        suffix = Path(urlsplit(url).path).suffix.lower()
        if media_type and not media_type.startswith("image/"):
            raise ValueError(f"Unsupported web asset content type: {content_type}")
        if media_type == "image/svg+xml" or suffix == ".svg":
            prefix = content[:4096].decode("utf-8", errors="ignore").lower()
            if "<svg" not in prefix:
                raise ValueError("Remote SVG asset does not contain an <svg> root")
            return ".svg"
        try:
            with Image.open(io.BytesIO(content)) as image:
                image_format = image.format
                image.verify()
        except Exception as exc:
            raise ValueError("Remote asset is not a valid supported image") from exc
        extension = Image.registered_extensions()
        for suffix, registered_format in extension.items():
            if registered_format == image_format and suffix in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".gif",
                ".bmp",
            }:
                return ".jpg" if suffix == ".jpeg" else suffix
        raise ValueError("Remote asset uses an unsupported image format")

    @classmethod
    def _resolve_filename(
        cls,
        requested: str | None,
        final_url: str,
        content_type: str,
        detected_extension: str,
    ) -> str:
        raw_name = requested or Path(unquote(urlsplit(final_url).path)).name
        filename = Path(str(raw_name)).name.strip()
        if not filename or filename in {".", ".."}:
            filename = "web_asset"
        if "\x00" in filename:
            raise ValueError("filename contains an invalid null byte")
        if not Path(filename).suffix:
            extension = (
                cls._extension_for_content_type(content_type) or detected_extension
            )
            filename = f"{filename}{extension}"
        return filename

    def _write_unique_output(self, filename: str, content: bytes) -> Path:
        for index in range(10_000):
            suffix = "" if index == 0 else f"_{index}"
            candidate = (
                self._workspace.output_dir
                / f"{Path(filename).stem}{suffix}{Path(filename).suffix}"
            )
            try:
                with candidate.open("xb") as output:
                    output.write(content)
                return candidate
            except FileExistsError:
                continue
        raise FileExistsError(f"Could not allocate a unique filename for {filename}")

    @staticmethod
    def _media_type(content_type: str) -> str:
        return content_type.split(";", 1)[0].strip().lower()

    @classmethod
    def _extension_for_content_type(cls, content_type: str) -> str | None:
        media_type = cls._media_type(content_type)
        if media_type == "image/svg+xml":
            return ".svg"
        return mimetypes.guess_extension(media_type)
