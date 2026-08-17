"""Download an exact web image asset into the current task workspace."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Mapping, Type
from urllib.parse import unquote, urlsplit

import httpx
from PIL import Image
from pydantic import BaseModel, Field

from ....file_ref import build_workspace_file_ref
from ....utils.security import fetch_public_http_bytes
from ....utils.svg import validate_svg_bytes
from ....workspace import TaskWorkspace
from ...core.web_content import (
    DEFAULT_MAX_CONTENT_BYTES,
    DEFAULT_USER_AGENT,
    get_trusted_proxy_url,
)
from .base import AbstractBaseTool, ToolCategory, ToolVisibility


class DownloadWebAssetArgs(BaseModel):
    url: str = Field(
        description=(
            "Exact HTTP or HTTPS image URL discovered on an authoritative page. "
            "For a brand asset, authoritative means the brand's own domain or "
            "its CDN, reached while carrying out a retrieval the user asked for."
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
            "fetch_web_content(include_assets=true) surfaced the asset the user "
            "asked you to retrieve; a downloaded image is not proof that it is a "
            "brand's authentic asset, so do not treat one as verified identity "
            "material for a retrieval nobody requested. It returns a trusted "
            "FileRef, so do not reproduce the asset through api_call plus "
            "write_file."
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
                detected_extension,
            )
            with self._workspace.auto_register_files():
                target = self._write_unique_output(filename, content)
            file_ref = build_workspace_file_ref(
                workspace=self._workspace,
                file_path=target,
                mime_type=self._mime_type_for_extension(detected_extension),
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
        proxy_url = get_trusted_proxy_url()
        if proxy_url:
            client_kwargs["proxy"] = proxy_url

        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await fetch_public_http_bytes(
                client,
                url,
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=30,
                max_content_bytes=self._max_content_bytes,
                resource_name="remote asset",
                require_non_empty=True,
                via_proxy=bool(proxy_url),
            )
        return response.content, response.url, response.content_type

    @classmethod
    def _validate_image(
        cls,
        content: bytes,
        content_type: str,
        url: str,
    ) -> str:
        media_type = cls._media_type(content_type)
        url_suffix = Path(urlsplit(url).path).suffix.lower()
        if media_type and not media_type.startswith("image/"):
            raise ValueError(f"Unsupported web asset content type: {content_type}")
        if media_type == "image/svg+xml" or url_suffix == ".svg":
            validate_svg_bytes(content)
            return ".svg"
        try:
            with Image.open(io.BytesIO(content)) as image:
                image_format = image.format
                image.verify()
        except Exception as exc:
            raise ValueError("Remote asset is not a valid supported image") from exc
        registered_extensions = Image.registered_extensions()
        for extension, registered_format in registered_extensions.items():
            if registered_format == image_format and extension in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".gif",
                ".bmp",
            }:
                return ".jpg" if extension == ".jpeg" else extension
        raise ValueError("Remote asset uses an unsupported image format")

    @classmethod
    def _resolve_filename(
        cls,
        requested: str | None,
        final_url: str,
        detected_extension: str,
    ) -> str:
        raw_name = requested or Path(unquote(urlsplit(final_url).path)).name
        filename = Path(str(raw_name)).name.strip()
        if not filename or filename in {".", ".."}:
            filename = "web_asset"
        if "\x00" in filename:
            raise ValueError("filename contains an invalid null byte")
        existing = Path(filename).suffix.lower()
        jpeg_aliases = {".jpg", ".jpeg"}
        if not existing:
            filename = f"{filename}{detected_extension}"
        elif existing != detected_extension and not (
            existing in jpeg_aliases and detected_extension in jpeg_aliases
        ):
            filename = f"{Path(filename).stem}{detected_extension}"
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

    @staticmethod
    def _mime_type_for_extension(extension: str) -> str:
        return {
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
        }[extension]
