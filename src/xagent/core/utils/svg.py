"""Safe SVG rasterization helpers for provider-facing image inputs."""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import cast

MAX_SVG_BYTES = 5 * 1024 * 1024
DEFAULT_RASTER_WIDTH = 1600
_REMOTE_REFERENCE_RE = re.compile(
    rb"(?:href\s*=\s*['\"]\s*(?:https?:|file:|//)|"
    rb"url\(\s*['\"]?\s*(?:https?:|file:|//)|@import)",
    re.IGNORECASE,
)


def rasterize_svg_bytes(
    svg_bytes: bytes,
    *,
    output_width: int = DEFAULT_RASTER_WIDTH,
) -> bytes:
    """Render trusted, self-contained SVG bytes to PNG for image providers."""

    if not svg_bytes:
        raise ValueError("SVG content is empty")
    if len(svg_bytes) > MAX_SVG_BYTES:
        raise ValueError(f"SVG exceeds maximum size of {MAX_SVG_BYTES} bytes")
    lowered_prefix = svg_bytes[:4096].lower()
    if b"<svg" not in lowered_prefix:
        raise ValueError("SVG content does not contain an <svg> root")
    if b"<!doctype" in lowered_prefix or b"<!entity" in lowered_prefix:
        raise ValueError("SVG declarations and entities are not supported")
    if _REMOTE_REFERENCE_RE.search(svg_bytes):
        raise ValueError("SVG must be self-contained and cannot load remote resources")
    if output_width <= 0 or output_width > 4096:
        raise ValueError("output_width must be between 1 and 4096 pixels")

    cairosvg_error: Exception | None = None
    try:
        import cairosvg  # type: ignore[import-not-found]

        try:
            return cast(
                bytes,
                cairosvg.svg2png(
                    bytestring=svg_bytes,
                    output_width=output_width,
                    unsafe=False,
                ),
            )
        except Exception as exc:
            cairosvg_error = exc
    except Exception as exc:
        cairosvg_error = exc

    rsvg_convert = shutil.which("rsvg-convert")
    if rsvg_convert:
        try:
            completed = subprocess.run(
                [
                    rsvg_convert,
                    "--format",
                    "png",
                    "--width",
                    str(output_width),
                    "-",
                ],
                input=svg_bytes,
                capture_output=True,
                check=True,
                timeout=30,
            )
            if completed.stdout:
                return completed.stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(
                f"Failed to rasterize SVG with rsvg-convert: {exc}"
            ) from exc

    detail = f": {cairosvg_error}" if cairosvg_error else ""
    raise RuntimeError(
        "SVG rasterization requires a working CairoSVG or rsvg-convert installation"
        f"{detail}"
    )
