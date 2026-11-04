"""Safe SVG rasterization helpers for provider-facing image inputs."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from typing import cast

MAX_SVG_BYTES = 5 * 1024 * 1024
DEFAULT_RASTER_WIDTH = 1600
MAX_RASTER_PIXELS = 4096 * 4096
_REMOTE_REFERENCE_RE = re.compile(
    rb"(?:href\s*=\s*['\"]\s*(?:https?:|file:|//)|"
    rb"url\(\s*['\"]?\s*(?:https?:|file:|//)|@import)",
    re.IGNORECASE,
)


def validate_svg_bytes(svg_bytes: bytes) -> None:
    """Reject SVG content that is unsafe to store or rasterize."""

    if not svg_bytes:
        raise ValueError("SVG content is empty")
    if len(svg_bytes) > MAX_SVG_BYTES:
        raise ValueError(f"SVG exceeds maximum size of {MAX_SVG_BYTES} bytes")
    lowered_all = svg_bytes.lower()
    if b"<svg" not in lowered_all[:4096]:
        raise ValueError("SVG content does not contain an <svg> root")
    if re.search(rb"<!\s*(?:doctype|entity)\b", lowered_all):
        raise ValueError("SVG declarations and entities are not supported")
    if b"<script" in lowered_all:
        raise ValueError("SVG must not contain <script> elements")
    if _REMOTE_REFERENCE_RE.search(svg_bytes):
        raise ValueError("SVG must be self-contained and cannot load remote resources")


def _parse_length(value: str | None) -> float | None:
    if not value:
        return None
    match = re.match(r"\s*([0-9]*\.?[0-9]+)", value)
    return float(match.group(1)) if match else None


def _svg_aspect_ratio(svg_bytes: bytes) -> float | None:
    """Return intrinsic height/width, or None if undeterminable."""

    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError:
        return None
    view_box = root.get("viewBox")
    if view_box:
        parts = re.split(r"[\s,]+", view_box.strip())
        if len(parts) == 4:
            try:
                _, _, vb_w, vb_h = (float(p) for p in parts)
                if vb_w > 0 and vb_h > 0:
                    return vb_h / vb_w
            except ValueError:
                pass
    width = _parse_length(root.get("width"))
    height = _parse_length(root.get("height"))
    if width and height and width > 0:
        return height / width
    return None


def _bounded_dimensions(svg_bytes: bytes, output_width: int) -> tuple[int, int]:
    ratio = _svg_aspect_ratio(svg_bytes)
    if ratio is None:
        return output_width, max(1, MAX_RASTER_PIXELS // output_width)
    height = max(1, round(output_width * ratio))
    if output_width * height > MAX_RASTER_PIXELS:
        scale = math.sqrt(MAX_RASTER_PIXELS / (output_width * height))
        output_width = max(1, int(output_width * scale))
        height = max(1, int(height * scale))
    return output_width, height


def rasterize_svg_bytes(
    svg_bytes: bytes,
    *,
    output_width: int = DEFAULT_RASTER_WIDTH,
) -> bytes:
    """Render trusted, self-contained SVG bytes to PNG for image providers."""

    validate_svg_bytes(svg_bytes)
    if output_width <= 0 or output_width > 4096:
        raise ValueError("output_width must be between 1 and 4096 pixels")
    width, height = _bounded_dimensions(svg_bytes, output_width)

    import cairosvg  # type: ignore[import-not-found]

    return cast(
        bytes,
        cairosvg.svg2png(
            bytestring=svg_bytes,
            output_width=width,
            output_height=height,
            unsafe=False,
        ),
    )
