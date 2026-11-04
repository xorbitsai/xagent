"""Tests for safe SVG rasterization."""

from types import SimpleNamespace

import pytest

from xagent.core.utils.svg import (
    MAX_RASTER_PIXELS,
    rasterize_svg_bytes,
    validate_svg_bytes,
)


def _patch_svg2png(monkeypatch):
    calls = []

    def svg2png(**kwargs):
        calls.append(kwargs)
        return b"png"

    monkeypatch.setitem(
        __import__("sys").modules,
        "cairosvg",
        SimpleNamespace(svg2png=svg2png),
    )
    return calls


def test_rasterize_svg_uses_safe_cairosvg_options(monkeypatch) -> None:
    calls = _patch_svg2png(monkeypatch)

    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">'
        b'<path d="M0 0" /></svg>'
    )
    assert rasterize_svg_bytes(svg, output_width=800) == b"png"
    assert calls == [
        {
            "bytestring": svg,
            "output_width": 800,
            "output_height": 400,
            "unsafe": False,
        }
    ]


@pytest.mark.parametrize(
    "svg",
    [
        b'<svg xmlns="http://www.w3.org/2000/svg"><image href="https://bad/x" /></svg>',
        b'<svg xmlns="http://www.w3.org/2000/svg"><style>@import "//bad/x";</style></svg>',
        b'<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><svg />',
    ],
)
def test_rasterize_svg_rejects_external_resources(svg: bytes) -> None:
    with pytest.raises(ValueError):
        rasterize_svg_bytes(svg)


def test_rasterize_svg_rejects_declaration_after_large_padding() -> None:
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg">'
        + b"<!-- padding -->" * 400
        + b'<!ENTITY xxe SYSTEM "file:///etc/passwd">'
        + b"</svg>"
    )

    with pytest.raises(ValueError, match="declarations and entities"):
        rasterize_svg_bytes(svg)


def test_rasterize_rejects_or_downscales_extreme_viewbox(monkeypatch) -> None:
    calls = _patch_svg2png(monkeypatch)

    svg = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 100000" />'
    rasterize_svg_bytes(svg, output_width=4096)

    kwargs = calls[0]
    assert kwargs["output_width"] * kwargs["output_height"] <= MAX_RASTER_PIXELS


def test_rasterize_caps_height_when_dimensions_unknown(monkeypatch) -> None:
    calls = _patch_svg2png(monkeypatch)

    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0" /></svg>'
    rasterize_svg_bytes(svg, output_width=800)

    kwargs = calls[0]
    assert kwargs["output_width"] == 800
    assert kwargs["output_height"] == MAX_RASTER_PIXELS // 800


def test_validate_svg_bytes_rejects_script() -> None:
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
    with pytest.raises(ValueError, match="script"):
        validate_svg_bytes(svg)


def test_validate_svg_bytes_accepts_clean_svg() -> None:
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0" /></svg>'
    validate_svg_bytes(svg)  # no raise


@pytest.mark.parametrize(
    "svg",
    [
        b'<! DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><svg />',
        b'<!\nENTITY xxe SYSTEM "file:///etc/passwd"><svg />',
        b'<!  DoCtYpE svg [<!  EnTiTy xxe SYSTEM "file:///etc/passwd">]><svg />',
        b"<!\tdoctype svg><svg />",
    ],
)
def test_validate_svg_bytes_rejects_whitespace_obfuscated_declarations(
    svg: bytes,
) -> None:
    with pytest.raises(ValueError, match="declarations and entities"):
        validate_svg_bytes(svg)


def test_validate_svg_bytes_rejects_exact_doctype_and_entity() -> None:
    with pytest.raises(ValueError, match="declarations and entities"):
        validate_svg_bytes(
            b'<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><svg />'
        )
    with pytest.raises(ValueError, match="declarations and entities"):
        validate_svg_bytes(b'<!ENTITY xxe SYSTEM "file:///etc/passwd"><svg />')
