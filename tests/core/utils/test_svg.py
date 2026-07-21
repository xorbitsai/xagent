"""Tests for safe SVG rasterization."""

from types import SimpleNamespace

import pytest

from xagent.core.utils.svg import rasterize_svg_bytes


def test_rasterize_svg_uses_safe_cairosvg_options(monkeypatch) -> None:
    calls = []

    def svg2png(**kwargs):
        calls.append(kwargs)
        return b"png"

    monkeypatch.setitem(
        __import__("sys").modules,
        "cairosvg",
        SimpleNamespace(svg2png=svg2png),
    )

    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0" /></svg>'
    assert rasterize_svg_bytes(svg, output_width=800) == b"png"
    assert calls == [
        {
            "bytestring": svg,
            "output_width": 800,
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
