"""Tests for safe SVG rasterization."""

import subprocess
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


def test_rasterize_svg_uses_rsvg_convert_fallback(monkeypatch) -> None:
    def fail_svg2png(**_kwargs):
        raise RuntimeError("Cairo unavailable")

    monkeypatch.setitem(
        __import__("sys").modules,
        "cairosvg",
        SimpleNamespace(svg2png=fail_svg2png),
    )
    monkeypatch.setattr(
        "xagent.core.utils.svg.shutil.which",
        lambda _name: "/usr/bin/rsvg-convert",
    )
    run_calls = []

    def run(*args, **kwargs):
        run_calls.append((args, kwargs))
        return SimpleNamespace(stdout=b"fallback-png")

    monkeypatch.setattr("xagent.core.utils.svg.subprocess.run", run)
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" />'

    assert rasterize_svg_bytes(svg, output_width=640) == b"fallback-png"
    assert run_calls[0][0][0] == [
        "/usr/bin/rsvg-convert",
        "--format",
        "png",
        "--width",
        "640",
        "-",
    ]
    assert run_calls[0][1]["input"] == svg


@pytest.mark.parametrize("failure", ["nonzero", "empty"])
def test_rasterize_svg_reports_rsvg_convert_failures(monkeypatch, failure) -> None:
    def fail_svg2png(**_kwargs):
        raise RuntimeError("Cairo unavailable")

    monkeypatch.setitem(
        __import__("sys").modules,
        "cairosvg",
        SimpleNamespace(svg2png=fail_svg2png),
    )
    monkeypatch.setattr(
        "xagent.core.utils.svg.shutil.which",
        lambda _name: "/usr/bin/rsvg-convert",
    )

    def run(*args, **_kwargs):
        if failure == "nonzero":
            raise subprocess.CalledProcessError(1, args[0])
        return SimpleNamespace(stdout=b"")

    monkeypatch.setattr("xagent.core.utils.svg.subprocess.run", run)
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" />'
    message = (
        "Failed to rasterize SVG with rsvg-convert"
        if failure == "nonzero"
        else "rsvg-convert returned empty PNG output"
    )

    with pytest.raises(ValueError, match=message):
        rasterize_svg_bytes(svg)
