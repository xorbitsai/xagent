"""Tests for the on-disk SVG preview rasterization cache.

Covers two review findings from PR #977:
 - the cache key must not alias distinct SVG assets that share a base
   ``file_id`` (e.g. relative-path assets under one uploaded file);
 - rasterization must not block the asyncio event loop.
"""

import asyncio
import time

import pytest

import xagent.web.api.files as files_module
from xagent.web.api.files import _rasterize_svg_preview, _svg_png_cache_path

SVG_A = (
    b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
    b'<rect width="10" height="10" fill="red"/></svg>'
)
SVG_B = (
    b'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">'
    b'<circle cx="10" cy="10" r="10" fill="blue"/></svg>'
)


@pytest.fixture()
def storage_root(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path))
    return tmp_path


def test_svg_png_cache_path_differs_for_different_svg_paths_same_file_id(
    storage_root, tmp_path
) -> None:
    """Two different SVG assets registered under the same base file_id must
    not resolve to the same cache file name — otherwise the second asset's
    preview request would read back the first asset's cached PNG."""

    svg_a = tmp_path / "a.svg"
    svg_b = tmp_path / "sub" / "b.svg"
    svg_b.parent.mkdir()
    svg_a.write_bytes(SVG_A)
    svg_b.write_bytes(SVG_B)

    path_a = _svg_png_cache_path(svg_a, file_id="shared-file-id")
    path_b = _svg_png_cache_path(svg_b, file_id="shared-file-id")

    assert path_a != path_b


def test_rasterize_svg_preview_does_not_alias_across_relative_assets(
    storage_root, tmp_path
) -> None:
    """Rendering two distinct SVGs registered under the same file_id must
    produce two distinct PNG caches, not a shared/reused one."""

    svg_a = tmp_path / "a.svg"
    svg_b = tmp_path / "sub" / "b.svg"
    svg_b.parent.mkdir()
    svg_a.write_bytes(SVG_A)
    svg_b.write_bytes(SVG_B)

    cache_a = _rasterize_svg_preview(svg_a, file_id="shared-file-id")
    cache_b = _rasterize_svg_preview(svg_b, file_id="shared-file-id")

    assert cache_a != cache_b
    assert cache_a.read_bytes() != cache_b.read_bytes()


@pytest.mark.asyncio
async def test_inline_preview_response_does_not_block_event_loop(
    storage_root, tmp_path, monkeypatch
) -> None:
    """``_inline_preview_response`` must offload the synchronous SVG
    read+rasterize work (e.g. via ``asyncio.to_thread``) rather than run it
    inline on the async request path — otherwise one expensive preview
    blocks every other request handled by the same event loop."""

    svg_path = tmp_path / "big.svg"
    svg_path.write_bytes(SVG_A)

    def slow_rasterize(svg_bytes: bytes) -> bytes:
        time.sleep(0.3)
        return SVG_A  # placeholder bytes; content isn't checked here

    monkeypatch.setattr(files_module, "rasterize_svg_bytes", slow_rasterize)

    ticks: list[float] = []

    async def ticker() -> None:
        for _ in range(20):
            ticks.append(time.monotonic())
            await asyncio.sleep(0.01)

    ticker_task = asyncio.ensure_future(ticker())
    await asyncio.sleep(0.01)
    await files_module._inline_preview_response(
        svg_path,
        filename="big.svg",
        media_type="image/svg+xml",
        file_id="event-loop-test-file-id",
    )
    await ticker_task

    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert gaps, "ticker should have produced measurable ticks"
    assert max(gaps) < 0.2, (
        "the ticker was starved for a stretch close to the 0.3s "
        "rasterization delay — the SVG preview is still running "
        "synchronously on the event loop thread"
    )
