"""Tests for the on-disk SVG preview rasterization cache.

Covers review findings from PR #977:
 - the cache key must not alias distinct SVG assets that share a base
   ``file_id`` (e.g. relative-path assets under one uploaded file);
 - rasterization must not block the asyncio event loop;
 - concurrent cache-miss requests for the same SVG must not race on a
   shared temp file.
"""

import asyncio
import threading

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
    """Rasterization must leave the loop running while its worker is blocked."""
    svg_path = tmp_path / "big.svg"
    svg_path.write_bytes(SVG_A)
    entered = threading.Event()
    release = threading.Event()
    loop_thread = threading.get_ident()

    def blocked_rasterize(svg_bytes: bytes) -> bytes:
        entered.set()
        assert threading.get_ident() != loop_thread
        assert release.wait(timeout=30), "rasterization was never released"
        return SVG_A

    monkeypatch.setattr(files_module, "rasterize_svg_bytes", blocked_rasterize)
    preview = asyncio.create_task(
        files_module._inline_preview_response(
            svg_path,
            filename="big.svg",
            media_type="image/svg+xml",
            file_id="event-loop-test-file-id",
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 30)
        assert not preview.done()
    finally:
        release.set()
        await asyncio.wait_for(preview, timeout=30)


@pytest.mark.asyncio
async def test_concurrent_same_key_rasterize_does_not_race(
    storage_root, tmp_path
) -> None:
    """Two concurrent cache-miss previews of the *same* SVG must not race
    on a shared temp file: offloading rasterization via ``asyncio.to_thread``
    lets multiple requests enter ``_rasterize_svg_preview`` for the same
    cache key at once, and a temp filename that isn't unique per call means
    the first ``replace()`` can consume the file the second is about to
    rename onto the cache path, raising ``FileNotFoundError``."""

    svg_path = tmp_path / "shared.svg"
    svg_path.write_bytes(SVG_A)

    async def run_once():
        return await asyncio.to_thread(
            _rasterize_svg_preview, svg_path, "shared-file-id"
        )

    # Run several rounds: a fresh svg per round forces a real cache miss
    # (and thus a real concurrent write) every time, since a hit from a
    # prior round would just short-circuit to the existing cache file.
    for round_index in range(20):
        round_svg = tmp_path / f"shared_{round_index}.svg"
        round_svg.write_bytes(SVG_A)

        results = await asyncio.gather(
            asyncio.to_thread(
                _rasterize_svg_preview, round_svg, f"shared-file-id-{round_index}"
            ),
            asyncio.to_thread(
                _rasterize_svg_preview, round_svg, f"shared-file-id-{round_index}"
            ),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, BaseException):
                raise result

        cache_a, cache_b = results
        assert cache_a == cache_b
        assert cache_a.exists()
        assert cache_a.read_bytes()
