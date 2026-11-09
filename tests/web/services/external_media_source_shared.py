"""Instagram source images for the #3596 tests.

Oversized sources and a byte-counting reader: Pillow reads a PNG chunk or JPEG
segment it does not decode in full and keeps it, so these sources cost memory
by their size, not their pixels. The tests lower ``INSTAGRAM_MAX_SOURCE_BYTES``
to ``SOURCE_LIMIT`` and write sources twice that size, so they stay small and
fast. Also small JPEGs of each scan layout, for the pixel limits.
"""

from __future__ import annotations

import os
import struct
import zlib
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO

import pytest
from PIL import Image

from xagent.web.services import external_media

SOURCE_LIMIT = 1024 * 1024
OVERSIZED = 2 * SOURCE_LIMIT


def _image_bytes(image_format: str) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 64), "blue").save(buffer, image_format)
    return buffer.getvalue()


def write_png_with_private_chunk(path: Path, size: int) -> None:
    """A valid 64x64 PNG of exactly ``size`` bytes, padded by a private chunk.

    The ``prVt`` chunk holds zeros that are skipped over rather than written,
    as a sandbox could do: in a sparse file a large chunk takes almost no disk,
    yet its full size counts on ``stat`` and when read.
    """
    png = _image_bytes("PNG")
    ihdr_end = 8 + 4 + 4 + 13 + 4  # signature, then IHDR's length, type, data, CRC
    chunk_size = size - len(png) - 12  # the chunk's length, type and CRC
    crc = zlib.crc32(b"prVt")
    zeros = memoryview(bytes(1024 * 1024))
    for offset in range(0, chunk_size, len(zeros)):
        crc = zlib.crc32(zeros[: chunk_size - offset], crc)
    with path.open("wb") as out:
        out.write(png[:ihdr_end])
        out.write(struct.pack(">I4s", chunk_size, b"prVt"))
        out.seek(chunk_size, os.SEEK_CUR)
        out.write(struct.pack(">I", crc))
        out.write(png[ihdr_end:])


def write_jpeg_with_app_segments(path: Path, size: int) -> None:
    """A valid 64x64 JPEG preceded by at least ``size`` bytes of APP9 segments."""
    jpeg = _image_bytes("JPEG")
    segment = b"\xff\xe9" + struct.pack(">H", 0xFFFF) + bytes(0xFFFF - 2)
    count = -(-size // len(segment))
    path.write_bytes(jpeg[:2] + segment * count + jpeg[2:])


OVERSIZED_SOURCES = pytest.mark.parametrize(
    ("filename", "write_oversized"),
    [
        ("launch.png", write_png_with_private_chunk),
        ("launch.jpg", write_jpeg_with_app_segments),
    ],
    ids=["png-private-chunk", "jpeg-app-segments"],
)


# A baseline (SOF0) 400x400 JPEG whose three components each have their own
# scan. It is a lossless rewrite of Image.new("RGB", (400, 400), "blue") saved
# as JPEG by Pillow 12.3.0, made with jpegtran from libjpeg-turbo 3.2.1:
#   printf '0;\n1;\n2;\n' > scans.txt
#   jpegtran -scans scans.txt -outfile noninterleaved_400x400.jpg interleaved.jpg
NONINTERLEAVED_JPEG = Path(__file__).parent / "fixtures" / "noninterleaved_400x400.jpg"


def jpeg_400px(layout: str) -> bytes:
    """A blue 400x400 JPEG: "progressive", "interleaved" in one scan, or
    "non-interleaved" with one scan per component."""
    if layout == "non-interleaved":
        return NONINTERLEAVED_JPEG.read_bytes()
    buffer = BytesIO()
    Image.new("RGB", (400, 400), "blue").save(
        buffer, "JPEG", progressive=layout == "progressive"
    )
    return buffer.getvalue()


# The scan layouts against the PNG pixel limit, lowered to 400 * 400 + slack:
# one scan per layout keeps the higher JPEG limit, more than one gets the PNG
# limit (#3596). Shared by the issuance and the route tests.
MULTI_SCAN_PIXEL_LIMIT_CASES = pytest.mark.parametrize(
    ("layout", "slack", "refused_as"),
    [
        ("progressive", -1, "progressive JPEG"),
        ("progressive", 0, None),
        ("non-interleaved", -1, "multi-scan JPEG"),
        ("non-interleaved", 0, None),
        ("interleaved", -1, None),
    ],
    ids=[
        "progressive-over",
        "progressive-at-limit",
        "non-interleaved-over",
        "non-interleaved-at-limit",
        "interleaved-same-size",
    ],
)


def write_small_image(path: Path) -> None:
    """A small valid image, in the format the file name's extension says."""
    image_format = "JPEG" if path.suffix == ".jpg" else "PNG"
    Image.new("RGB", (640, 640), "blue").save(path, image_format)


_UNCOUNTED_READS = frozenset(
    {"peek", "read1", "readinto", "readinto1", "readline", "readlines"}
)


class CountingReader:
    """An opened source that counts every byte read from it.

    ``before_first_read`` runs once, just before the first read, the way a
    sandbox writer replacing the file after it was opened would.
    """

    def __init__(
        self,
        stream: BinaryIO,
        before_first_read: Callable[[], None] | None = None,
    ) -> None:
        self._stream = stream
        self._before_first_read = before_first_read
        self.bytes_read = 0

    def _starting_read(self) -> None:
        hook, self._before_first_read = self._before_first_read, None
        if hook is not None:
            hook()

    def read(self, size: int = -1) -> bytes:
        self._starting_read()
        data = self._stream.read(size)
        self.bytes_read += len(data)
        return data

    def __getattr__(self, name: str) -> Any:
        if name in _UNCOUNTED_READS:
            raise AttributeError(f"{name}() would read without being counted")
        return getattr(self._stream, name)

    def __enter__(self) -> CountingReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stream.close()


def count_source_reads(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    *,
    before_first_read: Callable[[], None] | None = None,
) -> list[CountingReader]:
    """Count what ``module`` reads from each file it opens for delivery."""
    opened: list[CountingReader] = []
    open_delivery_file = external_media.open_delivery_file

    def open_counting(record: Any, path: Path) -> CountingReader:
        reader = CountingReader(open_delivery_file(record, path), before_first_read)
        opened.append(reader)
        return reader

    monkeypatch.setattr(module, "open_delivery_file", open_counting)
    return opened
