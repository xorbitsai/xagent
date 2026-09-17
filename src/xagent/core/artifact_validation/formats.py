"""Independent format checks; no business-content or minimum-size heuristics."""

import csv
import zlib
from io import BytesIO, StringIO

from .models import ArtifactContent, InvalidArtifact, UncheckedArtifact


def check_csv(content: ArtifactContent) -> None:
    data = content.data
    try:
        # BOMs are explicit encoding declarations. Do not guess latin-1 and
        # thereby accept arbitrary binary files as a one-column CSV.
        encoding = (
            "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        )
        text = data.decode(encoding)
    except UnicodeError as exc:
        raise UncheckedArtifact(
            "CSV encoding is not UTF-8 or BOM-declared UTF-16."
        ) from exc
    if "\x00" in text:
        raise InvalidArtifact("CSV contains binary NUL characters.")
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    try:
        csv.field_size_limit(content.limits.max_bytes)
        for index, _row in enumerate(
            csv.reader(StringIO(text, newline=""), dialect, strict=True)
        ):
            if index >= content.limits.max_units:
                raise UncheckedArtifact("CSV exceeds the row budget.")
        # Ragged rows, an empty file, or a single column can all be legitimate.
        # Schema/business expectations belong to additional explicit checks.
    except csv.Error as exc:
        # Lenient readers may recover nonstandard quoting, but can also merge
        # truncated records silently. Neither corruption nor validity is proven.
        raise UncheckedArtifact(
            "CSV quoting or record structure is ambiguous."
        ) from exc


def check_pdf(content: ArtifactContent) -> None:
    from pypdf import PdfReader
    from pypdf.errors import DependencyError, PdfReadError, PyPdfError
    from pypdf.generic import ArrayObject, EncodedStreamObject

    if not content.data.lstrip().startswith(b"%PDF-"):
        raise InvalidArtifact("PDF header is missing.")
    try:
        # Validate readability, not strict PDF conformance. Real readers can
        # recover common producer defects such as an inaccurate xref offset.
        reader = PdfReader(BytesIO(content.data), strict=False)
        if reader.is_encrypted:
            raise UncheckedArtifact("Encrypted PDFs require a password to validate.")
        # A fixed parser ceiling, not partial validation: larger PDFs are
        # unchecked in their entirety, never certified from a page sample.
        page_limit = min(content.limits.max_units, 500)
        if len(reader.pages) > page_limit:
            raise UncheckedArtifact(
                f"PDF exceeds the page budget ({page_limit} pages)."
            )
        expanded = 0
        for page in reader.pages:
            raw_contents = page.get("/Contents")
            if raw_contents is not None:
                raw_contents = raw_contents.get_object()
                parts = (
                    raw_contents
                    if isinstance(raw_contents, ArrayObject)
                    else [raw_contents]
                )
                for part in parts:
                    part = part.get_object()
                    if not isinstance(part, EncodedStreamObject):
                        continue
                    filters = part.get("/Filter")
                    if filters not in (
                        "/FlateDecode",
                        "/Fl",
                        ["/FlateDecode"],
                        ["/Fl"],
                    ):
                        continue
                    # pypdf's recovery decoder can silently return empty bytes
                    # for an unreadable FlateDecode stream. Distinguish that
                    # from a genuinely empty compressed stream, without treating
                    # successful nonempty recovery as strict-conformance failure.
                    if not part.get_data():
                        encoded = getattr(part, "_data", None)
                        if not isinstance(encoded, bytes):
                            raise UncheckedArtifact(
                                "PDF reader does not expose encoded stream bytes for recovery verification."
                            )
                        try:
                            decoder = zlib.decompressobj()
                            decoded = decoder.decompress(encoded, 1)
                        except zlib.error as exc:
                            raise UncheckedArtifact(
                                "PDF content stream recovery could not be verified."
                            ) from exc
                        if decoded or not decoder.eof:
                            raise UncheckedArtifact(
                                "PDF content stream recovery could not be verified."
                            )
            stream = page.get_contents()
            if stream is not None:
                expanded += len(stream.get_data())
                if expanded > content.limits.max_expanded_bytes:
                    raise UncheckedArtifact("PDF exceeds the decoded content budget.")
    except (PdfReadError, ValueError, KeyError, TypeError, OSError) as exc:
        raise InvalidArtifact(
            "PDF structure or page content cannot be decoded."
        ) from exc
    except (PyPdfError, DependencyError) as exc:
        raise UncheckedArtifact(
            "PDF reader could not complete validation (dependency or parser limit)."
        ) from exc


def check_image(content: ArtifactContent) -> None:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(BytesIO(content.data)) as image:
            width, height = image.size
            frames = getattr(image, "n_frames", 1)
            if width * height * frames > content.limits.max_pixels:
                raise UncheckedArtifact("Image exceeds the decoded pixel budget.")
            image.verify()
        # verify() checks container structure; load() also exercises decoding.
        with Image.open(BytesIO(content.data)) as image:
            for index in range(getattr(image, "n_frames", 1)):
                image.seek(index)
                image.load()
    except Image.DecompressionBombError as exc:
        raise UncheckedArtifact("Image exceeds the decoded pixel budget.") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise InvalidArtifact("Image data cannot be decoded.") from exc
