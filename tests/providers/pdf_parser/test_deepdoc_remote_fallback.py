"""Tests for DeepDoc remote routing, local fallback, and the element translator.

Every test here is a pure unit test: no network, no DeepDoc model cache, and no
writes outside ``tmp_path``. The remote client
(``deepdoc_remote.parse_document_remote``) and the local parser factory
(``DeepDocParser._get_parser_for_ext``) are the two seams that get monkeypatched,
which lets the routing decision be asserted without either side actually running.

The central guarantee under test is that remote mode never instantiates a local
parser. ``DeepDocPdfParser()`` eagerly loads ONNX models and may download them
from ModelScope, so the remote path must not touch it at all. That is asserted by
arming ``_get_parser_for_ext`` as a tripwire that raises ``AssertionError``: if the
production code ever moves local-parser construction back above the remote
dispatch, these tests fail immediately.
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from xagent.providers.pdf_parser import deepdoc as deepdoc_module
from xagent.providers.pdf_parser import deepdoc_remote
from xagent.providers.pdf_parser.deepdoc import (
    DeepDocParser,
    _translate_pdf_bboxes,
    _translate_remote_elements,
)
from xagent.providers.pdf_parser.deepdoc_remote import DeepDocRemoteError

REMOTE_URL_ENV = "XAGENT_DEEPDOC_XINFERENCE_URL"


# ==========================================
# HELPERS AND FIXTURES
# ==========================================


class RecordingProgressCallback:
    """Minimal ``ProgressCallback`` implementation that records status updates."""

    def __init__(self) -> None:
        self.statuses: List[str] = []

    def on_status_update(
        self, status: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        self.statuses.append(status)


class BrokenProgressCallback:
    """A progress sink that always raises.

    ``DeepDocProgressAdapter.get_callback()`` calls ``on_status_update`` again
    from inside its own ``except`` handler, so a sink that raises unconditionally
    makes the adapter's callback re-raise rather than swallow. Reporting the
    fallback must therefore be guarded by the caller, or a recoverable remote
    failure would turn into a hard parse failure.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def on_status_update(
        self, status: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        self.call_count += 1
        raise RuntimeError("progress sink is broken")


class FakeLocalParser:
    """Stand-in for ``DeepDocPdfParser`` that returns canned bboxes."""

    def __init__(self, bboxes: List[Dict[str, Any]]) -> None:
        self.bboxes = bboxes
        self.calls: List[Dict[str, Any]] = []

    def parse_into_bboxes(self, file_path: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        self.calls.append({"file_path": file_path, **kwargs})
        return self.bboxes


@pytest.fixture(autouse=True)
def isolate_artifacts_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect saved images into ``tmp_path``.

    ``_save_bytes_to_disk`` resolves against the module-level ``ARTIFACTS_DIR``
    (which defaults under the user's real ``~/.xagent``), so every test that
    exercises image bytes must have it pointed somewhere disposable.
    """
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(deepdoc_module, "ARTIFACTS_DIR", artifacts_dir)
    return artifacts_dir


@pytest.fixture(autouse=True)
def remote_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from "remote not configured".

    A developer ``.env`` (loaded by the root conftest with ``override=True``)
    could otherwise set the remote URL and silently flip the routing decision
    these tests are asserting.
    """
    monkeypatch.delenv(REMOTE_URL_ENV, raising=False)


@pytest.fixture
def remote_configured(monkeypatch: pytest.MonkeyPatch) -> str:
    """Configure a remote DeepDoc URL that is never actually contacted."""
    url = "http://deepdoc.invalid:9997"
    monkeypatch.setenv(REMOTE_URL_ENV, url)
    return url


class CapturedWarnings:
    """Records warnings straight off a named logger.

    These assertions were originally written against ``caplog``, which installs
    its handler on the *root* logger and therefore depends on propagation and on
    the root level surviving whatever else configured logging in the same
    process. In CI they intermittently read an empty capture -- the same two
    tests, failing on some runs and passing on others including a rerun of the
    identical commit. The trigger never reproduced locally, not even under the
    CI invocation (``-n 4 --dist=loadscope`` over the same paths).

    So rather than keep guessing at the cause, this removes every dependency it
    could plausibly have had: the handler goes on the emitting logger (no
    propagation needed), the level is forced (no inherited threshold), and
    ``disabled`` is cleared (``logging.disable()`` elsewhere in the run cannot
    suppress it). Everything is restored on exit.

    Even so, callers should treat a log assertion as a secondary check. The
    behavioural facts -- that the fallback ran, that the local parser was
    called, that ``deepdoc_backend`` says ``local`` -- are asserted separately
    and are what actually matter.
    """

    class _Collector(logging.Handler):
        def __init__(self, records: List[logging.LogRecord]) -> None:
            super().__init__(level=logging.NOTSET)
            self._records = records

        def emit(self, record: logging.LogRecord) -> None:
            self._records.append(record)

    def __init__(self, logger_name: str) -> None:
        self._logger = logging.getLogger(logger_name)
        self._records: List[logging.LogRecord] = []
        self._handler: logging.Handler = self._Collector(self._records)
        self._previous_level = self._logger.level
        self._previous_disabled = self._logger.disabled
        self._previous_manager_disable = logging.root.manager.disable

    def __enter__(self) -> "CapturedWarnings":
        # A module-level logging.disable() suppresses records before any handler
        # sees them, and it is process-global, so clear it for the duration.
        logging.root.manager.disable = 0
        self._logger.disabled = False
        self._logger.setLevel(logging.WARNING)
        self._logger.addHandler(self._handler)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._previous_level)
        self._logger.disabled = self._previous_disabled
        logging.root.manager.disable = self._previous_manager_disable

    @property
    def text(self) -> str:
        return "\n".join(record.getMessage() for record in self._records)


def capture_deepdoc_warnings() -> CapturedWarnings:
    """Capture warnings emitted by the DeepDoc parser module."""
    return CapturedWarnings("xagent.providers.pdf_parser.deepdoc")


def capture_remote_warnings() -> CapturedWarnings:
    """Capture warnings emitted by the remote client module."""
    return CapturedWarnings("xagent.providers.pdf_parser.deepdoc_remote")


def arm_local_parser_tripwire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any local-parser construction an immediate test failure."""

    def _tripwire(self: DeepDocParser, ext: str) -> Any:
        raise AssertionError(
            f"_get_parser_for_ext({ext!r}) was called on the remote path; "
            "remote mode must never instantiate a local DeepDoc parser"
        )

    monkeypatch.setattr(DeepDocParser, "_get_parser_for_ext", _tripwire)


def arm_remote_tripwire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any remote call an immediate test failure."""

    def _tripwire(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "parse_document_remote was called while remote mode is unconfigured"
        )

    monkeypatch.setattr(deepdoc_remote, "parse_document_remote", _tripwire)


def remote_pdf_elements() -> List[Dict[str, Any]]:
    """Canned remote response for a PDF: one text element, one table, one figure."""
    return [
        {
            "type": "text",
            "text": "Remote parsed paragraph",
            "image": None,
            "metadata": {
                "layout_type": "text",
                "page_number": 1,
                "col_id": 0,
                "positions": [[1, 10, 20, 30, 40]],
            },
        },
        {
            "type": "table",
            "text": "<table><tr><td>Remote</td></tr></table>",
            "image": None,
            "metadata": {
                "layout_type": "table",
                "page_number": 2,
                "col_id": 1,
                "positions": [[2, 11, 21, 31, 41]],
            },
        },
        {
            "type": "figure",
            "text": "Remote figure caption",
            "image": None,
            "metadata": {
                "layout_type": "figure",
                "page_number": 3,
                "col_id": 0,
                "positions": [[3, 12, 22, 32, 42]],
            },
        },
    ]


# ==========================================
# ROUTING: REMOTE SUCCESS
# ==========================================


class TestRemoteRoutingSuccess:
    """Remote mode must succeed without ever touching the local ONNX path."""

    @pytest.mark.asyncio
    async def test_pdf_goes_remote_without_local_parser(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, remote_configured: str
    ) -> None:
        """A configured remote server handles .pdf; the local parser stays untouched."""
        pdf_file = tmp_path / "remote.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")

        arm_local_parser_tripwire(monkeypatch)

        recorded: Dict[str, Any] = {}

        def fake_parse_document_remote(
            file_path: Any, **kwargs: Any
        ) -> List[Dict[str, Any]]:
            recorded["file_path"] = file_path
            recorded.update(kwargs)
            return remote_pdf_elements()

        monkeypatch.setattr(
            deepdoc_remote, "parse_document_remote", fake_parse_document_remote
        )

        parser = DeepDocParser()
        result = await parser.parse(str(pdf_file), doc_id="remote_pdf_doc")

        # The remote client was reached with the routing arguments it needs.
        assert recorded["file_path"] == str(pdf_file)
        assert recorded["ext"] == ".pdf"
        assert recorded["zoomin"] == 3
        assert callable(recorded["save_image"])

        # The parse succeeded purely from remote elements.
        assert [segment.text for segment in result.text_segments] == [
            "Remote parsed paragraph"
        ]
        assert len(result.tables) == 1
        assert result.tables[0].html == "<table><tr><td>Remote</td></tr></table>"
        assert len(result.figures) == 1
        assert result.figures[0].text == "Remote figure caption"

        # And the backend marker says remote, not local.
        assert result.metadata["deepdoc_backend"] == "remote"
        assert result.metadata["file_type"] == ".pdf"
        assert result.metadata["parse_method"] == "deepdoc"

    @pytest.mark.asyncio
    async def test_remote_success_reports_progress(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, remote_configured: str
    ) -> None:
        """Remote mode wires the progress adapter through to the remote client."""
        pdf_file = tmp_path / "remote_progress.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")

        arm_local_parser_tripwire(monkeypatch)

        def fake_parse_document_remote(
            file_path: Any, callback: Any = None, **kwargs: Any
        ) -> List[Dict[str, Any]]:
            assert callback is not None, (
                "remote mode should forward a progress callback"
            )
            callback(0.05, "Uploading document to remote DeepDoc server")
            callback(1.0, "Remote DeepDoc parse finished (0.12s)")
            return [{"type": "text", "text": "remote page", "image": None}]

        monkeypatch.setattr(
            deepdoc_remote, "parse_document_remote", fake_parse_document_remote
        )

        progress = RecordingProgressCallback()
        parser = DeepDocParser()
        result = await parser.parse(
            str(pdf_file), progress_callback=progress, doc_id="remote_progress_doc"
        )

        assert result.metadata["deepdoc_backend"] == "remote"
        assert [segment.text for segment in result.text_segments] == ["remote page"]
        # The adapter strips the timing suffix, matching local DeepDoc's shape.
        assert progress.statuses == [
            "Uploading document to remote DeepDoc server",
            "Remote DeepDoc parse finished",
        ]

    @pytest.mark.asyncio
    async def test_pdf_bytesio_goes_remote_without_local_parser(
        self, monkeypatch: pytest.MonkeyPatch, remote_configured: str
    ) -> None:
        """An in-memory PDF also routes remote, with no local parse."""
        arm_local_parser_tripwire(monkeypatch)

        def fake_parse_document_remote(
            file_path: Any, **kwargs: Any
        ) -> List[Dict[str, Any]]:
            assert isinstance(file_path, BytesIO)
            assert kwargs["ext"] == ".pdf"
            return [
                {
                    "type": "text",
                    "text": "In-memory remote paragraph",
                    "image": None,
                    "metadata": {"layout_type": "text", "page_number": 1},
                }
            ]

        monkeypatch.setattr(
            deepdoc_remote, "parse_document_remote", fake_parse_document_remote
        )

        parser = DeepDocParser()
        result = await parser._parse_impl(
            BytesIO(b"%PDF-1.4 not really a pdf"),
            file_ext=".pdf",
            doc_id="remote_bytesio_pdf_doc",
        )

        assert result.metadata["deepdoc_backend"] == "remote"
        assert result.metadata["source"] == "memory_buffer"
        assert [segment.text for segment in result.text_segments] == [
            "In-memory remote paragraph"
        ]


# ==========================================
# ROUTING: NON-PDF NEVER GOES REMOTE
# ==========================================


class TestNonPdfStaysLocal:
    """``task=parse`` consumes PDFs only, so nothing else may attempt a round trip.

    Before the gate existed, a configured remote server made every format upload
    itself just to be rejected -- a ``.docx`` bought a 500 before falling back to
    the same local parse that would have run anyway. These tests pin the gate by
    arming the remote client as a tripwire: any HTTP attempt is a test failure.
    """

    @pytest.mark.asyncio
    async def test_docx_never_calls_remote(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, remote_configured: str
    ) -> None:
        """The known-gap case: a .docx must reach the local parser directly."""
        docx_file = tmp_path / "local.docx"
        # A ZIP magic header, so the cheap Open XML pre-check passes without
        # needing a real DOCX on disk.
        docx_file.write_bytes(b"PK\x03\x04 not really a docx")

        arm_remote_tripwire(monkeypatch)

        # The .docx branch dispatches on isinstance(parser, DoclingParser), so
        # the stand-in has to be one; only parse_docx is exercised.
        class FakeDocxParser(deepdoc_module.DeepDocDoclingParser):  # type: ignore[misc]
            def __init__(self) -> None:
                pass

            def check_installation(self) -> bool:
                return True

            def parse_docx(self, file_path: Any, **kwargs: Any) -> Any:
                return ([("Local DOCX body", "Normal")], [])

        monkeypatch.setattr(
            DeepDocParser, "_get_parser_for_ext", lambda self, ext: FakeDocxParser()
        )

        parser = DeepDocParser()
        result = await parser.parse(str(docx_file), doc_id="local_docx_doc")

        assert [segment.text for segment in result.text_segments] == ["Local DOCX body"]
        # _translate_docx_output carries the metadata on the segments rather than
        # on the ParseResult, so that is where the backend marker lands.
        assert result.text_segments[0].metadata["deepdoc_backend"] == "local"
        assert result.text_segments[0].metadata["file_type"] == ".docx"

    @pytest.mark.asyncio
    async def test_txt_never_calls_remote(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, remote_configured: str
    ) -> None:
        """A plain-text file is read locally; the remote server never hears about it."""
        txt_file = tmp_path / "local.txt"
        txt_file.write_text("Local text content", encoding="utf-8")

        arm_remote_tripwire(monkeypatch)

        parser = DeepDocParser()
        result = await parser.parse(str(txt_file), doc_id="local_txt_doc")

        assert [segment.text for segment in result.text_segments] == [
            "Local text content"
        ]
        # _translate_text_output carries the metadata on the segments rather than
        # on the ParseResult, so that is where the backend marker lands.
        assert result.text_segments[0].metadata["deepdoc_backend"] == "local"
        assert result.text_segments[0].metadata["file_type"] == ".txt"

    @pytest.mark.asyncio
    async def test_xlsx_bytesio_never_calls_remote(
        self, monkeypatch: pytest.MonkeyPatch, remote_configured: str
    ) -> None:
        """An in-memory spreadsheet goes straight to the openpyxl row reader."""
        arm_remote_tripwire(monkeypatch)
        arm_local_parser_tripwire(monkeypatch)

        def fake_parse_xlsx_rows(file_path: Any, **kwargs: Any) -> Any:
            from xagent.providers.pdf_parser.base import ParsedTextSegment, ParseResult

            return ParseResult(
                text_segments=[
                    ParsedTextSegment(text="local xlsx row", metadata=dict(kwargs))
                ],
                metadata=dict(kwargs),
            )

        monkeypatch.setattr(deepdoc_module, "_parse_xlsx_rows", fake_parse_xlsx_rows)

        parser = DeepDocParser()
        result = await parser._parse_impl(
            BytesIO(b"not really an xlsx"),
            file_ext=".xlsx",
            doc_id="local_xlsx_doc",
        )

        assert [segment.text for segment in result.text_segments] == ["local xlsx row"]
        assert result.metadata["deepdoc_backend"] == "local"


# ==========================================
# ROUTING: REMOTE FAILURE FALLS BACK TO LOCAL
# ==========================================


class TestRemoteFailureFallsBackToLocal:
    """Any remote failure must degrade to local parsing with a warning."""

    @pytest.mark.asyncio
    async def test_remote_error_falls_back_to_local_pdf(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        remote_configured: str,
    ) -> None:
        """DeepDocRemoteError yields the local result, a warning, and backend=local."""
        pdf_file = tmp_path / "fallback.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")

        def failing_parse_document_remote(*args: Any, **kwargs: Any) -> Any:
            raise DeepDocRemoteError("Remote DeepDoc request failed: boom")

        monkeypatch.setattr(
            deepdoc_remote, "parse_document_remote", failing_parse_document_remote
        )

        local_bboxes = [
            {
                "layout_type": "text",
                "text": "Locally parsed paragraph",
                "positions": [[1, 10, 20, 30, 40]],
            },
            {
                "layout_type": "table",
                "text": "<table><tr><td>Local</td></tr></table>",
                "image": None,
            },
        ]
        fake_parser = FakeLocalParser(local_bboxes)
        monkeypatch.setattr(
            DeepDocParser, "_get_parser_for_ext", lambda self, ext: fake_parser
        )

        parser = DeepDocParser()
        with capture_deepdoc_warnings() as captured:
            result = await parser.parse(str(pdf_file), doc_id="fallback_pdf_doc")

        # The local result is what comes back.
        assert [segment.text for segment in result.text_segments] == [
            "Locally parsed paragraph"
        ]
        assert len(result.tables) == 1
        assert result.tables[0].html == "<table><tr><td>Local</td></tr></table>"
        assert result.metadata["deepdoc_backend"] == "local"

        # The local parser really ran, with the local call signature.
        assert len(fake_parser.calls) == 1
        assert fake_parser.calls[0]["file_path"] == str(pdf_file)
        assert fake_parser.calls[0]["zoomin"] == 3

        # And the failure was logged rather than swallowed.
        assert "Remote DeepDoc parse failed" in captured.text
        assert "falling back to local" in captured.text

    @pytest.mark.asyncio
    async def test_broken_progress_sink_does_not_abort_the_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        remote_configured: str,
    ) -> None:
        """Reporting the fallback must never be able to prevent the fallback.

        ``DeepDocProgressAdapter.get_callback()`` re-raises out of its own
        ``except`` handler when the sink misbehaves, so without the caller's
        try/except around the fallback notification this test fails with
        ``RuntimeError`` instead of returning the local result.
        """
        pdf_file = tmp_path / "broken_sink.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")

        def failing_parse_document_remote(*args: Any, **kwargs: Any) -> Any:
            raise DeepDocRemoteError("Remote DeepDoc request failed: boom")

        monkeypatch.setattr(
            deepdoc_remote, "parse_document_remote", failing_parse_document_remote
        )

        fake_parser = FakeLocalParser(
            [{"layout_type": "text", "text": "Local survived the broken sink"}]
        )
        monkeypatch.setattr(
            DeepDocParser, "_get_parser_for_ext", lambda self, ext: fake_parser
        )

        progress = BrokenProgressCallback()
        parser = DeepDocParser()
        with capture_deepdoc_warnings() as captured:
            result = await parser.parse(
                str(pdf_file),
                progress_callback=progress,
                doc_id="broken_sink_doc",
            )

        assert [segment.text for segment in result.text_segments] == [
            "Local survived the broken sink"
        ]
        assert result.metadata["deepdoc_backend"] == "local"
        # The sink was genuinely exercised and genuinely raised.
        assert progress.call_count > 0
        assert "Progress callback failed while reporting the DeepDoc fallback" in (
            captured.text
        )

    @pytest.mark.asyncio
    async def test_remote_error_falls_back_for_an_in_memory_pdf(
        self,
        monkeypatch: pytest.MonkeyPatch,
        remote_configured: str,
    ) -> None:
        """A BytesIO PDF falls back too, and the buffer is still readable locally."""

        def failing_parse_document_remote(*args: Any, **kwargs: Any) -> Any:
            raise DeepDocRemoteError("Remote DeepDoc returned an unusable response")

        monkeypatch.setattr(
            deepdoc_remote, "parse_document_remote", failing_parse_document_remote
        )

        fake_parser = FakeLocalParser(
            [{"layout_type": "text", "text": "Local in-memory paragraph"}]
        )
        monkeypatch.setattr(
            DeepDocParser, "_get_parser_for_ext", lambda self, ext: fake_parser
        )

        stream = BytesIO(b"%PDF-1.4 not really a pdf")
        parser = DeepDocParser()
        result = await parser._parse_impl(
            stream,
            file_ext=".pdf",
            doc_id="fallback_bytesio_pdf_doc",
        )

        assert [segment.text for segment in result.text_segments] == [
            "Local in-memory paragraph"
        ]
        assert result.metadata["deepdoc_backend"] == "local"
        assert fake_parser.calls[0]["file_path"] is stream


# ==========================================
# ROUTING: ENV UNSET STAYS LOCAL
# ==========================================


class TestEnvUnsetStaysLocal:
    """With no remote URL configured, nothing may reach the remote client."""

    @pytest.mark.asyncio
    async def test_pure_local_path_never_calls_remote(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The default configuration parses locally with no remote attempt."""
        pdf_file = tmp_path / "local_only.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")

        arm_remote_tripwire(monkeypatch)

        fake_parser = FakeLocalParser(
            [{"layout_type": "text", "text": "Local only paragraph"}]
        )
        monkeypatch.setattr(
            DeepDocParser, "_get_parser_for_ext", lambda self, ext: fake_parser
        )

        parser = DeepDocParser()
        result = await parser.parse(str(pdf_file), doc_id="local_only_doc")

        assert [segment.text for segment in result.text_segments] == [
            "Local only paragraph"
        ]
        assert result.metadata["deepdoc_backend"] == "local"
        assert len(fake_parser.calls) == 1

    @pytest.mark.asyncio
    async def test_malformed_remote_url_degrades_to_local(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A typo in the URL must not break every parse; it degrades to local."""
        pdf_file = tmp_path / "malformed_url.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")

        monkeypatch.setenv(REMOTE_URL_ENV, "ftp://not-http")
        arm_remote_tripwire(monkeypatch)

        fake_parser = FakeLocalParser(
            [{"layout_type": "text", "text": "Local despite a bad URL"}]
        )
        monkeypatch.setattr(
            DeepDocParser, "_get_parser_for_ext", lambda self, ext: fake_parser
        )

        parser = DeepDocParser()
        with capture_remote_warnings() as captured:
            result = await parser.parse(str(pdf_file), doc_id="malformed_url_doc")

        assert [segment.text for segment in result.text_segments] == [
            "Local despite a bad URL"
        ]
        assert result.metadata["deepdoc_backend"] == "local"
        assert "parsing locally" in captured.text


# ==========================================
# TRANSLATOR
# ==========================================


class TestTranslateRemoteElements:
    """``_translate_remote_elements`` must match the local translators' output."""

    @staticmethod
    def base_kwargs() -> Dict[str, Any]:
        return {
            "source": "remote.pdf",
            "file_type": ".pdf",
            "parse_method": "deepdoc",
            "deepdoc_backend": "remote",
        }

    def test_pdf_shaped_elements_match_local_translation(self) -> None:
        """A PDF-shaped element set translates exactly like the local bbox path."""
        doc_id = "translator_pdf_doc"
        kwargs = self.base_kwargs()

        elements = [
            {
                "type": "text",
                "text": "Paragraph one",
                "image": None,
                "metadata": {
                    "layout_type": "text",
                    "page_number": 4,
                    "col_id": 1,
                    "positions": [[4, 10, 20, 30, 40]],
                },
            },
            {
                "type": "table",
                "text": "<table><tr><td>T</td></tr></table>",
                "image": None,
                "metadata": {
                    "layout_type": "table",
                    "page_number": 5,
                    "col_id": 0,
                    "positions": [[5, 11, 21, 31, 41]],
                },
            },
            {
                "type": "figure",
                "text": "A caption",
                "image": None,
                "metadata": {
                    "layout_type": "figure",
                    "page_number": 6,
                    "col_id": 0,
                    "positions": [[6, 12, 22, 32, 42]],
                },
            },
        ]

        remote_result = _translate_remote_elements(doc_id, elements, **kwargs)

        # The equivalent local input: the same content, already bbox-shaped.
        local_bboxes = [
            {**element["metadata"], "text": element["text"], "image": element["image"]}
            for element in elements
        ]
        local_result = _translate_pdf_bboxes(doc_id, local_bboxes, **kwargs)

        assert len(remote_result.text_segments) == len(local_result.text_segments) == 1
        assert len(remote_result.tables) == len(local_result.tables) == 1
        assert len(remote_result.figures) == len(local_result.figures) == 1

        assert remote_result.text_segments[0].text == local_result.text_segments[0].text
        assert (
            remote_result.text_segments[0].metadata
            == local_result.text_segments[0].metadata
        )
        assert remote_result.tables[0].html == local_result.tables[0].html
        assert remote_result.tables[0].metadata == local_result.tables[0].metadata
        assert remote_result.figures[0].text == local_result.figures[0].text
        assert remote_result.figures[0].metadata == local_result.figures[0].metadata
        assert remote_result.metadata == local_result.metadata == kwargs

    def test_positions_are_enriched_with_col_id(self) -> None:
        """``positions`` gain ``col_id`` at index 1 and float coordinates."""
        result = _translate_remote_elements(
            "positions_doc",
            [
                {
                    "type": "text",
                    "text": "Positioned text",
                    "image": None,
                    "metadata": {
                        "layout_type": "text",
                        "page_number": 1,
                        "col_id": 0,
                        "positions": [[1, 10, 20, 30, 40]],
                    },
                }
            ],
            **self.base_kwargs(),
        )

        metadata = result.text_segments[0].metadata
        assert metadata["positions"] == [[1, 0, 10.0, 20.0, 30.0, 40.0]]
        assert metadata["col_id"] == 0
        assert metadata["page_number"] == 1
        assert metadata["layout_type"] == "text"
        assert metadata["doc_id"] == "positions_doc"

    def test_non_zero_col_id_is_inserted_into_positions(self) -> None:
        """A two-column element carries its own ``col_id`` into every position."""
        result = _translate_remote_elements(
            "two_column_doc",
            [
                {
                    "type": "text",
                    "text": "Right column",
                    "image": None,
                    "metadata": {
                        "layout_type": "text",
                        "page_number": 2,
                        "col_id": 1,
                        "positions": [[2, 300, 590, 100, 140], [3, 300, 590, 0, 60]],
                    },
                }
            ],
            **self.base_kwargs(),
        )

        assert result.text_segments[0].metadata["positions"] == [
            [2, 1, 300.0, 590.0, 100.0, 140.0],
            [3, 1, 300.0, 590.0, 0.0, 60.0],
        ]

    def test_title_elements_become_text_segments(self) -> None:
        """``title`` is a real server layout type, and must not be dropped."""
        result = _translate_remote_elements(
            "title_doc",
            [
                {
                    "type": "title",
                    "text": "Sample Document",
                    "image": None,
                    "metadata": {
                        "layout_type": "title",
                        "page_number": 1,
                        "col_id": 0,
                        "positions": [[1, 70.7, 256.3, 77.3, 96.3]],
                    },
                }
            ],
            **self.base_kwargs(),
        )

        assert [segment.text for segment in result.text_segments] == ["Sample Document"]
        assert result.text_segments[0].metadata["layout_type"] == "title"
        assert result.tables == []
        assert result.figures == []

    def test_raw_bbox_coordinates_survive_translation(self) -> None:
        """x0/x1/top/bottom and layoutno reach the segment metadata untouched."""
        result = _translate_remote_elements(
            "coords_doc",
            [
                {
                    "type": "text",
                    "text": "Positioned paragraph",
                    "image": None,
                    "metadata": {
                        "x0": 70.666,
                        "x1": 256.333,
                        "top": 77.333,
                        "bottom": 96.333,
                        "layoutno": "text-3",
                        "layout_type": "text",
                        "page_number": 1,
                    },
                }
            ],
            **self.base_kwargs(),
        )

        metadata = result.text_segments[0].metadata
        assert metadata["x0"] == 70.666
        assert metadata["bottom"] == 96.333
        assert metadata["layoutno"] == "text-3"

    def test_absent_col_id_defaults_to_zero(self) -> None:
        """The server omits ``col_id`` on some elements; translation must not fail.

        ``_assign_column`` only labels elements it assigned to a column, so the
        key is legitimately missing on tables and figures reinserted into the
        text flow. The local translator defaults those to column 0 and so must
        this one.
        """
        result = _translate_remote_elements(
            "no_col_id_doc",
            [
                {
                    "type": "table",
                    "text": "<table><tr><td>T</td></tr></table>",
                    "image": None,
                    "metadata": {
                        "layout_type": "table",
                        "page_number": 2,
                        "positions": [[2, 20.0, 400.0, 50.0, 200.0]],
                    },
                }
            ],
            **self.base_kwargs(),
        )

        assert result.tables[0].metadata["col_id"] == 0
        assert result.tables[0].metadata["positions"] == [
            [2, 0, 20.0, 400.0, 50.0, 200.0]
        ]

    def test_unknown_element_type_degrades_to_text(self) -> None:
        """A future element type must become a text segment, never be dropped."""
        result = _translate_remote_elements(
            "unknown_type_doc",
            [
                {
                    "type": "equation",
                    "text": "E = mc^2",
                    "image": None,
                    "metadata": {"page_number": 7},
                },
                {"type": "text", "text": "Ordinary text", "image": None},
            ],
            **self.base_kwargs(),
        )

        assert [segment.text for segment in result.text_segments] == [
            "E = mc^2",
            "Ordinary text",
        ]
        # The unknown type is preserved in metadata rather than rewritten.
        assert result.text_segments[0].metadata["layout_type"] == "equation"
        assert result.text_segments[0].metadata["page_number"] == 7
        assert result.tables == []
        assert result.figures == []

    def test_missing_and_none_metadata_do_not_blow_up(self) -> None:
        """Elements with absent, ``None``, or non-dict metadata still translate."""
        result = _translate_remote_elements(
            "sparse_metadata_doc",
            [
                {"type": "text", "text": "No metadata key"},
                {"type": "text", "text": "None metadata", "metadata": None},
                {"type": "text", "text": "List metadata", "metadata": ["nope"]},
                {"type": "table", "text": "<table></table>", "metadata": None},
                {"type": "figure", "text": "", "metadata": None},
            ],
            **self.base_kwargs(),
        )

        assert [segment.text for segment in result.text_segments] == [
            "No metadata key",
            "None metadata",
            "List metadata",
        ]
        for segment in result.text_segments:
            # _build_element_metadata supplies the defaults.
            assert segment.metadata["layout_type"] == "text"
            assert segment.metadata["page_number"] == 1
            assert segment.metadata["col_id"] == 0
            assert "positions" not in segment.metadata

        assert len(result.tables) == 1
        assert result.tables[0].metadata["image_path"] is None
        assert result.tables[0].metadata["type"] == "table"

        assert len(result.figures) == 1
        # An empty caption is backfilled so downstream processing has text.
        assert result.figures[0].text == "Figure"
        assert result.figures[0].metadata["image_path"] is None
        assert result.figures[0].metadata["type"] == "figure"

    def test_empty_element_list_yields_empty_result(self) -> None:
        """No elements means empty lists, with the shared metadata still set."""
        kwargs = self.base_kwargs()
        result = _translate_remote_elements("empty_doc", [], **kwargs)

        assert result.text_segments == []
        assert result.tables == []
        assert result.figures == []
        assert result.metadata == kwargs

    def test_image_path_strings_are_carried_onto_table_and_figure(
        self, tmp_path: Path
    ) -> None:
        """The remote client's saved-image paths flow through unchanged.

        The client rewrites ``image_base64`` into an on-disk path string, which is
        exactly what the local ``_handle_image`` string branch already accepts.
        """
        table_image = tmp_path / "table.png"
        table_image.write_bytes(b"fake png bytes")
        figure_image = tmp_path / "figure.png"
        figure_image.write_bytes(b"fake png bytes")

        result = _translate_remote_elements(
            "image_doc",
            [
                {
                    "type": "table",
                    "text": "<table><tr><td>with image</td></tr></table>",
                    "image": str(table_image),
                    "metadata": {"page_number": 1},
                },
                {
                    "type": "figure",
                    "text": "Figure with image",
                    "image": str(figure_image),
                    "metadata": {"page_number": 2},
                },
            ],
            **self.base_kwargs(),
        )

        assert result.tables[0].metadata["image_path"] == str(table_image)
        assert result.figures[0].metadata["image_path"] == str(figure_image)

    def test_real_image_bytes_are_saved_under_the_patched_artifacts_dir(
        self, isolate_artifacts_dir: Path
    ) -> None:
        """``_save_bytes_to_disk`` writes only inside the patched artifacts dir."""
        image_path = Path(
            deepdoc_module._save_bytes_to_disk("bytes_doc", b"fake png bytes", ".png")
        )

        assert image_path.is_file()
        assert image_path.read_bytes() == b"fake png bytes"
        assert isolate_artifacts_dir in image_path.parents


class TestTranslatorToleratesNullFields:
    """The translator must not crash on an explicit null `type` or `text`.

    The client rejects those before they get here, so this is the second layer:
    a `.get(key, default)` would hand a present-but-null value straight through,
    and `ParsedTextSegment` would then raise a pydantic ValidationError — which
    the remote branch does not catch, so the local fallback would be skipped.
    """

    def test_null_type_and_text_degrade_to_empty_text_segment(self) -> None:
        """A null `type` reads as text and a null `text` as the empty string."""
        result = deepdoc_module._translate_remote_elements(
            "null-doc",
            [
                {
                    "type": None,
                    "text": None,
                    "image": None,
                    "metadata": {"page_number": 1},
                }
            ],
        )

        assert len(result.text_segments) == 1
        assert result.text_segments[0].text == ""
        assert result.text_segments[0].metadata["layout_type"] == "text"
        assert not result.tables
        assert not result.figures


class TestUntrustedMetadataFallsBackRatherThanRaising:
    """Server-supplied `metadata` is untrusted and must not be able to fail a parse.

    `_normalize_elements` validates the top-level `type`/`text` but passes
    `metadata` through untouched by design, so anything in it reaches
    `_build_element_metadata`, where `int(pos[0])`/`float(pos[1:])` run bare.
    Those raise `ValueError`, and `_translate_remote_elements` runs inside the
    remote branch's `try`, so the fallback has to be broad enough to catch them
    -- otherwise the document fails to ingest even though local parsing was
    available.
    """

    @pytest.mark.asyncio
    # Only a well-formed 5-tuple reaches the bare int()/float() calls;
    # _build_element_metadata skips anything shorter or not a sequence, so those
    # shapes pass through harmlessly and are covered by the translator tests.
    @pytest.mark.parametrize(
        "positions",
        [
            pytest.param([["bad", 1, 2, 3, 4]], id="non-numeric-page"),
            pytest.param([[1, 2, 3, 4, "bad"]], id="non-numeric-coord"),
            pytest.param([[1, None, 2, 3, 4]], id="none-coord"),
        ],
    )
    async def test_malformed_positions_fall_back_to_local(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        remote_configured: str,
        positions: Any,
    ) -> None:
        """A parse still succeeds locally when the server sends unusable positions."""
        pdf_file = tmp_path / "untrusted.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")

        monkeypatch.setattr(
            deepdoc_remote,
            "parse_document_remote",
            lambda *args, **kwargs: [
                {
                    "type": "text",
                    "text": "remote paragraph",
                    "image": None,
                    "metadata": {"page_number": 1, "positions": positions},
                }
            ],
        )
        fake_parser = FakeLocalParser(
            [{"layout_type": "text", "text": "local paragraph"}]
        )
        monkeypatch.setattr(
            DeepDocParser, "_get_parser_for_ext", lambda self, ext: fake_parser
        )

        parser = DeepDocParser()
        with capture_deepdoc_warnings() as captured:
            result = await parser.parse(str(pdf_file), doc_id="untrusted")

        assert result.metadata["deepdoc_backend"] == "local"
        assert result.text_segments[0].text == "local paragraph"
        assert "falling back to local" in captured.text

    @pytest.mark.asyncio
    async def test_missing_image_path_falls_back_to_local(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        remote_configured: str,
    ) -> None:
        """`_handle_image` raises for a path that no longer exists; that must fall back."""
        pdf_file = tmp_path / "missing_crop.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")

        monkeypatch.setattr(
            deepdoc_remote,
            "parse_document_remote",
            lambda *args, **kwargs: [
                {
                    "type": "table",
                    "text": "<table></table>",
                    "image": str(tmp_path / "gone.png"),
                    "metadata": {"page_number": 1},
                }
            ],
        )
        fake_parser = FakeLocalParser(
            [{"layout_type": "text", "text": "local paragraph"}]
        )
        monkeypatch.setattr(
            DeepDocParser, "_get_parser_for_ext", lambda self, ext: fake_parser
        )

        parser = DeepDocParser()
        with capture_deepdoc_warnings() as captured:
            result = await parser.parse(str(pdf_file), doc_id="missing-crop")

        assert result.metadata["deepdoc_backend"] == "local"
        assert "falling back to local" in captured.text


class TestRawOutputPassthroughOnTheRemotePath:
    """`enable_raw_output` must mean the same thing remotely as it does locally.

    `ParseResult`'s own visualization helpers read `raw_parser_output["bboxes"]`
    and expect the coordinate keys flat. Keying the remote payload any other way
    leaves `has_visualization_data` true while `get_visualization_elements()`
    silently returns nothing — a parity break that looks like working code.
    """

    @pytest.mark.asyncio
    async def test_remote_raw_output_feeds_the_visualization_helpers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, remote_configured: str
    ) -> None:
        pdf_file = tmp_path / "viz.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")

        monkeypatch.setattr(
            deepdoc_remote,
            "parse_document_remote",
            lambda *a, **k: remote_pdf_elements(),
        )
        arm_local_parser_tripwire(monkeypatch)

        parser = DeepDocParser(enable_raw_output=True)
        result = await parser.parse(str(pdf_file), doc_id="viz_doc")

        assert result.parser_engine == "deepdoc"
        assert result.has_visualization_data
        assert result.raw_parser_output is not None
        assert result.raw_parser_output["total_elements"] == 3
        assert result.raw_parser_output["has_positions"] is True

        # The helpers must actually yield the elements, not an empty list.
        elements = result.get_visualization_elements()
        assert len(elements) == 3
        assert [element["type"] for element in elements] == ["text", "table", "figure"]
        assert [element["page"] for element in elements] == [1, 2, 3]

        summary = result.get_visualization_summary()
        assert summary["total_elements"] == 3
        assert summary["elements_by_type"] == {"text": 1, "table": 1, "figure": 1}

    @pytest.mark.asyncio
    async def test_raw_output_stays_absent_when_the_flag_is_off(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, remote_configured: str
    ) -> None:
        pdf_file = tmp_path / "noviz.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")

        monkeypatch.setattr(
            deepdoc_remote,
            "parse_document_remote",
            lambda *a, **k: remote_pdf_elements(),
        )
        arm_local_parser_tripwire(monkeypatch)

        result = await DeepDocParser().parse(str(pdf_file), doc_id="noviz_doc")

        assert result.raw_parser_output is None
        assert result.parser_engine is None
        assert not result.has_visualization_data


class TestBackendMarkerOnEveryFormat:
    """`deepdoc_backend` must survive every format's translator.

    Only the PDF translator used to forward the parse metadata onto the
    ParseResult, so a DOCX/XLSX/MD/TXT/CSV parse returned `deepdoc_backend=None`
    and the remote-versus-local outcome was unobservable for those formats.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("suffix", "payload"),
        [
            (".txt", b"plain text body"),
            (".md", b"# heading\n\nbody text\n"),
        ],
    )
    async def test_local_parse_marks_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        suffix: str,
        payload: bytes,
    ) -> None:
        """Formats that bypass the PDF branch still report the local backend."""
        source = tmp_path / f"sample{suffix}"
        source.write_bytes(payload)

        arm_remote_tripwire(monkeypatch)

        parser = DeepDocParser()
        result = await parser.parse(str(source), doc_id="backend-marker")

        assert result.metadata.get("deepdoc_backend") == "local"


class TestRawOutputFlagDoesNotChangeParsing:
    """`enable_raw_output` is a debug switch and must not decide if a parse works.

    Building the passthrough payload reads each element's `metadata`, which is
    untrusted — it can be absent, null, or not an object. `{**None}` raises, so an
    unguarded build made these inputs parse remotely with the flag off and fall
    back to local with it on.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "metadata",
        [
            pytest.param(None, id="metadata-null"),
            pytest.param("not-an-object", id="metadata-string"),
            pytest.param(["not", "an", "object"], id="metadata-list"),
        ],
    )
    async def test_odd_metadata_parses_the_same_either_way(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        remote_configured: str,
        metadata: Any,
    ) -> None:
        pdf_file = tmp_path / "flag.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")

        monkeypatch.setattr(
            deepdoc_remote,
            "parse_document_remote",
            lambda *args, **kwargs: [
                {
                    "type": "text",
                    "text": "remote paragraph",
                    "image": None,
                    "metadata": metadata,
                }
            ],
        )
        arm_local_parser_tripwire(monkeypatch)

        for enable_raw_output in (False, True):
            parser = DeepDocParser(enable_raw_output=enable_raw_output)
            result = await parser.parse(str(pdf_file), doc_id="flag_doc")

            # Remote either way: the tripwire would have fired on a fallback.
            assert result.metadata["deepdoc_backend"] == "remote"
            assert result.text_segments[0].text == "remote paragraph"
            if enable_raw_output:
                assert result.raw_parser_output is not None
                assert result.raw_parser_output["total_elements"] == 1


class TestCropsAreNotOrphanedByTranslationFailure:
    """A crop written before translation failed describes a discarded result.

    The client cleans up after its own failures, but once it has returned, a
    crop stranded by a translation error would sit in the artifacts tree
    alongside the fresh set the local fallback writes, and nothing prunes it.
    """

    @pytest.mark.asyncio
    async def test_written_crop_is_removed_when_translation_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        remote_configured: str,
        isolate_artifacts_dir: Path,
    ) -> None:
        pdf_file = tmp_path / "orphan.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")

        # The first element's crop lands on disk; the second trips the
        # translator on an uncoercible page number.
        crop = isolate_artifacts_dir / "already_written.png"
        crop.parent.mkdir(parents=True, exist_ok=True)
        crop.write_bytes(b"fake png")

        monkeypatch.setattr(
            deepdoc_remote,
            "parse_document_remote",
            lambda *args, **kwargs: [
                {
                    "type": "table",
                    "text": "<table></table>",
                    "image": str(crop),
                    "metadata": {"page_number": 1},
                },
                {
                    "type": "text",
                    "text": "trips the translator",
                    "image": None,
                    "metadata": {"page_number": "not-a-number"},
                },
            ],
        )
        fake_parser = FakeLocalParser(
            [{"layout_type": "text", "text": "local paragraph"}]
        )
        monkeypatch.setattr(
            DeepDocParser, "_get_parser_for_ext", lambda self, ext: fake_parser
        )

        result = await DeepDocParser().parse(str(pdf_file), doc_id="orphan_doc")

        assert result.metadata["deepdoc_backend"] == "local"
        assert not crop.exists(), "crop from the discarded remote result was kept"


class TestRawOutputCarriesTheImageKey:
    """`has_image` in the visualization helpers tests for key *presence*."""

    @pytest.mark.asyncio
    async def test_image_key_survives_into_raw_bboxes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, remote_configured: str
    ) -> None:
        pdf_file = tmp_path / "viz.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")
        crop = tmp_path / "table.png"
        crop.write_bytes(b"fake png")

        monkeypatch.setattr(
            deepdoc_remote,
            "parse_document_remote",
            lambda *args, **kwargs: [
                {
                    "type": "table",
                    "text": "<table></table>",
                    "image": str(crop),
                    "metadata": {"page_number": 1},
                }
            ],
        )
        arm_local_parser_tripwire(monkeypatch)

        result = await DeepDocParser(enable_raw_output=True).parse(
            str(pdf_file), doc_id="viz_image_doc"
        )

        assert result.raw_parser_output is not None
        assert "image" in result.raw_parser_output["bboxes"][0]
        element = result.get_visualization_elements()[0]
        assert element["metadata"]["has_image"] is True


class TestUntrustedNumericMetadataIsCoerced:
    """`page_number` and `col_id` are coerced like `positions` already were."""

    def test_numeric_strings_become_ints(self) -> None:
        result = _translate_remote_elements(
            "coerce-doc",
            [
                {
                    "type": "text",
                    "text": "paragraph",
                    "image": None,
                    "metadata": {"page_number": "3", "col_id": "2"},
                }
            ],
        )

        metadata = result.text_segments[0].metadata
        assert metadata["page_number"] == 3
        assert metadata["col_id"] == 2
        assert isinstance(metadata["page_number"], int)
        assert isinstance(metadata["col_id"], int)

    def test_uncoercible_value_raises_so_the_caller_falls_back(self) -> None:
        with pytest.raises(ValueError):
            _translate_remote_elements(
                "coerce-doc",
                [
                    {
                        "type": "text",
                        "text": "paragraph",
                        "image": None,
                        "metadata": {"page_number": "page one"},
                    }
                ],
            )
