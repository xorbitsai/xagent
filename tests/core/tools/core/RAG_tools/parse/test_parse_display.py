"""Tests for parse_display: reconstruct_parse_result_from_db and paginate_parse_results.

Validates multi-tenancy (user_id/is_admin) and latest-by-created_at behavior.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import DocumentNotFoundError
from xagent.core.tools.core.RAG_tools.core.schemas import ParseMethod
from xagent.core.tools.core.RAG_tools.file.register_document import register_document
from xagent.core.tools.core.RAG_tools.parse.parse_display import (
    paginate_parse_results,
    reconstruct_parse_result_from_db,
)
from xagent.core.tools.core.RAG_tools.parse.parse_document import parse_document

RESOURCES_DIR = Path("tests/resources/test_files")


@pytest.fixture
def temp_lancedb_dir():
    """Isolate LanceDB per test (same pattern as test_parse_document)."""
    base_dir = Path(os.environ.get("LANCEDB_DIR", "/tmp/.lancedb_test_root")).resolve()
    unique_dir = base_dir / f"pytest_parse_display_{uuid.uuid4().hex[:8]}"
    unique_dir.mkdir(parents=True, exist_ok=True)
    old_dir = os.environ.get("LANCEDB_DIR")
    os.environ["LANCEDB_DIR"] = str(unique_dir)
    try:
        yield str(unique_dir)
    finally:
        if old_dir is not None:
            os.environ["LANCEDB_DIR"] = old_dir
        else:
            os.environ.pop("LANCEDB_DIR", None)
        import shutil

        if unique_dir.exists():
            shutil.rmtree(unique_dir)


@pytest.fixture
def test_collection() -> str:
    return f"test_collection_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_doc_id() -> str:
    return str(uuid.uuid4())


class TestReconstructParseResultMultiTenancy:
    """Multi-tenancy: user_id and is_admin filtering."""

    def _require_file(self, relative: str) -> Path:
        p = RESOURCES_DIR / relative
        if not p.exists():
            pytest.skip(f"Sample file not found: {p}")
        return p

    def test_owner_sees_own_parse(
        self, temp_lancedb_dir: str, test_collection: str, test_doc_id: str
    ) -> None:
        """User who created the parse can reconstruct it."""
        sample = self._require_file("test.txt")
        register_document(
            collection=test_collection,
            source_path=str(sample),
            doc_id=test_doc_id,
            user_id=1,
        )
        parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=False,
        )
        elements, parse_hash = reconstruct_parse_result_from_db(
            test_collection,
            test_doc_id,
            parse_hash=None,
            user_id=1,
            is_admin=False,
        )
        assert parse_hash is not None and len(parse_hash) > 0
        assert isinstance(elements, list)
        # At least one element for non-empty txt
        if sample.read_text(encoding="utf-8").strip():
            assert len(elements) >= 1
        for el in elements:
            assert "type" in el and el["type"] in ("text", "table", "figure")
            assert "metadata" in el

    def test_other_user_cannot_see_parse(
        self, temp_lancedb_dir: str, test_collection: str, test_doc_id: str
    ) -> None:
        """Regular user cannot see another user's parse."""
        sample = self._require_file("test.txt")
        register_document(
            collection=test_collection,
            source_path=str(sample),
            doc_id=test_doc_id,
            user_id=1,
        )
        parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=False,
        )
        with pytest.raises(DocumentNotFoundError):
            reconstruct_parse_result_from_db(
                test_collection,
                test_doc_id,
                parse_hash=None,
                user_id=2,
                is_admin=False,
            )

    def test_admin_sees_other_user_parse(
        self, temp_lancedb_dir: str, test_collection: str, test_doc_id: str
    ) -> None:
        """Admin can reconstruct parse owned by another user."""
        sample = self._require_file("test.txt")
        register_document(
            collection=test_collection,
            source_path=str(sample),
            doc_id=test_doc_id,
            user_id=1,
        )
        parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=False,
        )
        elements, parse_hash = reconstruct_parse_result_from_db(
            test_collection,
            test_doc_id,
            parse_hash=None,
            user_id=99,
            is_admin=True,
        )
        assert parse_hash is not None
        assert isinstance(elements, list)

    def test_unauthenticated_sees_nothing(
        self, temp_lancedb_dir: str, test_collection: str, test_doc_id: str
    ) -> None:
        """Without user_id and not admin, no data is visible."""
        sample = self._require_file("test.txt")
        register_document(
            collection=test_collection,
            source_path=str(sample),
            doc_id=test_doc_id,
            user_id=1,
        )
        parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=False,
        )
        with pytest.raises(DocumentNotFoundError):
            reconstruct_parse_result_from_db(
                test_collection,
                test_doc_id,
                parse_hash=None,
                user_id=None,
                is_admin=False,
            )


class TestReconstructParseResultLatestByCreatedAt:
    """When parse_hash is None, the latest parse by created_at is returned."""

    def _require_file(self, relative: str) -> Path:
        p = RESOURCES_DIR / relative
        if not p.exists():
            pytest.skip(f"Sample file not found: {p}")
        return p

    def test_latest_parse_returned_when_no_hash(
        self, temp_lancedb_dir: str, test_collection: str, test_doc_id: str
    ) -> None:
        """With two parses (different methods -> different hashes), no parse_hash returns the latest by created_at."""
        sample = self._require_file("test.txt")
        register_document(
            collection=test_collection,
            source_path=str(sample),
            doc_id=test_doc_id,
            user_id=1,
        )
        first = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=True,
        )
        # Add a small delay to ensure different created_at timestamps
        time.sleep(0.1)
        second = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.DEFAULT,  # Different method -> different parse_hash -> new record
            user_id=1,
            is_admin=True,
        )
        hash1 = first["parse_hash"]
        hash2 = second["parse_hash"]
        # Different parse methods yield different hashes, so we have two distinct records
        assert hash1 != hash2

        # Without parse_hash we should get the latest by created_at (second write)
        elements, actual_hash = reconstruct_parse_result_from_db(
            test_collection,
            test_doc_id,
            parse_hash=None,
            user_id=1,
            is_admin=True,
        )
        assert actual_hash == hash2
        assert isinstance(elements, list)

        # With explicit parse_hash we get the corresponding version
        elements1, h1 = reconstruct_parse_result_from_db(
            test_collection, test_doc_id, parse_hash=hash1, user_id=1, is_admin=True
        )
        assert h1 == hash1
        elements2, h2 = reconstruct_parse_result_from_db(
            test_collection, test_doc_id, parse_hash=hash2, user_id=1, is_admin=True
        )
        assert h2 == hash2


class TestPaginateParseResults:
    """Unit tests for paginate_parse_results."""

    def test_pagination_first_page(self) -> None:
        """First page returns correct slice and pagination info."""
        elements = [
            {"type": "text", "text": f"seg{i}", "metadata": {}} for i in range(25)
        ]
        page_elements, info = paginate_parse_results(elements, page=1, page_size=10)
        assert len(page_elements) == 10
        assert info["page"] == 1
        assert info["page_size"] == 10
        assert info["total_elements"] == 25
        assert info["total_pages"] == 3
        assert info["has_next"] is True
        assert info["has_previous"] is False
        assert page_elements[0].text == "seg0"

    def test_pagination_last_page(self) -> None:
        """Last page returns remainder and has_previous True."""
        elements = [
            {"type": "text", "text": f"seg{i}", "metadata": {}} for i in range(25)
        ]
        page_elements, info = paginate_parse_results(elements, page=3, page_size=10)
        assert len(page_elements) == 5
        assert info["page"] == 3
        assert info["total_pages"] == 3
        assert info["has_next"] is False
        assert info["has_previous"] is True
        assert page_elements[0].text == "seg20"

    def test_pagination_single_page(self) -> None:
        """Single page of results."""
        elements = [
            {"type": "text", "text": "only", "metadata": {}},
        ]
        page_elements, info = paginate_parse_results(elements, page=1, page_size=20)
        assert len(page_elements) == 1
        assert info["total_elements"] == 1
        assert info["total_pages"] == 1
        assert info["has_next"] is False
        assert info["has_previous"] is False

    def test_pagination_empty_list(self) -> None:
        """Empty elements list returns empty page and total_pages 1."""
        page_elements, info = paginate_parse_results([], page=1, page_size=10)
        assert len(page_elements) == 0
        assert info["total_elements"] == 0
        assert info["total_pages"] == 1

    def test_pagination_table_and_figure_types(self) -> None:
        """Mixed types are converted to Parsed*Display models."""
        elements = [
            {"type": "text", "text": "a", "metadata": {}},
            {"type": "table", "html": "<table></table>", "metadata": {}},
            {"type": "figure", "text": "cap", "metadata": {}},
        ]
        page_elements, _ = paginate_parse_results(elements, page=1, page_size=5)
        assert len(page_elements) == 3
        assert page_elements[0].type == "text"
        assert page_elements[0].text == "a"
        assert page_elements[1].type == "table"
        assert page_elements[1].html == "<table></table>"
        assert page_elements[2].type == "figure"
        assert page_elements[2].text == "cap"
