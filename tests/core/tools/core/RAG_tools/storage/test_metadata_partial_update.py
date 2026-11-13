"""Partial collection_metadata writes: only the named columns change."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store


def _save(**kw: Any) -> None:
    asyncio.run(get_metadata_store().save_collection(CollectionInfo(**kw)))


def _row(name: str = "kb") -> CollectionInfo:
    return asyncio.run(get_metadata_store().get_collection(name))


def test_partial_update_leaves_other_columns_alone() -> None:
    _save(
        name="kb",
        embedding_model_id="model-new",
        embedding_dimension=8,
        documents=1,
        extra_metadata={"keep": "me"},
    )

    asyncio.run(
        get_metadata_store().update_collection_fields(
            CollectionInfo(name="kb", documents=7), ("documents",)
        )
    )

    row = _row()
    assert row.documents == 7
    assert (row.embedding_model_id, row.embedding_dimension) == ("model-new", 8)
    assert row.extra_metadata == {"keep": "me"}


def test_require_uninitialized_binds_an_unbound_row() -> None:
    _save(name="kb")

    applied = asyncio.run(
        get_metadata_store().update_collection_fields(
            CollectionInfo(
                name="kb", embedding_model_id="model-new", embedding_dimension=8
            ),
            ("embedding_model_id", "embedding_dimension"),
            require_uninitialized=True,
        )
    )

    assert applied is True
    assert (x := _row()).embedding_model_id == "model-new"
    assert x.embedding_dimension == 8


def test_require_uninitialized_never_overwrites_a_binding() -> None:
    _save(name="kb", embedding_model_id="model-new", embedding_dimension=8)

    applied = asyncio.run(
        get_metadata_store().update_collection_fields(
            CollectionInfo(
                name="kb", embedding_model_id="model-old", embedding_dimension=4
            ),
            ("embedding_model_id", "embedding_dimension"),
            require_uninitialized=True,
        )
    )

    assert applied is False
    assert (row := _row()).embedding_model_id == "model-new"
    assert row.embedding_dimension == 8


def test_expected_model_id_guards_a_rewrite() -> None:
    _save(name="kb", embedding_model_id="alias", embedding_dimension=4)

    applied = asyncio.run(
        get_metadata_store().update_collection_fields(
            CollectionInfo(
                name="kb", embedding_model_id="model-old", embedding_dimension=4
            ),
            ("embedding_model_id", "embedding_dimension"),
            expected_model_id="other",
        )
    )
    assert applied is False
    assert _row().embedding_model_id == "alias"

    applied = asyncio.run(
        get_metadata_store().update_collection_fields(
            CollectionInfo(
                name="kb", embedding_model_id="model-old", embedding_dimension=4
            ),
            ("embedding_model_id", "embedding_dimension"),
            expected_model_id="alias",
        )
    )
    assert applied is True
    assert _row().embedding_model_id == "model-old"


def test_save_if_absent_inserts_a_new_row() -> None:
    inserted = asyncio.run(
        get_metadata_store().save_collection_if_absent(
            CollectionInfo(name="kb", documents=3)
        )
    )

    assert inserted is True
    assert _row().documents == 3


def test_save_if_absent_never_overwrites_an_existing_row() -> None:
    _save(name="kb", embedding_model_id="model-new", embedding_dimension=8, documents=2)

    inserted = asyncio.run(
        get_metadata_store().save_collection_if_absent(
            CollectionInfo(
                name="kb", embedding_model_id="model-old", embedding_dimension=4
            )
        )
    )

    assert inserted is False
    row = _row()
    assert (row.embedding_model_id, row.embedding_dimension) == ("model-new", 8)
    assert row.documents == 2


def test_require_uninitialized_repairs_a_half_bound_row() -> None:
    store = get_metadata_store()
    asyncio.run(
        store.save_collection_if_absent(
            CollectionInfo(name="kb", embedding_model_id="model-old")
        )
    )

    applied = asyncio.run(
        store.update_collection_fields(
            CollectionInfo(
                name="kb", embedding_model_id="model-new", embedding_dimension=8
            ),
            ("embedding_model_id", "embedding_dimension"),
            require_uninitialized=True,
        )
    )

    assert applied is True
    row = _row()
    assert (row.embedding_model_id, row.embedding_dimension) == ("model-new", 8)


def test_missing_row_raises_unless_require_uninitialized() -> None:
    with pytest.raises(ValueError, match="not found"):
        asyncio.run(
            get_metadata_store().update_collection_fields(
                CollectionInfo(name="ghost", documents=1), ("documents",)
            )
        )

    applied = asyncio.run(
        get_metadata_store().update_collection_fields(
            CollectionInfo(name="ghost", embedding_model_id="m", embedding_dimension=1),
            ("embedding_model_id", "embedding_dimension"),
            require_uninitialized=True,
        )
    )
    assert applied is False


def test_both_guards_are_rejected() -> None:
    _save(name="kb")

    with pytest.raises(ValueError, match="not both"):
        asyncio.run(
            get_metadata_store().update_collection_fields(
                CollectionInfo(
                    name="kb", embedding_model_id="m", embedding_dimension=1
                ),
                ("embedding_model_id", "embedding_dimension"),
                require_uninitialized=True,
                expected_model_id="m",
            )
        )


def test_unknown_or_key_fields_are_rejected() -> None:
    _save(name="kb")

    for field in ("name", "not_a_column"):
        with pytest.raises(ValueError, match="Cannot update"):
            asyncio.run(
                get_metadata_store().update_collection_fields(
                    CollectionInfo(name="kb", documents=1), (field,)
                )
            )

    assert _row().documents == 0
