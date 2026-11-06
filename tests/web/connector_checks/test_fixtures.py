"""Fixture keys: values come from the identity's resolver, never from the caller."""

from __future__ import annotations

import pytest

from xagent.web.connector_checks.fixtures import (
    FixtureMissing,
    FixtureRef,
    FixtureValue,
    resolve_fixtures,
)


class _Resolver:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values
        self.asked: list[str] = []

    def resolve_fixture(self, key: str) -> FixtureValue | None:
        self.asked.append(key)
        return self.values.get(key)  # type: ignore[return-value]


def test_refs_are_replaced_at_any_depth_and_literals_pass_through() -> None:
    resolver = _Resolver(
        {"google-docs.document_id": "doc-1", "google-sheets.range_rows": 3}
    )
    arguments = {
        "document_id": FixtureRef("google-docs.document_id"),
        "page_size": 1,
        "ranges": [{"rows": FixtureRef("google-sheets.range_rows")}, "A1"],
        "options": ("x", FixtureRef("google-docs.document_id")),
    }

    resolved = resolve_fixtures(arguments, resolver)

    assert resolved == {
        "document_id": "doc-1",
        "page_size": 1,
        "ranges": [{"rows": 3}, "A1"],
        "options": ["x", "doc-1"],
    }


def test_a_bare_ref_resolves_to_its_value() -> None:
    assert (
        resolve_fixtures(
            FixtureRef("slack.workspace_id"), _Resolver({"slack.workspace_id": "T1"})
        )
        == "T1"
    )


def test_an_unresolved_key_raises_naming_only_the_key() -> None:
    resolver = _Resolver({"word.file_path": "secret/path.docx"})

    with pytest.raises(FixtureMissing) as excinfo:
        resolve_fixtures(
            {"a": FixtureRef("word.file_path"), "b": FixtureRef("excel.file_path")},
            resolver,
        )

    assert excinfo.value.key == "excel.file_path"
    assert "secret/path.docx" not in str(excinfo.value)


def test_a_resolver_value_of_the_wrong_type_is_refused_without_echoing_it() -> None:
    resolver = _Resolver({"excel.file_path": {"token": "abc"}})

    with pytest.raises(TypeError) as excinfo:
        resolve_fixtures(FixtureRef("excel.file_path"), resolver)

    assert "excel.file_path" in str(excinfo.value)
    assert "abc" not in str(excinfo.value)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_non_finite_number_is_refused(value: float) -> None:
    with pytest.raises(TypeError):
        resolve_fixtures(
            FixtureRef("excel.row_count"), _Resolver({"excel.row_count": value})
        )


@pytest.mark.parametrize(
    "key",
    ["google-docs.document_id", "slack.workspace_id", "jira.seed.project_key"],
)
def test_valid_keys_are_namespace_dot_snake_case_names(key: str) -> None:
    assert FixtureRef(key).key == key


@pytest.mark.parametrize(
    "key",
    [
        "",
        "document_id",
        "Google.doc",
        "google-docs.",
        ".doc",
        "google docs.id",
        "g.Doc-Id",
        "google-docs." + "a" * 69,
    ],
)
def test_malformed_keys_are_rejected(key: str) -> None:
    with pytest.raises(ValueError):
        FixtureRef(key)
