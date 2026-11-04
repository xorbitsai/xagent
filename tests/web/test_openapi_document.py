"""The served OpenAPI document must be self-contained.

A route that parses its own request body declares its ``requestBody`` by hand,
which takes it outside the framework's schema inference: nothing walks the
model, so its definitions never reach ``components.schemas``. A hand-declared
body that references one is then a dangling pointer in the assembled document,
and a client generator or a reader of ``/openapi.json`` cannot resolve it.
"""

from __future__ import annotations

from typing import Any, Iterator

import pytest

from xagent.web.app import app
from xagent.web.services.global_memory_embedding_authority import (
    AuthorityConfiguration,
    CredentialSource,
)

AUTHORITY_PATH = "/api/admin/memory/embedding-authority"


@pytest.fixture(scope="module")
def document() -> dict[str, Any]:
    return app.openapi()


def _refs(node: Any) -> Iterator[str]:
    """Every ``$ref`` in the document, wherever it is nested."""
    if isinstance(node, dict):
        target = node.get("$ref")
        if isinstance(target, str):
            yield target
        for value in node.values():
            yield from _refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from _refs(value)


def _resolve(document: dict[str, Any], ref: str) -> Any:
    assert ref.startswith("#/"), f"non-local $ref is not resolvable: {ref}"
    node: Any = document
    for segment in ref[2:].split("/"):
        segment = segment.replace("~1", "/").replace("~0", "~")
        assert isinstance(node, dict) and segment in node, (
            f"{ref} does not resolve: no {segment!r}"
        )
        node = node[segment]
    return node


def test_every_ref_in_the_document_resolves(document):
    unresolved = []
    for ref in set(_refs(document)):
        try:
            _resolve(document, ref)
        except AssertionError as failure:
            unresolved.append(str(failure))
    assert not unresolved, "\n".join(sorted(unresolved))


def test_the_authority_request_body_resolves_to_its_own_fields(document):
    body = document["paths"][AUTHORITY_PATH]["put"]["requestBody"]
    schema = _resolve(document, body["content"]["application/json"]["schema"]["$ref"])

    assert set(schema["properties"]) == set(AuthorityConfiguration.model_fields)
    # The nested definition is the point: this is the reference that used to
    # dangle, because a standalone schema's "$defs" has no root to hang from.
    credential_source = _resolve(
        document, schema["properties"]["credential_source"]["$ref"]
    )
    assert set(credential_source["enum"]) == {
        source.value for source in CredentialSource
    }


def test_the_document_carries_no_root_defs(document):
    """``$defs`` is a schema-local keyword; nothing may reference it as a root."""
    assert "$defs" not in document
    assert not [ref for ref in _refs(document) if ref.startswith("#/$defs/")]
