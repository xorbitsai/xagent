"""Security lower bounds must apply to wheel installs, not only uv.lock."""

import tomllib
from pathlib import Path
from typing import Iterator

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]


def _requirements() -> Iterator[Requirement]:
    with (ROOT / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)
    groups = [
        project["project"]["dependencies"],
        *project["project"]["optional-dependencies"].values(),
        *project["dependency-groups"].values(),
    ]
    for group in groups:
        for item in group:
            if isinstance(item, str):
                yield Requirement(item)


@pytest.mark.parametrize(
    ("name", "floor", "old"),
    [
        ("aiohttp", "3.14.5", "3.13.3"),
        ("asyncssh", "2.24.1", "2.23.1"),
        ("boxlite", "0.10.7", "0.7.5"),
        ("cairosvg", "2.9.1", "2.9.0"),
        ("cryptography", "50.0.2", "46.0.5"),
        ("docling", "2.140.0", "2.75.0"),
        ("fastapi", "0.143.3", "0.135.1"),
        ("filelock", "3.32.7", "3.15.4"),
        ("fsspec", "2026.9.0", "2026.2.0"),
        ("idna", "3.20", "3.11"),
        ("json-repair", "0.60.1", "0.35.0"),
        ("langchain-community", "0.4.2", "0.3.21"),
        ("langchain-core", "1.6.12", "1.3.2"),
        ("langchain-openai", "1.1.14", "0.3.35"),
        ("mcp", "1.30.0", "1.19.0"),
        ("pillow", "12.3.0", "12.1.1"),
        ("pydantic-settings", "2.16.0", "2.13.1"),
        ("pypdf", "6.21.1", "6.0.0"),
        ("pytest", "9.1.1", "9.0.2"),
        ("python-dotenv", "1.2.4", "1.0.1"),
        ("requests", "2.34.2", "2.31.0"),
        ("starlette", "1.8.0", "0.46.2"),
        ("unstructured", "0.27.28", "0.18.32"),
    ],
)
def test_all_declarations_keep_security_lower_bound(name, floor, old):
    declarations = [
        req for req in _requirements() if canonicalize_name(req.name) == name
    ]
    assert declarations, f"Missing direct security requirement for {name}"
    for req in declarations:
        assert req.specifier.contains(floor), str(req)
        assert not req.specifier.contains(old), str(req)
        assert any(
            spec.operator == ">=" and spec.version == floor for spec in req.specifier
        ), str(req)


def test_unused_top_level_langchain_is_not_a_runtime_requirement():
    assert all(canonicalize_name(req.name) != "langchain" for req in _requirements())
