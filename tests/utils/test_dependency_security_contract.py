"""Security lower bounds must apply to wheel installs, not only uv.lock."""

import tomllib
from pathlib import Path
from typing import Iterator

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

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
    ("name", "floor"),
    [
        ("aiohttp", "3.14.5"),
        ("asyncssh", "2.24.1"),
        ("boxlite", "0.10.7"),
        ("cairosvg", "2.9.1"),
        ("cryptography", "50.0.2"),
        ("docling", "2.140.0"),
        ("fastapi", "0.143.3"),
        ("filelock", "3.32.7"),
        ("fsspec", "2026.9.0"),
        ("idna", "3.20"),
        ("json-repair", "0.64.0"),
        ("langchain-community", "0.4.2"),
        ("langchain-core", "1.6.12"),
        ("langchain-openai", "1.1.14"),
        ("mcp", "1.30.0"),
        ("pillow", "12.3.0"),
        ("pydantic-settings", "2.16.0"),
        ("pypdf", "6.21.1"),
        ("pytest", "9.1.1"),
        ("python-dotenv", "1.2.4"),
        ("requests", "2.34.2"),
        ("starlette", "1.8.0"),
        ("unstructured", "0.27.28"),
    ],
)
def test_all_declarations_keep_security_lower_bound(name, floor):
    declarations = [
        req for req in _requirements() if canonicalize_name(req.name) == name
    ]
    assert declarations, f"Missing direct security requirement for {name}"
    for req in declarations:
        assert _has_reviewed_minimum(req.specifier, floor), str(req)


def _has_reviewed_minimum(specifier: SpecifierSet, floor: str) -> bool:
    # These are reviewed deployment baselines, not a CVE/advisory database.
    # A higher minimum or exact audited pin must not fail a change detector.
    return any(
        spec.operator in {">=", ">", "=="}
        and not spec.version.endswith(".*")
        and Version(spec.version) >= Version(floor)
        for spec in specifier
    )


@pytest.mark.parametrize(
    "specifier,allowed",
    [
        (">=1.30.0,<2", True),
        (">=1.31.0,<2", True),
        ("==1.31.0", True),
        (">=1.29.0,<2", False),
        ("==1.29.0", False),
        ("<2", False),
    ],
)
def test_reviewed_minimum_accepts_stronger_floors_and_pins(specifier, allowed):
    assert _has_reviewed_minimum(SpecifierSet(specifier), "1.30.0") is allowed


def test_runtime_sandbox_installers_keep_reviewed_floors():
    from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper import (
        _MCP_SANDBOX_EXTRA_PACKAGES,
    )
    from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
        SANDBOX_BASE_DEPENDENCIES,
    )

    requirements = {
        canonicalize_name(req.name): req
        for item in [*SANDBOX_BASE_DEPENDENCIES, *_MCP_SANDBOX_EXTRA_PACKAGES]
        for req in [Requirement(item)]
    }
    for name, floor in [("mcp", "1.30.0"), ("pydantic-settings", "2.16.0")]:
        assert _has_reviewed_minimum(requirements[name].specifier, floor)


def test_unused_top_level_langchain_is_not_a_runtime_requirement():
    assert all(canonicalize_name(req.name) != "langchain" for req in _requirements())
