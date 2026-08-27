from __future__ import annotations

import ast
import re
import subprocess
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]

# Distribution names and import names are separate interfaces. Keep the mapping
# explicit so packages such as pydantic-settings are checked without guessing
# their import name from punctuation.
SANDBOX_DISTRIBUTION_IMPORTS = {
    "pydantic": "pydantic",
    "pydantic-settings": "pydantic_settings",
    "cloudpickle": "cloudpickle",
    "mcp": "mcp",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "openpyxl": "openpyxl",
    "fsspec": "fsspec",
}
SANDBOX_DIRECT_REQUIREMENTS = {
    "pydantic": "pydantic>=2.11.7",
    "pydantic-settings": "pydantic-settings",
    "cloudpickle": "cloudpickle>=3.0.0",
    "mcp": "mcp>=1.12.4,<2",
    "pandas": "pandas>=1.3.0",
    "numpy": "numpy>=1.21.0",
    "matplotlib": "matplotlib>=3.5.0",
    "openpyxl": "openpyxl>=3.1.0",
    "fsspec": "fsspec>=2024.0.0",
}


def read_workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text()


def read_repo_file(path: str) -> str:
    return (ROOT / path).read_text()


def pyproject_section(pyproject: str, section_name: str) -> str:
    marker = f"[{section_name}]"
    start = pyproject.index(marker)
    next_section = pyproject.find("\n[", start + len(marker))
    if next_section == -1:
        return pyproject[start:]
    return pyproject[start:next_section]


def test_nightly_build_uses_pep440_package_version() -> None:
    workflow = read_workflow("nightly-build.yml")

    assert 'NIGHTLY_VERSION="nightly-$NIGHTLY_DATE"' in workflow
    assert 'PACKAGE_VERSION="0.0.dev$NIGHTLY_DATE"' in workflow
    assert (
        "XAGENT_VERSION=${{ steps.version-meta.outputs.nightly_version }}" in workflow
    )
    assert (
        "XAGENT_PACKAGE_VERSION=${{ steps.version-meta.outputs.package_version }}"
        in workflow
    )
    assert 'python scripts/write_package_version.py "$PACKAGE_VERSION"' not in workflow
    assert 'echo "package_version=$PACKAGE_VERSION" >> "$GITHUB_OUTPUT"' in workflow
    assert (
        "XAGENT_PACKAGE_VERSION=${{ steps.version-meta.outputs.nightly_version }}"
        not in workflow
    )


def test_release_build_sanitizes_package_version_for_manual_runs() -> None:
    workflow = read_workflow("docker-publish.yml")

    assert 'PACKAGE_VERSION="${RELEASE_VERSION#v}"' in workflow
    assert 'PACKAGE_VERSION="0.0.0+${GITHUB_SHA::12}"' in workflow
    assert (
        "XAGENT_VERSION=${{ steps.version-meta.outputs.release_version }}" in workflow
    )
    assert (
        "XAGENT_PACKAGE_VERSION=${{ steps.version-meta.outputs.package_version }}"
        in workflow
    )
    assert 'python scripts/write_package_version.py "$PACKAGE_VERSION"' not in workflow
    assert 'echo "package_version=$PACKAGE_VERSION" >> "$GITHUB_OUTPUT"' in workflow
    assert (
        "XAGENT_PACKAGE_VERSION=${{ steps.version-meta.outputs.release_version }}"
        not in workflow
    )


def test_backend_dockerfile_applies_package_specific_vcs_version() -> None:
    dockerfile = read_repo_file("docker/Dockerfile.backend")

    assert dockerfile.count('ARG XAGENT_PACKAGE_VERSION="0.0.0+docker"') == 2
    assert (
        dockerfile.count('SETUPTOOLS_SCM_PRETEND_VERSION="${XAGENT_PACKAGE_VERSION}"')
        == 2
    )
    dependency_sync = (
        'SETUPTOOLS_SCM_PRETEND_VERSION="${XAGENT_PACKAGE_VERSION}" \\\n'
        "    VIRTUAL_ENV=/opt/venv uv sync --active --locked --no-dev "
        "--no-install-project --no-editable"
    )
    build_sync = (
        'SETUPTOOLS_SCM_PRETEND_VERSION="${XAGENT_PACKAGE_VERSION}" \\\n'
        "    VIRTUAL_ENV=/opt/venv uv sync --active --locked --no-dev --no-editable"
    )
    assert dependency_sync in dockerfile
    assert build_sync in dockerfile
    assert "COPY .git .git" not in dockerfile


def test_backend_dockerfile_uses_uv_deployment_sync() -> None:
    dockerfile = read_repo_file("docker/Dockerfile.backend")

    assert "uv pip compile" not in dockerfile
    assert "uv pip sync" not in dockerfile
    assert "--no-emit-package xagent" not in dockerfile
    assert dockerfile.count("uv sync") == 2
    assert dockerfile.count("--active") == 2
    assert dockerfile.count("--locked") == 2
    assert dockerfile.count("--no-dev") == 2
    assert dockerfile.count("--no-editable") == 2
    assert dockerfile.count("--group backend-image") == 2
    assert "--torch-backend" not in dockerfile
    assert "--no-install-project" in dockerfile
    assert "COPY pyproject.toml uv.lock README.md ./" in dockerfile
    assert "ENV UV_COMPILE_BYTECODE=1" in dockerfile
    assert "ENV UV_LINK_MODE=copy" in dockerfile
    assert "ENV UV_PYTHON_DOWNLOADS=0" in dockerfile


def test_backend_image_dependencies_are_deployment_group() -> None:
    pyproject = read_repo_file("pyproject.toml")
    optional_dependencies = pyproject_section(
        pyproject, "project.optional-dependencies"
    )
    dependency_groups = pyproject_section(pyproject, "dependency-groups")

    assert "backend-image = [" not in optional_dependencies
    assert "backend-image = [" in dependency_groups
    assert '"torch"' in dependency_groups
    assert '"torchvision"' in dependency_groups


def test_pytorch_cpu_index_is_project_configured_for_uv_sync() -> None:
    pyproject = read_repo_file("pyproject.toml")

    assert 'torch = [{ index = "pytorch-cpu" }]' in pyproject
    assert 'torchvision = [{ index = "pytorch-cpu" }]' in pyproject
    assert 'name = "pytorch-cpu"' in pyproject
    assert 'url = "https://download.pytorch.org/whl/cpu"' in pyproject
    assert "explicit = true" in pyproject


def test_boxlite_is_not_declared_for_linux_aarch64() -> None:
    pyproject = read_repo_file("pyproject.toml")

    assert (
        "\"boxlite>=0.6.0; sys_platform == 'linux' and platform_machine == 'x86_64'\""
        in pyproject
    )
    assert (
        "\"boxlite>=0.6.0; sys_platform == 'darwin' and platform_machine == 'arm64'\""
        in pyproject
    )
    assert (
        "boxlite>=0.6.0; sys_platform == 'linux' and platform_machine == 'aarch64'"
        not in pyproject
    )


def test_publish_script_derives_package_version_from_valid_tags() -> None:
    publish_script = read_repo_file("docker/publish.sh")

    assert 'DEFAULT_PACKAGE_VERSION="${TAG#v}"' in publish_script
    assert (
        'PACKAGE_VERSION="${XAGENT_PACKAGE_VERSION:-${DEFAULT_PACKAGE_VERSION}}"'
        in publish_script
    )
    assert 'DEFAULT_PACKAGE_VERSION="0.0.0+${GIT_COMMIT::12}"' in publish_script
    assert 'XAGENT_VERSION="${XAGENT_VERSION:-${TAG}}"' in publish_script
    assert (
        'python "${REPO_ROOT}/scripts/write_package_version.py"' not in publish_script
    )
    assert '--build-arg "XAGENT_PACKAGE_VERSION=${PACKAGE_VERSION}"' in publish_script


def test_backend_dockerfile_uses_frontend_managed_pptxgenjs() -> None:
    dockerfile = read_repo_file("docker/Dockerfile.backend")
    package_json = read_repo_file("frontend/package.json")

    assert '"pptxgenjs": "4.0.1"' in package_json
    assert "npm install -g pptxgenjs" not in dockerfile
    assert "/usr/lib/node_modules/pptxgenjs" not in dockerfile
    assert 'ENV NODE_PATH="/opt/xagent/frontend/node_modules"' in dockerfile


def test_backend_runtime_keeps_uv_binaries() -> None:
    dockerfile = read_repo_file("docker/Dockerfile.backend")

    assert dockerfile.count("COPY --from=uv /uv /uvx /usr/local/bin/") == 2


def test_backend_package_version_is_vcs_based_for_normal_builds() -> None:
    pyproject = read_repo_file("pyproject.toml")

    assert 'dynamic = ["version"]' in pyproject
    assert 'requires = ["hatchling", "hatch-vcs"]' in pyproject
    assert 'source = "vcs"' in pyproject
    assert 'path = "src/xagent/_version.py"' not in pyproject
    assert not (ROOT / "src" / "xagent" / "_version.py").exists()
    assert not (ROOT / "scripts" / "write_package_version.py").exists()


def test_docker_workflows_pass_package_version_to_backend_build() -> None:
    release_workflow = read_workflow("docker-publish.yml")
    nightly_workflow = read_workflow("nightly-build.yml")

    assert (
        'python scripts/write_package_version.py "$PACKAGE_VERSION"'
        not in release_workflow
    )
    assert (
        'python scripts/write_package_version.py "$PACKAGE_VERSION"'
        not in nightly_workflow
    )
    assert (
        "XAGENT_PACKAGE_VERSION=${{ steps.version-meta.outputs.package_version }}"
        in release_workflow
    )
    assert (
        "XAGENT_PACKAGE_VERSION=${{ steps.version-meta.outputs.package_version }}"
        in nightly_workflow
    )


def test_docker_readme_documents_backend_and_sandbox_lockfile_requirements() -> None:
    readme = read_repo_file("docker/README.md")

    assert "`uv.lock` during the Docker build" in readme
    assert "uv sync --locked" in readme
    assert "`[dependency-groups].sandbox`" in readme
    assert "docker/Dockerfile.sandbox" in readme
    assert "uv.lock` is not copied" not in readme


def test_sandbox_image_dependencies_are_a_dedicated_locked_group() -> None:
    pyproject = tomllib.loads(read_repo_file("pyproject.toml"))

    assert pyproject["dependency-groups"]["sandbox"] == list(
        SANDBOX_DIRECT_REQUIREMENTS.values()
    )


def test_sandbox_group_covers_runtime_requirements_with_compatible_lock() -> None:
    runtime_constants = {
        "src/xagent/core/tools/adapters/vibe/sandboxed_tool/"
        "sandboxed_tool_wrapper.py": "SANDBOX_BASE_DEPENDENCIES",
        "src/xagent/core/tools/adapters/vibe/sandboxed_tool/"
        "sandboxed_mcp_tool_helper.py": "_MCP_SANDBOX_EXTRA_PACKAGES",
    }
    runtime_requirements: list[str] = []
    for path, constant_name in runtime_constants.items():
        module = ast.parse(read_repo_file(path))
        assignment = next(
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == constant_name
                for target in node.targets
            )
        )
        runtime_requirements.extend(ast.literal_eval(assignment.value))

    pyproject = tomllib.loads(read_repo_file("pyproject.toml"))
    sandbox_requirements = {
        canonicalize_name(requirement.name): requirement
        for raw_requirement in pyproject["dependency-groups"]["sandbox"]
        if (requirement := Requirement(raw_requirement))
    }
    lock = tomllib.loads(read_repo_file("uv.lock"))
    locked_versions = {
        canonicalize_name(package["name"]): Version(package["version"])
        for package in lock["package"]
        if "version" in package
    }

    for raw_requirement in runtime_requirements:
        runtime_requirement = Requirement(raw_requirement)
        name = canonicalize_name(runtime_requirement.name)
        assert name in sandbox_requirements
        assert locked_versions[name] in runtime_requirement.specifier


def test_backend_mcp_dependency_excludes_2x() -> None:
    # Companion to test_sandbox_group_covers_runtime_requirements_with_compatible_lock
    # above, which only guards the sandbox group's mcp bound: mcp 2.0 removed
    # mcp.client.streamable_http.streamablehttp_client, which the backend's own
    # sessions.py imports eagerly, so [project].dependencies must independently
    # keep mcp below 2.x even if that guard alone were ever satisfied.
    pyproject = tomllib.loads(read_repo_file("pyproject.toml"))
    backend_requirements = {
        canonicalize_name((requirement := Requirement(raw)).name): requirement
        for raw in pyproject["project"]["dependencies"]
    }

    mcp_requirement = backend_requirements[canonicalize_name("mcp")]
    assert Version("2.0.0") not in mcp_requirement.specifier


def test_sandbox_dockerfile_installs_locked_group_and_smoke_tests_imports() -> None:
    dockerfile = read_repo_file("docker/Dockerfile.sandbox")

    assert "FROM python:3.11-slim AS sandbox-requirements" in dockerfile
    assert "COPY pyproject.toml uv.lock ./" in dockerfile
    assert (
        "uv export --quiet --locked --only-group sandbox --no-emit-project"
        in dockerfile
    )
    assert "FROM node:22-slim AS sandbox" in dockerfile
    runtime_stage = dockerfile.split("FROM node:22-slim AS sandbox\n", maxsplit=1)[1]
    assert "COPY pyproject.toml uv.lock ./" not in runtime_stage
    assert "COPY --from=sandbox-requirements /sandbox-requirements.txt" in runtime_stage
    assert "uv pip install --system --break-system-packages" in runtime_stage
    assert "docker/sandbox-requirements.txt" not in dockerfile
    assert not (ROOT / "docker" / "sandbox-requirements.txt").exists()
    assert "USER sandbox" in runtime_stage


def test_sandbox_runtime_keeps_and_smoke_tests_uvx() -> None:
    dockerfile = read_repo_file("docker/Dockerfile.sandbox")
    runtime_stage = dockerfile.split("FROM node:22-slim AS sandbox\n", maxsplit=1)[1]

    assert "COPY --from=uv /uv /uvx /usr/local/bin/" in runtime_stage
    assert "uvx --version" in runtime_stage


def test_sandbox_export_is_locked_without_managed_python_downloads() -> None:
    dockerfile = read_repo_file("docker/Dockerfile.sandbox")
    requirements_stage = dockerfile.split(
        "FROM python:3.11-slim AS sandbox-requirements\n", maxsplit=1
    )[1].split("FROM node:22-slim AS sandbox\n", maxsplit=1)[0]

    assert "ENV UV_PYTHON_DOWNLOADS=0" in requirements_stage
    assert "uv export --quiet --locked --only-group sandbox --no-emit-project" in (
        requirements_stage
    )


def test_chrome_devtools_mcp_pin_matches_across_dockerfiles_and_registry() -> None:
    # builtin_mcp_registry.py is the single source of truth for the version
    # end users' MCP calls actually run; both Dockerfiles independently warm
    # an npx cache for "the same" pin so npx resolves offline instead of
    # hitting the npm registry on every sandboxed/backend-hosted launch (see
    # Dockerfile.sandbox's INSTALL_CHROME block and its npx warm-up comment).
    # A version bumped in the registry but missed in either warm-up command
    # would silently leave that path's cache cold for the *new* version
    # while still reporting success, not fail loudly -- this pins all three
    # together so a future bump can't drift one file behind the others.
    registry = read_repo_file("src/xagent/web/builtin_mcp_registry.py")
    pin_match = re.search(r'"chrome-devtools-mcp@([\w.\-]+)"', registry)
    assert pin_match, "chrome-devtools-mcp pin not found in builtin_mcp_registry.py"
    pinned_version = pin_match.group(1)

    # Extracts just the version each Dockerfile's own npx warm-up command
    # resolves against, rather than requiring an exact-substring match of
    # the whole npx invocation -- a future edit that legitimately reorders
    # or adds flags around the same pinned version shouldn't fail this
    # test, only an actual version mismatch should.
    for dockerfile_path in ("docker/Dockerfile.backend", "docker/Dockerfile.sandbox"):
        dockerfile = read_repo_file(dockerfile_path)
        dockerfile_match = re.search(
            r"npx\b[^\n]*\bchrome-devtools-mcp@([\w.\-]+)", dockerfile
        )
        assert dockerfile_match, (
            f"{dockerfile_path} has no npx chrome-devtools-mcp warm-up command"
        )
        assert dockerfile_match.group(1) == pinned_version, (
            f"{dockerfile_path} warms the npx cache for "
            f"chrome-devtools-mcp@{dockerfile_match.group(1)}, but "
            f"builtin_mcp_registry.py pins chrome-devtools-mcp@{pinned_version} -- "
            "update the warm-up command to match"
        )


def test_sandbox_uv_install_uses_buildkit_cache() -> None:
    dockerfile = read_repo_file("docker/Dockerfile.sandbox")
    runtime_stage = dockerfile.split("FROM node:22-slim AS sandbox\n", maxsplit=1)[1]

    assert "--mount=type=cache,target=/root/.cache/uv" in runtime_stage
    assert "--no-cache" not in runtime_stage


def test_sandbox_direct_distributions_have_explicit_smoke_imports() -> None:
    pyproject = tomllib.loads(read_repo_file("pyproject.toml"))
    direct_requirements = pyproject["dependency-groups"]["sandbox"]
    dockerfile = read_repo_file("docker/Dockerfile.sandbox")
    runtime_stage = dockerfile.split("FROM node:22-slim AS sandbox\n", maxsplit=1)[1]
    smoke_command = re.search(r'python -c "([^"]+)"', runtime_stage)

    assert SANDBOX_DIRECT_REQUIREMENTS.keys() == SANDBOX_DISTRIBUTION_IMPORTS.keys()
    assert direct_requirements == list(SANDBOX_DIRECT_REQUIREMENTS.values())
    assert smoke_command is not None
    imported_modules = {
        alias.name
        for node in ast.walk(ast.parse(smoke_command.group(1)))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imported_modules == set(SANDBOX_DISTRIBUTION_IMPORTS.values())


def test_uv_export_locked_sandbox_group_contains_direct_distributions() -> None:
    result = subprocess.run(
        [
            "uv",
            "export",
            "--quiet",
            "--locked",
            "--only-group",
            "sandbox",
            "--no-emit-project",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    for distribution in SANDBOX_DISTRIBUTION_IMPORTS:
        assert re.search(rf"^{re.escape(distribution)}==", result.stdout, re.MULTILINE)
