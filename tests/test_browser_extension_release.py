from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_browser_extension_ci_checks_out_xagent_tags() -> None:
    workflow = (REPOSITORY_ROOT / ".github/workflows/ci.yml").read_text()
    job = workflow.split("  browser-extension-build:", 1)[1].split("  ci-summary:", 1)[
        0
    ]

    assert "fetch-depth: 0" in job
    assert "npm run check" in job


def test_xagent_release_uploads_matching_browser_extension() -> None:
    workflow = (REPOSITORY_ROOT / ".github/workflows/pypi-publish.yml").read_text()
    job = workflow.split("  browser-extension:", 1)[1]

    assert "XAGENT_VERSION: ${{ github.event.release.tag_name }}" in job
    assert "npm run check" in job
    assert 'gh release upload "$XAGENT_VERSION"' in job
    assert "artifacts/xagent-browser-relay-*" in job
