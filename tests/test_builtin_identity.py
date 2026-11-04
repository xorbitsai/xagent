"""Unit tests for the builtin catalog identity helpers."""

from __future__ import annotations

from typing import Any

import pytest

from xagent.builtin_identity import owned_catalog_marker

_WORD_MARKER = {"registry": "xagent", "app_id": "word", "version": 1}


def _app_info(launch_config: Any, *, app_id: str = "word") -> dict[str, Any]:
    return {"id": app_id, "launch_config": launch_config}


@pytest.mark.parametrize(
    "marker",
    [
        pytest.param(_WORD_MARKER, id="catalog-marker"),
        pytest.param({**_WORD_MARKER, "version": 7}, id="other-version"),
        pytest.param({"registry": "xagent", "app_id": "word"}, id="no-version"),
    ],
)
def test_owned_catalog_marker_returns_the_apps_own_marker(marker) -> None:
    app_info = _app_info({"command": "unused", "builtin_provenance": marker})

    assert owned_catalog_marker(app_info) is marker


@pytest.mark.parametrize(
    "app_info",
    [
        pytest.param({"id": "word"}, id="no-launch-config"),
        pytest.param(_app_info(None), id="null-launch-config"),
        pytest.param(_app_info([_WORD_MARKER]), id="list-launch-config"),
        pytest.param(_app_info({}), id="no-marker"),
        pytest.param(_app_info({"builtin_provenance": None}), id="null-marker"),
        pytest.param(
            _app_info({"builtin_provenance": "xagent:word"}), id="string-marker"
        ),
        pytest.param(
            _app_info({"builtin_provenance": ["xagent", "word"]}), id="list-marker"
        ),
        pytest.param(
            _app_info({"builtin_provenance": {"app_id": "word", "version": 1}}),
            id="no-registry",
        ),
        pytest.param(
            _app_info({"builtin_provenance": {**_WORD_MARKER, "registry": ""}}),
            id="empty-registry",
        ),
        pytest.param(
            _app_info({"builtin_provenance": {**_WORD_MARKER, "registry": "custom"}}),
            id="other-registry",
        ),
        pytest.param(
            _app_info({"builtin_provenance": {**_WORD_MARKER, "app_id": "excel"}}),
            id="other-app",
        ),
        pytest.param(
            _app_info({"builtin_provenance": _WORD_MARKER}, app_id="excel"),
            id="marker-copied-to-another-app",
        ),
    ],
)
def test_owned_catalog_marker_ignores_markers_the_app_does_not_own(app_info) -> None:
    assert owned_catalog_marker(app_info) is None
