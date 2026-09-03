from __future__ import annotations

import pytest

from xagent.web.api.chat import _connect_apps_interactive_for_task


@pytest.mark.parametrize(
    ("source", "channel_id", "expected"),
    [
        # Web chat: source is None/"internal" and there's no IM channel.
        (None, None, True),
        ("internal", None, True),
        # IM-channel bot tasks share source=None/"internal" with web chat --
        # only channel_id tells them apart, and that surface can't render
        # the connect_apps card.
        (None, 5, False),
        ("internal", 5, False),
        # A2A, the v1 SDK, and public/shared-link runs are excluded by
        # source alone.
        ("a2a", None, False),
        ("sdk", None, False),
        ("external", None, False),
        ("widget", None, False),
        ("shared_link", None, False),
    ],
)
def test_connect_apps_interactive_for_task(source, channel_id, expected):
    assert (
        _connect_apps_interactive_for_task(source=source, channel_id=channel_id)
        is expected
    )
