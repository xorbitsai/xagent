"""The structured output format as the web layer sees it."""

from __future__ import annotations

from xagent.core.agent import structured_output
from xagent.web.services import structured_output_projection


def test_the_result_key_is_the_one_the_projection_reads() -> None:
    assert (
        structured_output.STRUCTURED_OUTPUT_RESULT_KEY
        == structured_output_projection._RESULT_KEY
    )
