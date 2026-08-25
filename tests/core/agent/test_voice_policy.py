"""Unit tests for apply_output_voice's own contract.

Tested here, once, rather than only indirectly through every call site:
review found the scoping caveat below had been added independently at
two call sites and was missing from a third that could reach the same
create_agent/update_agent-style tools (PR #1612's M3 finding) - baking
it into apply_output_voice itself means every current and future caller
gets it for free, and this is the one place that actually needs to prove
it's always present.
"""

from xagent.core.agent.voice_policy import VALID_VOICES, apply_output_voice


def test_apply_output_voice_is_a_noop_without_a_voice():
    assert apply_output_voice("Base prompt.", None) == "Base prompt."
    assert apply_output_voice(None, None) is None


def test_apply_output_voice_is_a_noop_for_an_unrecognized_voice():
    """The persisted JSON has no nested-type constraint, so a corrupted
    or hand-edited value could hold anything here."""
    assert apply_output_voice("Base prompt.", "shouting") == "Base prompt."
    assert apply_output_voice("Base prompt.", ["warm"]) == "Base prompt."
    assert apply_output_voice("Base prompt.", {"voice": "warm"}) == "Base prompt."


def test_apply_output_voice_always_pairs_the_instruction_with_the_scope_caveat():
    for voice in VALID_VOICES:
        result = apply_output_voice("Base prompt.", voice)
        assert result is not None
        assert result.startswith("Base prompt.\n\n## OUTPUT VOICE\n")
        assert "final natural-language reply" in result
        assert "persisted as configuration" in result
        # The instruction always precedes the caveat, not the reverse.
        assert result.index("## OUTPUT VOICE") < result.index(
            "final natural-language reply"
        )


def test_apply_output_voice_with_no_base_prompt_still_includes_the_caveat():
    result = apply_output_voice(None, "warm")
    assert result is not None
    assert result.startswith("## OUTPUT VOICE\n")
    assert "final natural-language reply" in result
