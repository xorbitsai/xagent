from copy import deepcopy

import pytest

from xagent.core.agent.transcript import build_assistant_transcript_content
from xagent.web.services.assistant_question_display import question_content_for_display


@pytest.mark.parametrize(
    "interaction",
    [
        {"type": "text_input", "label": "Owner", "placeholder": "Team name"},
        {"type": "number_input", "label": "Threshold", "min": 1, "max": 100},
        {"type": "confirm", "label": "Send to the team?", "default": False},
        {"type": "file_upload", "label": "Sample", "accept": [".csv"]},
        {
            "type": "select_one",
            "label": "Frequency",
            "options": [{"value": "weekly", "label": "Every week"}],
        },
        {
            "type": "select_multiple",
            "label": "Frequency",
            "options": [{"value": "weekly", "label": "Every week"}],
        },
    ],
)
def test_only_generated_appendix_is_removed(interaction):
    interactions = [interaction]
    before = deepcopy(interactions)
    introduction = "Please help me configure the report."
    transcript = build_assistant_transcript_content(introduction, interactions)

    assert question_content_for_display(transcript, interactions) == introduction
    assert interactions == before
    assert "Please answer the following questions:" in transcript


@pytest.mark.parametrize(
    "interaction",
    [
        {"type": "text_input", "label": "Owner", "placeholder": "Team name \t"},
        {"type": "number_input", "label": "Threshold \t"},
        {
            "type": "select_one",
            "label": "Frequency",
            "options": [{"value": "weekly", "label": "Every week \t"}],
        },
        {
            "type": "select_multiple",
            "label": "Frequency",
            "options": [{"value": "weekly", "label": "Every week \t"}],
        },
    ],
)
@pytest.mark.parametrize("strip_persisted_content", [False, True])
def test_generated_appendix_matches_persistence_whitespace_normalization(
    interaction, strip_persisted_content
):
    interactions = [interaction]
    introduction = "Please configure the report."
    content = build_assistant_transcript_content(introduction, interactions)
    assert content != content.rstrip()
    if strip_persisted_content:
        content = content.strip()

    assert question_content_for_display(content, interactions) == introduction


def test_keeps_authored_text_even_when_it_contains_the_same_question_list():
    interactions = [{"type": "text_input", "label": "Owner"}]
    authored = build_assistant_transcript_content(
        "An example, quoted verbatim:", interactions
    )
    transcript = build_assistant_transcript_content(authored, interactions)

    assert question_content_for_display(transcript, interactions) == authored
    assert (
        question_content_for_display("A plain question", interactions)
        == "A plain question"
    )
    assert (
        question_content_for_display(transcript + "\nOther instructions", interactions)
        == transcript + "\nOther instructions"
    )


@pytest.mark.parametrize(
    "interactions",
    [
        None,
        [],
        "not a form",
        [None],
        [{"type": "unknown"}],
        [{"type": "confirm", "label": ["not text"]}],
        [{"type": "select_one", "options": []}],
        [{"type": "select_one", "options": [{"value": 1, "label": "One"}]}],
        [{"type": "file_upload", "accept": 42}],
        [{"type": "text_input", "label": "Owner"}, {"type": "unknown"}],
    ],
)
def test_preserves_fallback_for_absent_unknown_or_malformed_controls(interactions):
    try:
        content = build_assistant_transcript_content("Question", interactions)
    except (TypeError, ValueError, AttributeError):
        content = (
            "Question\n\n\nPlease answer the following questions:\n- Owner: text input"
        )
    assert question_content_for_display(content, interactions) == content


def test_mismatched_controls_and_empty_introduction_preserve_fallback():
    interactions = [{"type": "text_input", "label": "Owner"}]
    transcript = build_assistant_transcript_content("Question", interactions)
    assert (
        question_content_for_display(
            transcript, [{"type": "text_input", "label": "Different question"}]
        )
        == transcript
    )
    empty_intro = build_assistant_transcript_content("", interactions).strip()
    assert question_content_for_display(empty_intro, interactions) == empty_intro
