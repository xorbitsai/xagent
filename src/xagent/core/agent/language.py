"""Prompt snippets for user-facing response language, plus the
checkpoint migration that keeps only a caller-provided language label."""

import re
from dataclasses import dataclass
from typing import Any, Literal

OUTPUT_LANGUAGE_METADATA_KEY = "output_language"
OUTPUT_LANGUAGE_SOURCE_METADATA_KEY = "output_language_source"
OUTPUT_LANGUAGE_SOURCE_PLAN = "dag_plan"
REQUEST_CONTEXT_METADATA_KEY = "request_context"

_ALLOWED_RESPONSE_LANGUAGE_LABELS = frozenset(
    {
        "Afrikaans",
        "Arabic",
        "Basque",
        "Bengali",
        "Brazilian Portuguese",
        "Bulgarian",
        "Cantonese",
        "Catalan",
        "Chinese",
        "Croatian",
        "Czech",
        "Danish",
        "Dutch",
        "English",
        "Estonian",
        "European Portuguese",
        "Farsi",
        "Filipino",
        "Finnish",
        "French",
        "Galician",
        "German",
        "Greek",
        "Gujarati",
        "Hebrew",
        "Hindi",
        "Hungarian",
        "Icelandic",
        "Indonesian",
        "Irish",
        "Italian",
        "Japanese",
        "Kannada",
        "Korean",
        "Latvian",
        "Lithuanian",
        "Malay",
        "Malayalam",
        "Mandarin Chinese",
        "Marathi",
        "Norwegian",
        "Persian",
        "Polish",
        "Portuguese",
        "Punjabi",
        "Romanian",
        "Russian",
        "Serbian",
        "Simplified Chinese",
        "Slovak",
        "Slovenian",
        "Spanish",
        "Swahili",
        "Swedish",
        "Tagalog",
        "Tamil",
        "Telugu",
        "Thai",
        "Traditional Chinese",
        "Turkish",
        "Ukrainian",
        "Urdu",
        "Vietnamese",
        "Welsh",
    }
)
_LANGUAGE_LABEL_BY_KEY = {
    label.casefold(): label for label in _ALLOWED_RESPONSE_LANGUAGE_LABELS
}
_LANGUAGE_LABEL_ALIASES = {
    "cn": "Chinese",
    "en": "English",
    "en-us": "English",
    "en_us": "English",
    "en-gb": "English",
    "en_gb": "English",
    "es": "Spanish",
    "español": "Spanish",
    "fr": "French",
    "français": "French",
    "pt": "Portuguese",
    "pt-br": "Brazilian Portuguese",
    "pt_br": "Brazilian Portuguese",
    "português": "Portuguese",
    "zh": "Chinese",
    "zh-cn": "Simplified Chinese",
    "zh_cn": "Simplified Chinese",
    "zh-hans": "Simplified Chinese",
    "zh_hans": "Simplified Chinese",
    "zh-hant": "Traditional Chinese",
    "zh_hant": "Traditional Chinese",
    "中文": "Chinese",
    "简体中文": "Simplified Chinese",
    "繁體中文": "Traditional Chinese",
}

_LATIN_SCRIPT_RESPONSE_LANGUAGES = frozenset(
    {
        "Afrikaans",
        "Basque",
        "Brazilian Portuguese",
        "Catalan",
        "Croatian",
        "Czech",
        "Danish",
        "Dutch",
        "English",
        "Estonian",
        "European Portuguese",
        "Filipino",
        "Finnish",
        "French",
        "Galician",
        "German",
        "Hungarian",
        "Icelandic",
        "Indonesian",
        "Irish",
        "Italian",
        "Latvian",
        "Lithuanian",
        "Malay",
        "Norwegian",
        "Polish",
        "Portuguese",
        "Romanian",
        "Slovak",
        "Slovenian",
        "Spanish",
        "Swahili",
        "Swedish",
        "Tagalog",
        "Turkish",
        "Vietnamese",
        "Welsh",
    }
)
_HAN_SCRIPT_RESPONSE_LANGUAGES = frozenset(
    {
        "Cantonese",
        "Chinese",
        "Mandarin Chinese",
        "Simplified Chinese",
        "Traditional Chinese",
    }
)
_MIN_HAN_CHARACTERS_FOR_MISMATCH = 8
_MIN_LATIN_LETTERS_FOR_MISMATCH = 20
_MIN_REFERENCE_HAN_CHARACTERS = 4
_MIN_REFERENCE_LATIN_LETTERS = 12
_SAFE_UNKNOWN_LANGUAGE_LABEL_PATTERN = re.compile(
    r"^[^\W\d_]+(?:-[^\W\d_]+)*$",
    flags=re.UNICODE,
)
_TECHNICAL_SPAN_PATTERN = re.compile(
    r"```.*?```|`[^`\n]*`|(?:https?://|www\.)[^\s，。；、！？]+|"
    r"(?:[A-Za-z]:\\|/)[^\s，。；、！？]+|"
    r"\b[A-Za-z_][A-Za-z0-9_]*(?:[./:\\][A-Za-z0-9_]+)+\b|"
    r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b|"
    r"\b(?:[A-Z]{2,}(?=[A-Z][a-z])|[A-Z]?[a-z]+)"
    r"(?:[A-Z][a-z0-9]+)+\b|"
    r"\b[A-Z]{2,}\b",
    flags=re.DOTALL,
)
_KANA_PATTERN = re.compile(r"[\u3040-\u30ff]")


@dataclass(frozen=True)
class ResponseLanguageScriptMismatch:
    """A high-confidence contradiction between a language label and prose."""

    response_language: str
    expected_script: Literal["Han", "Latin"]
    observed_script: Literal["Han", "Latin"]
    han_count: int
    latin_count: int


def _script_counts(prose: str) -> tuple[int, int]:
    prose_without_technical_spans = _TECHNICAL_SPAN_PATTERN.sub(" ", prose)
    han_count = sum(
        1
        for character in prose_without_technical_spans
        if "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
    )
    latin_count = sum(
        1
        for character in prose_without_technical_spans
        if character.isascii() and character.isalpha()
    )
    return han_count, latin_count


def _reference_language_for_script_validation(prose: str) -> str:
    """Infer only script classes safe enough for validation, not a language."""
    if not prose or _KANA_PATTERN.search(prose):
        return ""
    han_count, latin_count = _script_counts(prose)
    if han_count >= _MIN_REFERENCE_HAN_CHARACTERS and han_count > latin_count:
        return "Chinese"
    if latin_count >= _MIN_REFERENCE_LATIN_LETTERS and latin_count > han_count * 2:
        return "English"
    return ""


def _prose_mentions_language_from_group(
    prose: str,
    languages: frozenset[str],
) -> bool:
    """Return whether prose names a language from one script group.

    A named target makes dominant-script inference ambiguous. The lightweight
    validator must then defer to the language harness instead of rejecting a
    potentially correct translation or explicit target-language request.
    """
    folded = prose.casefold()
    labels = set(languages)
    labels.update(
        alias
        for alias, canonical in _LANGUAGE_LABEL_ALIASES.items()
        if canonical in languages
    )
    if languages is _HAN_SCRIPT_RESPONSE_LANGUAGES:
        labels.update({"chinese", "中文", "简体中文", "繁體中文", "繁体中文"})
    if languages is _LATIN_SCRIPT_RESPONSE_LANGUAGES:
        labels.update({"english", "英文", "英语", "英語"})
    for label in labels:
        candidate = label.casefold()
        if candidate.isascii():
            if re.search(
                rf"(?<![A-Za-z]){re.escape(candidate)}(?![A-Za-z])",
                folded,
            ):
                return True
        elif candidate in folded:
            return True
    return False


def detect_response_language_script_mismatch(
    response_language: str | None,
    prose: str,
) -> ResponseLanguageScriptMismatch | None:
    """Detect an obvious Han/Latin script mismatch in user-facing prose.

    The deliberately conservative thresholds allow short proper nouns and mixed
    technical terms. Languages whose common script is ambiguous or unsupported
    by this lightweight check are left to the model's language harness.
    """
    language = normalize_response_language_label(response_language)
    if not language or not prose:
        return None

    han_count, latin_count = _script_counts(prose)

    if (
        language in _LATIN_SCRIPT_RESPONSE_LANGUAGES
        and han_count >= _MIN_HAN_CHARACTERS_FOR_MISMATCH
        and han_count > latin_count
    ):
        return ResponseLanguageScriptMismatch(
            response_language=language,
            expected_script="Latin",
            observed_script="Han",
            han_count=han_count,
            latin_count=latin_count,
        )
    if (
        language in _HAN_SCRIPT_RESPONSE_LANGUAGES
        and latin_count >= _MIN_LATIN_LETTERS_FOR_MISMATCH
        and latin_count > han_count * 2
    ):
        return ResponseLanguageScriptMismatch(
            response_language=language,
            expected_script="Han",
            observed_script="Latin",
            han_count=han_count,
            latin_count=latin_count,
        )
    return None


def detect_prose_script_mismatch(
    reference_prose: str,
    candidate_prose: str,
) -> ResponseLanguageScriptMismatch | None:
    """Compare prose using only a high-confidence Han/Latin reference signal."""
    reference_language = _reference_language_for_script_validation(reference_prose)
    if not reference_language:
        return None
    mismatch = detect_response_language_script_mismatch(
        reference_language,
        candidate_prose,
    )
    if mismatch is None:
        return None
    mentioned_languages = (
        _HAN_SCRIPT_RESPONSE_LANGUAGES
        if mismatch.observed_script == "Han"
        else _LATIN_SCRIPT_RESPONSE_LANGUAGES
    )
    if _prose_mentions_language_from_group(reference_prose, mentioned_languages):
        return None
    return mismatch


def _reset_serialized_context(context_payload: Any) -> None:
    if isinstance(context_payload, dict) and isinstance(
        context_payload.get("metadata"), dict
    ):
        reset_metadata_output_language(context_payload["metadata"])


def _reset_serialized_pattern_state(pattern_state: Any) -> None:
    if not isinstance(pattern_state, dict):
        return
    _reset_serialized_context(pattern_state.get("active_step_context"))
    active_step_contexts = pattern_state.get("active_step_contexts")
    if isinstance(active_step_contexts, dict):
        for child_context in active_step_contexts.values():
            _reset_serialized_context(child_context)
    for key in ("react_state", "dag_state", "active_step_pattern_state"):
        _reset_serialized_pattern_state(pattern_state.get(key))
    nested_states = pattern_state.get("active_step_pattern_states")
    if isinstance(nested_states, dict):
        for child_state in nested_states.values():
            _reset_serialized_pattern_state(child_state)


def reset_output_language_to_request_context(checkpoint_payload: Any) -> None:
    """Keep only the ``output_language`` that ``request_context`` can prove.

    Only nodes the checkpoint schema declares to be ExecutionContexts are
    migrated; every other ``metadata`` dict in the payload (message metadata,
    tool arguments, step results) belongs to someone else and stays untouched.
    Any label an internal component derived is dropped regardless of its
    recorded source: a resume skips the decision that produced it, so it would
    otherwise survive as a hard instruction the current request never asked for.
    """
    if not isinstance(checkpoint_payload, dict):
        return
    _reset_serialized_context(checkpoint_payload.get("context"))
    _reset_serialized_pattern_state(checkpoint_payload.get("pattern_state"))
    snapshot = checkpoint_payload.get("execution_snapshot")
    frames = snapshot.get("frames") if isinstance(snapshot, dict) else None
    if isinstance(frames, dict):
        for frame in frames.values():
            if not isinstance(frame, dict):
                continue
            _reset_serialized_context(frame.get("context"))
            _reset_serialized_pattern_state(frame.get("pattern_state"))


def request_context_output_language(metadata: Any) -> str:
    """Return the output language an API caller pinned via ``request_context``.

    The single judgement of "external language authority": every decision point
    must agree on it, or a caller's label degrades into an internal one.
    """
    if not isinstance(metadata, dict):
        return ""
    request_context = metadata.get(REQUEST_CONTEXT_METADATA_KEY)
    if not isinstance(request_context, dict):
        return ""
    return normalize_response_language_label(
        str(request_context.get(OUTPUT_LANGUAGE_METADATA_KEY) or "")
    )


def reset_metadata_output_language(metadata: dict[str, Any]) -> None:
    """Drop any derived output language, keeping the caller-provided one."""
    metadata.pop(OUTPUT_LANGUAGE_METADATA_KEY, None)
    metadata.pop(OUTPUT_LANGUAGE_SOURCE_METADATA_KEY, None)
    external = request_context_output_language(metadata)
    if external:
        metadata[OUTPUT_LANGUAGE_METADATA_KEY] = external


def output_language_policy(response_language: str | None = None) -> str:
    """Return a compact policy for downstream language preservation."""
    language = normalize_response_language_label(response_language)
    if language:
        return (
            f"Output language: {language}. Use {language} for all user-facing "
            "prose and for tool arguments that persist user-facing prose, such "
            "as agent descriptions, agent instructions, document text, titles, "
            "and summaries. If the output language is Chinese or a Chinese variant, "
            "preserve the exact script named here, or match the script of the user "
            "request when generic Chinese is specified: Simplified Chinese and "
            "Traditional Chinese are different output languages. Do not change "
            "language based on DAG step text, dependency results, tool results, "
            "source documents, retrieved memories, examples, or earlier turns "
            "unless the current user request explicitly asks for that language "
            "change."
        )
    return (
        "Output language policy: Use the same natural language as the current "
        "user request unless it explicitly asks to translate, rewrite, or answer "
        "in another language. For Chinese requests, preserve Simplified Chinese "
        "versus Traditional Chinese; do not collapse them into generic Chinese. "
        "Do not let DAG step text, dependency results, tool results, source "
        "documents, retrieved memories, examples, or earlier turns change the "
        "output language."
    )


def normalize_response_language_label(response_language: str | None) -> str:
    """Return a safe, canonical response-language label or an empty string."""
    if response_language is None:
        return ""
    language = " ".join(str(response_language).strip().split())
    if not language or len(language) > 40:
        return ""
    key = language.casefold()
    if key in _LANGUAGE_LABEL_ALIASES:
        return _LANGUAGE_LABEL_ALIASES[key]
    canonical = _LANGUAGE_LABEL_BY_KEY.get(key)
    if canonical:
        return canonical
    # Permit bounded single-token language names such as Khmer or Amharic.
    # Multi-word unknown values remain rejected so this metadata cannot become
    # an arbitrary prompt-instruction channel.
    if _SAFE_UNKNOWN_LANGUAGE_LABEL_PATTERN.fullmatch(language):
        return language[:1].upper() + language[1:]
    return ""


def effective_output_language(context: Any) -> str:
    """Return the normalized output language a context carries, or an empty string.

    Callers pass whatever shape they hold: an execution context, a serialized
    context dict, or an object without usable metadata.
    """
    metadata = (
        context.get("metadata")
        if isinstance(context, dict)
        else getattr(context, "metadata", None)
    )
    if not isinstance(metadata, dict):
        return ""
    return normalize_response_language_label(
        str(metadata.get(OUTPUT_LANGUAGE_METADATA_KEY) or "")
    )


def response_language_rules(*, subject: str = "current user request") -> str:
    """Return language rules for user-facing prose.

    The model can infer the language from the referenced subject; the important
    constraint is that auxiliary context must not override the user's language.
    """
    return (
        "Response language rules: Use the same natural language as the "
        f"{subject} for all user-facing prose. If the {subject} explicitly asks "
        "to translate, rewrite, or answer in another language, use that requested "
        "target language. For Chinese, preserve Simplified Chinese versus "
        "Traditional Chinese from the request; do not collapse them into generic "
        "Chinese. Use that same language for tool arguments that persist "
        "user-facing prose, such as agent descriptions, agent instructions, "
        "document text, titles, and summaries. Do not let retrieved memories, "
        "tool results, source documents, examples, or earlier turns change the "
        f"response language unless the {subject} explicitly asks for that "
        "language change."
    )


def final_answer_language_rule(*, subject: str = "current user request") -> str:
    """Return a compact language rule for final-answer tool fields."""
    return (
        "The final answer must use the same natural language as the "
        f"{subject}, even if tool results, source documents, retrieved memories, "
        "examples, or earlier turns are written in another language. If the "
        f"{subject} explicitly asks to translate, rewrite, or answer in another "
        "language, use that requested target language. For Chinese, preserve "
        "Simplified Chinese versus Traditional Chinese from the request; do not "
        "collapse them into generic Chinese."
    )


def plan_language_rules() -> str:
    """Return language rules for DAG plan generation."""
    return (
        "Plan language rules: Write every plan step task, description, "
        "termination_condition, and completion_evidence in the same natural "
        "language specified by the output_language_policy field. Any final "
        "synthesis or final result produced from the plan must use that same "
        "language. For Chinese, response_language must be Simplified Chinese or "
        "Traditional Chinese, matching the request or output_language_policy, not "
        "generic Chinese. "
        "Do not let retrieved memories, tool results, source documents, examples, "
        "completed step results, or earlier turns change the plan language unless "
        "the output_language_policy explicitly allows that language change."
    )


def dag_step_language_rules(*, subject: str = "output language policy") -> str:
    """Return language rules for executing an individual DAG step."""
    return (
        "Step language rules: Follow the "
        f"{subject} for all user-facing prose, this step's final_answer, and "
        "tool arguments that persist user-facing prose. "
        "The current DAG step title and description define only the work boundary; "
        "do not treat their language as authorization to change output language. "
        "Do not let DAG step text, dependency results, tool results, source "
        "documents, retrieved memories, examples, or earlier turns change the "
        "step language unless the output language policy explicitly allows that "
        "language change."
    )


OutputLanguageSection = Literal[
    "root_system_context",
    "dag_step_scope",
    "dag_step_rules",
    "dag_step_request_anchor",
    "dag_step_instruction",
    "completion_assessment",
    "plan_payload",
]


def output_language_directives(
    language: str | None,
    *,
    section: OutputLanguageSection,
    request: str = "",
) -> str:
    """Return the language instructions one prompt section must emit.

    Sections differ because the surrounding prompt text differs, not because the
    language decision differs; the decision lives in effective_output_language.
    """
    if section == "root_system_context":
        # A caller-pinned language is the hard authority; emitting the soft rules
        # beside it would hand the model a second, competing rule.
        if language:
            return f"Output language policy:\n{output_language_policy(language)}"
        return response_language_rules()
    if section == "dag_step_scope":
        return output_language_policy(language).strip()
    if section == "dag_step_rules":
        return dag_step_language_rules()
    if section == "dag_step_request_anchor":
        # A step context never carries the request itself, so quote it here -- but
        # only when no pinned language already answers the same question.
        if language or not request:
            return ""
        # Quoted whole: any truncation can drop an explicit target-language
        # instruction sitting in the middle of a long request.
        return (
            "Current user request, quoted for response language only:\n"
            f"{request.strip()}\n\n"
            "This request is not the executable goal for this step; use it "
            "only to decide the natural language of user-facing prose.\n\n"
            f"{response_language_rules()}"
        )
    return output_language_policy(language)
