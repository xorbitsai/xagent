"""Prompt snippets for preserving user-facing response language."""

import re
from dataclasses import dataclass
from typing import Literal

OUTPUT_LANGUAGE_METADATA_KEY = "output_language"
OUTPUT_LANGUAGE_SOURCE_METADATA_KEY = "output_language_source"
OUTPUT_LANGUAGE_SOURCE_PLAN = "dag_plan"
OUTPUT_LANGUAGE_SOURCE_ROUTER = "auto_router"

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
_TECHNICAL_SPAN_PATTERN = re.compile(
    r"```.*?```|`[^`\n]*`|(?:https?://|www\.)[^\s，。；、！？]+|"
    r"(?:[A-Za-z]:\\|/)[^\s，。；、！？]+|"
    r"\b[A-Za-z_][A-Za-z0-9_]*(?:[./:\\][A-Za-z0-9_]+)+\b|"
    r"\b(?:[A-Z]{2,}|[a-z]+[A-Z][A-Za-z0-9]*)\b",
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
    return detect_response_language_script_mismatch(
        reference_language,
        candidate_prose,
    )


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
    return _LANGUAGE_LABEL_BY_KEY.get(key, "")


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
        "Chinese. Do not let retrieved memories, tool results, source documents, "
        "examples, or earlier turns change the response language unless "
        f"the {subject} explicitly asks for that language change."
    )


def final_answer_language_rule(*, subject: str = "current user request") -> str:
    """Return a compact language rule for final-answer tool fields."""
    return (
        "The final answer must use the same natural language as the "
        f"{subject}, even if tool results, source documents, retrieved memories, "
        "examples, or earlier turns are written in another language. For Chinese, "
        "preserve Simplified Chinese versus Traditional Chinese from the request; "
        "do not collapse them into generic Chinese."
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
