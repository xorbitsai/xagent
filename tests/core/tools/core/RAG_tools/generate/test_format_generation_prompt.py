import logging

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import ConfigurationError
from xagent.core.tools.core.RAG_tools.generate.format_generation_prompt import (
    format_generation_prompt,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_prompt_template_placeholder() -> str:
    """Provides a sample prompt template with placeholder."""
    return "Summarize this: {context}"


@pytest.fixture
def sample_prompt_template_plain() -> str:
    """Provides a sample prompt template without placeholder."""
    return "Please summarize the following context:"


@pytest.fixture
def sample_formatted_contexts() -> str:
    """Provides sample formatted contexts for testing."""
    return "This is the first chunk.\n---\nThis is the second chunk."


class TestFormatGenerationPrompt:
    """Tests for the format_generation_prompt core function."""

    def test_format_generation_prompt_with_placeholder(
        self,
        sample_prompt_template_placeholder: str,
        sample_formatted_contexts: str,
    ) -> None:
        """Test formatting when placeholder is present."""
        result = format_generation_prompt(
            prompt_template=sample_prompt_template_placeholder,
            formatted_contexts=sample_formatted_contexts,
        )

        expected = f"Summarize this: {sample_formatted_contexts}"
        assert result == expected

    def test_format_generation_prompt_plain_template(
        self,
        sample_prompt_template_plain: str,
        sample_formatted_contexts: str,
    ) -> None:
        """Test formatting when no placeholder is present (legacy behavior)."""
        result = format_generation_prompt(
            prompt_template=sample_prompt_template_plain,
            formatted_contexts=sample_formatted_contexts,
        )

        expected = f"{sample_prompt_template_plain}\n\nContext:\n{sample_formatted_contexts}\n\nAnswer:"
        assert result == expected

    def test_format_generation_prompt_empty_template_raises_error(
        self,
        sample_formatted_contexts: str,
    ) -> None:
        """Test that empty prompt template raises ConfigurationError."""
        with pytest.raises(
            ConfigurationError, match="Prompt template cannot be empty."
        ):
            format_generation_prompt(
                prompt_template="",
                formatted_contexts=sample_formatted_contexts,
            )

    def test_format_generation_prompt_empty_contexts_produces_warning_and_formats(
        self,
        sample_prompt_template_plain: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that empty formatted contexts produce a warning but still format."""
        with caplog.at_level(logging.WARNING):
            result = format_generation_prompt(
                prompt_template=sample_prompt_template_plain,
                formatted_contexts="",
            )
            # Check for warning log - using more flexible matching
            assert any(
                "Formatted contexts are empty" in record.message
                for record in caplog.records
            )

        expected_prompt_for_empty_context = (
            f"{sample_prompt_template_plain}\n\nContext:\n\n\nAnswer:"
        )
        assert result == expected_prompt_for_empty_context
