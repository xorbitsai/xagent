import logging

from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def format_generation_prompt(
    prompt_template: str,
    formatted_contexts: str,
) -> str:
    """Formats a prompt template and contexts into a single string for LLM input.

    This function takes a base prompt template and a string of formatted contexts,
    and combines them into a cohesive prompt string. If the template contains
    a "{context}" placeholder, it will be replaced with the formatted contexts.
    Otherwise, the contexts will be appended after the template.

    Args:
        prompt_template: The base template for the prompt.
        formatted_contexts: A string containing the relevant contexts.

    Returns:
        A single string representing the full prompt ready for LLM consumption.

    Raises:
        ConfigurationError: If `prompt_template` is empty.
    """
    if not prompt_template:
        raise ConfigurationError("Prompt template cannot be empty.")

    if not formatted_contexts:
        logger.warning(
            "Formatted contexts are empty, which might lead to non-grounded generation."
        )

    # Check if the template has a placeholder for context
    if "{context}" in prompt_template:
        try:
            full_prompt = prompt_template.format(context=formatted_contexts)
        except (KeyError, ValueError) as e:
            logger.error("Failed to format prompt template: %s", e)
            # Fallback to appending if formatting fails
            full_prompt = f"{prompt_template}\n\nContext:\n{formatted_contexts}"
    else:
        # Default behavior: append context and answer marker
        full_prompt = f"{prompt_template}\n\nContext:\n{formatted_contexts}\n\nAnswer:"

    logger.debug("Formatted prompt length: %s chars.", len(full_prompt))
    return full_prompt
