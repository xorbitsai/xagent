"""
JSON Translation Tool

Translates specific fields in JSON structures using LLM.
Supports nested structures and batch translation.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TranslateJSONToolCore:
    """Core JSON translation functionality"""

    def __init__(self, llm: Optional[Any] = None) -> None:
        """
        Initialize JSON translation tool.

        Args:
            llm: LLM instance for translation
        """
        self._llm = llm

    def _get_field_value(self, data: Dict[str, Any], field_path: str) -> List[Any]:
        """
        Get values from nested dict using dot notation.

        Args:
            data: Dictionary to search
            field_path: Dot-separated field path (e.g., "segments.text")

        Returns:
            List of matching values and their parent dicts for updating
        """
        results: list[dict[str, Any]] = []

        def traverse(
            obj: Any, path: str, parent: Any = None, field_idx: Optional[int] = None
        ) -> None:
            """Recursively traverse object to find matching paths."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if current_path == field_path or current_path.startswith(
                        field_path + "."
                    ):
                        if current_path == field_path:
                            # Found exact match
                            results.append(
                                {
                                    "value": value,
                                    "parent": obj,
                                    "key": key,
                                    "field_idx": len(results),
                                }
                            )
                    traverse(value, current_path, obj)
            elif isinstance(obj, list) and path:
                for idx, item in enumerate(obj):
                    traverse(item, path, obj, idx)

        traverse(data, "")
        return results

    def _set_field_value(
        self, data: Dict[str, Any], field_path: str, value: Any, field_idx: int = 0
    ) -> bool:
        """
        Set value in nested dict using dot notation.

        Args:
            data: Dictionary to update
            field_path: Dot-separated field path
            value: Value to set
            field_idx: Index when multiple fields match (for array elements)

        Returns:
            True if successful, False otherwise
        """
        parts = field_path.split(".")
        current = data

        for i, part in enumerate(parts[:-1]):
            if part.isdigit() and isinstance(current, list):
                idx = int(part)
                if idx < len(current):
                    current = current[idx]
                else:
                    return False
            elif part in current:
                current = current[part]
            else:
                return False

        last_part = parts[-1]
        if last_part.isdigit() and isinstance(current, list):
            idx = int(last_part)
            if idx < len(current):
                current[idx] = value
                return True
        elif last_part in current:
            current[last_part] = value
            return True

        return False

    async def translate_values(
        self,
        texts: List[str],
        target_lang: str,
        source_lang: Optional[str] = None,
    ) -> List[str]:
        """
        Batch translate texts using LLM.

        Args:
            texts: List of texts to translate
            target_lang: Target language
            source_lang: Source language (auto-detect if None)

        Returns:
            List of translated texts
        """
        if not texts:
            return []

        if not self._llm:
            raise ValueError("No LLM instance available")

        # Build translation prompt
        source_info = f" from {source_lang}" if source_lang else ""
        prompt = f"""Translate the following texts to {target_lang}{source_info}. Return only the translations, one per line, in the same order.

Texts to translate:
{chr(10).join(f"{i + 1}. {text}" for i, text in enumerate(texts))}

Translations:"""

        messages = [
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self._llm.chat(messages=messages)
            if isinstance(response, str):
                content = response
            elif isinstance(response, dict):
                content = response.get("content", response)
            else:
                content = str(response)

            # Parse translations
            lines = content.strip().split("\n")
            translations = []

            for line in lines:
                line = line.strip()
                # Remove numbering if present
                if line and line[0].isdigit() and line[1] == ".":
                    translations.append(line.split(".", 1)[1].strip())
                elif line:
                    translations.append(line)

            # Ensure we have the right number of translations
            if len(translations) != len(texts):
                logger.warning(
                    f"Translation count mismatch: expected {len(texts)}, got {len(translations)}"
                )
                # Fallback: return original texts
                return texts

            return translations

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise

    async def translate_json(
        self,
        json_data: str | Dict[str, Any],
        target_fields: List[str],
        output_field: str = "translated_text",
        target_lang: str = "en",
        source_lang: Optional[str] = None,
    ) -> str:
        """
        Translate specific fields in JSON structure.

        Args:
            json_data: JSON string or dict to process
            target_fields: List of field paths to translate (e.g., ["segments.text"])
            output_field: Field name for translated text (default: "translated_text")
            target_lang: Target language code (default: "en")
            source_lang: Source language code (auto-detect if None)

        Returns:
            Translated JSON string

        Example:
            Input: {"segments": [{"text": "你好"}]}
            Fields: ["segments.text"]
            Output: {"segments": [{"text": "你好", "translated_text": "Hello"}]}
        """
        # Parse JSON
        if isinstance(json_data, str):
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}")
        else:
            data = json_data

        # Collect all texts to translate with their context
        all_results = []
        for field_path in target_fields:
            results = self._get_field_value(data, field_path)
            for result in results:
                result["field_path"] = field_path
            all_results.extend(results)

        if not all_results:
            logger.warning(f"No fields found matching: {target_fields}")
            return json.dumps(data, ensure_ascii=False, indent=2)

        # Extract texts
        texts = [r["value"] for r in all_results if isinstance(r["value"], str)]

        if not texts:
            logger.warning("No text values found to translate")
            return json.dumps(data, ensure_ascii=False, indent=2)

        # Translate
        translated_texts = await self.translate_values(texts, target_lang, source_lang)

        # Update JSON with translated values
        trans_idx = 0
        for result in all_results:
            if not isinstance(result["value"], str):
                continue

            parent = result["parent"]
            key = result["key"]
            translated_text = (
                translated_texts[trans_idx]
                if trans_idx < len(translated_texts)
                else result["value"]
            )
            trans_idx += 1

            if isinstance(parent, dict):
                if key in parent and isinstance(parent[key], dict):
                    # The value is a dict, add translation field to it
                    parent[key][output_field] = translated_text
                elif key in parent:
                    # The value is a primitive, create field-specific output name
                    # e.g., "text" + "_" + "translated" -> "text_translated"
                    field_name = result["field_path"].split(".")[-1]
                    # Check if field_path is a simple (non-nested) field
                    if "." not in result["field_path"]:
                        # Root-level field, create field-specific output
                        output_name = f"{field_name}_{output_field}"
                        parent[output_name] = translated_text
                    else:
                        # Nested field, add to parent (would overwrite if multiple fields at same level)
                        parent[output_field] = translated_text
            elif isinstance(parent, list) and 0 <= key < len(parent):
                if isinstance(parent[key], dict):
                    parent[key][output_field] = translated_text

        return json.dumps(data, ensure_ascii=False, indent=2)


def translate_json(
    json_data: str | Dict[str, Any],
    target_fields: List[str],
    output_field: str = "translated_text",
    target_lang: str = "en",
    source_lang: Optional[str] = None,
) -> str:
    """
    Translate specific fields in JSON structure.

    Args:
        json_data: JSON string or dict to process
        target_fields: List of field paths to translate (e.g., ["segments.text"])
        output_field: Field name for translated text (default: "translated_text")
        target_lang: Target language code (default: "en")
        source_lang: Source language code (auto-detect if None)

    Returns:
        Translated JSON string

    Example:
        >>> json_str = '{"segments": [{"text": "你好"}]}'
        >>> result = translate_json(json_str, ["segments.text"], target_lang="en")
        >>> # Returns: {"segments": [{"text": "你好", "translated_text": "Hello"}]}

    Language codes:
        - 'zh': Chinese (Mandarin)
        - 'en': English
        - 'yue': Cantonese
        - 'ja': Japanese
        - 'ko': Korean
        And more...
    """
    import asyncio

    # This would need to be called in async context
    # For now, provide a sync wrapper using asyncio.run()
    tool = TranslateJSONToolCore(llm=None)  # LLM should be passed from config
    return asyncio.run(
        tool.translate_json(
            json_data, target_fields, output_field, target_lang, source_lang
        )
    )
