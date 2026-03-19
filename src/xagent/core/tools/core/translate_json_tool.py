"""
JSON Translation Tool

Translates specific fields in JSON structures using LLM.
Supports nested structures and batch translation.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from tqdm import tqdm as tqdm_std  # type: ignore[import-untyped]
from tqdm.asyncio import tqdm as tqdm_async  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class TranslateJSONToolCore:
    """Core JSON translation functionality"""

    def __init__(
        self, llm: Optional[Any] = None, workspace: Optional[Any] = None
    ) -> None:
        """
        Initialize JSON translation tool.

        Args:
            llm: LLM instance for translation
            workspace: Optional workspace for saving translated files
        """
        self._llm = llm
        self._workspace = workspace

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

    async def _translate_batch(
        self,
        batch_texts: List[str],
        batch_index: int,
        target_lang: str,
        source_lang: Optional[str],
        instructions_section: str,
    ) -> List[str]:
        """
        Translate a batch of texts.

        Args:
            batch_texts: List of texts to translate in this batch
            batch_index: Batch index for error reporting
            target_lang: Target language
            source_lang: Source language (auto-detect if None)
            instructions_section: Additional instructions section for prompt

        Returns:
            List of translated texts

        Raises:
            RuntimeError: If translation fails
            ValueError: If translation count mismatches
        """
        # Assert LLM is available (caller should have checked)
        assert self._llm is not None, "LLM must be available for translation"

        # Build translation prompt for this batch
        source_info = f" from {source_lang}" if source_lang else ""
        prompt = f"""Translate the following texts to {target_lang}{source_info}. Return only the translations, one per line, in the same order.{instructions_section}

Texts to translate:
{chr(10).join(f"{i + 1}. {text}" for i, text in enumerate(batch_texts))}

Translations:"""

        messages = [
            {"role": "user", "content": prompt},
        ]

        try:
            # Use stream_chat to avoid timeout
            content = ""
            async for chunk in self._llm.stream_chat(messages=messages):
                if chunk.is_token():
                    content += chunk.delta
                elif chunk.is_error():
                    raise RuntimeError(f"Translation error: {chunk.delta}")

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
            if len(translations) != len(batch_texts):
                raise ValueError(
                    f"Batch {batch_index}: Translation count mismatch: expected {len(batch_texts)}, got {len(translations)}. "
                    f"LLM response:\n{content}"
                )

            return translations

        except Exception:
            # Re-raise the exception to notify the user of translation failure
            raise

    async def translate_values(
        self,
        texts: List[str],
        target_lang: str,
        source_lang: Optional[str] = None,
        batch_size: int = 10,
        instructions: Optional[str] = None,
    ) -> List[str]:
        """
        Batch translate texts using LLM with parallel batch processing.

        Args:
            texts: List of texts to translate
            target_lang: Target language
            source_lang: Source language (auto-detect if None)
            batch_size: Number of texts to translate per batch (default: 10)
            instructions: Additional translation instructions (e.g., style, terminology, context)

        Returns:
            List of translated texts
        """
        if not texts:
            return []

        if not self._llm:
            raise ValueError("No LLM instance available")

        # Build instructions section for prompt
        instructions_section = ""
        if instructions:
            instructions_section = f"\n\nAdditional Instructions:\n{instructions}\n"

        try:
            # Split texts into batches
            batches = [
                texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
            ]

            if len(batches) == 1:
                # Single batch: translate directly
                logger.info(f"Translating single batch of {len(texts)} texts")
                return await self._translate_batch(
                    batches[0], 0, target_lang, source_lang, instructions_section
                )
            else:
                # Multiple batches: translate in parallel with progress tracking
                logger.info(
                    f"Translating {len(texts)} texts in {len(batches)} parallel batches (batch_size={batch_size})"
                )

                # Create translation tasks for all batches with progress tracking
                import asyncio

                # Create async progress bar for batch completion
                with tqdm_async(
                    total=len(batches),
                    desc="Translation batches",
                    unit="batch",
                    colour="green",
                ) as pbar:
                    # Create wrapper functions to update progress bar
                    async def translate_batch_with_progress(
                        batch_texts: List[str], batch_index: int
                    ) -> List[str]:
                        result = await self._translate_batch(
                            batch_texts,
                            batch_index,
                            target_lang,
                            source_lang,
                            instructions_section,
                        )
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "batch": f"{batch_index + 1}/{len(batches)}",
                                "texts": len(batch_texts),
                            }
                        )
                        return result

                    tasks = [
                        translate_batch_with_progress(batch, i)
                        for i, batch in enumerate(batches)
                    ]

                    # Execute all batches in parallel
                    results = await asyncio.gather(*tasks)

                # Combine results from all batches with item-level progress
                combined_translations = []
                with tqdm_std(
                    total=len(texts),
                    desc="Combining translations",
                    unit="text",
                    colour="blue",
                    leave=False,
                ) as pbar:
                    for batch_result in results:
                        combined_translations.extend(batch_result)
                        pbar.update(len(batch_result))

                return combined_translations

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
        batch_size: int = 10,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Translate specific fields in JSON structure with parallel batch processing.

        Args:
            json_data: JSON string or dict to process
            target_fields: List of field paths to translate (e.g., ["segments.text"])
            output_field: Field name for translated text (default: "translated_text")
            target_lang: Target language code (default: "en")
            source_lang: Source language code (auto-detect if None)
            batch_size: Number of texts to translate per batch (default: 10)
            instructions: Additional translation instructions (e.g., style, terminology, context)

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
                return {
                    "success": False,
                    "result": "",
                    "error": f"Invalid JSON: {e}",
                    "fields_translated": 0,
                    "target_lang": target_lang,
                }
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
            return {
                "success": False,
                "result": json.dumps(data, ensure_ascii=False, indent=2),
                "error": "No fields found matching the specified paths",
                "fields_translated": 0,
                "target_lang": target_lang,
            }

        # Extract texts
        texts = [r["value"] for r in all_results if isinstance(r["value"], str)]

        if not texts:
            logger.warning("No text values found to translate")
            return {
                "success": False,
                "result": json.dumps(data, ensure_ascii=False, indent=2),
                "error": "No text values found to translate",
                "fields_translated": 0,
                "target_lang": target_lang,
            }

        # Translate
        try:
            translated_texts = await self.translate_values(
                texts, target_lang, source_lang, batch_size, instructions
            )
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                "success": False,
                "result": "",
                "error": str(e),
                "fields_translated": 0,
                "target_lang": target_lang,
            }

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

        result_json = json.dumps(data, ensure_ascii=False, indent=2)

        # Save translation to JSON file if workspace is available
        file_id: Optional[str] = None
        translation_path = None
        saved_to_workspace = False

        if self._workspace:
            try:
                # Generate filename for translation
                filename = f"translation_{uuid.uuid4().hex[:8]}.json"

                # Build structured JSON data
                translation_data = {
                    "target_fields": target_fields,
                    "output_field": output_field,
                    "target_lang": target_lang,
                    "source_lang": source_lang,
                    "fields_translated": len(translated_texts),
                    "result": data,
                    "metadata": {
                        "input_type": "json_string"
                        if isinstance(json_data, str)
                        else "json_dict",
                        "total_input_fields": len(target_fields),
                    },
                }

                # Register and save file in workspace
                with self._workspace.auto_register_files():
                    save_path = self._workspace.output_dir / filename

                    # Write translation to JSON file
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(translation_data, f, ensure_ascii=False, indent=2)

                    translation_path = str(save_path)
                    logger.info(f"Saved translation to: {translation_path}")

                # Get file ID from workspace after registration
                if translation_path:
                    file_id = self._workspace.get_file_id_from_path(translation_path)
                    saved_to_workspace = True

            except Exception as e:
                logger.warning(f"Failed to save translation to workspace: {e}")

        return {
            "success": True,
            "result": result_json,
            "error": None,
            "fields_translated": len(translated_texts),
            "target_lang": target_lang,
            "file_id": file_id,
            "translation_path": translation_path,
            "saved_to_workspace": saved_to_workspace,
        }


def translate_json(
    json_data: str | Dict[str, Any],
    target_fields: List[str],
    output_field: str = "translated_text",
    target_lang: str = "en",
    source_lang: Optional[str] = None,
) -> Dict[str, Any]:
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
