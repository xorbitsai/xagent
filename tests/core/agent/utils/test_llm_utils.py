"""
Unit tests for LLM content cleaning utilities
"""

from json_repair import loads as repair_loads

from xagent.core.agent.utils.llm_utils import (
    clean_dict_content,
    clean_llm_content,
    clean_messages,
    extract_json_from_markdown,
)


class TestLLMContentCleaning:
    """Test cases for LLM content cleaning functionality"""

    def test_clean_llm_content_control_characters(self):
        """Test removal of control characters"""
        dirty_content = "Hello\x00\x01\x08\x0b\x0c\x0e\x1f\x7fWorld"
        cleaned = clean_llm_content(dirty_content)

        # Should remove control characters but keep visible text
        assert "\x00" not in cleaned
        assert "\x01" not in cleaned
        assert "\x08" not in cleaned
        assert "\x0b" not in cleaned
        assert "\x0c" not in cleaned
        assert "\x0e" not in cleaned
        assert "\x1f" not in cleaned
        assert "\x7f" not in cleaned
        assert "HelloWorld" in cleaned

    def test_clean_llm_content_preserves_valid_chars(self):
        """Test that valid whitespace characters are preserved/normalized"""
        content = "Hello\n\t\rWorld"
        cleaned = clean_llm_content(content)

        # Should preserve newlines and carriage returns, normalize tabs to spaces
        assert "\n" in cleaned
        assert "\r" in cleaned
        assert " " in cleaned  # Tab should be normalized to space
        assert "\t" not in cleaned  # Tab should be normalized
        assert "Hello" in cleaned
        assert "World" in cleaned

    def test_clean_llm_content_html_entities(self):
        """Test HTML entity unescaping"""
        html_content = "Hello &amp; &lt;world&gt; &quot;test&quot; &#39;single&#39;"
        cleaned = clean_llm_content(html_content)

        assert "&amp;" not in cleaned
        assert "&lt;" not in cleaned
        assert "&gt;" not in cleaned
        assert "&quot;" not in cleaned
        assert "&#39;" not in cleaned
        assert "Hello & <world> \"test\" 'single'" in cleaned

    def test_clean_llm_content_non_breaking_spaces(self):
        """Test removal of non-breaking spaces"""
        nbsp_content = "Hello\xa0world\u00a0test"
        cleaned = clean_llm_content(nbsp_content)

        assert "\xa0" not in cleaned
        assert "\u00a0" not in cleaned
        assert "Hello world test" in cleaned

    def test_clean_llm_content_whitespace_normalization(self):
        """Test whitespace normalization"""
        content = "Hello    world\n\n\n\ntest   again"
        cleaned = clean_llm_content(content)

        # Should collapse multiple spaces to single space
        assert "    " not in cleaned
        assert "   " not in cleaned
        assert "\n\n\n\n" not in cleaned  # Should reduce to max 2 newlines
        assert "Hello world" in cleaned
        assert "test again" in cleaned
        assert "\n\n" in cleaned  # Should preserve some structure

    def test_clean_llm_content_length_truncation(self):
        """Test content truncation when too long"""
        # Create content longer than the limit
        long_content = "A" * 60000
        cleaned = clean_llm_content(long_content)

        assert len(cleaned) < 60000
        assert "Content truncated" in cleaned
        assert cleaned.startswith("AAAAA")  # Should keep beginning

    def test_clean_llm_content_non_string_input(self):
        """Test handling of non-string inputs"""
        # Should return non-string inputs unchanged
        assert clean_llm_content(123) == 123
        assert clean_llm_content(None) is None
        assert clean_llm_content([]) == []

    def test_clean_llm_content_empty_string(self):
        """Test handling of empty and whitespace-only strings"""
        assert clean_llm_content("") == ""
        assert clean_llm_content("   \n\t  ") == ""

    def test_clean_messages_list(self):
        """Test cleaning a list of message dictionaries"""
        messages = [
            {"role": "user", "content": "Hello\x00world"},
            {"role": "assistant", "content": "Hi&amp;there"},
            {"role": "system", "content": "System\xa0message"},
        ]

        cleaned_messages = clean_messages(messages)

        assert len(cleaned_messages) == 3
        assert "\x00" not in cleaned_messages[0]["content"]
        assert "&amp;" not in cleaned_messages[1]["content"]
        assert "\xa0" not in cleaned_messages[2]["content"]
        assert cleaned_messages[0]["content"] == "Helloworld"
        assert cleaned_messages[1]["content"] == "Hi&there"
        assert cleaned_messages[2]["content"] == "System message"

    def test_clean_messages_preserves_structure(self):
        """Test that message cleaning preserves message structure"""
        messages = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Response", "other_field": "preserve_me"},
        ]

        cleaned_messages = clean_messages(messages)

        # Should preserve all fields and structure
        assert cleaned_messages[0]["role"] == "user"
        assert cleaned_messages[1]["role"] == "assistant"
        assert cleaned_messages[1]["other_field"] == "preserve_me"

    def test_clean_dict_content_simple(self):
        """Test cleaning a simple dictionary with string values"""
        data = {
            "clean_key": "Hello\x00world",
            "preserve_key": 123,
            "nested": {"content": "Test&amp;content"},
        }

        cleaned_data = clean_dict_content(data)

        assert "\x00" not in cleaned_data["clean_key"]
        assert cleaned_data["preserve_key"] == 123  # Non-string preserved
        assert "&amp;" not in cleaned_data["nested"]["content"]

    def test_clean_dict_content_nested_structures(self):
        """Test cleaning deeply nested structures with lists and dicts"""
        data = {
            "level1": {
                "level2": [
                    {"content": "Item\x001"},
                    {"content": "Item&amp;2"},
                    "string_item\xa0with_spaces",
                    123,  # Non-string in list
                ],
                "deep_content": "Deep\x7fcontent",
            }
        }

        cleaned_data = clean_dict_content(data)

        # Check all string values are cleaned
        assert "\x00" not in cleaned_data["level1"]["level2"][0]["content"]
        assert "&amp;" not in cleaned_data["level1"]["level2"][1]["content"]
        assert "\xa0" not in cleaned_data["level1"]["level2"][2]
        assert cleaned_data["level1"]["level2"][3] == 123  # Non-string preserved
        assert "\x7f" not in cleaned_data["level1"]["deep_content"]

    def test_clean_dict_content_non_dict_input(self):
        """Test handling of non-dictionary inputs"""
        assert clean_dict_content("string") == "string"
        assert clean_dict_content(123) == 123
        assert clean_dict_content(None) is None
        assert clean_dict_content(["list", "of", "items"]) == ["list", "of", "items"]

    def test_clean_dict_content_empty_dict(self):
        """Test handling of empty dictionary"""
        empty_dict = {}
        cleaned = clean_dict_content(empty_dict)
        assert cleaned == {}
        assert cleaned is not empty_dict  # Should return new dict

    def test_complex_real_world_example(self):
        """Test with a realistic example from web scraping"""
        # Simulate problematic content from web scraping
        scraped_content = "Found information about\x00Python\xa0on\xa0wikipedia&amp;other sources.\n\n\n\nThe language is great!"

        cleaned = clean_llm_content(scraped_content)

        # Should handle all the issues
        assert "\x00" not in cleaned
        assert "\xa0" not in cleaned
        assert "&amp;" not in cleaned
        assert "\n\n\n\n" not in cleaned
        assert "Python" in cleaned
        assert "wikipedia&other" in cleaned
        assert "The language is great!" in cleaned

    def test_edge_cases(self):
        """Test various edge cases"""
        # Only control characters
        assert clean_llm_content("\x00\x01\x02") == ""

        # Mixed content with various issues
        messy = "\x00Hello\xa0World&amp;\x7fTest\n\n\n\nEnd"
        cleaned = clean_llm_content(messy)
        assert "\x00" not in cleaned
        assert "\xa0" not in cleaned
        assert "&amp;" not in cleaned
        assert "\x7f" not in cleaned
        assert "\n\n\n\n" not in cleaned  # Should reduce to max 2 newlines
        assert "Hello World&Test" in cleaned
        assert "\n\n" in cleaned  # Should preserve 2 newlines
        assert "End" in cleaned


class TestExtractJsonFromMarkdown:
    """Test cases for extract_json_from_markdown functionality"""

    def test_json_object_with_inner_code_block(self):
        """Test that JSON objects with inner code blocks are NOT extracted"""
        # This is the problematic case: JSON string contains code blocks within string values
        json_with_inner_code = """{
  "achieved": true,
  "reason": "Test",
  "final_answer": "Here's the code:\\n```python\\nimport random\\nprint(random.randint(1, 10))\\n```",
  "memory_insights": {
    "should_store": false,
    "reason": "Test"
  }
}"""

        result = extract_json_from_markdown(json_with_inner_code)

        # Should NOT extract (content is already JSON)
        assert result == json_with_inner_code

        # Should be parseable as JSON
        parsed = repair_loads(result, logging=False)
        assert isinstance(parsed, dict)
        assert parsed["achieved"] is True
        assert "final_answer" in parsed
        assert "```python" in parsed["final_answer"]

    def test_json_wrapped_in_markdown_code_block(self):
        """Test that JSON wrapped in markdown code blocks IS extracted"""
        json_in_markdown = """```json
{
  "achieved": true,
  "reason": "Test successful",
  "confidence": 0.95
}
```"""

        result = extract_json_from_markdown(json_in_markdown)

        # Should extract the JSON from markdown
        assert result != json_in_markdown
        assert not result.startswith("```")
        assert result.startswith("{")

        # Should be parseable
        parsed = repair_loads(result, logging=False)
        assert isinstance(parsed, dict)
        assert parsed["achieved"] is True
        assert parsed["confidence"] == 0.95

    def test_json_in_generic_code_block(self):
        """Test extraction from generic (no language specified) code blocks"""
        json_in_generic_block = """```
{
  "achieved": true,
  "reason": "Test"
}
```"""

        result = extract_json_from_markdown(json_in_generic_block)

        # Should extract
        assert result != json_in_generic_block
        assert not result.startswith("```")
        assert result.startswith("{")

        parsed = repair_loads(result, logging=False)
        assert isinstance(parsed, dict)

    def test_plain_text_no_code_blocks(self):
        """Test that plain text without code blocks is returned unchanged"""
        plain_text = "This is just plain text without any JSON or code blocks."

        result = extract_json_from_markdown(plain_text)

        # Should return unchanged
        assert result == plain_text

    def test_json_array_with_inner_code_block(self):
        """Test that JSON arrays with inner code blocks are NOT extracted"""
        json_array_with_code = """[
  {
    "achieved": true,
    "code_example": "```python\\nprint('hello')\\n```"
  }
]"""

        result = extract_json_from_markdown(json_array_with_code)

        # Should NOT extract (already JSON array)
        assert result == json_array_with_code

        parsed = repair_loads(result, logging=False)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_json_array_wrapped_in_markdown(self):
        """Test extraction of JSON arrays from markdown"""
        json_array_in_markdown = """```json
[
  {"achieved": true},
  {"achieved": false}
]
```"""

        result = extract_json_from_markdown(json_array_in_markdown)

        # Should extract
        assert result != json_array_in_markdown
        assert result.startswith("[")

        parsed = repair_loads(result, logging=False)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_empty_string(self):
        """Test handling of empty string"""
        result = extract_json_from_markdown("")
        assert result == ""

    def test_non_string_input(self):
        """Test handling of non-string inputs"""
        assert extract_json_from_markdown(None) is None
        assert extract_json_from_markdown(123) == 123
        assert extract_json_from_markdown([]) == []

    def test_multiple_code_blocks_extracts_first(self):
        """Test that only the first code block is extracted"""
        multiple_blocks = """```json
{"first": "value"}
```

Some text in between

```json
{"second": "value"}
```"""

        result = extract_json_from_markdown(multiple_blocks)

        # Should extract first block
        assert '{"first": "value"}' in result
        assert '{"second": "value"}' not in result

        parsed = repair_loads(result, logging=False)
        assert isinstance(parsed, dict)
        assert parsed["first"] == "value"

    def test_markdown_with_language_variants(self):
        """Test extraction from code blocks with different language specifiers"""
        # Should work with json specifier
        with_json = extract_json_from_markdown('```json\n{"key": "value"}\n```')
        assert '{"key": "value"}' in with_json

        # Should work with no specifier
        with_no_spec = extract_json_from_markdown('```\n{"key": "value"}\n```')
        assert '{"key": "value"}' in with_no_spec

    def test_real_world_goal_check_response(self):
        """Test with a realistic goal check response that has inner code blocks"""
        realistic_response = """{
  "achieved": true,
  "reason": "Successfully wrote and executed Python script",
  "confidence": 1.0,
  "final_answer": "Python random number generation task completed successfully.\\n\\n### Execution Results Summary\\n1. **Random Integers:** `[82, 15, 4]`\\n\\n### Core Code Implementation\\n```python\\nimport random\\nrandom_integers = [random.randint(1, 100) for _ in range(5)]\\n```",
  "memory_insights": {
    "should_store": false,
    "reason": "Standard basic programming task",
    "classification": {
      "primary_domain": "Software Development",
      "task_type": "Code Generation",
      "complexity_level": "Simple"
    }
  }
}"""

        result = extract_json_from_markdown(realistic_response)

        # Should NOT extract inner code block
        assert result == realistic_response

        # Should be parseable and have correct structure
        parsed = repair_loads(result, logging=False)
        assert isinstance(parsed, dict)
        assert parsed["achieved"] is True
        assert "final_answer" in parsed
        # Verify the Python code is still in the final_answer
        assert "```python" in parsed["final_answer"]
        assert "import random" in parsed["final_answer"]
