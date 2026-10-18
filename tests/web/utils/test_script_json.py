"""HTML-tokenizer regression coverage for the shared script JSON utility."""

import json
from html.parser import HTMLParser

import pytest

from xagent.web.utils.script_json import script_safe_json


class _Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)


@pytest.mark.parametrize(
    "value",
    [
        "</script><script>window.injected=true</script>",
        '</ScRiPt><img src=x onerror="alert(1)">',
        "<!--<script>quoted & data</script>",
        '"\\\n\r\u2028\u2029中文',
        None,
        {"email": "</script><svg onload=alert(1)>", "provider": "a&b"},
    ],
)
def test_script_json_preserves_data_without_creating_html(value):
    encoded = script_safe_json(value)
    assert json.loads(encoded) == value
    assert all(character not in encoded for character in "<>&\u2028\u2029")
    parser = _Elements()
    parser.feed(f"<script>const payload = {encoded};</script>")
    assert parser.tags == ["script"]
