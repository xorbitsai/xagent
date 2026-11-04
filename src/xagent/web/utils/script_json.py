"""Encode JSON data safely inside an HTML script element."""

import json
from typing import Any


def script_safe_json(value: Any) -> str:
    """Keep JSON inert to the HTML tokenizer, without changing its value.

    JSON quoting alone leaves ``</script>`` intact. Escape HTML-sensitive
    characters so untrusted strings cannot close the surrounding element.
    The default ASCII encoding also escapes JavaScript line separators.
    This is for JSON expressions, not HTML attributes or executable code.
    """
    return (
        json.dumps(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
