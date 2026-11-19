"""Host-supplied frame-src additions; parity with frontend/extra-frame-src.mjs.

Entries are matched as plain strings, never parsed as URLs: only
``https://<host>`` and ``https://*.<host>`` with a canonical lowercase DNS host
pass, so ports, paths, credentials, IP literals, escapes and CSP delimiters
cannot reach the policy. A wildcard needs at least two labels after ``*.``.
"""

import re

_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_TOP_LABEL = r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?"
_ENTRY = re.compile(rf"https://(?:\*\.(?:{_LABEL}\.)+|(?:{_LABEL}\.)*){_TOP_LABEL}")


def extra_frame_src_origins(configured: str) -> list[str]:
    """Valid entries of a space-separated list, in order and without duplicates.

    Invalid entries are dropped one by one.
    """
    entries = re.split(r"[\t\n\f\r ]+", configured)
    return list(dict.fromkeys(e for e in entries if _ENTRY.fullmatch(e)))
