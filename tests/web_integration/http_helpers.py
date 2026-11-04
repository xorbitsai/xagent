"""HTTP helpers for web integration tests (richer assertion failures)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any


def http_detail(response: Any) -> str:
    """Return status line plus body so pytest truncation still shows the HTTP code."""
    code = getattr(response, "status_code", "?")
    try:
        body = json.dumps(response.json(), ensure_ascii=False, indent=2)[:8000]
    except Exception:
        body = (getattr(response, "text", None) or "")[:8000]
    return f"HTTP {code}\n{body}"


def eventually(
    check: Callable[[], bool],
    timeout: float = 10.0,
    detail: Callable[[], str] = lambda: "",
) -> None:
    """Wait out Milvus's Bounded reads, which can lag a write by up to 5 s."""
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, f"never settled: {detail()}"
        time.sleep(0.2)
