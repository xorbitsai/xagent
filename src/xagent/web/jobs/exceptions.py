from __future__ import annotations

from typing import Any


class BackgroundJobHandlerError(RuntimeError):
    def __init__(self, message: str, *, result: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.result = result
