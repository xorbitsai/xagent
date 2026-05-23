from __future__ import annotations

from typing import Any


class BackgroundJobHandlerError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        result: dict[str, Any] | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.result = result
        self.retryable = retryable
