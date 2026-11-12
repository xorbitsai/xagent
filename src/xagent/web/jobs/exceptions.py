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


class BackgroundJobDeferred(RuntimeError):
    """Run the same job again later without spending its retry budget."""

    def __init__(self, message: str, *, delay_seconds: int) -> None:
        super().__init__(message)
        self.delay_seconds = max(1, int(delay_seconds))
