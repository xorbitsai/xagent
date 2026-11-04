from typing import Any, Mapping

class Response:
    content: bytes
    headers: Mapping[str, str]
    status_code: int
    text: str
    url: str

class AsyncSession:
    def __init__(
        self,
        *,
        impersonate: str | None = ...,
        timeout: float | int | None = ...,
        **kwargs: Any,
    ) -> None: ...
    async def __aenter__(self) -> "AsyncSession": ...
    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...
    async def get(self, url: str, **kwargs: Any) -> Response: ...
