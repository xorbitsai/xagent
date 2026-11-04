from typing import Any

class DBConnection:
    def open_table(self, name: str) -> Any: ...
    def create_table(
        self,
        name: str,
        data: Any | None = ...,
        schema: object | None = ...,
        mode: str | None = ...,
    ) -> Any: ...
