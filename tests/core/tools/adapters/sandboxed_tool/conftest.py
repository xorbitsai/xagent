"""Shared test helpers for sandboxed tool adapter tests."""

from typing import Any, Mapping, Type

from pydantic import BaseModel, Field

from xagent.core.tools.adapters.vibe.base import AbstractBaseTool


class FakeArgsModel(BaseModel):
    """Common fake args model used by sandboxed tool adapter tests."""

    code: str = Field(default="")


class FakeResultModel(BaseModel):
    """Common fake result model used by sandboxed tool adapter tests."""

    output: str = Field(default="")


class FakeBaseTool(AbstractBaseTool):
    """Common fake AbstractBaseTool implementation for tests."""

    @property
    def name(self) -> str:
        return "fake_tool"

    @property
    def description(self) -> str:
        return "fake"

    @property
    def tags(self) -> list[str]:
        return []

    def args_type(self) -> Type[BaseModel]:
        return FakeArgsModel

    def return_type(self) -> Type[BaseModel]:
        return FakeResultModel

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        return {}

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return {}
