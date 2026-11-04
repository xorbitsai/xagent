"""
Demo tools for Web UI testing and development.

This module provides simple mock tools that can be used for testing
the DAG execution system in the web interface.
"""

import asyncio
import random
import time
from typing import Any, Dict, List, Mapping, Optional, Type

from pydantic import BaseModel

from xagent.core.tools.adapters.vibe.base import Tool, ToolMetadata, ToolVisibility


class SimpleToolArgs(BaseModel):
    """Arguments for simple tools"""

    query: Optional[str] = "default query"
    data: Optional[str] = None


class SimpleTool:
    """Simple tool for web demo and testing"""

    def __init__(self, name: str, description: str):
        self._metadata = ToolMetadata(
            name=name, description=description, visibility=ToolVisibility.PUBLIC
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def args_type(self) -> Type[BaseModel]:
        return SimpleToolArgs

    def return_type(self) -> Type[BaseModel]:
        class ReturnType(BaseModel):
            result: str
            success: bool

        return ReturnType

    def state_type(self) -> Optional[Type[BaseModel]]:
        return None

    def is_async(self) -> bool:
        return True

    def return_value_as_string(self, value: Any) -> str:
        return str(value)

    async def run_json_async(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        """Execute tool with mock functionality"""
        # Simulate some work
        await asyncio.sleep(0.5)

        return {
            "tool_name": self.metadata.name,
            "description": self.metadata.description,
            "args_received": dict(args),
            "result": f"Processed by {self.metadata.name}: {args.get('query', 'no query')}",
            "success": True,
            "timestamp": time.time(),
            "mock_data": f"Generated mock data from {self.metadata.name}",
            "score": random.uniform(0.8, 1.0),
        }

    def run_json_sync(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        return {"sync_result": "not used"}

    async def save_state_json(self) -> Mapping[str, Any]:
        return {}

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        pass


def create_demo_tools() -> List[Tool]:
    """Create demo tools for web UI testing"""
    return [
        SimpleTool("data_collector", "Collect data from various sources"),
        SimpleTool("data_processor", "Process and clean collected data"),
        SimpleTool("analyzer", "Analyze processed data for insights"),
        SimpleTool("visualizer", "Create charts and visualizations"),
        SimpleTool("reporter", "Generate comprehensive reports"),
    ]
