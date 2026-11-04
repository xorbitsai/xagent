from typing import Optional

from pydantic import BaseModel


class Tool(BaseModel):
    id: str


class PyTool(Tool):
    path: Optional[str]
    content: Optional[str]
