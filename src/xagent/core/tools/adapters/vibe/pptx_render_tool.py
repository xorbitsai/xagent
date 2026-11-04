"""Slide render tool adapter: render selected .pptx slides to PNG images."""

import logging
from typing import TYPE_CHECKING, Any, Dict

from pydantic import StrictInt

from ...core.pptx_render import render_pptx_to_images
from .base import ToolCategory
from .config import BaseToolConfig
from .factory import ToolFactory, register_tool
from .function import FunctionTool

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ....workspace import TaskWorkspace
else:
    TaskWorkspace = Any  # type: ignore[assignment]


class PPTXRenderFunctionTool(FunctionTool):
    """Render function with ToolCategory.PPT category."""

    category = ToolCategory.PPT


RENDER_PPTX_TO_IMAGES_DESCRIPTION = """Render the slides of a .pptx file from the current task workspace to PNG images.

Use this tool when the user wants to see, screenshot or visually inspect an
uploaded presentation: each selected slide becomes one PNG image that is
registered as a durable workspace file.

Args:
    pptx_path: A workspace-relative path, an authorized absolute path, or a
        file_id / file:<id> reference to an existing .pptx file.
    start_page: Optional first slide to render, 1-based and inclusive.
    end_page: Optional last slide to render, 1-based and inclusive.

Behavior and limits:
    - Slide numbering and total_pages include hidden slides.
    - Without start_page/end_page the first 20 slides are rendered; when the
      deck has more slides, truncated is true and the remaining pages are not
      rendered.
    - With only start_page, at most 20 slides from that page are rendered;
      truncated is true when later slides remain.
    - An explicit start_page/end_page range is rendered as requested. A range
      over 20 slides is refused instead of being silently rewritten.
    - Every PNG is capped at 4,000,000 pixels. The render scale is lowered to
      fit the cap, so very large slides are downscaled instead of producing an
      oversized raster.
    - The deck is converted with LibreOffice (soffice) on a PDF cache miss,
      with a 60 second conversion timeout. Repeated page requests reuse a
      content-keyed export that includes hidden slides, separate from previews.
      LibreOffice converts the whole deck; the 20 page
      bound only limits the PNG output.
    - This tool runs in the Xagent backend process, which ships LibreOffice;
      it does not run in the sandbox. When soffice is missing the call fails
      with a clear error instead of returning images.
    - Every returned PNG is registered as a durable workspace file. Each page
      entry carries the 1-based page_number and a file_ref; the result lists
      one inline image artifact per page. Rendered pages keep the slide
      layout; exact fonts and animations may differ from the original editor.
""".strip()


class PPTXRenderTool:
    """PPTX render tool wrapper that handles workspace integration."""

    def __init__(self, workspace: "TaskWorkspace | None" = None):
        self._workspace = workspace

    async def render_pptx_to_images(
        self,
        pptx_path: str,
        start_page: StrictInt | None = None,
        end_page: StrictInt | None = None,
    ) -> Dict[str, Any]:
        """Render selected slides of a workspace .pptx file to PNG images."""
        return await render_pptx_to_images(
            pptx_path,
            self._workspace,
            start_page=start_page,
            end_page=end_page,
        )

    def get_tools(self) -> list:
        """Get all tool instances."""
        return [
            PPTXRenderFunctionTool(
                self.render_pptx_to_images,
                name="render_pptx_to_images",
                description=RENDER_PPTX_TO_IMAGES_DESCRIPTION,
                tags=["pptx", "presentation", "render", "image"],
            )
        ]


@register_tool(categories={"ppt"})
async def create_pptx_render_tools(config: "BaseToolConfig") -> list:
    """Create the slide render tool with workspace support."""
    workspace = await ToolFactory.create_workspace_async(config)
    tool_instance = PPTXRenderTool(workspace)
    return tool_instance.get_tools()
