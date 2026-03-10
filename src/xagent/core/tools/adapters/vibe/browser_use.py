"""
Browser automation tools for xagent Vibe framework.

Provides browser automation capabilities with proper session lifecycle management.
Browser sessions are automatically cleaned up when tasks complete.
"""

import logging
import os
from typing import Any, Mapping, Optional, Type

from pydantic import BaseModel, Field

from ....tools.core.browser_use import (
    browser_click,
    browser_close,
    browser_evaluate,
    browser_extract_text,
    browser_fill,
    browser_list_sessions,
    browser_navigate,
    browser_pdf,
    browser_screenshot,
    browser_select_option,
    browser_wait_for_selector,
    get_browser_manager,
)
from ....workspace import TaskWorkspace
from .base import AbstractBaseTool, ToolCategory, ToolVisibility

logger = logging.getLogger(__name__)


# ============== Input/Output Schemas ==============


class BrowserNavigateArgs(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for the browser (auto-managed, usually not needed)",
    )
    url: str = Field(description="URL to navigate to")
    headless: bool = Field(
        default=False,
        description="Run mode (default: False). Use True only for simple content extraction",
    )
    wait_until: str = Field(default="networkidle", description="Wait condition")


class BrowserNavigateResult(BaseModel):
    success: bool = Field(description="Whether navigation succeeded")
    session_id: str = Field(description="Session ID")
    url: str = Field(description="Navigated URL")
    title: str = Field(default="", description="Page title")
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


class BrowserClickArgs(BaseModel):
    session_id: Optional[str] = Field(
        default=None, description="Session ID (auto-managed, usually not needed)"
    )
    selector: str = Field(description="CSS selector or XPath")
    timeout: int = Field(default=30000, description="Timeout in milliseconds")


class BrowserClickResult(BaseModel):
    success: bool = Field(description="Whether click succeeded")
    session_id: str = Field(description="Session ID")
    selector: str = Field(description="Clicked selector")
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


class BrowserFillArgs(BaseModel):
    session_id: Optional[str] = Field(
        default=None, description="Session ID (auto-managed, usually not needed)"
    )
    selector: str = Field(description="CSS selector or XPath")
    value: str = Field(description="Text value to fill")


class BrowserFillResult(BaseModel):
    success: bool = Field(description="Whether fill succeeded")
    session_id: str = Field(description="Session ID")
    selector: str = Field(description="Filled selector")
    value: str = Field(description="Filled value (preview)")
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


class BrowserScreenshotArgs(BaseModel):
    session_id: Optional[str] = Field(
        default=None, description="Session ID (auto-managed, usually not needed)"
    )
    full_page: bool = Field(default=False, description="Whether to capture full page")
    width: Optional[int] = Field(
        default=None,
        description="Desired output width in pixels (e.g., 2025, 1080, 1920). Set viewport size before screenshot.",
    )
    height: Optional[int] = Field(
        default=None,
        description="Desired output height in pixels (e.g., 2025, 1080, 1920). Set viewport size before screenshot.",
    )
    wait_for_lazy_load: bool = Field(
        default=False,
        description="Whether to scroll the page to trigger lazy-loaded content before screenshot. Only effective when full_page=True. Use this for pages with infinite scroll, lazy-loaded images, or dynamic content loading.",
    )
    output_filename: Optional[str] = Field(
        default=None,
        description="Custom filename for the screenshot (e.g., 'result.png'). If not provided, auto-generates with timestamp (e.g., 'screenshot_20250113_123456.png'). Always saves to output/ directory.",
    )


class BrowserScreenshotResult(BaseModel):
    success: bool = Field(description="Whether screenshot succeeded")
    session_id: str = Field(description="Session ID")
    screenshot: str = Field(default="", description="Base64 encoded screenshot")
    format: str = Field(description="Image format")
    full_page: bool = Field(description="Whether full page was captured")
    wait_for_lazy_load: bool = Field(description="Whether lazy loading was enabled")
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


class BrowserExtractTextArgs(BaseModel):
    session_id: Optional[str] = Field(
        default=None, description="Session ID (auto-managed, usually not needed)"
    )
    selector: str = Field(default="body", description="CSS selector or XPath")


class BrowserExtractTextResult(BaseModel):
    success: bool = Field(description="Whether extraction succeeded")
    session_id: str = Field(description="Session ID")
    selector: str = Field(description="Extracted selector")
    text: str = Field(default="", description="Extracted text")
    length: int = Field(default=0, description="Text length")
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


class BrowserEvaluateArgs(BaseModel):
    session_id: Optional[str] = Field(
        default=None, description="Session ID (auto-managed, usually not needed)"
    )
    javascript: str = Field(description="JavaScript code to execute")


class BrowserEvaluateResult(BaseModel):
    success: bool = Field(description="Whether execution succeeded")
    session_id: str = Field(description="Session ID")
    result: Any = Field(default=None, description="Execution result")
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


class BrowserSelectOptionArgs(BaseModel):
    session_id: Optional[str] = Field(
        default=None, description="Session ID (auto-managed, usually not needed)"
    )
    selector: str = Field(description="CSS selector for the select element")
    value: Optional[str] = Field(default=None, description="Option value to select")
    index: Optional[int] = Field(default=None, description="Option index to select")


class BrowserSelectOptionResult(BaseModel):
    success: bool = Field(description="Whether selection succeeded")
    session_id: str = Field(description="Session ID")
    selector: str = Field(description="Select element selector")
    selected_value: str = Field(default="", description="Selected option value")
    selected_index: Optional[int] = Field(
        default=None, description="Selected option index"
    )
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


class BrowserWaitForSelectorArgs(BaseModel):
    session_id: Optional[str] = Field(
        default=None, description="Session ID (auto-managed, usually not needed)"
    )
    selector: str = Field(description="CSS selector or XPath to wait for")
    timeout: int = Field(default=30000, description="Timeout in milliseconds")


class BrowserWaitForSelectorResult(BaseModel):
    success: bool = Field(description="Whether wait succeeded")
    session_id: str = Field(description="Session ID")
    selector: str = Field(description="Selector that was waited for")
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


class BrowserCloseArgs(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID to close (auto-managed, usually not needed)",
    )


class BrowserCloseResult(BaseModel):
    success: bool = Field(description="Whether close succeeded")
    session_id: str = Field(description="Closed session ID")
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


class BrowserPdfArgs(BaseModel):
    session_id: Optional[str] = Field(
        default=None, description="Session ID (auto-managed, usually not needed)"
    )
    output_filename: Optional[str] = Field(
        default=None,
        description="Output PDF filename (e.g., 'page.pdf'). Auto-generated with timestamp if not provided.",
    )
    landscape: bool = Field(
        default=False, description="PDF orientation (False=portrait, True=landscape)"
    )
    format: str = Field(default="A4", description="Paper format (A4, Letter, etc.)")
    print_background: bool = Field(
        default=True, description="Include background graphics in PDF"
    )


class BrowserPdfResult(BaseModel):
    success: bool = Field(description="Whether PDF generation succeeded")
    session_id: str = Field(description="Session ID")
    output_path: str = Field(
        description="Relative path to generated PDF file in workspace"
    )
    format: str = Field(description="File format (base64 or file)")
    size: int = Field(default=0, description="PDF file size in bytes")
    message: str = Field(description="Result message")
    error: str = Field(default="", description="Error message if failed")


# ============== Tool Implementations ==============


class BrowserNavigateTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Navigate to a URL in a browser session."""

    def __init__(
        self, task_id: Optional[str] = None, workspace: Optional["TaskWorkspace"] = None
    ):
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "browser_navigate"

    @property
    def description(self) -> str:
        return """Navigate to URL. Browser session auto-created.

        Workspace files: Use filename (e.g., "poster.html") - auto-searches input/output/temp dirs.

        Default: headless=False (shows browser window for debugging/interaction).
        Set headless=True only for simple content extraction without user interaction.

        Args:
            url: URL to navigate (http/https for websites, filename for workspace files)
            headless: Run mode (default: False). Use True only for automated tasks
            wait_until: "domcontentloaded" (fast) or "networkidle" (wait)
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "automation", "web", "navigation"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserNavigateArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserNavigateResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        # Use task_id as default session_id if not provided
        if not args.get("session_id") and self._task_id:
            args = dict(args)  # Make a mutable copy
            args["session_id"] = self._task_id

        # Convert workspace-relative paths to file:// URLs
        url = args.get("url", "")

        # Debug logging
        import logging

        logger = logging.getLogger(__name__)

        # ALWAYS log to help debugging
        logger.warning("[BrowserNavigateTool] ========== NAVIGATE START ==========")
        logger.warning(f"[BrowserNavigateTool] Input URL: {url}")
        logger.warning(
            f"[BrowserNavigateTool] Workspace available: {self._workspace is not None}"
        )

        if self._workspace is not None:
            logger.warning(
                f"[BrowserNavigateTool] Workspace path: {self._workspace.workspace_dir}"
            )
        else:
            logger.warning(
                "[BrowserNavigateTool] ⚠️  WARNING: Workspace is None! Cannot resolve relative paths."
            )

        if (
            self._workspace
            and url
            and not url.startswith(
                ("http://", "https://", "file://", "about:", "data:")
            )
        ):
            # Use workspace's intelligent file search (input → output → temp → root)
            try:
                resolved_path = self._workspace.resolve_path_with_search(url)
                logger.info(f"[BrowserNavigateTool] Resolved path: {resolved_path}")
                logger.info(
                    f"[BrowserNavigateTool] File exists: {resolved_path.exists()}"
                )

                if resolved_path.exists():
                    args = dict(args)  # Make a mutable copy
                    args["url"] = resolved_path.as_uri()
                    logger.info(
                        f"[BrowserNavigateTool] Converted to file:// URL: {args['url']}"
                    )
                else:
                    # File doesn't exist after search
                    logger.warning(
                        f"[BrowserNavigateTool] File not found after search: {resolved_path}"
                    )
                    # List what directories were searched
                    logger.info(
                        "[BrowserNavigateTool] Searched in: input -> output -> temp -> root"
                    )
                    return {
                        "success": False,
                        "session_id": args.get("session_id", ""),
                        "url": url,
                        "title": "",
                        "message": "",
                        "error": f"File not found: {url}. Searched in workspace directories (input/, output/, temp/). Please check if the file exists.",
                    }
            except ValueError as e:
                # Path is outside workspace
                logger.warning(f"[BrowserNavigateTool] Path error: {e}")
                return {
                    "success": False,
                    "session_id": args.get("session_id", ""),
                    "url": url,
                    "title": "",
                    "message": "",
                    "error": str(e),
                }
        else:
            logger.warning(
                f"[BrowserNavigateTool] Skipping path conversion (workspace={self._workspace is not None}, url={url[:50]})"
            )

        logger.warning(
            f"[BrowserNavigateTool] Calling browser_navigate with URL: {args.get('url', '')[:100]}"
        )

        result = await browser_navigate(**args)

        logger.warning(
            f"[BrowserNavigateTool] Navigation result: success={result.get('success')}"
        )
        if not result.get("success"):
            logger.warning(
                f"[BrowserNavigateTool] Navigation error: {result.get('error', 'Unknown error')[:200]}"
            )

        logger.warning("[BrowserNavigateTool] ========== NAVIGATE END ==========")

        return BrowserNavigateResult(**result).model_dump()

    async def setup(self, task_id: Optional[str] = None) -> None:
        """Setup called when task starts - store task_id for session tracking."""
        if task_id:
            self._task_id = task_id

    async def teardown(self, task_id: Optional[str] = None) -> None:
        """Cleanup browser sessions when task completes."""
        if self._task_id or task_id:
            target_task_id = self._task_id or task_id
            if target_task_id:  # Type guard for mypy
                try:
                    manager = get_browser_manager()
                    # Close the session associated with this task
                    # Session ID is typically the same as task_id in our pattern
                    await manager.close(target_task_id)
                    logger.info(
                        f"Cleaned up browser session for task {target_task_id} via teardown"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to cleanup browser session for task {target_task_id}: {e}",
                        exc_info=True,
                    )


class BrowserClickTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Click an element on the current page."""

    def __init__(self, task_id: Optional[str] = None):
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id

    @property
    def name(self) -> str:
        return "browser_click"

    @property
    def description(self) -> str:
        return """Click an element on the current page.

        Args:
            selector: CSS selector or XPath for element to click
            timeout: Max wait time in ms (default 30000)
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "automation", "interaction"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserClickArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserClickResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        # Use task_id as default session_id if not provided
        if not args.get("session_id") and self._task_id:
            args = dict(args)  # Make a mutable copy
            args["session_id"] = self._task_id

        result = await browser_click(**args)
        return BrowserClickResult(**result).model_dump()


class BrowserFillTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Fill an input field with text."""

    def __init__(self, task_id: Optional[str] = None):
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id

    @property
    def name(self) -> str:
        return "browser_fill"

    @property
    def description(self) -> str:
        return """Fill an input field with text.

        Args:
            selector: CSS selector or XPath for input element
            value: Text to fill
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "automation", "input", "form"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserFillArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserFillResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        # Use task_id as default session_id if not provided
        if not args.get("session_id") and self._task_id:
            args = dict(args)  # Make a mutable copy
            args["session_id"] = self._task_id

        result = await browser_fill(**args)
        return BrowserFillResult(**result).model_dump()


class BrowserScreenshotTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Take a screenshot of the current page."""

    def __init__(
        self, task_id: Optional[str] = None, workspace: Optional["TaskWorkspace"] = None
    ):
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "browser_screenshot"

    @property
    def description(self) -> str:
        return """Take a screenshot of the current page. Returns relative path to saved screenshot (output/ directory).

        Auto-saves to output/ with timestamp filename. Use output_filename for custom name.

        Args:
            full_page: Capture entire scrolling page (not just visible area)
            wait_for_lazy_load: Scroll to trigger lazy loading (use with full_page=True)
            output_filename: Custom filename (e.g., "result.png"). Auto-generated if not provided.
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "automation", "screenshot", "image"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserScreenshotArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserScreenshotResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        # Use task_id as default session_id if not provided
        if not args.get("session_id") and self._task_id:
            args = dict(args)  # Make a mutable copy
            args["session_id"] = self._task_id

        # Call async version directly
        import traceback

        try:
            result = await browser_screenshot(**args)
        except Exception as e:
            print(f"[BrowserScreenshotTool] Error type: {type(e).__name__}")
            print(f"[BrowserScreenshotTool] Error: {str(e)}")
            traceback.print_exc()
            raise

        # If workspace is available, save screenshot to file and return relative path
        if self._workspace and result.get("success"):
            try:
                import base64
                from datetime import datetime

                # Extract base64 data from data URI
                screenshot_data = result.get("screenshot", "")
                if screenshot_data.startswith("data:image/png;base64,"):
                    base64_data = screenshot_data.split(",", 1)[1]
                else:
                    base64_data = screenshot_data

                # Decode base64 to bytes
                image_bytes = base64.b64decode(base64_data)

                # Determine filename - always save to output directory
                output_filename = args.get("output_filename")
                if output_filename:
                    # Sanitize filename to prevent path traversal attacks
                    filename = os.path.basename(output_filename)
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.png"
                # Always save to output directory
                file_path = self._workspace.output_dir / filename

                # Save to file
                with open(file_path, "wb") as f:
                    f.write(image_bytes)

                relative_path = str(
                    file_path.relative_to(self._workspace.workspace_dir)
                )
                result["screenshot"] = relative_path
                result["format"] = "file"
                result["message"] = f"Screenshot saved to {relative_path}"
            except Exception as e:
                logger.error(
                    f"Failed to save screenshot to workspace: {e}", exc_info=True
                )
                result["message"] = (
                    f"Screenshot captured (base64 format, file save failed: {e})"
                )

        return BrowserScreenshotResult(**result).model_dump()


class BrowserExtractTextTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Extract text content from the page."""

    def __init__(self, task_id: Optional[str] = None):
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id

    @property
    def name(self) -> str:
        return "browser_extract_text"

    @property
    def description(self) -> str:
        return """Extract text content from page or element.

        Args:
            selector: CSS selector or XPath (default "body" for entire page)
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "automation", "scraping", "text"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserExtractTextArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserExtractTextResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        # Use task_id as default session_id if not provided
        if not args.get("session_id") and self._task_id:
            args = dict(args)  # Make a mutable copy
            args["session_id"] = self._task_id

        result = await browser_extract_text(**args)
        return BrowserExtractTextResult(**result).model_dump()


class BrowserEvaluateTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Execute JavaScript code in the browser."""

    def __init__(self, task_id: Optional[str] = None):
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id

    @property
    def name(self) -> str:
        return "browser_evaluate"

    @property
    def description(self) -> str:
        return """Execute JavaScript in browser context.

        For visual inspection before modifying pages (when vision tools are available):
        1. Use browser_screenshot to capture the current page
        2. Use vision tools to analyze the screenshot content and layout
        3. Use browser_evaluate with JavaScript to make targeted modifications

        Examples:
        - Change element style: document.querySelector('.header').style.backgroundColor = 'red'
        - Get element text: document.querySelector('h1').textContent
        - Scroll to element: document.querySelector('.footer').scrollIntoView()

        Args:
            javascript: JS code to execute (can access page DOM and make changes)
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "automation", "javascript", "advanced"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserEvaluateArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserEvaluateResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        # Use task_id as default session_id if not provided
        if not args.get("session_id") and self._task_id:
            args = dict(args)  # Make a mutable copy
            args["session_id"] = self._task_id

        result = await browser_evaluate(**args)
        return BrowserEvaluateResult(**result).model_dump()


class BrowserListSessionsTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """List all active browser sessions for debugging."""

    def __init__(self) -> None:
        self._visibility = ToolVisibility.PRIVATE  # Debug tool

    @property
    def name(self) -> str:
        return "browser_list_sessions"

    @property
    def description(self) -> str:
        return """List all active browser sessions (for debugging).

        Returns information about all active browser sessions, including:
        - Session IDs
        - Creation time
        - Last used time
        - Initialization status
        - Headless mode

        Use this tool to diagnose browser state issues.
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "debug", "internal"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserCloseArgs  # Empty args, reuse for simplicity

    def return_type(self) -> Type[BaseModel]:
        return BrowserCloseResult  # Reuse for simplicity

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        result = await browser_list_sessions()
        return result


class BrowserSelectOptionTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Select an option from a dropdown."""

    def __init__(self, task_id: Optional[str] = None):
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id

    @property
    def name(self) -> str:
        return "browser_select_option"

    @property
    def description(self) -> str:
        return """Select option from dropdown element.

        Args:
            selector: CSS selector for select element
            value: Option value to select (exclusive with index)
            index: Option index to select (exclusive with value)
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "automation", "interaction", "form"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserSelectOptionArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserSelectOptionResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        # Use task_id as default session_id if not provided
        if not args.get("session_id") and self._task_id:
            args = dict(args)  # Make a mutable copy
            args["session_id"] = self._task_id

        result = await browser_select_option(**args)
        return BrowserSelectOptionResult(**result).model_dump()


class BrowserWaitForSelectorTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Wait for an element to appear on the page."""

    def __init__(self, task_id: Optional[str] = None):
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id

    @property
    def name(self) -> str:
        return "browser_wait_for_selector"

    @property
    def description(self) -> str:
        return """Wait for element to appear on page.

        Args:
            selector: CSS selector or XPath to wait for
            timeout: Max wait time in ms (default 30000)
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "automation", "wait", "async"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserWaitForSelectorArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserWaitForSelectorResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        # Use task_id as default session_id if not provided
        if not args.get("session_id") and self._task_id:
            args = dict(args)  # Make a mutable copy
            args["session_id"] = self._task_id

        result = await browser_wait_for_selector(**args)
        return BrowserWaitForSelectorResult(**result).model_dump()


class BrowserCloseTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Close a browser session and free resources."""

    def __init__(self, task_id: Optional[str] = None):
        self._visibility = ToolVisibility.PRIVATE  # Internal tool
        self._task_id = task_id

    @property
    def name(self) -> str:
        return "browser_close"

    @property
    def description(self) -> str:
        return """Close a browser session and free resources. Sessions auto-close after 30min inactivity."""

    @property
    def tags(self) -> list[str]:
        return ["browser", "cleanup", "internal"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserCloseArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserCloseResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        # Use task_id as default session_id if not provided
        if not args.get("session_id") and self._task_id:
            args = dict(args)  # Make a mutable copy
            args["session_id"] = self._task_id

        result = await browser_close(**args)
        return BrowserCloseResult(**result).model_dump()


class BrowserPdfTool(AbstractBaseTool):
    category = ToolCategory.BROWSER
    """Save current page as PDF."""

    def __init__(
        self, task_id: Optional[str] = None, workspace: Optional["TaskWorkspace"] = None
    ):
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "browser_pdf"

    @property
    def description(self) -> str:
        return """Save current browser page as PDF.

        Generates a PDF of the current page content. Useful for:
        - Saving web pages as PDF documents
        - Archiving dynamic content
        - Converting HTML reports to PDF format

        Args:
            output_filename: Output PDF filename (e.g., 'page.pdf'). Auto-generated with timestamp if not provided.
            landscape: Orientation (default: False for portrait)
            format: Paper size (default: "A4", options: A4, Letter, etc.)
            print_background: Include background graphics (default: True)
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "automation", "pdf", "export"]

    def args_type(self) -> Type[BaseModel]:
        return BrowserPdfArgs

    def return_type(self) -> Type[BaseModel]:
        return BrowserPdfResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """
        Synchronous wrapper (not supported - use run_json_async instead).

        Browser tools are async-only. Please call them from async context.
        """
        raise NotImplementedError(
            f"{self.name} is async-only. Use await {self.name}() or call from async context."
        )

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        # Use task_id as default session_id if not provided
        if not args.get("session_id") and self._task_id:
            args = dict(args)  # Make a mutable copy
            args["session_id"] = self._task_id

        # Call core function to get PDF data (base64 encoded)
        result = await browser_pdf(**args)

        # If workspace is available, save PDF to file and return relative path
        if self._workspace and result.get("success"):
            try:
                import base64
                from datetime import datetime

                # Extract base64 data
                pdf_data = result.get("pdf", "")
                if pdf_data.startswith("data:application/pdf;base64,"):
                    base64_data = pdf_data.split(",", 1)[1]
                else:
                    base64_data = pdf_data

                # Decode base64 to bytes
                pdf_bytes = base64.b64decode(base64_data)

                # Determine filename - always save to output directory
                output_filename = args.get("output_filename")
                if output_filename:
                    # Sanitize filename to prevent path traversal attacks
                    filename = os.path.basename(output_filename)
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"page_{timestamp}.pdf"

                # Always save to output directory
                file_path = self._workspace.output_dir / filename

                # Save to file
                with open(file_path, "wb") as f:
                    f.write(pdf_bytes)

                relative_path = str(
                    file_path.relative_to(self._workspace.workspace_dir)
                )
                result["output_path"] = relative_path
                result["format"] = "file"
                result["message"] = f"PDF saved to {relative_path}"
            except Exception as e:
                logger.error(f"Failed to save PDF to workspace: {e}", exc_info=True)
                result["message"] = (
                    f"PDF generated (base64 format, file save failed: {e})"
                )

        return BrowserPdfResult(**result).model_dump()


# ============== Factory Functions ==============


def create_browser_tools(
    task_id: Optional[str] = None, workspace: Optional["TaskWorkspace"] = None
) -> list:
    """
    Create all browser automation tools for a task.

    Args:
        task_id: Optional task ID for session tracking
        workspace: Optional workspace for saving screenshots

    Returns:
        List of browser tool instances
    """
    return [
        BrowserNavigateTool(task_id=task_id, workspace=workspace),
        BrowserClickTool(task_id=task_id),
        BrowserFillTool(task_id=task_id),
        BrowserScreenshotTool(task_id=task_id, workspace=workspace),
        BrowserExtractTextTool(task_id=task_id),
        BrowserPdfTool(task_id=task_id, workspace=workspace),
        BrowserEvaluateTool(task_id=task_id),
        BrowserSelectOptionTool(task_id=task_id),
        BrowserWaitForSelectorTool(task_id=task_id),
        BrowserCloseTool(task_id=task_id),
        BrowserListSessionsTool(),  # Debug tool (no task_id needed)
    ]
