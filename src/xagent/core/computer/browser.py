from __future__ import annotations

import logging
import sys
from typing import Any, cast
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from ..tools.core.browser_use import BrowserSessionManager, get_browser_manager
from .environment import (
    ComputerEnvironment,
    ComputerFrameMismatchError,
    ComputerTargetNotFoundError,
)
from .schema import (
    ELEMENT_EXTRACTION_FAILED_KEY,
    ELEMENT_EXTRACTION_INCOMPLETE_KEY,
    ELEMENTS_TRUNCATED_KEY,
    MAX_OBSERVATION_ELEMENTS,
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerElement,
    ComputerElementSource,
    ComputerElementSurface,
    ComputerEnvironmentType,
    ComputerObservation,
    NormalizedPoint,
    ObservedUrl,
    Viewport,
)
from .store import ObservationStore

logger = logging.getLogger(__name__)

_OBSERVED_URL_ADAPTER = TypeAdapter(ObservedUrl)

_EDITABLE_ACTIVE_ELEMENT_SCRIPT = """() => {
  let currentDocument = document;
  const visited = new Set();
  while (currentDocument && !visited.has(currentDocument)) {
    visited.add(currentDocument);
    const node = currentDocument.activeElement;
    if (!node) return false;
    if (node.isContentEditable) return true;
    const tag = String(node.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea") {
      return !node.disabled && !node.readOnly;
    }
    if (tag !== "iframe" && tag !== "frame") return false;
    try {
      currentDocument = node.contentDocument;
    } catch (_) {
      return false;
    }
  }
  return false;
}"""

_INTERACTIVE_ELEMENTS_SCRIPT = """
(limit) => {
  const selector = [
    "a[href]", "button", "input", "textarea", "select", "summary",
    "[role='button']", "[role='link']", "[role='checkbox']", "[role='radio']",
    "[role='tab']", "[role='menuitem']", "[role='textbox']", "[onclick]",
    "[tabindex]", "[contenteditable='true']"
  ].join(",");
  const width = Math.max(1, window.innerWidth);
  const height = Math.max(1, window.innerHeight);
  const elements = [];
  let truncated = false;
  for (const node of document.querySelectorAll(selector)) {
    if (elements.length >= limit) {
      truncated = true;
      break;
    }
    const rect = node.getBoundingClientRect();
    const style = window.getComputedStyle(node);
    if (
      style.visibility === "hidden" ||
      style.visibility === "collapse" ||
      style.display === "none" ||
      Number(style.opacity) === 0 ||
      rect.width < 2 ||
      rect.height < 2 ||
      rect.right <= 0 ||
      rect.bottom <= 0 ||
      rect.left >= width ||
      rect.top >= height
    ) {
      continue;
    }
    const left = Math.max(0, rect.left);
    const top = Math.max(0, rect.top);
    const right = Math.min(width, rect.right);
    const bottom = Math.min(height, rect.bottom);
    const inputType = String(node.getAttribute("type") || "").toLowerCase();
    const autocomplete = String(
      node.getAttribute("autocomplete") || ""
    ).toLowerCase();
    const sensitive = (
      inputType === "password" ||
      inputType === "hidden" ||
      autocomplete === "current-password" ||
      autocomplete === "new-password" ||
      autocomplete === "one-time-code" ||
      autocomplete === "webauthn" ||
      autocomplete.startsWith("cc-")
    );
    const text = String(node.innerText || "").trim().slice(0, 240);
    const label = String(
      node.getAttribute("aria-label") ||
      node.getAttribute("title") ||
      node.getAttribute("placeholder") ||
      text
    ).trim().slice(0, 240);
    elements.push({
      bounds: {
        x: left / width,
        y: top / height,
        width: Math.max(0.000001, (right - left) / width),
        height: Math.max(0.000001, (bottom - top) / height)
      },
      label: label || null,
      role: node.getAttribute("role") || node.tagName.toLowerCase(),
      text: text || null,
      metadata: {
        tag: node.tagName.toLowerCase(),
        input_type: inputType || null,
        autocomplete: autocomplete || null,
        sensitive,
        disabled: Boolean(node.disabled),
        focused: node === document.activeElement
      }
    });
  }
  return {elements, truncated};
}
"""


class BrowserComputerEnvironment(ComputerEnvironment):
    """Ephemeral Playwright implementation of a computer environment."""

    def __init__(
        self,
        *,
        session_id: str,
        workspace: Any,
        manager: BrowserSessionManager | None = None,
        observation_store: ObservationStore | None = None,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
    ) -> None:
        super().__init__(session_id)
        if viewport_width <= 0 or viewport_height <= 0:
            raise ValueError("viewport dimensions must be positive")
        self.workspace = workspace
        self.manager = manager or get_browser_manager()
        self.observation_store = observation_store or ObservationStore(workspace)
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self._manager_session_id = f"{self.session_id}:computer"
        self._browser_session: Any | None = None

    async def _get_page(self, *, reject_recreated_session: bool) -> Any:
        session = await self.manager.get_or_create(
            self._manager_session_id,
            headless=self.headless,
        )
        recreated = (
            self._browser_session is not None and session is not self._browser_session
        )
        self._browser_session = session
        if recreated:
            self._invalidate_observation()
            if reject_recreated_session:
                raise ComputerFrameMismatchError(
                    "browser session changed after the expected frame was captured; "
                    "request a fresh observation before another action"
                )
        page = await session.get_page()
        requested = {
            "width": self.viewport_width,
            "height": self.viewport_height,
        }
        if page.viewport_size != requested:
            await page.set_viewport_size(cast(Any, requested))
        return page

    async def _close(self) -> None:
        await self.manager.close(self._manager_session_id)
        self._browser_session = None

    async def _observe(self) -> ComputerObservation:
        return await self._capture_observation(
            await self._get_page(reject_recreated_session=False)
        )

    async def _execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        page = await self._get_page(reject_recreated_session=True)
        action = batch.actions[0]
        await self._execute_action(page, action)
        if action.type not in {
            ComputerActionType.OBSERVE,
            ComputerActionType.SCREENSHOT,
            ComputerActionType.WAIT,
        }:
            await page.wait_for_timeout(250)
        return await self._capture_observation(page)

    async def _capture_observation(self, page: Any) -> ComputerObservation:
        viewport = await self._read_viewport(page)
        frame_id = f"frame-{uuid4().hex}"
        screenshot_bytes = await page.screenshot(full_page=False, type="png")
        screenshot = await self.observation_store.save_screenshot(
            session_id=self.session_id,
            frame_id=frame_id,
            image_bytes=screenshot_bytes,
            mime_type="image/png",
            viewport=viewport,
            text_fallback="Current browser viewport screenshot.",
        )
        try:
            title = await page.title()
        except Exception:  # noqa: BLE001 - title is optional observation metadata.
            title = None

        metadata: dict[str, Any] = {
            "browser_runtime_kind": "ephemeral_playwright",
            "supported_actions": [action.value for action in ComputerActionType],
        }
        try:
            elements, truncated = await self._read_interactive_elements(page)
        except Exception:  # noqa: BLE001 - screenshots remain useful without DOM hints.
            logger.warning(
                "Failed to collect interactive browser elements for %s",
                self.session_id,
                exc_info=True,
            )
            elements = []
            truncated = False
            metadata[ELEMENT_EXTRACTION_FAILED_KEY] = True
        if truncated:
            metadata[ELEMENTS_TRUNCATED_KEY] = True
        frames = getattr(page, "frames", ())
        if isinstance(frames, (list, tuple)) and len(frames) > 1:
            # This adapter intentionally reports only top-level DOM hints. The
            # screenshot remains authoritative for cross-origin child frames.
            metadata[ELEMENT_EXTRACTION_INCOMPLETE_KEY] = True

        raw_active_url = str(getattr(page, "url", "") or "").strip() or None
        active_url: str | None = None
        if raw_active_url is not None:
            try:
                active_url = _OBSERVED_URL_ADAPTER.validate_python(raw_active_url)
            except ValidationError:
                metadata["active_url_unavailable"] = True
        return ComputerObservation(
            session_id=self.session_id,
            frame_id=frame_id,
            environment=ComputerEnvironmentType.BROWSER,
            viewport=viewport,
            screenshot=screenshot,
            elements=elements,
            active_url=active_url,
            title=str(title) if title else None,
            metadata=metadata,
        )

    async def _read_viewport(self, page: Any) -> Viewport:
        size = page.viewport_size
        if not size:
            size = await page.evaluate(
                "() => ({width: window.innerWidth, height: window.innerHeight})"
            )
        try:
            device_pixel_ratio = float(
                await page.evaluate("() => window.devicePixelRatio || 1")
            )
        except Exception:  # noqa: BLE001 - DPR defaults safely.
            device_pixel_ratio = 1.0
        return Viewport(
            width=int(size["width"]),
            height=int(size["height"]),
            device_pixel_ratio=device_pixel_ratio,
        )

    async def _read_interactive_elements(
        self,
        page: Any,
    ) -> tuple[list[ComputerElement], bool]:
        payload = await page.evaluate(
            _INTERACTIVE_ELEMENTS_SCRIPT,
            MAX_OBSERVATION_ELEMENTS,
        )
        if not isinstance(payload, dict):
            return [], False
        raw_elements = payload.get("elements")
        if not isinstance(raw_elements, list):
            return [], bool(payload.get("truncated"))
        elements: list[ComputerElement] = []
        for index, raw_element in enumerate(
            raw_elements[:MAX_OBSERVATION_ELEMENTS],
            start=1,
        ):
            if not isinstance(raw_element, dict):
                continue
            try:
                elements.append(
                    ComputerElement(
                        element_id=f"dom-{index}",
                        source=ComputerElementSource.DOM,
                        bounds=raw_element["bounds"],
                        surface=ComputerElementSurface.DOCUMENT,
                        label=raw_element.get("label"),
                        role=raw_element.get("role"),
                        text=raw_element.get("text"),
                        metadata=raw_element.get("metadata") or {},
                    )
                )
            except (KeyError, TypeError, ValueError):
                logger.debug("Skipping invalid DOM element payload", exc_info=True)
        return elements, bool(payload.get("truncated"))

    async def _execute_action(self, page: Any, action: ComputerAction) -> None:
        if action.type in {
            ComputerActionType.OBSERVE,
            ComputerActionType.SCREENSHOT,
        }:
            return
        if action.type is ComputerActionType.NAVIGATE:
            if action.url is None:
                raise ValueError("navigate action requires a URL")
            await page.goto(
                self._resolve_navigation_url(action.url),
                wait_until="domcontentloaded",
                timeout=30_000,
            )
            return
        if action.type is ComputerActionType.WAIT:
            await page.wait_for_timeout(action.duration_ms)
            return
        if action.type is ComputerActionType.KEYPRESS:
            await page.keyboard.press(self._playwright_key_chord(action.keys))
            return
        if action.type is ComputerActionType.TYPE:
            await page.keyboard.insert_text(action.text or "")
            return
        if action.type is ComputerActionType.REPLACE_TEXT:
            x, y = self._target_pixels(action)
            await page.mouse.click(x, y)
            editable = await page.evaluate(_EDITABLE_ACTIVE_ELEMENT_SCRIPT)
            if not editable:
                raise ValueError("replace_text target is not an editable element")
            modifier = "Meta" if sys.platform == "darwin" else "Control"
            await page.keyboard.press(f"{modifier}+A")
            await page.keyboard.insert_text(action.text or "")
            return
        if action.type is ComputerActionType.SCROLL:
            viewport = self._current_viewport()
            await page.mouse.wheel(
                action.delta_x * viewport.width,
                action.delta_y * viewport.height,
            )
            return
        if action.type is ComputerActionType.DRAG:
            if action.start is None or action.end is None:
                raise ValueError("drag action requires start and end points")
            start_x, start_y = self._point_pixels(action.start)
            end_x, end_y = self._point_pixels(action.end)
            await page.mouse.move(start_x, start_y)
            await page.mouse.down()
            await page.mouse.move(end_x, end_y, steps=10)
            await page.mouse.up()
            return

        x, y = self._target_pixels(action)
        if action.type is ComputerActionType.CLICK:
            await page.mouse.click(x, y)
        elif action.type is ComputerActionType.DOUBLE_CLICK:
            await page.mouse.dblclick(x, y)
        elif action.type is ComputerActionType.MOVE:
            await page.mouse.move(x, y)
        else:  # pragma: no cover - schema and supported action list stay aligned.
            raise ValueError(
                f"unsupported browser computer action: {action.type.value}"
            )

    def _target_pixels(self, action: ComputerAction) -> tuple[float, float]:
        if action.target is None:
            raise ValueError(f"{action.type.value} requires a target")
        if action.target.point is not None:
            return self._point_pixels(action.target.point)
        target_id = action.target.element_id
        observation = self.current_observation
        if target_id is None or observation is None:
            raise ValueError("element target requires a current observation")
        element = next(
            (
                element
                for element in observation.elements
                if element.element_id == target_id
            ),
            None,
        )
        if element is None:
            raise ComputerTargetNotFoundError(
                f"element {target_id!r} is not present in the current observation"
            )
        return self._point_pixels(
            NormalizedPoint(
                x=element.bounds.x + element.bounds.width / 2,
                y=element.bounds.y + element.bounds.height / 2,
            )
        )

    def _point_pixels(self, point: NormalizedPoint) -> tuple[float, float]:
        viewport = self._current_viewport()
        return (
            min(point.x * viewport.width, viewport.width - 1),
            min(point.y * viewport.height, viewport.height - 1),
        )

    def _current_viewport(self) -> Viewport:
        observation = self.current_observation
        if observation is None:
            raise RuntimeError("computer action requires a current observation")
        return observation.viewport

    @staticmethod
    def _resolve_navigation_url(raw_url: str) -> str:
        url = raw_url.strip()
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError("navigate URL must use absolute HTTP or HTTPS")
        return url

    @staticmethod
    def _playwright_key_chord(keys: list[str]) -> str:
        aliases = {
            "ALT": "Alt",
            "ARROWDOWN": "ArrowDown",
            "ARROWLEFT": "ArrowLeft",
            "ARROWRIGHT": "ArrowRight",
            "ARROWUP": "ArrowUp",
            "BACKSPACE": "Backspace",
            "CMD": "Meta",
            "COMMAND": "Meta",
            "CTRL": "Control",
            "DELETE": "Delete",
            "END": "End",
            "ENTER": "Enter",
            "ESC": "Escape",
            "ESCAPE": "Escape",
            "HOME": "Home",
            "META": "Meta",
            "PAGEDOWN": "PageDown",
            "PAGEUP": "PageUp",
            "SHIFT": "Shift",
            "SPACE": "Space",
            "TAB": "Tab",
        }
        normalized = [aliases.get(key.upper(), key) for key in keys]
        modifiers = {"Alt", "Control", "Meta", "Shift"}
        if len(normalized) > 1 and any(key not in modifiers for key in normalized[:-1]):
            raise ValueError(
                "keypress accepts one key chord; send sequential keys as separate actions"
            )
        return "+".join(normalized)
