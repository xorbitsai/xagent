from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from ..tools.core.browser_use import BrowserSessionManager, get_browser_manager
from .environment import ComputerEnvironment
from .schema import (
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerElement,
    ComputerElementSource,
    ComputerEnvironmentType,
    ComputerObservation,
    NormalizedPoint,
    Viewport,
)
from .session import ComputerSessionBinding
from .store import ObservationStore

logger = logging.getLogger(__name__)

_INTERACTIVE_ELEMENTS_SCRIPT = """
() => {
  const selector = [
    "a[href]", "button", "input", "textarea", "select", "summary",
    "[role='button']", "[role='link']", "[role='checkbox']", "[role='radio']",
    "[role='tab']", "[role='menuitem']", "[onclick]", "[tabindex]"
  ].join(",");
  const width = Math.max(1, window.innerWidth);
  const height = Math.max(1, window.innerHeight);
  const elements = [];
  for (const node of document.querySelectorAll(selector)) {
    const rect = node.getBoundingClientRect();
    const style = window.getComputedStyle(node);
    if (
      style.visibility === "hidden" ||
      style.display === "none" ||
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
    const safeValue = sensitive ? "" : String(node.value || "");
    const text = String(node.innerText || safeValue).trim().slice(0, 240);
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
        disabled: Boolean(node.disabled)
      }
    });
    if (elements.length >= 100) break;
  }
  return elements;
}
"""


class BrowserComputerEnvironment(ComputerEnvironment):
    """ComputerEnvironment backed by the existing Playwright browser sessions."""

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
        session_binding: ComputerSessionBinding | None = None,
    ) -> None:
        super().__init__(session_id)
        if viewport_width <= 0 or viewport_height <= 0:
            raise ValueError("viewport dimensions must be positive")
        self.workspace = workspace
        self.manager = manager or get_browser_manager()
        self.observation_store = observation_store or ObservationStore(workspace)
        self.session_binding = session_binding or ComputerSessionBinding()
        self.headless = False if self.session_binding.is_persistent else headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

    async def _get_page(self) -> Any:
        manager_session_id = self.session_binding.manager_session_id(self.session_id)
        session = await self.manager.get_or_create(
            manager_session_id,
            headless=self.headless,
            persistent_profile_dir=self.session_binding.persistent_profile_dir(),
            owner_id=self.session_binding.manager_owner_id(),
        )
        page = await session.get_page()
        requested = {
            "width": self.viewport_width,
            "height": self.viewport_height,
        }
        if page.viewport_size != requested:
            await page.set_viewport_size(cast(Any, requested))
        return page

    async def close(self) -> None:
        await self.manager.close(
            self.session_binding.manager_session_id(self.session_id)
        )

    async def _observe(self) -> ComputerObservation:
        page = await self._get_page()
        return await self._capture_observation(page)

    async def _execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        page = await self._get_page()
        for action in batch.actions:
            await self._execute_action(page, action)
        if any(
            action.type not in {ComputerActionType.SCREENSHOT, ComputerActionType.WAIT}
            for action in batch.actions
        ):
            await page.wait_for_timeout(250)
        return await self._capture_observation(page)

    async def _capture_observation(self, page: Any) -> ComputerObservation:
        viewport = await self._read_viewport(page)
        screenshot_bytes = await page.screenshot(full_page=False, type="png")
        frame_id = f"frame-{uuid4().hex}"
        screenshot = self.observation_store.save_screenshot(
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
        try:
            elements = await self._read_interactive_elements(page)
        except Exception:  # noqa: BLE001 - screenshots remain usable without DOM hints.
            logger.warning(
                "Failed to collect interactive browser elements for %s",
                self.session_id,
                exc_info=True,
            )
            elements = []
        active_url = str(getattr(page, "url", "") or "").strip() or None
        return ComputerObservation(
            session_id=self.session_id,
            frame_id=frame_id,
            environment=ComputerEnvironmentType.BROWSER,
            viewport=viewport,
            screenshot=screenshot,
            elements=elements,
            active_url=active_url,
            title=str(title) if title else None,
            metadata={
                "browser_runtime_kind": self.session_binding.runtime_kind.value,
                "user_takeover_available": self.session_binding.is_persistent,
            },
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

    async def _read_interactive_elements(self, page: Any) -> list[ComputerElement]:
        raw_elements = await page.evaluate(_INTERACTIVE_ELEMENTS_SCRIPT)
        if not isinstance(raw_elements, list):
            return []
        elements: list[ComputerElement] = []
        for index, payload in enumerate(raw_elements[:100], start=1):
            if not isinstance(payload, dict):
                continue
            try:
                metadata = payload.get("metadata") or {}
                if not isinstance(metadata, dict):
                    metadata = {}
                sensitive = self._is_sensitive_element_metadata(metadata)
                safe_metadata = {
                    key: metadata[key]
                    for key in (
                        "tag",
                        "input_type",
                        "autocomplete",
                        "disabled",
                    )
                    if key in metadata
                }
                elements.append(
                    ComputerElement(
                        element_id=f"dom-{index}",
                        source=ComputerElementSource.DOM,
                        bounds=payload["bounds"],
                        label="Sensitive input" if sensitive else payload.get("label"),
                        role=payload.get("role"),
                        text=None if sensitive else payload.get("text"),
                        metadata={
                            **safe_metadata,
                            "sensitive": sensitive,
                        },
                    )
                )
            except (KeyError, TypeError, ValueError):
                logger.debug("Skipping invalid DOM element payload", exc_info=True)
        return elements

    @staticmethod
    def _is_sensitive_element_metadata(metadata: dict[str, Any]) -> bool:
        if metadata.get("sensitive") is True:
            return True
        input_type = str(metadata.get("input_type") or "").strip().lower()
        autocomplete = str(metadata.get("autocomplete") or "").strip().lower()
        return (
            input_type in {"password", "hidden"}
            or autocomplete
            in {
                "current-password",
                "new-password",
                "one-time-code",
                "webauthn",
            }
            or autocomplete.startswith("cc-")
        )

    async def _execute_action(self, page: Any, action: ComputerAction) -> None:
        if action.type == ComputerActionType.SCREENSHOT:
            return
        if action.type == ComputerActionType.NAVIGATE:
            assert action.url is not None
            await page.goto(
                self._resolve_navigation_url(action.url),
                wait_until="domcontentloaded",
                timeout=30_000,
            )
            return
        if action.type == ComputerActionType.WAIT:
            await page.wait_for_timeout(action.duration_ms or 1_000)
            return
        if action.type == ComputerActionType.KEYPRESS:
            await page.keyboard.press(self._playwright_key_chord(action.keys))
            return
        if action.type == ComputerActionType.TYPE:
            if action.target is not None:
                x, y = self._target_pixels(action)
                await page.mouse.click(x, y)
            await page.keyboard.insert_text(action.text or "")
            return
        if action.type == ComputerActionType.SCROLL:
            if action.target is not None:
                x, y = self._target_pixels(action)
                await page.mouse.move(x, y)
            viewport = self._current_viewport()
            await page.mouse.wheel(
                action.delta_x * viewport.width,
                action.delta_y * viewport.height,
            )
            return
        if action.type == ComputerActionType.DRAG:
            assert action.start is not None and action.end is not None
            start_x, start_y = self._point_pixels(action.start)
            end_x, end_y = self._point_pixels(action.end)
            await page.mouse.move(start_x, start_y)
            await page.mouse.down()
            steps = max(1, min(50, (action.duration_ms or 250) // 25))
            await page.mouse.move(end_x, end_y, steps=steps)
            await page.mouse.up()
            return

        x, y = self._target_pixels(action)
        if action.type == ComputerActionType.CLICK:
            await page.mouse.click(x, y)
        elif action.type == ComputerActionType.DOUBLE_CLICK:
            await page.mouse.dblclick(x, y)
        elif action.type == ComputerActionType.MOVE:
            await page.mouse.move(x, y)
        else:
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
            element
            for element in observation.elements
            if element.element_id == target_id
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
        if self.current_observation is None:
            raise RuntimeError("computer action requires a current observation")
        return self.current_observation.viewport

    def _resolve_navigation_url(self, raw_url: str) -> str:
        url = raw_url.strip()
        parsed = urlsplit(url)
        if parsed.scheme in {"http", "https"}:
            return url
        if parsed.scheme == "about" and url == "about:blank":
            return url
        if parsed.scheme == "file":
            if parsed.netloc:
                raise ValueError("remote file URLs are not supported")
            path = Path(unquote(parsed.path))
            resolve_path = getattr(self.workspace, "resolve_path", None)
            if not callable(resolve_path):
                raise ValueError("workspace cannot authorize local browser paths")
            return Path(resolve_path(str(path), default_dir="input")).as_uri()
        if not parsed.scheme:
            resolve_search = getattr(self.workspace, "resolve_path_with_search", None)
            if callable(resolve_search):
                try:
                    return Path(resolve_search(url)).as_uri()
                except (FileNotFoundError, ValueError):
                    pass
        raise ValueError(
            "navigate URL must use http://, https://, about:blank, or an allowed "
            "workspace file"
        )

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
        return "+".join(aliases.get(key.upper(), key) for key in keys)
