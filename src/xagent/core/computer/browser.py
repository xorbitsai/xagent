from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from ..tools.core.browser_use import BrowserSessionManager, get_browser_manager
from .environment import ComputerEnvironment, ComputerTargetNotFoundError
from .schema import (
    ELEMENT_EXTRACTION_FAILED_KEY,
    ELEMENTS_TRUNCATED_KEY,
    MAX_OBSERVATION_ELEMENTS,
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


class ComputerTargetObstructedError(RuntimeError):
    """Raised when the coordinate for a target resolves to another element."""


@dataclass(frozen=True)
class _ElementHitTarget:
    """Where an extracted element lives, so a click can be hit-tested."""

    frame: Any
    marker: str
    offset_x: float
    offset_y: float


#: Attribute stamped on extracted nodes so a later hit test can prove that a
#: coordinate still resolves to the element the model asked for.
_ELEMENT_MARKER_ATTRIBUTE = "data-xagent-eid"

_INTERACTIVE_ELEMENTS_SCRIPT = """
(options) => {
  const {frameToken, limit, markerAttribute, offsetX, offsetY,
         rootWidth, rootHeight} = options;
  const selector = [
    "a[href]", "button", "input", "textarea", "select", "summary",
    "[role='button']", "[role='link']", "[role='checkbox']", "[role='radio']",
    "[role='tab']", "[role='menuitem']", "[onclick]", "[tabindex]",
    "[contenteditable='true']"
  ].join(",");
  // Bounds are reported in the top-level viewport's coordinate space so that
  // a single normalized point is meaningful across nested frames.
  const width = Math.max(1, rootWidth);
  const height = Math.max(1, rootHeight);
  const localWidth = Math.max(1, window.innerWidth);
  const localHeight = Math.max(1, window.innerHeight);
  const elements = [];
  let truncated = false;

  const collect = (root, out) => {
    let nodes;
    try {
      nodes = root.querySelectorAll(selector);
    } catch (error) {
      return;
    }
    for (const node of nodes) out.push(node);
    // Shadow roots are invisible to querySelectorAll on the host document, so
    // open shadow trees are walked explicitly.
    let hosts;
    try {
      hosts = root.querySelectorAll("*");
    } catch (error) {
      return;
    }
    for (const host of hosts) {
      if (host.shadowRoot) collect(host.shadowRoot, out);
    }
  };

  const candidates = [];
  collect(document, candidates);

  for (const node of candidates) {
    if (elements.length >= limit) {
      truncated = true;
      break;
    }
    let rect;
    let style;
    try {
      rect = node.getBoundingClientRect();
      style = window.getComputedStyle(node);
    } catch (error) {
      continue;
    }
    if (
      style.visibility === "hidden" ||
      style.visibility === "collapse" ||
      style.display === "none" ||
      Number(style.opacity) === 0 ||
      rect.width < 2 ||
      rect.height < 2 ||
      rect.right <= 0 ||
      rect.bottom <= 0 ||
      rect.left >= localWidth ||
      rect.top >= localHeight
    ) {
      continue;
    }
    const left = Math.max(0, rect.left) + offsetX;
    const top = Math.max(0, rect.top) + offsetY;
    const right = Math.min(localWidth, rect.right) + offsetX;
    const bottom = Math.min(localHeight, rect.bottom) + offsetY;
    if (left >= width || top >= height || right <= 0 || bottom <= 0) continue;
    const clippedLeft = Math.max(0, Math.min(left, width));
    const clippedTop = Math.max(0, Math.min(top, height));
    const clippedRight = Math.max(clippedLeft, Math.min(right, width));
    const clippedBottom = Math.max(clippedTop, Math.min(bottom, height));
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
    const marker = frameToken + ":" + (elements.length + 1);
    try {
      node.setAttribute(markerAttribute, marker);
    } catch (error) {
      // A read-only node can still be reported; it just cannot be hit-tested.
    }
    elements.push({
      marker,
      bounds: {
        x: clippedLeft / width,
        y: clippedTop / height,
        width: Math.max(0.000001, (clippedRight - clippedLeft) / width),
        height: Math.max(0.000001, (clippedBottom - clippedTop) / height)
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

_HIT_TEST_SCRIPT = """
(options) => {
  const {x, y, markerAttribute} = options;
  let node = document.elementFromPoint(x, y);
  if (node === null) return {marker: null, tag: null, found: false};
  const tag = node.tagName ? node.tagName.toLowerCase() : null;
  // Walk outward: a coordinate usually lands on a label or icon inside the
  // control that was actually extracted.
  while (node !== null) {
    const marker = node.getAttribute
      ? node.getAttribute(markerAttribute)
      : null;
    if (marker) return {marker, tag, found: true};
    node = node.parentElement;
  }
  return {marker: null, tag, found: true};
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
        self._hit_targets: dict[str, _ElementHitTarget] = {}

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
        frame_id = f"frame-{uuid4().hex}"
        screenshot_bytes = await page.screenshot(full_page=False, type="png")
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

        extraction_failed = False
        truncated = False
        self._hit_targets = {}
        try:
            elements, truncated = await self._read_interactive_elements(
                page,
                viewport=viewport,
                frame_id=frame_id,
            )
        except Exception:  # noqa: BLE001 - screenshots remain usable without DOM hints.
            logger.warning(
                "Failed to collect interactive browser elements for %s",
                self.session_id,
                exc_info=True,
            )
            elements = []
            extraction_failed = True

        active_url = str(getattr(page, "url", "") or "").strip() or None
        metadata: dict[str, Any] = {
            "browser_runtime_kind": self.session_binding.runtime_kind.value,
            "user_takeover_available": self.session_binding.is_persistent,
        }
        if extraction_failed:
            metadata[ELEMENT_EXTRACTION_FAILED_KEY] = True
        if truncated:
            metadata[ELEMENTS_TRUNCATED_KEY] = True
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
        *,
        viewport: Viewport,
        frame_id: str,
    ) -> tuple[list[ComputerElement], bool]:
        """Extract interactive elements from the page and its visible frames.

        Bounds from nested frames are translated into the top-level viewport so
        one normalized coordinate space covers the whole screenshot.
        """
        elements: list[ComputerElement] = []
        truncated = False
        for frame_index, frame in enumerate(self._iter_frames(page)):
            remaining = MAX_OBSERVATION_ELEMENTS - len(elements)
            if remaining <= 0:
                truncated = True
                break
            offset = await self._frame_offset(page, frame)
            if offset is None:
                continue
            offset_x, offset_y = offset
            try:
                payload = await frame.evaluate(
                    _INTERACTIVE_ELEMENTS_SCRIPT,
                    {
                        "frameToken": f"{frame_id}-f{frame_index}",
                        "limit": remaining,
                        "markerAttribute": _ELEMENT_MARKER_ATTRIBUTE,
                        "offsetX": offset_x,
                        "offsetY": offset_y,
                        "rootWidth": viewport.width,
                        "rootHeight": viewport.height,
                    },
                )
            except Exception:  # noqa: BLE001 - a detached frame is not fatal.
                if frame is self._main_frame(page):
                    raise
                logger.debug(
                    "Skipping frame %s during element extraction",
                    frame_index,
                    exc_info=True,
                )
                continue
            frame_elements, frame_truncated = self._parse_frame_elements(
                payload,
                frame=frame,
                offset_x=offset_x,
                offset_y=offset_y,
                start_index=len(elements) + 1,
            )
            elements.extend(frame_elements)
            truncated = truncated or frame_truncated
        return elements, truncated

    def _parse_frame_elements(
        self,
        payload: Any,
        *,
        frame: Any,
        offset_x: float,
        offset_y: float,
        start_index: int,
    ) -> tuple[list[ComputerElement], bool]:
        raw_elements = payload.get("elements") if isinstance(payload, dict) else payload
        truncated = (
            bool(payload.get("truncated")) if isinstance(payload, dict) else False
        )
        if not isinstance(raw_elements, list):
            return [], truncated
        elements: list[ComputerElement] = []
        index = start_index
        for entry in raw_elements:
            if not isinstance(entry, dict):
                continue
            try:
                metadata = entry.get("metadata") or {}
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
                        "focused",
                    )
                    if key in metadata
                }
                element_id = f"dom-{index}"
                elements.append(
                    ComputerElement(
                        element_id=element_id,
                        source=ComputerElementSource.DOM,
                        bounds=entry["bounds"],
                        label="Sensitive input" if sensitive else entry.get("label"),
                        role=entry.get("role"),
                        text=None if sensitive else entry.get("text"),
                        metadata={
                            **safe_metadata,
                            "sensitive": sensitive,
                        },
                    )
                )
            except (KeyError, TypeError, ValueError):
                logger.debug("Skipping invalid DOM element payload", exc_info=True)
                continue
            marker = entry.get("marker")
            if isinstance(marker, str) and marker:
                self._hit_targets[element_id] = _ElementHitTarget(
                    frame=frame,
                    marker=marker,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
            index += 1
        return elements, truncated

    @staticmethod
    def _main_frame(page: Any) -> Any:
        return getattr(page, "main_frame", page)

    @classmethod
    def _iter_frames(cls, page: Any) -> list[Any]:
        """Return the main frame first, then any additional frames."""
        main_frame = cls._main_frame(page)
        frames = list(getattr(page, "frames", []) or [])
        if not frames:
            return [main_frame]
        ordered = [frame for frame in frames if frame is main_frame]
        ordered.extend(frame for frame in frames if frame is not main_frame)
        return ordered or [main_frame]

    async def _frame_offset(
        self,
        page: Any,
        frame: Any,
    ) -> tuple[float, float] | None:
        """Offset of ``frame`` inside the top-level viewport, or None if hidden."""
        if frame is self._main_frame(page):
            return (0.0, 0.0)
        try:
            frame_element = await frame.frame_element()
            box = await frame_element.bounding_box()
        except Exception:  # noqa: BLE001 - detached or cross-process frames.
            return None
        if not isinstance(box, dict):
            return None
        return (float(box.get("x", 0.0)), float(box.get("y", 0.0)))

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
                await self._verify_hit_target(action, x, y)
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
            await self._verify_hit_target(action, x, y)
            await page.mouse.click(x, y)
        elif action.type == ComputerActionType.DOUBLE_CLICK:
            await self._verify_hit_target(action, x, y)
            await page.mouse.dblclick(x, y)
        elif action.type == ComputerActionType.MOVE:
            await page.mouse.move(x, y)
        else:
            raise ValueError(
                f"unsupported browser computer action: {action.type.value}"
            )

    async def _verify_hit_target(
        self,
        action: ComputerAction,
        x: float,
        y: float,
    ) -> None:
        """Refuse a click whose coordinate resolves to a different element.

        The center of an element's box is not necessarily clickable: a sticky
        header, a consent overlay, or a wrapped inline link can put something
        else on top. Verifying first turns a silent mis-click into an error the
        model can react to.
        """
        target = action.target
        if target is None or target.element_id is None:
            return
        hit_target = self._hit_targets.get(target.element_id)
        if hit_target is None:
            return
        try:
            result = await hit_target.frame.evaluate(
                _HIT_TEST_SCRIPT,
                {
                    "x": x - hit_target.offset_x,
                    "y": y - hit_target.offset_y,
                    "markerAttribute": _ELEMENT_MARKER_ATTRIBUTE,
                },
            )
        except Exception:  # noqa: BLE001 - verification is best effort.
            logger.debug(
                "Could not hit-test %s before clicking",
                target.element_id,
                exc_info=True,
            )
            return
        if not isinstance(result, dict) or not result.get("found"):
            return
        marker = result.get("marker")
        if marker == hit_target.marker:
            return
        obstruction = str(result.get("tag") or "another element")
        raise ComputerTargetObstructedError(
            f"{target.element_id} is covered by {obstruction} at the clicked "
            "position. Take a fresh screenshot, then dismiss the overlay, "
            "scroll the target into the clear, or click a different element."
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
                f"element {target_id!r} is not present in frame "
                f"{observation.frame_id!r}"
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
