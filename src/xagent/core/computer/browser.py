from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from ..tools.core.browser_use import BrowserSessionManager, get_browser_manager
from .environment import ComputerEnvironment, ComputerTargetNotFoundError
from .policy import (
    find_computer_target_element,
    navigation_block_reason,
    normalize_host_patterns,
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

    handle: Any
    offset_x: float
    offset_y: float


_INTERACTIVE_ELEMENTS_SCRIPT = """
(options) => {
  const {limit, offsetX, offsetY, rootWidth, rootHeight} = options;
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
  const targets = [];
  let truncated = false;
  let incomplete = false;

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
      if (host.shadowRoot) {
        collect(host.shadowRoot, out);
      } else if (String(host.tagName || "").includes("-")) {
        // A custom element without an open shadow root may own a closed tree.
        // It cannot be enumerated, so unresolved coordinate actions must not
        // be treated as verified.
        incomplete = true;
      }
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
    elements.push({
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
    targets.push(node);
  }
  return {elements, targets, truncated, incomplete};
}
"""

_HIT_TEST_SCRIPT = """
(target, options) => {
  const {x, y} = options;
  const deepestElementFromPoint = (root, pointX, pointY) => {
    let hit = root.elementFromPoint(pointX, pointY);
    while (hit && hit.shadowRoot) {
      const nested = hit.shadowRoot.elementFromPoint(pointX, pointY);
      if (!nested || nested === hit) break;
      hit = nested;
    }
    return hit;
  };
  const node = deepestElementFromPoint(document, x, y);
  if (node === null) return {matches: false, tag: null, found: false};
  const tag = node.tagName ? node.tagName.toLowerCase() : null;
  return {
    matches: node === target || Boolean(target && target.contains(node)),
    tag,
    found: true
  };
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
        navigation_allowlist: Sequence[str] | None = None,
        navigation_denylist: Sequence[str] | None = None,
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
        self.navigation_allowlist = normalize_host_patterns(navigation_allowlist)
        self.navigation_denylist = normalize_host_patterns(navigation_denylist)
        self._hit_targets: dict[str, _ElementHitTarget] = {}
        self._guarded_page_ids: set[int] = set()

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
        await self._install_navigation_guard(page)
        return page

    async def close(self) -> None:
        await self._clear_hit_targets()
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
            self._assert_navigation_allowed(str(getattr(page, "url", "") or ""))
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
        extraction_incomplete = False
        truncated = False
        await self._clear_hit_targets()
        try:
            (
                elements,
                truncated,
                extraction_incomplete,
            ) = await self._read_interactive_elements(
                page,
                viewport=viewport,
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
        if extraction_incomplete:
            metadata[ELEMENT_EXTRACTION_INCOMPLETE_KEY] = True
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
    ) -> tuple[list[ComputerElement], bool, bool]:
        """Extract interactive elements from the page and its visible frames.

        Bounds from nested frames are translated into the top-level viewport so
        one normalized coordinate space covers the whole screenshot.
        """
        elements: list[ComputerElement] = []
        truncated = False
        incomplete = False
        for frame_index, frame in enumerate(self._iter_frames(page)):
            remaining = MAX_OBSERVATION_ELEMENTS - len(elements)
            if remaining <= 0:
                truncated = True
                break
            offset = await self._frame_offset(page, frame)
            if offset is None:
                incomplete = True
                continue
            offset_x, offset_y = offset
            try:
                (
                    payload,
                    target_handles,
                    identity_verified,
                ) = await self._evaluate_interactive_elements(
                    frame,
                    limit=remaining,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    viewport=viewport,
                )
            except Exception:  # noqa: BLE001 - a detached frame is not fatal.
                if frame is self._main_frame(page):
                    raise
                incomplete = True
                logger.debug(
                    "Skipping frame %s during element extraction",
                    frame_index,
                    exc_info=True,
                )
                continue
            frame_elements, frame_truncated, frame_incomplete = (
                self._parse_frame_elements(
                    payload,
                    target_handles=target_handles,
                    identity_verified=identity_verified,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    start_index=len(elements) + 1,
                )
            )
            elements.extend(frame_elements)
            truncated = truncated or frame_truncated
            incomplete = incomplete or frame_incomplete
        return elements, truncated, incomplete

    async def _evaluate_interactive_elements(
        self,
        frame: Any,
        *,
        limit: int,
        offset_x: float,
        offset_y: float,
        viewport: Viewport,
    ) -> tuple[Any, list[Any], bool]:
        """Collect descriptors plus provider-owned handles to their DOM nodes.

        Element handles are kept out of page-visible attributes.  A hostile
        page therefore cannot copy or rewrite an Xagent marker to make a click
        appear to target a different node.
        """
        options = {
            "limit": limit,
            "offsetX": offset_x,
            "offsetY": offset_y,
            "rootWidth": viewport.width,
            "rootHeight": viewport.height,
        }
        evaluate_handle = getattr(frame, "evaluate_handle", None)
        if not callable(evaluate_handle):
            # Test doubles and unusual Playwright-compatible providers may only
            # expose evaluate(). Descriptors remain useful, but element-target
            # actions fail closed because no unforgeable identity was captured.
            payload = await frame.evaluate(_INTERACTIVE_ELEMENTS_SCRIPT, options)
            return payload, [], False

        result_handle = await evaluate_handle(
            _INTERACTIVE_ELEMENTS_SCRIPT,
            options,
        )
        disposable_handles: list[Any] = [result_handle]
        target_handles: list[Any] = []
        try:
            properties = await result_handle.get_properties()
            elements_handle = properties.get("elements")
            targets_handle = properties.get("targets")
            truncated_handle = properties.get("truncated")
            incomplete_handle = properties.get("incomplete")
            if elements_handle is None or targets_handle is None:
                raise RuntimeError(
                    "interactive element extraction returned no target handles"
                )
            disposable_handles.extend(
                handle
                for handle in (
                    elements_handle,
                    targets_handle,
                    truncated_handle,
                    incomplete_handle,
                )
                if handle is not None
            )
            raw_elements = await elements_handle.json_value()
            target_properties = await targets_handle.get_properties()
            for index in range(
                len(raw_elements) if isinstance(raw_elements, list) else 0
            ):
                item_handle = target_properties.get(str(index))
                if item_handle is None:
                    target_handles.append(None)
                    continue
                element_handle = item_handle.as_element()
                if element_handle is None:
                    await self._dispose_handle(item_handle)
                    target_handles.append(None)
                    continue
                target_handles.append(element_handle)
            payload = {
                "elements": raw_elements,
                "truncated": (
                    bool(await truncated_handle.json_value())
                    if truncated_handle is not None
                    else False
                ),
                "incomplete": (
                    bool(await incomplete_handle.json_value())
                    if incomplete_handle is not None
                    else False
                ),
            }
            return payload, target_handles, True
        except Exception:
            for handle in target_handles:
                await self._dispose_handle(handle)
            raise
        finally:
            for handle in disposable_handles:
                if handle not in target_handles:
                    await self._dispose_handle(handle)

    def _parse_frame_elements(
        self,
        payload: Any,
        *,
        target_handles: list[Any],
        identity_verified: bool,
        offset_x: float,
        offset_y: float,
        start_index: int,
    ) -> tuple[list[ComputerElement], bool, bool]:
        raw_elements = payload.get("elements") if isinstance(payload, dict) else payload
        truncated = (
            bool(payload.get("truncated")) if isinstance(payload, dict) else False
        )
        incomplete = (
            bool(payload.get("incomplete")) if isinstance(payload, dict) else False
        )
        if not isinstance(raw_elements, list):
            return [], truncated, True
        elements: list[ComputerElement] = []
        index = start_index
        for target_index, entry in enumerate(raw_elements):
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
                incomplete = True
                continue
            handle = (
                target_handles[target_index]
                if target_index < len(target_handles)
                else None
            )
            if identity_verified and handle is not None:
                self._hit_targets[element_id] = _ElementHitTarget(
                    handle=handle,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
            else:
                incomplete = True
            index += 1
        return elements, truncated, incomplete

    async def _clear_hit_targets(self) -> None:
        targets = list(self._hit_targets.values())
        self._hit_targets.clear()
        for target in targets:
            await self._dispose_handle(target.handle)

    @staticmethod
    async def _dispose_handle(handle: Any) -> None:
        dispose = getattr(handle, "dispose", None)
        if not callable(dispose):
            return
        try:
            await dispose()
        except Exception:  # noqa: BLE001 - stale remote handles are disposable.
            logger.debug("Could not dispose browser element handle", exc_info=True)

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
            target_url = self._resolve_navigation_url(action.url)
            self._assert_navigation_allowed(target_url)
            await page.goto(
                target_url,
                wait_until="domcontentloaded",
                timeout=30_000,
            )
            self._assert_navigation_allowed(str(getattr(page, "url", "") or ""))
            return
        if action.type == ComputerActionType.WAIT:
            await page.wait_for_timeout(action.duration_ms or 1_000)
            return
        self._assert_navigation_allowed(str(getattr(page, "url", "") or ""))
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
        if target is None:
            return
        target_id = target.element_id
        if target_id is None and self.current_observation is not None:
            element = find_computer_target_element(action, self.current_observation)
            target_id = element.element_id if element is not None else None
        if target_id is None:
            # An unresolved point is elevated by policy and bound to a fresh
            # screenshot when the user approves it. There is no element
            # identity to verify in addition to that grant.
            return
        hit_target = self._hit_targets.get(target_id)
        if hit_target is None:
            raise ComputerTargetNotFoundError(
                f"Cannot verify the identity of {target_id!r} in the "
                "current frame. Take a fresh screenshot and try again."
            )
        try:
            result = await hit_target.handle.evaluate(
                _HIT_TEST_SCRIPT,
                {
                    "x": x - hit_target.offset_x,
                    "y": y - hit_target.offset_y,
                },
            )
        except Exception as exc:  # noqa: BLE001 - provider errors fail closed.
            raise ComputerTargetNotFoundError(
                f"Could not verify {target_id!r} before clicking. "
                "Take a fresh screenshot and try again."
            ) from exc
        if (
            isinstance(result, dict)
            and result.get("found") is True
            and result.get("matches") is True
        ):
            return
        obstruction = (
            str(result.get("tag") or "another element")
            if isinstance(result, dict)
            else "an unknown element"
        )
        raise ComputerTargetObstructedError(
            f"{target_id} is covered by {obstruction} at the clicked "
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

    def _assert_navigation_allowed(self, raw_url: str) -> None:
        reason = navigation_block_reason(
            raw_url,
            allowlist=self.navigation_allowlist,
            denylist=self.navigation_denylist,
        )
        if reason is not None:
            raise ValueError(reason)

    async def _install_navigation_guard(self, page: Any) -> None:
        """Abort disallowed top-level document requests, including redirects."""
        page_key = id(page)
        if page_key in self._guarded_page_ids:
            return
        route_method = getattr(page, "route", None)
        if not callable(route_method):
            return

        async def guard(route: Any, request: Any) -> None:
            try:
                is_navigation = request.is_navigation_request()
            except (AttributeError, TypeError):
                is_navigation = False
            is_main_frame = getattr(request, "frame", None) is self._main_frame(page)
            reason = (
                navigation_block_reason(
                    str(getattr(request, "url", "") or ""),
                    allowlist=self.navigation_allowlist,
                    denylist=self.navigation_denylist,
                )
                if is_navigation and is_main_frame
                else None
            )
            if reason is not None:
                logger.warning("Blocked browser navigation request: %s", reason)
                await route.abort("blockedbyclient")
                return
            await route.continue_()

        await route_method("**/*", guard)
        self._guarded_page_ids.add(page_key)

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
