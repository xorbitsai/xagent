from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from ...config import SUPPORTED_NATIVE_BROWSER_APP_NAMES


class NativeBrowserWindowTarget(Protocol):
    @property
    def pid(self) -> int: ...

    @property
    def window_id(self) -> int: ...

    @property
    def app_name(self) -> str: ...

    @property
    def title(self) -> str | None: ...

    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...

    @property
    def width(self) -> float: ...

    @property
    def height(self) -> float: ...


class NativeBrowserNavigationError(RuntimeError):
    """Raised when an exact native browser window cannot be navigated safely."""


@dataclass(frozen=True)
class NativeBrowserNavigationResult:
    route: str
    browser_window_id: int
    actual_url: str


class NativeBrowserNavigator(Protocol):
    def supports(self, target: NativeBrowserWindowTarget) -> bool: ...

    async def navigate(
        self,
        target: NativeBrowserWindowTarget,
        url: str,
    ) -> NativeBrowserNavigationResult: ...


JXARunner = Callable[[str, list[str]], Awaitable[Mapping[str, Any]]]

_CHROMIUM_APP_NAMES = frozenset(
    name.casefold() for name in SUPPORTED_NATIVE_BROWSER_APP_NAMES
)
_CHROMIUM_BUNDLE_IDS = frozenset(
    {
        "com.brave.Browser",
        "com.google.Chrome",
        "com.google.Chrome.canary",
        "com.microsoft.edgemac",
        "com.vivaldi.Vivaldi",
        "org.chromium.Chromium",
    }
)
_WINDOW_ENUMERATION_SCRIPT = r"""
function normalizedBounds(browserWindow) {
  const bounds = browserWindow.bounds();
  return [
    Number(bounds.x),
    Number(bounds.y),
    Number(bounds.x) + Number(bounds.width),
    Number(bounds.y) + Number(bounds.height),
  ];
}

function run(argv) {
  ObjC.import("AppKit");
  const pid = Number(argv[0]);
  const running = $.NSRunningApplication.runningApplicationWithProcessIdentifier(pid);
  if (!running) {
    throw new Error("authorized browser process is no longer running");
  }
  const bundleId = String(ObjC.unwrap(running.bundleIdentifier));
  const app = Application(bundleId);
  const windows = app.windows();
  const result = [];
  for (let index = 0; index < windows.length; index += 1) {
    const browserWindow = windows[index];
    result.push({
      id: Number(browserWindow.id()),
      title: String(browserWindow.name() || ""),
      bounds: normalizedBounds(browserWindow),
    });
  }
  return JSON.stringify({bundle_id: bundleId, windows: result});
}
"""
_WINDOW_NAVIGATION_SCRIPT = r"""
function normalizedBounds(browserWindow) {
  const bounds = browserWindow.bounds();
  return [
    Number(bounds.x),
    Number(bounds.y),
    Number(bounds.x) + Number(bounds.width),
    Number(bounds.y) + Number(bounds.height),
  ];
}

function sameBounds(actual, expected) {
  if (!Array.isArray(actual) || actual.length !== 4) return false;
  for (let index = 0; index < 4; index += 1) {
    if (Math.abs(Number(actual[index]) - Number(expected[index])) > 2) return false;
  }
  return true;
}

function run(argv) {
  ObjC.import("AppKit");
  const pid = Number(argv[0]);
  const browserWindowId = Number(argv[1]);
  const expectedTitle = argv[2];
  const expectedBounds = JSON.parse(argv[3]);
  const requestedUrl = argv[4];
  const running = $.NSRunningApplication.runningApplicationWithProcessIdentifier(pid);
  if (!running) {
    throw new Error("authorized browser process is no longer running");
  }
  const bundleId = String(ObjC.unwrap(running.bundleIdentifier));
  const app = Application(bundleId);
  const windows = app.windows();
  let matched = null;
  for (let index = 0; index < windows.length; index += 1) {
    const candidate = windows[index];
    if (Number(candidate.id()) === browserWindowId) {
      matched = candidate;
      break;
    }
  }
  if (matched === null) {
    throw new Error("authorized browser window disappeared before navigation");
  }
  const currentTitle = String(matched.name() || "");
  const currentBounds = normalizedBounds(matched);
  if (currentTitle !== expectedTitle || !sameBounds(currentBounds, expectedBounds)) {
    throw new Error("authorized browser window changed before navigation");
  }
  const tab = matched.activeTab();
  tab.url = requestedUrl;
  const loadingDeadline = Date.now() + 5000;
  delay(0.1);
  while (Boolean(tab.loading()) && Date.now() < loadingDeadline) {
    delay(0.1);
  }
  return JSON.stringify({
    bundle_id: bundleId,
    browser_window_id: browserWindowId,
    actual_url: String(tab.url() || ""),
    loading: Boolean(tab.loading()),
  });
}
"""


@dataclass(frozen=True)
class _ScriptBrowserWindow:
    window_id: int
    title: str
    bounds: tuple[float, float, float, float]


class MacOSChromiumNavigator:
    """Navigate one authorized Chromium window without keyboard or CDP.

    Chrome's scriptable window ids are not CGWindowIDs. The adapter therefore
    resolves the authorized CG window to exactly one scriptable browser window
    using its fresh title and bounds, and rechecks that fingerprint immediately
    before changing the active tab URL. Ambiguous mappings fail before mutation.
    """

    def __init__(
        self,
        *,
        runner: JXARunner | None = None,
        platform: str | None = None,
    ) -> None:
        self._runner = runner or _run_jxa
        self._platform = platform or sys.platform

    def supports(self, target: NativeBrowserWindowTarget) -> bool:
        return (
            self._platform == "darwin"
            and target.app_name.strip().casefold() in _CHROMIUM_APP_NAMES
        )

    async def navigate(
        self,
        target: NativeBrowserWindowTarget,
        url: str,
    ) -> NativeBrowserNavigationResult:
        if not self.supports(target):
            raise NativeBrowserNavigationError(
                f"native navigation is unavailable for {target.app_name}"
            )
        inventory = await self._runner(
            _WINDOW_ENUMERATION_SCRIPT,
            [str(target.pid)],
        )
        bundle_id = str(inventory.get("bundle_id") or "")
        if bundle_id not in _CHROMIUM_BUNDLE_IDS:
            raise NativeBrowserNavigationError(
                "authorized process is not a supported Chromium browser"
            )
        windows = self._parse_windows(inventory.get("windows"))
        matched = self._match_window(target, windows)
        expected_bounds = self._target_bounds(target)
        result = await self._runner(
            _WINDOW_NAVIGATION_SCRIPT,
            [
                str(target.pid),
                str(matched.window_id),
                matched.title,
                json.dumps(expected_bounds, separators=(",", ":")),
                url,
            ],
        )
        if str(result.get("bundle_id") or "") != bundle_id:
            raise NativeBrowserNavigationError(
                "authorized browser process changed during navigation"
            )
        if int(result.get("browser_window_id") or -1) != matched.window_id:
            raise NativeBrowserNavigationError(
                "browser reported a different window after navigation"
            )
        actual_url = str(result.get("actual_url") or "").strip()
        if not actual_url.startswith(("http://", "https://")):
            raise NativeBrowserNavigationError(
                "browser did not report an HTTP URL after navigation"
            )
        return NativeBrowserNavigationResult(
            route="macos_chromium_apple_events",
            browser_window_id=matched.window_id,
            actual_url=actual_url,
        )

    @staticmethod
    def _parse_windows(raw_windows: Any) -> list[_ScriptBrowserWindow]:
        if not isinstance(raw_windows, list):
            raise NativeBrowserNavigationError(
                "browser did not return a scriptable window inventory"
            )
        parsed: list[_ScriptBrowserWindow] = []
        for raw in raw_windows:
            if not isinstance(raw, Mapping):
                continue
            bounds = raw.get("bounds")
            if not isinstance(bounds, list) or len(bounds) != 4:
                continue
            try:
                parsed.append(
                    _ScriptBrowserWindow(
                        window_id=int(raw["id"]),
                        title=str(raw.get("title") or ""),
                        bounds=tuple(float(value) for value in bounds),  # type: ignore[arg-type]
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return parsed

    @classmethod
    def _match_window(
        cls,
        target: NativeBrowserWindowTarget,
        windows: list[_ScriptBrowserWindow],
    ) -> _ScriptBrowserWindow:
        expected_title = target.title or ""
        expected_bounds = cls._target_bounds(target)
        matches = [
            window
            for window in windows
            if window.title == expected_title
            and all(
                abs(actual - expected) <= 2
                for actual, expected in zip(window.bounds, expected_bounds, strict=True)
            )
        ]
        if len(matches) != 1:
            detail = "ambiguous" if len(matches) > 1 else "not found"
            raise NativeBrowserNavigationError(
                "authorized browser window mapping is "
                f"{detail}; navigation was not attempted"
            )
        return matches[0]

    @staticmethod
    def _target_bounds(
        target: NativeBrowserWindowTarget,
    ) -> tuple[float, float, float, float]:
        return (
            target.x,
            target.y,
            target.x + target.width,
            target.y + target.height,
        )


async def _run_jxa(script: str, arguments: list[str]) -> Mapping[str, Any]:
    process = await asyncio.create_subprocess_exec(
        "/usr/bin/osascript",
        "-l",
        "JavaScript",
        "-e",
        script,
        "--",
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=8)
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise NativeBrowserNavigationError(
            "native browser navigation timed out"
        ) from exc
    except BaseException:
        # The subprocess outlives a cancelled coroutine unless it is explicitly
        # terminated and reaped. Always clean up, then preserve cancellation or
        # another process-control exception unchanged.
        process.kill()
        await process.wait()
        raise
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise NativeBrowserNavigationError(
            detail[:500] or "native browser navigation failed"
        )
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeBrowserNavigationError(
            "native browser navigation returned invalid output"
        ) from exc
    if not isinstance(payload, Mapping):
        raise NativeBrowserNavigationError(
            "native browser navigation returned an invalid result"
        )
    return payload


def default_native_browser_navigator() -> NativeBrowserNavigator | None:
    navigator = MacOSChromiumNavigator()
    return navigator if sys.platform == "darwin" else None
