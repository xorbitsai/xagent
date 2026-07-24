from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from xagent.core.computer.browser import BrowserComputerEnvironment
from xagent.core.computer.schema import (
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerTarget,
    NormalizedPoint,
)
from xagent.core.context_ref import ContextReference, ContextReferencePurpose


class FakeObservationStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def save_screenshot(self, **kwargs: Any) -> ContextReference:
        self.calls.append(kwargs)
        return ContextReference(
            file_ref={
                "file_id": f"image-{len(self.calls)}",
                "filename": f"{kwargs['frame_id']}.png",
                "mime_type": "image/png",
            },
            purpose=ContextReferencePurpose.OBSERVATION,
            frame_id=kwargs["frame_id"],
        )


class FakeMouse:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def click(self, x: float, y: float) -> None:
        self.calls.append(("click", x, y))

    async def dblclick(self, x: float, y: float) -> None:
        self.calls.append(("dblclick", x, y))

    async def move(self, x: float, y: float, **kwargs: Any) -> None:
        self.calls.append(("move", x, y, kwargs))

    async def wheel(self, x: float, y: float) -> None:
        self.calls.append(("wheel", x, y))

    async def down(self) -> None:
        self.calls.append(("down",))

    async def up(self) -> None:
        self.calls.append(("up",))


class FakeKeyboard:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def press(self, keys: str) -> None:
        self.calls.append(("press", keys))

    async def insert_text(self, text: str) -> None:
        self.calls.append(("insert_text", text))


class FakePage:
    def __init__(self) -> None:
        self.viewport_size = {"width": 1280, "height": 720}
        self.url = "about:blank"
        self.mouse = FakeMouse()
        self.keyboard = FakeKeyboard()
        self.goto_calls: list[tuple[str, str, int]] = []
        self.wait_calls: list[int] = []

    async def set_viewport_size(self, viewport: dict[str, int]) -> None:
        self.viewport_size = viewport

    async def screenshot(self, **kwargs: Any) -> bytes:
        assert kwargs == {"full_page": False, "type": "png"}
        return b"browser-png"

    async def title(self) -> str:
        return "Browser page"

    async def evaluate(self, script: str) -> Any:
        if "devicePixelRatio" in script:
            return 2
        if "querySelectorAll" in script:
            return [
                {
                    "bounds": {
                        "x": 0.1,
                        "y": 0.2,
                        "width": 0.2,
                        "height": 0.1,
                    },
                    "label": "Continue",
                    "role": "button",
                    "text": "Continue",
                    "metadata": {"tag": "button"},
                }
            ]
        raise AssertionError(f"unexpected evaluate script: {script}")

    async def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
        self.goto_calls.append((url, wait_until, timeout))
        self.url = url

    async def wait_for_timeout(self, duration_ms: int) -> None:
        self.wait_calls.append(duration_ms)


class FakeSession:
    def __init__(self, page: FakePage) -> None:
        self.page = page

    async def get_page(self) -> FakePage:
        return self.page


class FakeManager:
    def __init__(self, page: FakePage) -> None:
        self.page = page
        self.calls: list[tuple[str, bool]] = []
        self.closed: list[str] = []

    async def get_or_create(self, session_id: str, headless: bool) -> FakeSession:
        self.calls.append((session_id, headless))
        return FakeSession(self.page)

    async def close(self, session_id: str) -> None:
        self.closed.append(session_id)


class FakeWorkspace:
    def resolve_path(self, path: str, default_dir: str) -> Path:
        raise ValueError(f"outside workspace: {path} ({default_dir})")

    def resolve_path_with_search(self, path: str) -> Path:
        raise FileNotFoundError(path)


@pytest.fixture
def browser_environment() -> tuple[
    BrowserComputerEnvironment,
    FakePage,
    FakeObservationStore,
]:
    page = FakePage()
    store = FakeObservationStore()
    environment = BrowserComputerEnvironment(
        session_id="browser-1",
        workspace=FakeWorkspace(),
        manager=FakeManager(page),  # type: ignore[arg-type]
        observation_store=store,  # type: ignore[arg-type]
    )
    return environment, page, store


@pytest.mark.asyncio
async def test_browser_observation_captures_screenshot_and_dom_elements(
    browser_environment,
) -> None:
    environment, _page, store = browser_environment

    observation = await environment.observe()

    assert observation.active_url == "about:blank"
    assert observation.title == "Browser page"
    assert observation.viewport.model_dump() == {
        "width": 1280,
        "height": 720,
        "device_pixel_ratio": 2.0,
    }
    assert observation.elements[0].element_id == "dom-1"
    assert observation.elements[0].label == "Continue"
    assert store.calls[0]["image_bytes"] == b"browser-png"
    assert environment.current_observation == observation


@pytest.mark.asyncio
async def test_browser_actions_use_normalized_and_element_coordinates(
    browser_environment,
) -> None:
    environment, page, _store = browser_environment
    first = await environment.observe()

    second = await environment.execute(
        ComputerActionBatch(
            session_id="browser-1",
            expected_frame_id=first.frame_id,
            actions=[
                ComputerAction(
                    type=ComputerActionType.CLICK,
                    target=ComputerTarget(element_id="dom-1"),
                ),
                ComputerAction(
                    type=ComputerActionType.TYPE,
                    target=ComputerTarget(point=NormalizedPoint(x=0.5, y=0.5)),
                    text="hello",
                ),
                ComputerAction(
                    type=ComputerActionType.SCROLL,
                    delta_y=0.5,
                ),
                ComputerAction(
                    type=ComputerActionType.KEYPRESS,
                    keys=["CTRL", "A"],
                ),
            ],
        )
    )

    assert page.mouse.calls[0] == ("click", 256.0, 180.0)
    assert page.mouse.calls[1] == ("click", 640.0, 360.0)
    assert page.mouse.calls[2] == ("wheel", 0, 360.0)
    assert page.keyboard.calls == [
        ("insert_text", "hello"),
        ("press", "Control+A"),
    ]
    assert second.frame_id != first.frame_id


@pytest.mark.asyncio
async def test_browser_navigation_returns_new_observation(
    browser_environment,
) -> None:
    environment, page, _store = browser_environment
    first = await environment.observe()

    observation = await environment.execute(
        ComputerActionBatch(
            session_id="browser-1",
            expected_frame_id=first.frame_id,
            actions=[
                ComputerAction(
                    type=ComputerActionType.NAVIGATE,
                    url="https://example.com",
                )
            ],
        )
    )

    assert page.goto_calls == [("https://example.com", "domcontentloaded", 30_000)]
    assert observation.active_url == "https://example.com"


@pytest.mark.asyncio
async def test_browser_navigation_rejects_unapproved_scheme(
    browser_environment,
) -> None:
    environment, _page, _store = browser_environment
    first = await environment.observe()

    with pytest.raises(ValueError, match="navigate URL"):
        await environment.execute(
            ComputerActionBatch(
                session_id="browser-1",
                expected_frame_id=first.frame_id,
                actions=[
                    ComputerAction(
                        type=ComputerActionType.NAVIGATE,
                        url="javascript:alert(1)",
                    )
                ],
            )
        )


@pytest.mark.asyncio
async def test_browser_environment_closes_its_session(browser_environment) -> None:
    environment, _page, _store = browser_environment

    await environment.close()

    assert environment.manager.closed == ["browser-1"]  # type: ignore[attr-defined]
