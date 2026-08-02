from __future__ import annotations

from typing import Any

import pytest

from xagent.core.computer.browser import (
    _EDITABLE_ACTIVE_ELEMENT_SCRIPT,
    _INTERACTIVE_ELEMENTS_SCRIPT,
    BrowserComputerEnvironment,
)
from xagent.core.computer.environment import (
    ComputerFrameMismatchError,
    ComputerTargetNotFoundError,
)
from xagent.core.computer.schema import (
    COMPUTER_FRAME_ID_METADATA_KEY,
    COMPUTER_SESSION_ID_METADATA_KEY,
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerTarget,
    NormalizedPoint,
)
from xagent.core.context_ref import ContextReference


class FakeObservationStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def save_screenshot(self, **kwargs: Any) -> ContextReference:
        self.calls.append(kwargs)
        return ContextReference(
            file_ref={
                "file_id": f"image-{len(self.calls)}",
                "filename": f"{kwargs['frame_id']}.png",
                "mime_type": "image/png",
            },
            metadata={
                COMPUTER_SESSION_ID_METADATA_KEY: kwargs["session_id"],
                COMPUTER_FRAME_ID_METADATA_KEY: kwargs["frame_id"],
            },
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
        self.frames: list[Any] = [object()]
        self.mouse = FakeMouse()
        self.keyboard = FakeKeyboard()
        self.goto_calls: list[tuple[str, str, int]] = []
        self.wait_calls: list[int] = []
        self.active_element_editable = True
        self.element_payload: Any = {
            "elements": [
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
                    "metadata": {"tag": "button", "sensitive": False},
                }
            ],
            "truncated": False,
        }

    async def set_viewport_size(self, viewport: dict[str, int]) -> None:
        self.viewport_size = viewport

    async def screenshot(self, **kwargs: Any) -> bytes:
        assert kwargs == {"full_page": False, "type": "png"}
        return b"browser-png"

    async def title(self) -> str:
        return "Browser page"

    async def evaluate(self, script: str, arg: Any = None) -> Any:
        if "devicePixelRatio" in script:
            return 2
        if "querySelectorAll" in script:
            assert arg == 100
            if isinstance(self.element_payload, BaseException):
                raise self.element_payload
            return self.element_payload
        if "currentDocument.activeElement" in script:
            return self.active_element_editable
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
        self.session = FakeSession(page)
        self.calls: list[tuple[str, bool]] = []
        self.closed: list[str] = []

    async def get_or_create(self, session_id: str, headless: bool) -> FakeSession:
        self.calls.append((session_id, headless))
        return self.session

    async def close(self, session_id: str) -> None:
        self.closed.append(session_id)


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
        workspace=object(),
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
    assert observation.metadata["browser_runtime_kind"] == "ephemeral_playwright"
    assert environment.current_observation == observation
    assert environment.manager.calls == [  # type: ignore[attr-defined]
        ("browser-1:computer", True)
    ]


def test_browser_element_script_never_reads_input_values() -> None:
    assert "node.value" not in _INTERACTIVE_ELEMENTS_SCRIPT


def test_editable_check_descends_into_same_origin_frames() -> None:
    assert "node.contentDocument" in _EDITABLE_ACTIVE_ELEMENT_SCRIPT


@pytest.mark.asyncio
async def test_browser_actions_use_fresh_normalized_coordinates(
    browser_environment,
) -> None:
    environment, page, _store = browser_environment
    first = await environment.observe()

    clicked = await environment.execute(
        ComputerActionBatch(
            session_id="browser-1",
            expected_frame_id=first.frame_id,
            actions=[
                ComputerAction(
                    type=ComputerActionType.CLICK,
                    target=ComputerTarget(element_id="dom-1"),
                )
            ],
        )
    )
    typed = await environment.execute(
        ComputerActionBatch(
            session_id="browser-1",
            expected_frame_id=clicked.frame_id,
            actions=[ComputerAction(type=ComputerActionType.TYPE, text="hello")],
        )
    )
    scrolled = await environment.execute(
        ComputerActionBatch(
            session_id="browser-1",
            expected_frame_id=typed.frame_id,
            actions=[ComputerAction(type=ComputerActionType.SCROLL, delta_y=0.5)],
        )
    )
    await environment.execute(
        ComputerActionBatch(
            session_id="browser-1",
            expected_frame_id=scrolled.frame_id,
            actions=[
                ComputerAction(type=ComputerActionType.KEYPRESS, keys=["CTRL", "A"])
            ],
        )
    )

    assert page.mouse.calls[0] == ("click", 256.0, 180.0)
    assert page.mouse.calls[1] == ("wheel", 0, 360.0)
    assert page.keyboard.calls == [
        ("insert_text", "hello"),
        ("press", "Control+A"),
    ]


@pytest.mark.asyncio
async def test_browser_point_click_uses_normalized_viewport(
    browser_environment,
) -> None:
    environment, page, _store = browser_environment
    first = await environment.observe()

    await environment.execute(
        ComputerActionBatch(
            session_id="browser-1",
            expected_frame_id=first.frame_id,
            actions=[
                ComputerAction(
                    type=ComputerActionType.CLICK,
                    target=ComputerTarget(point=NormalizedPoint(x=0.5, y=0.5)),
                )
            ],
        )
    )

    assert page.mouse.calls[0] == ("click", 640.0, 360.0)


@pytest.mark.asyncio
async def test_browser_element_lookup_fails_with_clear_target_error(
    browser_environment,
) -> None:
    environment, _page, _store = browser_environment
    await environment.observe()
    action = ComputerAction(
        type=ComputerActionType.CLICK,
        target=ComputerTarget(element_id="missing-element"),
    )

    with pytest.raises(ComputerTargetNotFoundError, match="missing-element"):
        environment._target_pixels(action)


@pytest.mark.asyncio
async def test_browser_navigation_returns_new_observation(browser_environment) -> None:
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
async def test_browser_omits_unsupported_active_url(browser_environment) -> None:
    environment, page, _store = browser_environment
    page.url = "data:text/html,secret"

    observation = await environment.observe()

    assert observation.active_url is None
    assert observation.metadata["active_url_unavailable"] is True


@pytest.mark.asyncio
async def test_browser_recreated_session_requires_fresh_frame(
    browser_environment,
) -> None:
    environment, _page, _store = browser_environment
    first = await environment.observe()
    manager = environment.manager
    replacement_page = FakePage()
    manager.session = FakeSession(replacement_page)  # type: ignore[attr-defined]

    with pytest.raises(ComputerFrameMismatchError, match="session changed"):
        await environment.execute(
            ComputerActionBatch(
                session_id="browser-1",
                expected_frame_id=first.frame_id,
                actions=[
                    ComputerAction(
                        type=ComputerActionType.CLICK,
                        target=ComputerTarget(point=NormalizedPoint(x=0.5, y=0.5)),
                    )
                ],
            )
        )

    assert replacement_page.mouse.calls == []
    assert environment.current_observation is None
    refreshed = await environment.observe()
    assert refreshed.frame_id != first.frame_id


@pytest.mark.asyncio
async def test_replace_text_rejects_non_editable_target(browser_environment) -> None:
    environment, page, _store = browser_environment
    first = await environment.observe()
    page.active_element_editable = False

    with pytest.raises(ValueError, match="not an editable element"):
        await environment.execute(
            ComputerActionBatch(
                session_id="browser-1",
                expected_frame_id=first.frame_id,
                actions=[
                    ComputerAction(
                        type=ComputerActionType.REPLACE_TEXT,
                        target=ComputerTarget(point=NormalizedPoint(x=0.5, y=0.5)),
                        text="replacement",
                    )
                ],
            )
        )

    assert page.keyboard.calls == []


def test_keypress_rejects_non_modifier_sequences() -> None:
    assert BrowserComputerEnvironment._playwright_key_chord(["CTRL", "A"]) == (
        "Control+A"
    )
    with pytest.raises(ValueError, match="one key chord"):
        BrowserComputerEnvironment._playwright_key_chord(["ArrowDown", "Enter"])


@pytest.mark.asyncio
async def test_browser_reports_incomplete_or_failed_element_extraction(
    browser_environment,
) -> None:
    environment, page, _store = browser_environment
    page.frames.append(object())

    incomplete = await environment.observe()
    assert incomplete.metadata["element_extraction_incomplete"] is True

    page.element_payload = RuntimeError("detached document")
    failed = await environment.observe()
    assert failed.elements == []
    assert failed.metadata["element_extraction_failed"] is True


@pytest.mark.asyncio
async def test_browser_environment_closes_its_session(browser_environment) -> None:
    environment, _page, _store = browser_environment

    await environment.close()

    assert environment.manager.closed == [  # type: ignore[attr-defined]
        "browser-1:computer"
    ]
