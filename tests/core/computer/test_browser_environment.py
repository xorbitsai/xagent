from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from xagent.core.computer.browser import (
    BrowserComputerEnvironment,
    ComputerTargetObstructedError,
)
from xagent.core.computer.schema import (
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerTarget,
    NormalizedPoint,
)
from xagent.core.computer.session import ComputerSessionBinding
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


class FakeRequest:
    def __init__(self, url: str, frame: Any) -> None:
        self.url = url
        self.frame = frame

    def is_navigation_request(self) -> bool:
        return True


class FakeRoute:
    def __init__(self) -> None:
        self.aborted = False
        self.continued = False

    async def abort(self, _reason: str) -> None:
        self.aborted = True

    async def continue_(self) -> None:
        self.continued = True


class FakeHandle:
    def __init__(
        self,
        value: Any,
        *,
        page: "FakePage | None" = None,
        is_element: bool = False,
    ) -> None:
        self.value = value
        self.page = page
        self.is_element = is_element
        self.disposed = False

    async def json_value(self) -> Any:
        return self.value

    async def get_properties(self) -> dict[str, "FakeHandle"]:
        if isinstance(self.value, dict):
            return {
                key: FakeHandle(value, page=self.page)
                for key, value in self.value.items()
            }
        if isinstance(self.value, list):
            return {
                str(index): (
                    value
                    if isinstance(value, FakeHandle)
                    else FakeHandle(value, page=self.page)
                )
                for index, value in enumerate(self.value)
            }
        return {}

    def as_element(self) -> "FakeHandle | None":
        return self if self.is_element else None

    async def evaluate(self, script: str, arg: Any = None) -> Any:
        assert "elementFromPoint" in script
        assert self.page is not None
        self.page.hit_test_calls.append(dict(arg or {}))
        return {
            "matches": self.page.hit_marker is None,
            "tag": "div",
            "found": True,
        }

    async def dispose(self) -> None:
        self.disposed = True


class FakeFrame:
    """A nested frame reporting bounds in its own coordinate space."""

    def __init__(
        self,
        *,
        offset_x: float,
        offset_y: float,
        elements: list[dict[str, Any]],
    ) -> None:
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.elements = elements
        self.issued_markers: list[str] = []

    async def frame_element(self) -> "FakeFrame":
        return self

    async def bounding_box(self) -> dict[str, float]:
        return {"x": self.offset_x, "y": self.offset_y, "width": 640, "height": 360}

    async def evaluate(self, script: str, arg: Any = None) -> Any:
        if "querySelectorAll" not in script:
            raise AssertionError(f"unexpected frame script: {script}")
        options = arg or {}
        offset_x = float(options.get("offsetX", 0))
        offset_y = float(options.get("offsetY", 0))
        width = float(options.get("rootWidth", 1))
        height = float(options.get("rootHeight", 1))
        elements = []
        for entry in self.elements:
            bounds = entry["bounds"]
            # The real script reports local pixels shifted by the frame offset;
            # 640x360 is this frame's own viewport.
            elements.append(
                {
                    **entry,
                    "bounds": {
                        "x": (bounds["x"] * 640 + offset_x) / width,
                        "y": (bounds["y"] * 360 + offset_y) / height,
                        "width": max(1e-6, bounds["width"] * 640 / width),
                        "height": max(1e-6, bounds["height"] * 360 / height),
                    },
                }
            )
        return {
            "elements": elements,
            "truncated": False,
            "incomplete": False,
        }

    async def evaluate_handle(self, script: str, arg: Any = None) -> FakeHandle:
        payload = await self.evaluate(script, arg)
        # Nested-frame tests only inspect geometry, but production still
        # requires an opaque provider handle for every reported element.
        targets = [
            FakeHandle({}, page=None, is_element=True) for _entry in payload["elements"]
        ]
        return FakeHandle({**payload, "targets": targets})


class FakePage:
    def __init__(self) -> None:
        self.viewport_size = {"width": 1280, "height": 720}
        self.url = "about:blank"
        self.child_frames: list[FakeFrame] = []
        self.truncate_elements = False
        self.fail_element_extraction = False
        self.mouse = FakeMouse()
        self.keyboard = FakeKeyboard()
        self.goto_calls: list[tuple[str, str, int]] = []
        self.goto_redirect_url: str | None = None
        self.route_handler: Any = None
        self.wait_calls: list[int] = []
        self.hit_test_calls: list[dict[str, Any]] = []
        self.issued_markers: list[str] = []
        #: Marker the hit test reports. ``None`` means the click lands on the
        #: element that was extracted first, i.e. nothing is in the way.
        self.hit_marker: str | None = None
        self.interactive_elements: list[dict[str, Any]] = [
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

    @property
    def main_frame(self) -> "FakePage":
        return self

    @property
    def frames(self) -> list[Any]:
        return [self, *self.child_frames]

    async def set_viewport_size(self, viewport: dict[str, int]) -> None:
        self.viewport_size = viewport

    async def route(self, pattern: str, handler: Any) -> None:
        assert pattern == "**/*"
        self.route_handler = handler

    async def screenshot(self, **kwargs: Any) -> bytes:
        assert kwargs == {"full_page": False, "type": "png"}
        return b"browser-png"

    async def title(self) -> str:
        return "Browser page"

    async def evaluate(self, script: str, arg: Any = None) -> Any:
        if "devicePixelRatio" in script:
            return 2
        if "innerWidth" in script and "innerHeight" in script and arg is None:
            return {"width": 1280, "height": 720}
        if "elementFromPoint" in script:
            raise AssertionError("hit testing must use an opaque element handle")
        if "querySelectorAll" in script:
            if self.fail_element_extraction:
                raise RuntimeError("element extraction failed")
            return {
                "elements": list(self.interactive_elements),
                "truncated": self.truncate_elements,
                "incomplete": False,
            }
        raise AssertionError(f"unexpected evaluate script: {script}")

    async def evaluate_handle(self, script: str, arg: Any = None) -> FakeHandle:
        payload = await self.evaluate(script, arg)
        targets = [
            FakeHandle({}, page=self, is_element=True) for _entry in payload["elements"]
        ]
        return FakeHandle({**payload, "targets": targets}, page=self)

    async def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
        self.goto_calls.append((url, wait_until, timeout))
        for candidate in [url, self.goto_redirect_url]:
            if candidate is None:
                continue
            if self.route_handler is not None:
                route = FakeRoute()
                await self.route_handler(route, FakeRequest(candidate, self))
                if route.aborted:
                    raise RuntimeError("navigation was blocked")
            self.url = candidate

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
        self.calls: list[dict[str, Any]] = []
        self.closed: list[str] = []

    async def get_or_create(
        self,
        session_id: str,
        headless: bool,
        **kwargs: Any,
    ) -> FakeSession:
        self.calls.append(
            {
                "session_id": session_id,
                "headless": headless,
                **kwargs,
            }
        )
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
async def test_browser_observation_redacts_sensitive_input_values(
    browser_environment,
) -> None:
    environment, page, _store = browser_environment
    page.interactive_elements = [
        {
            "bounds": {
                "x": 0.1,
                "y": 0.2,
                "width": 0.2,
                "height": 0.1,
            },
            "label": "Password label secret",
            "role": "input",
            "text": "plain-text-secret",
            "metadata": {
                "tag": "input",
                "input_type": "password",
                "autocomplete": "current-password",
                "focused": True,
                "value": "metadata-secret",
            },
        }
    ]

    observation = await environment.observe()

    assert observation.elements[0].text is None
    assert observation.elements[0].label == "Sensitive input"
    assert observation.elements[0].metadata["sensitive"] is True
    assert observation.elements[0].metadata["focused"] is True
    assert "plain-text-secret" not in observation.model_dump_json()
    assert "metadata-secret" not in observation.model_dump_json()
    assert "Password label secret" not in observation.model_dump_json()


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
async def test_browser_navigation_guard_blocks_disallowed_redirect() -> None:
    page = FakePage()
    page.goto_redirect_url = "https://outside.test/landing"
    environment = BrowserComputerEnvironment(
        session_id="browser-1",
        workspace=FakeWorkspace(),
        manager=FakeManager(page),  # type: ignore[arg-type]
        observation_store=FakeObservationStore(),  # type: ignore[arg-type]
        navigation_allowlist=["example.com"],
    )
    first = await environment.observe()

    with pytest.raises(RuntimeError, match="navigation was blocked"):
        await environment.execute(
            ComputerActionBatch(
                session_id="browser-1",
                expected_frame_id=first.frame_id,
                actions=[
                    ComputerAction(
                        type=ComputerActionType.NAVIGATE,
                        url="https://example.com/start",
                    )
                ],
            )
        )

    assert page.url == "https://example.com/start"


@pytest.mark.asyncio
async def test_click_is_refused_when_the_target_is_covered(
    browser_environment,
) -> None:
    """The centre of an element's box is not always what a click would hit.

    A consent overlay or sticky header on top of the target would otherwise turn
    into a silent mis-click on whatever is actually there.
    """
    environment, page, _store = browser_environment
    first = await environment.observe()
    page.hit_marker = "some-other-element"

    with pytest.raises(ComputerTargetObstructedError, match="covered by div"):
        await environment.execute(
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

    assert page.mouse.calls == []


@pytest.mark.asyncio
async def test_point_clicks_skip_hit_verification(browser_environment) -> None:
    """Only element targets carry an identity that a hit test can check."""
    environment, page, _store = browser_environment
    first = await environment.observe()
    page.hit_marker = "some-other-element"

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

    assert page.mouse.calls == [("click", 640.0, 360.0)]
    assert page.hit_test_calls == []


@pytest.mark.asyncio
async def test_element_click_fails_closed_without_provider_identity(
    browser_environment,
) -> None:
    environment, page, _store = browser_environment
    page.evaluate_handle = None  # type: ignore[method-assign]
    first = await environment.observe()

    with pytest.raises(RuntimeError, match="Cannot verify the identity"):
        await environment.execute(
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

    assert first.metadata["element_extraction_incomplete"] is True
    assert page.mouse.calls == []


@pytest.mark.asyncio
async def test_iframe_elements_are_mapped_into_the_top_level_viewport() -> None:
    """A screenshot spans nested frames, so one coordinate space must too."""
    page = FakePage()
    frame = FakeFrame(
        offset_x=640.0,
        offset_y=360.0,
        elements=[
            {
                "bounds": {"x": 0.0, "y": 0.0, "width": 0.25, "height": 0.1},
                "label": "Pay now",
                "role": "button",
                "text": "Pay now",
                "metadata": {"tag": "button"},
            }
        ],
    )
    page.child_frames = [frame]
    environment = BrowserComputerEnvironment(
        session_id="browser-1",
        workspace=FakeWorkspace(),
        manager=FakeManager(page),  # type: ignore[arg-type]
        observation_store=FakeObservationStore(),  # type: ignore[arg-type]
    )

    observation = await environment.observe()

    assert [element.label for element in observation.elements] == [
        "Continue",
        "Pay now",
    ]
    nested = observation.elements[1]
    # The frame sits at (640, 360) in a 1280x720 viewport, so its origin is the
    # centre of the page.
    assert nested.bounds.x == pytest.approx(0.5)
    assert nested.bounds.y == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_truncated_element_lists_are_reported(browser_environment) -> None:
    environment, page, _store = browser_environment
    page.truncate_elements = True

    observation = await environment.observe()

    assert observation.metadata["elements_truncated"] is True


@pytest.mark.asyncio
async def test_failed_element_extraction_is_reported(browser_environment) -> None:
    """The policy needs to know the page structure is unknown, not empty."""
    environment, page, _store = browser_environment
    page.fail_element_extraction = True

    observation = await environment.observe()

    assert observation.elements == []
    assert observation.metadata["element_extraction_failed"] is True


@pytest.mark.asyncio
async def test_browser_environment_closes_its_session(browser_environment) -> None:
    environment, _page, _store = browser_environment

    await environment.close()

    assert environment.manager.closed == ["browser-1"]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_persistent_environment_uses_stable_profile_session(tmp_path) -> None:
    page = FakePage()
    manager = FakeManager(page)
    binding = ComputerSessionBinding.from_values(
        runtime_kind="persistent_playwright",
        owner_task_id="task-9",
        user_id=7,
        profile_id="work",
        profile_root=tmp_path,
    )
    environment = BrowserComputerEnvironment(
        session_id="task-9:step-1",
        workspace=FakeWorkspace(),
        manager=manager,  # type: ignore[arg-type]
        observation_store=FakeObservationStore(),  # type: ignore[arg-type]
        session_binding=binding,
    )

    await environment.observe()

    assert manager.calls[0] == {
        "session_id": "computer-profile:user_7:work",
        "headless": False,
        "persistent_profile_dir": tmp_path / "user_7" / "work",
        "owner_id": "task-9",
    }
