from __future__ import annotations

import asyncio
from typing import Any

import pytest

from xagent.core.computer.cua_driver import CuaDriverError, CuaDriverResult
from xagent.core.computer.environment import (
    ComputerFrameMismatchError,
    ComputerTargetNotFoundError,
)
from xagent.core.computer.native_browser import NativeBrowserEnvironment
from xagent.core.computer.native_navigation import NativeBrowserNavigationResult
from xagent.core.computer.schema import (
    COMPUTER_CONTROL_METADATA_KEY,
    COMPUTER_FRAME_ID_METADATA_KEY,
    COMPUTER_PERCEPTION_METADATA_KEY,
    COMPUTER_SESSION_ID_METADATA_KEY,
    ELEMENT_EXTRACTION_INCOMPLETE_KEY,
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerPerceptionMode,
    ComputerTarget,
    NormalizedPoint,
)
from xagent.core.context_ref import ContextReference


class FakeObservationStore:
    async def save_screenshot(self, **kwargs: Any) -> ContextReference:
        return ContextReference(
            file_ref={
                "file_id": "native-shot",
                "filename": "native.png",
                "mime_type": kwargs["mime_type"],
            },
            metadata={
                COMPUTER_SESSION_ID_METADATA_KEY: kwargs["session_id"],
                COMPUTER_FRAME_ID_METADATA_KEY: kwargs["frame_id"],
            },
        )


class FakeCuaDriver:
    def __init__(
        self,
        *,
        windows: list[dict[str, Any]] | None = None,
        elements: list[dict[str, Any]] | None = None,
        escalation: dict[str, Any] | None = None,
        background_input: dict[str, Any] | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False
        self.windows = windows if windows is not None else self._default_windows()
        self.elements = elements if elements is not None else self._default_elements()
        self.escalation = escalation
        self.background_input = background_input

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> CuaDriverResult:
        payload = dict(arguments or {})
        self.calls.append((name, payload))
        if name == "list_windows":
            return CuaDriverResult(structured={"windows": self.windows})
        if name == "get_window_state":
            structured: dict[str, Any] = {
                "window_id": payload["window_id"],
                "pid": payload["pid"],
                "url": "https://example.com/inbox",
                "element_count": 2,
                "screenshot_width": 1200,
                "screenshot_height": 800,
                "elements": self.elements,
            }
            if self.escalation is not None:
                structured["escalation"] = self.escalation
            if self.background_input is not None:
                structured["background_input"] = self.background_input
            return CuaDriverResult(
                structured=structured,
                image_bytes=b"native-png",
                image_mime_type="image/png",
            )
        if name == "health_report":
            return CuaDriverResult(structured={"schema_version": "1", "overall": "ok"})
        return CuaDriverResult(
            structured={"effect": "confirmed", "verified": True},
            text=f"{name} completed",
        )

    async def close(self) -> None:
        self.closed = True

    @staticmethod
    def _default_elements() -> list[dict[str, Any]]:
        return [
            {
                "element_index": 4,
                "element_token": "snapshot-1:4",
                "role": "AXButton",
                "label": "Continue",
                "frame": {"x": 200, "y": 300, "w": 200, "h": 80},
            },
            {
                "element_index": 5,
                "element_token": "snapshot-1:5",
                "role": "AXSecureTextField",
                "label": "Password",
                "value": "do-not-leak",
                "frame": {"x": 300, "y": 450, "w": 300, "h": 50},
            },
            {
                "element_index": 6,
                "element_token": "snapshot-1:6",
                "role": "AXTextField",
                "label": "Address and search bar",
                "value": "https://example.com/inbox",
                "frame": {"x": 180, "y": 220, "w": 760, "h": 40},
            },
        ]

    @staticmethod
    def _default_windows() -> list[dict[str, Any]]:
        return [
            {
                "window_id": 10,
                "pid": 100,
                "app_name": "Google Chrome",
                "title": "Background",
                "bounds": {"x": 10, "y": 10, "width": 900, "height": 700},
                "z_index": 1,
                "is_on_screen": True,
                "on_current_space": False,
            },
            {
                "window_id": 20,
                "pid": 200,
                "app_name": "Google Chrome",
                "title": "Inbox",
                "bounds": {"x": 100, "y": 200, "width": 1000, "height": 800},
                "z_index": 9,
                "is_on_screen": True,
                "on_current_space": True,
            },
        ]


class FakeNativeBrowserNavigator:
    def __init__(self, *, supported: bool) -> None:
        self.supported = supported
        self.calls: list[tuple[Any, str]] = []

    def supports(self, target: Any) -> bool:
        return self.supported

    async def navigate(
        self,
        target: Any,
        url: str,
    ) -> NativeBrowserNavigationResult:
        self.calls.append((target, url))
        return NativeBrowserNavigationResult(
            route="fake_native_navigation",
            browser_window_id=777,
            actual_url=url,
        )


def make_environment(
    driver: FakeCuaDriver,
    *,
    perception_mode: ComputerPerceptionMode | str = ComputerPerceptionMode.AUTO,
    native_navigator: FakeNativeBrowserNavigator | None = None,
) -> NativeBrowserEnvironment:
    return NativeBrowserEnvironment(
        session_id="task-1",
        workspace=object(),
        driver=driver,
        observation_store=FakeObservationStore(),  # type: ignore[arg-type]
        perception_mode=perception_mode,
        native_browser_navigator=(
            native_navigator or FakeNativeBrowserNavigator(supported=False)
        ),
    )


def batch(frame_id: str, action: ComputerAction) -> ComputerActionBatch:
    return ComputerActionBatch(
        session_id="task-1",
        expected_frame_id=frame_id,
        actions=[action],
    )


def test_local_browser_requires_explicit_enablement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XAGENT_NATIVE_BROWSER_ENABLED", raising=False)

    with pytest.raises(RuntimeError, match="Local browser access is disabled"):
        NativeBrowserEnvironment(session_id="task-1", workspace=object())


@pytest.mark.asyncio
async def test_local_browser_binds_frontmost_window_and_redacts_password() -> None:
    driver = FakeCuaDriver()
    environment = make_environment(driver)

    observation = await environment.observe()

    assert observation.title == "Inbox"
    assert observation.active_url == "https://example.com/inbox"
    assert observation.viewport.width == 1200
    assert observation.metadata["pid"] == 200
    assert observation.metadata["window_id"] == 20
    assert observation.environment.value == "desktop"
    assert observation.metadata["computer_runtime_kind"] == "local_browser"
    assert observation.metadata[COMPUTER_PERCEPTION_METADATA_KEY] == {
        "mode": "auto",
        "available": ["vision", "semantic"],
        "semantic_source": "accessibility",
    }
    assert observation.metadata[COMPUTER_CONTROL_METADATA_KEY] == {
        "transport": "native_accessibility",
        "scope": "window",
        "browser_debugging": False,
    }
    assert "available_windows" not in observation.metadata
    assert "Background" not in observation.model_dump_json()
    assert "move" not in observation.metadata["supported_actions"]
    assert "navigate" not in observation.metadata["supported_actions"]
    assert observation.metadata["unsupported_actions"]["navigate"] == (
        "pid_keyboard_capability_unknown"
    )
    assert observation.elements[0].element_id == "snapshot-1:4"
    assert observation.elements[1].label == "Sensitive input"
    assert observation.elements[1].text is None
    assert "do-not-leak" not in observation.model_dump_json()
    assert [name for name, _payload in driver.calls[:3]] == [
        "start_session",
        "list_windows",
        "get_window_state",
    ]


@pytest.mark.asyncio
async def test_local_browser_vision_mode_does_not_expose_semantic_targets() -> None:
    driver = FakeCuaDriver()
    environment = make_environment(
        driver,
        perception_mode=ComputerPerceptionMode.VISION,
    )

    observation = await environment.observe()

    assert observation.elements == []
    assert observation.metadata[COMPUTER_PERCEPTION_METADATA_KEY] == {
        "mode": "vision",
        "available": ["vision", "semantic"],
        "semantic_source": "accessibility",
    }
    assert ELEMENT_EXTRACTION_INCOMPLETE_KEY not in observation.metadata
    assert all(not name.startswith("browser_") for name, _ in driver.calls)


@pytest.mark.asyncio
async def test_local_browser_marks_ax_surfaces_from_parent_chain() -> None:
    elements = [
        {
            "element_index": 0,
            "element_token": "snapshot-1:0",
            "role": "AXWindow",
            "depth": 0,
            "frame": {"x": 100, "y": 200, "w": 1000, "h": 800},
        },
        {
            "element_index": 13,
            "element_token": "snapshot-1:13",
            "role": "AXButton",
            "label": "Xuye",
            "depth": 1,
            "parent_index": 0,
            "frame": {"x": 1060, "y": 220, "w": 20, "h": 20},
        },
        {
            "element_index": 15,
            "element_token": "snapshot-1:15",
            "role": "AXWebArea",
            "label": "Zhihu",
            "depth": 1,
            "parent_index": 0,
            "frame": {"x": 100, "y": 260, "w": 1000, "h": 740},
        },
        {
            "element_index": 16,
            "element_token": "snapshot-1:16",
            "role": "AXLink",
            "label": "Qin Xuye",
            "depth": 2,
            "parent_index": 15,
            "frame": {"x": 900, "y": 300, "w": 80, "h": 30},
        },
        {
            "element_index": 20,
            "element_token": "snapshot-1:20",
            "role": "AXSheet",
            "depth": 1,
            "parent_index": 0,
            "frame": {"x": 400, "y": 300, "w": 400, "h": 300},
        },
        {
            "element_index": 21,
            "element_token": "snapshot-1:21",
            "role": "AXButton",
            "label": "Confirm",
            "depth": 2,
            "parent_index": 20,
            "frame": {"x": 650, "y": 520, "w": 100, "h": 40},
        },
        {
            "element_index": 30,
            "element_token": "snapshot-1:30",
            "role": "AXButton",
            "label": "Incomplete",
            "depth": 2,
            "parent_index": 999,
            "frame": {"x": 200, "y": 300, "w": 100, "h": 40},
        },
    ]
    environment = make_environment(FakeCuaDriver(elements=elements))

    observation = await environment.observe()
    by_index = {
        element.metadata["element_index"]: element for element in observation.elements
    }

    assert by_index[13].surface is not None
    assert by_index[13].surface.value == "application_chrome"
    assert by_index[13].metadata["surface"] == "application_chrome"
    assert by_index[13].metadata["surface_root_index"] == 0
    assert by_index[15].surface is not None
    assert by_index[15].surface.value == "document"
    assert by_index[15].metadata["surface"] == "document"
    assert by_index[15].metadata["surface_root_index"] == 15
    assert by_index[16].metadata["surface"] == "document"
    assert by_index[16].metadata["surface_root_index"] == 15
    assert by_index[20].metadata["surface"] == "overlay"
    assert by_index[20].metadata["surface_root_index"] == 20
    assert by_index[21].metadata["surface"] == "overlay"
    assert by_index[21].metadata["surface_root_index"] == 20
    assert by_index[30].metadata["surface"] == "unknown"
    assert "surface_root_index" not in by_index[30].metadata


@pytest.mark.asyncio
async def test_native_browser_click_uses_bound_window_and_element_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("xagent.core.computer.native_browser.asyncio.sleep", no_sleep)
    driver = FakeCuaDriver()
    environment = make_environment(driver)
    first = await environment.observe()

    second = await environment.execute(
        batch(
            first.frame_id,
            ComputerAction(
                type=ComputerActionType.CLICK,
                target=ComputerTarget(element_id="snapshot-1:4"),
            ),
        )
    )

    name, payload = next(
        (name, payload) for name, payload in driver.calls if name == "click"
    )
    assert name == "click"
    assert payload["pid"] == 200
    assert payload["window_id"] == 20
    assert payload["element_token"] == "snapshot-1:4"
    assert payload["delivery_mode"] == "background"
    assert second.metadata["last_action_result"]["effect"] == "confirmed"
    assert [name for name, _payload in driver.calls].count("list_windows") == 3


@pytest.mark.asyncio
async def test_local_browser_refuses_ambiguous_pid_keyboard_actions() -> None:
    driver = FakeCuaDriver(
        background_input={
            "exact_window": {"pid": 200, "window_id": 20, "status": "matched"},
            "routes": [
                {"route": "accessibility", "status": "available"},
                {"route": "window_pointer", "status": "available"},
                {
                    "route": "pid_keyboard",
                    "status": "refused",
                    "reason": "same_pid_keyboard_ambiguity",
                },
            ],
        }
    )
    environment = make_environment(
        driver,
        native_navigator=FakeNativeBrowserNavigator(supported=True),
    )
    observation = await environment.observe()

    assert observation.metadata["background_input"] == driver.background_input
    assert "click" in observation.metadata["supported_actions"]
    assert "type" not in observation.metadata["supported_actions"]
    assert "replace_text" in observation.metadata["supported_actions"]
    assert "keypress" not in observation.metadata["supported_actions"]
    assert "navigate" in observation.metadata["supported_actions"]
    assert observation.metadata["unsupported_actions"] == {
        "type": "unscoped_keyboard_input_disabled",
        "keypress": "unscoped_keyboard_input_disabled",
    }

    with pytest.raises(ValueError, match="unscoped_keyboard_input_disabled"):
        await environment.execute(
            batch(
                observation.frame_id,
                ComputerAction(type=ComputerActionType.TYPE, text="unsafe"),
            )
        )


@pytest.mark.asyncio
async def test_local_browser_refuses_keyboard_when_driver_capability_is_absent() -> (
    None
):
    observation = await make_environment(FakeCuaDriver()).observe()

    assert "type" not in observation.metadata["supported_actions"]
    assert observation.metadata["unsupported_actions"]["type"] == (
        "unscoped_keyboard_input_disabled"
    )


@pytest.mark.asyncio
async def test_local_browser_limits_keyboard_input_to_document_text_targets() -> None:
    elements = [
        {
            "element_index": 0,
            "element_token": "snapshot-1:0",
            "role": "AXWindow",
            "depth": 0,
            "frame": {"x": 100, "y": 200, "w": 1000, "h": 800},
        },
        {
            "element_index": 1,
            "element_token": "snapshot-1:1",
            "role": "AXTextField",
            "label": "Address and search bar",
            "depth": 1,
            "parent_index": 0,
            "frame": {"x": 180, "y": 220, "w": 760, "h": 40},
        },
        {
            "element_index": 2,
            "element_token": "snapshot-1:2",
            "role": "AXWebArea",
            "depth": 1,
            "parent_index": 0,
            "frame": {"x": 100, "y": 280, "w": 1000, "h": 720},
        },
        {
            "element_index": 3,
            "element_token": "snapshot-1:3",
            "role": "AXTextField",
            "label": "Search",
            "depth": 2,
            "parent_index": 2,
            "frame": {"x": 300, "y": 350, "w": 400, "h": 40},
        },
    ]
    driver = FakeCuaDriver(
        elements=elements,
        background_input={
            "routes": [{"route": "pid_keyboard", "status": "available"}],
        },
    )
    environment = make_environment(driver)
    observation = await environment.observe()

    assert "replace_text" in observation.metadata["supported_actions"]
    assert "type" not in observation.metadata["supported_actions"]
    assert "keypress" not in observation.metadata["supported_actions"]
    assert observation.metadata["unsupported_actions"]["type"] == (
        "unscoped_keyboard_input_disabled"
    )

    after_document_edit = await environment.execute(
        batch(
            observation.frame_id,
            ComputerAction(
                type=ComputerActionType.REPLACE_TEXT,
                target=ComputerTarget(element_id="snapshot-1:3"),
                text="safe site search",
            ),
        )
    )
    action_call_count = len(driver.calls)

    with pytest.raises(ValueError, match="limited to document elements"):
        await environment.execute(
            batch(
                after_document_edit.frame_id,
                ComputerAction(
                    type=ComputerActionType.REPLACE_TEXT,
                    target=ComputerTarget(element_id="snapshot-1:1"),
                    text="file:///Users/admin/.ssh/id_rsa",
                ),
            )
        )

    set_value = next(
        payload
        for name, payload in driver.calls[:action_call_count]
        if name == "set_value"
    )
    assert set_value["element_token"] == "snapshot-1:3"
    assert set_value["value"] == "safe site search"
    assert all(
        name not in {"hotkey", "type_text"}
        for name, _ in driver.calls[:action_call_count]
    )
    assert len(driver.calls) == action_call_count + 1  # Bound-window revalidation only.


@pytest.mark.asyncio
async def test_local_browser_replace_text_does_not_use_pixel_occlusion() -> None:
    elements = [
        {
            "element_index": 0,
            "element_token": "snapshot-1:0",
            "role": "AXWindow",
            "depth": 0,
            "frame": {"x": 100, "y": 200, "w": 1000, "h": 800},
        },
        {
            "element_index": 1,
            "element_token": "snapshot-1:1",
            "role": "AXWebArea",
            "depth": 1,
            "parent_index": 0,
            "frame": {"x": 100, "y": 280, "w": 1000, "h": 720},
        },
        {
            "element_index": 2,
            "element_token": "snapshot-1:2",
            "role": "AXTextField",
            "label": "Search",
            "depth": 2,
            "parent_index": 1,
            "frame": {"x": 300, "y": 350, "w": 400, "h": 40},
        },
    ]
    driver = FakeCuaDriver(elements=elements)
    driver.windows.append(
        {
            "window_id": 31,
            "pid": 300,
            "app_name": "Terminal",
            "title": "Overlapping window",
            "bounds": {"x": 100, "y": 200, "width": 1000, "height": 800},
            "z_index": 10,
            "is_on_screen": True,
            "on_current_space": True,
        }
    )
    environment = make_environment(driver)
    observation = await environment.observe()

    await environment.execute(
        batch(
            observation.frame_id,
            ComputerAction(
                type=ComputerActionType.REPLACE_TEXT,
                target=ComputerTarget(element_id="snapshot-1:2"),
                text="semantic input",
            ),
        )
    )

    set_value = next(payload for name, payload in driver.calls if name == "set_value")
    assert set_value["element_token"] == "snapshot-1:2"
    assert set_value["value"] == "semantic input"


@pytest.mark.asyncio
async def test_local_browser_rejects_sensitive_document_text_targets() -> None:
    elements = [
        {
            "element_index": 0,
            "element_token": "snapshot-1:0",
            "role": "AXWindow",
            "depth": 0,
            "frame": {"x": 100, "y": 200, "w": 1000, "h": 800},
        },
        {
            "element_index": 1,
            "element_token": "snapshot-1:1",
            "role": "AXWebArea",
            "depth": 1,
            "parent_index": 0,
            "frame": {"x": 100, "y": 280, "w": 1000, "h": 720},
        },
        {
            "element_index": 2,
            "element_token": "snapshot-1:2",
            "role": "AXTextField",
            "label": "一次性验证码 (OTP)",
            "depth": 2,
            "parent_index": 1,
            "frame": {"x": 300, "y": 350, "w": 400, "h": 40},
        },
    ]
    driver = FakeCuaDriver(elements=elements)
    environment = make_environment(driver)
    observation = await environment.observe()

    sensitive = next(
        element
        for element in observation.elements
        if element.element_id == "snapshot-1:2"
    )
    assert sensitive.label == "Sensitive input"
    assert sensitive.metadata["sensitive"] is True

    with pytest.raises(ValueError, match="sensitive input"):
        await environment.execute(
            batch(
                observation.frame_id,
                ComputerAction(
                    type=ComputerActionType.REPLACE_TEXT,
                    target=ComputerTarget(element_id="snapshot-1:2"),
                    text="123456",
                ),
            )
        )

    assert all(name != "set_value" for name, _ in driver.calls)


@pytest.mark.asyncio
async def test_local_browser_does_not_expose_snapshot_indices_without_tokens() -> None:
    driver = FakeCuaDriver(
        elements=[
            {
                "element_index": 4,
                "role": "AXButton",
                "label": "Stale target",
                "frame": {"x": 200, "y": 300, "w": 200, "h": 80},
            }
        ]
    )

    observation = await make_environment(driver).observe()

    assert observation.elements == []


@pytest.mark.asyncio
async def test_native_browser_requires_driver_proof_for_foreground_delivery() -> None:
    driver = FakeCuaDriver()
    environment = make_environment(driver)
    first = await environment.observe()

    with pytest.raises(ValueError, match="escalation recommendation"):
        await environment.execute(
            batch(
                first.frame_id,
                ComputerAction(
                    type=ComputerActionType.CLICK,
                    target=ComputerTarget(element_id="snapshot-1:4"),
                    metadata={"delivery_mode": "foreground"},
                ),
            )
        )


@pytest.mark.asyncio
async def test_native_browser_allows_driver_recommended_foreground_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("xagent.core.computer.native_browser.asyncio.sleep", no_sleep)
    driver = FakeCuaDriver(
        escalation={"recommended": "foreground", "reason": "background did not land"}
    )
    environment = make_environment(driver)
    first = await environment.observe()

    await environment.execute(
        batch(
            first.frame_id,
            ComputerAction(
                type=ComputerActionType.CLICK,
                target=ComputerTarget(element_id="snapshot-1:4"),
                metadata={"delivery_mode": "foreground"},
            ),
        )
    )

    click = next(payload for name, payload in driver.calls if name == "click")
    assert click["delivery_mode"] == "foreground"


@pytest.mark.asyncio
async def test_local_browser_rejects_pixel_action_after_window_geometry_changes() -> (
    None
):
    driver = FakeCuaDriver()
    environment = make_environment(driver)
    first = await environment.observe()
    assert first.metadata["coordinate_frame"] == {
        "screenshot_width": 1200,
        "screenshot_height": 800,
        "screenshot_fresh": True,
        "window_bounds": {
            "x": 100.0,
            "y": 200.0,
            "width": 1000.0,
            "height": 800.0,
        },
    }
    driver.windows[1]["bounds"]["width"] = 900

    with pytest.raises(ComputerFrameMismatchError, match="moved or resized"):
        await environment.execute(
            batch(
                first.frame_id,
                ComputerAction(
                    type=ComputerActionType.CLICK,
                    target=ComputerTarget(
                        point=NormalizedPoint(x=0.5, y=0.5),
                    ),
                ),
            )
        )

    assert all(name != "click" for name, _payload in driver.calls)


@pytest.mark.asyncio
async def test_local_browser_rejects_actions_after_bound_window_disappears() -> None:
    driver = FakeCuaDriver()
    environment = make_environment(driver)
    first = await environment.observe()
    driver.windows = [window for window in driver.windows if window["window_id"] != 20]

    with pytest.raises(ComputerTargetNotFoundError, match="disappeared"):
        await environment.execute(
            batch(
                first.frame_id,
                ComputerAction(
                    type=ComputerActionType.CLICK,
                    target=ComputerTarget(element_id="snapshot-1:4"),
                ),
            )
        )

    assert all(name != "click" for name, _payload in driver.calls)


@pytest.mark.asyncio
async def test_local_browser_rejects_pixel_action_under_cross_app_occluding_window() -> (
    None
):
    driver = FakeCuaDriver()
    driver.windows.append(
        {
            "window_id": 31,
            "pid": 300,
            "app_name": "Terminal",
            "title": "Unrelated foreground window",
            "bounds": {"x": 100, "y": 200, "width": 1000, "height": 800},
            "z_index": 10,
            "is_on_screen": True,
            "on_current_space": True,
        }
    )
    environment = NativeBrowserEnvironment(
        session_id="task-1",
        workspace=object(),
        driver=driver,
        observation_store=FakeObservationStore(),  # type: ignore[arg-type]
        target_pid=200,
        target_window_id=20,
        native_browser_navigator=FakeNativeBrowserNavigator(supported=False),
    )
    first = await environment.observe()

    with pytest.raises(ComputerTargetNotFoundError, match="Pixel action is ambiguous"):
        await environment.execute(
            batch(
                first.frame_id,
                ComputerAction(
                    type=ComputerActionType.CLICK,
                    target=ComputerTarget(
                        point=NormalizedPoint(x=0.5, y=0.5),
                    ),
                ),
            )
        )

    assert all(name != "click" for name, _payload in driver.calls)


@pytest.mark.asyncio
async def test_local_browser_treats_unknown_overlapping_z_index_as_occluding() -> None:
    driver = FakeCuaDriver()
    driver.windows.append(
        {
            "window_id": 31,
            "pid": 300,
            "app_name": "Terminal",
            "title": "Unknown stacking order",
            "bounds": {"x": 100, "y": 200, "width": 1000, "height": 800},
            "is_on_screen": True,
            "on_current_space": True,
        }
    )
    environment = NativeBrowserEnvironment(
        session_id="task-1",
        workspace=object(),
        driver=driver,
        observation_store=FakeObservationStore(),  # type: ignore[arg-type]
        target_pid=200,
        target_window_id=20,
        native_browser_navigator=FakeNativeBrowserNavigator(supported=False),
    )
    first = await environment.observe()

    with pytest.raises(ComputerTargetNotFoundError, match="Pixel action is ambiguous"):
        await environment.execute(
            batch(
                first.frame_id,
                ComputerAction(
                    type=ComputerActionType.CLICK,
                    target=ComputerTarget(point=NormalizedPoint(x=0.5, y=0.5)),
                ),
            )
        )

    assert all(name != "click" for name, _payload in driver.calls)


@pytest.mark.asyncio
async def test_local_browser_prioritizes_overlay_elements_beyond_context_limit() -> (
    None
):
    elements: list[dict[str, Any]] = [
        {
            "element_index": 0,
            "element_token": "snapshot-1:0",
            "role": "AXWindow",
            "depth": 0,
            "frame": {"x": 100, "y": 200, "w": 1000, "h": 800},
        }
    ]
    elements.extend(
        {
            "element_index": index,
            "element_token": f"snapshot-1:{index}",
            "role": "AXButton",
            "label": f"Action {index}",
            "depth": 1,
            "parent_index": 0,
            "enabled": True,
            "frame": {
                "x": 120 + (index % 20) * 30,
                "y": 260 + (index % 15) * 25,
                "w": 24,
                "h": 20,
            },
        }
        for index in range(1, 131)
    )
    elements.extend(
        [
            {
                "element_index": 200,
                "element_token": "snapshot-1:200",
                "role": "AXMenu",
                "depth": 1,
                "parent_index": 0,
                "frame": {"x": 800, "y": 260, "w": 200, "h": 180},
            },
            {
                "element_index": 201,
                "element_token": "snapshot-1:201",
                "role": "AXMenuItem",
                "label": "My profile",
                "depth": 2,
                "parent_index": 200,
                "enabled": True,
                "frame": {"x": 820, "y": 280, "w": 160, "h": 30},
            },
        ]
    )
    driver = FakeCuaDriver(elements=elements)
    environment = make_environment(driver)

    observation = await environment.observe()

    assert len(observation.elements) == 100
    profile = next(
        element for element in observation.elements if element.label == "My profile"
    )
    assert profile.surface is not None
    assert profile.surface.value == "overlay"
    state_call = next(
        payload for name, payload in driver.calls if name == "get_window_state"
    )
    assert state_call["max_elements"] == 2_000


@pytest.mark.asyncio
async def test_local_browser_refuses_pixel_fallback_when_ax_owner_is_unprovable() -> (
    None
):
    class RefusingElementDriver(FakeCuaDriver):
        async def call_tool(
            self,
            name: str,
            arguments: dict[str, Any] | None = None,
        ) -> CuaDriverResult:
            payload = dict(arguments or {})
            if name == "click" and "element_token" in payload:
                self.calls.append((name, payload))
                raise CuaDriverError(
                    "Background input refused (element_outside_target_window)"
                )
            return await super().call_tool(name, arguments)

    driver = RefusingElementDriver()
    environment = make_environment(driver)
    first = await environment.observe()

    with pytest.raises(CuaDriverError, match="element_outside_target_window"):
        await environment.execute(
            batch(
                first.frame_id,
                ComputerAction(
                    type=ComputerActionType.CLICK,
                    target=ComputerTarget(element_id="snapshot-1:4"),
                ),
            )
        )

    clicks = [payload for name, payload in driver.calls if name == "click"]
    assert len(clicks) == 1
    assert clicks[0]["element_token"] == "snapshot-1:4"
    assert "x" not in clicks[0]
    assert "y" not in clicks[0]


@pytest.mark.asyncio
async def test_local_browser_keeps_fresh_semantics_when_transient_capture_is_unavailable() -> (
    None
):
    class TransientCaptureDriver(FakeCuaDriver):
        async def call_tool(
            self,
            name: str,
            arguments: dict[str, Any] | None = None,
        ) -> CuaDriverResult:
            if name == "get_window_state" and any(
                called_name == "click" for called_name, _payload in self.calls
            ):
                payload = dict(arguments or {})
                self.calls.append((name, payload))
                return CuaDriverResult(
                    structured={
                        "window_id": payload["window_id"],
                        "pid": payload["pid"],
                        "element_count": 1,
                        "elements": [
                            {
                                "element_index": 20,
                                "element_token": "snapshot-2:20",
                                "role": "AXLink",
                                "label": "My profile",
                                "frame": {
                                    "x": 800,
                                    "y": 260,
                                    "w": 120,
                                    "h": 30,
                                },
                            }
                        ],
                        "background_input": {
                            "observation": {"one_shot_capture": "unavailable"}
                        },
                    }
                )
            return await super().call_tool(name, arguments)

    driver = TransientCaptureDriver()
    environment = make_environment(driver)
    first = await environment.observe()

    second = await environment.execute(
        batch(
            first.frame_id,
            ComputerAction(
                type=ComputerActionType.CLICK,
                target=ComputerTarget(element_id="snapshot-1:4"),
            ),
        )
    )

    assert second.frame_id != first.frame_id
    assert second.screenshot.file_id == first.screenshot.file_id
    assert second.screenshot.metadata["reused_from_frame_id"] == first.frame_id
    assert second.metadata["screenshot_fresh"] is False
    assert second.metadata["computer_perception"]["available"] == ["semantic"]
    assert [element.label for element in second.elements] == ["My profile"]

    with pytest.raises(ComputerFrameMismatchError, match="fresh semantic elements"):
        await environment.execute(
            batch(
                second.frame_id,
                ComputerAction(
                    type=ComputerActionType.CLICK,
                    target=ComputerTarget(
                        point=NormalizedPoint(x=0.5, y=0.5),
                    ),
                ),
            )
        )


@pytest.mark.asyncio
async def test_local_browser_waits_for_unverified_semantic_state_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("xagent.core.computer.native_browser.asyncio.sleep", no_sleep)

    class DelayedSemanticDriver(FakeCuaDriver):
        state_reads = 0

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, Any] | None = None,
        ) -> CuaDriverResult:
            if name == "click":
                payload = dict(arguments or {})
                self.calls.append((name, payload))
                return CuaDriverResult(
                    structured={"effect": "unverifiable"},
                    text="AXPress posted",
                )
            if name == "get_window_state":
                self.state_reads += 1
                if self.state_reads >= 3:
                    self.elements = [
                        {
                            "element_index": 20,
                            "element_token": "snapshot-2:20",
                            "role": "AXLink",
                            "label": "My profile",
                            "frame": {"x": 250, "y": 360, "w": 120, "h": 30},
                        }
                    ]
            return await super().call_tool(name, arguments)

    driver = DelayedSemanticDriver()
    environment = make_environment(driver)
    first = await environment.observe()

    second = await environment.execute(
        batch(
            first.frame_id,
            ComputerAction(
                type=ComputerActionType.CLICK,
                target=ComputerTarget(element_id="snapshot-1:4"),
            ),
        )
    )

    assert driver.state_reads == 3
    assert [element.label for element in second.elements] == ["My profile"]
    assert second.metadata["last_action_result"]["effect"] == "unverifiable"


@pytest.mark.asyncio
async def test_local_browser_records_stall_without_authorizing_foreground(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("xagent.core.computer.native_browser.asyncio.sleep", no_sleep)

    class UnchangedSemanticDriver(FakeCuaDriver):
        async def call_tool(
            self,
            name: str,
            arguments: dict[str, Any] | None = None,
        ) -> CuaDriverResult:
            if name == "click":
                payload = dict(arguments or {})
                self.calls.append((name, payload))
                return CuaDriverResult(
                    structured={"effect": "unverifiable"},
                    text="AXPress posted",
                )
            return await super().call_tool(name, arguments)

    driver = UnchangedSemanticDriver()
    environment = make_environment(driver)
    first = await environment.observe()
    stalled = await environment.execute(
        batch(
            first.frame_id,
            ComputerAction(
                type=ComputerActionType.CLICK,
                target=ComputerTarget(element_id="snapshot-1:4"),
            ),
        )
    )

    assert stalled.metadata["last_action_result"]["code"] == (
        "no_observable_state_change_after_background_delivery"
    )
    assert "escalation" not in stalled.metadata["last_action_result"]

    with pytest.raises(ValueError, match="current cua-driver escalation"):
        await environment.execute(
            batch(
                stalled.frame_id,
                ComputerAction(
                    type=ComputerActionType.CLICK,
                    target=ComputerTarget(element_id="snapshot-1:4"),
                    metadata={"delivery_mode": "foreground"},
                ),
            )
        )


@pytest.mark.asyncio
async def test_native_browser_defensively_rejects_incomplete_drag() -> None:
    driver = FakeCuaDriver()
    environment = make_environment(driver)
    first = await environment.observe()
    invalid_drag = ComputerAction.model_construct(
        type=ComputerActionType.DRAG,
        target=None,
        url=None,
        text=None,
        keys=[],
        delta_x=0,
        delta_y=0,
        start=None,
        end=None,
        duration_ms=0,
        metadata={},
    )

    with pytest.raises(ValueError, match="drag requires start and end points"):
        await environment.execute(batch(first.frame_id, invalid_drag))


@pytest.mark.asyncio
async def test_local_browser_navigates_with_native_browser_adapter() -> None:
    driver = FakeCuaDriver()
    navigator = FakeNativeBrowserNavigator(supported=True)
    environment = NativeBrowserEnvironment(
        session_id="task-1",
        workspace=object(),
        driver=driver,
        observation_store=FakeObservationStore(),  # type: ignore[arg-type]
        native_browser_navigator=navigator,
    )
    first = await environment.observe()

    await environment.execute(
        batch(
            first.frame_id,
            ComputerAction(
                type=ComputerActionType.NAVIGATE,
                url="https://example.com/account",
            ),
        )
    )

    assert len(navigator.calls) == 1
    navigated_target, navigated_url = navigator.calls[0]
    assert navigated_target.pid == 200
    assert navigated_target.window_id == 20
    assert navigated_url == "https://example.com/account"
    assert environment.current_observation is not None
    assert (
        environment.current_observation.metadata["last_action_result"]["actual_url"]
        == "https://example.com/account"
    )
    assert all(name not in {"set_value", "press_key"} for name, _ in driver.calls)
    assert [name for name, _payload in driver.calls].count("get_window_state") == 2
    await environment.close()
    assert ("end_session", {"session": "task-1"}) in driver.calls
    assert driver.closed is True
    assert environment.closed is True


@pytest.mark.asyncio
async def test_local_browser_close_can_retry_after_cancellation() -> None:
    class CancelOnceDriver(FakeCuaDriver):
        close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise asyncio.CancelledError
            self.closed = True

    driver = CancelOnceDriver()
    environment = make_environment(driver)
    await environment.observe()

    with pytest.raises(asyncio.CancelledError):
        await environment.close()
    await environment.close()

    assert driver.close_calls == 2
    assert driver.closed is True


@pytest.mark.asyncio
async def test_local_browser_falls_back_to_address_field_when_keyboard_is_safe() -> (
    None
):
    driver = FakeCuaDriver(
        background_input={
            "routes": [{"route": "pid_keyboard", "status": "available"}],
        }
    )
    environment = NativeBrowserEnvironment(
        session_id="task-1",
        workspace=object(),
        driver=driver,
        observation_store=FakeObservationStore(),  # type: ignore[arg-type]
        native_browser_navigator=FakeNativeBrowserNavigator(supported=False),
    )
    first = await environment.observe()

    await environment.execute(
        batch(
            first.frame_id,
            ComputerAction(
                type=ComputerActionType.NAVIGATE,
                url="https://example.com/account",
            ),
        )
    )

    set_value = next(payload for name, payload in driver.calls if name == "set_value")
    assert set_value["element_token"] == "snapshot-1:6"
    assert set_value["value"] == "https://example.com/account"
    press_key = next(payload for name, payload in driver.calls if name == "press_key")
    assert press_key["element_token"] == "snapshot-1:6"
    assert press_key["key"] == "return"
    assert press_key["delivery_mode"] == "background"


@pytest.mark.asyncio
async def test_local_browser_refuses_missing_or_hidden_window() -> None:
    driver = FakeCuaDriver(windows=[])
    with pytest.raises(RuntimeError, match="No visible Google Chrome window"):
        await make_environment(driver).observe()

    hidden = FakeCuaDriver(
        windows=[
            {
                "window_id": 10,
                "pid": 100,
                "app_name": "Google Chrome",
                "title": "Hidden",
                "bounds": {"x": 10, "y": 10, "width": 900, "height": 700},
                "z_index": 1,
                "is_on_screen": False,
                "on_current_space": False,
            }
        ]
    )
    with pytest.raises(RuntimeError, match="No visible Google Chrome window"):
        await make_environment(hidden).observe()


@pytest.mark.asyncio
async def test_local_browser_never_selects_a_non_browser_window() -> None:
    driver = FakeCuaDriver(
        windows=[
            {
                "window_id": 30,
                "pid": 300,
                "app_name": "Music",
                "title": "Songs",
                "bounds": {"x": 10, "y": 10, "width": 900, "height": 700},
                "z_index": 20,
                "is_on_screen": True,
                "on_current_space": True,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="No visible Google Chrome window"):
        await make_environment(driver).observe()


@pytest.mark.asyncio
async def test_local_browser_honors_exact_user_selected_window() -> None:
    driver = FakeCuaDriver()
    environment = NativeBrowserEnvironment(
        session_id="task-1",
        workspace=object(),
        driver=driver,
        observation_store=FakeObservationStore(),  # type: ignore[arg-type]
        target_pid=100,
        target_window_id=10,
    )

    # Window 10 is on another Space and is therefore not a valid selected target.
    with pytest.raises(RuntimeError, match="no longer visible"):
        await environment.observe()

    visible_driver = FakeCuaDriver()
    visible_driver.windows[0]["on_current_space"] = True
    selected = NativeBrowserEnvironment(
        session_id="task-2",
        workspace=object(),
        driver=visible_driver,
        observation_store=FakeObservationStore(),  # type: ignore[arg-type]
        target_pid=100,
        target_window_id=10,
    )
    observation = await selected.observe()
    assert observation.metadata["pid"] == 100
    assert observation.metadata["window_id"] == 10
