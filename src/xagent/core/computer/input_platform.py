from __future__ import annotations

import sys
from enum import Enum
from typing import Any


class ComputerInputPlatform(str, Enum):
    MACOS = "macos"
    WINDOWS = "windows"
    LINUX = "linux"
    CHROMEOS = "chromeos"
    ANDROID = "android"
    UNKNOWN = "unknown"


class ComputerPrimaryModifier(str, Enum):
    META = "META"
    CTRL = "CTRL"


def normalize_computer_input_platform(value: str | None) -> ComputerInputPlatform:
    normalized = str(value or "").strip().lower()
    aliases = {
        "darwin": ComputerInputPlatform.MACOS,
        "mac": ComputerInputPlatform.MACOS,
        "macos": ComputerInputPlatform.MACOS,
        "win": ComputerInputPlatform.WINDOWS,
        "win32": ComputerInputPlatform.WINDOWS,
        "windows": ComputerInputPlatform.WINDOWS,
        "linux": ComputerInputPlatform.LINUX,
        "openbsd": ComputerInputPlatform.LINUX,
        "chromeos": ComputerInputPlatform.CHROMEOS,
        "android": ComputerInputPlatform.ANDROID,
    }
    return aliases.get(normalized, ComputerInputPlatform.UNKNOWN)


def host_computer_input_platform() -> ComputerInputPlatform:
    return normalize_computer_input_platform(sys.platform)


def primary_modifier_for_platform(
    platform: ComputerInputPlatform | str,
) -> ComputerPrimaryModifier:
    normalized = (
        platform
        if isinstance(platform, ComputerInputPlatform)
        else normalize_computer_input_platform(platform)
    )
    if normalized is ComputerInputPlatform.MACOS:
        return ComputerPrimaryModifier.META
    return ComputerPrimaryModifier.CTRL


def computer_input_metadata(
    platform: ComputerInputPlatform | str,
) -> dict[str, Any]:
    normalized = (
        platform
        if isinstance(platform, ComputerInputPlatform)
        else normalize_computer_input_platform(platform)
    )
    return {
        "platform": normalized.value,
        "primary_modifier": primary_modifier_for_platform(normalized).value,
    }
