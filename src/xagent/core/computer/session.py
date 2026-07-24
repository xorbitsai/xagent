from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class BrowserRuntimeKind(str, Enum):
    """Browser execution backends supported by the computer tool."""

    EPHEMERAL_PLAYWRIGHT = "ephemeral_playwright"
    PERSISTENT_PLAYWRIGHT = "persistent_playwright"


def validate_browser_profile_id(profile_id: str) -> str:
    """Validate one filesystem-safe browser profile component."""
    value = profile_id.strip()
    if not (
        1 <= len(value) <= 64
        and value[0].isalnum()
        and all(char.isalnum() or char in {"_", ".", "-"} for char in value)
        and value not in {".", ".."}
    ):
        raise ValueError(
            "browser profile ID must be 1-64 safe filename characters and "
            "start with a letter or number"
        )
    return value


@dataclass(frozen=True)
class ComputerSessionBinding:
    """Out-of-band binding from one task to an approved browser backend."""

    runtime_kind: BrowserRuntimeKind = BrowserRuntimeKind.EPHEMERAL_PLAYWRIGHT
    owner_task_id: str | None = None
    user_id: int | None = None
    profile_id: str = "default"
    profile_root: Path | None = None

    @property
    def is_persistent(self) -> bool:
        return self.runtime_kind is BrowserRuntimeKind.PERSISTENT_PLAYWRIGHT

    @classmethod
    def from_values(
        cls,
        *,
        runtime_kind: BrowserRuntimeKind | str,
        owner_task_id: str | None,
        user_id: int | None,
        profile_id: str,
        profile_root: Path | None,
    ) -> "ComputerSessionBinding":
        return cls(
            runtime_kind=BrowserRuntimeKind(runtime_kind),
            owner_task_id=(
                str(owner_task_id).strip() if owner_task_id is not None else None
            ),
            user_id=user_id,
            profile_id=validate_browser_profile_id(profile_id),
            profile_root=profile_root,
        )

    def persistent_profile_dir(self) -> Path | None:
        if not self.is_persistent:
            return None
        if self.user_id is None or self.user_id <= 0:
            raise ValueError(
                "persistent browser profiles require an authenticated user_id"
            )
        if self.profile_root is None:
            raise ValueError("persistent browser profiles require a profile root")
        root = self.profile_root.expanduser().resolve()
        profile_dir = (
            root / f"user_{self.user_id}" / validate_browser_profile_id(self.profile_id)
        ).resolve()
        if not profile_dir.is_relative_to(root):
            raise ValueError("browser profile path escapes the configured root")
        return profile_dir

    def manager_session_id(self, logical_session_id: str) -> str:
        if not self.is_persistent:
            return logical_session_id
        if self.user_id is None or self.user_id <= 0:
            raise ValueError(
                "persistent browser profiles require an authenticated user_id"
            )
        profile_id = validate_browser_profile_id(self.profile_id)
        return f"computer-profile:user_{self.user_id}:{profile_id}"

    def manager_owner_id(self) -> str | None:
        if not self.is_persistent:
            return None
        owner = str(self.owner_task_id or "").strip()
        if not owner:
            raise ValueError(
                "persistent browser profiles require an owning task execution"
            )
        return owner
