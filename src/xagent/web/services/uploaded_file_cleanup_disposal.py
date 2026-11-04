"""Bounded filesystem disposal for retained upload cleanup manifests."""

from __future__ import annotations

import os
from dataclasses import dataclass

DISPOSALS_PER_RUN = 128
DIRECTORY_ENTRIES_PER_RUN = 256
MAX_DISPOSAL_DEPTH = 64


class CleanupResourceUncertain(RuntimeError):
    """A retained manifest needs retry or operator reconciliation."""


class CleanupBudgetExhausted(RuntimeError):
    """The retained claim has more bounded work for a later activation."""


@dataclass
class CleanupWorkBudget:
    """Limit destructive attempts and directory enumeration per activation."""

    disposals: int = DISPOSALS_PER_RUN
    entries: int = DIRECTORY_ENTRIES_PER_RUN

    def consume_disposal(self) -> None:
        if self.disposals <= 0:
            raise CleanupBudgetExhausted()
        self.disposals -= 1

    def consume_entry(self) -> None:
        if self.entries <= 0:
            raise CleanupBudgetExhausted()
        self.entries -= 1


def _directory_identity(info: os.stat_result) -> tuple[int, int, int]:
    return info.st_dev, info.st_ino, info.st_mode


def dispose_directory(
    parent: int,
    name: str,
    budget: CleanupWorkBudget,
    depth: int = 0,
) -> None:
    """Drain an owned quarantine without following links or retaining offsets."""
    if depth >= MAX_DISPOSAL_DEPTH:
        raise CleanupResourceUncertain(
            "Converter directory exceeds cleanup depth limit"
        )
    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    descriptor = os.open(
        name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
    )
    try:
        if _directory_identity(os.fstat(descriptor)) != _directory_identity(before):
            raise CleanupResourceUncertain("Converter directory was replaced")
        while True:
            budget.consume_entry()
            # Closing before mutation makes filesystem deletions the only durable
            # progress; a later process never depends on a directory offset.
            with os.scandir(descriptor) as entries:
                entry = next(entries, None)
            if entry is None:
                break
            if entry.is_dir(follow_symlinks=False):
                dispose_directory(descriptor, entry.name, budget, depth + 1)
            else:
                budget.consume_disposal()
                os.unlink(entry.name, dir_fd=descriptor)
        if _directory_identity(
            os.stat(name, dir_fd=parent, follow_symlinks=False)
        ) != _directory_identity(before):
            raise CleanupResourceUncertain("Converter directory was replaced")
        budget.consume_disposal()
        os.rmdir(name, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(descriptor)
