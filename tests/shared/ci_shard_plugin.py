"""Run one slice of the collected tests, so a CI leg can be split across jobs.

``--shard=K/N`` keeps the tests of the K-th of N groups of test files. A file
always lands in one shard as a whole, so module fixtures still run once and
tests that rely on running after one another in their file keep doing so.

Files are spread greedily by test count, largest first, which keeps the shards
close in size without a recorded duration file to maintain. Every shard -- and
every xdist worker inside one -- derives the same assignment from the same
collection, so the N shards together run each selected test exactly once.
"""

from __future__ import annotations

from collections import Counter

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--shard",
        default=None,
        metavar="K/N",
        help="run only the K-th of N file groups of the selected tests (1-based)",
    )


def parse_shard(value: str) -> tuple[int, int]:
    """Return ``(index, total)`` for ``"K/N"``, 1-based."""
    try:
        index, total = (int(part) for part in value.split("/"))
    except ValueError:
        raise pytest.UsageError(f"--shard expects K/N, got {value!r}") from None
    if not 1 <= index <= total:
        raise pytest.UsageError(f"--shard {value!r} needs 1 <= K <= N")
    return index, total


def assign_files(counts: dict[str, int], total: int) -> dict[str, int]:
    """Map each file to a 0-based shard, balancing the test count per shard."""
    loads = [0] * total
    owner: dict[str, int] = {}
    for path, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        shard = min(range(total), key=lambda s: (loads[s], s))
        owner[path] = shard
        loads[shard] += count
    return owner


def _file_of(item: pytest.Item) -> str:
    return item.nodeid.split("::", 1)[0]


# Last, so the split is computed over what `-m`/`-k` left selected and the
# shards balance the tests that will actually run.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    value = config.getoption("--shard")
    if value is None:
        return
    index, total = parse_shard(value)
    owner = assign_files(Counter(_file_of(item) for item in items), total)

    keep: list[pytest.Item] = []
    drop: list[pytest.Item] = []
    for item in items:
        (keep if owner[_file_of(item)] == index - 1 else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
    items[:] = keep
