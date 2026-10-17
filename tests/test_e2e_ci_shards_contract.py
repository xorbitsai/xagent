"""Every E2E test file runs in exactly one CI e2e shard."""

from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def _selected(paths: str) -> set[str]:
    selected: set[str] = set()
    ignored: set[str] = set()
    for arg in paths.split():
        if arg.startswith("--ignore="):
            ignored.add(arg.removeprefix("--ignore="))
        elif (ROOT / arg).is_dir():
            selected.update(
                p.relative_to(ROOT).as_posix() for p in (ROOT / arg).rglob("test_*.py")
            )
        else:
            selected.add(arg)
    return selected - ignored


def test_every_e2e_file_is_in_exactly_one_shard():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    shards = workflow["jobs"]["e2e"]["strategy"]["matrix"]["include"]
    counts = Counter(path for shard in shards for path in _selected(shard["paths"]))
    files = {
        p.relative_to(ROOT).as_posix() for p in (ROOT / "tests/e2e").rglob("test_*.py")
    }
    assert set(counts) == files
    assert {path: n for path, n in counts.items() if n != 1} == {}
