"""Every E2E and Milvus test file runs in exactly one CI shard of its job."""

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


def _shard_counts(job: str) -> Counter[str]:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    shards = workflow["jobs"][job]["strategy"]["matrix"]["include"]
    return Counter(path for shard in shards for path in _selected(shard["paths"]))


def test_every_e2e_file_is_in_exactly_one_shard():
    counts = _shard_counts("e2e")
    files = _selected("tests/e2e")
    assert set(counts) == files
    assert {path: n for path, n in counts.items() if n != 1} == {}


# Everything the Milvus job ran before it was sharded; `-m milvus` then picks
# the tests that need the server.
MILVUS_ROOTS = (
    "tests/core/tools/core/RAG_tools tests/web_integration "
    "tests/web/api/test_kb_milvus_rollback.py"
)


def test_every_milvus_file_is_in_exactly_one_shard():
    counts = _shard_counts("pytest-milvus")
    assert set(counts) == _selected(MILVUS_ROOTS)
    assert {path: n for path, n in counts.items() if n != 1} == {}
