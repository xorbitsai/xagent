"""`--shard=K/N` splits a selection into disjoint, whole-file slices."""

import pytest

from tests.shared.ci_shard_plugin import assign_files, parse_shard

PLUGIN = "tests.shared.ci_shard_plugin"

# File name -> number of tests in it; one file in each is marked `fast`.
FILES = {"test_a": 5, "test_b": 3, "test_c": 2, "test_d": 1, "test_e": 1}


@pytest.fixture
def suite(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeini("[pytest]\nmarkers =\n    fast: a marked test\n")
    for name, count in FILES.items():
        body = ["import pytest"]
        for i in range(count):
            if i == 0:
                body.append("@pytest.mark.fast")
            body.append(f"def test_{i}(): pass")
        pytester.makepyfile(**{name: "\n".join(body)})
    return pytester


def _collected(pytester: pytest.Pytester, *args: str) -> list[str]:
    result = pytester.runpytest("-p", PLUGIN, "--collect-only", "-q", *args)
    return [line for line in result.outlines if "::" in line]


@pytest.mark.parametrize("total", [1, 2, 3, 7])
@pytest.mark.parametrize("marker", [(), ("-m", "fast")])
def test_shards_partition_the_selection_by_whole_files(suite, total, marker):
    everything = _collected(suite, *marker)
    shards = [
        _collected(suite, *marker, f"--shard={k}/{total}") for k in range(1, total + 1)
    ]

    merged = [nodeid for shard in shards for nodeid in shard]
    assert sorted(merged) == sorted(everything)

    files = [{nodeid.split("::")[0] for nodeid in shard} for shard in shards]
    for i, mine in enumerate(files):
        for theirs in files[i + 1 :]:
            assert not mine & theirs


def test_shards_balance_the_test_count():
    # Name order puts the largest file last; only largest-first comes out even.
    counts = {"a": 1, "b": 1, "c": 2, "d": 3, "e": 5}
    owner = assign_files(counts, 2)
    loads = [sum(n for f, n in counts.items() if owner[f] == s) for s in range(2)]
    assert loads == [6, 6]


@pytest.mark.parametrize("value", ["0/2", "3/2", "1", "a/b", "1/2/3"])
def test_a_malformed_shard_is_a_usage_error(value):
    with pytest.raises(pytest.UsageError):
        parse_shard(value)


def test_a_malformed_shard_stops_the_run(suite):
    result = suite.runpytest("-p", PLUGIN, "--shard=2/1")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
