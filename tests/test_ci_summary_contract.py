"""Invariants that keep `CI Summary` fail-closed.

`CI Summary` is a required status check on `main`. GitHub reports a *skipped*
job to branch protection as a success, so the summary can only stay fail-closed
while two properties hold:

1. every job it gathers is also named in its ``check_job`` list, and
2. none of those jobs is ever skipped, which is why their conditions live on
   their steps rather than on the job.

Both are prose in ``docs/branch-protection.md``. This module turns them into
something that fails. See that document's "Required contexts must be summary
jobs" and "Gate at the step, not at the job" sections for the reasoning.

``ci.yml`` has a second contract test, ``frontend/src/ci/frontend-test-manifest.test.ts``.
It freezes the summary script by exact text, pins both ``jobs.changes`` filter
rule sets, and checks the six required frontend-build steps semantically --
command, working directory, shell, and an ``if:`` that is either absent or
exactly the path-filter gate. A change to either
region has to update both files or CI fails in the frontend lane.
"""

from __future__ import annotations

import copy
import re
import tomllib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT = REPO_ROOT / "pyproject.toml"

# This suite validates the `changes` outputs, so a job gated on them cannot be
# the only place it runs.
CONTRACT_TEST_PATH = "tests/test_ci_summary_contract.py"

# The no-op step that keeps a gated job reporting success instead of skipping.
# Matched by name, because it is the one step whose guard is legitimately inverted.
SKIP_SENTINEL_PREFIX = "Skip"

DRAFT_GUARD = (
    "github.event_name != 'pull_request' || github.event.pull_request.draft == false"
)

# Jobs whose steps are gated on the `changes` job, and the output each one
# reads. Everything else CI Summary gathers runs unconditionally.
GATED_JOBS = {
    "pytest-fast": "code",
    "pytest-fast-deepdoc": "code",
    "pytest-slow": "code",
    "e2e": "code",
    "frontend-build": "frontend",
}

# Widening this set means docs-only pull requests stop running some part of the
# suite. Changing it here as well as in the workflow is the point: it has to be
# a deliberate act, not a one-line edit that reads as harmless in review.
#
# `!frontend/src/**` rather than `!frontend/**` because Python tests read
# frontend/package.json and frontend/public, and paths-filter cannot re-include a
# file an earlier pattern excluded. See docs/branch-protection.md.
CODE_FILTER_EXCLUDES = frozenset(
    {"!docs/**", "!assets/**", "!*.md", "!frontend/src/**"}
)

# The wheel-build inputs frontend-build owns. A rule dropped or negated here
# stops the frontend lane running for changes that need it.
FRONTEND_FILTER_RULES = (
    "frontend/**",
    "pyproject.toml",
    # Hatchling honours .gitignore when it selects the files that go into the
    # wheel, so an edit there can silently drop one.
    ".gitignore",
    "README.md",
    # The whole package tree, not just the entrypoint: hatchling packages all
    # of src/xagent and drops any file .gitignore matches, so a backend-only
    # change can break the wheel (PR #1848 review).
    "src/xagent/**",
    ".github/workflows/ci.yml",
)

# This action decides whether the test suite runs at all, so it is pinned by
# commit. Bumping it must be deliberate -- update this constant in the same
# change.
PATHS_FILTER_PIN = "dorny/paths-filter@ceb8a2b8f2d89434be7ff52d3de7ec3738c5cc9d"

# pulls.listFiles caps its response at this many files, and paths-filter does
# not compare the rows it received against the pull request's changed_files.
LIST_FILES_MAX = 3000

# The step each gated job exists to run, and the command it must execute. Named
# rather than searched: a substring scan over every step's text lets a decoy in
# the sentinel stand in for a deleted workload. frontend-build is absent -- the
# TS contract pins its steps by exact command.
GATED_JOB_WORK_STEPS = {
    "pytest-fast": ("Run tests", "python -m pytest"),
    "pytest-fast-deepdoc": ("Run tests", "python -m pytest"),
    "pytest-slow": ("Run slow tests", "python -m pytest"),
    "e2e": ("Run e2e tests", "python -m pytest"),
}

_CACHE_HIT = "(needs.prepare-deepdoc-cache.outputs.cache-hit == 'true')"
_CACHE_MISS = "(needs.prepare-deepdoc-cache.outputs.cache-hit != 'true')"

# Every step whose gate may carry more than the `changes` output, and the exact
# extra it carries. Keyed by (job, step), not allowlisted globally: each of these
# predicates is false on some reachable run, so one that is right on a setup step
# silently skips a workload step (PR #1848 review).
STEP_GUARD_EXTRAS = {
    ("pytest-fast", "Pre-pull the sandbox image"): "(matrix.name == 'web')",
    ("pytest-fast-deepdoc", "Pre-pull the sandbox image"): "(matrix.name == 'core')",
    ("pytest-fast-deepdoc", "Restore Deepdoc cache"): _CACHE_HIT,
    ("pytest-fast-deepdoc", "Download Deepdoc cache artifact"): _CACHE_MISS,
    ("pytest-slow", "Restore Deepdoc cache"): _CACHE_HIT,
    ("pytest-slow", "Download Deepdoc cache artifact"): _CACHE_MISS,
}

# Applied to the value above, not to the workflow: renaming a leg in `strategy`
# alone leaves a guard that is false on every leg.
MATRIX_CONJUNCT = re.compile(r"^\(matrix\.name == '([a-z0-9-]+)'\)$")


class _NoDuplicateKeyLoader(yaml.SafeLoader):
    """A loader that rejects duplicate mapping keys.

    PyYAML accepts them and keeps the last one, which silently drops a step's
    guard when an edit adds a second ``if:`` to the same step. GitHub rejects
    the workflow outright, so the permissive parse is the wrong default here.
    """


def _no_duplicates(loader: yaml.Loader, node: yaml.MappingNode, deep: bool = False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise AssertionError(
                f"duplicate key {key!r} at line {key_node.start_mark.line + 1}"
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_NoDuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates
)


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.load(
        CI_WORKFLOW.read_text(encoding="utf-8"), Loader=_NoDuplicateKeyLoader
    )


@pytest.fixture(scope="module")
def jobs(workflow: dict) -> dict:
    return workflow["jobs"]


@pytest.fixture(scope="module")
def summary(jobs: dict) -> dict:
    return jobs["ci-summary"]


def _filter_step(jobs: dict) -> dict:
    return next(step for step in jobs["changes"]["steps"] if step.get("id") == "filter")


def _matrix_leg_names(job: dict) -> list[str]:
    include = (job.get("strategy") or {}).get("matrix", {}).get("include") or []
    return [leg["name"] for leg in include if "name" in leg]


def _normalise(expression: str) -> str:
    return re.sub(r"\s+", " ", expression).strip()


def test_summary_gathers_every_job_it_checks(jobs: dict, summary: dict) -> None:
    """`needs` and the `check_job` calls must name the same set of jobs.

    A job missing from either list is advisory only: it can fail without
    failing the required check.
    """
    script = summary["steps"][0]["run"]
    checked = set(re.findall(r'check_job "([^"]+)"', script))
    gathered = set(summary["needs"])

    assert checked == gathered, (
        "ci-summary's needs and its check_job list have drifted apart; "
        f"only in needs: {sorted(gathered - checked)}, "
        f"only in check_job: {sorted(checked - gathered)}"
    )

    # Nothing may be left out of the summary altogether.
    everything_else = set(jobs) - {"ci-summary"}
    assert gathered == everything_else, (
        "every job in ci.yml must be gathered by ci-summary; "
        f"missing: {sorted(everything_else - gathered)}"
    )


def test_gathered_jobs_are_never_skipped(summary: dict, jobs: dict) -> None:
    """No gathered job may carry a condition beyond the draft guard.

    Any other job-level `if:` makes the job skippable, and a skipped required
    job reports success to branch protection. Conditions belong on the steps.
    """
    for name in summary["needs"]:
        condition = _normalise(jobs[name].get("if", DRAFT_GUARD))
        assert condition == _normalise(DRAFT_GUARD), (
            f"job '{name}' carries a job-level condition: {condition!r}. "
            "A gathered job that can be skipped reports success to branch "
            "protection -- gate its steps instead. See "
            "docs/branch-protection.md 'Gate at the step, not at the job'."
        )


def test_summary_rejects_a_non_literal_changes_output(summary: dict) -> None:
    """An empty output leaves only the sentinel running, and the job goes green.

    `check_job` cannot see that, so the summary has to reject the value itself.
    """
    script = summary["steps"][0]["run"]
    checked = set(re.findall(r'check_flag "([^"]+)"', script))
    assert checked == set(GATED_JOBS.values()), (
        "every output the gated jobs read must be asserted to be a literal "
        f"true/false; missing: {sorted(set(GATED_JOBS.values()) - checked)}"
    )


@pytest.mark.parametrize("job_name", sorted(GATED_JOBS))
def test_every_step_of_a_gated_job_is_guarded(jobs: dict, job_name: str) -> None:
    """Guards must have the right polarity, not merely name the output.

    Referencing the output is not enough: flipping a work step to ``!= 'true'``
    skips the real tests on exactly the pull requests that need them while the
    job, and so the summary, stays green.
    """
    output = GATED_JOBS[job_name]
    run_guard = f"needs.changes.outputs.{output} == 'true'"
    skip_guard = f"needs.changes.outputs.{output} != 'true'"

    sentinels = []
    for step in jobs[job_name]["steps"]:
        label = step.get("name") or step.get("uses") or "<unnamed>"
        condition = _normalise(step.get("if") or "")
        assert condition, (
            f"step '{label}' in job '{job_name}' has no condition; every step "
            f"of a gated job must be guarded on {run_guard}"
        )

        if label.startswith(SKIP_SENTINEL_PREFIX):
            sentinels.append(label)
            assert condition == skip_guard, (
                f"sentinel '{label}' in job '{job_name}' is conditional on "
                f"{condition!r}; it must be exactly {skip_guard!r} so the job "
                "reports success on a change it does not apply to"
            )
            continue

        # The gate has to lead and cannot be OR-ed away.
        assert condition == run_guard or condition.startswith(f"{run_guard} && "), (
            f"step '{label}' in job '{job_name}' is conditional on "
            f"{condition!r}; a work step must start with {run_guard!r} and may "
            "only add '&&' conjunctions"
        )

        remainder = condition[len(run_guard) :].removeprefix(" && ")
        expected = STEP_GUARD_EXTRAS.get((job_name, label), "")
        assert remainder == expected, (
            f"step '{label}' in job '{job_name}' is gated on "
            f"{run_guard!r} && {remainder!r}, but the only extra this step may "
            f"carry is {expected!r}. Every permitted predicate is false on some "
            "reachable run, so one on the wrong step skips it there while the "
            "job still reports success."
        )

        leg = MATRIX_CONJUNCT.match(expected)
        if leg:
            declared = _matrix_leg_names(jobs[job_name])
            assert leg.group(1) in declared, (
                f"step '{label}' in job '{job_name}' narrows itself to matrix "
                f"leg {leg.group(1)!r}, but the job declares "
                f"{declared or 'no matrix at all'}. The predicate is false on "
                "every leg, so the step never runs while the job still reports "
                "success."
            )

    assert len(sentinels) == 1, (
        f"job '{job_name}' must have exactly one '{SKIP_SENTINEL_PREFIX}...' "
        f"sentinel step so it never reports an empty run; found {sentinels}"
    )


# Words that open or close a compound statement, and their effect on nesting.
# `if false; then <workload>; fi` is not proof that the workload runs.
_BLOCK_WORDS = {
    "if": 1,
    "while": 1,
    "until": 1,
    "for": 1,
    "case": 1,
    "select": 1,
    "{": 1,
    "fi": -1,
    "done": -1,
    "esac": -1,
    "}": -1,
    "then": 0,
    "else": 0,
    "elif": 0,
    "do": 0,
    "in": 0,
    "(": 0,
    ")": 0,
}

_SEPARATOR = re.compile(r"\|\||&&|[;\n|&]")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# The one wrapper a workload is legitimately launched through, so the check
# cannot be satisfied by another command that merely mentions pytest.
WORKLOAD_RUNNER_PREFIXES = ("python3 -m uv run ",)


def _top_level_commands(script: str) -> list[str]:
    """The commands a `run:` script reaches unconditionally, in command position.

    Substring membership is not execution: `echo python -m pytest`,
    `if false; then python -m pytest; fi`, `true || python -m pytest` and a
    function body that is never called all contain the workload and none of them
    runs it. So quoted literals and comments go first, then a command counts only
    at nesting depth zero and only from the start of a command.

    This is a command-shape contract, not a shell parser: it models nesting,
    command position and `||`, and deliberately accepts the right-hand side of
    `&&`, which the leading `set -e` already makes load-bearing in these steps.
    """
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "", script)
    text = re.sub(r"#[^\n]*", "", text)
    text = re.sub(r"\\\n", " ", text)

    commands: list[str] = []
    depth = 0
    preceded_by = ""
    cursor = 0
    for match in [*_SEPARATOR.finditer(text), None]:
        segment = text[cursor : match.start()] if match else text[cursor:]
        words = segment.split()
        while words and words[0] in _BLOCK_WORDS:
            depth = max(depth + _BLOCK_WORDS[words.pop(0)], 0)
        if words and words[-1] == "{":
            # `run_tests() { ... }` defines the workload, it does not call it.
            depth += 1
            words.pop()
        if words and depth == 0 and preceded_by != "||":
            commands.append(" ".join(words))
        if match:
            preceded_by = match.group(0)
            cursor = match.end()
    return commands


def _invokes(segment: str, command: str) -> bool:
    words = segment.split()
    while words and _ASSIGNMENT.match(words[0]):
        words.pop(0)
    text = " ".join(words)
    for prefix in WORKLOAD_RUNNER_PREFIXES:
        text = text.removeprefix(prefix)
    return text == command or text.startswith(f"{command} ")


@pytest.mark.parametrize("job_name", sorted(GATED_JOB_WORK_STEPS))
def test_a_gated_job_still_runs_its_test_command(jobs: dict, job_name: str) -> None:
    """A correctly guarded job that no longer runs anything is still green."""
    step_name, command = GATED_JOB_WORK_STEPS[job_name]
    matching = [
        step for step in jobs[job_name]["steps"] if step.get("name") == step_name
    ]

    assert len(matching) == 1, (
        f"job '{job_name}' must have exactly one '{step_name}' step, the one it "
        f"exists to run; found {len(matching)}"
    )
    commands = _top_level_commands(matching[0].get("run") or "")
    assert any(_invokes(c, command) for c in commands), (
        f"step '{step_name}' in job '{job_name}' never reaches {command!r} in "
        f"command position; the commands it does run are {commands}. Every step "
        "of a gated job is skippable by design, so without this the job reports "
        "success having executed no tests at all."
    )


@pytest.mark.parametrize("job_name", sorted(GATED_JOBS))
def test_gated_jobs_depend_on_changes(jobs: dict, job_name: str) -> None:
    needs = jobs[job_name]["needs"]
    needs = [needs] if isinstance(needs, str) else needs
    assert "changes" in needs, (
        f"job '{job_name}' reads needs.changes.outputs but does not list "
        "'changes' in its needs"
    )


def test_changes_job_can_read_the_pull_request(jobs: dict) -> None:
    """paths-filter uses the pulls.listFiles API whenever a token is present.

    Without this scope the call is a 403 and every pull request fails here.
    """
    permissions = jobs["changes"]["permissions"]
    assert permissions.get("pull-requests") == "read"
    assert permissions.get("contents") == "read"


def test_paths_filter_is_pinned_by_commit(jobs: dict) -> None:
    """The pin has to bind to the step that produces the outputs.

    Looking for the SHA anywhere in the job would also accept a dead pinned
    step sitting beside a real gate that uses a mutable tag.
    """
    assert _filter_step(jobs)["uses"] == PATHS_FILTER_PIN, (
        f"the 'filter' step must use exactly {PATHS_FILTER_PIN}, found "
        f"{_filter_step(jobs)['uses']!r}. Bumping the action is a supply-chain "
        "decision: it gates whether the test suite runs at all."
    )

    invocations = [
        step["uses"]
        for step in jobs["changes"]["steps"]
        if "dorny/paths-filter" in step.get("uses", "")
    ]
    assert invocations == [PATHS_FILTER_PIN], (
        "the changes job must invoke paths-filter exactly once, pinned; found "
        f"{invocations}"
    )


def test_code_filter_exclusions_are_exactly_as_declared(jobs: dict) -> None:
    step = _filter_step(jobs)
    filters = yaml.safe_load(step["with"]["filters"])

    excludes = {p for p in filters["code"] if p.startswith("!")}
    assert excludes == CODE_FILTER_EXCLUDES, (
        "the set of paths treated as 'not code' changed; anything listed here "
        "stops triggering the test suite on its own. "
        f"added: {sorted(excludes - CODE_FILTER_EXCLUDES)}, "
        f"removed: {sorted(CODE_FILTER_EXCLUDES - excludes)}"
    )

    # The include side has to stay open: everything is code until excluded, so
    # a newly added top-level directory runs the suite rather than skipping it.
    assert "**" in filters["code"]

    # Without this quantifier the exclusions above do not mean what they read
    # as -- the default `some` does not combine includes with excludes.
    assert step["with"]["predicate-quantifier"] == "some-with-excludes"


def test_frontend_filter_rules_are_exactly_as_declared(jobs: dict) -> None:
    """Both suites pin these rules, which is overlap rather than duplication.

    This suite's load-bearing run is the ungated `pre-commit` job, so no filter
    edit can skip it; the frontend suite is still `frontend`-gated. Dropping or
    negating a rule has to get past both.
    """
    filters = yaml.safe_load(_filter_step(jobs)["with"]["filters"])

    assert tuple(filters["frontend"]) == FRONTEND_FILTER_RULES, (
        "the frontend filter rules changed; anything removed here stops the "
        f"frontend lane running for that path. expected "
        f"{list(FRONTEND_FILTER_RULES)}, found {filters['frontend']}"
    )

    negated = [rule for rule in filters["frontend"] if rule.startswith("!")]
    assert not negated, (
        f"the frontend filter must not exclude anything; found {negated}. An "
        "exclusion cannot be undone by a later rule, so it silently narrows the "
        "lane."
    )


def test_non_pull_request_events_never_filter(jobs: dict) -> None:
    """The merge queue must behave exactly as it did before this job existed."""
    for output in GATED_JOBS.values():
        expression = jobs["changes"]["outputs"][output]
        assert "github.event_name != 'pull_request' ||" in expression, (
            f"output '{output}' must short-circuit to true on non-pull_request "
            "events so merge_group, push and workflow_dispatch run everything"
        )
        assert f"steps.filter.outputs.{output} != 'false'" in expression, (
            f"output '{output}' must use != 'false' so a missing filter output "
            "runs everything instead of silently skipping"
        )


def test_a_truncated_file_list_runs_everything(jobs: dict) -> None:
    """paths-filter cannot see past the 3000th file, and does not notice.

    It paginates pulls.listFiles without comparing the rows it got back to
    changed_files, so a larger pull request whose visible portion is all
    excluded paths yields a genuine `false` with code after the cutoff.
    """
    for output in sorted(set(GATED_JOBS.values())):
        expression = _normalise(jobs["changes"]["outputs"][output])
        assert (
            f"github.event.pull_request.changed_files > {LIST_FILES_MAX}" in expression
        ), (
            f"output '{output}' must fall back to true once the pull request "
            f"exceeds {LIST_FILES_MAX} files, which truncates the file list "
            f"paths-filter reads; got {expression!r}"
        )
        assert "github.event.pull_request.changed_files == null" in expression, (
            f"output '{output}' must also fall back to true when the file count "
            "is unavailable -- an absent count compares as 0 and would pass the "
            "size check silently"
        )


def _expected_output_expression(output: str) -> str:
    return (
        "${{ github.event_name != 'pull_request'"
        " || github.event.pull_request.changed_files == null"
        f" || github.event.pull_request.changed_files > {LIST_FILES_MAX}"
        f" || steps.filter.outputs.{output} != 'false' }}}}"
    )


def test_the_changes_outputs_are_exactly_as_declared(jobs: dict) -> None:
    """The two tests above assert substrings, which precedence can defeat.

    `&&` binds tighter than `||`, so a chain rewritten to use it keeps every
    substring they look for while forcing the output to false on an ordinary
    pull request -- skipping the whole suite.
    """
    for output in sorted(set(GATED_JOBS.values())):
        expression = _normalise(jobs["changes"]["outputs"][output])
        assert expression == _expected_output_expression(output), (
            f"the '{output}' output expression changed. Each clause is a "
            "fallback to running everything, and they have to stay a flat "
            f"'||' chain. expected {_expected_output_expression(output)!r}, "
            f"found {expression!r}"
        )


def test_frontend_filter_covers_the_readme_the_wheel_needs(jobs: dict) -> None:
    """frontend-build is the only job that builds a wheel.

    pyproject points `readme` at a root markdown file, which `code` excludes, so
    unless the frontend filter names it too, renaming it skips that build.
    """
    readme = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["readme"]
    filters = yaml.safe_load(_filter_step(jobs)["with"]["filters"])

    assert readme in filters["frontend"], (
        f"pyproject declares readme = {readme!r}, so it is a wheel build input, "
        f"but the frontend filter is {filters['frontend']}. A pull request "
        "touching only that file would skip 'Verify package bundles the "
        "frontend', the one step that builds the wheel."
    )


def _repo_path_for_wheel_member(member: str, packages: list[str]) -> str | None:
    """Where a wheel member lives in the repository, per hatch's `packages`."""
    head, _, rest = member.partition("/")
    for package in packages:
        if package.rsplit("/", 1)[-1] == head:
            return f"{package}/{rest}" if rest else package
    return None


def _covered_by_filter(path: str, rules: list[str]) -> bool:
    return any(
        path.startswith(rule[:-2]) if rule.endswith("/**") else path == rule
        for rule in rules
    )


def test_wheel_members_are_covered_by_the_frontend_filter(jobs: dict) -> None:
    """The grep targets and the filter rules are literals in two namespaces.

    The archive holds `xagent/...` while the filter names `src/xagent/...`, so
    nothing textual links them and either side can be repointed alone. Hatch's
    `packages` is the mapping, which makes the relationship assertable.
    """
    step = _step(jobs, "frontend-build", "Verify package bundles the frontend")
    members = re.findall(r'grep -q "([^"]+)"', step["run"])
    assert members, (
        "the wheel step no longer asserts any package member is in the archive"
    )

    wheel = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["hatch"][
        "build"
    ]["targets"]["wheel"]
    rules = yaml.safe_load(_filter_step(jobs)["with"]["filters"])["frontend"]

    for member in members:
        path = _repo_path_for_wheel_member(member, wheel["packages"])
        assert path is not None, (
            f"the wheel step asserts {member!r}, which is under none of hatch's "
            f"packages {wheel['packages']}. Either the grep target or the "
            "packaging config was repointed without the other."
        )
        assert _covered_by_filter(path, rules), (
            f"the wheel step asserts {member!r}, which lives at {path!r}, but "
            f"the frontend filter {rules} does not cover it. A change to that "
            "file would skip the only job that inspects a built wheel."
        )


def _mutated(jobs: dict) -> dict:
    return copy.deepcopy(jobs)


def _step(jobs: dict, job_name: str, step_name: str) -> dict:
    return next(
        step for step in jobs[job_name]["steps"] if step.get("name") == step_name
    )


def test_a_work_guard_widened_with_an_event_check_is_rejected(jobs: dict) -> None:
    mutated = _mutated(jobs)
    _step(mutated, "pytest-fast", "Run tests")["if"] = (
        "needs.changes.outputs.code == 'true' && github.event_name == 'push'"
    )

    with pytest.raises(AssertionError):
        test_every_step_of_a_gated_job_is_guarded(mutated, "pytest-fast")


def test_a_gated_job_stripped_to_its_sentinel_is_rejected(jobs: dict) -> None:
    mutated = _mutated(jobs)
    mutated["pytest-fast"]["steps"] = [
        step
        for step in mutated["pytest-fast"]["steps"]
        if (step.get("name") or "").startswith(SKIP_SENTINEL_PREFIX)
    ]

    with pytest.raises(AssertionError):
        test_a_gated_job_still_runs_its_test_command(mutated, "pytest-fast")


def test_an_and_chained_output_expression_is_rejected(jobs: dict) -> None:
    """`&&` binds tighter than `||`, so this skips CI on every ordinary PR."""
    mutated = _mutated(jobs)
    mutated["changes"]["outputs"]["code"] = (
        _normalise(mutated["changes"]["outputs"]["code"])
        .replace("== null ||", "== null &&")
        .replace("> 3000 ||", "> 3000 &&")
    )

    test_non_pull_request_events_never_filter(mutated)
    test_a_truncated_file_list_runs_everything(mutated)

    with pytest.raises(AssertionError):
        test_the_changes_outputs_are_exactly_as_declared(mutated)


def test_the_contract_runs_in_a_job_the_filter_cannot_gate(
    jobs: dict, summary: dict
) -> None:
    """This suite is the thing that catches a neutered `changes` job.

    Behind a gate it validates nothing: an output pinned to a literal `false`
    skips the jobs, skips this suite with them, and `check_flag` accepts
    `false` by design, so the summary stays green having run no tests at all.
    """
    ungated = [name for name in summary["needs"] if name not in GATED_JOBS]
    running = [
        name
        for name in ungated
        for step in jobs[name]["steps"]
        if CONTRACT_TEST_PATH in (step.get("run") or "") and not step.get("if")
    ]

    assert running, (
        f"no unconditional step in any of {sorted(ungated)} runs "
        f"{CONTRACT_TEST_PATH}. Every job that does run it is gated on the "
        "very outputs it exists to validate."
    )


def test_a_workload_replaced_with_an_echo_decoy_is_rejected(jobs: dict) -> None:
    mutated = _mutated(jobs)
    _step(mutated, "pytest-fast", "Run tests")["run"] = "echo 'python -m pytest'\n"

    with pytest.raises(AssertionError):
        test_a_gated_job_still_runs_its_test_command(mutated, "pytest-fast")


def test_a_matrix_leg_renamed_out_from_under_a_guard_is_rejected(
    jobs: dict,
) -> None:
    """The rename is on the `strategy` side; the guard still names the old leg."""
    mutated = _mutated(jobs)
    for leg in mutated["pytest-fast"]["strategy"]["matrix"]["include"]:
        if leg["name"] == "web":
            leg["name"] = "webapp"

    with pytest.raises(AssertionError):
        test_every_step_of_a_gated_job_is_guarded(mutated, "pytest-fast")


def test_a_matrix_guard_on_a_job_without_a_matrix_is_rejected(jobs: dict) -> None:
    mutated = _mutated(jobs)
    _step(mutated, "e2e", "Run e2e tests")["if"] = (
        "needs.changes.outputs.code == 'true' && (matrix.name == 'web')"
    )

    with pytest.raises(AssertionError):
        test_every_step_of_a_gated_job_is_guarded(mutated, "e2e")


def test_a_wheel_member_outside_the_frontend_filter_is_rejected(jobs: dict) -> None:
    mutated = _mutated(jobs)
    step = _step(mutated, "frontend-build", "Verify package bundles the frontend")
    step["run"] = step["run"].replace(
        "xagent/web/__main__.py", "somewhere/else/__main__.py"
    )

    with pytest.raises(AssertionError):
        test_wheel_members_are_covered_by_the_frontend_filter(mutated)


def test_an_unquoted_echo_decoy_is_rejected(jobs: dict) -> None:
    mutated = _mutated(jobs)
    _step(mutated, "pytest-fast", "Run tests")["run"] = (
        "set -e\necho python -m pytest tests\n"
    )

    with pytest.raises(AssertionError):
        test_a_gated_job_still_runs_its_test_command(mutated, "pytest-fast")


def test_a_workload_in_an_unreachable_branch_is_rejected(jobs: dict) -> None:
    mutated = _mutated(jobs)
    _step(mutated, "pytest-fast", "Run tests")["run"] = (
        "set -e\nif false; then python3 -m uv run python -m pytest; fi\n"
    )

    with pytest.raises(AssertionError):
        test_a_gated_job_still_runs_its_test_command(mutated, "pytest-fast")


def test_a_declared_matrix_leg_is_rejected_on_a_workload_step(jobs: dict) -> None:
    mutated = _mutated(jobs)
    _step(mutated, "pytest-fast", "Run tests")["if"] = (
        "needs.changes.outputs.code == 'true' && (matrix.name == 'web')"
    )

    with pytest.raises(AssertionError):
        test_every_step_of_a_gated_job_is_guarded(mutated, "pytest-fast")


def test_a_cache_predicate_is_rejected_on_a_workload_step(jobs: dict) -> None:
    mutated = _mutated(jobs)
    _step(mutated, "pytest-slow", "Run slow tests")["if"] = (
        "needs.changes.outputs.code == 'true' && "
        "(needs.prepare-deepdoc-cache.outputs.cache-hit != 'true')"
    )

    with pytest.raises(AssertionError):
        test_every_step_of_a_gated_job_is_guarded(mutated, "pytest-slow")


def test_a_pre_pull_leg_swapped_for_another_declared_leg_is_rejected(
    jobs: dict,
) -> None:
    mutated = _mutated(jobs)
    _step(mutated, "pytest-fast", "Pre-pull the sandbox image")["if"] = (
        "needs.changes.outputs.code == 'true' && (matrix.name == 'integration')"
    )

    with pytest.raises(AssertionError):
        test_every_step_of_a_gated_job_is_guarded(mutated, "pytest-fast")
