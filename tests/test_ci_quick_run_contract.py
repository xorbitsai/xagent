"""An unapproved pull request gets only the quick CI run.

`CI Summary` and `Migrations Summary` are required checks. A quick run skips
the long jobs at the job level, and GitHub reports a skipped check as passed,
so the required names must exist only on a head that had a full run: a quick
run reports `CI Quick Summary` and `Migrations Quick Summary` instead. See
docs/branch-protection.md "Quick runs before approval".
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# (workflow file, deciding job, summary job, required name, quick name,
#  jobs a quick run skips)
WORKFLOW_CASES = {
    "ci.yml": (
        "changes",
        "ci-summary",
        "CI Summary",
        "CI Quick Summary",
        {
            "prepare-deepdoc-cache",
            "pytest-fast",
            "pytest-fast-deepdoc",
            "pytest-slow",
            "pytest-milvus",
            "e2e",
            "frontend-build",
            "lancedb-memory-compatibility",
        },
    ),
    "test-migrations.yml": (
        "detect-migration-changes",
        "migrations-summary",
        "Migrations Summary",
        "Migrations Quick Summary",
        {"test-sqlite-migrations", "test-postgresql-migrations"},
    ),
}
APPROVAL_GUARD = (
    "github.event_name != 'pull_request_review'"
    " || github.event.review.state == 'approved'"
)
PR_EVENTS = ("pull_request", "pull_request_review")
OTHER_EVENTS = ("push", "merge_group", "workflow_dispatch")


def _workflow(name: str) -> dict:
    data = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    data["on"] = data.pop(True)  # YAML 1.1 parses `on:` as True
    return data


def _normalise(expression: str) -> str:
    return re.sub(r"\s+", " ", str(expression)).strip()


def _needs(job: dict) -> set[str]:
    needs = job.get("needs", [])
    return {needs} if isinstance(needs, str) else set(needs)


def _scope_step(name: str) -> dict:
    decider = WORKFLOW_CASES[name][0]
    steps = _workflow(name)["jobs"][decider]["steps"]
    # First, so no later step can fail before the output is written.
    assert steps[0]["id"] == "scope", name
    return steps[0]


def _scope(
    tmp_path: Path, event: str, gh: str, verdicts: str = "0", workflow: str = "ci.yml"
) -> tuple[str, int]:
    """Run a scope step with `gh` as its stand-in; count its review lookups.

    `verdicts` is what the lookup of earlier verdict check runs answers;
    "fail" makes it fail.
    """
    step = _scope_step(workflow)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    called = tmp_path / "called"
    answer = "exit 1" if verdicts == "fail" else f"echo {verdicts}"
    fake = bin_dir / "gh"
    fake.write_text(
        "#!/bin/bash\n"
        f'if [[ "$*" == *check-runs* ]]; then {answer}; exit; fi\n'
        f'echo call >> "{called}"\n{gh}\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    # The step waits between attempts; the test does not.
    sleep = bin_dir / "sleep"
    sleep.write_text("#!/bin/bash\n", encoding="utf-8")
    sleep.chmod(0o755)
    output = tmp_path / "output"
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        env={
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "GITHUB_EVENT_NAME": event,
            "GITHUB_OUTPUT": str(output),
            "REPO": "xorbitsai/xagent",
            "PR_NUMBER": "3853",
            "HEAD_SHA": "0123456789abcdef0123456789abcdef01234567",
            "VERDICT_CHECK": step["env"]["VERDICT_CHECK"],
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    calls = (
        len(called.read_text(encoding="utf-8").splitlines()) if called.exists() else 0
    )
    return output.read_text(encoding="utf-8").strip(), calls


def test_both_workflows_decide_the_same_way():
    ci, migrations = _scope_step("ci.yml"), _scope_step("test-migrations.yml")
    assert ci["run"] == migrations["run"]
    assert ci["env"]["VERDICT_CHECK"] == "CI Summary"
    assert migrations["env"]["VERDICT_CHECK"] == "Migrations Summary"
    assert {k: v for k, v in ci["env"].items() if k != "VERDICT_CHECK"} == {
        k: v for k, v in migrations["env"].items() if k != "VERDICT_CHECK"
    }


@pytest.mark.parametrize("event", PR_EVENTS)
def test_an_approved_pull_request_gets_the_full_run(tmp_path, event):
    assert _scope(tmp_path, event, "echo APPROVED") == ("full=true", 1)


# NONE is what the step's own filter prints when GitHub reports no decision.
@pytest.mark.parametrize("decision", ["REVIEW_REQUIRED", "CHANGES_REQUESTED", "NONE"])
def test_an_unapproved_pull_request_gets_the_quick_run(tmp_path, decision):
    assert _scope(tmp_path, "pull_request", f"echo {decision}") == ("full=false", 1)
    # An approval whose decision does not settle asks again before giving up.
    (tmp_path / "review").mkdir()
    assert _scope(tmp_path / "review", "pull_request_review", f"echo {decision}") == (
        "full=false",
        5,
    )


def test_an_approval_waits_for_the_decision_it_caused(tmp_path):
    gh = f'[ "$(wc -l < "{tmp_path}/called")" -ge 3 ] && echo APPROVED || echo REVIEW_REQUIRED'
    assert _scope(tmp_path, "pull_request_review", gh) == ("full=true", 3)


@pytest.mark.parametrize("event", PR_EVENTS)
def test_an_unreadable_review_decision_runs_the_full_ci(tmp_path, event):
    # An approved pull request must not be left without its required checks.
    assert _scope(tmp_path, event, "exit 1") == ("full=true", 1)


def test_a_second_approval_does_not_repeat_the_full_run(tmp_path):
    for workflow in WORKFLOW_CASES:
        work = tmp_path / workflow
        work.mkdir()
        assert _scope(
            work,
            "pull_request_review",
            "echo APPROVED",
            verdicts="1",
            workflow=workflow,
        ) == ("full=false", 1)
    # A push after the approval gets its full run whatever the old head had.
    (tmp_path / "push").mkdir()
    assert _scope(tmp_path / "push", "pull_request", "echo APPROVED", verdicts="1") == (
        "full=true",
        1,
    )


@pytest.mark.parametrize("verdicts", ["0", "fail", "", "x"])
def test_an_approval_without_a_known_verdict_runs_the_full_ci(tmp_path, verdicts):
    assert _scope(
        tmp_path, "pull_request_review", "echo APPROVED", verdicts=verdicts
    ) == (
        "full=true",
        1,
    )


@pytest.mark.parametrize("event", OTHER_EVENTS)
def test_merge_groups_pushes_and_manual_runs_are_always_full(tmp_path, event):
    assert _scope(tmp_path, event, "echo REVIEW_REQUIRED") == ("full=true", 0)


@pytest.mark.parametrize("name", sorted(WORKFLOW_CASES))
def test_only_an_approval_starts_a_review_run(name):
    workflow = _workflow(name)
    decider, summary, *_ = WORKFLOW_CASES[name]
    assert workflow["on"]["pull_request_review"] == {"types": ["submitted"]}
    jobs = workflow["jobs"]
    assert APPROVAL_GUARD in _normalise(jobs[decider]["if"])
    assert APPROVAL_GUARD in _normalise(jobs[summary]["if"])
    for job_name, job in jobs.items():
        if job_name in (decider, summary):
            continue
        # Anything else is skipped with the decider, or rules reviews out itself.
        assert decider in _needs(job) or APPROVAL_GUARD in _normalise(
            job.get("if", "")
        ), job_name


@pytest.mark.parametrize("name", sorted(WORKFLOW_CASES))
def test_only_a_full_run_can_report_the_required_check(name):
    decider, summary, required, quick, full_only = WORKFLOW_CASES[name]
    jobs = _workflow(name)["jobs"]
    full = f"needs.{decider}.outputs.full == 'true'"
    assert jobs[summary]["name"] == (
        "${{ (" + full + " || (github.event_name != 'pull_request'"
        " && github.event_name != 'pull_request_review'))"
        f" && '{required}' || '{quick}' }}}}"
    )
    assert jobs[decider]["outputs"]["full"] == "${{ steps.scope.outputs.full }}"
    for job_name, job in jobs.items():
        if job_name != summary:
            assert required not in str(job.get("name", "")), job_name
        condition = _normalise(job.get("if", ""))
        # Exactly the jobs a quick run skips are conditioned on `full`.
        assert (full in condition) == (job_name in full_only), job_name
        if job_name in full_only:
            assert decider in _needs(job), job_name
            assert "always()" not in condition, job_name


def test_ci_concurrency_lets_an_approval_wait_and_isolates_other_reviews():
    concurrency = _workflow("ci.yml")["concurrency"]
    assert concurrency["group"] == (
        "${{ github.workflow }}-${{ (github.event_name == 'pull_request_review'"
        " && github.event.review.state != 'approved')"
        " && format('review-{0}', github.run_id)"
        " || github.event.pull_request.number || github.ref }}"
    )
    assert concurrency["cancel-in-progress"] == (
        "${{ github.event_name != 'pull_request_review' }}"
    )


def test_an_approval_run_detects_migrations_from_the_pull_request_diff():
    step = next(
        s
        for s in _workflow("test-migrations.yml")["jobs"]["detect-migration-changes"][
            "steps"
        ]
        if s.get("id") == "changes"
    )
    assert "pull_request|pull_request_review)" in step["run"]


def _ci_summary(results: dict[str, str], full: str) -> int:
    step = next(
        s
        for s in _workflow("ci.yml")["jobs"]["ci-summary"]["steps"]
        if s.get("name") == "Check required jobs"
    )

    def value(match: re.Match) -> str:
        expression = match.group(1)
        if expression.endswith(".result"):
            job = re.search(
                r"needs(?:\['([^']+)'\]|\.([a-z0-9-]+))\.result", expression
            )
            return results[job.group(1) or job.group(2)]
        if expression == "needs.changes.outputs.full":
            return full
        return "true"

    script = re.sub(r"\$\{\{ (.+?) \}\}", value, step["run"])
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=30, check=False
    ).returncode


def _migrations_summary(results: dict[str, str], full: str) -> int:
    step = next(
        s
        for s in _workflow("test-migrations.yml")["jobs"]["migrations-summary"]["steps"]
        if s.get("name") == "Check required jobs"
    )
    env = {
        "PATH": os.environ["PATH"],
        "DETECT_RESULT": results["detect-migration-changes"],
        "SQLITE_RESULT": results["test-sqlite-migrations"],
        "POSTGRESQL_RESULT": results["test-postgresql-migrations"],
        "FULL": full,
    }
    return subprocess.run(
        ["bash", "-c", step["run"]],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    ).returncode


SUMMARIES = {"ci.yml": _ci_summary, "test-migrations.yml": _migrations_summary}


def _shapes(name: str) -> tuple[dict[str, str], dict[str, str]]:
    _, summary, _, _, full_only = WORKFLOW_CASES[name]
    gathered = _needs(_workflow(name)["jobs"][summary])
    quick = {job: ("skipped" if job in full_only else "success") for job in gathered}
    return quick, dict.fromkeys(gathered, "success")


@pytest.mark.parametrize("name", sorted(WORKFLOW_CASES))
def test_the_summary_accepts_a_quick_run_and_a_full_run(name):
    quick, full = _shapes(name)
    assert SUMMARIES[name](quick, "false") == 0
    assert SUMMARIES[name](full, "true") == 0


@pytest.mark.parametrize("name", sorted(WORKFLOW_CASES))
def test_the_summary_rejects_a_full_run_that_skipped_a_job(name):
    _, full = _shapes(name)
    for job in sorted(WORKFLOW_CASES[name][4]):
        assert SUMMARIES[name]({**full, job: "skipped"}, "true") == 1, job


@pytest.mark.parametrize("name", sorted(WORKFLOW_CASES))
def test_the_summary_rejects_a_quick_run_that_ran_a_full_job(name):
    quick, _ = _shapes(name)
    for job in sorted(WORKFLOW_CASES[name][4]):
        assert SUMMARIES[name]({**quick, job: "success"}, "false") == 1, job


@pytest.mark.parametrize("name", sorted(WORKFLOW_CASES))
@pytest.mark.parametrize("full", ["", "maybe"])
def test_the_summary_rejects_a_non_literal_full_output(name, full):
    quick, every = _shapes(name)
    assert SUMMARIES[name](quick, full) == 1
    assert SUMMARIES[name](every, full) == 1
