# Branch protection and the merge queue

How `main` is protected, why it is set up this way, and what to do when it
blocks something urgent.

> **Status: the merge queue is live.** `required_status_checks.strict` is off, so
> PRs no longer have to sync `main` by hand. Merging is now "Merge when ready",
> which enqueues the PR; the queue validates it against the latest `main` and
> merges it for you.
>
> **The push restriction on `main` was removed to make this possible**, so
> merging is no longer limited to administrators, and `block_creations` went
> with it. See "The cost: merging is no longer admin-only" below -- that
> trade-off is the most important thing on this page.
>
> The required contexts are `Migrations Summary` and `CI Summary`. Both are
> summary jobs rather than individual job names, for a reason worth reading:
> see "Required contexts must be summary jobs" below.
>
> Still pending: `enforce_admins`.

## What is being protected against

Two migrations branched off the same Alembic revision will each declare the same
`down_revision`. Individually both are valid; merged together they leave `main`
with two heads, and `alembic upgrade head` can no longer resolve a single
target. This has happened repeatedly -- the repository carries 13 hand-written
`*_merge_*.py` revisions and a trail of `fix(migrations): merge alembic heads`
commits -- and each occurrence is a fire drill that blocks everyone.

Alembic does not prevent any of this. A branched graph is legal to Alembic, and
`ScriptDirectory.get_heads()` returns every head with no warning and no error.
The single-head invariant is a policy this repository enforces, not something
the tool gives us.

Note that git cannot catch it either: two migrations are two different files, so
the merge is clean. Only the graph check sees the problem.

## How it is enforced

| Layer | What it catches | Where |
| --- | --- | --- |
| `alembic-check` pre-commit hook | multiple heads, duplicate revision IDs, unresolvable `down_revision` | locally on commit, and on every PR via the `pre-commit` job |
| Merge queue | the same defects, evaluated against the tree that will actually land | at merge time |
| `Test SQLite/PostgreSQL Migrations` | a real `alembic upgrade head` on both backends | reported via the required `Migrations Summary` context |
| `ci.yml` on `push: main` | post-merge detection if something slipped through | red build on `main`, emailed to the pusher |

`scripts/check_alembic_heads.sh` is the single implementation of the graph
check. It became the single implementation when
`.github/scripts/refresh_migration_prs.py` was removed: that script existed to
merge `main` into open migration PRs while `strict` was on, and along the way it
hand-parsed `down_revision` out of the source with `ast` and re-derived the head
graph without going through Alembic. A second implementation that can disagree
with the authoritative one, while posting a status that looks just as official,
is not extra protection -- it is a second answer that can quietly contradict the
real one.

Its regression coverage lives in
`tests/migrations/test_check_alembic_heads.py`, which injects each defect into a
throwaway Alembic environment and asserts the script rejects it. Those negative
tests exist because a check that cannot fail is worse than no check: it reads as
protection that is not there.

The conflict-detection chain was verified by reconstructing the exact tree the
queue builds: two branches each adding a migration off the same head are
individually valid and merge without git conflict, but the combined tree fails
both `alembic-check` ("expected exactly 1 head, found 2") and `alembic upgrade
head` (`CommandError: Multiple head revisions are present`), and
`detect-migration-changes` reports `should-test=true` for it.

## Why a merge queue instead of "require branches to be up to date"

`required_status_checks.strict` is a per-branch boolean -- it cannot be scoped to
PRs that touch migrations. Turning it on therefore serialized *every* PR: after
each merge, every other open PR went stale and had to re-sync `main` by hand.

The merge queue reaches the same guarantee without that cost. For each entry it
builds `gh-readonly-queue/main/pr-N-<sha>` containing `main` plus the entries
ahead of it plus that PR, and runs the required checks on that branch. Two
conflicting migrations produce two heads there, so the second entry fails and is
ejected while the first merges. This is evaluated at merge time against the exact
tree that will land, not as a point-in-time snapshot that can go stale.

It also catches defects the graph check cannot -- two migrations that each add
the same column produce a valid single-head graph that still fails on upgrade.

### Queue configuration

```
merge_method:                      SQUASH   # the only method enabled on this repo
max_entries_to_build:              5        # speculative builds; this is what keeps throughput up
max_entries_to_merge:              1        # merge one PR at a time
min_entries_to_merge:              1
min_entries_to_merge_wait_minutes: 5        # inactive while min_entries_to_merge is 1
grouping_strategy:                 ALLGREEN
check_response_timeout_minutes:    60       # worst observed ci.yml run is 29.4 min
```

`max_entries_to_merge: 1` means every commit on `main` was validated on its own,
so `git bisect` and reverts stay meaningful. It does not slow the queue down:
throughput comes from `max_entries_to_build`, which builds later entries while
earlier ones are still being checked.

`grouping_strategy` has no effect while `max_entries_to_merge` is 1. It is set to
`ALLGREEN` so that raising the batch size later cannot silently downgrade to the
weaker "only the group head must pass" semantics.

## The cost: merging is no longer admin-only

**A merge queue and "Restrict who can push to matching branches" cannot both be
enabled.** The queue merges by pushing to `main`, and it does so as
`github-merge-queue[bot]`, which is not a repository admin. With a push
restriction in place that push is rejected, and the entry is ejected with
`branch_protection_failure` -- *after* every required check has gone green, which
makes it look like a mystery.

The bot cannot be added to the allowlist: `github-merge-queue` is an internal
GitHub service, not an installable GitHub App, so it cannot be selected as a
user, team, or app. GitHub's own answer to this is that using a merge queue
means granting push access to the branch.

Removing the restriction also silently disabled `block_creations`, which depends
on it. That one is close to harmless here -- it only blocked *creating* a branch
named `main`, which cannot happen while `main` exists and `allow_deletions` is
false -- but it is gone, and the API reports it as `false` even if a request
tries to set it `true`.

### What actually changed, precisely

The removed `restrictions` was an **empty allowlist** (`users: []`, `teams: []`,
`apps: []`). Combined with `enforce_admins: false`, that meant only
administrators could merge anything into `main` -- which is why, before the
rollout, every merge on the branch was performed by an admin.

| | before | after |
| --- | --- | --- |
| direct push by a write collaborator | rejected | rejected |
| direct push by an administrator | allowed (`enforce_admins: false`) | allowed (unchanged) |
| merging an approved PR, write collaborator | rejected -- not on the allowlist | allowed |
| merging an approved PR, administrator | allowed | allowed |

**Direct pushes by ordinary collaborators were not what `restrictions` was
holding back, and they are still blocked.** That gate is
`required_pull_request_reviews`: with it enabled, a write collaborator can only
change `main` through a pull request carrying the required approval.
`restrictions` was a second, narrower gate layered on top, and what it gated was
*who may perform the merge*.

So the change is that merging widened from administrators to all write
collaborators -- still by way of an approved PR, resolved conversations, and the
required checks. That is the intended effect, not a side effect: self-service
merging is the point of the queue.

What still holds `main`:

| | |
| --- | --- |
| `required_pull_request_reviews` | no change to `main` outside a PR; 1 approval, conversations resolved |
| required status checks | `Migrations Summary`, `CI Summary` |
| merge queue | validates the combined tree before it lands |
| `allow_force_pushes: false` | history cannot be rewritten |
| `allow_deletions: false` | `main` cannot be deleted |

The one bypass left is an administrator, because `enforce_admins` is `false` --
and that was equally true before the rollout. Closing it is a matter of enabling
`enforce_admins`; it does not require giving up the queue.

If narrowing merge permission back to a named set ever becomes necessary, the
lever is `restrictions`, and restoring it means giving up the queue. `strict` is
not that lever -- it governs whether a PR must be up to date, not who may merge
or whether a PR is required at all.

## Required contexts must be summary jobs

**GitHub reports a skipped job to branch protection as a success.** Successful
check statuses are `success`, `skipped` and `neutral`. GitHub's
"Troubleshooting required status checks" spells out both ways a job gets there:
skipped by a conditional -- "the job reports 'Success'"; skipped because a job in
its `needs` failed -- "the dependent job is skipped and may not block merging",
with the documented fix being to use `always()`.

So requiring an individual job name is not fail-closed. If whatever the
migration jobs depend on fails, they are skipped, their contexts go green, and
the merge lands with no migration test having run. Only a workflow that never
triggers at all stays pending and blocks -- which is a different case, and the
reason a missing `merge_group:` trigger stalls the queue instead of waving it
through.

The required contexts are therefore the two summary jobs:

| context | job | covers |
| --- | --- | --- |
| `Migrations Summary` | `test-migrations.yml` | `detect-migration-changes`, `Test SQLite Migrations`, `Test PostgreSQL Migrations` |
| `CI Summary` | `ci.yml` | `changes`, `pre-commit`, `pytest-fast`, `pytest-fast-deepdoc`, `pytest-slow`, `e2e`, `frontend-build`, `prepare-deepdoc-cache` |

Both declare `if: always()` and fail unless every job they gather reports
`success` -- a gathered job that is `skipped`, `failure` or `cancelled` fails
the summary. **`always()` is load-bearing.** Removing it looks harmless and
silently restores the hole, which is why it carries a comment in the workflow
saying so.

### Gate at the step, not at the job

Failing on `skipped` only works because **no gathered job is ever skipped**.
`Test SQLite Migrations` and `Test PostgreSQL Migrations` carry no job-level
`if:` at all. When `detect-migration-changes` reports `should-test=false`, the
jobs still run; it is their *steps* that are gated, and the job reports
`success` in a few seconds. A PR that touches no migration still gets a green
`Migrations Summary` -- this page's own PR is an example.

That is a deliberate property and an easy one to destroy. Moving the condition
up to the job level looks like an optimisation -- same outcome, one less job
started -- but it turns every non-migration PR into a skipped required job,
which reports success, which is the exact hole the summary exists to close.
**If a job must be conditional, gate its steps.**

Adding a job to either workflow does not automatically gate on it. It has to be
added to that summary's `needs` and to its `check_job` list, or it is advisory
only.

`ci.yml` follows the same shape. Its `changes` job reports which halves of the
tree a pull request touches, and `pytest-fast`, `pytest-fast-deepdoc`,
`pytest-slow`, `e2e` and `frontend-build` gate every one of their *steps* on
that output. On a docs-only pull request those five jobs still start and still
report `success`, their own steps finishing in about twenty seconds.
`prepare-deepdoc-cache` is not gated, so on a cache miss `pytest-fast-deepdoc`
and `pytest-slow` wait on it first. Non-`pull_request` events --
`merge_group`, `push` to `main`, `workflow_dispatch` -- never filter at all, so
the queue re-runs the full suite against the tree that will actually land and
stays the backstop for anything the filters get wrong.

Step-level gating has a failure mode of its own that job-level gating does not: if the `changes` outputs come back empty, every guarded work step skips, only the `Skip` sentinel runs, and the job reports `success` having tested nothing. A skipped job at least shows up grey in the UI; this one is green, and `check_job` cannot tell the difference. `CI Summary` therefore also asserts that each output is a literal `true` or `false` and fails on an empty one -- that is the only way this summary could otherwise pass vacuously, so it is checked explicitly rather than trusted to the expression that produces it.

The detector has a blind spot in the same family. `paths-filter` reads the changed file list from `pulls.listFiles`, which returns at most 3000 files, and it never compares the rows it received against the pull request's own `changed_files` count -- so past that cutoff it can report a perfectly genuine `false` while code sits in the part it could not see. Both outputs therefore fall back to "run everything" once `changed_files` exceeds 3000, or if that count is missing entirely; an absent count compares as `0`, so the `== null` clause is doing real work rather than restating the size check.

The `frontend` filter names `.gitignore` because hatchling honours it when selecting the files that go into the wheel -- this repository's own `artifacts` override for `frontend_dist` exists precisely because gitignored paths are otherwise dropped -- and `frontend-build` runs the only step that inspects a built wheel. It names `README.md` for a related reason: `pyproject.toml` declares it as the project `readme`, root markdown is excluded from `code`, and `frontend-build` owns the only `python -m build --wheel` in the workflow. Without that rule a pull request that renames or deletes the README skips the one job that would notice hatchling can no longer resolve it. It names `src/xagent/**` rather than the entrypoint alone for the same reason read the other way: `pyproject.toml` packages the whole tree, and hatchling drops any file `.gitignore` matches -- without consulting git's index, so a tracked file force-added under a pattern like `data/` or `default.yaml` is simply absent from the wheel. That is a change no rule above would catch, because it touches only backend sources. Anything narrower here reopens the gap that the previously unconditional wheel build used to cover (PR #1848 review).

`code` excludes `frontend/src/**` and not `frontend/**`, which looks timid but is the widest exclusion that is actually safe: `tests/test_docker_workflow_versions.py` reads `frontend/package.json` and `tests/templates/test_manager.py` reads `frontend/public`, and `paths-filter` cannot express "exclude the directory but keep those two" -- once a pattern excludes a file, no later pattern includes it back. So a frontend-source-only pull request skips the Python lanes, while a `package.json` or asset change still runs them.

`tests/test_ci_summary_contract.py` holds all of the above as tests: that
`needs` and `check_job` name the same set of jobs, that no gathered job carries
a condition beyond the draft guard, that both outputs fall back to running
everything on a truncated or missing file count, that the `frontend` filter still
covers whatever `pyproject` points `readme` at, and that the paths-filter action
stays pinned to its reviewed commit.

Several of those assertions are narrower than they first look, because the
obvious version of each passes while still being bypassable. A step guard is
checked for *polarity*, not merely for naming the output: work steps must lead
with `== 'true'`, the single `Skip` sentinel must be exactly `!= 'true'`, and any
`&&` conjunct a work step adds has to match the matrix or cache predicates it is
allowed to narrow itself with -- accepting an arbitrary conjunct would let
`&& github.event_name == 'push'` skip the step on every pull request with the job
still green. Guard polarity alone still describes a job that runs nothing, so
each gated Python job must also carry a step invoking `python -m pytest`. The
output expressions are frozen whole rather than checked for their clauses,
because `&&` binds tighter than `||` and a rewritten chain keeps every clause
while inverting what it means. The pin is asserted on the step carrying
`id: filter`, and `paths-filter` must appear in the job exactly once -- searching
the job for the SHA would also accept a dead pinned step sitting beside a real
gate on a mutable tag.

Each of these assertions was confirmed to fail against a deliberately broken
workflow before being committed.

### Two contract tests own ci.yml

`ci.yml` is guarded from both sides, and a change to it usually has to update both:

- `tests/test_ci_summary_contract.py` (runs ungated in `pre-commit`, and again in `pytest-fast-deepdoc`) checks the properties above structurally, over the parsed YAML.
- `frontend/src/ci/frontend-test-manifest.test.ts` (runs in `frontend-build`) freezes exactly one region by *exact text*: the `Check required jobs` script, whose non-comment lines must match a hard-coded list element for element. It also executes that script under `bash` to prove failures propagate, so keep the `${{ needs.* }}` interpolations inline in `run:` -- moving them to `env:` leaves that execution with unset variables and silently guts the check.

The six required `frontend-build` test steps are *not* frozen by text. They are checked semantically, over the parsed YAML: each `npm run` command must appear in exactly one step, with `working-directory: ./frontend`, no `continue-on-error`, a bash-compatible shell, and an `if:` that is either absent or exactly `needs.changes.outputs.frontend == 'true'`. Renaming a step, reordering it, or rewriting its key layout is therefore free; changing what it runs or how it is guarded is not. The gate is allowlisted by exact value and nothing else is, so the rule it protects still holds: no *arbitrary* condition can turn a required frontend step off.

That file also pins both `jobs.changes` filter rule sets, which is a deliberate overlap rather than duplication. The Python contract's load-bearing run is the `pre-commit` job: it carries no `changes` condition, so no filter edit can skip it, and `CI Summary` gathers it. Its `pytest-fast-deepdoc` registration is duplicate coverage on top of that, not the thing holding the invariant up -- do not remove the `pre-commit` step as redundant, because that is what reopens the self-gating hole (`Run CI contract tests`, and `test_the_contract_runs_in_a_job_the_filter_cannot_gate` is what fails if it goes). The frontend contract is still `frontend`-gated, which is why pinning the rules in both places earns its keep: dropping `.github/workflows/ci.yml` from the `frontend` filter skips the frontend contract, and the ungated Python run is what still objects. The merge queue's unfiltered run is a further backstop behind both, not the only detector.

This is also why `CI Summary` had to become required at all: before it did, only
the migration checks gated, and a PR with red `pytest`, `e2e` or `pre-commit`
was mergeable. Three open PRs were in exactly that state when it was switched on.

## Gotchas worth knowing

**The queue-branch exclusion needs a trailing `/*`.** Ruleset "Restrict new
branches to main and rls" applies to `~ALL`, which does match
`refs/heads/gh-readonly-queue/<target>/pr-N-<sha>` and rejects its creation. The
exclusion must be spelled `refs/heads/gh-readonly-queue/**/*`;
`refs/heads/gh-readonly-queue/**` does **not** match those refs -- GitHub's rules
API reports zero applicable rules for them -- so that spelling looks like a fix
while leaving every merge stalled.

**A ruleset whose include and exclude are identical is rejected**, silently
leaving the previous value in place. Check the response, do not assume the PUT
applied.

**Removal reasons are only in GraphQL.** The REST timeline event carries no
reason; `RemovedFromMergeQueueEvent.reason` does:

```bash
gh api graphql -F number=PR_NUMBER -f query='
query($number: Int!) { repository(owner:"xorbitsai",name:"xagent"){ pullRequest(number:$number){
  timelineItems(last:5, itemTypes:[REMOVED_FROM_MERGE_QUEUE_EVENT]){ nodes{
    ... on RemovedFromMergeQueueEvent { createdAt reason } } } } } }'
```

Note that a *successful* merge also emits a removal event, with
`reason: merged`. "Removed from merge queue" on its own says nothing about
whether anything went wrong -- read the reason.

**Merge queue is organization-only.** It cannot be configured on a
personal-account repository -- a `merge_queue` ruleset there is rejected with
`Invalid rule 'merge_queue'` regardless of any other setting -- so a personal
fork cannot be used to rehearse queue changes.

### Workflows must opt in

Any workflow producing a required status check needs a `merge_group:` trigger.
Without it the context is never reported on a merge group, the entry times out,
and **every merge stalls**. `ci.yml` and `test-migrations.yml` both carry it.
Adding a new required check means adding that trigger first.

## Break-glass (`enforce_admins` pending)

`enforce_admins` is **not yet enabled**, so administrators can still merge around
the required checks and this procedure is not needed today. It is documented for
when that changes.

Once enabled, nobody -- administrators included -- can bypass the required
checks, and the only override is to disable admin enforcement, merge, and
immediately re-enable it.

**Authorized:** `qinxuye`, `rogercloud` (organization owners).

**Use it only for:**

1. the queue is stuck and a fix must land now;
2. a required check is failing for infrastructure reasons and is blocking every PR;
3. an urgent revert of something already merged.

```bash
gh api -X DELETE repos/xorbitsai/xagent/branches/main/protection/enforce_admins
```

Merge the PR, then restore immediately:

```bash
gh api -X POST repos/xorbitsai/xagent/branches/main/protection/enforce_admins
```

Use these dedicated endpoints rather than `PUT .../protection`, which replaces
the entire configuration -- any field left out of the body is cleared.

Say in the PR why the override was used, so it stays auditable.
