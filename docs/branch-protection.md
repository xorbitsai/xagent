# Branch protection and the merge queue

How `main` is protected, why it is set up this way, and what to do when it
blocks something urgent.

> **Status.** Live today: the `alembic-check` hook, the required
> `Test SQLite/PostgreSQL Migrations` checks, and the `merge_group:` triggers
> (inert until a queue exists). Still pending, in this order:
>
> 1. `CI Summary` as a required status check -- independent of the queue.
> 2. Exclude `refs/heads/gh-readonly-queue/**` from the branch-creation ruleset,
>    or the queue cannot build anything.
> 3. Enable the merge queue, **leaving `required_status_checks.strict` on**.
> 4. Verify the queue end to end: a merge-group branch is created, and every
>    required context reports on it. An already-up-to-date PR satisfies strict, so
>    it can be enqueued for this without relaxing anything first.
> 5. **Only now set `required_status_checks.strict` to `false`.** It is still
>    `true` on `main`, and it has to go: a PR must satisfy its branch-protection
>    requirements before it is admitted to the queue, so leaving strict on
>    permanently would keep every PR syncing `main` and re-running checks first --
>    exactly the serialization the queue exists to remove.
> 6. `enforce_admins` -- deliberately last, once the queue is proven.
>
> **The order of 3 and 5 matters.** Turning strict off first would leave a window
> where neither protection applies: `main` would accept an approved but stale PR
> on the strength of green checks that never saw the combined tree, which is the
> migration race this whole change exists to prevent. If queue activation then
> failed, the repository would be stuck in that weaker state. Keeping strict on
> until the queue is confirmed working means the worst case is the status quo.
>
> Sections below describe the mechanism; anything above that is still pending is
> not in force yet. Update this block as each step lands.

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

## How it is enforced

| Layer | What it catches | Where |
| --- | --- | --- |
| `alembic-check` pre-commit hook | multiple heads, duplicate revision IDs, unresolvable `down_revision` | locally on commit, and on every PR via the `pre-commit` job |
| Merge queue | the same defects, evaluated against the tree that will actually land | at merge time |
| `Test SQLite/PostgreSQL Migrations` | a real `alembic upgrade head` on both backends | required status check |
| `ci.yml` on `push: main` | post-merge detection if something slipped through | red build on `main`, emailed to the pusher |

`scripts/check_alembic_heads.sh` is the single implementation of the graph
check. Its regression coverage lives in
`tests/migrations/test_check_alembic_heads.py`, which injects each defect into a
throwaway Alembic environment and asserts the script rejects it. Those negative
tests exist because a check that cannot fail is worse than no check: it reads as
protection that is not there.

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
mergeMethod:                  SQUASH   # the only method enabled on this repo
maxEntriesToBuild:            5        # speculative builds; this is what keeps throughput up
maxEntriesToMerge:            1        # merge one PR at a time
minEntriesToMerge:            1
minEntriesToMergeWaitMinutes: 5        # inactive while minEntriesToMerge is 1
groupingStrategy:             ALLGREEN
checkResponseTimeoutMinutes:  60       # worst observed ci.yml run is 29.4 min
```

`maxEntriesToMerge: 1` means every commit on `main` was validated on its own, so
`git bisect` and reverts stay meaningful. It does not slow the queue down:
throughput comes from `maxEntriesToBuild`, which builds later entries while
earlier ones are still being checked.

`groupingStrategy` has no effect while `maxEntriesToMerge` is 1. It is set to
`ALLGREEN` so that raising the batch size later cannot silently downgrade to the
weaker "only the group head must pass" semantics.

### Workflows must opt in

Any workflow producing a required status check needs a `merge_group:` trigger.
Without it the context is never reported on a merge group, the entry times out,
and **every merge stalls**. `ci.yml` and `test-migrations.yml` both carry it.
Adding a new required check means adding that trigger first.

The queue also needs to be able to create its branches: ruleset
"Restrict new branches to main and rls" must exclude
`refs/heads/gh-readonly-queue/**`.

And `required_status_checks.strict` must end up `false`. GitHub tracks the
up-to-date requirement separately from the queue, and a PR has to satisfy its
branch-protection requirements before it is admitted -- so leaving strict on
keeps every PR syncing `main` by hand before it can even enter the queue.

Turn it off *after* the queue is confirmed working, not before; see the ordering
note in the status block above.

## Break-glass (`enforce_admins` pending)

Pushes to `main` are restricted with an empty allowlist, so nobody can push
directly. `enforce_admins` is **not yet enabled**: until it is, administrators
can still merge around the required checks, and this procedure is not needed.

Once it is enabled, nobody -- administrators included -- can bypass the required
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

Use these dedicated endpoints rather than `PUT .../protection`. The full-object
PUT replaces the entire configuration, and omitting the empty-allowlist
`restrictions` block silently removes the push restriction with no error.

Say in the PR why the override was used, so it stays auditable.
