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
> Still pending: `CI Summary` as a required status check, and `enforce_admins`.

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
| `Test SQLite/PostgreSQL Migrations` | a real `alembic upgrade head` on both backends | required status check |
| `ci.yml` on `push: main` | post-merge detection if something slipped through | red build on `main`, emailed to the pusher |

`scripts/check_alembic_heads.sh` is the single implementation of the graph
check. Its regression coverage lives in
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
| required status checks | `Test SQLite/PostgreSQL Migrations` |
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
