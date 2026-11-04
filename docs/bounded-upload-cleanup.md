# Bounded upload disposal (2C-1a)

Refs #1086 and #2853. This extends [2B complete cleanup](complete-upload-cleanup.md)
with bounded disposal of captured resources. Persistent bounded **discovery** is
tracked separately in [#2872](https://github.com/xorbitsai/xagent/issues/2872).
Detached retention, exact eligibility, indexes and independent fair collection
remain 2C-2 after that prerequisite. Neither this PR nor #1086's closed state
means the full retention lifecycle is delivered.

## Capture, deletion and completion

The existing version **1** manifest captures managed sources, key-owned
materializations and producer temporaries eagerly. Preview resources are captured
under the execution guard and saved before preview deletion. These namespace
scans and the final preview verification remain complete directory scans; **they
are not bounded by this PR**. Local capture retains 2B's existing reference-gate
behavior. #2872 must address source temporaries and materialization generations
as well as flat PDF/SVG caches, without silently skipping old layouts.

The exact claim, original ownership evidence, quarantine names and retired-ID
fence commit before destructive work. Storage deletion runs under the existing
execution lock after SQL connections and KB reference locks are released.
Durable, local and preview receipts retain their existing order. Settlement
requires every receipt; a partial converter tree cannot authorize a preview
receipt or row removal.

Each activation adds optional `disposal_positions` to the version 1 manifest.
The position for a phase advances only after successful resource disposal and is
the authoritative completed prefix. Budget exhaustion commits that prefix and
returns `yielded`, leaving the claimed row unavailable and recoverable. Old
version 1 manifests that contain per-resource `disposed` fields remain valid;
the fields are harmless and ignored. A resource's quarantine locator is durable
before its original path is renamed.

## Disposal budgets

| Work across one cleanup activation | Limit |
| --- | ---: |
| Captured resource attempts, child unlinks and directory removals | 128 |
| Converter directory entries and EOF probes | 256 |
| Converter traversal depth | 64 |

Local and preview disposal share the same budget. Resource attempts include
already absent resources; child deletion consumes that budget too. Converter
scans close before modifying their directory. A later worker reopens the reduced
quarantine and deletes remaining children; filesystem removals provide durable
progress without a directory offset or a process-local iterator. Parent and
converter device/inode/mode checks reject replacements, and descriptor walks do
not follow symlinks. Direct cache-deletion paths remain outside this protocol.

These are disposal operation bounds, not wall-clock or discovery bounds. Whole
file checksums, provider calls, namespace capture, final preview verification and
serialization of the existing JSON manifest can still be expensive. Deep trees,
configuration drift and ownership uncertainty retain their handle for
reconciliation rather than being called complete.

## Recovery, concurrency and cancellation

Compensation recovery counts `yielded` as `deferred_budget`, separately from
uncertainty failures. Registered-upload rollback treats a committed budget yield
as handoff to recovery and does not report a durable-storage double fault.
`exists`, `unknown`, `pending` and actual operation errors keep their existing
failure contracts.

The default recovery cadence still uses a 300-second stale window and a
60-second poll. A takeover writes a current claim token, and a budget yield saves
that token, so the next activation normally waits for the token to become stale
again. The initial activation is often immediate. With fast local I/O, `K`
activations therefore spend roughly `(K - 1) * 300` seconds waiting for stale
eligibility, plus polling alignment, queue backlog and I/O. This is not a hard
completion bound, and the final activation settles immediately rather than
waiting through another stale window. A converter with mostly regular files
needs about four activations for 400 files and eight for 1,000 files: roughly 15
and 35 minutes of stale-window waiting respectively, before the additional
factors above. [#2891](https://github.com/xorbitsai/xagent/issues/2891) tracks
prompt rescheduling after `yielded` as a later 2C-2 enhancement; this PR does not
change scheduler behavior or add scheduling state.

A failed progress commit leaves the previous SQL obligations intact. Retry can
revisit absent resources or partially reduced quarantines using their original
evidence. No stream registry or uncommitted discovery page survives an exception.
Workers serialize destructive work and token takeover using the existing file
execution lock; an old token cannot apply progress or settle a newer claim.
Cancellation drains the current worker activation before propagating, preserving
unfinished work. Each activation may run in a different process without losing
disposal progress, even while unrelated preview files are being generated.

The removed discovery proposal used size/mtime invalidation and process-local
streams. It could repeatedly rescan a directory's first page without completing.
Restoring eager capture removes that failure from this slice; it does not solve
strict bounded discovery. That requirement remains explicitly outstanding in
#2872 and blocks claiming all of 2C complete.

## Ownership and matching axes

- **Stable ID:** preserve row/owner/file/key identity, exact claim token and
  manifest generation. Retired IDs remain fenced after settlement.
- **Name:** keep literal file-ID preview prefixes and producer temporary prefixes.
  Names never replace captured parent/inode/checksum evidence.
- **Transport:** preserve captured backend, URI and provider routing validation.
- **Configuration:** use canonical roots from unified configuration and the
  effective shared coordination directory; reject escaping or replaced parents.
- **Authentication:** use current credentials; manifests contain no authentication
  material. Rotation does not change captured ownership or routing.
- **Ownership:** preserve external/shared sources, unproven historical temporaries
  and replacements. Capture evidence before destructive work.
- **Scope:** preserve normalized owner/workspace keys and 2A's cross-owner KB
  document/active-target protection. Task-before-upload SQL locking is unchanged.

## Deployment and rollback

No schema or environment-variable change is required. Existing populated
SQLite/PostgreSQL installations keep version 1 manifests and receipts; existing
captured temporary files are disposed directly, without new discovery jobs.
Deploy compatible writers and all cleanup/recovery workers with the same
resource roots, coordination directory and effective OS file-lock semantics.
Older 2B workers understand version 1 evidence but do not enforce the new budgets;
consistent disposal limits require updating every cleanup/recovery worker.

Before replacing any experimental or separately deployed worker that wrote an
unsupported manifest version, quiesce writers and drain or reconcile those
claims with a worker compatible with that version. This implementation retains
unsupported versions as pending; it never strips unknown obligations or treats
them as version 1. Preserve those handles until their outstanding work has been
accounted for.

Rolling back to 2B preserves version 1 obligations and captured ownership, but
returns to unbounded disposal. Quiesce active workers before changing versions;
never erase retired IDs or replace live coordination lock inodes. Keep SQL,
resource roots and coordination files together. Safe fence/lock compaction is
separate work.

Historical inventory, ambiguous unmarked rows and managed legacy/local-only
backfill remain later delivery. Uploads predating rollout must not be permanently
excluded from a later detached policy solely because of their creation date.
#2835, #2836, #2848, #2849 and #2851 retain their existing independent scopes.
Account/team erasure, backup replay and recovery UI remain outside this PR.
