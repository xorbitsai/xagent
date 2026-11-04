# Complete upload cleanup (2B)

Refs #1086. This extends [2A reference coordination](kb-reference-cleanup-coordination.md)
for the existing task-less public-share collector and registered upload compensation.
It does not finish #1086 or enable detached collection.

## Completion and durable state

An exact SQL claim changes an available upload to `compensating`, commits its
`updated_at` generation token, cleanup manifest, and retired-ID fence, and only
then permits destructive I/O. The upload row remains the recovery handle until
**durable object, disposable managed local copies, and owned previews** have
successful, committed phase receipts. Object absence alone cannot settle it.

The nullable JSON `cleanup_manifest` belongs only to this lifecycle. Keeping
existing upload columns longer is necessary but insufficient: after object
deletion, its content-hash locator cannot be recovered from storage; after local
rename, original inode evidence and quarantine ownership cannot be reconstructed;
and retry must distinguish completed phases from unfinished obligations. The
manifest retains the exact key, backend and sanitized routing URI, original
checksum/ETag, configured roots, captured resource and parent identities, owned
quarantine names, and receipts. A random manifest generation stays fixed across
worker takeovers so their quarantine locators remain usable.

The existing compensation status, recovery scan, SQL timestamp CAS, and cleanup
fence are reused. No new inventory, scheduler, eligibility policy, or general
resource registry is introduced.

## Phases, transactions, and interruption

| Point | Persisted obligation and restart behavior |
| --- | --- |
| Before claim commit | The upload is available and no bytes have been removed. An exact version CAS or reference failure leaves it untouched. A rolled-back claim may leave 2A's conservative `.claimed` marker; live SQL validation reconciles it. |
| After claim commit | The upload is unavailable to binders and reference writers. A fresh recovery session takes over the exact token and resumes the retained manifest. |
| Durable deletion | The existing delete/probe contract reports absent, present, or uncertain. Only absence records the durable receipt. A lost acknowledgement retries the same immutable key; uncertain presence retains every obligation. |
| Local cleanup | Source and materialization locators were captured before claim. A bounded descriptor walk rejects escaping symlinks and replaced parents. Captured resources are renamed to a recorded quarantine, the parent is fsynced, and only the captured inode is disposed. Exit after rename resumes at that quarantine. |
| Preview discovery | With producers excluded, owned PDF/PNG caches and converter temporaries are enumerated and their identities committed before removal. Discovery failure can retry because the manifest and source identity remain present. |
| Preview disposal | Partial removal retries the recorded list. Already absent files succeed; directory identity ignores size/mtime changes caused by partial removal but retains device/inode/type checks. A final enumeration detects unexpected surviving owned previews. |
| Receipt commit | Failure leaves the preceding durable receipt state. Repeating the phase is safe, including an already absent object or quarantine. |
| Settlement | Exact row/user/file/task/key/status/token predicates and all three receipts authorize row deletion. Commit failure leaves the handle recoverable. The SQL retired-ID fence and standalone marker survive successful settlement. |

Each SQL transaction is short: initial claim, takeover, manifest/receipt updates,
and settlement. Metadata/configuration capture and filesystem operations occur
after the connection is returned. No SQL connection or KB reference lock spans
checksums, storage deletion, local unlink, or preview conversion/removal.

Cancellation retains the existing worker-draining contract. Request cancellation
waits for registered compensation to finish or retain its claim. Its local
finalizer preserves still-registered files, including rebound/protected files
and partial compensation, and checks the original staged inode before removing
an unregistered path. Original caller errors and cancellations remain primary;
registered compensation still reports unresolved work as its existing durable
storage error. Recovery counts local/preview uncertainty as pending failure.

## Lock order and generation ownership

A persistent per-file `.cleanup.lock` serializes claim capture, execution,
takeover, settlement, local restoration/materialization, and preview publication.
It shares 2A's configured LanceDB coordination directory but has a distinct inode
from the reference `.lock`. Claim acquisition order is cleanup execution lock,
then reference lock, then short SQL work. Batch claims acquire each sorted file
ID's execution/reference pair. They release reference locks after commit and
before entering storage work. Reference writers do not acquire execution locks;
local/preview producers do not acquire reference locks. The existing SQL
attachment task-before-upload order stays intact.

A takeover acquires the execution lock before changing the exact SQL token. It
cannot settle while an older deleter still has I/O in flight. An old worker
arriving later fails its token check before I/O. Receipt and settlement updates
also compare the exact token. Local publishers validate current available/legacy
metadata through an independent short session under the execution guard. A
previously loaded `ManagedFileRef` cannot restore a retired upload. Async preview
conversion drains publication and releases its guard before propagating cancellation.
Existing local copies, validated materialization copies, and fresh preview caches
are read without taking the execution lock. Materialized reads validate the owner
scope and checksum and recheck availability/generation after probing the cache.
File APIs, Chat, WebSocket attachment resolution, and KB ingestion snapshot ORM
values and return clean request connections before offloading storage and lock
waits. Dirty transactions retain their changes; a copy requiring storage I/O reports the typed
storage error instead of publishing through that transaction. Claimed, retired,
and changed generations have a distinct unavailable error and cannot use the
durable-missing local fallback.
Cache-miss conversion keeps its execution guard while the converter can write
owned temporary output. Moving only the final rename under the guard would let
cleanup settle while the converter can recreate files; narrowing that interval
requires a separate producer/recovery contract. Cached reads do not wait for it.
Publication contention uses the existing storage-error/503 path for copies and
SVGs, and the converter's existing None/503 canvas fallback for PPTX. The
execution-lock timeout does not escape as an unhandled request error.

## Ownership and matching axes

| Axis | Gate or reason it does not select cleanup resources |
| --- | --- |
| Stable ID | SQL row ID, owner ID, file ID, exact storage key, task ID, compensating status, and persisted claim token fence execution and settlement. Retired file IDs remain recognizable after row removal. |
| Name | Filename only locates the basename of the key/checksum-specific materialization copy. Preview ownership uses the literal file-ID prefix, never a broad filename glob. Names alone do not authorize deletion. |
| Transport | Stored backend, base URI, object URI and provider endpoint/region must match captured routing. Changing a backend or destination leaves the claim pending. |
| Configuration | Unified configuration supplies upload/materialization/preview roots. Captured canonical roots must still match. Descriptor walks reject symlinks below these roots and pin the expected parent. All workers must share the coordination directory and OS file-lock semantics. |
| Authentication | Current configured provider credentials are used on retry and may rotate. Passwords, signed query strings and URI userinfo are not persisted. Credentials do not confer ownership or change the captured destination. |
| Ownership | Sources must lie inside the configured owner's managed upload root, outside external-read roots, have SHA-256 evidence, and match captured inode/content evidence. Shared hard links are preserved. Materialization ownership derives from the exact normalized key/hash directory; previews and new restoration temporaries carry the stable owner identity. |
| Scope | The existing storage consumer validates the exact owner prefix using the same tolerant key normalization. Existing workspace segments stay in the captured key; a deleted task is not used to invent another namespace. Cross-owner KB references still protect cleanup without granting read access. |

Configured root symlinks are mapped to their canonical roots without resolving
children beneath them. Explicit external-read roots remain preserved, including
aliases nested in uploads. Local materializations are captured from the storage
layer's exact normalized-key namespace, including backend-derived hash directories;
manifest capture does not read durable bytes or probe the provider under reference
locks. Unverifiable namespaces remain pending before durable deletion. Quarantine
and producer temp names are bounded in bytes; restoration reserves enough space
for the nested atomic-copy name to retain the same ownership prefix after a crash.

Managed source restoration temporaries now include a hash of the stable file ID;
materialization temporaries live in the exact key/checksum directory. Owned
partial copies and converter directories can therefore be reclaimed without
assuming complete contents. Unknown historical source-side temporary names are
preserved for later inventory/backfill. Managed copies and cache files must be
regular files; only owned converter temporary directories permit recursive
disposal. A missing child cannot stand in for absence of its retained quarantine.
External/shared source paths are preserved;
they are never made disposable by the external-read allowlist. Missing resources
are successful steps. Replacement, checksum mismatch, escaping symlink, missing
ownership evidence, or configuration drift retains the manifest and quarantine
locators for reconciliation; it never authorizes deleting the replacement.

Unbound staged-object compensation and superseded immutable-object cleanup retain
their existing interfaces and do not become row-backed inventory. Direct deletion
gets only a narrow guard against removing a compensating row; its broader caller
transaction and error contract is unchanged.

## Deployment, migration, and remaining work

Stop upload/local/preview producers, KB reference writers, and cleanup/recovery
workers; upgrade to `20261005_uploaded_file_cleanup_manifest`; restart all of them
on the same version. Mixed workers cannot honor the execution/publication gate.
Preserve the SQL database, configured resource roots, and coordination directory
together. This is online DDL; fresh model installation and populated SQLite/
PostgreSQL upgrade have matching nullable JSON schema. Retrying an interrupted
upgrade is safe. A schema without an upload table defers to fresh model creation.
Downgrade refuses any pending manifest; complete or reconcile
those claims before removing the column. Prior retired-ID fence downgrade
constraints still apply.

A pre-upgrade compensating row can adopt a manifest from retained metadata before
any new destructive operation. Ambiguous old metadata remains pending. Objects or
copies whose metadata was already lost need later historical inventory/backfill.

SQL fence rows, `.claimed` markers, reference `.lock` and execution `.cleanup.lock`
files remain monotonic. **Do not delete retired identities or replace live lock
inodes to reduce growth.** Safe compaction requires its own retention/coordination
protocol and remains explicit follow-up work; this PR performs none.

- #1086 2C still owns detached seven-day TTL, scan indexes, scheduling and work
  budget. Detached uploads remain excluded from the existing collector.
  [2C-1a bounded disposal](bounded-upload-cleanup.md) limits captured-resource
  disposal while retaining unfinished manifests. Persistent bounded resource
  discovery remains [#2872](https://github.com/xorbitsai/xagent/issues/2872),
  including local/temp and flat preview namespaces. Detached retention and
  scheduler activation remain 2C-2 after that prerequisite.
- #1086 legacy/local-only backfill and historical inventory remain later delivery.
- [#2835](https://github.com/xorbitsai/xagent/issues/2835) owns fresh canonical-path
  publication while old metadata is compensating; the existing safe conflict remains.
- [#2836](https://github.com/xorbitsai/xagent/issues/2836) owns per-file isolation of
  uncertain reference checks before compensation claims; batch behavior remains.
- [#2848](https://github.com/xorbitsai/xagent/issues/2848) owns removing SQL/row-lock
  ownership across immediate direct deletion. Deferred KB cleanup now queues
  durable/local/preview work only after its generation CAS succeeds and runs it
  after commit; the immediate-delete caller contract remains separate.
- [#2849](https://github.com/xorbitsai/xagent/issues/2849) owns consolidating existing
  managed preview locators/matchers; current PDF/SVG producer and cleanup layouts
  are covered by lifecycle regressions.
- [#2851](https://github.com/xorbitsai/xagent/issues/2851) owns classifying retained
  cleanup uncertainty for reconciliation. An already claimed row with ambiguous
  ownership or changed resource identity remains compensating with its manifest;
  later recovery attempts preserve it and report failure until reconciled.
- Pre-existing collection rollback/concurrent ingestion
  [#1242](https://github.com/xorbitsai/xagent/issues/1242), job takeover/duplicate
  execution [#1566](https://github.com/xorbitsai/xagent/issues/1566), worker bookkeeping
  sessions [#1535](https://github.com/xorbitsai/xagent/issues/1535), backup replay
  [#2781](https://github.com/xorbitsai/xagent/issues/2781), and account/team erasure
  [#1576](https://github.com/xorbitsai/xagent/issues/1576) are out of scope.
