# KB references and upload cleanup (2A)

Refs #1086. Baseline `d1aeafc9` includes the task-attachment lifecycle (#2783),
SQLite task identity fix (#2802), and completed KB reference/snapshot work
(#2662 and #2665). This slice enables **no detached-file collector**.

## Contract

For a stable `file_id`, document publication and destructive cleanup claims
share one exclusive, cross-process lock:

- If a document or an active ingest target establishes its reference first,
  the cleanup claim fails. The existing task-less collector also preserves its
  local source before attempting its claim.
- If cleanup claims first, later document upserts/restores and target admission
  fail. Claim state is committed before durable deletion. A cleanup fence keeps
  the retired identity recognizable after compensation settles and removes the
  `UploadedFile` row.
- Reference lookup errors are errors, never evidence that a source is unused.
  No claim or local unlink happens on an uncertain lookup.
- Removing the last document or releasing/tombstoning its target permits a new
  eligibility decision. It does not change `detached_at` or bypass the retained
  detachment policy. The task-less collector still excludes detached rows.

Protection crosses document/target owners; it grants no new permission to read,
download, or reuse an upload. File names and paths are not reference identities.
Independent standalone RAG sources remain valid without SQL initialization.
Retired managed IDs require a successful new upload publication before reuse.

## Writers and consumers

| Boundary | Actual paths and coordination |
| --- | --- |
| Sync document upsert | `LanceDBVectorIndexStore.upsert_documents`; normal registration through the legacy facade/coordinator/handle and `restore_document` converge here. |
| Native async document upsert | `upsert_documents_async` uses the same gate; acquisition/SQL checking runs off the event loop, and cancellation drains native publication before releasing it. |
| Whole-document snapshot restore | `LanceDBCollectionHandle.restore_document_rows` gates all restored document file IDs before its raw, multi-table merge/delete operations. It remains the coordinator boundary added by #2743. |
| Pending target publication | `admit_kb_ingest_target`, including its uniqueness-conflict retry, validates file availability inside the gate before updating SQL. Failed admission, including lock timeout, releases only its generation and marks the already-created job failed when SQL is available. |
| Fresh staged upload | `/ingest/jobs` preserves the deterministic missing-row ID unless it was retired by cleanup; then it selects a new ID before admission. A fresh upload cannot be mistaken for a late reference to the cleaned identity. |
| Staged/background document ingestion | `handle_kb_ingest_document` binds generation identity through copied contexts; the final document write rechecks that generation. Success or a terminal error releases only that generation; retryable attempts retain it. A superseded worker cannot release its replacement. |
| Target removal | Explicit generation release, file tombstone, and collection tombstone cease pending protection through `deleted_at`. They do not remove protection supplied by surviving LanceDB documents. |
| Cleanup claims | `_claim_orphan` and `compensate_registered_uploads_sync` use uncapped `find_referenced_file_ids` plus a cross-owner active-target query under the same locks. |
| Cleanup settlement/recovery | Common settlement preserves the retired identity. Existing recovery continues owning compensating rows and their exact tokens; it does not restore availability. |
| New durable upload publication | `UploadedFileStore.add_already_durable` schedules fence reconciliation after the root SQL transaction commits and returns its connection. Rollback does not clear the fence. |
| Other reference consumers | KB orphan/collection-delete classification, reconcile, file status, source lookup, and preview/download authorization retain their current interfaces and scope. This slice reuses the exact-query facade instead of replacing those APIs. |

Offline LanceDB schema/file-ID backfill scripts also mutate document rows. They
are maintenance writers, require stopped runtime writers/collectors, and are not
converted into online migration or backfill services here.

## Transaction and lock boundaries

Lock files live under the configured, normalized LanceDB directory in
`.file-references/`, keyed by SHA-256 of `file_id`; all owners use the same key.
Current LanceDB connection management supports local filesystem directories.
Processes accessing the same KB must share that directory and effective OS
file-lock semantics. Never delete its lock files while any process is running:
replacing the inode creates two independent ownership locks. A 15-second lock
timeout fails the operation safely.

Batch locks are acquired in sorted file-ID order. A caller must release its
clean SQL transaction before acquiring reference locks; pending writes are
rejected rather than rolled back. Reference writers use a short, independent
SQL Session for validation, close it, and then publish to LanceDB under the
filesystem lock. Target admission owns its existing SQL transaction and commit.
Staged handlers carry their caller's SQL engine in the generation context, so
validation uses the same database through an independent short Session.
Cleanup checks active targets in SQL, returns that connection, queries fresh
LanceDB document references, and then performs its exact SQL claim. The existing
task-before-upload row-lock order is unchanged; no task lock is added here.

No transaction spans both SQL and LanceDB. Serialization instead spans their
short operations, without retaining a SQL connection over LanceDB publication,
checksum work, or durable-object I/O. The existing compensation batch behavior
is preserved, including finishing the batch before reporting unresolved deletes.
Reference-query or lock failures abort the compensation batch before any claim;
this conservative behavior is distinct from finishing already-claimed durable
deletes. Per-file isolation is tracked in
[#2836](https://github.com/xorbitsai/xagent/issues/2836). Referenced files are
logged and skipped. Compensation
releases reference locks after the claim/fence transaction closes, before durable
deletion or preview cleanup. Admission waits off the event loop with its own
short SQL Session on the caller's engine. Contention returns a sanitized 503
with `Retry-After`; unavailable or superseded identities return a sanitized 409.

## Persistent fences and interruption

`uploaded_file_cleanup_fences` stores only retired file identities and claim
timestamps. It deliberately has no upload/user foreign key. A corresponding
`.claimed` marker, fsynced before the SQL claim commits, lets standalone RAG
writers reject retired IDs even when no Web SQL factory is installed.
Retired SQL fence rows, their `.claimed` markers, and per-ID `.lock` files grow
monotonically in this slice. Safe compaction remains separate follow-up work:
retired identities must remain fenced, and live lock inodes cannot be replaced.

These are rejection fences, not reference pins: they cannot prevent subsequent
cleanup of an unreferenced file. A surviving available/legacy SQL upload is
authoritative after rollback or a legitimate new publication. Validation under
the reference lock reconciles its conservative marker. A new durable upload
also reconciles after root commit; a failed post-commit reconciliation retains
the marker and retries on a subsequent Web reference operation.

- Before LanceDB publication: process exit releases the OS lock and leaves no
  new reference state. A retry or cleanup can proceed.
- After an atomic document merge: process exit releases the lock; the next
  exact document query protects the committed reference.
- After successful ingest, failure to release its target is logged without
  replacing success. The target conservatively retains protection until an
  exact generation release, replacement, or tombstone succeeds. Release failure
  also preserves an original ingest error; retryable attempts retain the target.
- Before SQL claim commit: SQL rolls back. A conservative standalone marker
  may survive; a Web operation validates the still-live upload and clears it.
  It does not permanently pin that upload against a later cleanup attempt.
- After claim commit: SQL and standalone fences reject publication, including
  after the upload row disappears. Existing compensation recovery retains its
  generation token and continues the deletion protocol.
- During async cancellation: the owned async task is drained before unlock,
  so cleanup cannot race an abandoned native document commit.

## Upgrade and rollout

Fresh ingestion while an old canonical-path upload is still compensating remains
a safe conflict until cleanup settles. A fresh-ID publication/cleanup handoff is
tracked separately in [#2835](https://github.com/xorbitsai/xagent/issues/2835).

Deploy all reference writers and cleanup workers together; mixed old/new writers
cannot honor the gate. Stop them before the online Alembic upgrade, retain the
shared LanceDB directory (including its coordination files), and restart all
workers on the same version. Quiesce pre-upgrade compensation work as well;
this is prospective coordination, not a legacy/backfill rollout.

Fresh model creation and SQLite/PostgreSQL upgrades produce the same additive
fence table. Upgrade preserves upload/target data and is idempotent. An
interrupted transactional upgrade can be retried. Offline DDL is rejected.
Downgrade refuses to remove a populated fence table: erasing retired identities
would permit stale references. Preserve the database and coordination directory
together when moving a deployment; backup replay reconciliation is #2781.

## Remaining delivery dependencies

- #1086 **2B** now extends these claims with the
  [complete cleanup protocol](complete-upload-cleanup.md). Claims commit before
  local unlink, and recovery retains upload metadata until every phase succeeds.
- #1086 **2C** owns the detached seven-day TTL, scan indexes, scheduler, and work
  budget. Future detached claims must use this gate and fence exact detachment
  provenance/version in their CAS; no detached sweep is added here.
- #1086 later slices own legacy/backfill/inventory; SaaS #1576 owns account/team
  closure. Direct/standalone deletion is not redesigned here.
- Pre-existing whole-collection rollback versus concurrent ingest is #1242;
  background-job takeover/duplicate execution is #1566; per-operation worker
  bookkeeping Sessions are #1535. This change does not claim those are solved.
