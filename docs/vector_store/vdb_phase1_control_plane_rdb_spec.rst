VDB Phase 1: Control Plane Decoupling + Metadata RDB Migration
=============================================================

Goal
----
- Decouple KB control-plane logic from LanceDB implementation details.
- Move control-plane metadata to RDB with safe migration gates.

Scope
-----
- Introduce control-plane facade and ``MetadataStore`` contract.
- Migrate to RDB (phase-1 entities):
  - ``collection_metadata``
  - ``main_pointers``
  - ``prompt_templates``
  - ``ingestion_runs``
  - ``documents``
- Keep vector index data in VDB.

Non-goals
---------
- No Milvus/Chroma runtime switching in this phase.
- No full migration of large payload fields (e.g. parse/chunk large blobs).

Execution Plan
--------------
1. Add anti-corruption layer for control-plane operations.
2. Add RDB tables + SQLAlchemy repository implementation.
3. Enable dual-write (RDB primary, VDB compatibility).
4. Backfill historical metadata.
5. Switch read path to RDB behind feature flag.
6. Stop VDB control-plane writes after stable window.

Audit Gates
-----------
- Contract tests for control-plane API behavior parity.
- Data consistency checks (count/hash/sampling) >= 99.99%.
- Verified rollback path via feature flags.

Definition of Done
------------------
- API/service layer no longer directly depends on LanceDB control-plane semantics.
- Control-plane reads are stable from RDB in production-like validation.
