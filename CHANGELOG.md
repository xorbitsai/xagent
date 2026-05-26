# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- **Documented `LANCEDB_AUTO_MIGRATE` in environment template**
  Added `LANCEDB_AUTO_MIGRATE` usage notes to `example.env`, including default behavior (`false`) and when to enable startup auto-migration.

### Changed

- **Knowledge Base embedding model binding (breaking / migration)**
  The Knowledge Base now treats the **Model Hub ID** as the single source of truth for embedding model identity:
  - `collection_metadata.embedding_model_id` stores the Hub ID (trimmed; no other normalization).
  - Embeddings tables are named by Hub ID: `embeddings_{to_model_tag(hub_id)}`.
  - The `model` field stored alongside each embedding vector is the Hub ID.

  **Migration / backward compatibility:** Older deployments may have created embeddings tables using the provider `model_name`
  (e.g. `embeddings_text-embedding-v4`). During search and embedding reads, the system will **try the new Hub-ID table first**
  and automatically **fall back to the legacy table name** derived from the resolved `model_name` when the new table is missing.
  Rebuild/inference helpers were updated to prefer Hub IDs when they can be resolved from Model Hub metadata.

- **Knowledge Base upload: default parse method (breaking)**
  The default parse method on the KB detail upload form is now `"default"` instead of `"pypdf"`. The backend chooses the parser by file type (e.g. .docx, .pdf). If you rely on the previous default (always use PyPDF), select `"pypdf"` explicitly in the parse method dropdown when uploading.

- **Knowledge Base document registration (breaking)**
  Document IDs for new uploads are now generated deterministically from `(collection, source_path)` instead of a random UUID. Re-uploading the same file in the same collection updates the existing record instead of creating a duplicate.
  **Impact on existing data:** Documents that were registered with the previous logic (random UUID in `doc_id`) will get a *different* `doc_id` when re-uploaded. Re-uploading such a file will create a new record rather than updating the old one, so you may briefly see two entries for the same filename until the old one is removed. If you rely on idempotent re-uploads for previously registered documents, consider deleting the old document from the KB before re-uploading, or plan a one-time cleanup of legacy duplicates.

- **LanceDB user_id migration hardening**
  Startup and migration logic now include cross-process file locking, legacy `-1` orphan marker remapping to reserved int64 sentinel values, zero-progress loop protection, and shared embeddings-table listing utilities to avoid API-compat drift.

- **Chunk multi-page metadata and document page stats (breaking)**
  Chunking now preserves multi-page origins via `metadata["spanning_pages"]` (a sorted list of 1-based page numbers) and derives `page_number` from the first page when missing. Parse records also persist document-level page statistics (`page_count`, `page_numbers`) in `params_json`.
  **Impact on existing data:** Previously generated chunks only stored a single `page_number` and dropped cross-page information. It is not possible to retroactively reconstruct full page coverage for those chunks without re-parsing and re-chunking the original documents. As a result, page-based views for legacy data may still undercount pages or show chunks only on their representative page until those documents are reprocessed.
