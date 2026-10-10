# Milvus as the knowledge-base engine

By default Xagent keeps knowledge-base chunks, vectors and search indexes in LanceDB, on the `xagent_data` volume (with `docker/docker-compose.sandbox.docker.yml`, on the host directory it binds at `/root/.xagent`, see [Start](#start)). The `docker/docker-compose.milvus.yml` add-on starts a Milvus standalone server next to the stack and makes it the knowledge-base engine. Xagent still parses, chunks and embeds documents. Milvus stores the searchable chunk copies and vectors and runs dense, keyword (BM25) and substring search. The LanceDB directory stays as the ledger of documents, parses, ingestion status and chunks.

Use it on a **new** deployment. A deployment that already holds LanceDB knowledge-base data is refused at startup (see [The engine lock](#the-engine-lock)), and there is no migration between engines.

The images in `docker-compose.yml` must contain the Milvus engine; the `0.8.1` images do not, because that release predates it. With them the add-on starts and Milvus idles, but the first knowledge-base request fails with `Vector backend 'milvus' is not implemented yet`. Use a later release, or build `docker/Dockerfile.backend` from `main` and point `backend`, `worker` and `scheduler` at it.

The release must also include #2963 (or be `main` at or after it). Without it, the Celery worker's first-start engine check opens LanceDB in the prefork parent process and the forked children crash on their first LanceDB access, so the first ingestion fails until `docker compose restart worker`.

## Start

From the project root:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker/docker-compose.milvus.yml \
  up -d
```

It combines with a sandbox overlay; list every `-f` file on every Compose command:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker/docker-compose.milvus.yml \
  -f docker/docker-compose.sandbox.docker.yml \
  up -d
```

With `docker/docker-compose.sandbox.docker.yml`, `backend`, `worker`, `scheduler` and `nginx` all bind `${XAGENT_HOST_STORAGE_ROOT:-/root/.xagent}` at `/root/.xagent`, so the LanceDB ledger, the collection ids and the engine record are in one place.

The add-on adds three services, the same three containers as the official Milvus v2.6.25 standalone Compose file:

| Service | Image | Volume |
| --- | --- | --- |
| `milvus-etcd` | `quay.io/coreos/etcd:v3.5.25` | `milvus_etcd_data` |
| `milvus-minio` | `milvusdb/minio:RELEASE.2024-12-18T13-15-44Z` | `milvus_minio_data` |
| `milvus` | `milvusdb/milvus:v2.6.25` | `milvus_data` |

`backend`, `worker` and `scheduler` set `XAGENT_VECTOR_BACKEND=milvus` and `MILVUS_URI=http://milvus:19530`. `backend` and `worker` also wait for `milvus` to be healthy; `scheduler` only dispatches tasks and never reads the engine, so a slow Milvus does not hold back scheduled triggers. Nothing is published on the host: Milvus and MinIO (default credentials) are reachable only from the Compose network. To run a Milvus client, use the backend container, which has `pymilvus` and `MILVUS_URI`.

Keep the Milvus image at 2.6.13 or later, the supported floor, and below 3.0 (`pymilvus` is pinned below 3.0).

## Resources

The official guidance for Milvus standalone is 8 GB of RAM (16 GB recommended) and an SSD. Measured on a Docker VM with 11.7 GiB, the three containers used about 0.5 GiB idle and about 2.6 GiB at the peak of a stress run (an index build on 20,000 rows and 1,000 writes of 25 KB each). In the lab run of the official Compose file, a start on empty volumes took about 11 s until the first request succeeded, and a restart with data about 17 s until it could be searched. With this add-on, `docker compose up -d --wait` returned about 22 s after `up` on fresh volumes (21.6 s and 21.5 s), with the images already pulled. The images are 1.37 GB (Milvus), 232 MB (MinIO) and 90 MB (etcd).

Milvus applies writes in steps of 200 ms. Committing a document, the last step of an ingest, which makes its rows visible, took about 1 s whether it had 3 or 300 chunks.

## The engine lock

A deployment uses one engine. The first start records it in `.kb-engine` in the LanceDB data directory (by default `/root/.xagent/data/lancedb` in the backend container, on the `xagent_data` volume or the host directory that the sandbox overlay binds at `/root/.xagent`; `LANCEDB_DIR` and `LANCEDB_PATH` change it). From then on, the backend, the Celery worker and the agent workers refuse to start when `XAGENT_VECTOR_BACKEND` differs from the record. The scheduler only dispatches tasks and does not check. The refusal names the recorded engine and the setting:

```text
This deployment's KB engine is lancedb (documents hold data), recorded in /root/.xagent/data/lancedb/.kb-engine, but XAGENT_VECTOR_BACKEND is milvus. To change the engine of an empty deployment, delete the record file and restart.
```

A deployment that has no record yet (one that started before the lock existed) is detected from its data: rows in the `kb_ids` table mean Milvus, rows in `documents`, `collection_config`, `collection_metadata` or an `embeddings_*` table mean LanceDB, and with neither the setting is recorded. So an existing LanceDB deployment records `lancedb` on its first start and refuses the add-on. When the LanceDB directory cannot be reached or listed, startup logs a warning and does not check. When the record cannot be locked or written, the engine is detected from the data on every start instead; keep the data volume writable by the backend.

To change the engine of an **empty** deployment, stop the stack, delete the record, and start with the new setting. This example moves a LanceDB deployment to Milvus:

```bash
docker compose down
docker compose run --rm --no-deps --entrypoint rm backend -f /root/.xagent/data/lancedb/.kb-engine
docker compose -f docker-compose.yml -f docker/docker-compose.milvus.yml up -d
```

To go back to LanceDB, add `-f docker/docker-compose.milvus.yml` to the first two commands and leave it out of the last.

Empty means that `documents`, `collection_config`, `collection_metadata` and the `embeddings_*` tables hold no rows (deleting every knowledge base empties them; the tables stay) and, to leave Milvus, that `kb_ids` holds none either. If a start is still refused, the message names the tables that hold data. The detection runs again after the record is deleted, so nothing here moves data between engines.

## Consistency window

Search (dense, keyword and hybrid) and the vector counts use Milvus's default Bounded consistency. For a short window after a write returns, a document that was just committed may be missing from the results and one that was just deleted may still be returned. Measured on an idle single node, the new state was visible after 0.2 to 0.45 s (median about 0.22 s); Milvus caps the staleness at 5 s (`common.gracefulTime`). The agent's knowledge search skips a knowledge base whose vector count reads 0, so the first search right after the first ingest into a new knowledge base can skip it.

Ingest does not depend on the window: its reads (which chunks Milvus holds, and the check at commit) use Strong consistency, each about 0.2 s slower after a write.

## Chunk size

Milvus stores each chunk's text in a field of at most 65,535 UTF-8 bytes. An ingest that produces a longer chunk fails before anything is embedded, with an error that names the chunk and says to lower the chunk size. The chunk size setting counts characters (tokens with `use_token_count`), and a character takes 1 byte in ASCII and usually 3 in Chinese or Japanese, so the default of 1,000 characters is far below the limit, while a setting above about 21,800 characters of Chinese text, or a document chunked by semantic splitting with no size limit, can exceed it.

## Backup and restore

The knowledge-base state is in two places that have to agree: the LanceDB ledger (documents, parses, chunks, ingestion status, and the collection ids that Milvus rows are filed under), on `xagent_data` or on the host directory bound at `/root/.xagent` when the sandbox overlay is used, and the three Milvus volumes (`milvus_etcd_data`, `milvus_minio_data`, `milvus_data`). Back them up together, from the same moment, along with `postgres_data` as described under [Backup](README.md#backup). The simplest consistent copy is a cold one: stop the stack, copy the volumes, start it again.

Restoring them to different points in time is not detected or repaired:

- Milvus older than the ledger, or empty: the ledger lists documents whose chunks Milvus lacks, and search does not find them. Re-ingest such a document: it compares the ledger's chunks with the rows Milvus holds under the same collection id and document id and embeds only the missing ones. Deleting the document and importing it again also works.
- Milvus newer than the ledger: rows that belong to a document the ledger does not list are still searched, because search filters on the collection id, but they are listed nowhere, and nothing removes them. Delete them by hand in Milvus (filter on `kb_id` and `doc_id`). Rows of a collection whose id is not in the ledger are not reached at all.

## Differences from LanceDB

Keyword search runs on Milvus's built-in `chinese` analyzer (jieba word segmentation, keeping only tokens that contain a Han character or an ASCII letter or digit). When it finds nothing, a case-sensitive substring search runs on the whole query, and the response carries an `FTS_FALLBACK` warning if that finds something. Compared with LanceDB:

- Keyword search is case-sensitive: `milvus` does not find `Milvus`.
- Full-width letters and digits, kana and Hangul are not indexed as words. A token made only of them is dropped when the text is indexed and when a query is analyzed, so `型号ＡＢＣ１２３已停产` is indexed without `ＡＢＣ１２３`, and a query that mixes them with other words matches on the other words only. A query made only of such tokens matches nothing as words, and the substring search then looks for it as literal text: `ＡＢＣ１２３` finds a chunk containing it, `ＡＢＣ１２３ ひらがな` finds a chunk only if that exact string is in it. LanceDB finds these characters as words.
- There is no English stemming and no stop-word removal: `runs` does not find `running`, and a word such as `the` is indexed and matched like any other. (When nothing else matches, the substring search lets `run` find `running`.) LanceDB applies both.

Dense search uses the same embeddings as on LanceDB, and hybrid search fuses the two result lists with the same code, but scores and the order of close results can differ between the engines.

## Not supported

- Multiple hosts. The LanceDB ledger and the engine record live on the local `xagent_data` volume (or the host directory of the sandbox overlay), which `backend`, `worker` and `scheduler` share, and the add-on starts Milvus in the same Compose project.
- Moving existing knowledge bases between engines. Re-import the documents into a new deployment instead.
