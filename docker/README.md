# Xagent Docker Deployment

This directory contains Docker configuration files for deploying Xagent with Docker Compose.
Note: the base `docker-compose.yml` is located in the project root directory. Advanced
Compose overlays, including sandbox runtime options, live in this directory.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │     │    Backend      │     │   PostgreSQL   │
│  (Next.js)      │────│   (FastAPI)     │────│   Database      │
│  Port: 80       │ API │  Port: 8000      │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Services

- **Frontend**: Next.js standalone build served by nginx
- **Backend**: FastAPI with Python 3.11, Node.js 22, Playwright, LibreOffice
- **PostgreSQL**: PostgreSQL 17 database; the image tag is overridable via `POSTGRES_IMAGE_TAG`

> **Required action on upgrade:** A `postgres_data` volume initialized by PostgreSQL 16 cannot be switched to 17 in place. PostgreSQL 17 refuses to open it and exits, so `postgres` restart-loops and `backend`, `worker`, and `scheduler` never start. Pin `POSTGRES_IMAGE_TAG="16-bookworm"` to defer, then migrate with [PostgreSQL major version upgrade (16 to 17)](#postgresql-major-version-upgrade-16-to-17). Fresh installations are unaffected.

## Quick Start

Browser sign-in requires local browser storage and the Web Locks API for safe
cross-tab session coordination. Use HTTPS for non-local deployments. The
`localhost`, `127.0.0.1`, and `::1` loopback forms are suitable secure contexts
for local development. The bundled HTTP Compose endpoint is a local-development
reference; production deployments must terminate TLS and use a browser that
supports Web Locks.

### 1. Configure Environment

Copy and edit the environment file:

```bash
cp example.env .env
# Edit .env with your API keys
```

Required environment variables:

```bash
# LLM API Keys (at least one required)
OPENAI_API_KEY="your-openai-api-key"
DEEPSEEK_API_KEY="your-deepseek-api-key"

# Database Password (auto-generated if using docker-compose)
POSTGRES_PASSWORD="xagent_password"
```

Optional Gmail incoming-email trigger provisioning:

```bash
# Canonical public backend URL used by browser-facing API and MCP OAuth flows.
# This is not the frontend XAGENT_APP_BASE_URL.
XAGENT_PUBLIC_API_BASE_URL="https://api.example.com"

# Optional server-to-server backend URL advertised to Gmail Pub/Sub and A2A
# clients. Regional deployments should use their direct regional origin.
# When unset, this falls back to XAGENT_PUBLIC_API_BASE_URL.
XAGENT_S2S_API_BASE_URL="https://region-origin.example.com"

# Deprecated Gmail-only fallback used only when XAGENT_S2S_API_BASE_URL is
# unset. A2A never advertises this legacy URL.
XAGENT_TRIGGER_CALLBACK_BASE_URL="https://legacy-callback.example.com"

# Google Cloud project and deterministic per-mailbox resource prefixes.
XAGENT_GMAIL_PUBSUB_PROJECT_ID="your-gcp-project"
XAGENT_GMAIL_PUBSUB_TOPIC_PREFIX="xagent-gmail"
XAGENT_GMAIL_PUBSUB_SUBSCRIPTION_PREFIX="xagent-gmail-push"

# Service account used by Pub/Sub push OIDC tokens.
XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT="pubsub-push@your-gcp-project.iam.gserviceaccount.com"

# Local/container credential file path when not running on GCP with ADC.
GOOGLE_APPLICATION_CREDENTIALS="/run/secrets/google-application-credentials.json"
```

The backend uses Google Application Default Credentials. Grant the backend
service account `roles/pubsub.editor` on `XAGENT_GMAIL_PUBSUB_PROJECT_ID`
(create/delete topics and subscriptions, and update subscription push config);
provisioning also reads subscription config to skip redundant updates, but
degrades to re-applying it when that read is unavailable. Allow
`gmail-api-push@system.gserviceaccount.com` to publish to each per-mailbox
topic. Xagent grants the Gmail publisher IAM binding during provisioning when
the credentials have permission to update topic IAM policy.

Backend startup applies the
`20260729_add_gmail_audience_grace` Alembic migration before serving requests.
It adds two nullable audience-grace columns to `gmail_watch_states`; no data
backfill or separate migration command is required. Complete that backend
startup before running the endpoint reconciler below.

When introducing or changing `XAGENT_S2S_API_BASE_URL`, deploy and verify its
direct-origin ingress before changing the backend environment. Existing Gmail
subscriptions persist their push endpoint and OIDC audience in Pub/Sub, so
reconcile them after the backend deployment:

```bash
# Read-only audit: inspects the existing database and Pub/Sub configuration
# without running database initialization or changing either system.
python -m xagent.web.reconcile_gmail_push_endpoints

# Apply each reported Pub/Sub and stored-audience change independently.
# Execute mode performs the normal database initialization before reconciling.
python -m xagent.web.reconcile_gmail_push_endpoints --execute

# Verify convergence. A successful rerun reports changed=0 and failed=0.
python -m xagent.web.reconcile_gmail_push_endpoints
```

Run these commands in the backend container or an equivalent environment with
the production database configuration, Google credentials, and Gmail
environment variables. The command emits a JSON summary and exits nonzero when
any watch fails. It preserves each Gmail callback identifier, history cursor,
watch expiration, and existing `users.watch` registration.

To roll back the callback URL, restore the previous
`XAGENT_S2S_API_BASE_URL` (or unset it to use
the deprecated `XAGENT_TRIGGER_CALLBACK_BASE_URL`, then
`XAGENT_PUBLIC_API_BASE_URL`), redeploy the backend, and run the same audit
and `--execute` sequence. Keep both origins routable until the final audit
reports no failed or changed watches.

## 2026-07-30 — Owner deployment target discovery

### Deployment impact

The owner frontend now loads `GET /api/deployment-config` before generating
Agent or Workforce API, SDK, widget, and share artifacts. Standalone XAgent
preserves the existing API-config, browser-origin, and
`XAGENT_APP_BASE_URL` behavior. Hosting layers may replace this route to
advertise a shared external ingress and a region bootstrap.

### Prerequisites and configuration

No new standalone environment variable is required. Existing reverse proxies
must continue forwarding `/api/*` to the backend. The Gmail audience-grace
migration described above is part of the same backend release, but the owner
deployment-target route itself does not require a data backfill.

### Deployment and migration steps

Deploy the backend containing `/api/deployment-config` before the matching
frontend. If a new frontend temporarily reaches an older backend, it uses the
browser origin and keeps a warning with a retry action visible. That fallback
preserves standalone behavior, but it can be the wrong external target for a
regional deployment; finish the backend rollout and retry before copying an
artifact there.

### Verification and monitoring

Verify that `/api/deployment-config` returns the expected `app_origin`, with
`deployment_origin` and `region` unset for standalone XAgent. In the owner UI,
verify one Agent or Workforce API snippet, widget snippet, and public share
link against the deployment's existing external origins. Also verify that a
failed configuration request shows the persistent fallback warning and that
Retry clears it after the route becomes available.

### Rollback

No persistent state is changed. Roll back the frontend and backend together;
the previous clients resume deriving their targets directly from runtime and
browser configuration.

### 2. Start Services

From the project root directory:

```bash
docker compose up -d
```

This will start all services in the background.

### 3. Access Services

- **Frontend**: http://localhost:80
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 4. View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f postgres
```

### 5. Stop Services

```bash
docker compose down
```

## Advanced Usage

### Custom Port

By default, the frontend runs on port 80. To use a different port (e.g., 8080):

```bash
# In .env file
NGINX_PORT="8080"

# Then start
docker compose up -d
```

### Sandbox Runtime Overlays

Sandbox deployment is an advanced option. Use one sandbox overlay at a time, from
the project root.

Boxlite/KVM sandbox:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker/docker-compose.sandbox.boxlite.yml \
  up -d
```

This requires Linux or WSL2 with KVM support and grants the backend container
KVM access.

Docker sibling sandbox:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker/docker-compose.sandbox.docker.yml \
  up -d
```

This mounts the host Docker socket into the backend container so Xagent can
create sibling sandbox containers through the host Docker daemon. Treat this as
a privileged deployment mode: Docker socket access is effectively host-level
container control.

Docker sibling mode also resolves sandbox bind mounts on the Docker host. The
overlay defaults `XAGENT_SANDBOX_HOST_PROJECT_ROOT` to the current project root
and binds `${XAGENT_HOST_STORAGE_ROOT:-/root/.xagent}` to `/root/.xagent`. It
also passes `XAGENT_SANDBOX_HOST_STORAGE_ROOT` into the backend so sandbox
workspace mounts under `/root/.xagent` are translated back to the host storage
path before they reach the host Docker daemon. When set,
`XAGENT_SANDBOX_HOST_STORAGE_ROOT` must be an absolute Docker-host path;
relative paths and `~` are not supported. Override these values when the host
checkout or storage directory lives elsewhere:

```bash
XAGENT_SANDBOX_HOST_PROJECT_ROOT="$PWD" \
XAGENT_HOST_STORAGE_ROOT="$HOME/.xagent" \
docker compose \
  -f docker-compose.yml \
  -f docker/docker-compose.sandbox.docker.yml \
  up -d
```

In Docker sibling mode, `SANDBOX_VOLUMES` sources are host-side paths. Use
absolute host paths; relative paths and `~` are rejected instead of being
expanded inside the backend container.

Sandbox workspace guest paths are reserved below
`<XAGENT_UPLOADS_DIR>/user_<id>`. A `SANDBOX_VOLUMES` destination or
`XAGENT_EXTERNAL_UPLOAD_DIRS` mount at that directory or any descendant now
fails backend readiness, because Docker would let the more-specific bind hide
the managed workspace. Move shared mounts elsewhere under the uploads root
(for example `<uploads>/shared`) or mount a suitable ancestor before upgrading.

External-upload symlink aliases authorize the physical directory they resolve
to. Ordinary aliases are supported, but ambiguous spellings such as
`symlink/..` are rejected; configure the intended directory directly.
Xagent creates and owns each `<uploads>/user_<id>` directory as a normal
directory; replacing one with a symlink or changing that topology while the
service is running is unsupported.

> **Required action on upgrade:** Every existing deployment with
> `SANDBOX_ENABLED=true` and `SANDBOX_IMPLEMENTATION=docker` must provide
> `XAGENT_SANDBOX_NAMESPACE` before upgrading. The Compose overlay below sets
> it automatically; pip/systemd deployments must set their own stable, unique
> deployment identifier (see `example.env`). Missing values stop backend
> startup rather than falling back to unsafe daemon-global ownership.

The overlay sets `XAGENT_SANDBOX_NAMESPACE` on `backend`, `worker`, and
`scheduler` from the resolved `${COMPOSE_PROJECT_NAME}`. Docker sibling mode
treats that Compose project name as authoritative and overrides any namespace
from `example.env`; a missing value fails during Compose interpolation. Every
sandbox container a deployment creates is scoped to that namespace (physical
name and owner labels), so multiple deployments sharing one Docker daemon
never discover or manage each other's sandboxes. Co-located stacks must use
distinct Compose project names and distinct `XAGENT_HOST_STORAGE_ROOT` paths.

`XAGENT_SANDBOX_MAX_CONTAINERS` applies independently to each deployment
namespace, not globally to the shared daemon. Size the daemon for up to the
per-deployment limit multiplied by the number of co-located stacks, plus any
legacy containers awaiting removal.

Containers created before this scheme existed carry the legacy
`xagent.managed=true` label and are ignored by current code: they are never
listed, reclaimed, or counted against `XAGENT_SANDBOX_MAX_CONTAINERS`, and
a backend restart recreates their sandboxes fresh (host bind mounts
survive; container-layer state does not). The backend logs how many such
containers exist at startup.

Use one coordinated maintenance window to upgrade:

1. Stop new sandbox admission and drain every task on the old stacks.
2. Stop the old backends and their legacy sandbox containers before starting
   any namespaced backend.
3. Upgrade and start every co-located stack with a distinct Compose project
   name and host storage root.
4. After confirming that no old backend or task uses them, inventory and
   remove the stopped legacy containers:

   ```bash
   docker ps -a --filter label=xagent.managed=true
   ```

Never run a legacy container and its v2 replacement concurrently. Both mount
the same host workspace, so concurrent execution creates two live writers.

Rollback also requires a maintenance window. Drain and stop the namespaced
backends and their v2 containers before starting pre-namespace code; the old
backend cannot see v2 containers and otherwise double-provisions sandboxes.

## Docker Files

- `Dockerfile.backend` - Backend image (FastAPI, Python, Node.js)
- `Dockerfile.frontend` - Frontend image (Next.js, nginx)
- `Dockerfile.sandbox` - Sandbox image for isolated code execution
- `../docker-compose.yml` - Base multi-service orchestration
- `docker-compose.sandbox.boxlite.yml` - Boxlite/KVM sandbox overlay
- `docker-compose.sandbox.docker.yml` - Docker sibling sandbox overlay
- `.dockerignore` - Backend build exclusions
- `.dockerignore.frontend` - Frontend build exclusions
- `nginx.conf` - Frontend nginx configuration
- `entrypoint.sh` - Backend startup script

## Cloud Ingest Limits and Timeouts

`POST /api/kb/ingest-cloud` accepts one to five files per request. The bundled
`nginx.conf` gives this route a 900-second read timeout. For native Google
Workspace files, Drive polling has a 600-second default application deadline.
The final transfer, parsing, chunking, and embedding all continue within the
same HTTP request.

Custom reverse proxies must allow for the full end-to-end request. Increase the
proxy timeout when you increase
`XAGENT_GOOGLE_DRIVE_DOWNLOAD_TIMEOUT_SECONDS`. A closed client request does
not stop the active worker thread.

## Building Individual Images

### Backend

Backend image dependencies are resolved from the committed `pyproject.toml` and
`uv.lock` during the Docker build. Keep `uv.lock` up to date before publishing;
the backend image build runs `uv sync --locked` for reproducible installs.

**Build args:**

| Arg | Default | Effect |
|-----|---------|--------|
| `INSTALL_CHROME` | `true` | Installs Google Chrome (amd64) or Chromium (arm64) plus a warmed `npx` cache for the built-in Chrome MCP connector (`chrome-devtools-mcp`). Pass `--build-arg INSTALL_CHROME=false` to skip both, dropping the image size and the `/opt/google/chrome/chrome` binary + npx cache for deployments that never enable the connector — it ships hidden from the connector catalog until #1200 lands regardless of this flag. **This flag does not remove all root/`--no-sandbox` browser exposure in the image**: Playwright Chromium is installed unconditionally in a separate build stage and is already launched with `--no-sandbox` as root by the pre-existing `browser_use` tool, independent of this connector and this flag. Operator note: the connector launches via `npx` with an exact version pin; on a deployment whose npx cache is cold (a non-Docker install, or an `INSTALL_CHROME=false` image later flipped visible), the first tool call fetches that pinned package from the npm registry, as the backend user, before the server starts. |

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile.backend \
  -t xprobe/xagent-backend:latest \
  --push .
```

### Sandbox

The sandbox remains a separate image for untrusted code execution. Its Python
packages come only from the dedicated `[dependency-groups].sandbox` group in
`pyproject.toml`. Whenever that group changes, run `uv lock` from the repository
root and commit the updated `uv.lock`. `docker/Dockerfile.sandbox` exports the
group from the lockfile and checks all supported imports during the image build.
The build stage does not copy `pyproject.toml` or `uv.lock` into the runtime
image.

Custom `SANDBOX_IMAGE` images used with sandboxed uvx MCP connections must
provide `uvx` on `PATH`; Xagent no longer installs uv dynamically.

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile.sandbox \
  -t xprobe/xagent-sandbox:latest \
  --push .
```

The `Publish Sandbox Image` workflow in
`.github/workflows/sandbox-publish.yml` publishes release tags and supports
manual tags. After a new sandbox tag is published, update the `SANDBOX_IMAGE`
pins in `docker/docker-compose.sandbox.boxlite.yml` and
`docker/docker-compose.sandbox.docker.yml` to reference it. Rolling back only
requires restoring the previous sandbox image tag.

### Frontend

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile.frontend \
  -t xprobe/xagent-frontend:latest \
  --push ./frontend
```

## Publishing Images

Images are published to Docker Hub under the `xprobe` organization:
- Backend: `xprobe/xagent-backend:latest`
- Frontend: `xprobe/xagent-frontend:latest`
- Sandbox: `xprobe/xagent-sandbox:latest`

`docker/publish.sh` builds and publishes only the backend and frontend images.
The sandbox image is published separately by the `Publish Sandbox Image`
workflow in `.github/workflows/sandbox-publish.yml`; it is not part of
`publish.sh`.

Manual GitHub Container Registry publishing is also available for the backend
and frontend through their GitHub Actions workflow:
- Backend: `ghcr.io/<owner>/xagent-backend`
- Frontend: `ghcr.io/<owner>/xagent-frontend`

### Publish to Docker Hub

To publish the backend and frontend, run these commands from the `docker/`
directory:

```bash
# Publish with default tag (latest)
PUSH=true ./publish.sh

# Publish with version tag
PUSH=true ./publish.sh v1.0.0

# Local single-platform build without pushing
PLATFORMS=linux/arm64 ./publish.sh
```

`publish.sh` behavior:

- `PUSH=true` (or `CI=true`) -> publish images (`--push`)
- local default (`PUSH=false`) -> local build only (`--load`, single platform)
- local multi-platform without push will fail fast with a hint

Or manually:

```bash
# Build and tag
docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile.backend -t xprobe/xagent-backend:latest --push .
docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile.frontend -t xprobe/xagent-frontend:latest --push ./frontend
```

> If Docker Buildx is not initialized locally, run:
>
> ```bash
> docker buildx create --use
> docker run --privileged --rm tonistiigi/binfmt --install all
> ```

### First Time Setup

1. **Create Docker Hub repositories** (one-time):
   - Go to https://hub.docker.com/
   - Create repositories: `xagent-backend`, `xagent-frontend`, and
     `xagent-sandbox`
   - Or they will be auto-created on first push

2. **Login to Docker Hub** (one-time):
   ```bash
   docker login
   ```

3. **Publish backend and frontend images** (on each release):
   ```bash
   ./docker/publish.sh
   ```

4. **Publish the sandbox image** separately by pushing a release tag or by
   manually running the `Publish Sandbox Image` workflow defined in
   `.github/workflows/sandbox-publish.yml`.

### Docker Hub Repositories

- https://hub.docker.com/r/xprobe/xagent-backend
- https://hub.docker.com/r/xprobe/xagent-frontend
- https://hub.docker.com/r/xprobe/xagent-sandbox

### Automatic Publishing (GitHub Actions)

Backend and frontend images are automatically published to Docker Hub when you
create a GitHub release. The sandbox image is published separately by
`.github/workflows/sandbox-publish.yml` when a tag is pushed. Both workflows
can also be run manually; only the backend/frontend workflow supports optional
GHCR publishing.

**Setup (one-time):**

1. Configure GitHub secrets:
   - Go to repository Settings → Secrets and variables → Actions
   - Add `DOCKERHUB_USERNAME`: Your Docker Hub username
   - Add `DOCKERHUB_PASSWORD`: Your Docker Hub access token (not your password)
     - Create at: https://hub.docker.com/settings/security
     - Use "Read & Write" permissions for pushing images

2. Ensure Docker Hub repositories exist:
   - `xprobe/xagent-backend`
   - `xprobe/xagent-frontend`
   - `xprobe/xagent-sandbox`

**Publish on release:**

```bash
# Create a new release (triggers GitHub Actions)
git tag v1.0.0
git push origin v1.0.0
gh release create v1.0.0
```

GitHub Actions will:
- Build and publish the backend and frontend images from the GitHub release
- Build and publish `xprobe/xagent-sandbox` separately when the tag is pushed
- Tag the images with their workflow's release tags

### Manual GHCR Publish

1. Open the `Publish Docker Images` or `Nightly Build` workflow in GitHub Actions.
2. Run it manually with `push_to_ghcr=true`.
3. Leave `push_to_dockerhub=false` if you only want GHCR.

GitHub Actions will:
- Log in to GHCR with `GITHUB_TOKEN`
- Build backend and frontend images
- Push to `ghcr.io/<owner>/xagent-backend` and `ghcr.io/<owner>/xagent-frontend`

**Workflow file:** `.github/workflows/docker-publish.yml`

## Production Deployment

### Environment Variables

Key production variables:

```bash
# Database (set via docker-compose.yml)
DATABASE_URL="postgresql://xagent:password@postgres:5432/xagent"

# Image tag of the bundled postgres service (default: 17-bookworm).
# Pin "16-bookworm" for a postgres_data volume that v16 initialized.
POSTGRES_IMAGE_TAG="17-bookworm"

# Security
ENCRYPTION_KEY="your-encryption-key"
```

### Volumes

Data persists in Docker volumes:

- `postgres_data` - PostgreSQL database
- `xagent_data` - User data (~/.xagent/)
- `xagent_uploads` - Uploaded files

### Backup

```bash
# Backup database
docker compose exec postgres pg_dump -U xagent xagent > backup.sql

# Restore database
docker compose exec -T postgres psql -U xagent xagent < backup.sql
```

### PostgreSQL major version upgrade (16 to 17)

The bundled `postgres` service defaults to `postgres:17-bookworm`. PostgreSQL never upgrades a data directory across major versions: a v17 server refuses to open a v16 directory, exits with `FATAL: database files are incompatible with server`, and restart-loops as `unhealthy`, which blocks `backend`, `worker`, and `scheduler`. It does not modify the directory, so pinning `POSTGRES_IMAGE_TAG="16-bookworm"` restores service at any point.

The steps below are the standard major-version path — dump under v16, initialize a fresh v17 cluster, restore. **This is a general method, not a procedure tuned to your deployment: use the backup and verification process you already trust for this database, and rehearse it against a copy first.** Release-level context is the dated entry in [`docs/deployment.md`](../docs/deployment.md).

> **Data loss warning:** `docker compose down -v` and removing the live `<project>_postgres_data` volume destroy the database irreversibly. Neither is an upgrade step. Removing the separate `_pg16_backup` copy created in step 4 is expected.

> **Sandbox overlays:** keep the `-f docker/docker-compose.sandbox.*.yml` arguments on every Compose command below, or `backend`, `worker`, and `scheduler` come back without the overlay.

Set up. `PGVOL` is the Compose-prefixed volume name — find it with `docker volume ls | grep postgres_data`. Keep `BACKUP` outside the checkout; the dump carries password hashes, OAuth tokens, and encrypted provider credentials.

```bash
PGVOL=xagent_postgres_data
BACKUP="$HOME/xagent-pg16-backup.sql"
unset POSTGRES_IMAGE_TAG   # Compose prefers a shell value over .env
```

**1. Pin v16 and confirm the deployment is healthy.**

```bash
printf '\nPOSTGRES_IMAGE_TAG="16-bookworm"\n' >> .env
docker compose up -d postgres
docker compose exec postgres psql -U xagent -d xagent -tAc 'SHOW server_version'
```

**2. Stop the writers**, leaving `postgres` running. `nginx` and `frontend` keep serving errors for the whole window; stop them too if that is unacceptable.

```bash
docker compose stop backend worker scheduler
```

**3. Back up and record what you will compare against.** An interrupted `pg_dump` still leaves a plausible-looking file, so check the result.

```bash
docker compose exec -T postgres pg_dump -U xagent -d xagent > "$BACKUP"
grep -q 'PostgreSQL database dump complete' "$BACKUP" && echo OK || echo 'INCOMPLETE - do not proceed'
docker compose exec -T postgres psql -U xagent -d xagent -tAc 'SELECT version_num FROM alembic_version'
```

**4. Stop the stack and copy the v16 volume**, so rollback never depends on the dump alone. No `-v`; `-t 60` lets Postgres finish its shutdown checkpoint. The destination must be a fresh volume, because `cp -a` merges rather than mirrors.

```bash
docker compose down -t 60
docker volume rm "${PGVOL}_pg16_backup" 2>/dev/null   # absent on a first attempt
docker volume create "${PGVOL}_pg16_backup"
docker run --rm -v "$PGVOL":/from:ro -v "${PGVOL}_pg16_backup":/to alpine sh -c 'cp -a /from/. /to/'
docker run --rm -v "${PGVOL}_pg16_backup":/d alpine cat /d/pgdata/PG_VERSION   # expect: 16
```

**5. Remove the v16 data directory from the live volume.** The guard blocks the destructive command unless the dump completed and the copy really is v16.

```bash
if grep -q 'PostgreSQL database dump complete' "$BACKUP" \
   && [ "$(docker run --rm -v "${PGVOL}_pg16_backup":/d alpine cat /d/pgdata/PG_VERSION)" = 16 ]; then
  docker run --rm -v "$PGVOL":/d alpine sh -c 'rm -rf /d/pgdata'
else
  echo 'PRECONDITIONS NOT MET - nothing changed; do not continue'
fi
```

**6. Delete the `POSTGRES_IMAGE_TAG` line from `.env`** and start `postgres`, which initializes a fresh v17 cluster. `pg_isready` needs `-h 127.0.0.1`: during initialization the image runs a temporary server on the Unix socket only, which a socket check reports as ready.

```bash
docker compose up -d postgres
docker compose exec postgres pg_isready -h 127.0.0.1 -U xagent -d xagent
```

**7. Restore and verify.** `ON_ERROR_STOP=1` fails loudly instead of leaving a half-populated database.

```bash
docker compose exec -T postgres psql -U xagent -d xagent -v ON_ERROR_STOP=1 < "$BACKUP"
docker compose exec -T postgres psql -U xagent -d xagent -tAc 'SHOW server_version'                        # expect: 17.x
docker compose exec -T postgres psql -U xagent -d xagent -tAc 'SELECT version_num FROM alembic_version'    # expect: the step 3 value
docker compose exec -T postgres vacuumdb -U xagent --all --analyze-in-stages
```

Those two queries establish only that the schema arrived. Run whatever proves the *data* arrived for your deployment — row counts on the tables you care about, spot checks against recent records, an application smoke test. `vacuumdb` rebuilds the optimizer statistics that `pg_dump` does not carry across.

**8. Bring the writers back, only after your verification passes.**

```bash
docker compose up -d
```

**Rollback — valid only until step 8 restarts the writers.** Until then nothing has written to the v16 copy from step 4. Confirm it exists and really is v16 *before* removing anything: `docker run -v` creates a named volume that does not exist, so an unchecked copy-back from a missing volume restores nothing over a directory it has already deleted.

```bash
docker compose down -t 60
if docker volume inspect "${PGVOL}_pg16_backup" >/dev/null 2>&1 \
   && [ "$(docker run --rm -v "${PGVOL}_pg16_backup":/d alpine cat /d/pgdata/PG_VERSION)" = 16 ]; then
  docker run --rm -v "$PGVOL":/d alpine sh -c 'rm -rf /d/pgdata'
  docker run --rm -v "${PGVOL}_pg16_backup":/from:ro -v "$PGVOL":/to alpine sh -c 'cp -a /from/. /to/'
  printf '\nPOSTGRES_IMAGE_TAG="16-bookworm"\n' >> .env
  docker compose up -d
else
  echo "BACKUP VOLUME MISSING OR NOT v16 - live data untouched; restore from $BACKUP instead"
fi
```

After v17 accepts writes the copy is stale, and copying it back discards everything written since the cutover; recovery from that point means a fresh v17 backup plus reconciliation. Keep the copy while you are still deciding whether the upgrade held, then remove it with `docker volume rm "${PGVOL}_pg16_backup"`.

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs backend

# Check health status
docker compose ps
```

### Database Connection Issues

```bash
# Verify postgres is running
docker compose exec postgres pg_isready -U xagent

# Check database logs
docker compose logs postgres
```

### PostgreSQL Won't Start After an Upgrade

`docker compose ps` reports `postgres` as `restarting` and `unhealthy`, and `backend`, `worker`, and `scheduler` never start because they wait for `service_healthy`. A data directory initialized by an older major version produces this in `docker compose logs postgres`:

```
FATAL:  database files are incompatible with server
DETAIL:  The data directory was initialized by PostgreSQL version 16, which is not compatible with this version 17.11 (Debian 17.11-1.pgdg12+2).
```

PostgreSQL 17 refuses to open the directory rather than modifying it, so the v16 data is intact. Pin the previous major version to restore service immediately, then upgrade deliberately using [PostgreSQL major version upgrade (16 to 17)](#postgresql-major-version-upgrade-16-to-17).

```bash
printf '\nPOSTGRES_IMAGE_TAG="16-bookworm"\n' >> .env
docker compose up -d postgres
```

If `postgres` still starts on 17 after this, a `POSTGRES_IMAGE_TAG` exported in the shell is overriding `.env`; `unset` it and retry.

### Rebuild After Code Changes

```bash
# Rebuild specific service
docker compose build backend
docker compose up -d backend

# Rebuild all
docker compose build
docker compose up -d
```

## Development

### Running Tests in Docker

```bash
# Run backend tests
docker compose exec backend pytest

# Run with coverage
docker compose exec backend pytest --cov=src/xagent --cov-report=html
```

### Hot Reload (Development Mode)

For development with hot reload, use the standard setup instead of Docker:

```bash
# Backend (from project root)
python -m xagent.web.__main__

# Frontend (from frontend/)
cd frontend
npm run dev
```

## Security Notes

- Change default passwords in production
- Use `.env` file (never commit secrets)
- Enable SSL/TLS for production deployments
- Use Docker secrets for sensitive data
- Keep images updated with security patches
