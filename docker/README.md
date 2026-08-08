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
- **PostgreSQL**: PostgreSQL 16 database

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
path before they reach the host Docker daemon. Override these values when the
host checkout or storage directory lives elsewhere:

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
- `../docker-compose.yml` - Base multi-service orchestration
- `docker-compose.sandbox.boxlite.yml` - Boxlite/KVM sandbox overlay
- `docker-compose.sandbox.docker.yml` - Docker sibling sandbox overlay
- `.dockerignore` - Backend build exclusions
- `.dockerignore.frontend` - Frontend build exclusions
- `nginx.conf` - Frontend nginx configuration
- `entrypoint.sh` - Backend startup script

## Building Individual Images

### Backend

Backend image dependencies are resolved from the committed `pyproject.toml` and
`uv.lock` during the Docker build. Keep `uv.lock` up to date before publishing;
the backend image build runs `uv sync --locked` for reproducible installs.

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile.backend \
  -t xprobe/xagent-backend:latest \
  --push .
```

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

Manual GitHub Container Registry publishing is also available through the GitHub Actions workflow:
- Backend: `ghcr.io/<owner>/xagent-backend`
- Frontend: `ghcr.io/<owner>/xagent-frontend`

### Publish to Docker Hub

From the `docker/` directory:

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
   - Create repositories: `xagent-backend` and `xagent-frontend`
   - Or they will be auto-created on first push

2. **Login to Docker Hub** (one-time):
   ```bash
   docker login
   ```

3. **Publish images** (on each release):
   ```bash
   ./docker/publish.sh
   ```

### Docker Hub Repositories

- https://hub.docker.com/r/xprobe/xagent-backend
- https://hub.docker.com/r/xprobe/xagent-frontend

### Automatic Publishing (GitHub Actions)

Images are automatically published to Docker Hub when you create a GitHub release.
You can also run the same workflow manually and optionally publish to GHCR.

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

**Publish on release:**

```bash
# Create a new release (triggers GitHub Actions)
git tag v1.0.0
git push origin v1.0.0
gh release create v1.0.0
```

GitHub Actions will:
- Build backend and frontend images
- Tag with version (e.g., `v1.0.0`, `v1.0`, `v1`, `latest`)
- Push to Docker Hub

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
