# Xagent Docker Deployment

This directory contains Docker configuration files for deploying Xagent with Docker Compose.
Note: docker-compose.yml is located in the project root directory.

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

### 网络时间受限时：分步拉取镜像

若 `docker compose up -d` 因可用网络时间很短（例如每次只有十几秒）导致镜像拉取不全，可先**只拉取镜像**，再在本地启动：

```bash
# 在项目根目录执行，按服务逐个拉取（postgres → nginx → backend → frontend）
./docker/pull-all.sh
```

脚本会依次拉取：`postgres`、`nginx`、`backend`、`frontend`。每次只拉一个服务，拉完再拉下一个；若某次失败，可重新执行脚本（已存在的镜像不会重复拉）。全部拉取完成后执行：

```bash
docker compose up -d
```

也可手动按需拉取单个服务：

```bash
docker compose pull postgres   # 或 nginx / backend / frontend
```

### 配置镜像源并升级到最新版

若需使用国内镜像源（如 `https://docker.1ms.run`）并一次性拉取最新镜像、升级 xagent：

```bash
# 在 xagent 项目根目录执行（会提示输入 sudo 密码）
./docker/set-mirror-and-upgrade.sh
```

脚本会：1）将 Docker 镜像源设为 `https://docker.1ms.run`；2）重启 Docker；3）`docker compose pull` 拉取最新镜像；4）`docker compose up -d` 启动。仅修改镜像源可手动编辑 `/etc/docker/daemon.json` 后执行 `sudo systemctl restart docker`。

### 离线导入：从其他机器拷镜像包到本机

在**有外网**的机器上拉取并导出为一个 tar 包。**若本机是 x86_64/AMD64，而导出机是 ARM（如 Mac M1），必须指定目标平台**，否则拷过来的镜像是 ARM 架构，在本机无法运行（会报 platform 不匹配、postgres unhealthy 等）：

```bash
# 本机是 x86_64 时，在有网络的机器上执行（指定 linux/amd64）
PLATFORM=linux/amd64 ./docker/save-images.sh
# 或
./docker/save-images.sh xagent-images.tar linux/amd64
```

不指定平台时，导出的是「当前机器架构」的镜像（ARM 机导出 ARM，x86 机导出 x86）：

```bash
# 在有网络的机器上执行（可指定输出路径，默认为当前目录下的 xagent-images.tar）
./docker/save-images.sh
# 或指定路径，例如：
./docker/save-images.sh /path/to/xagent-images.tar
```

将生成的 **`xagent-images.tar`** 拷贝到**本机**项目下的 **`docker/offline-images/`** 目录（若目录不存在可创建），然后在本机项目根目录执行：

```bash
./docker/load-images.sh
```

脚本默认会加载 `docker/offline-images/xagent-images.tar`；若你把 tar 放在别处或改了文件名，可传入路径：

```bash
./docker/load-images.sh /path/to/你的镜像包.tar
```

导入完成后执行 `docker compose up -d` 即可启动（无需再拉取）。

### Custom Port

By default, the frontend runs on port 80. To use a different port (e.g., 8080):

```bash
# In .env file
NGINX_PORT="8080"

# Then start
docker compose up -d
```

## Docker Files

- `Dockerfile.backend` - Backend image (FastAPI, Python, Node.js)
- `Dockerfile.frontend` - Frontend image (Next.js, nginx)
- `docker-compose.yml` - Multi-service orchestration
- `.dockerignore` - Backend build exclusions
- `.dockerignore.frontend` - Frontend build exclusions
- `nginx.conf` - Frontend nginx configuration
- `entrypoint.sh` - Backend startup script

## Building Individual Images

### Backend

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

**Setup (one-time):**

1. Configure GitHub secrets:
   - Go to repository Settings → Secrets and variables → Actions
   - Add `DOCKERHUB_USERNAME`: Your Docker Hub username
   - Add `DOCKERHUB_TOKEN`: Your Docker Hub access token
     - Create at: https://hub.docker.com/settings/security

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
