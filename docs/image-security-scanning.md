# Image security scanning

Weekly vulnerability scan of the images published to Docker Hub, run by [`.github/workflows/image-security-scan.yml`](../.github/workflows/image-security-scan.yml).

## What is scanned

| Image | Tag | Package ecosystems Trivy sees |
| --- | --- | --- |
| `xprobe/xagent-backend` | `latest` | Debian packages, pip/uv (the venv at `/opt/venv`), npm, and Go/Rust binaries vendored into `node_modules` |
| `xprobe/xagent-frontend` | `latest` | Debian packages, npm |
| `xprobe/xagent-sandbox` | `latest` | Debian packages, pip, npm |

Three targets, `linux/amd64` only. The images are multi-arch, but CVE differences between architectures are marginal and scanning both would double runner time for little signal.

Only `latest` is scanned — what users actually pull. `nightly` is deliberately excluded: it is rebuilt every day, so its findings churn constantly without telling you anything about what is deployed. The `buildcache` tags are excluded too; they are buildx cache manifests, not runnable images.

## Where to look

Go to the repository's **Security and quality** tab → **Code scanning** in the left sidebar (GitHub [renamed the tab from "Security" in April 2026](https://github.blog/changelog/2026-04-02-the-security-tab-is-now-security-quality/); the URL is unchanged):

<https://github.com/xorbitsai/xagent/security/code-scanning>

Each alert names the CVE, the package, the installed version and the version that fixes it. Severity is shown per alert and the sidebar counts them by level, so "how many Critical do we have" is answered without filtering anything.

Alerts have a lifecycle: an alert closes itself once the next scan no longer finds the CVE, and reopens if it comes back. You are reading a list of current problems, not a weekly pile of fresh reports.

### Filtering

The page has dropdowns for the common cases, and the search box takes qualifiers. The ones worth knowing:

| Qualifier | Use |
| --- | --- |
| `is:open` / `is:closed` | Default view is open alerts |
| `severity:critical` | Also `high`, `medium`, `low`. Combine them: `severity:critical severity:high` |
| `tool:Trivy` | Separates image findings from any other scanner added later |
| `path:opt/venv` | Restrict to a path inside the image — see below |
| `branch:main` | Alerts as of a given branch |

### Viewing one image at a time

**Not from the alert list, and this is worth understanding before you trust any per-image number you see there.**

Every image is uploaded under its own SARIF category (`image-scan/backend-latest`, `image-scan/frontend-latest`, `image-scan/sandbox-latest`). That is what keeps the three scans from resolving each other's alerts, but it is not a view you can filter by: [GitHub does not support filtering the alert list by category](https://github.com/orgs/community/discussions/70050).

Worse, **counting alerts by category does not give you per-image totals.** GitHub deduplicates alerts by rule and location, so an identical finding at an identical path in two images becomes a single alert. That alert can belong to more than one configuration, but a per-alert view surfaces only one category for it — `most_recent_instance.category`, which the query below reads, is by definition the most recent one — so a shared finding counts towards one image and disappears from the other's total. The frontend and sandbox images share a `node:22` base, and 17 of the frontend's npm findings sit at the same `usr/local/lib/node_modules/npm/...` paths as the sandbox's. Measured on a fork: the frontend's own report lists 47 actionable findings, while counting alerts by category gives the frontend 30 and the sandbox the other 17. Neither number is wrong, but neither is "the frontend's vulnerability count" either.

So:

1. **For an authoritative per-image list, use the artifact.** One JSON report per image, per run, deduplicated against nothing. This is the only per-image view that means what it looks like. See below.
2. **For a quick look in the UI, use the `path:` filter.** Findings carry the path they were found at *inside the image*: backend application dependencies under `path:opt/venv` (Python) and `path:opt/xagent` (npm and vendored binaries), frontend ones under `path:app/node_modules`. OS-package findings carry the image reference itself as their path, so `path:xagent-backend` isolates the backend's Debian findings. `usr/local` appears in all three images, which is exactly where the attribution above gets shared around.
3. **Via the API**, if you want to see how alerts are currently attributed — remembering that this is attribution, not the image's real total:

   ```bash
   gh api "repos/xorbitsai/xagent/code-scanning/alerts?state=open&per_page=100" --paginate \
     --jq '.[] | select(.most_recent_instance.category == "image-scan/backend-latest")
                 | "\(.rule.security_severity_level)\t\(.rule.id)"'
   ```

### The full report

The alert list deliberately shows a filtered subset (next section). The **complete** report — every severity, including everything with no fix available — is attached to each workflow run as the `trivy-report-<slug>` artifact (JSON, kept 30 days):

```bash
gh run download <run-id> --repo xorbitsai/xagent -n trivy-report-backend-latest
```

Go there when you want the real total rather than the actionable subset. One caveat: this artifact is unfiltered by severity and by fix availability, but not by [`.trivyignore`](../.trivyignore) — Trivy reads that file from the repository root by default, so an accepted-risk entry drops out of the artifact as well as out of the alert list.

## Why the alert list shows less than the artifact

The backend image carries Debian, pip and npm dependency trees plus Chrome, Playwright and LibreOffice, and an unfiltered scan of it reports **thousands** of vulnerabilities. An alert list that long gets ignored, which is worse than no alert list. So the code scanning upload is filtered down to what someone can act on:

- `ignore-unfixed: true` — only CVEs that have a fixed version available. A vulnerability nobody can patch yet is not a to-do item. This is the filter that does the real work: on the first backend scan it cut 4,554 findings to 399, almost entirely because Debian marks the overwhelming majority of its CVEs as not-to-be-fixed (4,173 Debian findings became 30). What survives is nearly all application-level: pip, npm, and Go/Rust binaries vendored into `node_modules`.
- `CRITICAL,HIGH,MEDIUM` only. LOW and UNKNOWN stay in the artifact.
- `limit-severities-for-sarif: true` — required, and this is `trivy-action`'s behaviour rather than Trivy's. The action's `entrypoint.sh` unsets `TRIVY_SEVERITY` before invoking Trivy whenever the format is SARIF and this input is not `true`, which makes the filter above silently do nothing. The Trivy CLI on its own honours `--severity` with `--format sarif`.

Note that the severity shown on the alert is GitHub's own, derived from the CVSS score, so it will not always match Trivy's label — a Trivy MEDIUM can appear as `low`.

## Handling an alert

1. Open the alert; it names the package, the installed version and the fixed version.
2. Decide where the package comes from:
   - **Debian package** — the base images use floating tags (`python:3.11-bookworm`, `node:22-slim`), so a rebuild usually picks the fix up on its own and the alert closes at the next scan. If it survives several scans, the base image tag itself needs bumping in the relevant `docker/Dockerfile.*`. Note that Debian marks most of its CVEs as not-to-be-fixed, and those never become alerts at all — see the filtering section above.
   - **pip / npm package** — bump it in `pyproject.toml` or `frontend/package.json`.
   - **Go or Rust binary** — a compiled binary vendored into `node_modules` or `/usr/local/bin`, reported against the toolchain that built it rather than against a package you declare. Bump whatever npm/pip package ships the binary.
3. If the fix is not viable, add the CVE to [`.trivyignore`](../.trivyignore) with a comment saying why and the PR that decided it. Do not add entries to quiet alerts nobody has reviewed.

## Failure policy

The workflow never fails. `exit-code` is `0` on every scan step regardless of what is found. A weekly job that goes permanently red because of upstream CVEs stops being read within a month, which defeats the point. Notification is the code scanning alerts' job.

A red run therefore means the scan itself broke — a pull failure, a rate limit, disk exhaustion — not that a vulnerability was found.

## When it runs

Weekly, `cron: '10 19 * * 2'` — Wednesday 03:10 UTC+8. You can also run it on demand from the Actions tab (`workflow_dispatch`).

**Treat that time as "no earlier than", not "at".** GitHub delivers scheduled events on a best-effort basis and [documents the top of every hour as a high-load window](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows), which is why the minute is `:10` rather than `:00`. Delays of hours are normal on low-activity repositories. Nothing in this workflow depends on the exact time, so this is a non-issue — but do not debug a "missing" run until it is several hours late.

Two other scheduling facts worth knowing: the `schedule` event only fires for workflow files that exist on the **default branch**, and GitHub disables scheduled workflows in a public repository after 60 days with no activity.

### Forks

**The weekly run is upstream-only.** The job is guarded by `github.repository_owner == 'xorbitsai'`, because the scan targets images that only this org publishes — a fork running it on a schedule would spend its own Actions minutes filling its own alert list with our CVEs, which it can do nothing about. GitHub [already disables scheduled workflows when a public repository is forked](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/disable-and-enable-workflows); the guard also covers a fork that re-enables them.

`workflow_dispatch` is exempt from the guard, so a fork can still run the scan by hand — which is how changes to this workflow get tested without pushing to upstream. A fork that genuinely wants the weekly run should change the `image` values in the matrix to its own images, at which point the guard is the thing to edit too.

## Operational notes

- **Docker Hub rate limits.** The workflow authenticates with `DOCKERHUB_USERNAME` / `DOCKERHUB_PASSWORD` when those secrets exist, because shared Actions runner IPs hit the anonymous per-IP pull limit easily. On forks, which have no such secrets, the login step skips itself and three weekly anonymous pulls stay well inside the limit.
- **Disk.** The backend image unpacks to roughly 10 GB, more than a default runner has free, so `free-disk-space` runs first for that image only.
- **One pull, two scans.** The image is pulled once with `docker pull`; both Trivy steps then read it from the local daemon. Running Trivy twice against the registry would download the backend's 3.6 GB twice.
- **Trivy DB.** Both scans in a job share one on-disk `TRIVY_CACHE_DIR`, so the vulnerability database is fetched at most once per image — that, not the Actions cache, is why the download shows up once per job. The Actions cache is enabled on the first scan as well, but its key is per-day and GitHub evicts entries unused for 7 days, so on a weekly schedule reuse across runs is plausible rather than reliable.
- **Cost.** Zero. Code scanning and GitHub-hosted runner minutes are both free for public repositories.

## Not covered

Released tags such as `0.7.2` are not scanned. Unlike `latest` they are never rebuilt, so they are what users running a pinned version stay exposed to — worth adding once the alert volume from the current three targets is understood.

Also out of scope: the `nightly` tags, the third-party images in `docker-compose.yml` (`nginx:latest`, `postgres:17-bookworm`, `redis:7-alpine`), arm64, and scanning on pull requests.
