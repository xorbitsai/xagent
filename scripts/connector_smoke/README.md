# Connector smoke checks

Black-box HTTP checks for connectors on a deployed Xagent instance, tracked in
[#3244](https://github.com/xorbitsai/xagent/issues/3244). They log in as a
dedicated low-privilege monitor account and call only public routes. They never
run an LLM agent.

## L1: configuration smoke

L1 needs no vendor accounts. It checks two things.

**Catalog.** `GET /api/mcp/apps` must list every expected app, and each one must
be connectable (its `auth_type` is not `unconnectable`). Any app the catalog
lists that is not expected also fails. The expected set is:

- every app that the builtin registry (`src/xagent/web/builtin_mcp_registry.py`)
  shows by default,
- minus the apps this deployment hides (`--exclude`),
- plus the apps this deployment shows beyond that default (`--include`).

New connectors are picked up from the registry without editing any list. An app
an administrator created is not in the registry, so it fails as unexpected
until you pass it with `--include`.

The registry decides Gmail's and full Google Drive's default visibility from
`XAGENT_GOOGLE_RESTRICTED_SCOPES_ENABLED` in the environment of the process
running the check, not the deployment's. Leave it unset where the check runs.
Against a deployment that has the gate open, pass `--include gmail,google-drive`.

**Builtin OAuth providers.** The check launches each provider once, through one
of its visible catalog apps (a registry app when there is one):

1. `POST /api/auth/{provider}/login` starts the launch. The returned callback
   origin must equal the environment's origin. An unset `<PROVIDER>_REDIRECT_URI`
   falls back to `http://localhost:8000`, so it fails here.
2. `GET` on the returned launch path redeems it, without following the
   redirect. The provider `Location` must carry a non-empty `client_id` and a
   `redirect_uri` of `<origin>/api/auth/{provider}/callback`.

Remote MCP OAuth apps (Notion, Atlassian, HubSpot, …) are not launched, because
their connect may register a client with the vendor on every call. L2 covers
them.

L1 cannot detect a missing client secret: the server first reads the secret in
the OAuth callback, after the vendor's consent screen. Missing secrets are
covered by `python -m xagent.web.oauth_readiness` instead
([#3248](https://github.com/xorbitsai/xagent/issues/3248)). Run it in the
backend container through the image entrypoint, which loads `ENCRYPTION_KEY`
from its key file the same way it does for the server:

```bash
docker compose exec -T backend /opt/xagent/deploy/entrypoint.sh \
  python -m xagent.web.oauth_readiness
```

It prints one line per provider saying whether the client ID and secret
resolve, and from the `oauth_providers` row or the environment, never a value.
A row encrypted under another key fails as `unreadable`; a process with no
usable key fails once as `encryption-key`. It proves a value is present, not
that the vendor accepts it. Its `--exclude` takes OAuth provider names
(`google`), not the catalog app IDs this script's `--exclude` takes.

### Side effects

- Starting a launch writes nothing.
- Redeeming one stores a single-use ticket record (`oidc_consumed_tokens`,
  `oauth-launch:<id>`) that expires after 30 seconds. Every redeem deletes the
  records that have already expired, so they do not pile up across runs: a run
  more than 30 seconds after the previous one leaves one row per provider.
- Logging in replaces the monitor account's refresh token. Give the check its
  own account; do not share one that another client keeps a session for.

### Usage

Run from the repository root:

```bash
export XAGENT_CONNECTOR_SMOKE_LOGIN=monitor@example.com
export XAGENT_CONNECTOR_SMOKE_PASSWORD=...
PYTHONPATH=src:scripts python -m connector_smoke l1 --base-url https://xagent.example.com
```

On Xagent SaaS, pick the region with a cookie, start the login through
`login/start`, and log in by email. `selected_region` is sent on every request,
and `login_route` is sent only on the login request:

```bash
PYTHONPATH=src:scripts python -m connector_smoke l1 \
  --base-url https://cloud.xagent.co \
  --cookie selected_region=sg \
  --login-start --login-field email \
  --oauth-account zendesk=<subdomain> --include zendesk
```

| Option | Environment variable | Meaning |
| --- | --- | --- |
| `--base-url` | `XAGENT_CONNECTOR_SMOKE_BASE_URL` | Deployment base URL. Required. |
| (none) | `XAGENT_CONNECTOR_SMOKE_LOGIN`, `XAGENT_CONNECTOR_SMOKE_PASSWORD` | Monitor account. Accepted only from the environment. |
| `--callback-origin` | `XAGENT_CONNECTOR_SMOKE_CALLBACK_ORIGIN` | Origin every OAuth callback must use. Defaults to the base URL's origin. |
| `--exclude` | `XAGENT_CONNECTOR_SMOKE_EXCLUDE` | Registry apps this deployment hides. Each one must be shown by the registry by default. |
| `--include` | `XAGENT_CONNECTOR_SMOKE_INCLUDE` | Apps this deployment shows beyond the registry default. |
| `--cookie NAME=VALUE` | `XAGENT_CONNECTOR_SMOKE_COOKIES` (`;`-separated) | Cookies sent on every request. |
| `--oauth-account PROVIDER=LABEL` | `XAGENT_CONNECTOR_SMOKE_OAUTH_ACCOUNTS` | Account label for a provider hosted on the customer's own account, for example Zendesk. The label must resolve, so use a real account. |
| `--login-field` | (none) | `username` (default) or `email`. |
| `--login-start` | (none) | Call `/api/auth/login/start` before logging in. |

The list options are repeatable or comma-separated. A value given on the command
line replaces the environment variable's value; it does not add to it.

### Output

The check prints one line per check, then a summary:

```text
PASS catalog:slack
PASS oauth:slack:callback
FAIL oauth:slack:authorize: no provider redirect: HTTP 500 (Error: OAuth provider not configured - Missing SLACK_CLIENT_ID for provider slack.)
42 checks, 1 failed
```

The exit code is `0` when every check passes, `1` when any check fails, and `2`
for a usage error. The output never contains tokens, tickets, client IDs, or
redirect URLs.

### Tests

```bash
pytest tests/scripts/test_connector_smoke_l1.py tests/scripts/test_connector_smoke_cli.py
```

The tests use recorded response shapes and an in-process fake deployment. None
of them opens a network connection.

## L2: connection smoke

L2 starts every catalog connector the monitor account has connected and
requires each one to list at least one tool
([#3246](https://github.com/xorbitsai/xagent/issues/3246)). It calls
`GET /api/mcp/servers/{id}/tools`, which starts the server exactly as an agent
task would: the stdio launch with the user's token for builtin OAuth, the env
layer the user picked for API-key connectors, and the user's grant for remote
MCP OAuth (`src/xagent/web/services/mcp_connector_session.py`).

**Which connectors.** Every app that `GET /api/mcp/apps` reports as connected
(`is_connected` with a `server_id`) is started, whether or not it is expected,
so a connector the account gains is checked without editing a list. Custom MCP
servers and custom APIs are not catalog apps and are not started. Pass the apps
each environment must have connected with `--expect`. An expected app that is
not connected fails as `needs reconnect`, `not connected`, or `not in the
catalog`. A builtin OAuth app whose stored token was cleared (`connection_status`
is `needs_reconnect` in `GET /api/mcp/servers`) fails as `needs reconnect` even
when it is not expected. Any other connector the account loses drops out of the
run (for example a builtin OAuth token that expired with no refresh token,
which `GET /api/mcp/servers` still reports as `connected`), so the scheduled
job should always pass `--expect`. A run that finds no
connected connector and expects none fails.
Vendor test accounts for the monitor account are tracked in
[#3262](https://github.com/xorbitsai/xagent/issues/3262).

**What a pass proves** depends on the connector type:

| Type | A pass proves |
| --- | --- |
| Remote MCP OAuth (Notion, HubSpot, Atlassian, …) | The credential works: `tools/list` reaches the vendor's server, and the grant is refreshed when it expires within 5 minutes. A vendor that rejects the token (HTTP 401/403) fails as `not connected (auth_rejected)`. |
| Builtin OAuth (Google, Microsoft, Slack, GitHub, …) | The connector starts with a token injected; an expiring token is refreshed first. The in-repo servers' tool lists are static, so a token revoked at the vendor still passes. Credential validity is L3's ([#3247](https://github.com/xorbitsai/xagent/issues/3247)). |
| API key (Stripe, AWS, PostHog, …) | The connector starts with the user's env resolved. The key is read only inside tool calls, so its validity is L3's. |
| `chrome-devtools` | Not covered. It is hidden from the catalog, so the monitor account cannot connect it, and it can only start inside a task (`unsupported outside a task`). |

### Side effects

- Each connector runs once, one at a time. A stdio connector starts a process on
  the backend host for the length of one `tools/list`; an `npx` connector runs
  in a sandbox when the deployment has a sandbox manager. A load the server
  abandons at its timeout may still be cleaned up in the background while the
  next connector starts.
- A builtin OAuth token that expires within 5 minutes is refreshed, and the new
  token is stored, as a task would do. A refresh token the vendor has revoked
  permanently is cleared, so that connector then needs a reconnect.
- Logging in replaces the monitor account's refresh token, as in L1.

### Run time

Each connector takes as long as its server needs to start and answer
`tools/list`, usually a few seconds. A connector that hangs costs at most
`--tools-timeout` (120 s by default). That default sits above the server's bound
on one answer: it responds within `XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS` (60 s
by default) of starting the load, after the token refresh and, for an `npx`
connector, the sandbox lease. A shorter client timeout gives up on a connector
the server is still starting. If the deployment raises
`XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS`, raise `--tools-timeout` by the same
amount. The worst case is the number of
connected apps times `--tools-timeout`: 50 connectors that all hang take 100
minutes. The access token from the login lasts
`XAGENT_ACCESS_TOKEN_EXPIRE_MINUTES` (120 by default), so raise it for the
monitor account's deployment if a run can take longer. The summary line reports
the elapsed time.

### Usage

```bash
export XAGENT_CONNECTOR_SMOKE_LOGIN=monitor@example.com
export XAGENT_CONNECTOR_SMOKE_PASSWORD=...
PYTHONPATH=src:scripts python -m connector_smoke l2 \
  --base-url https://xagent.example.com \
  --expect slack,notion,stripe
```

L2 takes the same `--base-url`, `--cookie`, `--login-field`, `--login-start` and
`--timeout` options as L1, plus:

| Option | Environment variable | Meaning |
| --- | --- | --- |
| `--expect` | `XAGENT_CONNECTOR_SMOKE_EXPECT` | Apps the monitor account must have connected. Repeatable or comma-separated. |
| `--tools-timeout` | (none) | Seconds to wait for one connector to start and list its tools. Default 120. |

`--timeout` applies to the login, catalog and server-list requests only.

### Output

```text
PASS l2:notion: 14 tools
FAIL l2:slack: failed to start (initialize)
FAIL l2:stripe: not connected (oauth_token_required)
FAIL l2:gmail: needs reconnect
4 checks, 3 failed in 131s
```

| Failure | Meaning |
| --- | --- |
| `not connected (<code>)` | The server answered 400 with a credential code (`oauth_token_required`, `authorization_required`, `insufficient_scope`, `token_refresh_failed`, …), or the vendor rejected the token (`auth_rejected`). |
| `needs reconnect`, `not connected`, `not in the catalog` | An expected app is not connected; see above. |
| `failed to start (<phase or reason>)` | The server could not be started or did not answer the MCP handshake. The phase is where the loader gave up (`src/xagent/core/tools/adapters/vibe/mcp_adapter.py`): `session_start` when the session could not be opened (for example the command could not be spawned), `initialize` when the process exited or never answered the handshake (a missing Python module ends here), `list_tools`, `sandbox_list_tools` (an `npx` connector that fails or hangs in the sandbox ends here). A code such as `sandbox_unavailable`, or a remote OAuth discovery code such as `metadata_not_found`, means the server was never started. A host (non-`npx`) connector that hangs past the server's own load bound ends as `initialize`: the route reports the phase, not that it timed out. |
| `no tools` | The server started and listed no tools. |
| `timed out after <n>s` | No answer within `--tools-timeout`. The server gives up on a hung connector before this, so this means the request itself stalled, for example in the token refresh that runs before the load. |
| `unsupported outside a task` | An execution-scoped connector; see the table above. |
| `server not found` | The server row disappeared between the catalog and the call (HTTP 404). |
| `<METHOD> request failed: <Class>` | No HTTP answer, for example a connection error. |
| `unexpected response (HTTP 200)` | The server answered 200 with a body that is not a JSON object. |
| `HTTP <status>` | Any other answer, for example `HTTP 504` from a reverse proxy whose idle timeout is shorter than the server's load bound. |

Two checks are not about one connector. `FAIL servers: ...` means the server
list request failed; expected apps are then reported without the `needs
reconnect` distinction. `FAIL connectors: no connected connector to check`
means the run found nothing to start and expected nothing.

Reasons are the codes and failure phases the server reports. The route does not
return exception class names, and the output never contains a token, a key, a
diagnostic message, or a server's error text. The exit codes are the same as
L1's.

### Tests

```bash
pytest tests/scripts/test_connector_smoke_l2.py
```

## L3: tool-call smoke

L3 runs one read-only tool call against the vendor for every catalog connector
the monitor account has connected
([#3247](https://github.com/xorbitsai/xagent/issues/3247)). It is the only layer
that proves a builtin OAuth or API-key credential works, because those
connectors' `tools/list` is static and never reaches the vendor. It calls
`POST /api/mcp/servers/{id}/probe`, which starts the server the way L2 does and
then calls the one tool declared for that connector in
`src/xagent/web/connector_probes.py`, with the arguments declared there
(`src/xagent/web/services/connector_probe.py`). The script sends only the server
id: it cannot choose the tool or its arguments.

**Which connectors.** The same apps as L2: every connected catalog app, plus
`--expect` and `needs reconnect` reporting exactly as in L2. A connector with no
probe is reported as `SKIP` and is not a check:

| Skip | Connectors | Why |
| --- | --- | --- |
| `unprobeable (covered_by_l2)` | Remote MCP (Notion, HubSpot, Atlassian, …) | Their `tools/list` reaches the vendor, so L2 already proves the credential. |
| `unprobeable (sandboxed)` | `google-maps` | An `npx` connector; the call would run on the backend host after the listing released its sandbox. |
| `unprobeable (execution_scoped)` | `chrome-devtools` | It can only run inside a task. |
| `unprobeable (needs_fixture)` | Google Docs, Sheets, Slides; Word, Excel, PowerPoint | Every read-only tool needs a document the monitor account would have to own. |
| `no probe declared` | A custom catalog app | Only builtin apps declare probes. The route must name the app itself; see `not the catalog's row` below. |

A run that probes nothing, because every connected app was skipped or none is
connected, fails.

**Account.** The deployment must name the monitor account in
`XAGENT_CONNECTOR_PROBE_USERS` (username, or email compared case-insensitively;
see `example.env`). The account needs no admin flag. Otherwise the route refuses
it, and the run stops at the first connector the route answers with one
failure.

### Side effects

- Each probe is one read-only, idempotent vendor call that asks for the
  smallest page the tool accepts, for example `github_get_current_user`,
  `slack_list_channels` with `limit=1`, `stripe_list_customers` with
  `limit=1`. Nothing is written at the vendor. Intercom's probe is a read
  through Intercom's `POST /conversations/search`.
- The connector is started exactly as in L2, including the token refresh, so
  L2's side effects apply.
- The server runs one probe per web process at a time. A probe that overlaps
  another (two runs at once) fails as `probe busy`. So can the connector after
  one that hit `--probe-timeout`: the server keeps running the probe the client
  gave up on.
- The backend logs one `No user ID found in environment` warning per probe:
  the in-repo tools read the caller only from a task's environment.

### Run time

A probe costs a listing plus one call, usually a few seconds. The server bounds
each by `XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS` (60 s by default), so
`--probe-timeout` defaults to 180 s. If the deployment raises the server bound,
raise `--probe-timeout` by twice the same amount. The worst case is the number
of probed apps times `--probe-timeout`; a skipped app answers before anything
starts. The 20 Phase 1 connectors in
[#3262](https://github.com/xorbitsai/xagent/issues/3262) that have a probe take
60 minutes if every one hangs, within the 120-minute default of
`XAGENT_ACCESS_TOKEN_EXPIRE_MINUTES`.

### Usage

```bash
export XAGENT_CONNECTOR_SMOKE_LOGIN=monitor@example.com
export XAGENT_CONNECTOR_SMOKE_PASSWORD=...
PYTHONPATH=src:scripts python -m connector_smoke l3 \
  --base-url https://xagent.example.com \
  --expect github,slack,stripe
```

L3 takes the same common options as L2, plus `--expect` (as in L2) and:

| Option | Environment variable | Meaning |
| --- | --- | --- |
| `--probe-timeout` | (none) | Seconds to wait for one connector to start and answer its probe. Default 180. |

### Output

```text
PASS l3:github: github_get_current_user
SKIP l3:notion: unprobeable (covered_by_l2)
FAIL l3:slack: unexpected probe result (status_not_success)
FAIL l3:stripe: not connected (oauth_token_required)
3 checks, 2 failed, 1 skipped in 12s
```

A pass names the tool that answered. Failures that are not L2's:

| Failure | Meaning |
| --- | --- |
| `unexpected probe result (status_not_success)` | The tool ran and reported an error in its own result, which is how the in-repo servers report a vendor error: a revoked or expired token, a missing scope, an API change. Other reasons: `not_a_json_object`, `no_status`, `missing_keys` (the result had an unexpected shape). |
| `probe failed (<reason>)` | `tool_error`: the call came back flagged as an MCP error. `exception`: the call raised. `invalid_result`: the adapter returned something that is not a result. |
| `probe tool missing (<tool>)` | The server started but does not offer the declared tool. |
| `timed out on the server (<phase or probe_call>)` | The server gave up on the listing or, with `probe_call`, on the call itself. |
| `not the catalog's row for this app (no probe)` | The catalog lists the app as connected, but the server row is not the catalog's own: a custom server claiming the app, a user-owned row under the app's id, or a row whose command or URL no longer matches the catalog. The route declines to probe it. |
| `cannot be probed: runs in a sandbox` | The row's effective command is one a task runs in a sandbox (`npx`, `uvx`); the call would run outside it. |
| `probe busy (another probe, or one that timed out, still runs)` | Another probe was running in the same web process, possibly the previous connector's after a client timeout. |
| `FAIL probe: this account may not run connector probes (XAGENT_CONNECTOR_PROBE_USERS)` | The account is not on the deployment's probe list. The run stops. |
| `HTTP 404` on every connector | The deployment predates the probe route. |

`not connected (<code>)`, `failed to start (<phase or reason>)`, `no tools`,
`unsupported outside a task (<reason>)`, `server not found`,
`timed out after <n>s` (here against `--probe-timeout`) and the request errors
mean what they mean in L2. As in L2, reasons are codes only: the output never
contains a token, a key, or a tool's or vendor's error text.

### Tests

```bash
pytest tests/scripts/test_connector_smoke_l3.py
```

## Startability check: every connector, no accounts

L1 proves a connector is listed and its OAuth client is configured; it cannot
tell whether the connector starts. `python -m xagent.web.connector_readiness`
([#3264](https://github.com/xorbitsai/xagent/issues/3264)) does, for every
`public_mcp_apps` row of the deployment, hidden rows included, without any
vendor account. Run it in the backend container the same way as
`oauth_readiness`:

```bash
docker compose exec -T backend /opt/xagent/deploy/entrypoint.sh \
  python -m xagent.web.connector_readiness
```

What it checks depends on the row's `launch_config`:

| Shape | Check |
| --- | --- |
| `command` (`python -m xagent.web.tools.mcp.<app>`) | Spawns the command with a placeholder for every `env_mapping` and `required_env` key, runs MCP `initialize` and `tools/list`, and requires at least one tool. The in-repo servers' tool lists are static, and the check stops after `tools/list`, so the placeholder never reaches a vendor. |
| `command` `npx`/`uvx` | `SKIP sandboxed` when `SANDBOX_ENABLED` is on, because tasks run these in the sandbox image, not the backend image. With the sandbox off, spawned as above with npm and uv forced offline, so a package missing from the image's cache fails instead of being fetched from a registry. |
| `url` (remote MCP, e.g. Notion, Atlassian, HubSpot) | One unauthenticated `GET` through the runtime's MCP HTTP client, so the probe sees the same proxy and egress policy a task does. Any HTTP response passes. No OAuth flow is started. |
| `url` with a `ws://`/`wss://` scheme | `SKIP websocket`; websocket rows are not probed. |

It prints one line per connector, labelled `builtin` (a registry app) or
`custom` (an administrator-created row), plus `hidden` for a row the catalog
does not show, then a summary:

```text
PASS connector:slack builtin: 11 tools
SKIP sandboxed connector:chrome-devtools builtin hidden: tasks run npx/uvx rows in the sandbox image, not this one
SKIP sandboxed connector:google-maps builtin: tasks run npx/uvx rows in the sandbox image, not this one
FAIL missing connector:xero builtin: /usr/local/bin/python: No module named xagent.web.tools.mcp.xero
FAIL unreachable connector:notion builtin: ConnectError: [Errno -3] Temporary failure in name resolution
51 checks, 2 failed, 2 skipped
```

A failure's class is one of `spawn` (command not found, or a row with neither
a command nor a url), `missing` (the child exited before the MCP handshake: a
Python import error, or an npm package absent from the offline cache),
`handshake` (no usable `initialize`/`tools/list` reply within the timeout, an
error reply, or a connector not reached before the total bound), `no-tools`, or `unreachable`. The
detail is the last line of the child's stderr or the exception, never the
environment. When the database cannot be read, `FAIL database: <ExceptionClass>`
is printed once and the run exits `1`; when no row is checked (an empty
catalog, or every row excluded), `FAIL catalog: no public_mcp_apps rows to
check` is printed and the run exits `1`. The exit code is also `1` when any
connector fails and `2` for a usage error.

Each connector is bounded by `XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS` (60 s by
default, the same bound task startup uses) and the whole run by
`--total-timeout`, which defaults to the connector count times that value; a
connector not reached in time fails as `handshake`.
`XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS=0` disables both bounds, and
`--total-timeout 0` disables the total bound. `--exclude` takes catalog app
IDs, like this script's `--exclude`, for a connector the deployment leaves
unstartable on purpose. On a developer machine the 44 python-module rows took
12 seconds in total; an npx row that resolves offline and a remote probe add a
few seconds each.

What a pass proves: the module or package is present in the backend image and
answers the handshake, or the remote host answers HTTP from the backend
network. It does not prove a credential works (L3) or that an `npx` row
starts in the sandbox image (#1692).

```bash
pytest tests/web/test_connector_readiness.py
```

The tests spawn fake stdio servers with the test interpreter and replace the
HTTP client; none of them opens a network connection.
