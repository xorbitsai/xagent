# Code Review Report: PR #1403

## PR summary
This PR extends `/api/mcp/apps` and the connector picker so team-shared catalog-backed MCP connectors appear alongside personal connections, with explicit `is_team_shared` and `can_attach` states. It also overlays team-owned local MCP/Custom API rows, improves catalog launch matching, and adds focused API coverage; the follow-up head only removes an unused test binding. Blocking: yes — recommended event: REQUEST_CHANGES

## Update since the previous review
Since the previous review at `d3df23cd`, commit `40d1628d` only removes the unused test binding flagged by Ruff F841; it does not change the runtime or API behavior. This review therefore rechecks the full PR scope while carrying forward the unresolved prior findings below.

## Approach verdict and design reservations
**acceptable-with-reservations.** The third-state model is directionally right: `is_connected` remains personal association state, `is_team_shared` records team provenance, and `can_attach` is computed server-side. The implementation also preserves catalog-only remote representation, avoids inventing a team-only `server_id`, and keeps per-user OAuth restrictions explicit.

The main architectural reservation is an implicit scope mismatch: `/api/mcp/apps` computes visibility from the current user's team, while runtime connector resolution uses the governing agent's team, and the picker request sends no agent/team context. The endpoint also necessarily approximates credential availability for coarse cases such as API keys. The scope mismatch is a blocking prior finding below; the coarse API-key contract is recorded as an accepted refactoring rather than an active finding.

## Findings

### CRITICAL

None.

### MAJOR

#### [prior] F1/N1 — PARTIAL — Same-URL team rows are not checked for the `mcp_oauth` auth shape

**Location:** `src/xagent/web/api/mcp.py:2040-2041` (with the downstream attachability predicate at `src/xagent/web/api/mcp.py:1952-1960`).

**Evidence / impact:** The `mcp_oauth` branch of `_team_row_matches_catalog_launch` accepts a row solely when `server.url == launch_config["url"]`; it never requires `_is_mcp_oauth_server(server)`. Consequently, a same-URL `streamable_http` row whose auth is bearer, API-key, or absent can pass the catalog identity check. `_local_mcp_can_attach` then treats that row as a non-`mcp_oauth` shape and returns `True`, so the picker can advertise and attach a catalog identity to a row with different credential semantics. This is the remaining F1/N1 identity/config issue and subsumes the fresh same-URL OAuth candidate; it is not reported twice.

**Recommendation:** Require the expected OAuth transport/auth shape (at minimum `_is_mcp_oauth_server(server)`) in the `mcp_oauth` catalog-match arm before claiming the row. Add a regression with the official URL but a non-`mcp_oauth` auth shape, and verify that it is not marked team-shared or attachable. For non-OAuth arms, enforce the complete launch/credential shape or a persistent catalog identity so legacy rows cannot claim catalog provenance through partial matches.

#### [prior] F3/N3 — NOT FIXED — User-scoped team visibility can diverge from the governing agent's runtime team

**Location:** `src/xagent/web/api/mcp.py:2222-2227`; the picker request is `frontend/src/components/mcp/connect-mcp-dialog.tsx:244-253`; runtime scope is selected from `src/xagent/web/services/connector_runtime.py:600-605` and `:666-671`.

**Evidence / impact:** The endpoint calls `visible_team_connector_ids(db, current_user.id)`, while the picker sends only search/category/location/status parameters and no agent or team identifier. Runtime resolution instead calls the governing-team resolver with `agent.team_id`. A user can therefore select a connector surfaced as team-shared for their own team, persist that selection to an agent belonging to another team (or to no team), and have runtime omit it because it is not visible in the agent's governing scope. The linked issues `#1409` and `#1366` are open tracking items, not a fix for this mismatch.

**Recommendation:** Make listing/attachability agent-aware by passing the selected agent (or governing team) through the picker/API and resolving team visibility with the same runtime scope, or revalidate the selection at persistence against that scope. The scope used to emit `can_attach` must be the scope runtime will actually load.

#### [new] Builtin OAuth team rows ignore resolver-backed credentials in `can_attach`

**Location:** `src/xagent/web/api/mcp.py:2123-2125`; runtime credential precedence is `src/xagent/web/tools/config.py:3203-3267`.

**Evidence / impact:** `_catalog_team_shared_can_attach` first delegates to `_local_mcp_can_attach`, which already receives `token_resolver_installed`, but the `builtin_oauth` branch then returns only whether a usable `UserOAuth` account exists. Runtime resolves every `transport == "oauth"` server through the installed token resolver first and falls back to the legacy `UserOAuth` path. Thus a team-visible builtin OAuth row with a resolver-supplied token and no user account is runnable, while the API emits `can_attach: false` and the picker withholds the attach path.

**Recommendation:** Include resolver-backed credentials in the builtin OAuth attachability gate, using the same documented approximation as the other OAuth path (at minimum the existing resolver-availability signal, or a shared provider-aware credential predicate). Add a regression for a team-shared builtin OAuth row with the resolver installed and no `UserOAuth` account.

### MINOR

#### [prior] F4/P1 — PARTIAL — Legacy team-visibility IDs remain unvalidated at the listing boundary

**Location:** `src/xagent/web/services/connector_team_scope.py:138-143` and `src/xagent/web/api/mcp.py:2237-2244`.

**Evidence / impact:** `visible_team_connector_ids` returns the legacy hook result directly, while the new preload turns those values into an SQL `IN` list and later uses them for Python membership. Numeric strings may be coerced by SQLite but fail against stricter PostgreSQL typing, and `bool` values can alias integer connector IDs. A strict validator exists at `src/xagent/web/services/connector_team_scope.py:145-189`, but this legacy accessor does not invoke it.

**Recommendation:** Route the legacy accessor through the same strict shape/type validation, or enforce equivalent integer-and-not-boolean validation before SQL construction and Python membership. Add coverage for nonnumeric strings and booleans on the supported database backends.

#### [new] Normalized non-OAuth key collisions can hide the valid team row

**Location:** `src/xagent/web/api/mcp.py:1773-1780` and `:2074-2076`.

**Evidence / impact:** `_build_active_non_oauth_server_lookup` stores only the first row for each normalized `(transport, name)` key via `setdefault`. `_team_shared_server_for_app` then retrieves that one candidate and returns `None` immediately when its launch configuration does not match, without examining another row with the same normalized key. A foreign legacy row inserted first can therefore mask an official team-shared row whose raw name collides only after normalization.

**Recommendation:** Index a list of candidates per normalized key, scan candidates against the official launch configuration, and apply a deterministic tie-breaker when more than one valid row remains. Add tests with both insertion orders.

### SUGGESTION

#### [new] Remote preload cardinality scales with every team-visible MCP ID

**Location:** `src/xagent/web/api/mcp.py:2237-2244` (before the remote catalog loop at `:2269-2270`).

**Evidence / impact:** For `location=remote` or `all`, the endpoint loads every team-visible MCP row missing from the user's personal associations before applying catalog search, category, visibility, or status filtering. This is one set-based query rather than an N+1, but its SQL `IN` parameter count, memory, and row materialization scale with all team-visible IDs, including rows that cannot match any catalog app.

**Recommendation:** Narrow the preload to catalog candidate IDs, preferably with a database-side join/subquery or equivalent set-based filter. Preserve the single-query approach; do not replace it with per-app queries.

## Prior finding status checklist

- **F1/N1 — PARTIAL (MAJOR):** The d3df23cd changes reject foreign command/args and foreign URL rows, as covered by `tests/web/api/test_mcp_apps_team_catalog.py:407-456`, but the same-URL wrong-auth case remains at `src/xagent/web/api/mcp.py:2040-2041`. The fresh same-URL candidate is deduplicated into this status.
- **F2/N2 — REFACTORED (non-blocking; accepted coarse contract):** API-key attachability is intentionally not credential-gated because the governing agent's team environment is outside this user-scoped endpoint, as documented at `src/xagent/web/api/mcp.py:2105-2113`; the required-`API_KEY`/no-known-source behavior is pinned by `tests/web/api/test_mcp_apps_team_catalog.py:379-404`. It is not an active finding.
- **F3/N3 — NOT FIXED (MAJOR):** User-scoped visibility is still computed at `src/xagent/web/api/mcp.py:2222-2227`, the picker still sends no agent/team context at `frontend/src/components/mcp/connect-mcp-dialog.tsx:244-253`, and runtime still resolves `agent.team_id` at `src/xagent/web/services/connector_runtime.py:600-605,666-671`. Open `#1409`/`#1366` do not resolve the current mismatch.
- **F4/P1 — PARTIAL (MINOR):** The legacy raw accessor remains at `src/xagent/web/services/connector_team_scope.py:138-143`, and its values feed the new SQL preload at `src/xagent/web/api/mcp.py:2237-2244`; strict validation is present only in the separate validator at `:145-189`.
- **F5/N5 — FIXED:** Team IDs and missing MCP rows are fetched once at `src/xagent/web/api/mcp.py:2227-2244`, reused to build remote indexes at `:2257-2266`, and reused by the local overlay at `:2410-2411`. The remaining preload-cardinality concern is the separate suggestion above, not a duplicate-preload finding.
- **F6/N6 — FIXED:** `tests/web/api/test_mcp_apps_team_catalog.py:379-404` covers the required `API_KEY`/no-source case and asserts the intentional optimistic attachability contract.
- **F7/N7 — FIXED:** `tests/web/api/test_mcp_apps_team_catalog.py:628-663` covers both team-shared and personal-only Custom API rows and their `is_team_shared` values.

## Testing, review limitations, and criteria coverage

- **Testing and limitations:** No local tests were run, per the review workflow. Preflight CI status checks were observed completed successfully with no `FAILURE`, `ERROR`, `TIMED_OUT`, or `CANCELLED` conclusion, but the GitHub `reviewDecision` was `CHANGES_REQUESTED`. The changed tests were reviewed statically; coverage is still missing for resolver-backed builtin OAuth attachability and normalized-candidate insertion-order collisions.
- **Correctness:** Reviewed personal/team overlays, catalog identity and launch matching, attachability, status filtering, and local/remote branch behavior. The F1, F3, builtin OAuth, and collision findings are the remaining correctness risks.
- **Security:** Reviewed connector identity, OAuth credential shape/grant gates, malformed team-ID handling, and team visibility boundaries. F1, F3, and F4 can respectively expose the wrong credential semantics, apply the wrong team authorization scope, or fail on malformed authorization input.
- **Performance:** Reviewed query count and preload reuse. F5 is fixed; the remaining all-team-ID materialization cost is recorded as a suggestion rather than an N+1 defect.
- **Code quality/architecture:** The explicit three-state response model and shared attachability helpers are coherent, with the scope boundary noted in the approach reservation. No separate style or simplification finding is reported.
- **Tests:** The new suite covers the accepted API-key behavior, launch mismatch cases, and Custom API shared/personal states; it does not cover the two new regression scenarios called out above.
- **Docs/API:** The `is_team_shared`/`can_attach` response contract is represented in `frontend/src/components/mcp/types.ts` and consumed by `frontend/src/components/mcp/connect-mcp-dialog.tsx`. The missing agent/team context is an API contract issue covered by F3; no separate documentation-only finding is reported.

## Blocking status & recommended decision

**Blocking: yes — recommended event: REQUEST_CHANGES**

Blocking issues:

- `src/xagent/web/api/mcp.py:2040-2041` — **MAJOR** — Same-URL team rows can still claim an `mcp_oauth` catalog identity without the required OAuth auth shape. `[prior]`
- `src/xagent/web/api/mcp.py:2222-2227` (picker `frontend/src/components/mcp/connect-mcp-dialog.tsx:244-253`; runtime `src/xagent/web/services/connector_runtime.py:600-605,666-671`) — **MAJOR** — User-scoped team visibility can be dropped when the selected agent's governing team differs. `[prior]`
- `src/xagent/web/api/mcp.py:2123-2125` — **MAJOR** — Builtin OAuth rows are reported unattachable when runtime can obtain their token from the resolver without a `UserOAuth` account. `[new]`
