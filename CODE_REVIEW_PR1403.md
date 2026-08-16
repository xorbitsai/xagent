# Code Review: PR #1403

## PR overview

- **Title:** `fix(mcp): offer a team-shared catalog connector in the connector picker`
- **Author:** `codeacme17`
- **Intent:** Make a catalog-backed MCP connector shared through the team visibility layer appear once in the connector picker without presenting it as personally connected, while preserving per-member OAuth credentials and the existing standalone behavior.
- **Actual scope:** Five files, 818 additions and 30 deletions: three frontend files, `src/xagent/web/api/mcp.py`, and the new `tests/web/api/test_mcp_apps_team_catalog.py` backend test file.
- **Base:** `main` at `9fb08960a799c6f968675ada3fd02fb28ef70c91`
- **Head:** `0f82d1dcdc84f5f004c8852daee02fa89c8960e6`
- **CI/local testing:** CI preflight passed with no failing checks. No local tests were run, per the review workflow.

## PR summary

PR #1403 adds a team-visibility overlay to `GET /api/mcp/apps`, matching visible `MCPServer` rows back to catalog applications so a team-shared connector can be represented by one catalog/Remote card rather than a duplicate custom row. The frontend uses the new provenance and attachability fields to show a Team tool and either select an attachable connector or leave an OAuth connector on its personal Connect path. The overall direction is useful, but catalog identity, static-credential availability, and target-agent scope are not reliably represented by the new `can_attach` contract.

Blocking: yes — recommended event: REQUEST_CHANGES

## Approach verdict

**Verdict: acceptable-with-reservations.** The single catalog representation plus explicit `is_team_shared` provenance addresses the cross-layer discovery problem and avoids reviving the duplicate `is_custom` representation that caused the earlier picker issue. However, the implementation stretches heuristic name matching and an unscoped user-level visibility result into an executable catalog identity and runtime capability; those boundaries need to be corrected before relying on the picker as an attachability decision.

### Design and control-flow trace

- `PublicMCPApp` stores catalog metadata and launch configuration, while `MCPServer` stores the executable server row. These models do not have a persistent catalog foreign key. Non-OAuth association therefore relies on normalized catalog/name and transport matching; OAuth-specific matching uses `app_id`/provider information only in the paths that explicitly perform those checks.
- `GET /api/mcp/apps` loads the member's `UserMCPServer` and OAuth state, resolves team connector IDs through the visibility hook, preloads missing team `MCPServer` rows, and builds team lookup indexes. In the catalog branch, a matched team row controls `is_team_shared` and participates in the new team attachability predicate. In the local/all branch, normalized catalog keys suppress a duplicate custom representation.
- The picker consumes `can_attach` directly: an attachable card is selected into the agent, while an unattachable catalog OAuth card remains discoverable and can open its personal Connect flow. This makes the backend boolean a capability contract, not merely display metadata.
- The actual runtime is not resolved solely from the current user's visible team IDs. With the modern team hook installed, runtime connector visibility and team environment are resolved from the governing agent's team; a personal agent has no governing team. Runtime environment construction separately combines global, shared, team, and personal layers. The new list endpoint does not receive the target agent/team context or all of the credential state needed to predict that result.

The central design reservations are reflected in N1–N3 below. Apart from those boundaries, the helper split, backend-owned attachability decision, and preservation of personal OAuth semantics are coherent.

## Findings

### Critical

No critical findings confirmed.

### Major

#### N1 [new] — A normalized name/transport match can turn a custom row into an official catalog identity

- **Location:** `src/xagent/web/api/mcp.py:2038` (with related matching at `1859-1880` and local deduplication at `2361-2390`)
- **Severity:** major
- **Evidence / impact:** `_team_shared_server_for_app` routes non-builtin applications through `_non_oauth_server_for_app`. That resolver compares the catalog transport and normalized application/name keys, but does not verify an immutable catalog identity, stdio command/args, HTTP URL/auth, ownership, or a persisted catalog foreign key. A pre-existing custom or legacy `MCPServer` with the same normalized name and transport can therefore be returned by the visibility hook, displayed with the official catalog name/icon/category, marked `is_team_shared`, and made attachable. The local normalized-key suppression then hides the real custom representation. When the picker resolves the selected name back to the server row, runtime can execute the custom row's command, URL, arguments, or environment under the official catalog card. This is an identity/configuration-integrity failure and can cause the wrong authorized connector or credential domain to run; it is not merely a cosmetic collision.
- **Recommendation:** Give `MCPServer` a persistent, immutable catalog identity/FK and use it consistently for provisioning, listing, attachability, selector resolution, and deduplication. If a schema migration is not part of this change, require shape-specific configuration equivalence before treating a team row as catalog (including stdio command/args, HTTP URL/auth, and OAuth app/provider fields), reject user-owned mismatches, and retain them as custom/unattachable rows. Add a regression for a pre-existing same-normalized-name/same-transport row whose configuration differs.

#### N2 [new] — A team-only `api_key` catalog entry can be marked attachable without a usable key

- **Location:** `src/xagent/web/api/mcp.py:2041-2076`, especially `2074-2075` (consumed at `2302-2312`)
- **Severity:** major
- **Evidence / impact:** After `_local_mcp_can_attach` succeeds, `_catalog_team_shared_can_attach` returns `True` for every shape other than `builtin_oauth`. The delegated local predicate credential-gates `mcp_oauth`, but it does not validate static `api_key` required environment variables. Catalog API keys are persisted in the connecting member's `UserMCPServer.env`; they are not automatically present in the shared `MCPServer` row. The visibility hook is also independent of the shared/team credential hooks, so a visibility-only deployment or a team with no required environment can still produce `is_team_shared=true, can_attach=true`. The picker then selects the connector directly, while the runtime receives no required key and the stdio process cannot run correctly.
- **Recommendation:** Make static-auth attachability depend on the actual target runtime credential scope. For `api_key` applications, return `true` only when every `launch_config.required_env` key is available from the governing team/shared/platform/personal layers that the target runtime will use; visibility-hook presence alone must not satisfy the check. Add coverage for visibility-only, missing/partial credentials, complete team credentials, complete platform credentials, and credential-hook failure/empty responses.

#### N3 [new] — User-level team visibility is used as if it were target-agent runtime scope

- **Location:** `src/xagent/web/api/mcp.py:2178` and `2302-2312`
- **Severity:** major
- **Evidence / impact:** The endpoint accepts search/category/location/status filters but no target agent or team context. It obtains `team_ids` from the current user's `visible_team_connector_ids` at line 2178 and uses those IDs to produce `can_attach`. The frontend likewise requests `/api/mcp/apps` without an agent/team identifier. Runtime, however, uses the governing agent's `agent_team_id` with the modern team hook; a personal agent has `None` and receives no team overlay, while another team agent can have a different connector set. Thus the same member can receive `can_attach=true` for a personal agent, a different team agent, and the current team agent even though runtime visibility differs. The save paths do not provide a complete authoritative check for arbitrary personal-agent selections, so a connector can be saved and then silently omitted by runtime, or a runtime-visible connector can be omitted from the picker.
- **Recommendation:** Define attachability as a target-scoped capability. Pass the target agent or governing team ID to the picker/list request, or provide a dedicated capability check that reuses the runtime scope resolver and credential scope. Keep an authoritative save/runtime validation for scope changes and races; without target context, do not advertise a user-visible team connector as unconditionally attachable.

### Minor

#### P1 [prior, PARTIAL] — Malformed legacy team-ID answers remain unvalidated, with mixed-type duplicate loads and boolean hazards

- **Location:** `src/xagent/web/api/mcp.py:2178-2204`, especially `2197` and `2202` (membership checks at `2259-2264`)
- **Severity:** minor
- **Evidence / impact:** `team_ids` is assigned directly from the legacy visibility hook without element validation or normalization. The code casts database IDs to `int` for Python membership but passes the hook's raw values into `MCPServer.id.in_(...)`. Round 2 did not reproduce the original high-severity claim that a numeric string alone necessarily raises a PostgreSQL type error: the normal SQLAlchemy/psycopg path sends numeric text that PostgreSQL can coerce, and a string-only set does not itself duplicate a row. The residual issue is still real: a mixed `{1, "1"}` answer leaves the string in `missing_team_mcp` and can load the same row again, while a boolean can either bind as an invalid integer comparison or alias ID `1` through Python equality. The hook boundary also accepts other malformed representations. This is integration robustness, not the original blocking database-failure claim.
- **Recommendation:** Validate the legacy hook response at its boundary using the same shape checks as the team-keyed hook: reject strings, booleans, non-set containers, and preferably non-positive/non-integer IDs, or normalize to a set of exact positive integers once and reuse it for Python predicates and SQL `IN` clauses. Add coverage for string-only, boolean, negative, mixed int/string, and duplicate representations. P1 remains partially valid and is not fixed.

#### N5 [new] — Team MCP rows are prefetched and then queried again

- **Location:** `src/xagent/web/api/mcp.py:2190-2207` and `2349-2359`
- **Severity:** minor
- **Evidence / impact:** The new team lookup is built before the location branch. For `location=local`, the lookup is not used, so the request pays for an unnecessary team query and index construction. For `location=all`, the local branch issues another `MCPServer.id.in_(...)` query over the same missing team IDs instead of reusing the rows already loaded for the remote catalog branch. SQLAlchemy may return the same Python instances through its identity map, but it still incurs the second SQL round trip, `IN` list handling, and row processing. The impact is proportional to visible team connector count and does not change response semantics.
- **Recommendation:** Construct the team indexes/query only for `location=remote` or `location=all`. For `location=all`, reuse the preloaded team rows when constructing `local_mcps`; for `location=local`, retain only the original local query.

#### N6 [new] — The core team-catalog test covers only keyless apps, not API-key attachability

- **Location:** `tests/web/api/test_mcp_apps_team_catalog.py:271-289`
- **Severity:** minor
- **Evidence / impact:** The test uses `_add_keyless_app`, whose launch configuration has no `required_env`, and creates no shared/team/platform credential state. It therefore cannot fail if the production helper incorrectly returns `can_attach=true` for an `api_key` application with no key. Existing API-key tests cover personal environment persistence or standalone availability helpers, but do not exercise the team catalog attachability path.
- **Recommendation:** Add a team-shared catalog app with nonempty `required_env` and assert `can_attach=false` without credentials, `true` with complete team/shared credentials, and `true` with complete platform credentials. Cover partial keys and credential-hook failure/empty responses, and have at least one case build the actual runtime configuration and assert that required environment variables are present.

#### N7 [new] — The every-entry `is_team_shared` contract test omits the Custom API branch

- **Location:** `tests/web/api/test_mcp_apps_team_catalog.py:504-533` (production field at `src/xagent/web/api/mcp.py:2520`)
- **Severity:** minor
- **Evidence / impact:** The test creates only shared and personal `MCPServer`/`UserMCPServer` rows and returns an empty Custom API visibility set. It never creates `CustomApi`/`UserCustomApi` entries, although production emits `is_team_shared` for that separate local loop at line 2520. Removing the production field would therefore leave this test passing, so the stated every-entry boolean contract is not protected for Custom API responses.
- **Recommendation:** Add team-shared and personal-only Custom API fixtures, assert `is_team_shared` is `true` and `false` respectively, and sweep the field across remote catalog, local MCP, and local Custom API entries.

### Suggestion

No separate suggestion-level findings were confirmed. The recommendations above are required fixes or targeted coverage improvements for the confirmed issues.

## Prior findings status

- **P1 — PARTIAL, not fixed.** The original review raised a high-severity string-ID/PostgreSQL type-mismatch and duplicate-loading claim at `src/xagent/web/api/mcp.py:2197`. Round 2 confirmed that the literal string-only failure mode and severity are overstated under the normal database path, but confirmed the remaining unvalidated malformed-hook boundary, mixed-representation duplicate load, and boolean/type hazards as a minor robustness issue. It is therefore narrowed to the minor finding above, not treated as resolved.
- **N4 — DROPPED, not a finding.** The `status=verified` change at `src/xagent/web/api/mcp.py:2332` intentionally retains team-shared OAuth entries even when `can_attach=false`, so members can discover them and use the personal Connect flow. The PR/issue contract distinguishes team sharing from personal authorization, and the frontend keeps Connect available for these entries. Under that stated contract, this is not a confirmed defect; it must not be included as a finding.
- The exported review history contained the single prior P1 review/inline comment from `gemini-code-assist`. There was no author reply, technically correct explanation, linked tracking issue, or follow-up PR for P1. No prior discussion or tracking existed for N4 or the new N1–N7 findings.

## Positive aspects

- The PR addresses the intended duplicate-card problem with one catalog representation rather than reviving an `is_custom` duplicate.
- `is_connected`, `is_team_shared`, and `can_attach` preserve distinct concepts instead of pretending that a team share is a personal connection. In particular, the OAuth paths fail closed without a member grant/account and retain a personal Connect recovery path.
- The backend owns the attachability decision and the frontend follows it for selection versus Connect/authorization behavior, avoiding a second frontend implementation of policy.
- Team rows are prefetched and indexed rather than queried once per catalog application; no new per-app N+1 pattern was confirmed. N5 is a separate duplicate preload/query cost.
- The response does not newly serialize secrets, and the existing team visibility boundary remains the source of which rows are visible. Standalone behavior and the empty-overlay path remain covered by the design.

## Criteria coverage

| Criterion | Assessment |
|---|---|
| Correctness | N1, N2, and N3 are confirmed major correctness/architecture issues. P1 remains partially valid as minor malformed-hook robustness. No other correctness issue was confirmed. |
| Security and permissions | N1 can confuse the identity/configuration of an authorized row, and N2 can advertise execution without required credentials. No direct secret disclosure or new database privilege bypass was confirmed. |
| Performance | N5 confirms an avoidable duplicate query/indexing path. The remote lookup itself is batched and no additional per-catalog-item query pattern was found. |
| Tests | N6 and N7 identify concrete coverage gaps in the new backend test file. CI preflight had no failing checks, but no local tests were run. |
| Documentation and code quality | The new fields and frontend mapping are understandable, and no separate documentation/configuration/migration omission was confirmed. N1's heuristic identity boundary and N5's repeated loading are the confirmed code-quality/design issues. |
| Design and architecture | The approach is acceptable-with-reservations: the catalog/provenance model fits the intended feature, but identity, credential scope, and agent scope are not yet authoritative. |
| Simplification lens | `Lean already.` No independent simplification finding survived review. |
| Other issues | No other confirmed issues are included in this review. N4 was dropped rather than reported. |

## Blocking status & recommended decision

**Blocking: yes**

Blocking issues:

- `src/xagent/web/api/mcp.py:2038` — **major** — normalized name/transport can bind a differently configured custom row to an official catalog identity. `[new]`
- `src/xagent/web/api/mcp.py:2074-2075` — **major** — a team-only `api_key` catalog row can be marked attachable without required runtime credentials. `[new]`
- `src/xagent/web/api/mcp.py:2178, 2302-2312` — **major** — user-scoped visibility is advertised as attachability without the target agent/team runtime scope. `[new]`

**Recommended decision:** `REQUEST_CHANGES`. The author is `codeacme17`, not `qinxuye`, so the qinxuye-specific COMMENT-only exception does not apply. Review limitation: CI preflight passed with no failing checks; local tests were not run.

## Inline comment map

- `src/xagent/web/api/mcp.py:2038` — **N1, major, [new]** — normalized name/transport matching can present a differently configured custom `MCPServer` as the official catalog connector and execute it through the catalog card.
- `src/xagent/web/api/mcp.py:2074-2075` — **N2, major, [new]** — the team attachability helper returns true for static `api_key` shapes without proving that the target runtime has every required environment variable.
- `src/xagent/web/api/mcp.py:2178, 2302-2312` — **N3, major, [new]** — `can_attach` is computed from current-user visibility even though runtime visibility is governed by the target agent's team scope.
- `src/xagent/web/api/mcp.py:2197` — **P1, minor, [prior]** — the malformed legacy team-ID response is still not normalized; the original string-only claim is narrowed, but mixed types and booleans remain unsafe.
- `src/xagent/web/api/mcp.py:2190-2207, 2349-2359` — **N5, minor, [new]** — the team rows are loaded for the catalog path and then queried again for `location=all`, while `location=local` pays for an unused preload.
- `tests/web/api/test_mcp_apps_team_catalog.py:271-289` — **N6, minor, [new]** — the test uses only a keyless app and cannot catch a false-positive team `api_key` attachability result.
- `tests/web/api/test_mcp_apps_team_catalog.py:504-533` — **N7, minor, [new]** — the every-entry `is_team_shared` test has no Custom API rows and therefore does not protect the production Custom API branch.
