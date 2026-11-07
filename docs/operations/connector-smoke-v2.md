# Connector smoke V2: check declarations, engine and run results

Smoke V2 (xorbitsai/xagent-saas#2532) runs many checks per connector, under more than one identity, and writes a result that a later job compares across runs. This page documents the two interfaces the rest of V2 builds on: the **declaration format**, which says what a connector's checks are, and the **result document**, which says what a run found. It also documents the **engine** that runs the checks under an identity and produces the result. All three live in `src/xagent/web/connector_checks/`, and the two interfaces are versioned.

The engine is a library call (`engine.run_checks`). Its CLI and a local `user` identity adapter come in a follow-up to #3563; until then the engine runs under an adapter its caller supplies, such as the Toby adapter (xorbitsai/xagent-saas#2534).

V1 is unchanged: L1 to L3 still run from `scripts/connector_smoke`, and the probe route still serves L3 (`scripts/connector_smoke/README.md`).

## Declarations

A connector declares a `ConnectorChecks` (`declarations.py`) with read checks, write checks and tool exemptions. `validate_connector_checks` enforces every rule below; `tests/web/connector_checks/test_registry.py` runs it over every declaration.

```python
ConnectorChecks(
    app_id="jira",
    reads=(
        ReadCheck(
            id="jira.list_sites",
            tool="jira_list_accessible_sites",
            assertions=(
                StatusSuccess(),
                EachHasKeys(("sites",), ("id", "name")),
                Equals(("sites", 0, "id"), FixtureRef("jira.site_id")),
            ),
        ),
    ),
    exemptions=(ToolExemption("jira_transition_issue", "moves a real issue"),),
)
```

- **Check id.** Every check has a declared id, `<app_id>.<snake_case_name>`, at most 80 characters and unique within the connector. The result and the result diff key on it, so an id is never derived from a position and never reused for a different check.
- **Read check.** One tool, its arguments and assertions on the result. `require_read_only_hint=True` skips the check unless the server advertises the tool with `readOnlyHint: true`, for servers whose tools are vendor-owned (#3566).
- **Unprobeable check.** A slot that cannot run, with an L3 reason (`covered_by_l2`, `sandboxed`, `execution_scoped`, `needs_fixture`). Its row is always `unprobeable`.
- **Write check.** `create`, `verify` and `cleanup` steps, a `tenant_pin` and `expected_verdicts`. `create` must name its object with a `RunScopedName`, which the engine fills in as `xagent-smoke-<run_id>-<suffix>`. The tenant pin reads the connection's tenant id and compares it with the registry's (a fixture key). `expected_verdicts` gives, for exactly the `create` and `cleanup` tools, the approval gate's verdict (`allow`, `require_approval`, `deny`), because the probe path bypasses the gate. The write switch is per connector, keyed by its app id. **No write check runs until #3564 lands**: each one reports `skip` with `writes_disabled`.
- **Exemption.** A tool that has no check carries an exemption and a reason, keyed by its wire name. A tool cannot be both checked and exempted.

Arguments are JSON values. Any value, at any depth, can be a `FixtureRef`; a write step's value can also be a `RunScopedName`. Neither ever comes from the caller of the engine.

### Assertions

Assertions read a **payload** taken from the tool's normalized result (`{"content": [...], "structured_content": ..., "is_error": ...}`). A result flagged `is_error` fails before any assertion runs.

| `payload` | The payload is |
|---|---|
| `json_text` (default) | the one text item, parsed as a JSON object: the in-repo servers' shape |
| `json_text_items` | the list of every text item, each parsed as JSON |
| `structured_content` | the MCP `structuredContent` object |

A path is a tuple of object keys and list indexes into the payload; `()` is the payload itself.

| Assertion | Passes when |
|---|---|
| `StatusSuccess()` | the payload's `status` is `"success"` (the in-repo envelope) |
| `HasKeys(path, keys)` | the object at `path` has every key |
| `EachHasKeys(path, keys)` | every item of the array at `path` has every key |
| `IsType(path, type)` | the value at `path` has JSON type `object`, `array`, `string`, `number`, `boolean` or `null` |
| `Length(path, min=0, max=None)` | the array at `path` has between `min` and `max` items |
| `Equals(path, value)` | the value at `path` equals `value`, typically a seeded fixture |
| `L3Envelope(expect_keys=())` | the L3 probe would pass the result (`evaluate_probe_result`, which reads the whole result); must be the only assertion, with the default `json_text` payload |

`assertions.judge_result` runs the assertions in order, and the first that does not hold fails the check as `assertion_failed` with its kind as the `reason`: `status_success`, `has_keys`, `each_has_keys`, `is_type`, `length` or `equals`. A path that does not exist fails the assertion that reads it. `Equals` compares as JSON does: a boolean never equals a number, and `1` equals `1.0`. A payload that cannot be taken fails the same way, with the reason `not_a_json_object` (`json_text`), `not_json_text` (a text item of `json_text_items` that is not JSON) or `no_structured_content`. A result that is not a mapping is `call_failed`/`invalid_result`, and one flagged `is_error` is `call_failed`/`tool_error`. `L3Envelope` keeps the probe's own reasons (`not_a_json_object`, `status_not_success`, `no_status`, `missing_keys`).

### Fixture keys

A fixture key is `<namespace>.<name>`, at most 80 characters: the namespace is kebab-case, by convention the app id of the connector or tenant family that owns the value, and the name is one or more dot-separated snake_case parts (`google-docs.document_id`, `jira.seed.project_key`). The identity adapter resolves keys from server-side configuration, per environment and identity (`fixtures.FixtureResolver`). An unresolved key reports the check as `skip` with `fixture_missing` and the key as the row's `reason`. Errors and results name the key, never its value. The probe route's document fixtures (#3284) use the same keys.

### Today's L3 probes

A connector that has not adopted V2 keeps `web/connector_probes.py` as its single source (`registry.connector_checks_for`):

- a declared probe becomes one read check, `<app_id>.l3_probe`, with the probe's tool and arguments and `L3Envelope(expect_keys)` as its only assertion;
- an unprobeable connector gets an `UnprobeableCheck` with its reason;
- an app with neither (a custom catalog app) has no declaration, and its `<connector>.l3_probe` row is `skip` with `not_declared`.

A connector adopts V2 by adding an entry to `CONNECTOR_CHECKS`. To keep its L3 call, it lists `l3_read_check(app_id)` among its reads.

`notion`, `miro` and `atlassian` have adopted V2 with a `current_user` read check that sets `require_read_only_hint`, because their tools are vendor-owned: the check is skipped unless the server advertises the tool with `readOnlyHint: true`. The check takes the place of their unprobeable `l3_probe` row. `connector_probes.py` has no such guard, so the probe route still records them as `covered_by_l2`. The other remote MCP connectors keep the derived `covered_by_l2` row until their result shapes are verified against a test account; `fireflies` returns plain text, which no payload source reads. A remote (`url`) connector's tool list is vendor-owned and changes without a release, so per-tool coverage (#3565) does not apply to it: it declares the checks it has and no exemptions.

## Identity adapters

The engine runs as an identity through an `IdentityAdapter` (`identity.py`):

- `label`: the identity's public label in the result (`toby`, `user`), never a user id, workspace id or email. It matches `results.IDENTITY_PATTERN`;
- `enumerate_targets()`: the identity's connectors as `ConnectorTarget`s, one per app id. A connector the adapter cannot start carries a `skip_reason` code, decided from the enumeration and never from a result. When the identity or its connectors cannot be resolved (for example a user outside `XAGENT_CONNECTOR_PROBE_USERS`), it raises `IdentitySetupError(stage, reason)`, and the run records `RunResult.failure = error.code` (`<stage>:<reason>`) with no rows;
- `open_session(target, purpose=...)`: starts the connector as a task of that identity would and returns its `ConnectorToolsResult`. A failure is reported through the listing's outcome, not raised. The engine finds a check's tool by its MCP wire name, so the listing's name prefix is the adapter's choice;
- `resolve_fixture(key)`: the fixture value, or None.

An adapter offers a target only for a connection that is the catalog's own: for a server row, the probe route's catalog-identity rule (`_owns_catalog_identity` in `services/connector_probe.py`); an actor stdio connection is synthesized from the catalog. A lookalike or custom row never gets a builtin app's checks. App ids are unique per enumeration: where an identity has several rows of one app, the adapter offers the one its catalog resolution selects, the one a task of that identity would load.

`SessionPurpose.CALL_TOOLS` refuses a sandboxed (`npx`/`uvx`) connection, because the listing's lease is released before any call. `SessionPurpose.LIST_ONLY`, for a connector where no check will call a tool (each is unprobeable, or the app has no declaration), lets such a connection list through a lease, as V1's Toby probe does. This is the one departure from "sandboxed connections stay refused": no tool is called after the lease is released.

## Engine

`engine.run_checks(adapter, ...)` runs the identity's checks and returns a `RunResult`. It is meant for a one-shot CLI process: it does not take the web process's probe slot (`services/connector_probe.py`), so it must not run inside the web server. Its keywords are `environment` (the result's label), `connectors` (connector labels) and `checks` (check ids) as selectors, `check_timeout` (each call's bound, by default the L3 probe call's, `connector_probe.probe_call_timeout_seconds`), `total_timeout` (the run's bound, 900 seconds by default) and `declarations` (`registry.connector_checks_for` by default). A selected connector with no selected check is left out.

Before any connector starts, the engine enumerates the targets, validates every selected declaration and resolves every selected read check's fixtures through the adapter. An invalid declaration, or one that names another app, raises `ValueError`. A key without a value makes its check `skip`/`fixture_missing`, and the check is never called; a value that is not a JSON scalar raises `TypeError`, and anything the resolver raises propagates, because a broken fixture store is a failure of the run, never a skipped check. Then, for each target in the adapter's order:

1. A target with a `skip_reason` is never started.
2. One session: one config build and one tool listing (`adapter.open_session`), which becomes the `session` row. It opens as `LIST_ONLY` when the app has no declaration or every check it declares is unprobeable, otherwise as `CALL_TOOLS`. The purpose follows the whole declaration, so neither a selector, setting a fixture nor turning on writes changes how a connector starts or its session row. An exception while opening it is `failed_to_start`/`exception`.
3. The checks run in declaration order on the listed tools. A read check's tool is found by its wire name (`connector_probe.find_tool`), and each call opens its own MCP session, so an in-repo server spawns one process per check. A check that requires `readOnlyHint` is skipped unless the tool's raw annotations say `readOnlyHint: true` without `destructiveHint` (`MCPToolAdapter.write_hint`). A call that exceeds `check_timeout` is `timed_out`/`check_call`, and the next check still runs.

A check's row depends on the session:

| Check | Session passed | Session did not pass |
|---|---|---|
| read check | run; or `fixture_missing`, `read_only_hint_missing`, `tool_missing` | `skip`/`session_unavailable` |
| unprobeable check | `unprobeable` | `unprobeable` |
| write check | `skip`/`writes_disabled` | `skip`/`writes_disabled` |
| app with no declaration | `<connector>.l3_probe` `skip`/`not_declared` | the same |

Connectors run in sequence under the total timeout. The step it interrupts, a session or a call, fails as `timed_out` with the reason `run_deadline`, also when the step swallowed the cancellation and returned. A connector it keeps from starting has a `fail`/`not_reached` session row and the check rows of any session that did not pass; under a session that passed, a call it keeps from starting is `fail`/`not_reached`. The run is then not `complete`.

A run with no rows carries a `failure`. It is the adapter's `IdentitySetupError` code, or one of the engine's codes: `enumeration:exception` (the enumeration raised something else), `enumeration:no_connectors`, `enumeration:duplicate_connector` (two targets with one label) or `selection:no_targets` (the selectors matched nothing).

Today's L3 probes run as their derived `<app_id>.l3_probe` checks with the probe route's verdicts: `tests/web/connector_checks/test_l3_parity.py` feeds the same listing and tool behaviour to the route's call path and to the engine, and compares them. A probe outcome maps to a check row as `ok` → `pass`, `probe_failed` → `call_failed`, `probe_assertion_failed` → `assertion_failed`, `probe_tool_missing` → `tool_missing` and `timed_out` → `timed_out`, with the same reasons except the call's own bound (`probe_call` there, `check_call` here). A listing outcome is the session row's class.

## Result document

`results.RunResult.to_document()` builds one document per run and identity. It validates against the packaged JSON Schema, `src/xagent/web/connector_checks/result.schema.json` (`results.load_result_schema()`).

```json
{
  "schema": "xagent.connector_smoke.result",
  "schema_version": 1,
  "declaration_format_version": 1,
  "identity": "toby",
  "environment": "production-au",
  "started_at": "2026-11-06T04:00:00Z",
  "finished_at": "2026-11-06T04:01:35Z",
  "complete": true,
  "failure": null,
  "rows": [
    {"row": "session", "connector": "gmail", "check": null, "kind": null, "tool": null,
     "outcome": "pass", "failure_class": null, "reason": null, "duration_ms": 1800},
    {"row": "check", "connector": "gmail", "check": "gmail.l3_probe", "kind": "read",
     "tool": "gmail_search_messages", "outcome": "pass", "failure_class": null,
     "reason": null, "duration_ms": 900},
    {"row": "session", "connector": "jira", "check": null, "kind": null, "tool": null,
     "outcome": "fail", "failure_class": "client_rejected", "reason": "client_rejected",
     "duration_ms": 400},
    {"row": "check", "connector": "jira", "check": "jira.l3_probe", "kind": "read",
     "tool": "jira_list_accessible_sites", "outcome": "skip",
     "failure_class": "session_unavailable", "reason": null, "duration_ms": 0}
  ]
}
```

- **Rows.** Each connector has one `session` row (starting it and listing its tools), followed by one `check` row per declared check. The diff keys on `(identity, connector, check)`, with `check` null for the session row. When a session does not pass, none of its checks runs: each is `skip` with `session_unavailable` (`writes_disabled` for a write check) or stays `unprobeable`, so one broken connector is one failure.
- **`connector`** is the catalog app id. An admin-created app's id is free text, so one that is not a kebab-case id of at most 64 characters, or that starts with `custom-`, is named `custom-<16 hex digits>` from its SHA-256 (`ConnectorTarget.connector`): stable across runs, and never the text itself. Hashing ids that start with `custom-` too means a generated label never names another app.
- **`environment`** is the deployment's label, or null when the caller gives none, so the diff can confirm it compares runs of one environment.
- **`complete`** is false when the run's total timeout cut it, or when it failed to set up; what the timeout kept from starting fails with `not_reached` (see [Engine](#engine)).
- **`failure`** is a code for a run that could not be set up (`<stage>:<reason>`, such as `enumeration:no_connectors`; the engine's codes are listed under [Engine](#engine)), and null otherwise. Such a run has no rows; the diff reads it as no data, never as every connector recovering.
- **Outcomes.** `pass` and `unprobeable` carry no class; an `unprobeable` row's `reason` is its L3 reason. `fail` and `skip` carry a class from the tables below, and `reason` optionally refines it with a code.
- **What a document never holds.** Tokens, secrets, response bodies, fixture values or end-user ids. Every reason is a code (`^[a-z0-9][a-z0-9_.:-]{0,79}$`): a listing's reason that is not a code is dropped, and a row refuses one. A code-shaped value can still be anything, so the callers that build rows put only codes in them.
- **The schema enforces the pairing** of outcome and class: `pass` carries none, `fail` only a failure class, `skip` only a skip class, and `unprobeable` (a check row) an L3 reason. The schema checks each row on its own. The cross-row rules (a check id starts with its connector, one session row per connector, a session row precedes its checks, no check runs under a failed session) are enforced only by `RunResult`.

| Failure class (`fail`) | Meaning |
|---|---|
| `not_found` | the identity has no such connector, or the runtime dropped it |
| `not_connected` | the user's credential is missing or unusable: a dead grant; `reason` is the builder's code (`oauth_token_required`, `auth_rejected`, …) |
| `client_rejected` | the vendor rejected the platform's own client credential (#3580): every user of the provider is affected; `reason` is `client_rejected`, or `client_rejected:<provider>` when the session names the credential |
| `unsupported` | the connector cannot run outside a task (`execution_scoped`) or with tool calls (`sandboxed`) |
| `failed_to_start` | the server did not start; `reason` is the load phase, or `exception` when opening the session raised |
| `no_tools` | the server listed no tools |
| `timed_out` | starting, listing or a check's call exceeded its bound; `reason` is `check_call` for a call's own bound and `run_deadline` for the run's |
| `not_reached` | the run's total timeout ended before this connector, or before this check's call |
| `tool_missing` | the server does not list the check's tool |
| `call_failed` | the call raised (`exception`), the result was flagged as an error (`tool_error`), or it was not a mapping (`invalid_result`) |
| `assertion_failed` | an assertion did not hold; `reason` names which kind |
| `tenant_mismatch`, `verdict_mismatch`, `cleanup_failed` | write-check safety (#3564) |

| Skip class (`skip`) | Meaning |
|---|---|
| `skipped_by_identity` | the adapter's enumeration skips the connector; `reason` is its code |
| `session_unavailable` | the connector's session did not pass |
| `not_declared` | the app has no declaration |
| `fixture_missing` | a fixture key has no value in this environment and identity; `reason` is the key |
| `read_only_hint_missing` | the check requires `readOnlyHint` and the server does not list the tool with it, including when the tool is gone: such a check is skipped, never called and never `tool_missing` |
| `writes_disabled` | write checks are off |

## Versioning

- `DECLARATION_FORMAT_VERSION` (`declarations.py`) changes when a declared check or a consumer of declarations must adapt.
- `RESULT_SCHEMA_VERSION` (`results.py`) and the schema's `schema_version` const change together when a consumer of results must adapt: a field removed or renamed, a class removed, a meaning changed. A test pins the schema's enums to the Python enums.
- A new failure class or a new optional field is still a version bump, because the schema refuses unknown values; the result diff must be updated in the same change.
