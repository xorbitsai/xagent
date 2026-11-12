# Connector smoke V2: check declarations, engine and run results

Smoke V2 (xorbitsai/xagent-saas#2532) runs many checks per connector, under more than one identity, and writes a result that a later job compares across runs. This page documents the two interfaces the rest of V2 builds on: the **declaration format**, which says what a connector's checks are, and the **result document**, which says what a run found. It also documents the **engine** that runs the checks under an identity and produces the result. All three live in `src/xagent/web/connector_checks/`, and the two interfaces are versioned.

The engine is a library call (`engine.run_checks`). A `python -m` CLI runs it as a local user (see [CLI and the `user` identity](#cli-and-the-user-identity)), and Xagent SaaS wraps the same call with its Toby adapter (xorbitsai/xagent-saas#2534).

V1 is unchanged: L1 to L3 still run from `scripts/connector_smoke`, and the probe route still serves L3 (`scripts/connector_smoke/README.md`).

## Declarations

A connector declares a `ConnectorChecks` (`declarations.py`) with read checks, write checks, tool exemptions and, when it has write checks, a sweep. `validate_connector_checks` enforces every rule below, except the run-time and review items of [Write checks and the sweep](#write-checks-and-the-sweep); `tests/web/connector_checks/test_registry.py` runs it over every declaration.

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
                Equals(("sites", 0, "id"), FixtureRef("jira.tenant_id")),
            ),
        ),
    ),
    exemptions=(
        ToolExemption(
            "jira_transition_issue",
            "write; whether it gets a write check under the rules of #3653 is decided in #3570",
        ),
    ),
)
```

- **Check id.** Every check has a declared id, `<app_id>.<snake_case_name>`, at most 80 characters and unique within the connector. The result and the result diff key on it, so an id is never derived from a position and never reused for a different check.
- **Read check.** One tool, its arguments and assertions on the result. `require_read_only_hint=True` skips the check unless the server advertises the tool with `readOnlyHint: true`, for servers whose tools are vendor-owned (#3566).
- **Unprobeable check.** A slot that cannot run, with an L3 reason (`covered_by_l2`, `sandboxed`, `execution_scoped`, `needs_fixture`, `write_only`). Its row is always `unprobeable`.
- **Write check.** `create`, `verify` and `cleanup` steps, an optional `cleanup_guard`, a `tenant_pin` and `expected_verdicts`; see [Write checks and the sweep](#write-checks-and-the-sweep). A write check runs only for a connector whose app id is listed in `XAGENT_CONNECTOR_SMOKE_WRITE_APPS` (see [Running write checks](#running-write-checks)); otherwise it reports `skip` with `writes_disabled`.
- **Exemption.** A tool that has no check carries an exemption and a reason, keyed by its wire name. A tool cannot be both checked and exempted.

- **Sweep.** A connector with a write check declares one `Sweep`, which removes the objects aborted runs left behind; a connector without write checks declares none.

**Coverage (#3565).** A builtin connector is either in `CONNECTOR_CHECKS` or on `LEGACY_CONNECTORS` (`registry.py`), the connectors that have not adopted V2 yet. `tests/web/connector_checks/test_tool_coverage.py` holds these rules:

- Every tool an in-repo (`python -m`) connector's server registers has a check or an exemption, and every tool a declaration names is registered. A remote (`url`) or `npx` connector is not held to per-tool coverage (see Today's L3 probes).
- The legacy list only shrinks, and the test pins it. A connector leaves both `LEGACY_CONNECTORS` and the test's pinned list in the PR that adds its declaration; a new builtin connector ships with its declaration and cannot join it, and a name on it must still be a builtin connector.
- An exemption's reason says why the tool has no check yet. A read or write whose check is still to come names the issue of its connector family that decides it (#3567 for the Google connectors, #3569 for the Microsoft 365 ones; a write check also needs the rules of #3653). A write that edits content in place says so in its reason: a write check creates, verifies and deletes its own object and has no step for an edit. A write that creates an object no tool of the server deletes has a reason of its own, since a write check would have nothing to clean up with. A tool that reads and writes in one step has its own reason too, since a write check's verify or cleanup guard would have to call that same tool, which the read-step rule of #3653 refuses. When a check is added for a tool, its exemption goes in the same change: a tool cannot be both.

Arguments are JSON values. Any value, at any depth, can be a `FixtureRef`. A write check's steps can also use a `RunScopedName`, its `verify`, `cleanup` and `cleanup_guard` a `CreatedRef`, and a sweep's `delete` an `ItemRef`. None of them ever comes from the caller of the engine.

### Write checks and the sweep

```python
TENANT = FixtureRef("jira.tenant_id")
ISSUE = RunScopedName("issue")
ISSUE_KEY = CreatedRef(("issue", "key"))
PIN = TenantPin(
    ToolStep("jira_list_accessible_sites", assertions=(Length(("sites",), 1, 1),)),
    path=("sites", 0, "id"),
    expected=TENANT,
)

ConnectorChecks(
    app_id="jira",
    writes=(
        WriteCheck(
            id="jira.issue_round_trip",
            create=ToolStep(
                "jira_create_issue",
                {
                    "cloud_id": TENANT,
                    "project_key": FixtureRef("jira.write_project_key"),
                    "summary": ISSUE,
                },
                assertions=(StatusSuccess(), IsType(("issue", "key"), "string")),
            ),
            verify=ToolStep(
                "jira_get_issue",
                {"cloud_id": TENANT, "issue_key": ISSUE_KEY},
                assertions=(Equals(("issue", "summary"), ISSUE),),
            ),
            cleanup=ToolStep(
                "jira_delete_issue",
                {"cloud_id": TENANT, "issue_key": ISSUE_KEY},
                assertions=(StatusSuccess(),),
            ),
            cleanup_guard=ToolStep(
                "jira_get_issue",
                {"cloud_id": TENANT, "issue_key": ISSUE_KEY},
                assertions=(Equals(("issue", "summary"), ISSUE),),
            ),
            tenant_pin=PIN,
            expected_verdicts={"jira_create_issue": "deny", "jira_delete_issue": "deny"},
        ),
    ),
    sweep=Sweep(
        tenant_pin=PIN,
        find=ToolStep(
            "jira_search_issues",
            {"cloud_id": TENANT, "jql": FixtureRef("jira.write_sweep_jql"), "raw_fields": False},
            assertions=(IsType(("issues",), "array"),),
        ),
        items=("issues",),
        name=("summary",),
        delete=ToolStep(
            "jira_delete_issue",
            {"cloud_id": TENANT, "issue_key": ItemRef(("key",))},
            assertions=(StatusSuccess(),),
        ),
    ),
)
```

Validation enforces these:

- **Steps.** Every step of a write check (`create`, `verify`, `cleanup`, `cleanup_guard`) and of a sweep (`find`, `delete`) needs at least one assertion. A tenant pin's read, a cleanup guard and the sweep are not checked tools: they do not count as a check of their tool.
- **Tenant pin.** The pin's step reads the connection's tenant id at `path`; `expected` is the registry's id (a fixture key). Every tenant pin of a connector, its sweep's included, names the same fixture key. A path, here and in an assertion, is a tuple.
- **Write area.** `create`, and the sweep's `find`, take at least one write-area fixture key, `<app_id>.write_...`: `jira.write_project_key` above, or `google-drive.write_folder_id`. It names the isolated container the tenant registry (xorbitsai/xagent-saas#2533) sets aside for the connector's write checks. Its namespace is the connector's, which can differ from the pin's (`google.tenant_id`).
- **Run-scoped name.** `create` names its object with a `RunScopedName`, which the engine fills in as `xagent-smoke-<run_id>-<suffix>`. No two write checks of a connector use the same suffix, so in one run a name matches one check's object.
- **Cleanup.** `cleanup`'s arguments name the run's object: by the `RunScopedName` that `create` gave it, or by a `CreatedRef` (the value at a path of the create step's payload, typically the id the vendor gave the new object). They take no fixture key but the tenant pin's and write-area keys, so no seeded object outside the write area is the target. A cleanup with neither reference, with another run-scoped name or with another fixture key is refused.
- **Cleanup guard.** A cleanup that deletes by a `CreatedRef` needs a `cleanup_guard`, and no other cleanup has one. The guard's arguments carry the same reference and no run-scoped name, and its `Equals` asserts the `RunScopedName` that `create` gave the object. Its fixture keys are limited as cleanup's are.
- **Read steps.** Read checks, `verify`, every tenant pin's step, the cleanup guard and the sweep's `find` only read: none may call a tool that any of the connector's write checks creates or cleans up with.
- **Approval verdict.** `expected_verdicts` gives, for exactly the `create` and `cleanup` tools, the verdict the deployment's write policy gives a restricted actor (`allow`, `require_approval`, `deny`): for Toby, `deny` when the tool is withheld from a non-installer member. A change to Toby's write classification can therefore turn a check declared here red until its declaration follows; that is intended.
- **Sweep.** A connector with a write check declares one, and only such a connector does. `find` lists the write area, `items` is the path of the array of objects in its payload, and `name` the path of an item's name. `delete` takes one item by an `ItemRef` (a path inside the item), takes no fixture key but the pin's and write-area keys, and uses the cleanup tool of every write check of the connector, so its verdict is declared. A sweep lists one kind of object, so a connector's write checks all create, and clean up, that one kind.
- **Literal addresses.** No string in the arguments of a write check's or a sweep's step contains an `@`, as a key or a value: an address comes from a fixture in the test tenant.

At run time (see [Running write checks](#running-write-checks)):

- The pin runs before any write, and a tenant id that differs from the registry's refuses the write.
- The probe path bypasses both the approval gate and the per-user tool overrides, so the engine compares each declared verdict with the identity's write policy before any call.
- Cleanup runs even when verify fails, which is why the guard belongs to cleanup and not to verify. The guard's `Equals` compares the whole run-scoped name.
- The sweeper (see [The sweeper](#the-sweeper)) lists only the write area, so an object created outside it would never be swept. It matches each item's name against `xagent-smoke-` and reads the object's age from it.

Review each write flow against these, which validation cannot check:

- Cleanup and sweep use the connector's own tools, so a connector whose server cannot delete what its write check creates declares no write check until its Tier 1 issue settles how to clean up. Today that includes the HubSpot MCP server (#3572), the in-repo GitHub server (#3568), Slack's messages (#3571), Gmail's drafts (#3567) and Teams' messages (#3569).
- The sweep's `find` lists the same container `create` writes into. Jira's search takes only JQL, so its sweep reads a query fixture rather than the project key `create` takes.
- Each step's target comes from its reference or a write-area fixture, never from a literal, or from the pin's key, next to one: validation sees that a reference is present, not which argument the tool treats as its target.
- The cleanup guard finds the object by its `CreatedRef` alone, not by a query fixture or a literal that sits next to it. Only then does it catch a wrongly declared path before anything is deleted.
- Cleanup and the sweep's `delete` never target the write-area container itself, which is a seeded object too.
- The sweep's `ItemRef` is the id of the item whose `name` the sweeper matched, not a parent's or a sibling's.
- Every read step only reads. Validation refuses the connector's own create and cleanup tools there, not every tool that writes.
- A pin over a list, such as Jira's sites, asserts that the list has exactly one item.
- A cleanup that deletes by run-scoped name matches the whole name, never part of it: one suffix can begin another (`issue`, `issue-2`), and the engine passes the name to the tool as it is.
- Gmail creates drafts and never sends.
- Jira sets no assignee and no watcher.
- Slack posts only to the tenant's write channel.
- Nothing invites, shares with or notifies a user outside the test tenant.

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
| `Contains(path, text)` | the string at `path` contains `text` |
| `IsType(path, type)` | the value at `path` has JSON type `object`, `array`, `string`, `number`, `boolean` or `null` |
| `Length(path, min=0, max=None)` | the array at `path` has between `min` and `max` items |
| `Equals(path, value)` | the value at `path` equals `value`, typically a seeded fixture |
| `L3Envelope(expect_keys=())` | the L3 probe would pass the result (`evaluate_probe_result`, which reads the whole result); must be the only assertion, with the default `json_text` payload |

`assertions.judge_result` runs the assertions in order, and the first that does not hold fails the check as `assertion_failed` with its kind as the `reason`: `status_success`, `has_keys`, `each_has_keys`, `contains`, `is_type`, `length` or `equals`. A path that does not exist fails the assertion that reads it. `Equals` compares as JSON does: a boolean never equals a number, and `1` equals `1.0`. A payload that cannot be taken fails the same way, with the reason `not_a_json_object` (`json_text`), `not_json_text` (a text item of `json_text_items` that is not JSON) or `no_structured_content`. A result that is not a mapping is `call_failed`/`invalid_result`, and one flagged `is_error` is `call_failed`/`tool_error`. `L3Envelope` keeps the probe's own reasons (`not_a_json_object`, `status_not_success`, `no_status`, `missing_keys`).

### Fixture keys

A fixture key is `<namespace>.<name>`, at most 80 characters: the namespace is kebab-case, by convention the app id of the connector or tenant family that owns the value, and the name is one or more dot-separated snake_case parts (`google-docs.document_id`, `jira.seed.project_key`). The identity adapter resolves keys from server-side configuration, per environment and identity (`fixtures.FixtureResolver`). An unresolved key reports the check as `skip` with `fixture_missing` and the key as the row's `reason`. Errors and results name the key, never its value. The document, Google and Microsoft 365 connectors' read checks use these keys; the probe route does not take fixtures yet (#3284).

### Today's L3 probes

A connector that has not adopted V2 keeps `web/connector_probes.py` as its single source (`registry.connector_checks_for`):

- a declared probe becomes one read check, `<app_id>.l3_probe`, with the probe's tool and arguments and `L3Envelope(expect_keys)` as its only assertion;
- an unprobeable connector gets an `UnprobeableCheck` with its reason;
- an app with neither (a custom catalog app) has no declaration, and its `<connector>.l3_probe` row is `skip` with `not_declared`.

A connector adopts V2 by adding an entry to `CONNECTOR_CHECKS`. To keep its L3 call, it lists `l3_read_check(app_id)` among its reads.

`notion`, `miro` and `atlassian` have adopted V2 with a `current_user` read check that sets `require_read_only_hint`, because their tools are vendor-owned: the check is skipped unless the server advertises the tool with `readOnlyHint: true`. The check takes the place of their unprobeable `l3_probe` row. `connector_probes.py` has no such guard, so the probe route still records them as `covered_by_l2`. The other remote MCP connectors keep the derived `covered_by_l2` row until their result shapes are verified against a test account; `fireflies` returns plain text, which no payload source reads. A remote (`url`) connector's tool list is vendor-owned and changes without a release, so per-tool coverage (#3565) does not apply to it: it declares the checks it has and no exemptions.

The six document connectors (`google-docs`, `google-sheets`, `google-slides`, `powerpoint`, `word` and `excel`) declare a `read_fixture` read check that reads the object named by its fixture key: `google-docs.document_id`, `google-sheets.spreadsheet_id`, `google-slides.presentation_id`, `powerpoint.file_path`, `word.file_path` and `excel.file_path`. Each object is the seed's document or file in the identity's account (the Google and Microsoft 365 paragraphs below list what the checks assert of it), and a `file_path` is relative to that account's drive root, not a local task-workspace path. Without a value the check reports `skip` with `fixture_missing`, and the probe route, which does not take fixtures yet (#3284), still records these connectors as `needs_fixture`. `word`'s reads also need a tool output cap (`XAGENT_TOOL_MAX_OUTPUT_LENGTH`) of at least 16,384 characters.

`zendesk` and `zoom` have adopted V2: each keeps its L3 call through `l3_read_check` and adds structural read checks on list tools that take no object id, and exempts every other tool, because Tier 2 seeds no objects and runs no write checks (#3573).

`google-ads` has adopted V2 the same way as `zendesk` and `zoom`: it keeps its L3 call through `l3_read_check` and adds a structural read check on `google_ads_list_accessible_customers`, its one list tool that takes no object id and the tool its probe calls, which asserts the `customer_ids` list; `google_ads_search`, which takes a customer id and a query, is exempted (#3573).

`outlook`, `sharepoint` and `planner` have adopted V2 the same way: each keeps its L3 call and adds structural read checks on the reads that need no object id or query. `sharepoint` passes Graph's `root` site alias as the site id, which needs no fixture, and a `planner` list item asserts only the keys that survive the connector's output projection. A connector whose test account is not connected yet still declares its checks: the run has no target for it and reports no row until the account connects (#3573).

`jira` has adopted V2 at the Tier 1 bar (#3570). It keeps its L3 call and adds four read checks on the test tenant's seeded read project: `jira_search_issues` with the query `jira.seed.search_jql`, whose first issue is `jira.seed.issue_key`; `jira_list_comments` on that issue, whose first comment is `jira.seed.comment_id`; `jira_get_issue` on it; and `jira_get_current_user`. Its write check is the round trip shown under [Write checks and the sweep](#write-checks-and-the-sweep), with the sweep reading `jira.write_sweep_jql`. Each step that reads or writes issues passes `cloud_id` from `jira.tenant_id` instead of relying on the tool's single-site auto-detection; the tenant pin still refuses the write when the grant reaches any other site. The other seven tools are exempt: three reads the checked tools make unnecessary (they carry over 80% of Jira's calls), and four writes the write-check format cannot express, since each edits or adds to an existing issue rather than creating one it can delete.

`gmail`, `google-analytics`, `google-calendar`, `google-docs`, `google-drive`, `google-sheets` and `google-slides` have adopted V2 against the Google test tenant (#3567). Every read check but the probe-derived `l3_probe` is bound to a fixture key the tenant registry records for the seed layout: `gmail.seed_label_id` (the label's one seeded message), `google-drive.seed_folder_id` (the closed seed folder, found by a Drive search), `google-docs.document_id` (read directly, and again through Drive's text export), `google-sheets.spreadsheet_id` (the spreadsheet's one tab, and the seeded `item`/`quantity` block through `google_sheets_read_range`), `google-slides.presentation_id` (the deck's title and its one slide), and `google-analytics.property_id`. The checks assert the seed's own values where the seed pins one, such as the document's text `xagent smoke seed document`, the spreadsheet's `alpha`/`1` row and the deck's one slide, so a change to that seed is a change to the checks that read it. Assertions the seed does not pin are shape guards: a spreadsheet tab's `row_count` and `column_count` are Google's default grid, and an Analytics report's `rows` is an array. `google_analytics_run_report` returns no property field of GA's own: `query.property_id` is its normalization of the argument it was given, so no assertion reads it, and `google-analytics.property_id` must be a bare-digit string (`563334905`), the form the tenant registry records. `google-analytics` also runs its probe-derived `l3_probe`, which takes no fixture. `google-calendar` keeps only that check, because every Calendar tool addresses `primary`, while the seed's calendar is a secondary one, so no seeded event is reachable.

Every other Google tool is exempt: the reads outside the 80% cut; the writes, whose checks are decided with the write flows (#3567); the writes that create an object no tool of their server deletes (`google_docs_create_document`, `google_sheets_create_spreadsheet`, `google_slides_create_presentation`, `google_slides_import_pptx`, `gmail_send_messages`); and the in-place Docs edits (`google_docs_append_text`, `google_docs_replace_text`, `google_docs_batch_update`), which stay pending: a write check creates, verifies and deletes its own object and has no step for an edit, and the tenant registry sets aside no write-area document to edit. `google_slides_get_slide_thumbnail` is exempt because the tenant registry records no slide-id fixture: the deck the seed creates holds Google's default slide, but no fixture names it, and `gmail_manage_labels` is exempt because it is the only tool that reads a label and the only one that creates or deletes one, so a label write check's verify or cleanup guard would call its own create and cleanup tool (`#3653`); no separate label-reading tool is added.

`onedrive`, `teams`, `word`, `excel` and `powerpoint` have adopted V2 against the Microsoft 365 test account (#3569), with read checks only. Every check but four shape guards is bound to a fixture key the tenant registry records for the seed layout: the closed OneDrive folder `xagent-smoke-seed` (`onedrive.seed_folder_path`) holds exactly four files, the text note `xagent-smoke-seed-note.txt` (`onedrive.seed_file_path` and `onedrive.seed_file_id`) and the Word, Excel and PowerPoint seed files (`word.file_path`, `excel.file_path`, `powerpoint.file_path`); the read team (`teams.seed_team_id`) has one channel (`teams.seed_channel_id`), which lists the seeded message (`teams.seed_message_id`) first. Graph orders a channel's messages by the last activity of each reply chain, most recently active first, and includes system messages, so the seeded message must stay the channel's most recently active thread. The checks assert the seed's own values: the folder's four items; the note, found first by a search for `xagent-smoke-seed-note`, read by id under its seeded name, and read by path to its exact text; the document's two paragraphs, `xagent smoke seed document` and `second seeded paragraph`; the workbook's one worksheet `seed`, its `item`/`quantity`, `alpha`/`1`, `beta`/`2` block (numbers as numbers) and its used range `seed!A1:B3`; the deck's one slide, whose one text box reads `xagent smoke seed presentation` and which has no speaker notes; the team's one channel and the id and text of the message it lists first. `onedrive_get_profile`, `auth_status`, `list_teams` and `list_chats` (`teams.list_chats`) take no fixture and are shape guards; the team list is not counted or searched, since the account also belongs to other teams, and the chat list passes when it is empty, since the seed cannot hold a chat. These checks replace the probe-derived `onedrive.l3_probe` and `teams.l3_probe` rows, which called `onedrive_get_profile` and `get_current_user`, the latter now exempt; the document connectors keep their `read_fixture` ids.

The checked tools carry over 80% of each connector's calls over seven days on production-au, except Teams: OneDrive 52 of 63 calls (82.5%), Word 15 of 18 (83%), Excel 27 of 31 (87%) and PowerPoint 21 of 26 (80.8%). Teams stops at 39 of 51 (76.5%) under the cap of five read checks per connector, and at that volume the 80% cut means little. #3569 accepted that share with one swap: `list_chats` is checked in place of `get_current_user`, at the same share, because `auth_status` and `get_current_user` both read `/me`, the chat side had no check, and `/me/chats` needs a different Graph permission, so a grant that lost it now fails a check. `get_current_user` is exempt as a second read of `/me`, and `get_chat_messages` and `list_team_members` as left under the cap. The other reads are exempt as left unchecked. No Microsoft 365 connector declares a write check. A write check runs after a tenant pin, and the registry pins a Microsoft 365 connection by `microsoft-365.tenant_id`, the Entra tenant id: no tool of the OneDrive, Excel or PowerPoint servers returns a tenant id, so whether they get write checks is decided separately in #3569. Teams' `list_teams` does return each team's `tenantId`, but its writes are blocked by cleanup instead: no tool of its server deletes a message. The writes a write check could clean up (OneDrive's, Excel's worksheet and PowerPoint's slide tools) say so in their exemption. The writes that edit content in place (Word's paragraph and text edits, Excel's range, row, column and table-row edits, `powerpoint_set_shape_text`) carry the in-place reason, and those that create an object no tool of their server deletes (`word_create_document`, `excel_add_table`, `powerpoint_create_presentation`, and Teams' `send_channel_message` and `send_chat_message`) carry the no-cleanup reason.

## Identity adapters

The engine runs as an identity through an `IdentityAdapter` (`identity.py`):

- `label`: the identity's public label in the result (`toby`, `user`), never a user id, workspace id or email. It matches `results.IDENTITY_PATTERN`;
- `enumerate_targets()`: the identity's connectors as `ConnectorTarget`s, one per app id. A connector the adapter cannot start carries a `skip_reason` code, decided from the enumeration and never from a result. When the identity or its connectors cannot be resolved (for example a user outside `XAGENT_CONNECTOR_PROBE_USERS`), it raises `IdentitySetupError(stage, reason)`, and the run records `RunResult.failure = error.code` (`<stage>:<reason>`) with no rows;
- `open_session(target, purpose=...)`: starts the connector as a task of that identity would and returns its `ConnectorToolsResult`. A failure is reported through the listing's outcome, not raised. The engine finds a check's tool by its MCP wire name, so the listing's name prefix is the adapter's choice;
- `resolve_fixture(key)`: the fixture value, or None.

An identity whose deployment governs writes also implements `WritePolicy.write_policy_verdict(target, tool)`: the verdict its write policy gives a restricted actor for a tool, by its wire name, or None. Xagent SaaS's Toby adapter answers `deny` for a tool withheld from a non-installer member and `allow` otherwise (xorbitsai/xagent-saas#2534). The `user` adapter has no write policy, so its switched-on write checks fail as `verdict_mismatch`/`no_write_policy` once their session has passed.

An adapter starts a target only for a connection that is the catalog's own: for a server row, the probe route's catalog-identity rule (`owns_catalog_identity` in `services/connector_probe.py`); an actor stdio connection is synthesized from the catalog. A lookalike or custom row never gets a builtin app's checks run: the adapter leaves it out, or offers it with a skip reason, as the `user` adapter does with `not_catalog_identity`. App ids are unique per enumeration: where an identity has several rows of one app, the adapter offers the one its catalog resolution selects, the one a task of that identity would load.

`SessionPurpose.CALL_TOOLS` refuses a sandboxed (`npx`/`uvx`) connection, because the listing's lease is released before any call. `SessionPurpose.LIST_ONLY`, for a connector where no check will call a tool (each is unprobeable, or the app has no declaration), lets such a connection list through a lease, as V1's Toby probe does. This is the one departure from "sandboxed connections stay refused": no tool is called after the lease is released.

## Engine

`engine.run_checks(adapter, ...)` runs the identity's checks and returns a `RunResult`. It is meant for a one-shot CLI process: it does not take the web process's probe slot (`services/connector_probe.py`), so it must not run inside the web server. Its keywords are `environment` (the result's label), `connectors` (connector labels) and `checks` (check ids) as selectors, `check_timeout` (each call's bound, by default the L3 probe call's, `connector_probe.probe_call_timeout_seconds`), `total_timeout` (the run's bound, 900 seconds by default) and `declarations` (`registry.connector_checks_for` by default). A timeout that is a bool, NaN, infinite or not above 0 raises `ValueError` (a non-number raises `TypeError`). With a check selector, a selected connector none of whose checks it names is left out; without one, a connector whose declaration has only exemptions still gets its session row (it opens as `LIST_ONLY`) and no check rows.

Before any connector starts, the engine enumerates the targets, validates every selected declaration and resolves, through the adapter, the fixtures of every selected read check on a target the adapter does not skip. An invalid declaration, or one that names another app, raises `ValueError`. A key without a value makes its check `skip`/`fixture_missing`, and the check is never called; a value that is not a JSON scalar raises `TypeError`, and a resolver that raises fails the run as `fixtures:exception`, because a broken fixture store is a failure of the run, never a skipped check. The write checks of a connector whose writes are switched on are prepared at the same point (see [Running write checks](#running-write-checks)). Then, for each target in the adapter's order:

1. A target with a `skip_reason` is never started.
2. One session: one config build and one tool listing (`adapter.open_session`), which becomes the `session` row. It opens as `LIST_ONLY` when the app has no declaration or every check it declares is unprobeable, otherwise as `CALL_TOOLS`. The purpose follows the whole declaration, so neither a selector, setting a fixture nor turning on writes changes how a connector starts or its session row. An exception while opening it is `failed_to_start`/`exception`.
3. The checks run in declaration order on the listed tools. A read check's tool is found by its wire name (`connector_probe.find_tool`), and each call opens its own MCP session, so an in-repo server spawns one process per check. A check that requires `readOnlyHint` is skipped unless the tool's raw annotations say `readOnlyHint: true` without `destructiveHint` (`MCPToolAdapter.write_hint`). A call that exceeds `check_timeout` is `timed_out`/`check_call`, and the next check still runs.

A check's row depends on the session:

| Check | Session passed | Session did not pass |
|---|---|---|
| read check | run; or `fixture_missing`, `read_only_hint_missing`, `tool_missing` | `skip`/`session_unavailable` |
| unprobeable check | `unprobeable` | `unprobeable` |
| write check, writes switched on | run; or a refusal (see [Running write checks](#running-write-checks)) | `skip`/`session_unavailable` |
| write check, writes switched off | `skip`/`writes_disabled` | `skip`/`writes_disabled` |
| app with no declaration | `<connector>.l3_probe` `skip`/`not_declared` | the same |

Connectors run in sequence under the total timeout. The step it interrupts, a session or a call, fails as `timed_out` with the reason `run_deadline`, also when the step swallowed the cancellation and returned. A connector it keeps from starting has a `fail`/`not_reached` session row and the check rows of any session that did not pass; under a session that passed, a call it keeps from starting is `fail`/`not_reached`. The run is then not `complete`. A write check refused as `not_reached`/`time_budget` is the exception: it is refused before the deadline, and the run goes on (see [Running write checks](#running-write-checks)).

A run with no rows carries a `failure`. It is the adapter's `IdentitySetupError` code, or one of the engine's codes: `enumeration:exception` (the enumeration raised something else), `enumeration:no_connectors`, `enumeration:duplicate_connector` (two targets with one label), `fixtures:exception` (the fixture resolver raised), `write_policy:exception` (the identity's write policy raised) or `selection:no_targets` (the selectors matched nothing).

Today's L3 probes run as their derived `<app_id>.l3_probe` checks with the probe route's verdicts: `tests/web/connector_checks/test_l3_parity.py` feeds the same listing and tool behaviour to the route's call path and to the engine, and compares them. A probe outcome maps to a check row as `ok` → `pass`, `probe_failed` → `call_failed`, `probe_assertion_failed` → `assertion_failed`, `probe_tool_missing` → `tool_missing` and `timed_out` → `timed_out`, with the same reasons except the call's own bound (`probe_call` there, `check_call` here). A listing outcome is the session row's class.

### Running write checks

**Do not switch writes on until the V2 workflow runs the sweeper in that environment** ([The sweeper](#the-sweeper)). Scheduling it before each run is xorbitsai/xagent-saas#2536, which has not landed yet. A write that could not delete its object (a failed create, a failed guard or cleanup, a killed run) leaves it to the sweeper.

A write check creates, verifies and deletes a real object in the connector's test tenant (`writes.py`), so it runs only for a connector whose catalog app id is listed in `XAGENT_CONNECTOR_SMOKE_WRITE_APPS` (comma-separated, empty by default, set per environment; `config.get_connector_smoke_write_apps`). The engine reads the switch once at the start of a run. It is not a CLI argument, and an identity adapter cannot set it; the V2 workflow's kill switch still turns the whole run off (xorbitsai/xagent-saas#2536).

Each run has one id, `<yyyymmddhhmm>-<6 hex digits>` in UTC, and every `RunScopedName` becomes `xagent-smoke-<run_id>-<suffix>`. The id is logged on stderr when the run has a switched-on write check; it is not in the result, since the time in an object's name, with the environment, ties it to a run's `started_at`. Before any connector starts, the engine resolves a switched-on write check's fixtures (the pin's `expected` and every step's arguments and `Equals` values) and asks the identity's write policy for its create and cleanup tools' verdicts.

Once its connector's session has passed, a write check is refused, without calling anything, when:

| Refusal | Row |
|---|---|
| the identity has no write policy, or says nothing about a write tool | `fail`/`verdict_mismatch`, `no_write_policy` |
| the policy's verdict for the create or the cleanup tool is not the declared one | `fail`/`verdict_mismatch`, `create` or `cleanup` |
| a fixture has no value | `fail`/`tenant_mismatch`, the key |
| the listing lacks a tool the check calls | `fail`/`tool_missing`, the step (`pin`, `create`, `verify`, `guard`, `cleanup`) |
| the run's remaining time does not cover one call timeout per step | `fail`/`not_reached`, `time_budget`; later checks still run |

Otherwise it runs, each call under `check_timeout`:

1. **Pin.** The pin's step is called. A call that fails is `tenant_mismatch`/`pin_failed`, and a tenant id that is not the registry's is `tenant_mismatch`/`mismatch`.
2. **Create.** If the call raises, times out or returns an error, nothing is cleaned up: whatever it may have made is the sweeper's. If only an assertion fails, the object may exist, so cleanup still runs.
3. **Verify** runs when create passed. Every `CreatedRef` reads create's payload; a step whose id is not there, or is not a non-empty string or a finite number, makes create `assertion_failed`/`create:created_ref_missing`, and a cleanup that would delete by that id is left to the sweeper.
4. **Cleanup guard and cleanup** run whenever create returned something to clean up, including after a failed verify. The guard reads the object by its `CreatedRef` and must confirm its run-scoped name; if not, nothing is deleted and the row is `cleanup_failed`/`ownership` (`cleanup_failed`/`guard:<detail>` when the guard's call fails). A failed cleanup is `cleanup_failed`/`cleanup:<detail>`. Both calls run under their own call timeouts rather than the run's deadline, and finish even when the run is cancelled, which is re-raised afterwards.

The row reports the first of: a refusal, `cleanup_failed`, a failed create (`create:<detail>`), a failed verify (`verify:<detail>`), `pass`. A step's detail is the reason a read check would report: `exception`, `check_call`, `tool_error`, `invalid_result` or an assertion kind such as `status_success` or `equals`. Rows and logs carry no fixture value, response body or name.

### The sweeper

`sweeper.run_sweep(adapter, ...)` removes what write checks left behind. It sweeps each connector that declares a `Sweep` and whose writes are switched on; a connector whose writes are off is `skip`/`writes_disabled`, and one without a sweep has no row. Its keywords are `environment`, `connectors` (labels), `call_timeout` (each call's bound, by default the L3 probe call's) and `total_timeout` (600 seconds by default; the CLI's `--sweep` uses the same default).

Before any connector starts, it asks the identity's write policy for the sweep's delete tool, which is the write checks' cleanup tool, and resolves the sweep's fixtures; a policy or a resolver that raises fails the sweep as `write_policy:exception` or `fixtures:exception`. A connector is then refused, without calling anything, as a write check is: `verdict_mismatch` (`no_write_policy`, or `delete` when the policy's verdict is not the one the write checks declare), `tenant_mismatch` (a fixture with no value), `tool_missing` (`pin`, `find` or `delete`), `tenant_mismatch` (`pin_failed`, `mismatch`). A session that does not pass fails with its own class.

Otherwise the sweeper lists the write area with `find`, its first page only, so further leftovers wait for the next sweep, and reads each item's `name`:

| Item | What the sweeper does | Counted as |
|---|---|---|
| no `xagent-smoke-` prefix | nothing | not counted |
| prefixed, minute in its name more than 2 hours ago | deletes it by its `ItemRef` | `found`, then `deleted` or `failed` |
| prefixed, two hours old or younger, or dated in the future | keeps it: a run may still be cleaning it up | `found`, `kept` |
| prefixed, but no minute parses | keeps it, and reports it | `found`, `unparsed` |

The minute in a name is when its run started. A run with the engine's default total timeout lasts at most 15 minutes, so two hours keeps a sweep off every object a run in any environment may still be using; keep a run's `--total-timeout` well under two hours. An item listed twice is swept once. One sweep deletes at most 50 items of a connector (`cleanup_failed`/`cap_reached` beyond); the rest are counted as `kept` and wait for the next sweep. After a failed delete, `find` lists the area once more: an item missing from that second listing counts as deleted, as when another environment's sweeper took it first. A second listing that is empty, or in which no item has an id at the `ItemRef` path, proves nothing, since search can lag behind a delete; an id is compared as text, so `17`, `17.0` and `"17"` are one item. An item still listed, or one whose `ItemRef` finds no id, is `cleanup_failed`/`delete:<detail>`. The pin, `find`, each delete and the second listing start only when a call timeout of the sweep's total timeout is left; otherwise the connector is `not_reached`/`time_budget`, the items still due are counted as `kept`, and the sweep is not `complete`.

The result is its own document, `xagent.connector_smoke.sweep` (`sweep.schema.json`, `SWEEP_SCHEMA_VERSION` 1): the run fields of the result document, `older_than_seconds` and `max_deletes`, and one row per connector with its outcome, failure class, reason and the counts `found`, `deleted`, `failed`, `kept` and `unparsed`. It carries no name, id or value. The CLI prints it with `--sweep`.

## CLI and the `user` identity

`python -m xagent.web.connector_checks` runs the engine as a local user, to verify it locally and on stage. The V2 workflow does not schedule it: the web identity's scheduled coverage stays V1's HTTP L2/L3. Run it in the backend container with the service's environment, the same way as `oauth_readiness`:

```bash
docker compose exec -T backend /opt/xagent/deploy/entrypoint.sh \
    python -m xagent.web.connector_checks --user monitor
```

The options are selectors and bounds, never a tool name or an argument:

- `--user` (required): a username, or an email address compared case-insensitively;
- `--connector LABEL` and `--check CHECK_ID`, each repeatable or comma-separated: the engine's selectors;
- `--environment LABEL`: the result's environment label;
- `--check-timeout SECONDS` and `--total-timeout SECONDS`: the engine's bounds, finite and above 0;
- `--sweep`: run the sweeper instead and print the sweep document. `--check` does not apply. The `user` identity has no write policy, so its sweep refuses every switched-on connector: the option shows the wiring, while a runner whose identity governs writes, such as Xagent SaaS's Toby module, calls `run_sweep` itself.

Stdout carries exactly one JSON document, the run result (with `--sweep`, the sweep document), and logs go to stderr. While the run lasts, file descriptor 1 and `sys.stdout` point at stderr, so nothing a connector, a tool, a library or a child process writes can reach the document's stream. A run that cannot be set up still prints its document, with a `failure`: the adapter's or the engine's code, `setup:database` when the database cannot be opened, or `run:exception` when the engine raises (the log names the exception class only). Invalid arguments exit 2 with the usage on stderr and nothing on stdout. Otherwise the exit status is 0 when the run is `complete` and no row is `fail`, and 1 when it is not; a `skip` or `unprobeable` row does not fail the run.

The `user` adapter (`user_identity.py`):

- refuses a name no account has (`identity:user_not_found`) and a user outside `XAGENT_CONNECTOR_PROBE_USERS` (`identity:user_not_allowed`) before any connector starts. The user is matched by `is_connector_probe_user`, the probe route's rule, so `is_admin` grants nothing;
- offers the catalog apps the Connectors page shows the user as connected (`GET /api/mcp/apps`, the listing V1's L2/L3 smoke reads), in app id order, each with the server row that page selects for it. As in V1, an app hidden from the catalog (`is_visible_in_connector` false) is not a target even while connected. A catalog app that needs reconnecting (a builtin OAuth token that was cleared or expired with no refresh token, or an `mcp_oauth` association without an active grant) is not shown as connected either, so it is not a target and the run does not report it (#3635);
- offers a selected row that is not the catalog's own (`owns_catalog_identity`: a user-owned row under a catalog id, or a catalog row whose launch has drifted) with the skip reason `not_catalog_identity`. Such a row is never started, as the probe route answers `not_declared` for it. A custom server whose name is not a catalog app id (stamped with an app id or not) is not a catalog app the page shows as connected, so it is not a target;
- starts each connector as the user's own task would start it, with no actor policy and no team (`list_connector_tools`);
- resolves no fixture yet. The server-side source comes with #3284, so until then a check that needs a value is `skip`/`fixture_missing`.

## Result document

`results.RunResult.to_document()` builds one document per run and identity. It validates against the packaged JSON Schema, `src/xagent/web/connector_checks/result.schema.json` (`results.load_result_schema()`).

```json
{
  "schema": "xagent.connector_smoke.result",
  "schema_version": 2,
  "declaration_format_version": 2,
  "identity": "toby",
  "environment": "production-au",
  "started_at": "2026-11-06T04:00:00Z",
  "finished_at": "2026-11-06T04:01:35Z",
  "complete": true,
  "failure": null,
  "rows": [
    {"row": "session", "connector": "slack", "check": null, "kind": null, "tool": null,
     "outcome": "pass", "failure_class": null, "reason": null, "duration_ms": 1800},
    {"row": "check", "connector": "slack", "check": "slack.l3_probe", "kind": "read",
     "tool": "slack_list_channels", "outcome": "pass", "failure_class": null,
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

- **Rows.** Each connector has one `session` row (starting it and listing its tools), followed by one `check` row per declared check. The diff keys on `(identity, connector, check)`, with `check` null for the session row. When a session does not pass, none of its checks runs: each is `skip` with `session_unavailable` (`writes_disabled` for a write check whose connector's writes are off) or stays `unprobeable`, so one broken connector is one failure.
- **`connector`** is the catalog app id. An admin-created app's id is free text, so one that is not a kebab-case id of at most 64 characters, or that starts with `custom-`, is named `custom-<16 hex digits>` from its SHA-256 (`ConnectorTarget.connector`): stable across runs, and never the text itself. Hashing ids that start with `custom-` too means a generated label never names another app.
- **`environment`** is the deployment's label, or null when the caller gives none, so the diff can confirm it compares runs of one environment.
- **`complete`** is false when the run's total timeout cut it, or when it failed to set up; what the timeout kept from starting fails with `not_reached` (see [Engine](#engine)). A write check refused as `not_reached`/`time_budget` does not make the run incomplete: the run goes on.
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
| `not_reached` | the run's total timeout ended before this connector, or before this check's call; with reason `time_budget`, too little time was left to start this write check |
| `tool_missing` | the server does not list the check's tool |
| `call_failed` | the call raised (`exception`), the result was flagged as an error (`tool_error`), or it was not a mapping (`invalid_result`) |
| `assertion_failed` | an assertion did not hold; `reason` names which kind |
| `tenant_mismatch`, `verdict_mismatch`, `cleanup_failed` | a write check's safety rail stopped it, or its cleanup failed (see [Running write checks](#running-write-checks)) |

| Skip class (`skip`) | Meaning |
|---|---|
| `skipped_by_identity` | the adapter's enumeration skips the connector; `reason` is its code |
| `session_unavailable` | the connector's session did not pass |
| `not_declared` | the app has no declaration |
| `fixture_missing` | a fixture key has no value in this environment and identity; `reason` is the key |
| `read_only_hint_missing` | the check requires `readOnlyHint` and the server does not list the tool with it, including when the tool is gone: such a check is skipped, never called and never `tool_missing` |
| `writes_disabled` | the connector's writes are not switched on (`XAGENT_CONNECTOR_SMOKE_WRITE_APPS`) |

## Versioning

- `DECLARATION_FORMAT_VERSION` (`declarations.py`) changes when a declared check or a consumer of declarations must adapt. Version 2 added `CreatedRef`, `ItemRef`, the cleanup guard, the sweep and the rules on write areas, fixture keys in cleanup, read steps, run-scoped suffixes and `@` in arguments (#3564). It also refuses a path that is not a tuple: version 1 took a list, or a string as a path of one-letter keys.
- `SWEEP_SCHEMA_VERSION` (`sweeper.py`) and `sweep.schema.json` follow the same rule for the sweep document.
- `RESULT_SCHEMA_VERSION` (`results.py`) and the schema's `schema_version` const change together when a consumer of results must adapt: a field removed or renamed, a class removed, a meaning changed. A test pins the schema's enums to the Python enums.
- A new failure class or a new optional field is still a version bump, because the schema refuses unknown values; the result diff must be updated in the same change. Version 2 added the unprobeable reason `write_only` (#3655).
