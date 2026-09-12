// Client for the per-task connector-runtime endpoints and the pure
// classification/routing rules the connector-runtime dialog builds on. This
// file (and every symbol it exports) has exactly one production consumer:
// the connector-runtime dialog and the app-chat context that opens it. It is
// not a general connector client -- it does not know about connection
// management, only about the two endpoints that read and write a task's
// missing runtime inputs.
import type { ClientErrorCode } from "@/lib/client-errors"
import { apiRequest, isJsonRecord, parseApiResponse } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"

// section and type are closed sets on the wire (schemas/connector_runtime.py);
// a client may switch on them exhaustively, and adding a member is a wire
// change. Kept as `as const` arrays so a test can bind its parameterized rows
// to these members instead of re-typing the set by hand.
export const CONNECTOR_RUNTIME_SECTIONS = ["context", "secrets", "auth_selector"] as const
export type ConnectorRuntimeSection = (typeof CONNECTOR_RUNTIME_SECTIONS)[number]

export const CONNECTOR_RUNTIME_TYPES = ["string", "object"] as const
export type ConnectorRuntimeType = (typeof CONNECTOR_RUNTIME_TYPES)[number]

export interface ConnectorRuntimeRef {
  connector_type: string
  connector_id: number
}

export interface ConnectorRuntimeInput {
  section: ConnectorRuntimeSection
  key: string
  type: ConnectorRuntimeType
  required: boolean
  satisfied: boolean
  expired: boolean
}

export interface ConnectorRuntimeConnector {
  connector_ref: ConnectorRuntimeRef
  name: string
  inputs: ConnectorRuntimeInput[]
}

export interface ConnectorRuntimeReport {
  satisfied: boolean
  secrets_expires_at: string | null
  connectors: ConnectorRuntimeConnector[]
}

function isConnectorRuntimeSection(value: unknown): value is ConnectorRuntimeSection {
  return (CONNECTOR_RUNTIME_SECTIONS as readonly unknown[]).includes(value)
}

function isConnectorRuntimeType(value: unknown): value is ConnectorRuntimeType {
  return (CONNECTOR_RUNTIME_TYPES as readonly unknown[]).includes(value)
}

function readConnectorRuntimeRef(value: unknown): ConnectorRuntimeRef | null {
  if (!isJsonRecord(value)) return null
  if (typeof value.connector_type !== "string") return null
  if (typeof value.connector_id !== "number") return null
  return { connector_type: value.connector_type, connector_id: value.connector_id }
}

function readConnectorRuntimeInput(value: unknown): ConnectorRuntimeInput | null {
  if (!isJsonRecord(value)) return null
  const { section, key, type, required, satisfied, expired } = value
  if (!isConnectorRuntimeSection(section)) return null
  if (typeof key !== "string") return null
  if (!isConnectorRuntimeType(type)) return null
  if (typeof required !== "boolean") return null
  if (typeof satisfied !== "boolean") return null
  if (typeof expired !== "boolean") return null
  return { section, key, type, required, satisfied, expired }
}

function readConnectorRuntimeConnector(value: unknown): ConnectorRuntimeConnector | null {
  if (!isJsonRecord(value)) return null
  const connectorRef = readConnectorRuntimeRef(value.connector_ref)
  if (!connectorRef) return null
  if (typeof value.name !== "string") return null
  if (!Array.isArray(value.inputs)) return null
  const inputs: ConnectorRuntimeInput[] = []
  for (const rawInput of value.inputs) {
    const input = readConnectorRuntimeInput(rawInput)
    if (!input) return null
    inputs.push(input)
  }
  return { connector_ref: connectorRef, name: value.name, inputs }
}

/**
 * Validates an unknown value against the closed report shape. Never throws:
 * any cell that does not fit -- an unrecognized `section`/`type`, a missing
 * field, a non-array `connectors` -- returns null rather than a partially
 * trusted object. A future caller must not soften an unrecognized `section`
 * or `type` into a guessed member (`"context"`/`"string"`); the server
 * treats both fields as closed and a silent fallback would render a field
 * the server has not actually declared as fillable.
 */
export function readConnectorRuntimeReport(value: unknown): ConnectorRuntimeReport | null {
  if (!isJsonRecord(value)) return null
  const { satisfied, secrets_expires_at, connectors } = value
  if (typeof satisfied !== "boolean") return null
  if (secrets_expires_at !== null && typeof secrets_expires_at !== "string") return null
  if (!Array.isArray(connectors)) return null
  const parsedConnectors: ConnectorRuntimeConnector[] = []
  for (const rawConnector of connectors) {
    const connector = readConnectorRuntimeConnector(rawConnector)
    if (!connector) return null
    parsedConnectors.push(connector)
  }
  return { satisfied, secrets_expires_at, connectors: parsedConnectors }
}

// The seven status codes the read endpoint's own failure paths can return
// (see the per-task read endpoint's own docstring); anything else that is
// not 200 is still a read failure, it just does not need its own entry here.
export const READ_FAILURE_STATUSES = [400, 401, 403, 404, 422, 500, 503] as const

export type FetchTaskConnectorRuntimeRequirementsResult =
  | { ok: true; report: ConnectorRuntimeReport }
  | { ok: false; kind: "transport" }
  | { ok: false; kind: "http"; status: number }
  | { ok: false; kind: "malformed" }

/**
 * GET /api/chat/task/{task_id}/connector-runtime-requirements. Never throws:
 * a transport failure, a non-200 response, or a 200 whose body does not
 * parse into a report all come back as a tagged failure, kept distinct from
 * "read a report that says nothing is missing" -- collapsing the two would
 * make a task whose read genuinely fails look permanently satisfied. The
 * response body's `detail` string is never read: it is the server's English
 * safe message, not something to show a user.
 */
export async function fetchTaskConnectorRuntimeRequirements(
  taskId: number,
): Promise<FetchTaskConnectorRuntimeRequirementsResult> {
  let response: Response
  try {
    response = await apiRequest(
      `${getApiUrl()}/api/chat/task/${taskId}/connector-runtime-requirements`,
    )
  } catch {
    return { ok: false, kind: "transport" }
  }
  if (!response.ok) return { ok: false, kind: "http", status: response.status }
  const parsed = await parseApiResponse(response)
  const report = readConnectorRuntimeReport(parsed.data)
  if (!report) return { ok: false, kind: "malformed" }
  return { ok: true, report }
}

export interface ConnectorRuntimeSubmitItem {
  connector_ref: ConnectorRuntimeRef
  context: Record<string, unknown>
}

export type SubmitTaskConnectorRuntimeValuesFailure =
  | { ok: false; kind: "transport" }
  | {
    ok: false
    kind: "coded"
    status: number
    code: string
    reason?: string
    connectorRef?: ConnectorRuntimeRef
  }
  | { ok: false; kind: "http"; status: number }
  | { ok: false; kind: "malformed" }

export type SubmitTaskConnectorRuntimeValuesResult =
  | { ok: true; report: ConnectorRuntimeReport }
  | SubmitTaskConnectorRuntimeValuesFailure

function readSubmitErrorEnvelope(
  value: unknown,
): { code: string; reason?: string; connectorRef?: ConnectorRuntimeRef } | null {
  if (!isJsonRecord(value)) return null
  const error = value.error
  if (!isJsonRecord(error)) return null
  if (typeof error.code !== "string") return null
  const details = isJsonRecord(error.details) ? error.details : null
  // A key is present-with-any-value (including "") the moment its type is
  // string; absent -- the 503 shape that carries no reason at all -- is a
  // different input from present-and-empty, so this must not fall back to
  // `|| undefined`.
  const reason = details && typeof details.reason === "string" ? details.reason : undefined
  const connectorRef = details ? readConnectorRuntimeRef(details.connector_ref) ?? undefined : undefined
  return { code: error.code, reason, connectorRef }
}

/**
 * POST /api/chat/task/{task_id}/connector-runtime-values. Never throws.
 * Distinguishes four failure shapes because the write endpoint's non-200
 * responses are not uniform: some carry the `{"error": {...}}` envelope
 * (a `ConnectorRuntimeError`), the task-not-found 404 does not, and a 200
 * whose body fails report validation is its own case (a server bug, not a
 * network problem). A 200 response does not by itself mean the connector is
 * now satisfied -- resolveDialogOutcome decides that from the returned
 * report, not from this function's `ok` flag.
 */
export async function submitTaskConnectorRuntimeValues(
  taskId: number,
  items: ConnectorRuntimeSubmitItem[],
): Promise<SubmitTaskConnectorRuntimeValuesResult> {
  let response: Response
  try {
    response = await apiRequest(
      `${getApiUrl()}/api/chat/task/${taskId}/connector-runtime-values`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items }),
      },
    )
  } catch {
    return { ok: false, kind: "transport" }
  }
  const parsed = await parseApiResponse(response)
  if (response.ok) {
    const report = readConnectorRuntimeReport(parsed.data)
    return report ? { ok: true, report } : { ok: false, kind: "malformed" }
  }
  const envelope = readSubmitErrorEnvelope(parsed.data)
  if (!envelope) return { ok: false, kind: "http", status: response.status }
  return {
    ok: false,
    kind: "coded",
    status: response.status,
    code: envelope.code,
    reason: envelope.reason,
    connectorRef: envelope.connectorRef,
  }
}

// Exact-match reason strings the disposition table below branches on by
// value. Deliberately excludes the two "stored selection is corrupted" reason
// strings the server can produce (services/connector_runtime.py's
// _load_task_selected_refs): one of them interpolates a field name from
// whatever the corrupted row actually stored, so the full set is not
// enumerable, and neither string ever equals KEY_NAME_REJECTED_REASON below
// -- both fall through to the generic disposition instead.
export const CONNECTOR_RUNTIME_KNOWN_REASONS = [
  "empty_items",
  "empty_item_payload",
  "payload_too_large",
  "duplicate_ref",
  "connector_not_selected",
  "undeclared_context_key",
] as const

// The one exact sentence the per-turn gate raises for a malformed key name
// (validate_runtime_source_key in
// core/tools/adapters/vibe/connector_runtime.py). Written out rather than
// imported because this is TypeScript and that constant lives in the Python
// package; it mirrors that function's literal message, and a change to the
// message on that side has to be mirrored here.
export const KEY_NAME_REJECTED_REASON = "runtime input key must match [A-Za-z0-9_-]+"

const TYPE_MISMATCH_CONTEXT_PREFIX = "type_mismatch.context."
const EMPTY_VALUE_CONTEXT_PREFIX = "empty_value.context."
const CONFLICT_CONTEXT_PREFIX = "conflict.context."

export interface ConnectorRuntimeFailureLocation {
  connectorRef?: ConnectorRuntimeRef
  key?: string
}

// The full set of connectorRuntime.errors.* leaves classifySubmitFailure can
// select. Kept as a literal union (not `string`) so a caller mapping this to
// a translation key does so through an exhaustive lookup table rather than a
// template-literal key that a source scan cannot verify against the i18n
// tree (see the dialog component's own error-message lookup).
export type ConnectorRuntimeErrorMessageKey =
  | "network"
  | "busyRetry"
  | "contactAdmin"
  | "conflict"
  | "typeString"
  | "typeObject"
  | "emptyValue"
  | "keyNameRejected"
  | "configChanged"
  | "notInSession"
  | "connectorUnavailable"
  | "tooLarge"

export interface ConnectorRuntimeFailureDisposition {
  messageKey: ConnectorRuntimeErrorMessageKey
  retry: boolean
  // Whether the dialog should re-read the report before the next submit. A
  // conflict disposition sets this because the stored value moved out from
  // under the user; the next `buildSubmitItems(refreshedReport, drafts)` call
  // already drops any key the refreshed report now reports satisfied, so no
  // separate "drop the keys the retry no longer needs" field is needed here.
  refresh: boolean
  locate: ConnectorRuntimeFailureLocation
}

function findDeclaredInputType(
  report: ConnectorRuntimeReport,
  connectorRef: ConnectorRuntimeRef | undefined,
  key: string,
): ConnectorRuntimeType | null {
  if (!connectorRef) return null
  for (const connector of report.connectors) {
    if (
      connector.connector_ref.connector_type !== connectorRef.connector_type
      || connector.connector_ref.connector_id !== connectorRef.connector_id
    ) continue
    for (const input of connector.inputs) {
      if (input.key === key) return input.type
    }
  }
  return null
}

const GENERIC_DISPOSITION: ConnectorRuntimeFailureDisposition = {
  messageKey: "contactAdmin",
  retry: false,
  refresh: false,
  locate: {},
}

/**
 * The single place that turns a write failure into what the dialog shows,
 * whether the retry button appears, whether the report is re-read
 * automatically, and where the error attaches. Never throws. Judged in a
 * fixed order because two shapes both carry `code: "invalid_runtime_context"`
 * and an English `reason` (the malformed-key sentence and a corrupted stored
 * selection): only the malformed-key sentence is checked by exact text, so a
 * future call site must not loosen that check to "looks like English" --
 * that would misclassify a corrupted-selection failure as a fixable key name.
 * `report` is the dialog's own most recently read report, consulted only to
 * pick between the two field-type dispositions (`typeString`/`typeObject`)
 * for a mismatch reason that does not itself carry the field's declared type.
 */
export function classifySubmitFailure(
  outcome: SubmitTaskConnectorRuntimeValuesFailure,
  report: ConnectorRuntimeReport,
): ConnectorRuntimeFailureDisposition {
  if (outcome.kind === "transport") {
    return { messageKey: "network", retry: true, refresh: false, locate: {} }
  }

  if (outcome.kind === "http") return GENERIC_DISPOSITION
  if (outcome.kind === "malformed") {
    return { ...GENERIC_DISPOSITION, refresh: true }
  }

  const { status, code, reason, connectorRef } = outcome

  if (status === 503 && code === "connector_runtime_unavailable" && reason === undefined) {
    return { messageKey: "busyRetry", retry: true, refresh: false, locate: {} }
  }
  // Any 503 with a reason key present (including an empty string) is treated
  // uniformly: the one case this client mints on purpose
  // (team_scope_resolution_failed) and any deployment-installed hook's own
  // 503 both land here, because neither is something a user retry fixes.
  if (status === 503 && reason !== undefined) return GENERIC_DISPOSITION

  if (
    status === 409
    && code === "runtime_context_immutable"
    && reason !== undefined
    && reason.startsWith(CONFLICT_CONTEXT_PREFIX)
  ) {
    const key = reason.slice(CONFLICT_CONTEXT_PREFIX.length)
    return { messageKey: "conflict", retry: false, refresh: true, locate: { connectorRef, key } }
  }

  if (status === 400 && code === "invalid_runtime_context" && reason !== undefined) {
    if (reason.startsWith(TYPE_MISMATCH_CONTEXT_PREFIX)) {
      const key = reason.slice(TYPE_MISMATCH_CONTEXT_PREFIX.length)
      const declaredType = findDeclaredInputType(report, connectorRef, key)
      return {
        messageKey: declaredType === "object" ? "typeObject" : "typeString",
        retry: false,
        refresh: false,
        locate: { connectorRef, key },
      }
    }
    if (reason.startsWith(EMPTY_VALUE_CONTEXT_PREFIX)) {
      const key = reason.slice(EMPTY_VALUE_CONTEXT_PREFIX.length)
      return { messageKey: "emptyValue", retry: false, refresh: false, locate: { connectorRef, key } }
    }
    if (reason === KEY_NAME_REJECTED_REASON) {
      // The response never carries the offending key -- only which
      // connector it belongs to -- so this cannot locate a specific row.
      return { messageKey: "keyNameRejected", retry: false, refresh: false, locate: { connectorRef } }
    }
    if (reason === "undeclared_context_key") {
      return { messageKey: "configChanged", retry: false, refresh: true, locate: { connectorRef } }
    }
    if (reason === "connector_not_selected") {
      return { messageKey: "notInSession", retry: false, refresh: true, locate: { connectorRef } }
    }
    if (reason === "payload_too_large") {
      return { messageKey: "tooLarge", retry: false, refresh: false, locate: { connectorRef } }
    }
    if ((CONNECTOR_RUNTIME_KNOWN_REASONS as readonly string[]).includes(reason)) {
      // empty_items / empty_item_payload / duplicate_ref: shapes this
      // client's own request-building should never produce.
      return GENERIC_DISPOSITION
    }
  }

  if (status === 404 && code === "connector_not_found") {
    return { messageKey: "connectorUnavailable", retry: false, refresh: true, locate: { connectorRef } }
  }

  // Closed set exhausted: an unrecognized code, a reason outside every
  // matched prefix and the known-reason set, or a corrupted stored
  // selection's unenumerable sentence. A hook a deployment installs can also
  // raise a ConnectorRuntimeError with any code/status of its own choosing,
  // and it lands here the same way.
  return GENERIC_DISPOSITION
}

/**
 * Whether the connector-runtime dialog is a supported presence on the
 * current page. The failure notification that would open it reaches every
 * page an authenticated user can be on (the triggering frame is filtered
 * only by task identity, not by route), but the dialog is a UI surface for
 * exactly the pages a user fills it in from: the conversation page and the
 * two workforce run/edit pages. Everywhere else the existing failure bubble
 * is the only notification, same as before this dialog existed. This is a
 * product-surface decision, not an authorization boundary: whether a read or
 * write is *allowed* is decided solely by the per-task endpoints' own
 * owner-of-the-task check, and narrowing where the dialog appears can only
 * make it appear less, never grant it access it would not otherwise have.
 *
 * These are the same route shapes components/layout/sidebar.tsx already
 * matches off `usePathname()` to find the viewed conversation, and the
 * static-export server (frontend_static.py) maps the same three shapes to
 * a page shell -- a trailing slash and the `__shell__` placeholder id both
 * still match, so this holds under either deployment.
 */
export const CONNECTOR_RUNTIME_DIALOG_HOST_PATTERNS = [
  /^\/task\/[^/]+\/?$/,
  /^\/workforces\/[^/]+\/run\/?$/,
  /^\/workforces\/[^/]+\/?$/,
] as const

export function isConnectorRuntimeDialogHostPath(pathname: string | null): boolean {
  if (!pathname) return false
  return CONNECTOR_RUNTIME_DIALOG_HOST_PATTERNS.some(pattern => pattern.test(pathname))
}

// The three terminal task_error codes that open the dialog. The other
// two connector-runtime codes the client error table knows about --
// invalid_runtime_context, connector_runtime_unavailable -- settle the turn
// without anything left for this dialog to collect: the first means the
// server-stored selection or a declared key is itself invalid, the second
// means a dependency the gate needs is down, and neither is answered by
// filling in a context value.
export const CONNECTOR_RUNTIME_DIALOG_TRIGGER_CODES = [
  "missing_runtime_context",
  "runtime_secret_unavailable",
  "scheduled_secret_unavailable",
] as const

export function isConnectorRuntimeDialogTriggerCode(
  code: ClientErrorCode,
): code is (typeof CONNECTOR_RUNTIME_DIALOG_TRIGGER_CODES)[number] {
  return (CONNECTOR_RUNTIME_DIALOG_TRIGGER_CODES as readonly string[]).includes(code)
}

export const DIALOG_OUTCOME_KINDS = ["met", "unsupported_only", "nothing_fillable", "fillable"] as const
export type DialogOutcomeKind = (typeof DIALOG_OUTCOME_KINDS)[number]

export interface ConnectorRuntimeInputLocation {
  connectorRef: ConnectorRuntimeRef
  key: string
}

export type DialogOutcome =
  | { kind: "met" }
  | { kind: "unsupported_only"; blocking: ConnectorRuntimeInputLocation[] }
  | { kind: "nothing_fillable" }
  | { kind: "fillable"; blocking: ConnectorRuntimeInputLocation[] }

/**
 * The single derivation both "should the dialog open" and "should it close
 * after a save" read from: never recomputed per-key, always
 * `report.satisfied` for "met" -- a report can have every required key
 * satisfied yet carry a top-level `false` (one malformed-but-optional
 * secrets key forces the aggregate false; services/connector_runtime.py's
 * aggregation rule), and recomputing from the per-key flags here would read
 * that report as met when the server does not consider it so. Two groups
 * partition every unsatisfied *required* input: "A" (`context`, something
 * this dialog can still collect) and "S" (`secrets`/`auth_selector`,
 * required but not fillable in this phase). Optional inputs never appear in
 * `blocking` -- an unfilled optional secret is not something to tell the
 * user "is still needed".
 */
export function resolveDialogOutcome(report: ConnectorRuntimeReport): DialogOutcome {
  if (report.satisfied) return { kind: "met" }

  const blockingA: ConnectorRuntimeInputLocation[] = []
  const blockingS: ConnectorRuntimeInputLocation[] = []
  for (const connector of report.connectors) {
    for (const input of connector.inputs) {
      if (!input.required || input.satisfied) continue
      const location = { connectorRef: connector.connector_ref, key: input.key }
      if (input.section === "context") blockingA.push(location)
      else blockingS.push(location)
    }
  }

  if (blockingA.length === 0 && blockingS.length === 0) return { kind: "nothing_fillable" }
  if (blockingA.length === 0) return { kind: "unsupported_only", blocking: blockingS }
  return { kind: "fillable", blocking: blockingS }
}

export type ConnectorRuntimeDialogAction = "saveAndResend" | "saveOnly" | "acknowledge"

/**
 * The dialog's button set, derived from the outcome and whether this
 * request carries a resend snapshot -- never from whether the provider's
 * stash is non-empty, which is cleared the moment the snapshot is handed to
 * the request; reading the stash here instead would make the resend button
 * disappear right after the handoff that is supposed to enable it.
 */
export function resolveDialogActions(
  outcome: DialogOutcome,
  hasResendPayload: boolean,
): ConnectorRuntimeDialogAction[] {
  if (outcome.kind === "unsupported_only" || outcome.kind === "nothing_fillable") {
    return ["acknowledge"]
  }
  if (outcome.kind === "met") return []
  return hasResendPayload ? ["saveAndResend", "saveOnly"] : ["saveOnly"]
}

/** Shared by the dialog's draft state and buildSubmitItems so a draft value
 *  written under one key is always read back under the same key. */
export function connectorRuntimeInputDraftKey(ref: ConnectorRuntimeRef, key: string): string {
  return `${ref.connector_type}:${ref.connector_id}:${key}`
}

/**
 * The context-section items worth submitting: unsatisfied, non-blank. Reads
 * `section === "context"` explicitly rather than `!satisfied` alone --
 * `secrets`/`auth_selector` rows are also always `satisfied: false` in this
 * phase, and the request body has no field for either, so including them
 * would submit a shape the server 422s. Does not consult `required`: a
 * `fillable` dialog lets a user fill in an optional context key too, and
 * this is the one place that decides what actually gets submitted for a
 * connector, keyed by the connector_ref exactly as the report gave it (never
 * the whole connector object, which the request schema does not accept
 * extra fields on without a silent drop, schemas/connector_runtime.py:112).
 */
export function buildSubmitItems(
  report: ConnectorRuntimeReport,
  drafts: Record<string, string>,
): ConnectorRuntimeSubmitItem[] {
  const items: ConnectorRuntimeSubmitItem[] = []
  for (const connector of report.connectors) {
    const context: Record<string, unknown> = {}
    for (const input of connector.inputs) {
      if (input.section !== "context" || input.satisfied) continue
      const rawValue = drafts[connectorRuntimeInputDraftKey(connector.connector_ref, input.key)]
      if (rawValue === undefined) continue
      if (input.type === "string") {
        if (rawValue.trim() === "") continue
        context[input.key] = rawValue
      } else {
        let parsed: unknown
        try {
          parsed = JSON.parse(rawValue)
        } catch {
          continue
        }
        if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) continue
        // An empty object is the object-draft equivalent of a blank string:
        // the server's own blank check (`not value` for an object-typed
        // context field) treats `{}` as empty and 400s the whole submission,
        // taking every other filled-in key in the same batch down with it.
        if (Object.keys(parsed).length === 0) continue
        context[input.key] = parsed
      }
    }
    if (Object.keys(context).length === 0) continue
    items.push({ connector_ref: connector.connector_ref, context })
  }
  return items
}

/**
 * The submit button's only enabling rule: at least one submittable key, and
 * no context draft that failed object-JSON parsing. Never adds "every
 * required key filled" -- that would be a second, independently-maintained
 * copy of the server's own completeness rule, the exact drift the design
 * forbids. A malformed key-name warning never participates here either: it
 * is informational, and the row it warns about can still be submitted (the
 * server, not this rule, is the one that will reject it).
 */
export function isSubmitEnabled(items: ConnectorRuntimeSubmitItem[], hasInvalidObjectDraft: boolean): boolean {
  return items.length > 0 && !hasInvalidObjectDraft
}

// Mirrors core/tools/adapters/vibe/connector_runtime.py's
// validate_runtime_source_key exactly, for a hint only: it never
// participates in submission gating, and a server-side rename of the
// accepted character set would require updating this constant to match.
const ACCEPTED_RUNTIME_KEY_NAME_RE = /^[A-Za-z0-9_-]+$/

export function isAcceptedRuntimeKeyName(key: string): boolean {
  return ACCEPTED_RUNTIME_KEY_NAME_RE.test(key)
}
