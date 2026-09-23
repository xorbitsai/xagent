// Client for the per-task connector-runtime endpoints and the pure
// classification/routing rules the connector-runtime dialog builds on. This
// file (and every symbol it exports) has exactly one production consumer:
// the connector-runtime dialog and the app-chat context that opens it. It is
// not a general connector client -- it does not know about connection
// management, only about the two endpoints that read and write a task's
// missing runtime inputs.
import type { ClientErrorCode } from "@/lib/client-errors"
import { apiRequest, isJsonRecord, parseApiResponse, type ParsedApiResponse } from "@/lib/api-wrapper"
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

/**
 * How long either call below waits before giving up on a request that is not
 * answering. Nothing else cancels these two: the dialog's read effect drops a
 * late answer but does not stop the request behind it, and the save runs to
 * completion on purpose even when the dialog unmounts mid-flight. Without a
 * bound here, a request that never finishes answering -- no response at all,
 * or headers with a body that never arrives -- leaves the dialog unable to
 * save and, while the save is the call in flight, unable to close at all.
 *
 * The one comparable constant in this frontend is api-wrapper's
 * AUTH_REFRESH_TIMEOUT_MS (15s, for the token refresh). These two calls are
 * the heavier pair -- the write endpoint runs a full validation pass and one
 * encryption before it answers -- so this is one notch longer rather than a
 * number carried in from somewhere outside this repository.
 */
const CONNECTOR_RUNTIME_REQUEST_TIMEOUT_MS = 20_000

interface AnsweredConnectorRuntimeRequest {
  response: Response
  parsed: ParsedApiResponse
}

/**
 * Runs one connector-runtime request under the timeout above and hands back
 * both the response and its parsed body. Throws if the request does not get
 * that far in time.
 *
 * Both of apiRequest's paths forward the whole RequestInit to fetch -- direct
 * with no stored token, through withBearer and fetchWithRetry with one -- so
 * the signal reaches every attempt without api-wrapper's shared helpers
 * needing to know this timeout exists, and callers that pass no signal keep
 * behaving exactly as before.
 *
 * The body is read here rather than by the callers, because fetch resolves
 * as soon as the response *headers* arrive: a server that sends headers and
 * then stalls would leave `response.text()` waiting with no bound on it at
 * all, which is the unbounded wait this timeout exists to rule out. Holding
 * the read inside the window costs the callers nothing they did not already
 * do -- both parse every response they get -- except that a non-200 read
 * response now has its body parsed too, and that parse is discarded unread
 * exactly as before (this endpoint's `detail` string is never shown).
 *
 * Shaped like api-wrapper's own performTokenRefresh: an explicit controller
 * and timer rather than AbortSignal.timeout, because the timer has to be
 * cleared on the way out -- a request that answered in time must not leave a
 * pending abort behind it.
 *
 * No distinct failure kind for a timeout: both callers below already map a
 * rejection to `transport`, which is what a request that never answered is
 * from the caller's side.
 */
async function requestConnectorRuntime(
  url: string,
  init: RequestInit = {},
): Promise<AnsweredConnectorRuntimeRequest> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), CONNECTOR_RUNTIME_REQUEST_TIMEOUT_MS)
  try {
    const response = await apiRequest(url, { ...init, signal: controller.signal })
    const parsed = await parseApiResponse(response)
    // An aborted body read does not reach here as a rejection the way an
    // aborted fetch does: parseApiResponse turns it into an empty body
    // (`response.text().catch(() => "")`, api-wrapper.ts). So a body this
    // timeout cut short would arrive looking like a well-formed empty
    // response, which both callers report as `malformed` -- their word for a
    // server bug, carrying no retry. Reading the signal is what tells a
    // cut-short read apart from a genuinely empty one.
    if (controller.signal.aborted) throw new Error("connector-runtime request timed out")
    return { response, parsed }
  } finally {
    clearTimeout(timer)
  }
}

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
 * safe message, not something to show a user. A request whose headers and
 * body are not both in hand within CONNECTOR_RUNTIME_REQUEST_TIMEOUT_MS is
 * abandoned and reported the same way as any other transport failure.
 */
export async function fetchTaskConnectorRuntimeRequirements(
  taskId: number,
): Promise<FetchTaskConnectorRuntimeRequirementsResult> {
  let answered: AnsweredConnectorRuntimeRequest
  try {
    answered = await requestConnectorRuntime(
      `${getApiUrl()}/api/chat/task/${taskId}/connector-runtime-requirements`,
    )
  } catch {
    return { ok: false, kind: "transport" }
  }
  const { response, parsed } = answered
  if (!response.ok) return { ok: false, kind: "http", status: response.status }
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
 * report, not from this function's `ok` flag. A request whose headers and
 * body are not both in hand within CONNECTOR_RUNTIME_REQUEST_TIMEOUT_MS is
 * abandoned and reported as a transport failure; the values it carried may or
 * may not have been written, which is already true of every other transport
 * failure on this endpoint.
 */
export async function submitTaskConnectorRuntimeValues(
  taskId: number,
  items: ConnectorRuntimeSubmitItem[],
): Promise<SubmitTaskConnectorRuntimeValuesResult> {
  let answered: AnsweredConnectorRuntimeRequest
  try {
    answered = await requestConnectorRuntime(
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
  const { response, parsed } = answered
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

// The enumerable exact-match `reason` strings the server raises alongside
// code `invalid_runtime_context`. classifySubmitFailure does not read this
// constant: it branches on the individual strings it has a specific
// disposition for and lets every other reason reach the generic fallthrough.
// The constant exists so a test can bind each member to the disposition it
// actually produces, which is what makes "a new reason silently becomes
// generic" a visible choice rather than an accident.
//
// Deliberately excludes the two "stored selection is corrupted" reason
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
  "auth_selector_not_supported",
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
  | "typeUnknown"
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
      // Every type_mismatch reason this backs is section-scoped to "context"
      // (see locateFieldError in connector-runtime-dialog.tsx, which filters
      // the same way): without this filter a same-named "secrets" row could
      // win the search and report that row's declared type instead.
      if (input.section === "context" && input.key === key) return input.type
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
 * pick among the field-type dispositions (`typeString`/`typeObject`/
 * `typeUnknown`) for a mismatch reason that does not itself carry the
 * field's declared type.
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
      // The connector's own edit endpoint writes a new declaration in place,
      // with no version and no snapshot held by the task -- so the type this
      // dialog read and the type the write endpoint just checked against can
      // differ. Refreshing here is what lets the row pick up the new
      // declaration on its next render, rather than staying keyed to the
      // stale one. The messageKey below is chosen against the pre-refresh
      // report passed into this call and is not itself recomputed after the
      // refresh; isTypeMismatchDispositionStale (below) is what the caller
      // checks against the refreshed report to clear this hint outright
      // when the row's type no longer matches it, instead of leaving a
      // type-specific message attached to a row that has since changed type.
      // Three answers, not two: "object", "string", and "this report
      // declares no type for this row at all" -- a rejection that carries
      // no connector_ref, or a row this report does not carry. Folding the
      // third into "string" tells the user a field needs text on the
      // strength of a declaration nobody ever read.
      return {
        messageKey: declaredType === null
          ? "typeUnknown"
          : declaredType === "object" ? "typeObject" : "typeString",
        retry: false,
        refresh: true,
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
  }

  if (status === 404 && code === "connector_not_found") {
    return { messageKey: "connectorUnavailable", retry: false, refresh: true, locate: { connectorRef } }
  }

  // Closed set exhausted: an unrecognized code, a reason outside every
  // prefix and exact string matched above, or a corrupted stored selection's
  // unenumerable sentence. The reasons this client's own request-building
  // should never produce (empty_items, empty_item_payload, duplicate_ref)
  // and auth_selector_not_supported also land here: they carry nothing a
  // user can act on. A hook a deployment installs can raise a
  // ConnectorRuntimeError with any code/status of its own choosing, and it
  // lands here the same way.
  return GENERIC_DISPOSITION
}

/**
 * Whether a type hint on a disposition no longer matches the row it would
 * attach to, given a report re-read after the disposition's own
 * `refresh: true` landed. Three messageKeys carry a type hint:
 * `typeObject` and `typeString` name a declared type, and `typeUnknown`
 * says the report the hint was derived from named none. Every other
 * messageKey is never type-related, so this always reads false for it.
 *
 * This answers one question only: does the refreshed report still declare
 * the type the hint names. A row that report no longer declares under
 * "context", and one it now reports satisfied, both leave the two named
 * hints alone -- neither is a changed type, and neither means the save
 * stopped being rejected. What such a row does change is where the hint
 * can attach and therefore what it may claim: the dialog's field-error
 * location falls back to whole-dialog scope, where it is reworded to stop
 * naming a type no field on screen is asking for
 * (translateDialogScopeFailure in connector-runtime-dialog.tsx).
 *
 * What the dialog should then *do* about a hint this calls stale is not
 * this predicate's question, and the dialog does not ask it directly:
 * reconcileTypeMismatchDisposition below is the one caller, because two of
 * the three hints are dropped on a stale answer and one is re-derived.
 */
export function isTypeMismatchDispositionStale(
  disposition: ConnectorRuntimeFailureDisposition,
  refreshedReport: ConnectorRuntimeReport,
): boolean {
  const { messageKey } = disposition
  if (messageKey !== "typeObject" && messageKey !== "typeString" && messageKey !== "typeUnknown") return false
  const { connectorRef, key } = disposition.locate
  if (key === undefined) return false
  const currentType = findDeclaredInputType(refreshedReport, connectorRef, key)
  // The unknown hint's whole claim is "this report declares no type here",
  // so any declared type in the refreshed report ends that claim -- including
  // the type the server was enforcing all along, which is the common case.
  // Ending the claim is not the same as having nothing left to say, which is
  // why the reconciler below re-derives this one instead of dropping it.
  if (messageKey === "typeUnknown") return currentType !== null
  const expectedType = messageKey === "typeObject" ? "object" : "string"
  return currentType !== null && currentType !== expectedType
}

/**
 * What a type hint becomes once a report re-read after its own
 * `refresh: true` has landed: the same hint, a re-derived one, or none.
 *
 * Three answers rather than the two "keep it or drop it" the staleness
 * check alone gives, because a hint stops matching a refreshed report for
 * two opposite reasons. `typeObject`/`typeString` name a type the row has
 * since stopped declaring, and nothing in the refreshed report says what
 * the server was enforcing instead, so there is nothing left to say and
 * the hint goes. `typeUnknown` is the other way round: its whole claim is
 * that the report named no type here, and a refreshed report that does
 * name one answers exactly the question the hint said it could not. So it
 * is re-derived into the named hint rather than dropped -- dropping it
 * would take a rejection the user was shown off the screen while the draft
 * that provoked it is still in the box and the save button still live, and
 * the next save would be refused the same way for the same unstated
 * reason.
 *
 * The caller compares the result with what it passed in: the same object
 * back means nothing changed.
 */
export function reconcileTypeMismatchDisposition(
  disposition: ConnectorRuntimeFailureDisposition,
  refreshedReport: ConnectorRuntimeReport,
): ConnectorRuntimeFailureDisposition | null {
  if (!isTypeMismatchDispositionStale(disposition, refreshedReport)) return disposition
  if (disposition.messageKey !== "typeUnknown") return null
  const { connectorRef, key } = disposition.locate
  const currentType = key === undefined ? null : findDeclaredInputType(refreshedReport, connectorRef, key)
  // Unreachable: the staleness check calls an unknown-type hint stale only
  // when the refreshed report declares a type for this row. Kept as a guard
  // rather than a non-null assertion so that this function proves the fact
  // instead of assuming it, and so a future loosening of that check cannot
  // turn it into a hint naming a type nobody read.
  if (currentType === null) return null
  return { ...disposition, messageKey: currentType === "object" ? "typeObject" : "typeString" }
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
 * The static-export server (frontend_static.py) maps all three shapes to a
 * page shell -- a trailing slash and the `__shell__` placeholder id both
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
 *
 * Every outcome maps to at least one button, `met` included. A met report
 * normally closes the dialog before it renders: the first read does, and so
 * does a save, once any resend it promised has settled. Three paths still
 * render one in an open dialog -- the refresh a failed save triggers, a
 * same-task re-request's read that finds nothing missing, and a
 * save-and-resend's own successful save while its resend is still in flight
 * -- and returning no buttons there left the footer empty with only the
 * window chrome's close control to get out of it.
 */
export function resolveDialogActions(
  outcome: DialogOutcome,
  hasResendPayload: boolean,
): ConnectorRuntimeDialogAction[] {
  switch (outcome.kind) {
    case "fillable":
      return hasResendPayload ? ["saveAndResend", "saveOnly"] : ["saveOnly"]
    case "met":
    case "unsupported_only":
    case "nothing_fillable":
      return ["acknowledge"]
    default: {
      // Listing the acknowledge kinds instead of defaulting to them is what
      // makes this assignment stop compiling once a kind is added to
      // DialogOutcomeKind without a case here, rather than letting the new
      // kind inherit a button set nobody chose for it. The return below
      // still keeps a footer from rendering empty if a value from outside
      // the union reaches this at runtime.
      const unhandled: never = outcome
      void unhandled
      return ["acknowledge"]
    }
  }
}

/**
 * Shared by the dialog's draft state and buildSubmitItems so a draft value
 * written under one key is always read back under the same key. Keyed by the
 * input's full identity -- connector, section, key name and declared type --
 * not just connector and key name: the connector's own edit endpoint can
 * change a key's declared type in place between the report a draft was
 * written against and the next one the dialog reads (no version, no
 * per-task snapshot), and two different sections of the same connector may
 * legitimately reuse a key name. Including type means a stale draft cannot
 * silently survive a type change under a new, unrelated meaning; including
 * section means two same-named rows in different sections never collide,
 * including as React list keys (the dialog reuses this same string there).
 */
export function connectorRuntimeInputDraftKey(
  ref: ConnectorRuntimeRef,
  section: ConnectorRuntimeSection,
  key: string,
  type: ConnectorRuntimeType,
): string {
  return `${ref.connector_type}:${ref.connector_id}:${section}:${key}:${type}`
}

/**
 * Whether a parsed JSON value is an object-typed context draft worth
 * submitting: a plain object (not an array, not null) with at least one key.
 * An empty object parses as valid JSON, but the server's own blank check
 * (`not value` for an object-typed context field) treats `{}` as empty and
 * 400s the whole submission, taking every other filled-in key in the same
 * batch down with it -- so this is the one predicate both the dialog's blur
 * validation and buildSubmitItems below read, instead of each hand-rolling
 * its own "is this submittable" check and drifting apart on `{}`.
 *
 * Built on api-wrapper's isJsonRecord rather than restating the plain-object
 * test, so the dialog -- which needs the two halves apart, to tell an empty
 * object from a value that is no object at all -- reads the same plain-object
 * rule this does instead of a second copy of it.
 */
export function isSubmittableObjectValue(parsed: unknown): boolean {
  return isJsonRecord(parsed) && Object.keys(parsed).length > 0
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
      const rawValue = drafts[connectorRuntimeInputDraftKey(connector.connector_ref, input.section, input.key, input.type)]
      if (rawValue === undefined) continue
      if (input.type === "string") {
        // Submit the trimmed value, not the raw one. The server's merge makes
        // a stored context value immutable, so surrounding whitespace a paste
        // carried in would be written permanently: resubmitting the same
        // secret without it returns 409 runtime_context_immutable, and the
        // only recovery left is a new task.
        const value = rawValue.trim()
        if (value === "") continue
        context[input.key] = value
      } else {
        let parsed: unknown
        try {
          parsed = JSON.parse(rawValue)
        } catch {
          continue
        }
        if (!isSubmittableObjectValue(parsed)) continue
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
 * `hasInvalidObjectDraft` is false. That flag means an invalid-object draft
 * on a row the current report still offers an editable control for --
 * deciding which marks are still live is the caller's job, not this
 * function's; a mark against a row the report no longer renders as editable
 * must not reach this parameter. Never adds "every required key filled" --
 * that would be a second, independently-maintained copy of the server's own
 * completeness rule, the exact drift the design forbids. A malformed
 * key-name warning never participates here either: it is informational, and
 * the row it warns about can still be submitted (the server, not this rule,
 * is the one that will reject it).
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
