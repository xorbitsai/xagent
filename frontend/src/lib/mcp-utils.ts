/**
 * Shared utilities for MCP Server and Custom API configurations.
 */

import type { TranslationKey } from "@/i18n/translations"

type JsonObject = Record<string, unknown>
type RuntimeBindings = JsonObject[]
const MASKED_SECRET_VALUE = "********"

// Transports whose connectors may carry signed-upload settings. Must equal
// SIGNED_UPLOAD_TRANSPORTS in
// src/xagent/core/tools/adapters/vibe/signed_upload.py; a test compares them.
// Narrower than the MCP OAuth transports: websocket is not one of them.
export const SIGNED_UPLOAD_TRANSPORTS: ReadonlySet<string> = new Set(["sse", "streamable_http"])

export function isSignedUploadTransport(transport: unknown): boolean {
  return typeof transport === "string" && SIGNED_UPLOAD_TRANSPORTS.has(transport)
}

export interface CustomApiDetail {
  id: number
  name: string
  description: string | null
  url: string | null
  method: string | null
  headers: Record<string, string> | null
  body: string | null
  env: Record<string, string> | null
  runtime_input_schema: JsonObject | null
  runtime_bindings: RuntimeBindings | null
  allow_delegated_authorization: boolean
}

export interface CustomApiEditFormData {
  name: string
  transport: "custom_api"
  description: string
  url: string
  method: string
  headers: Record<string, string>
  body: string
  config: Record<string, unknown>
  runtime_input_schema: JsonObject | null
  runtime_bindings: RuntimeBindings | null
  allow_delegated_authorization: boolean
}

export interface CustomApiEditState {
  formData: CustomApiEditFormData
  env: { key: string; value: string }[]
}

export interface McpServerDetail {
  id: number
  user_id: number
  name: string
  transport: string
  description: string | null
  config: JsonObject
  user_env: Record<string, string> | null
  runtime_input_schema: JsonObject | null
  runtime_bindings: RuntimeBindings | null
  allow_delegated_authorization: boolean
  can_edit_global: boolean
  signed_uploads: unknown[] | null
  signed_uploads_error: string | null
}

export interface McpServerEditFormData {
  name: string
  transport: string
  description: string
  config: JsonObject
  user_env: Record<string, string>
  can_edit_global: boolean
  runtime_input_schema: JsonObject | null
  runtime_bindings: RuntimeBindings | null
  allow_delegated_authorization: boolean
  signed_uploads: unknown[] | null
  signed_uploads_error: string | null
}

export interface McpServerEditState {
  formData: McpServerEditFormData
}

type CustomApiMutationPayload = Partial<{
  name: string
  description: string
  url: string
  method: string
  headers: Record<string, string>
  body: string
  env: Record<string, string>
  runtime_input_schema: JsonObject | null
  runtime_bindings: RuntimeBindings | null
  allow_delegated_authorization: boolean
}>

type McpServerMutationPayload = Partial<{
  name: string
  transport: string
  description: string
  config: JsonObject
  user_env: Record<string, string>
  runtime_input_schema: JsonObject | null
  runtime_bindings: RuntimeBindings | null
  allow_delegated_authorization: boolean
}>

export type McpServerPayload = McpServerMutationPayload & { signed_uploads?: JsonObject[] }

interface CustomApiPayloadFormData {
  name?: unknown
  description?: unknown
  url?: unknown
  method?: unknown
  headers?: unknown
  body?: unknown
  runtime_input_schema?: unknown
  runtime_bindings?: unknown
  allow_delegated_authorization?: unknown
}

interface McpServerPayloadFormData {
  name?: unknown
  transport?: unknown
  description?: unknown
  config?: unknown
  user_env?: unknown
  runtime_input_schema?: unknown
  runtime_bindings?: unknown
  allow_delegated_authorization?: unknown
  signed_uploads?: unknown
}

export function isValidMcpName(name: string): boolean {
  const nameRegex = /^[a-zA-Z0-9_-]+$/
  return nameRegex.test(name.trim())
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function isStringRecord(value: unknown): value is Record<string, string> {
  return isObject(value) && Object.values(value).every((item) => typeof item === "string")
}

function hasOwn(object: JsonObject, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(object, key)
}

function cloneJson<T>(value: T): T {
  if (Array.isArray(value)) return value.map((item) => cloneJson(item)) as T
  if (isObject(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, cloneJson(item)]),
    ) as T
  }
  return value
}

function invalidDetail(field: string): never {
  throw new Error(`Invalid Custom API detail response: ${field}`)
}

function invalidMcpDetail(field: string): never {
  throw new Error(`Invalid MCP server detail response: ${field}`)
}

/**
 * Validate the authoritative Custom API detail before it becomes editable state.
 * In particular, runtime fields must be present; accepting their UI defaults here
 * would recreate the response-mapper data-loss failure this boundary prevents.
 */
export function parseCustomApiDetail(value: unknown): CustomApiDetail {
  if (!isObject(value)) invalidDetail("expected an object")

  if (!Number.isInteger(value.id)) invalidDetail("id")
  if (typeof value.name !== "string") invalidDetail("name")
  if (value.description !== null && typeof value.description !== "string") invalidDetail("description")
  if (value.url !== null && typeof value.url !== "string") invalidDetail("url")
  if (value.method !== null && typeof value.method !== "string") invalidDetail("method")
  if (value.headers !== null && !isStringRecord(value.headers)) invalidDetail("headers")
  if (value.body !== null && typeof value.body !== "string") invalidDetail("body")
  if (value.env !== null && !isStringRecord(value.env)) invalidDetail("env")

  if (!hasOwn(value, "runtime_input_schema")) invalidDetail("runtime_input_schema is missing")
  if (value.runtime_input_schema !== null && !isObject(value.runtime_input_schema)) {
    invalidDetail("runtime_input_schema")
  }

  if (!hasOwn(value, "runtime_bindings")) invalidDetail("runtime_bindings is missing")
  if (
    value.runtime_bindings !== null
    && (!Array.isArray(value.runtime_bindings) || !value.runtime_bindings.every(isObject))
  ) {
    invalidDetail("runtime_bindings")
  }

  if (!hasOwn(value, "allow_delegated_authorization")) {
    invalidDetail("allow_delegated_authorization is missing")
  }
  if (typeof value.allow_delegated_authorization !== "boolean") {
    invalidDetail("allow_delegated_authorization")
  }

  return {
    id: value.id as number,
    name: value.name,
    description: value.description as string | null,
    url: value.url as string | null,
    method: value.method as string | null,
    headers: value.headers as Record<string, string> | null,
    body: value.body as string | null,
    env: value.env as Record<string, string> | null,
    runtime_input_schema: value.runtime_input_schema as JsonObject | null,
    runtime_bindings: value.runtime_bindings as RuntimeBindings | null,
    allow_delegated_authorization: value.allow_delegated_authorization,
  }
}

export function customApiDetailToEditState(detail: CustomApiDetail): CustomApiEditState {
  const headers = cloneJson(detail.headers ?? {})
  const envObject = cloneJson(detail.env ?? {})
  const env = Object.entries(envObject).map(([key, value]) => ({ key, value }))

  const url = detail.url ?? ""
  const method = detail.method ?? "GET"
  const body = detail.body ?? ""

  return {
    formData: {
      name: detail.name,
      transport: "custom_api",
      description: detail.description ?? "",
      url,
      method,
      headers,
      body,
      config: {
        url,
        method,
        headers,
        body,
        env: envObject,
      },
      runtime_input_schema: cloneJson(detail.runtime_input_schema),
      runtime_bindings: cloneJson(detail.runtime_bindings),
      allow_delegated_authorization: detail.allow_delegated_authorization,
    },
    env,
  }
}

/**
 * Validate the authoritative MCP detail at the edit boundary. Fields that
 * participate in update semantics are deliberately required so a drifting
 * response mapper cannot be hidden by UI defaults.
 */
export function parseMcpServerDetail(value: unknown): McpServerDetail {
  if (!isObject(value)) invalidMcpDetail("expected an object")

  if (!Number.isInteger(value.id)) invalidMcpDetail("id")
  if (!Number.isInteger(value.user_id)) invalidMcpDetail("user_id")
  if (typeof value.name !== "string") invalidMcpDetail("name")
  if (typeof value.transport !== "string") invalidMcpDetail("transport")
  if (value.description !== null && typeof value.description !== "string") {
    invalidMcpDetail("description")
  }
  if (!isObject(value.config)) invalidMcpDetail("config")

  if (!hasOwn(value, "user_env")) invalidMcpDetail("user_env is missing")
  if (value.user_env !== null && !isStringRecord(value.user_env)) {
    invalidMcpDetail("user_env")
  }

  if (!hasOwn(value, "runtime_input_schema")) {
    invalidMcpDetail("runtime_input_schema is missing")
  }
  if (value.runtime_input_schema !== null && !isObject(value.runtime_input_schema)) {
    invalidMcpDetail("runtime_input_schema")
  }

  if (!hasOwn(value, "runtime_bindings")) invalidMcpDetail("runtime_bindings is missing")
  if (
    value.runtime_bindings !== null
    && (!Array.isArray(value.runtime_bindings) || !value.runtime_bindings.every(isObject))
  ) {
    invalidMcpDetail("runtime_bindings")
  }

  if (!hasOwn(value, "allow_delegated_authorization")) {
    invalidMcpDetail("allow_delegated_authorization is missing")
  }
  if (typeof value.allow_delegated_authorization !== "boolean") {
    invalidMcpDetail("allow_delegated_authorization")
  }

  if (!hasOwn(value, "can_edit_global")) invalidMcpDetail("can_edit_global is missing")
  if (typeof value.can_edit_global !== "boolean") invalidMcpDetail("can_edit_global")

  if (!hasOwn(value, "signed_uploads")) invalidMcpDetail("signed_uploads is missing")
  if (!hasOwn(value, "signed_uploads_error")) invalidMcpDetail("signed_uploads_error is missing")
  if (value.signed_uploads_error !== null && typeof value.signed_uploads_error !== "string") {
    invalidMcpDetail("signed_uploads_error")
  }

  return {
    id: value.id as number,
    user_id: value.user_id as number,
    name: value.name,
    transport: value.transport,
    description: value.description as string | null,
    config: cloneJson(value.config),
    user_env: cloneJson(value.user_env as Record<string, string> | null),
    runtime_input_schema: cloneJson(value.runtime_input_schema as JsonObject | null),
    runtime_bindings: cloneJson(value.runtime_bindings as RuntimeBindings | null),
    allow_delegated_authorization: value.allow_delegated_authorization,
    can_edit_global: value.can_edit_global,
    // The stored value is echoed as it is, valid or not; signed_uploads_error
    // alone says whether the connector can use it. A value that is not a list
    // cannot be shown as entries, so it reads as no settings and the editor
    // still opens; clearing it still sends [] (see signedUploadsMutation).
    signed_uploads: Array.isArray(value.signed_uploads) ? cloneJson(value.signed_uploads) : null,
    signed_uploads_error: value.signed_uploads_error as string | null,
  }
}

export function mcpServerDetailToEditState(detail: McpServerDetail): McpServerEditState {
  return {
    formData: {
      name: detail.name,
      transport: detail.transport,
      description: detail.description ?? "",
      config: cloneJson(detail.config),
      user_env: cloneJson(detail.user_env ?? {}),
      can_edit_global: detail.can_edit_global,
      runtime_input_schema: cloneJson(detail.runtime_input_schema),
      runtime_bindings: cloneJson(detail.runtime_bindings),
      allow_delegated_authorization: detail.allow_delegated_authorization,
      signed_uploads: cloneJson(detail.signed_uploads),
      signed_uploads_error: detail.signed_uploads_error,
    },
  }
}

function envEntriesToObject(
  customApiEnv: { key: string; value: string }[],
  baselineEnv?: Record<string, string> | null,
): Record<string, string> {
  const env: Record<string, string> = {}

  customApiEnv.forEach((entry) => {
    const key = entry.key.trim()
    const value = entry.value.trim()
    if (key && value) env[key] = value
    else if (key && baselineEnv?.[key] === MASKED_SECRET_VALUE) {
      // A transient empty edit is not deletion. Deletion is represented by
      // removing the key from the collection explicitly.
      env[key] = MASKED_SECRET_VALUE
    }
  })

  return env
}

export function deepEqual(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) return true
  if (Array.isArray(left) || Array.isArray(right)) {
    return Array.isArray(left)
      && Array.isArray(right)
      && left.length === right.length
      && left.every((item, index) => deepEqual(item, right[index]))
  }
  if (!isObject(left) || !isObject(right)) return false

  const leftKeys = Object.keys(left)
  const rightKeys = Object.keys(right)
  return leftKeys.length === rightKeys.length
    && leftKeys.every((key) => hasOwn(right, key) && deepEqual(left[key], right[key]))
}

function mutationValues(
  formDataValue: object,
  customApiEnv: { key: string; value: string }[],
  baseline?: CustomApiDetail,
): Required<CustomApiMutationPayload> {
  const formData = formDataValue as CustomApiPayloadFormData
  const runtimeInputSchema = formData.runtime_input_schema === undefined
    ? baseline?.runtime_input_schema ?? null
    : formData.runtime_input_schema
  const runtimeBindings = formData.runtime_bindings === undefined
    ? baseline?.runtime_bindings ?? null
    : formData.runtime_bindings
  const allowDelegatedAuthorization = formData.allow_delegated_authorization === undefined
    ? baseline?.allow_delegated_authorization ?? false
    : formData.allow_delegated_authorization

  if (runtimeInputSchema !== null && !isObject(runtimeInputSchema)) {
    throw new Error("Invalid Custom API form data: runtime_input_schema")
  }
  if (
    runtimeBindings !== null
    && (!Array.isArray(runtimeBindings) || !runtimeBindings.every(isObject))
  ) {
    throw new Error("Invalid Custom API form data: runtime_bindings")
  }
  if (typeof allowDelegatedAuthorization !== "boolean") {
    throw new Error("Invalid Custom API form data: allow_delegated_authorization")
  }

  return {
    name: typeof formData.name === "string" ? formData.name : "",
    description: typeof formData.description === "string" ? formData.description : "",
    url: typeof formData.url === "string" ? formData.url : "",
    method: typeof formData.method === "string" ? formData.method : "GET",
    headers: isStringRecord(formData.headers) ? formData.headers : {},
    body: typeof formData.body === "string" ? formData.body : "",
    env: envEntriesToObject(customApiEnv, baseline?.env),
    runtime_input_schema: runtimeInputSchema,
    runtime_bindings: runtimeBindings,
    allow_delegated_authorization: allowDelegatedAuthorization,
  }
}

function baselineMutationValues(detail: CustomApiDetail): Required<CustomApiMutationPayload> {
  return {
    name: detail.name,
    description: detail.description ?? "",
    url: detail.url ?? "",
    method: detail.method ?? "GET",
    headers: detail.headers ?? {},
    body: detail.body ?? "",
    env: detail.env ?? {},
    runtime_input_schema: detail.runtime_input_schema,
    runtime_bindings: detail.runtime_bindings,
    allow_delegated_authorization: detail.allow_delegated_authorization,
  }
}

function mcpMutationValues(
  formDataValue: object,
  baseline?: McpServerDetail,
): Required<McpServerMutationPayload> {
  const formData = formDataValue as McpServerPayloadFormData
  const configValue = formData.config === undefined ? baseline?.config ?? {} : formData.config
  const userEnvValue = formData.user_env === undefined
    ? baseline?.user_env ?? {}
    : formData.user_env
  const runtimeInputSchema = formData.runtime_input_schema === undefined
    ? baseline?.runtime_input_schema ?? null
    : formData.runtime_input_schema
  const runtimeBindings = formData.runtime_bindings === undefined
    ? baseline?.runtime_bindings ?? null
    : formData.runtime_bindings
  const allowDelegatedAuthorization = formData.allow_delegated_authorization === undefined
    ? baseline?.allow_delegated_authorization ?? false
    : formData.allow_delegated_authorization

  if (!isObject(configValue)) throw new Error("Invalid MCP server form data: config")
  if (!isStringRecord(userEnvValue)) throw new Error("Invalid MCP server form data: user_env")
  if (runtimeInputSchema !== null && !isObject(runtimeInputSchema)) {
    throw new Error("Invalid MCP server form data: runtime_input_schema")
  }
  if (
    runtimeBindings !== null
    && (!Array.isArray(runtimeBindings) || !runtimeBindings.every(isObject))
  ) {
    throw new Error("Invalid MCP server form data: runtime_bindings")
  }
  if (typeof allowDelegatedAuthorization !== "boolean") {
    throw new Error("Invalid MCP server form data: allow_delegated_authorization")
  }

  const config = cloneJson(configValue)
  const baselineConfigEnv = baseline?.config.env
  if (isStringRecord(config.env) && isStringRecord(baselineConfigEnv)) {
    config.env = preserveBlankMaskedValues(config.env, baselineConfigEnv)
  }
  const userEnv = preserveBlankMaskedValues(userEnvValue, baseline?.user_env)

  return {
    name: typeof formData.name === "string" ? formData.name : baseline?.name ?? "",
    transport: typeof formData.transport === "string"
      ? formData.transport
      : baseline?.transport ?? "stdio",
    description: typeof formData.description === "string"
      ? formData.description
      : baseline?.description ?? "",
    config: cloneJson(config),
    user_env: cloneJson(userEnv),
    runtime_input_schema: cloneJson(runtimeInputSchema),
    runtime_bindings: cloneJson(runtimeBindings),
    allow_delegated_authorization: allowDelegatedAuthorization,
  }
}

function preserveBlankMaskedValues(
  current: Record<string, string>,
  baseline?: Record<string, string> | null,
): Record<string, string> {
  return Object.fromEntries(
    Object.entries(current).map(([key, value]) => [
      key,
      value.trim() === "" && baseline?.[key] === MASKED_SECRET_VALUE
        ? MASKED_SECRET_VALUE
        : value,
    ]),
  )
}

function baselineMcpMutationValues(detail: McpServerDetail): Required<McpServerMutationPayload> {
  return {
    name: detail.name,
    transport: detail.transport,
    description: detail.description ?? "",
    config: detail.config,
    user_env: detail.user_env ?? {},
    runtime_input_schema: detail.runtime_input_schema,
    runtime_bindings: detail.runtime_bindings,
    allow_delegated_authorization: detail.allow_delegated_authorization,
  }
}

/**
 * Build a whitelisted Custom API request. Creation emits the complete contract;
 * editing emits only values changed from the authoritative detail baseline.
 */
export function buildCustomApiPayload(
  mcpFormData: object,
  customApiEnv: { key: string; value: string }[],
  baseline?: CustomApiDetail,
): { isValid: boolean; payload: CustomApiMutationPayload; errorKey?: TranslationKey } {
  const current = mutationValues(mcpFormData, customApiEnv, baseline)
  if (!baseline) return { isValid: true, payload: current }

  const original = baselineMutationValues(baseline)
  const payload: CustomApiMutationPayload = {}

  ;(Object.keys(current) as (keyof CustomApiMutationPayload)[]).forEach((key) => {
    if (!deepEqual(current[key], original[key])) {
      ;(payload as Record<string, unknown>)[key] = current[key]
    }
  })

  return { isValid: true, payload }
}

/**
 * The signed_uploads value an MCP save sends, or undefined to leave it out.
 *
 * Kept out of the field-by-field diff. A create sends the field only for sse
 * and streamable_http connectors with at least one entry. An edit that moves
 * a connector off those transports always sends [] so the stored settings are
 * cleared, valid or not: the settings are usually untouched then, so a diff
 * would leave them out and the server would refuse the move. A viewer who
 * cannot edit the shared config never sends it. Otherwise an edit sends it
 * only when it differs from the detail; null and [] are compared as they are,
 * so clearing a stored value that is not a list, which the detail reads as
 * null, still sends [].
 */
function signedUploadsMutation(
  formDataValue: object,
  transport: string,
  baseline?: McpServerDetail,
): JsonObject[] | undefined {
  const formData = formDataValue as McpServerPayloadFormData
  const value = formData.signed_uploads === undefined
    ? baseline?.signed_uploads ?? null
    : formData.signed_uploads
  let next: unknown
  if (!baseline) {
    if (!isSignedUploadTransport(transport) || !Array.isArray(value) || value.length === 0) {
      return undefined
    }
    next = value
  } else if (!baseline.can_edit_global) {
    return undefined
  } else if (!isSignedUploadTransport(transport)) {
    return isSignedUploadTransport(baseline.transport) ? [] : undefined
  } else {
    if (deepEqual(value, baseline.signed_uploads)) return undefined
    next = value ?? []
  }
  if (!Array.isArray(next) || !next.every(isObject)) {
    throw new Error("Invalid MCP server form data: signed_uploads")
  }
  return cloneJson(next)
}

/**
 * Build the single MCP mutation contract. Creation emits the full whitelist;
 * editing emits only changes from the authoritative detail baseline. Per-user
 * env remains a complete replacement whenever it changes, preserving masked
 * baseline entries that the backend resolves to their stored secret values.
 * signed_uploads follows signedUploadsMutation instead of the diff.
 */
export function buildMcpServerPayload(
  mcpFormData: object,
  baseline?: McpServerDetail,
): McpServerPayload {
  const current = mcpMutationValues(mcpFormData, baseline)
  const signedUploads = signedUploadsMutation(mcpFormData, current.transport, baseline)
  if (!baseline) {
    return signedUploads === undefined ? current : { ...current, signed_uploads: signedUploads }
  }

  const original = baselineMcpMutationValues(baseline)
  const payload: McpServerPayload = {}
  ;(Object.keys(current) as (keyof McpServerMutationPayload)[]).forEach((key) => {
    if (!deepEqual(current[key], original[key])) {
      ;(payload as Record<string, unknown>)[key] = current[key]
    }
  })
  if (signedUploads !== undefined) payload.signed_uploads = signedUploads
  return payload
}

// --- MCP OAuth (remote server) connect helpers -----------------------------

export interface McpOAuthConnectResponse {
  authorization_url: string
}

/** Extract a human-readable message from an MCP OAuth endpoint error body.
 * Shared by the custom-server form and the catalog connect dialog. */
export async function parseMcpOAuthErrorMessage(response: Response, fallback: string): Promise<string> {
  try {
    const payload = await response.json()
    if (typeof payload?.detail === "string") return payload.detail
    if (payload?.detail?.message) return payload.detail.message
    if (payload?.detail?.code) return payload.detail.code
  } catch {
    // Keep the provided fallback.
  }
  return fallback
}

/** The window name the mcp_oauth catalog connect popup opens under
 * (connect-mcp-dialog.tsx), and the query param the callback appends on
 * success. Shared so that popup-open side and the tools page's self-close
 * guard can't drift apart on either literal.
 *
 * Not (yet) used by custom-mcp-form.tsx's own OAuth flow: that popup still
 * opens under the reserved target "_blank", so window.name is empty there
 * and this self-close mechanism never fires for it — a pre-existing gap,
 * not something this constant's naming implies is already covered. */
export const MCP_OAUTH_POPUP_WINDOW_NAME = "mcp-oauth"
export const MCP_OAUTH_SUCCESS_PARAM = "mcp_oauth_success"

/** Whether the tools page, loaded with these search params under this window
 * name, is the mcp_oauth connect popup landing on a successful callback and
 * should self-close. Keyed on the explicit success param (not on the absence
 * of error params) so a future error param added to the callback redirect
 * can't be mistaken for success here — see the PR review that flagged this
 * coupling (N5). Extracted as a pure function so it's unit-testable without
 * mounting the full tools page. */
export function shouldSelfCloseMcpOauthPopup(
  windowName: string,
  searchParams: URLSearchParams,
): boolean {
  return (
    searchParams.get(MCP_OAUTH_SUCCESS_PARAM) !== null &&
    windowName === MCP_OAUTH_POPUP_WINDOW_NAME
  )
}
