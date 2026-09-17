import type { TranslationKey } from "@/i18n/translations"

export const CLIENT_ERROR_CODES = [
  "message_processing_failed",
  "task_execution_failed",
  "guidance_in_progress",
  "message_rate_limited",
  "message_id_conflict",
  "message_delivery_failed",
  "message_continuation_unsupported",
  "task_pause_in_progress",
  "message_acceptance_pending",
  "task_unavailable",
  "task_busy",
  "workforce_unavailable",
  "workforce_archived",
  "message_attachment_corrupt",
  "message_attachment_unavailable",
  "task_checkpoint_unreadable",
  "authentication_required",
  "task_access_denied",
  "invalid_message",
  "upload_too_large",
  "upload_proxy_error",
  "upload_failed",
  // Connector-runtime codes that reach the client on a terminal task_error
  // frame's `code` field. They come from V1ErrorCode rather than the backend's
  // ClientErrorCode enum -- this list is already a superset of that enum (the
  // three upload_* codes are client-side only) and stays one table so a code
  // has one wording for every audience. Only codes with a producer that can
  // reach that frame today are listed: a listed code nothing produces is an
  // entry with no expiry date. Which codes those are is a fact about this
  // repository's raise sites, not a property the wire holds -- the field is
  // typed as a bare string and validated against the connector-runtime closed
  // set the server keeps next to its fallback table, and a resolver installed
  // through set_connector_runtime_resolver lives outside this repository and
  // can raise any member of that set. The closed set has eight members, and
  // only five have their own entry below; the other three --
  // connector_not_found, runtime_context_immutable, and
  // runtime_secret_not_allowed -- have no producer that can reach a terminal
  // frame in this repository today, so they keep the generic prefixed
  // wording. Two further connector-runtime codes are outside the closed set
  // entirely (mcp_oauth_authorization_failed, delegated_authorization_failed):
  // each names the outcome of an authorization check, so the server drops
  // them before this frame is built. A code this table does not list keeps
  // the generic prefixed wording.
  "missing_runtime_context",
  "runtime_secret_unavailable",
  "scheduled_secret_unavailable",
  "invalid_runtime_context",
  "connector_runtime_unavailable",
] as const

export type ClientErrorCode = (typeof CLIENT_ERROR_CODES)[number]

const CLIENT_ERROR_TRANSLATION_KEYS: Record<ClientErrorCode, TranslationKey> = {
  message_processing_failed: "clientErrors.messageProcessingFailed",
  task_execution_failed: "clientErrors.taskExecutionFailed",
  guidance_in_progress: "clientErrors.guidanceInProgress",
  message_rate_limited: "clientErrors.messageRateLimited",
  message_id_conflict: "clientErrors.messageIdConflict",
  message_delivery_failed: "clientErrors.messageDeliveryFailed",
  message_continuation_unsupported: "clientErrors.messageContinuationUnsupported",
  task_pause_in_progress: "clientErrors.taskPauseInProgress",
  message_acceptance_pending: "clientErrors.messageAcceptancePending",
  task_unavailable: "clientErrors.taskUnavailable",
  task_busy: "clientErrors.taskBusy",
  workforce_unavailable: "clientErrors.workforceUnavailable",
  workforce_archived: "clientErrors.workforceArchived",
  message_attachment_corrupt: "clientErrors.messageAttachmentCorrupt",
  message_attachment_unavailable: "clientErrors.messageAttachmentUnavailable",
  task_checkpoint_unreadable: "clientErrors.taskCheckpointUnreadable",
  authentication_required: "clientErrors.authenticationRequired",
  task_access_denied: "clientErrors.taskAccessDenied",
  invalid_message: "clientErrors.invalidMessage",
  upload_too_large: "clientErrors.uploadTooLarge",
  upload_proxy_error: "clientErrors.uploadProxyError",
  upload_failed: "clientErrors.uploadFailed",
  missing_runtime_context: "clientErrors.missingRuntimeContext",
  runtime_secret_unavailable: "clientErrors.runtimeSecretUnavailable",
  scheduled_secret_unavailable: "clientErrors.scheduledSecretUnavailable",
  invalid_runtime_context: "clientErrors.invalidRuntimeContext",
  connector_runtime_unavailable: "clientErrors.connectorRuntimeUnavailable",
}

const CLIENT_ERROR_FALLBACKS: Record<ClientErrorCode, string> = {
  message_processing_failed: "The message could not be processed. Please try again.",
  task_execution_failed: "Task execution failed.",
  guidance_in_progress: "A previous guidance message is still being applied. Please wait for it to finish.",
  message_rate_limited: "You're sending messages too quickly. Please wait a moment and try again.",
  message_id_conflict: "Message id was already used for different content or files.",
  message_delivery_failed: "The message could not be delivered. Please retry the draft.",
  message_continuation_unsupported: "Task does not support message continuation.",
  task_pause_in_progress: "Task pause is still being applied; please retry shortly.",
  message_acceptance_pending: "Message acceptance is still being reconciled. Please retry shortly.",
  task_unavailable: "Task is no longer available.",
  task_busy: "Task is currently busy; please wait for the previous turn to finish before sending another message.",
  workforce_unavailable: "This workforce conversation can no longer accept messages; please start a new conversation.",
  workforce_archived: "This workforce has been archived. Unarchive and publish it before starting a new conversation, or select an active workforce.",
  message_attachment_corrupt: "A stored file for this message failed its integrity check and must be re-uploaded.",
  message_attachment_unavailable: "A stored file for this message could not be read. Please try again.",
  task_checkpoint_unreadable: "The task's saved progress could not be read.",
  authentication_required: "Authentication is required to send this message.",
  task_access_denied: "You do not have access to this task.",
  invalid_message: "The message format is invalid.",
  upload_too_large: "File is too large. Please reduce the upload size and try again.",
  upload_proxy_error: "Upload failed before reaching the application. Please check the server upload limit.",
  upload_failed: "Upload failed. Please try again.",
  missing_runtime_context: "This connector needs additional runtime input before it can run.",
  runtime_secret_unavailable: "This connector needs a runtime credential that is not available.",
  scheduled_secret_unavailable: "A scheduled run needs a runtime credential that is not available.",
  invalid_runtime_context: "This connector's runtime input is not valid, so the task could not run.",
  connector_runtime_unavailable: "A service this connector needs is unavailable. Please try again later.",
}

const CLIENT_ERROR_CODE_SET = new Set<string>(CLIENT_ERROR_CODES)

export function readClientErrorCode(value: unknown): ClientErrorCode | null {
  return typeof value === "string" && CLIENT_ERROR_CODE_SET.has(value)
    ? value as ClientErrorCode
    : null
}

export function clientErrorTranslationKey(code: ClientErrorCode): TranslationKey {
  return CLIENT_ERROR_TRANSLATION_KEYS[code]
}

export function clientErrorFallback(code: ClientErrorCode): string {
  return CLIENT_ERROR_FALLBACKS[code]
}
