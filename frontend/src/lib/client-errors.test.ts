import { describe, expect, it } from "vitest"

import en from "@/i18n/locales/en"
import {
  CLIENT_ERROR_CODES,
  clientErrorFallback,
  clientErrorTranslationKey,
  readClientErrorCode,
  readServerRejectionCode,
  type ClientErrorCode,
} from "@/lib/client-errors"

// The codes this client derives itself. A server rejection frame never
// legitimately names one, so readServerRejectionCode drops it; every other
// listed code is a server code and passes through unchanged.
const CLIENT_ONLY_CODES = new Set<ClientErrorCode>([
  "upload_too_large",
  "upload_proxy_error",
  "upload_failed",
  "upload_rate_limited",
  "upload_quota_exceeded",
  "upload_rejected",
  "connection_not_ready",
  "message_not_accepted",
])

describe("client error wire contract", () => {
  it.each([
    ["execution_queue_full", "clientErrors.executionQueueFull", "The team execution queue is full. Please retry shortly."],
    ["message_processing_failed", "clientErrors.messageProcessingFailed", "The message could not be processed. Please try again."],
    ["task_execution_failed", "clientErrors.taskExecutionFailed", "Task execution failed."],
    ["guidance_in_progress", "clientErrors.guidanceInProgress", "A previous guidance message is still being applied. Please wait for it to finish."],
    ["message_rate_limited", "clientErrors.messageRateLimited", "You're sending messages too quickly. Please wait a moment and try again."],
    ["message_id_conflict", "clientErrors.messageIdConflict", "Message id was already used for different content or files."],
    ["message_delivery_failed", "clientErrors.messageDeliveryFailed", "The message could not be delivered. Please retry the draft."],
    ["message_continuation_unsupported", "clientErrors.messageContinuationUnsupported", "Task does not support message continuation."],
    ["task_pause_in_progress", "clientErrors.taskPauseInProgress", "Task pause is still being applied; please retry shortly."],
    ["message_acceptance_pending", "clientErrors.messageAcceptancePending", "Message acceptance is still being reconciled. Please retry shortly."],
    ["task_unavailable", "clientErrors.taskUnavailable", "Task is no longer available."],
    ["task_busy", "clientErrors.taskBusy", "Task is currently busy; please wait for the previous turn to finish before sending another message."],
    ["workforce_unavailable", "clientErrors.workforceUnavailable", "This workforce conversation can no longer accept messages; please start a new conversation."],
    ["workforce_archived", "clientErrors.workforceArchived", "This workforce has been archived. Unarchive and publish it before starting a new conversation, or select an active workforce."],
    ["message_attachment_corrupt", "clientErrors.messageAttachmentCorrupt", "A stored file for this message failed its integrity check and must be re-uploaded."],
    ["message_attachment_unavailable", "clientErrors.messageAttachmentUnavailable", "A stored file for this message could not be read. Please try again."],
    ["task_checkpoint_unreadable", "clientErrors.taskCheckpointUnreadable", "The task's saved progress could not be read."],
    ["authentication_required", "clientErrors.authenticationRequired", "Authentication is required to send this message."],
    ["task_access_denied", "clientErrors.taskAccessDenied", "You do not have access to this task."],
    ["invalid_message", "clientErrors.invalidMessage", "The message format is invalid."],
    ["message_outcome_unknown", "clientErrors.messageOutcomeUnknown", "The message may or may not have been applied. Check the conversation before sending it again."],
    ["external_turn_interrupted", "clientErrors.externalTurnInterrupted", "This response was interrupted."],
    ["external_stop_not_applied", "clientErrors.externalStopNotApplied", "Stopping this response didn't go through — please try again."],
    ["upload_too_large", "clientErrors.uploadTooLarge", "File is too large. Please reduce the upload size and try again."],
    ["upload_proxy_error", "clientErrors.uploadProxyError", "Upload failed before reaching the application. Please check the server upload limit."],
    ["upload_failed", "clientErrors.uploadFailed", "Upload failed. Please try again."],
    ["upload_rate_limited", "clientErrors.uploadRateLimited", "The file could not be uploaded right now. Please try again in a minute."],
    ["upload_quota_exceeded", "clientErrors.uploadQuotaExceeded", "You have reached the file limit for this chat."],
    ["upload_rejected", "clientErrors.uploadRejected", "This file can't be uploaded. Check its name and type."],
    ["connection_not_ready", "clientErrors.connectionNotReady", "Not connected. Your message was not sent."],
    ["message_not_accepted", "clientErrors.messageNotAccepted", "The message was not accepted."],
    ["missing_runtime_context", "clientErrors.missingRuntimeContext", "This connector needs additional runtime input before it can run."],
    ["runtime_secret_unavailable", "clientErrors.runtimeSecretUnavailable", "This connector needs a runtime credential that is not available."],
    ["scheduled_secret_unavailable", "clientErrors.scheduledSecretUnavailable", "A scheduled run needs a runtime credential that is not available."],
    ["invalid_runtime_context", "clientErrors.invalidRuntimeContext", "This connector's runtime input is not valid, so the task could not run."],
    ["connector_runtime_unavailable", "clientErrors.connectorRuntimeUnavailable", "A service this connector needs is unavailable. Please try again later."],
  ] as const)("maps %s to a typed translation key", (code, key, fallback) => {
    expect(readClientErrorCode(code)).toBe(code)
    expect(clientErrorTranslationKey(code)).toBe(key)
    expect(clientErrorFallback(code)).toBe(fallback)
  })

  it("rejects unknown and non-string codes", () => {
    expect(readClientErrorCode("provider_secret")).toBeNull()
    expect(readClientErrorCode({ error_code: "upload_failed" })).toBeNull()
  })

  it.each(
    CLIENT_ERROR_CODES.map((code) => [code, CLIENT_ONLY_CODES.has(code) ? null : code] as const),
  )("readServerRejectionCode reads %s as %s", (code, expected) => {
    expect(readServerRejectionCode(code)).toBe(expected)
  })

  it.each(["upload_rate_limited", "upload_quota_exceeded", "upload_rejected"])(
    "never accepts %s from a server rejection frame",
    (code) => {
      expect(readClientErrorCode(code)).toBe(code)
      expect(readServerRejectionCode(code)).toBeNull()
    },
  )

  it("keeps the fallback strings identical to the English locale", () => {
    for (const code of CLIENT_ERROR_CODES) {
      const translationKey = clientErrorTranslationKey(code)
      const localeKey = translationKey.replace(
        /^clientErrors\./,
        "",
      ) as keyof typeof en.clientErrors
      expect(clientErrorFallback(code)).toBe(en.clientErrors[localeKey])
    }
  })
})
