import { describe, expect, it } from "vitest"

import {
  clientErrorFallback,
  clientErrorTranslationKey,
  readClientErrorCode,
} from "@/lib/client-errors"

describe("client error wire contract", () => {
  it.each([
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
    ["upload_too_large", "clientErrors.uploadTooLarge", "File is too large. Please reduce the upload size and try again."],
    ["upload_proxy_error", "clientErrors.uploadProxyError", "Upload failed before reaching the application. Please check the server upload limit."],
    ["upload_failed", "clientErrors.uploadFailed", "Upload failed. Please try again."],
  ] as const)("maps %s to a typed translation key", (code, key, fallback) => {
    expect(readClientErrorCode(code)).toBe(code)
    expect(clientErrorTranslationKey(code)).toBe(key)
    expect(clientErrorFallback(code)).toBe(fallback)
  })

  it("rejects unknown and non-string codes", () => {
    expect(readClientErrorCode("provider_secret")).toBeNull()
    expect(readClientErrorCode({ error_code: "upload_failed" })).toBeNull()
  })
})
