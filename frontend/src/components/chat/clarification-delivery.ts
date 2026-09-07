import type { MessageDeliveryDisposition } from "@/hooks/use-websocket"
import type { TranslationKey } from "@/i18n/translations"
import { readClientErrorCode, type ClientErrorCode } from "@/lib/client-errors"

/**
 * The delivery contract shared by ClarificationForm's two send paths (#1485).
 *
 * The internal path (AppContext.sendMessage) and any injected `onSend`
 * provider are held to the same shape: a send is one delivery attempt with a
 * stable identity, and a failed send rejects with a discriminated failure the
 * form can act on. Failures are still probed structurally rather than by
 * `instanceof`, because a provider's rejection need not be an `Error`
 * subclass — the contract is the fields, not the class.
 */

/** Metadata a clarification answer travels with. */
export interface ClarificationSendMetadata {
  request_id?: string
  url?: string
}

/**
 * The identity of one delivery attempt. The form keeps the same
 * `clientMessageId` for an unresolved submission across resubmits (and hands
 * it to both send paths), so a consumer that dedups on it — the internal
 * path's server does — can recognize a retry instead of recording the answer
 * twice. A provider that cannot dedup may ignore it; the form's failure copy
 * only warns about resubmits, it never promises safety.
 */
export interface ClarificationSendAttempt {
  clientMessageId: string
}

export type ClarificationOnSend = (
  message: string,
  files: File[],
  metadata: ClarificationSendMetadata,
  attempt: ClarificationSendAttempt,
) => Promise<void> | void

/**
 * The discriminated failure a send path rejects with — the same shape
 * use-websocket's MessageDeliveryError carries, declared here so an `onSend`
 * provider can construct it without depending on the websocket hook.
 */
export type ClarificationSendFailure = Error & {
  disposition: MessageDeliveryDisposition
  userFacing: boolean
  errorCode: ClientErrorCode | null
  retryWithNewId: boolean
}

export interface ClarificationSendFailureOptions {
  /** Whether `message` is the sender-actionable reason, safe to display. */
  userFacing?: boolean
  errorCode?: ClientErrorCode | null
  /** The attempt's identity is burned; a retry must mint a new one. */
  retryWithNewId?: boolean
}

export const clarificationSendFailure = (
  message: string,
  disposition: MessageDeliveryDisposition,
  options: ClarificationSendFailureOptions = {},
): ClarificationSendFailure => Object.assign(new Error(message), {
  disposition,
  userFacing: options.userFacing ?? false,
  errorCode: options.errorCode ?? null,
  retryWithNewId: options.retryWithNewId ?? false,
})

/**
 * Delivery failures carry whether the turn definitely never reached the agent.
 * Plain errors (local validation, unexpected throws) carry nothing, and are
 * left unqualified rather than guessed at: telling a visitor to resubmit a
 * turn that may have landed is worse than saying nothing.
 */
export const readSendDisposition = (error: unknown): MessageDeliveryDisposition | null => {
  if (typeof error !== "object" || error === null || !("disposition" in error)) {
    return null
  }
  const disposition = (error as { disposition: unknown }).disposition
  return disposition === "not_sent"
    || disposition === "rejected"
    || disposition === "outcome_unknown"
    ? disposition
    : null
}

/**
 * Whether the failure says this attempt's identity must not be reused. Only
 * an explicit `true` counts: an untyped rejection keeps the identity, so an
 * unchanged resubmit stays recognizable as a retry.
 */
export const readSendRetryWithNewId = (error: unknown): boolean =>
  typeof error === "object"
  && error !== null
  && (error as { retryWithNewId?: unknown }).retryWithNewId === true

/**
 * Only the reasons the sender can act on — the backend's rejection text — are
 * shown as-is. Connection plumbing messages stay behind the localized string:
 * they are English diagnostics, and a widget visitor is not the audience for
 * them. Nothing new becomes displayable through an arbitrary provider either
 * way — `userFacing` still has to be set.
 */
export const readSendReason = (error: unknown): string => {
  if (
    typeof error !== "object"
    || error === null
    || (error as { userFacing?: unknown }).userFacing !== true
  ) {
    return ""
  }
  const message = (error as { message?: unknown }).message
  return typeof message === "string" ? message.trim() : ""
}

export const readSendErrorCode = (error: unknown): ClientErrorCode | null => {
  if (typeof error !== "object" || error === null) return null
  return readClientErrorCode((error as { errorCode?: unknown }).errorCode)
}

/**
 * The hint that belongs with a disposition, as a key rather than a translated
 * string: the toast needs it once at failure time, while the persistent alert
 * has to re-resolve it on every render so a locale switch is not stuck behind
 * whatever language was active when the send failed.
 */
export const sendHintKey = (
  disposition: MessageDeliveryDisposition | null,
): TranslationKey | null => disposition === "outcome_unknown"
  ? "chatPage.clarification.sendOutcomeUnknown"
  : disposition === "not_sent" || disposition === "rejected"
    ? "chatPage.clarification.sendNotSent"
    : null
