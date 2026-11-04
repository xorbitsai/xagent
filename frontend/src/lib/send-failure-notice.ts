import { toast } from "@/components/ui/sonner"
import type { Translate } from "@/contexts/i18n-context"
import { clientErrorTranslationKey, readClientErrorCode } from "@/lib/client-errors"

// Fields a failed send may carry. A thrown value that is not an object has
// none of them.
type SendFailureFields = {
  notificationHandled?: unknown
  connectionLost?: unknown
  errorCode?: unknown
  userFacing?: unknown
}

/**
 * Shows the toast for a chat message that failed to send, so every sender
 * that calls it gives the same notice for the same failure:
 * - nothing when another surface already notified it (`notificationHandled`);
 * - nothing for a lost connection (`connectionLost`) when the caller passes
 *   `suppressConnectionLost`;
 * - otherwise the error code's text, else the message of an `Error` marked
 *   `userFacing`, else the generic "not sent" text.
 */
export function notifySendFailure(
  error: unknown,
  t: Translate,
  { suppressConnectionLost }: { suppressConnectionLost: boolean },
): void {
  const fields: SendFailureFields = typeof error === "object" && error !== null
    ? error as SendFailureFields
    : {}
  if (fields.notificationHandled === true) return
  if (suppressConnectionLost && fields.connectionLost === true) return
  const errorCode = readClientErrorCode(fields.errorCode)
  const userFacing = fields.userFacing === true
  toast.error(errorCode
    ? t(clientErrorTranslationKey(errorCode))
    : userFacing && error instanceof Error
      ? error.message
      : t("builds.list.chat.sendFailed") || "Message was not sent")
}
