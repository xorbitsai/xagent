import { useCallback } from "react"
import { W_SEND_FAILED, type WidgetOpenRequest } from "@/hooks/use-widget-open-message"
import {
  useFirstMessageDriver,
  type FirstMessageOutcome,
  type UseFirstMessageDriverInput,
} from "./use-first-message-driver"

export { PHASE_ROUTE } from "./use-first-message-driver"

export const W_UNSUPPORTED = "Xagent Widget: this chat server cannot start a new conversation for XagentWidget.open(); the message was not sent."
export const W_RELOAD_REQUIRED = "Xagent Widget: the chat must be reloaded before XagentWidget.open() can send a message; the message was not sent."
export const W_RESET_FAILED = "Xagent Widget: a new conversation could not be started; the message from XagentWidget.open() was not sent."
export const W_RESET_SUPERSEDED = "Xagent Widget: a new conversation was started, but the chat moved to another conversation before XagentWidget.open() could send its message; the message was not sent."
export const W_SESSION_ENDED = "Xagent Widget: the chat session has ended; the message from XagentWidget.open() was not sent."

// The console warning for each way an open message ends; a sent message
// gives none.
const OUTCOME_WARNING: Readonly<Record<FirstMessageOutcome, string | null>> = {
  sent: null,
  send_failed: W_SEND_FAILED,
  reset_failed: W_RESET_FAILED,
  superseded: W_RESET_SUPERSEDED,
  unsupported: W_UNSUPPORTED,
  reload_required: W_RELOAD_REQUIRED,
  ended: W_SESSION_ENDED,
}

const warnOutcome = (outcome: FirstMessageOutcome) => {
  const warning = OUTCOME_WARNING[outcome]
  if (warning) console.warn(warning)
}

export interface UseSessionOpenMessageInput
  extends Omit<UseFirstMessageDriverInput<WidgetOpenRequest>, "send" | "onOutcome"> {
  sendStartMessage: (message: string) => Promise<void>
}

// Sends the host page's XagentWidget.open(message) as the first message of a
// new conversation, through the first message driver: one message at a time,
// a newer one replacing an older one still in the slot, nothing sent until
// the current socket has attached, a reset at most once and no retry. Waiting
// for the attach means a conversation the server is still replaying is reset
// first instead of receiving the message. Attaching does not prove there is
// no previous conversation: if the server's replay finds nothing to send (its
// task was deleted or changed owner after the attach check), the page stays
// unbound and the message is sent as a first message, exactly as one typed on
// the start screen would be. A message that is not sent ends with its console
// warning, unless its reset or send settles after unmount.
//
// `busy` is true from the render a message reaches the slot until it has
// ended, whether it was sent, failed or dropped.
export function useSessionOpenMessage({
  sendStartMessage,
  ...input
}: UseSessionOpenMessageInput): { busy: boolean } {
  const send = useCallback(
    (request: WidgetOpenRequest) => sendStartMessage(request.message),
    [sendStartMessage],
  )
  return useFirstMessageDriver({ ...input, send, onOutcome: warnOutcome })
}
