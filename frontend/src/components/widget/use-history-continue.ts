import { useCallback } from "react"
import {
  useFirstMessageDriver,
  type FirstMessageOutcome,
  type UseFirstMessageDriverInput,
} from "./use-first-message-driver"

// A message the visitor wrote to go on from an earlier conversation.
export interface HistoryContinueRequest {
  seq: number
  message: string
  sourceTaskId: number
}

// "send_failed" means sendContinueMessage rejected.
export type HistoryContinueOutcome = FirstMessageOutcome

export interface UseHistoryContinueInput
  extends Omit<UseFirstMessageDriverInput<HistoryContinueRequest>, "send"> {
  // True while the provider is delivering any message, including a host page
  // open message already being sent; this driver waits for it. An open
  // message still in its own slot does not count: this driver never waits
  // for one. The order runs the other way only through the page: a page that
  // also runs the open message driver passes this hook's `busy` into that
  // driver's `isMessageDeliveryPending`, so an open message waits until the
  // continuation has ended.
  isMessageDeliveryPending: boolean
  sendContinueMessage: (message: string, sourceTaskId: number) => Promise<void>
}

// Sends a message written over an earlier conversation as the first message
// of a new conversation that continues from it, through the first message
// driver, the same one the host page's open messages go through: one message
// at a time, a newer one replacing an older one still in the slot, nothing
// sent until the current socket has attached, a conversation that is still
// bound reset first, a reset at most once and no retry. Every way a message
// ends is reported through onOutcome, except a reset or send that settles
// after unmount, which reports nothing.
//
// `busy` is true from the render a request reaches the slot until its
// message has ended, whether it was sent, failed or dropped.
export function useHistoryContinue({
  sendContinueMessage,
  ...input
}: UseHistoryContinueInput): { busy: boolean } {
  const send = useCallback(
    (request: HistoryContinueRequest) => sendContinueMessage(request.message, request.sourceTaskId),
    [sendContinueMessage],
  )
  return useFirstMessageDriver({ ...input, send })
}
