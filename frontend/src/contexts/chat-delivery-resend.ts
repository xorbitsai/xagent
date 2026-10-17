import { MessageDeliveryError } from "@/hooks/use-websocket"

// A fixed upper bound on how long a send that lost its connection waits for the
// next open. The send button stays locked while the send waits, so the wait must
// end even though reconnecting itself never gives up. A send that times out here
// fails as it came; a manual retry reuses the same client_message_id, which the
// server deduplicates.
export const RESEND_RECONNECT_DEADLINE_MS = 90_000

// Where a send was made from, captured with the first attempt.
export interface ResendOwnership {
  loginIdentity: string | null
  boundTaskId: number | null
  // Present only for the Session transport.
  session: { deliveryGeneration: number; phase: string } | null
}

// Every check must hold; an unreadable login never matches. A Session send made
// while unbound may go again while still unbound: the Session server never starts
// a second task or turn for a client_message_id it already took on another
// connection of the same session, which a manual same-id retry relies on too.
export const mayResend = (first: ResendOwnership, now: ResendOwnership): boolean => {
  if (first.loginIdentity === null || now.loginIdentity !== first.loginIdentity) return false
  if (first.session === null) return now.boundTaskId === first.boundTaskId
  if (now.session === null || now.session.deliveryGeneration !== first.session.deliveryGeneration) return false
  if (now.session.phase !== "unbound" && now.session.phase !== "bound") return false
  return now.boundTaskId === first.boundTaskId
    || (first.boundTaskId === null && now.boundTaskId !== null)
}

// Sends that lost their connection wait here, first in first out, for the next
// open. Each wait ends at that open, when the owner calls `release`, or at the
// deadline.
export class ResendWaiters {
  private waiting: Array<(reconnected: boolean) => void> = []
  private ended = false

  wait(): Promise<boolean> {
    if (this.ended) return Promise.resolve(false)
    return new Promise((resolve) => {
      const settle = (reconnected: boolean) => {
        clearTimeout(timer)
        this.waiting = this.waiting.filter((entry) => entry !== settle)
        resolve(reconnected)
      }
      const timer = setTimeout(() => settle(false), RESEND_RECONNECT_DEADLINE_MS)
      this.waiting.push(settle)
    })
  }

  // While no reconnect is coming, current and later waits end at once.
  setEnded(ended: boolean) {
    this.ended = ended
    if (ended) this.release(false)
  }

  release(reconnected: boolean) {
    if (reconnected) this.ended = false
    for (const settle of [...this.waiting]) settle(reconnected)
  }
}

// A failed send whose connection was lost is sent once more, with the same
// arguments, after the next open if `stillOwned` still holds. Any other failure,
// and a second failure, is thrown as it came.
export async function sendWithResend<T>(
  send: () => Promise<T>,
  waiters: ResendWaiters,
  stillOwned: () => boolean,
): Promise<T> {
  try {
    return await send()
  } catch (error) {
    if (!(error instanceof MessageDeliveryError) || !error.connectionLost) throw error
    if (!(await waiters.wait()) || !stillOwned()) throw error
    return await send()
  }
}
