import React, { useLayoutEffect, useRef, useState } from "react"
import { useApp } from "@/contexts/app-context-chat"
import { ConnectorConnectCard } from "./ConnectorConnectCard"
import { ConnectorProbeCard } from "./ConnectorProbeCard"
import type { ConnectorInfo } from "./types"

const SEEN_KEY = "connector-probe-seen"

/**
 * True the first time this browser tab sees the interaction, false after a
 * reload: the probe replays once, never on history replay. Storage that is
 * unavailable (private mode, no window) means the probe animates.
 */
function claimProbeReplay(interactionId: string): boolean {
  try {
    const seen: unknown = JSON.parse(window.sessionStorage.getItem(SEEN_KEY) ?? "[]")
    const ids = Array.isArray(seen) ? seen.filter((x): x is string => typeof x === "string") : []
    if (ids.includes(interactionId)) return false
    window.sessionStorage.setItem(SEEN_KEY, JSON.stringify([...ids, interactionId].slice(-50)))
  } catch {
    // storage unavailable
  }
  return true
}

/** Probe card above the connect card; the unit the clarification form renders. */
export function ConnectorInteraction({
  connector, active,
}: {
  connector: ConnectorInfo
  /** The message carrying this card is the one the task is waiting on. */
  active: boolean
}) {
  // Outside the app context (no task state) a card is never pending.
  let waitingTask: { status?: string; waitingInteractions?: Array<{ type: string; interaction_id?: string }> } | null | undefined
  let historyLoading = false
  try {
    const { state } = useApp()
    waitingTask = state?.currentTask
    historyLoading = state?.isHistoryLoading === true
  } catch {
    waitingTask = null
  }
  // Only the interaction the task is waiting on right now may be acted on.
  // While history replays, an older message briefly re-asserts its own waiting
  // state; that is not the task waiting on it now.
  const pending = active && !historyLoading && waitingTask?.status === "waiting_for_user"
    && (waitingTask.waitingInteractions ?? []).some(
      (i) => i.type === "connect_connector" && i.interaction_id === connector.interaction_id,
    )
  // Claimed in an effect, not while rendering, so a re-run render (StrictMode)
  // cannot see its own claim. The ref keeps a StrictMode remount from claiming twice.
  const [animate, setAnimate] = useState(pending)
  const claimedRef = useRef(false)
  useLayoutEffect(() => {
    if (claimedRef.current) return
    claimedRef.current = true
    if (!pending || !claimProbeReplay(connector.interaction_id)) setAnimate(false)
  }, [pending, connector.interaction_id])
  return (
    <div>
      {connector.probe && <ConnectorProbeCard connector={connector} animate={animate} />}
      <ConnectorConnectCard connector={connector} pending={pending} />
    </div>
  )
}
