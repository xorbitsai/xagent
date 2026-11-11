"use client"

import React, { createContext, useContext, useEffect, useMemo, useRef, useSyncExternalStore } from "react"
import type { VoiceCall } from "@/hooks/use-voice-call"

/**
 * Where a running voice call tells the composer what to show.
 *
 * The call itself lives in a host (the task page's provider, or the task
 * panel) that is not always an ancestor of the composer, so it publishes its
 * state here and the composer reads it. At most one call is shown at a time.
 */
class VoiceCallStore {
  private views = new Map<symbol, VoiceCall>()
  private listeners = new Set<() => void>()
  private snapshot: VoiceCall | null = null

  publish(owner: symbol, view: VoiceCall | null) {
    if (view) this.views.set(owner, view)
    else this.views.delete(owner)
    // Prefer a call that is actually running over one that has only been opened.
    const all = [...this.views.values()]
    const next = all.find(call => call.active) ?? all[0] ?? null
    if (next === this.snapshot) return
    this.snapshot = next
    this.listeners.forEach(listener => listener())
  }

  subscribe = (listener: () => void) => {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }

  get = () => this.snapshot
}

const StoreContext = createContext<VoiceCallStore | null>(null)

export function VoiceCallStoreProvider({ children }: { children: React.ReactNode }) {
  const store = useMemo(() => new VoiceCallStore(), [])
  return <StoreContext.Provider value={store}>{children}</StoreContext.Provider>
}

const NO_LISTENERS = () => () => {}
const NO_CALL = () => null

/** The voice call currently open or running, if any. */
export function useVoiceCallView(): VoiceCall | null {
  const store = useContext(StoreContext)
  return useSyncExternalStore(store?.subscribe ?? NO_LISTENERS, store?.get ?? NO_CALL, NO_CALL)
}

/** Publish `call` for the composer while it is open. */
export function usePublishVoiceCall(call: VoiceCall) {
  const store = useContext(StoreContext)
  const owner = useRef(Symbol("voice-call"))
  useEffect(() => {
    store?.publish(owner.current, call.open ? call : null)
  }, [store, call])
  useEffect(() => {
    const id = owner.current
    return () => store?.publish(id, null)
  }, [store])
}
