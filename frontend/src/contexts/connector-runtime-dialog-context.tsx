"use client"

// A thin state container for the connector-runtime dialog: it holds exactly
// two things -- which task (if any) the dialog is being asked to open a
// request for, and the most recently delivered turn on this tab that a
// dialog could offer to resend. It never issues a request itself, never
// reads the viewed-task/`sendMessage` context (it sits above that provider in
// the tree and cannot reach it), and never looks at the route.
//
// A widget or share guest never reaches this provider, and that is a
// structural fact rather than a runtime check, backed by three independent
// layers: (1) the shell that mounts this provider takes an early-return
// branch for widget/share paths before this provider's tree exists at all,
// so a consumer there gets the no-provider default below, not a live
// instance; (2) a public conversation-create path answers the connector
// runtime report field with a constant null rather than resolving anything,
// and a public page builds its own task through its own request path,
// bypassing the authenticated create flow this dialog listens on; (3) even if
// both of those were bypassed, a guest's access token is a distinct token
// type that the per-task read/write endpoints' own auth dependency rejects
// before any owner check runs. If a future change ever moves this provider
// above the widget/share branch, or points the per-task endpoints at a public
// or share dependency instead of the authenticated one, layers (1) and (3)
// stop holding at the same time, and this dialog's page-scoping must be
// re-examined from scratch rather than assumed to still isolate guests.
import React, { createContext, useContext, useEffect, useMemo, useState } from "react"

import { useAuth } from "@/contexts/auth-context"

export interface ConnectorRuntimeResendPayload {
  taskId: number
  clientMessageId: string
  text: string
  files: File[]
}

export type ConnectorRuntimeDialogCloseOutcome = "dismissed" | "resent" | "not-shown" | "left-host"

export interface ConnectorRuntimeDialogRequest {
  taskId: number
  seq: number
  resendPayload: ConnectorRuntimeResendPayload | null
}

interface ConnectorRuntimeDialogState {
  seq: number
  request: ConnectorRuntimeDialogRequest | null
  payload: ConnectorRuntimeResendPayload | null
}

export interface ConnectorRuntimeDialogActions {
  openForTask: (taskId: number) => void
  close: (outcome: ConnectorRuntimeDialogCloseOutcome) => void
  recordDelivery: (delivery: {
    taskId: number
    clientMessageId: string
    text: string
    files?: File[]
  }) => void
  retainOnlyTask: (taskId: number | null) => void
  forgetDelivery: (taskId: number) => void
}

export interface ConnectorRuntimeDialogValue {
  request: ConnectorRuntimeDialogRequest | null
  payload: ConnectorRuntimeResendPayload | null
}

// Matches file-access-context.tsx's shape for a capability that may have no
// provider above it: a consumer outside this provider's tree (a widget or
// share page) gets an object whose functions do nothing and whose fields are
// always null, not a thrown error.
const NOOP_ACTIONS: ConnectorRuntimeDialogActions = {
  openForTask: () => {},
  close: () => {},
  recordDelivery: () => {},
  retainOnlyTask: () => {},
  forgetDelivery: () => {},
}

const NOOP_VALUE: ConnectorRuntimeDialogValue = { request: null, payload: null }

const ConnectorRuntimeDialogActionsContext =
  createContext<ConnectorRuntimeDialogActions>(NOOP_ACTIONS)
const ConnectorRuntimeDialogStateContext =
  createContext<ConnectorRuntimeDialogValue>(NOOP_VALUE)

const INITIAL_STATE: ConnectorRuntimeDialogState = { seq: 0, request: null, payload: null }

export function ConnectorRuntimeDialogProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<ConnectorRuntimeDialogState>(INITIAL_STATE)
  const userId = useAuth().user?.id ?? null

  // A signed-in identity change (logout, or another tab switching accounts)
  // clears both the request and the stash. This also runs on mount, when
  // both are already empty, so it is harmless there; React's strict-mode
  // double-invoke of effects is likewise harmless for the same reason.
  useEffect(() => {
    setState(prev =>
      prev.request === null && prev.payload === null ? prev : { ...prev, request: null, payload: null },
    )
  }, [userId])

  const actions = useMemo<ConnectorRuntimeDialogActions>(() => ({
    openForTask: (taskId) => {
      if (!Number.isInteger(taskId) || taskId <= 0) return
      setState(prev => {
        // Hand the stash to the request and clear it in the same update: the
        // frame that opens the dialog also settles the turn, so the handoff
        // and the clearing cannot be two separate setState calls without a
        // window where a "save and resend" click would read an
        // already-cleared stash.
        const claimed = prev.payload?.taskId === taskId ? prev.payload : null
        // A second request for a task whose dialog is already open (or
        // already read) keeps whatever snapshot the first request carried,
        // so a second tab's broadcast frame cannot make an in-flight resend
        // button disappear.
        const kept = prev.request?.taskId === taskId ? prev.request.resendPayload : null
        const seq = prev.seq + 1
        return {
          seq,
          request: { taskId, seq, resendPayload: claimed ?? kept },
          payload: claimed ? null : prev.payload,
        }
      })
    },
    close: (outcome) => {
      setState((prev) => {
        if (prev.request === null) return prev
        // "dismissed": the user ended a dialog they saw (close, Esc, Got it,
        //   or a save that settled it without a resend) -- drop the stash too.
        // "resent": the stash now holds the turn that was just resent; keep it.
        // "not-shown": the dialog never became visible; keep the stash.
        // "left-host": the user navigated off the host pages after seeing
        //   it; they did not choose to give up, so keep the stash.
        return { ...prev, request: null, payload: outcome === "dismissed" ? null : prev.payload }
      })
    },
    recordDelivery: (delivery) => {
      setState(prev => ({
        ...prev,
        payload: {
          taskId: delivery.taskId,
          clientMessageId: delivery.clientMessageId,
          text: delivery.text,
          files: delivery.files ?? [],
        },
      }))
    },
    retainOnlyTask: (taskId) => {
      setState((prev) => {
        const nextRequest = prev.request && prev.request.taskId === taskId ? prev.request : null
        const nextPayload = prev.payload && prev.payload.taskId === taskId ? prev.payload : null
        if (nextRequest === prev.request && nextPayload === prev.payload) return prev
        return { ...prev, request: nextRequest, payload: nextPayload }
      })
    },
    forgetDelivery: (taskId) => {
      setState(prev => (prev.payload?.taskId === taskId ? { ...prev, payload: null } : prev))
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [])

  const value = useMemo<ConnectorRuntimeDialogValue>(
    () => ({ request: state.request, payload: state.payload }),
    [state.request, state.payload],
  )

  return (
    <ConnectorRuntimeDialogActionsContext.Provider value={actions}>
      <ConnectorRuntimeDialogStateContext.Provider value={value}>
        {children}
      </ConnectorRuntimeDialogStateContext.Provider>
    </ConnectorRuntimeDialogActionsContext.Provider>
  )
}

/** The identity-stable half: safe to hold in a ref or skip from a dependency
 *  array, and safe for AppProvider to read without joining the state half's
 *  re-render cycle. */
export function useConnectorRuntimeDialogActions(): ConnectorRuntimeDialogActions {
  return useContext(ConnectorRuntimeDialogActionsContext)
}

export function useConnectorRuntimeDialog(): ConnectorRuntimeDialogActions & ConnectorRuntimeDialogValue {
  const actions = useContext(ConnectorRuntimeDialogActionsContext)
  const value = useContext(ConnectorRuntimeDialogStateContext)
  return { ...actions, ...value }
}
