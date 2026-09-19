"use client"

// A thin state container for the connector-runtime dialog: it holds which
// task (if any) the dialog is being asked to open a request for, the most
// recently delivered turn on this tab that a dialog could offer to resend,
// and the turns handed to the transport but not yet acknowledged as
// delivered (see ConnectorRuntimePendingDelivery below). It never issues a
// request itself, never reads the viewed-task/`sendMessage` context (it sits
// above that provider in the tree and cannot reach it), and never looks at
// the route.
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
  // Written on every recordDelivery call but not read anywhere today -- a
  // resend always mints its own fresh id instead. Reserved for per-turn
  // attribution on a terminal frame, see xorbitsai/xagent#2465.
  clientMessageId: string
  text: string
  files: File[]
}

// A turn handed to the transport but not yet acknowledged as delivered,
// keyed by its own clientMessageId. sendMessage stages one of these right
// before it awaits the delivery acknowledgement, so a terminal frame that
// arrives first (the transport gives no ordering guarantee between the two,
// see the docs above) can still claim it in openForTask. Normally empty or
// one entry per task; kept as an array so the "exactly one in flight"
// check in openForTask is honest about two turns racing for the same task.
export interface ConnectorRuntimePendingDelivery {
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
  pending: ConnectorRuntimePendingDelivery[]
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
  // Called right before sendMessage awaits the delivery acknowledgement for
  // this turn, keyed by clientMessageId. See ConnectorRuntimePendingDelivery.
  stagePendingDelivery: (delivery: {
    taskId: number
    clientMessageId: string
    text: string
    files?: File[]
  }) => void
  // Called when the staged turn above never gets delivered (the send threw,
  // or an early return skipped the write-back) -- withdraws the ticket
  // recordDelivery would otherwise still be waiting to redeem.
  discardPendingDelivery: (clientMessageId: string) => void
}

export interface ConnectorRuntimeDialogValue {
  request: ConnectorRuntimeDialogRequest | null
  payload: ConnectorRuntimeResendPayload | null
}

// Matches file-access-context.tsx's shape for a capability that may have no
// provider above it: a consumer outside this provider's tree (a widget or
// share page) gets an object whose functions do nothing and whose fields are
// always null, not a thrown error.
//
// Doing nothing is the correct production behavior on a widget or share
// page, but it is indistinguishable from a wiring mistake: a component
// mounted as a sibling of the provider rather than inside it -- the shell
// keeps TaskErrorController and VoiceInputController as siblings today --
// would call these and get no dialog, no crash, and no signal. The warning
// below is that signal. It is limited to non-production builds because the
// widget and share pages reach this default legitimately, and it fires once
// per action name so a call from a render loop cannot flood the console.
const reportedNoProviderActions = new Set<string>()

function warnCalledOutsideProvider(action: string): void {
  if (process.env.NODE_ENV === "production" || reportedNoProviderActions.has(action)) return
  reportedNoProviderActions.add(action)
  console.warn(
    `ConnectorRuntimeDialog: ${action}() was called outside ConnectorRuntimeDialogProvider and did nothing.`,
  )
}

// Exported so a consumer that legitimately expects no provider above it (see
// ConnectorRuntimeDialog's own mount effect) can tell that case apart from an
// actual wiring mistake by reference identity, instead of adding a second,
// provider-shaped field to the context value.
export const NOOP_ACTIONS: ConnectorRuntimeDialogActions = {
  openForTask: () => warnCalledOutsideProvider("openForTask"),
  close: () => warnCalledOutsideProvider("close"),
  recordDelivery: () => warnCalledOutsideProvider("recordDelivery"),
  retainOnlyTask: () => warnCalledOutsideProvider("retainOnlyTask"),
  forgetDelivery: () => warnCalledOutsideProvider("forgetDelivery"),
  stagePendingDelivery: () => warnCalledOutsideProvider("stagePendingDelivery"),
  discardPendingDelivery: () => warnCalledOutsideProvider("discardPendingDelivery"),
}

// Silent counterpart to NOOP_ACTIONS: same do-nothing behavior, but without
// the dev-only warning -- for a consumer that is not itself the wiring this
// dialog depends on, and so cannot tell a genuine mistake apart from the
// widget/share pages' expected no-provider shape. useConnectorRuntimeDialogActionsIfMounted
// returns this instead of NOOP_ACTIONS in that case.
const NOOP_ACTIONS_SILENT: ConnectorRuntimeDialogActions = {
  openForTask: () => {},
  close: () => {},
  recordDelivery: () => {},
  retainOnlyTask: () => {},
  forgetDelivery: () => {},
  stagePendingDelivery: () => {},
  discardPendingDelivery: () => {},
}

const NOOP_VALUE: ConnectorRuntimeDialogValue = { request: null, payload: null }

const ConnectorRuntimeDialogActionsContext =
  createContext<ConnectorRuntimeDialogActions>(NOOP_ACTIONS)
const ConnectorRuntimeDialogStateContext =
  createContext<ConnectorRuntimeDialogValue>(NOOP_VALUE)

const INITIAL_STATE: ConnectorRuntimeDialogState = { seq: 0, request: null, payload: null, pending: [] }

export function ConnectorRuntimeDialogProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<ConnectorRuntimeDialogState>(INITIAL_STATE)
  const userId = useAuth().user?.id ?? null

  // A signed-in identity change (logout, or another tab switching accounts)
  // clears the request, the stash, and any pending candidates. This also
  // runs on mount, when all three are already empty, so it is harmless
  // there; React's strict-mode double-invoke of effects is likewise
  // harmless for the same reason.
  useEffect(() => {
    setState(prev =>
      prev.request === null && prev.payload === null && prev.pending.length === 0
        ? prev
        : { ...prev, request: null, payload: null, pending: [] },
    )
  }, [userId])

  const actions = useMemo<ConnectorRuntimeDialogActions>(() => ({
    openForTask: (taskId) => {
      if (!Number.isInteger(taskId) || taskId <= 0) return
      setState(prev => {
        // A staged, not-yet-acknowledged turn for this task outranks the
        // confirmed stash: it is the more recently sent one. Only an
        // unambiguous single in-flight turn can be claimed, though -- the
        // terminal frame carries no turn identity (xorbitsai/xagent#2465),
        // so with two staged turns there is no way to tell which one
        // failed. Ambiguity withholds the whole fallback chain, not just
        // the staged pick: neither staged candidate nor the confirmed
        // stash is offered. The one exception is a snapshot an
        // already-open dialog for this task is already carrying (`kept`)
        // -- that one survives, so a resend button already on screen does
        // not disappear out from under the user. Offering save-only, or
        // leaving that button alone, is the safe degradation; resending
        // the wrong turn is not.
        const mine = prev.pending.filter(p => p.taskId === taskId)
        const ambiguous = mine.length > 1
        const staged = mine.length === 1 ? mine[0] : null
        // Hand the stash to the request and clear it in the same update: the
        // frame that opens the dialog also settles the turn, so the handoff
        // and the clearing cannot be two separate setState calls without a
        // window where a "save and resend" click would read an
        // already-cleared stash.
        const stashed = prev.payload?.taskId === taskId ? prev.payload : null
        // A second request for a task whose dialog is already open (or
        // already read) keeps whatever snapshot the first request carried,
        // so a second tab's broadcast frame cannot make an in-flight resend
        // button disappear.
        const kept = prev.request?.taskId === taskId ? prev.request.resendPayload : null
        const seq = prev.seq + 1
        return {
          seq,
          request: {
            taskId,
            seq,
            resendPayload: ambiguous ? (kept ?? null) : (staged ?? stashed ?? kept),
          },
          payload: stashed ? null : prev.payload,
          pending: prev.pending.filter(p => p.taskId !== taskId),
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
      // Ticket taken up: only a delivery whose clientMessageId matches a
      // staged candidate writes the stash. Without a matching ticket, this
      // acknowledgement belongs to a turn this tab already settled (its
      // pending entry was cleared by openForTask or forgetDelivery), and
      // writing it anyway would resurface a stale candidate for the next
      // failure to wrongly claim -- see the stash lifecycle docs.
      setState(prev => {
        if (!prev.pending.some(p => p.clientMessageId === delivery.clientMessageId)) return prev
        return {
          ...prev,
          pending: prev.pending.filter(p => p.clientMessageId !== delivery.clientMessageId),
          payload: {
            taskId: delivery.taskId,
            clientMessageId: delivery.clientMessageId,
            text: delivery.text,
            files: delivery.files ?? [],
          },
        }
      })
    },
    retainOnlyTask: (taskId) => {
      setState((prev) => {
        const nextRequest = prev.request && prev.request.taskId === taskId ? prev.request : null
        const nextPayload = prev.payload && prev.payload.taskId === taskId ? prev.payload : null
        const nextPending = prev.pending.filter(p => p.taskId === taskId)
        if (
          nextRequest === prev.request
          && nextPayload === prev.payload
          && nextPending.length === prev.pending.length
        ) return prev
        return { ...prev, request: nextRequest, payload: nextPayload, pending: nextPending }
      })
    },
    forgetDelivery: (taskId) => {
      // Drops every pending candidate for this task, not just the one that
      // settled: a settlement frame carries no turn identity, so "which
      // turn just ended" and "which other turn is still in flight" cannot
      // be told apart (xorbitsai/xagent#2465). Clearing all of them trades
      // an interleaved send's still-live candidate for the guarantee that a
      // later failure never resends the wrong message.
      setState(prev => {
        const nextPayload = prev.payload?.taskId === taskId ? null : prev.payload
        const nextPending = prev.pending.some(p => p.taskId === taskId)
          ? prev.pending.filter(p => p.taskId !== taskId)
          : prev.pending
        if (nextPayload === prev.payload && nextPending === prev.pending) return prev
        return { ...prev, payload: nextPayload, pending: nextPending }
      })
    },
    stagePendingDelivery: (delivery) => {
      setState(prev => {
        const entry: ConnectorRuntimePendingDelivery = {
          taskId: delivery.taskId,
          clientMessageId: delivery.clientMessageId,
          text: delivery.text,
          files: delivery.files ?? [],
        }
        // A repeat stage under the same clientMessageId (a retry reusing the
        // caller-supplied id) replaces the earlier entry in place rather
        // than appending a second one.
        const withoutExisting = prev.pending.filter(p => p.clientMessageId !== delivery.clientMessageId)
        return { ...prev, pending: [...withoutExisting, entry] }
      })
    },
    discardPendingDelivery: (clientMessageId) => {
      setState(prev => {
        if (!prev.pending.some(p => p.clientMessageId === clientMessageId)) return prev
        return { ...prev, pending: prev.pending.filter(p => p.clientMessageId !== clientMessageId) }
      })
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

/**
 * Same actions as useConnectorRuntimeDialogActions(), for a call site that
 * may legitimately run with no ConnectorRuntimeDialogProvider above it and
 * does not want the dev-only "called outside provider" warning that
 * combination would otherwise trip -- the chat context is mounted on the
 * widget/share pages, which intentionally omit this provider (see this
 * module's own top-of-file docstring). A real provider's actions object is
 * always distinct by reference from NOOP_ACTIONS (the useMemo below never
 * recreates it), so this can tell the two cases apart without a second
 * context value, and without every call site adding its own reference check.
 */
export function useConnectorRuntimeDialogActionsIfMounted(): ConnectorRuntimeDialogActions {
  const actions = useContext(ConnectorRuntimeDialogActionsContext)
  return actions === NOOP_ACTIONS ? NOOP_ACTIONS_SILENT : actions
}
