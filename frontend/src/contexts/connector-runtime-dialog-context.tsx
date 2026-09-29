"use client"

// A thin state container for the connector-runtime dialog: it holds which
// task (if any) the dialog is being asked to open a request for, the most
// recently delivered turn on this tab that a dialog could offer to resend,
// the turns handed to the transport but not yet acknowledged as delivered
// (see ConnectorRuntimePendingDelivery below), the task whose dialog the user
// closed (dismissedCheck), and the account of each first message a create path
// holds behind a first gate (gates). It never issues a request itself, never
// reads the viewed-task/`sendMessage` context (it sits above that provider in
// the tree and cannot reach it), and never looks at the route.
//
// A held message's ending: only transitionRequest writes the account, in the
// step that moves its request; one effect hands each decision over once the
// state carrying it is committed, answering "cleared" for any entry still held
// by a request that is not the committed one.
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
import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react"

import { useAuth } from "@/contexts/auth-context"
import type { ConnectorRuntimeDialogTrigger } from "@/lib/connector-runtime-api"

export interface ConnectorRuntimeResendPayload {
  taskId: number
  // Written on every recordDelivery call. connector-runtime-dialog.tsx's
  // doResend reads this to bind the id it carries between resend attempts to
  // the snapshot that id was minted for, so a same-task retarget that swaps
  // in a different snapshot cannot carry an old id over onto different text.
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

// Why the task page asks: it started viewing the task, or its connection came back.
export type SessionCheckCause = "opened" | "reconnected"

export interface ConnectorRuntimeDialogRequest {
  taskId: number
  seq: number
  // Only a turn_failure request carries one: a session check has no message,
  // and a first gate's message stays with the create path holding it.
  resendPayload: ConnectorRuntimeResendPayload | null
  trigger: ConnectorRuntimeDialogTrigger
  // Non-null exactly when trigger is "first_gate": the message it holds.
  gateId: number | null
}

// How a held first message ends. `released`: the dialog let it go (nothing to
// fill, a failed read, a save or "send", or leaving before it was visible).
// `discarded`: the user closed it. `cleared`: the request went away for any
// other reason (task switch, sign-out, replaced, left after it was visible).
export type FirstGateDecision = "released" | "discarded" | "cleared"

// One held first message's account; `decision: null` means still held.
export interface HeldGate { id: number; decision: FirstGateDecision | null }

export interface ConnectorRuntimeDialogState {
  seq: number
  request: ConnectorRuntimeDialogRequest | null
  payload: ConnectorRuntimeResendPayload | null
  pending: ConnectorRuntimePendingDelivery[]
  // Whose dialog the user closed this view (see transitionRequest).
  dismissedCheck: number | null
  // Written only by transitionRequest (accountGates, gates-drained).
  gates: readonly HeldGate[]
}

export interface ConnectorRuntimeDialogActions {
  openForTask: (taskId: number) => void
  openSessionCheck: (taskId: number, cause: SessionCheckCause) => void
  // How a new task's held first message ends; null (hold nothing) for an
  // invalid task id or with no provider mounted.
  openFirstGate: (taskId: number) => Promise<FirstGateDecision> | null
  // Closes the request only if it is still for `taskId`, the closing dialog's.
  close: (outcome: ConnectorRuntimeDialogCloseOutcome, taskId: number) => void
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
  openSessionCheck: () => warnCalledOutsideProvider("openSessionCheck"),
  openFirstGate: () => { warnCalledOutsideProvider("openFirstGate"); return null },
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
  openSessionCheck: () => {},
  openFirstGate: () => null,
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

// Everything that can move the current request, as data.
export type RequestInput =
  | { type: "open"; taskId: number; trigger: "turn_failure" }
  | { type: "open"; taskId: number; trigger: "session_open"; cause: SessionCheckCause }
  | { type: "open"; taskId: number; trigger: "first_gate"; gateId: number }
  | { type: "close"; taskId: number; outcome: ConnectorRuntimeDialogCloseOutcome }
  | { type: "retain"; taskId: number | null }
  | { type: "forget"; taskId: number }
  | { type: "identity-changed" }
  | { type: "gates-drained"; ids: readonly number[] }

// Pure, and the one place the current request and the held-message account
// are written: every rule for how one is opened, closed or dropped reads (and
// is tested) as one table. React may run an updater more than once, so nothing
// here may have a side effect: a decision is recorded, and handed over later.
export function transitionRequest(prev: ConnectorRuntimeDialogState, input: RequestInput): ConnectorRuntimeDialogState {
  if (input.type !== "gates-drained") return accountGates(prev, moveRequest(prev, input), input)
  const gates = prev.gates.filter(g => !input.ids.includes(g.id))
  // Keeps the request and payload objects, so no subscriber re-renders.
  return gates.length === prev.gates.length ? prev : { ...prev, gates }
}

const CLOSE_DECISIONS: Record<ConnectorRuntimeDialogCloseOutcome, FirstGateDecision> = {
  "resent": "released", "not-shown": "released", "dismissed": "discarded", "left-host": "cleared",
}

// Whatever step takes a first gate's request away records its decision in that
// same step, so no input can drop a held message unanswered. Whatever `gates`
// moveRequest returns is ignored: they are carried over from `prev`, even past
// a branch that builds its next state from scratch.
function accountGates(prev: ConnectorRuntimeDialogState, next: ConnectorRuntimeDialogState, input: RequestInput): ConnectorRuntimeDialogState {
  const before = prev.request?.gateId ?? null
  const after = next.request?.gateId ?? null
  if (before === after) return next.gates === prev.gates ? next : { ...next, gates: prev.gates }
  let gates = prev.gates
  if (before !== null) {
    const decision = input.type === "close" ? CLOSE_DECISIONS[input.outcome] : "cleared"
    gates = gates.map(g => (g.id === before && g.decision === null ? { ...g, decision } : g))
  }
  if (after !== null) gates = [...gates, { id: after, decision: null }]
  return { ...next, gates }
}

// The answers due once `state` is committed: every recorded decision, and, as
// a backstop for a write that bypassed accountGates, "cleared" for an entry
// still held by a request that is not the committed one (a hang made visible).
export function gatesToSettle(state: ConnectorRuntimeDialogState): Array<{ id: number; decision: FirstGateDecision }> {
  return state.gates.flatMap(g => g.decision !== null ? [{ id: g.id, decision: g.decision }]
    : g.id !== state.request?.gateId ? [{ id: g.id, decision: "cleared" as const }] : [])
}

function moveRequest(prev: ConnectorRuntimeDialogState, input: Exclude<RequestInput, { type: "gates-drained" }>): ConnectorRuntimeDialogState {
  switch (input.type) {
    case "open": {
      const { taskId } = input
      if (input.trigger === "session_open") {
        // A session check never touches the stash or the tickets, and yields to
        // a message on its way: a ticket (queued or unacknowledged), or on a
        // fresh view a stash (delivered, turn not ended). A reconnect ignores
        // the stash, since a failure frame lost while disconnected is never
        // replayed, but stays quiet for a dialog the user closed this view.
        const { cause } = input
        const base = cause === "opened" && prev.dismissedCheck !== null ? { ...prev, dismissedCheck: null } : prev
        if (
          base.request?.taskId === taskId
          || base.pending.some(p => p.taskId === taskId)
          || (cause === "opened" ? base.payload?.taskId === taskId : base.dismissedCheck === taskId)
        ) return base
        const seq = base.seq + 1
        return { ...base, seq, request: { taskId, seq, resendPayload: null, trigger: "session_open", gateId: null } }
      }
      if (input.trigger === "first_gate") {
        // The held message has no ticket or stash entry; this touches neither.
        const seq = prev.seq + 1
        return { ...prev, seq, request: { taskId, seq, resendPayload: null, trigger: "first_gate", gateId: input.gateId } }
      }
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
      // Clear the stash in the same update that builds the request. When
      // this frame's snapshot comes from the stash, the handoff and the
      // clearing cannot be two separate setState calls without a window
      // where a "save and resend" click would read an already-cleared
      // stash. The other two branches clear it with no handoff at all --
      // a staged turn outranks it, or the frame is ambiguous and nothing
      // is claimed -- for forgetDelivery's reason rather than this one:
      // the frame carries no turn identity (xorbitsai/xagent#2465), so a
      // stash left behind is a turn this very frame may be about. It
      // would then survive a "not-shown" or "left-host" close, be
      // claimed unambiguously by the next frame for this task, and be
      // offered as a one-click resend of a turn nobody could confirm had
      // failed.
      const stashed = prev.payload?.taskId === taskId ? prev.payload : null
      // A held first message is not on the wire, so this frame is about some
      // other message: the request stays as it is (`seq` too, so a save in
      // flight is not superseded) and the frame claims nothing, as if ambiguous.
      if (prev.request?.taskId === taskId && prev.request.trigger === "first_gate") {
        return { ...prev, payload: stashed ? null : prev.payload, pending: prev.pending.filter(p => p.taskId !== taskId) }
      }
      // A second request for a task whose dialog is already open (or
      // already read) keeps whatever snapshot the first request carried,
      // so a second tab's broadcast frame cannot make an in-flight resend
      // button disappear.
      const kept = prev.request?.taskId === taskId ? prev.request.resendPayload : null
      // An open session check for this task is upgraded (its `kept` is null).
      const seq = prev.seq + 1
      return {
        seq,
        request: {
          taskId,
          seq,
          resendPayload: ambiguous ? (kept ?? null) : (staged ?? stashed ?? kept),
          trigger: "turn_failure",
          gateId: null,
        },
        payload: stashed ? null : prev.payload,
        pending: prev.pending.filter(p => p.taskId !== taskId),
        dismissedCheck: prev.dismissedCheck === taskId ? null : prev.dismissedCheck,
        gates: prev.gates,
      }
    }
    case "close": {
      const { outcome } = input
      // Another task's dialog (mounted for a render after a switch) cannot close it.
      if (prev.request === null || prev.request.taskId !== input.taskId) return prev
      // "dismissed": the user ended a dialog they saw (close, Esc, Got it,
      //   or a save that settled it without a resend) -- drop the stash too,
      //   but only for a turn_failure request, the one the stash is about.
      //   Any dismissed dialog is remembered (dismissedCheck), so a reconnect
      //   check does not bring back a dialog the user just closed.
      // "resent": the stash now holds the turn that was just resent; keep it.
      // "not-shown": the dialog never became visible; keep the stash.
      // "left-host": the user navigated off the host pages after seeing
      //   it; they did not choose to give up, so keep the stash.
      // A first gate's held message ends as CLOSE_DECISIONS maps the outcome.
      const dismissed = outcome === "dismissed" ? prev.request.trigger : null
      const dismissedCheck = dismissed === null ? prev.dismissedCheck : input.taskId
      return { ...prev, request: null, payload: dismissed === "turn_failure" ? null : prev.payload, dismissedCheck }
    }
    case "retain": {
      const { taskId } = input
      const nextRequest = prev.request && prev.request.taskId === taskId ? prev.request : null
      const nextPayload = prev.payload && prev.payload.taskId === taskId ? prev.payload : null
      const nextPending = prev.pending.filter(p => p.taskId === taskId)
      const dismissedCheck = prev.dismissedCheck === taskId ? taskId : null
      if (
        nextRequest === prev.request
        && nextPayload === prev.payload
        && nextPending.length === prev.pending.length
        && dismissedCheck === prev.dismissedCheck
      ) return prev
      return { ...prev, request: nextRequest, payload: nextPayload, pending: nextPending, dismissedCheck }
    }
    case "forget": {
      const { taskId } = input
      // Drops every pending candidate for this task, not just the one that
      // settled: a settlement frame carries no turn identity, so "which
      // turn just ended" and "which other turn is still in flight" cannot
      // be told apart (xorbitsai/xagent#2465). Clearing all of them trades
      // an interleaved send's still-live candidate for the guarantee that a
      // later failure never resends the wrong message.
      //
      // An open dialog for this task holds its own copy of the snapshot,
      // handed over by openForTask, and that copy is what its resend button
      // reads -- so clearing only the stash would leave a one-click resend
      // on screen for a turn this frame has just settled. The copy goes
      // too, but nothing else about the request does: the request stays,
      // `seq` does not move, so the dialog stays open on the report it is
      // showing, the user's draft survives, and only the resend button
      // disappears. Closing the dialog here instead would throw away a
      // draft the user is in the middle of typing over a frame they did
      // not cause.
      const nextRequest = prev.request?.taskId === taskId && prev.request.resendPayload !== null
        ? { ...prev.request, resendPayload: null }
        : prev.request
      const nextPayload = prev.payload?.taskId === taskId ? null : prev.payload
      const nextPending = prev.pending.some(p => p.taskId === taskId)
        ? prev.pending.filter(p => p.taskId !== taskId)
        : prev.pending
      if (
        nextRequest === prev.request
        && nextPayload === prev.payload
        && nextPending === prev.pending
      ) return prev
      return { ...prev, request: nextRequest, payload: nextPayload, pending: nextPending }
    }
    case "identity-changed":
      return prev.request === null && prev.payload === null && prev.pending.length === 0
        && prev.dismissedCheck === null
        ? prev
        : { ...prev, request: null, payload: null, pending: [], dismissedCheck: null }
  }
}

const INITIAL_STATE: ConnectorRuntimeDialogState = { seq: 0, request: null, payload: null, pending: [], dismissedCheck: null, gates: [] }

type GateResolvers = Map<number, (decision: FirstGateDecision) => void>

// Hands a held message its answer at most once: taken out, then called.
function settleGate(resolvers: GateResolvers, id: number, decision: FirstGateDecision): void {
  const resolve = resolvers.get(id)
  if (!resolve) return
  resolvers.delete(id)
  resolve(decision)
}

export function ConnectorRuntimeDialogProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<ConnectorRuntimeDialogState>(INITIAL_STATE)
  const userId = useAuth().user?.id ?? null
  // How every action and effect below moves the request (setState is stable).
  const apply = useCallback((input: RequestInput) => setState(prev => transitionRequest(prev, input)), [])
  const gateSeqRef = useRef(0)
  const resolversRef = useRef<GateResolvers>(new Map())

  // A signed-in identity change (logout, or another tab switching accounts)
  // clears the request, the stash, any pending candidates and dismissedCheck;
  // a held first message it takes away is recorded as cleared. This also
  // runs on mount, when all four are already empty, so it is harmless there;
  // so is React's strict-mode double-invoke of effects.
  useEffect(() => {
    apply({ type: "identity-changed" })
  }, [userId, apply])

  // The only place a held first message learns its ending (the unmount fallback
  // aside): an account entry is final only once the state carrying it commits.
  useEffect(() => {
    const due = gatesToSettle(state)
    if (due.length === 0) return
    for (const { id, decision } of due) settleGate(resolversRef.current, id, decision)
    apply({ type: "gates-drained", ids: due.map(d => d.id) })
  }, [state, apply])

  // No commit follows an unmount, so the effect above cannot answer then;
  // this does, but only for a real one. React also tears effects down and
  // sets them up again on a provider that stays mounted (Fast Refresh in
  // development, and StrictMode's mount re-run), and a held request is still
  // held there: answering "cleared" then would end the message while its
  // dialog stays up, and the real decision would later find nobody to
  // answer. So the teardown only marks the provider gone and checks again
  // once the current task finishes -- a re-setup in between marks it back.
  // Assumes the provider is never kept alive with its effects torn down (a
  // hidden <Activity>): that would read as an unmount.
  const mountedRef = useRef(false)
  useEffect(() => {
    mountedRef.current = true
    const resolvers = resolversRef.current
    return () => {
      mountedRef.current = false
      queueMicrotask(() => {
        if (mountedRef.current) return
        for (const id of Array.from(resolvers.keys())) settleGate(resolvers, id, "cleared")
      })
    }
  }, [])

  const actions = useMemo<ConnectorRuntimeDialogActions>(() => ({
    openForTask: (taskId) => {
      if (!Number.isInteger(taskId) || taskId <= 0) return
      apply({ type: "open", taskId, trigger: "turn_failure" })
    },
    openSessionCheck: (taskId, cause) => {
      if (!Number.isInteger(taskId) || taskId <= 0) return
      apply({ type: "open", taskId, trigger: "session_open", cause })
    },
    openFirstGate: (taskId) => {
      if (!Number.isInteger(taskId) || taskId <= 0) return null
      const gateId = ++gateSeqRef.current
      // Registered first, so no commit finds its entry with nobody to answer.
      const decision = new Promise<FirstGateDecision>(resolve => { resolversRef.current.set(gateId, resolve) })
      apply({ type: "open", taskId, trigger: "first_gate", gateId })
      return decision
    },
    close: (outcome, taskId) => {
      apply({ type: "close", taskId, outcome })
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
      apply({ type: "retain", taskId })
    },
    forgetDelivery: (taskId) => {
      apply({ type: "forget", taskId })
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
