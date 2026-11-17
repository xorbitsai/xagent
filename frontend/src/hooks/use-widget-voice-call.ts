"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useI18n } from "@/contexts/i18n-context"
import { generateClientMessageId } from "@/lib/utils"
import {
  createWidgetVoiceAudio, VoiceAudioError,
  type VoiceAudio, type VoiceAudioFailure, type VoiceAudioTransport, type VoiceLevelSource, type VoiceTransportEvent,
} from "@/lib/voice-audio"
import { VoiceTurns, type VoicePhase } from "@/lib/voice-conversation"
import type { VoiceCall, VoiceReplyMessage } from "@/hooks/use-voice-call"

/** Which transport refusals a state-table row stands for. */
type TransportMatch =
  | { kind: "asr_error"; codes: string[] }
  | { kind: "asr_close"; codes: number[]; reasons?: string[] }
  /** With `codes`, only responses whose body carries one of these codes. */
  | { kind: "tts_response"; statuses: number[]; codes?: string[] }

interface WidgetVoiceStateRow {
  row: number
  event: string
  result: string
  /** Transport refusals this row covers, and the failure they end the call with (undefined: only a note). */
  match?: TransportMatch[]
  failure?: VoiceAudioFailure
}

/**
 * Every event a widget voice call reacts to and what the visitor then sees.
 * The rows with `match` are also the transport classifier: an event is read
 * by the first row with a match that names a body code and fits it, and
 * otherwise by the first row with any match that fits it (see
 * `classifyWidgetVoiceEvent`). An error frame sent before a close is read on
 * its own, so its code, not the close that follows, decides the outcome.
 */
export const WIDGET_VOICE_CALL_STATES: readonly WidgetVoiceStateRow[] = [
  { row: 1, event: "call button pressed", result: "audio unlocked in the click, then the microphone, then recognition" },
  { row: 2, event: "call button pressed while busy", result: "error, try later, retryable" },
  { row: 3, event: "microphone refused or missing", result: "error naming the microphone, no backend request" },
  { row: 4, event: "recognition closed 4003 (credential, voice switch or recheck refused)", result: "unavailable, not retryable",
    match: [{ kind: "asr_close", codes: [4003] }], failure: "voice_unavailable" },
  { row: 5, event: "recognition refused as busy (error frame) or closed 1008 busy or rate limited", result: "busy, retryable",
    match: [
      { kind: "asr_error", codes: ["voice_busy"] },
      { kind: "asr_close", codes: [1008], reasons: ["Voice is busy", "Too many requests"] },
    ], failure: "voice_busy" },
  { row: 6, event: "recognition refused: budget exhausted", result: "error, retry later",
    match: [{ kind: "asr_error", codes: ["voice_budget_exhausted"] }], failure: "voice_retry_later" },
  { row: 7, event: "recognition refused: unavailable, model unavailable or invalid request", result: "unavailable, not retryable",
    match: [{ kind: "asr_error", codes: ["voice_unavailable", "model_unavailable", "invalid_request"] }], failure: "voice_unavailable" },
  { row: 8, event: "recognition failed 1011 / 1006, or the handshake was refused", result: "error, retryable",
    match: [{ kind: "asr_error", codes: ["stream_failed"] }, { kind: "asr_close", codes: [1011, 1006] }], failure: "connection_failed" },
  { row: 9, event: "recognition closed 1012 (server restart)", result: "error, retryable",
    match: [{ kind: "asr_close", codes: [1012] }], failure: "connection_failed" },
  { row: 10, event: "usage limit reached", result: "unavailable, not retryable, no quota detail",
    match: [{ kind: "asr_error", codes: ["standalone_usage_refused"] }], failure: "voice_unavailable" },
  { row: 11, event: "600 second limit reached, or a write timeout (close 1000 \"Speech session timed out\")", result: "timed out, retryable",
    match: [
      { kind: "asr_error", codes: ["session_timeout"] },
      { kind: "asr_close", codes: [1000], reasons: ["Speech session timed out"] },
    ], failure: "session_timeout" },
  { row: 12, event: "sending the words failed", result: "call ends, fixed send-failed text, retryable" },
  { row: 13, event: "one sentence could not be synthesized", result: "call continues with a playback note",
    match: [{ kind: "tts_response", statuses: [400, 422, 500, 502, 504] }] },
  { row: 14, event: "synthesis 401 / 403", result: "unavailable, not retryable",
    match: [{ kind: "tts_response", statuses: [401, 403] }], failure: "voice_unavailable" },
  { row: 15, event: "synthesis 402 (usage limit), or 404 (the session widget API switched off)", result: "unavailable, not retryable",
    match: [{ kind: "tts_response", statuses: [402, 404] }], failure: "voice_unavailable" },
  { row: 16, event: "synthesis 503, or 400 with model_unavailable", result: "unavailable, not retryable",
    match: [{ kind: "tts_response", statuses: [503] }, { kind: "tts_response", statuses: [400], codes: ["model_unavailable"] }],
    failure: "voice_unavailable" },
  { row: 17, event: "synthesis 429", result: "call ends, temporarily unavailable, retry later",
    match: [{ kind: "tts_response", statuses: [429] }], failure: "voice_retry_later" },
  { row: 18, event: "page unloaded or call closed", result: "microphone, recognition and playback released" },
  { row: 19, event: "disabled turns true", result: "hang up, error" },
  { row: 20, event: "task id changes from a number to anything else", result: "hang up and close the bar" },
  { row: 21, event: "start pressed again during a call", result: "ignored" },
  { row: 22, event: "voice switched off during a call", result: "server closes recognition within 30 s, error (row 4)" },
  { row: 23, event: "Session recognition handshake closed 4401 (token refused)", result: "error, retryable; the page replaces the refused token",
    match: [{ kind: "asr_close", codes: [4401] }], failure: "connection_failed" },
]

function matches(match: TransportMatch, event: VoiceTransportEvent): boolean {
  if (event.kind !== match.kind) return false
  if (match.kind === "tts_response") {
    const response = event as { status: number; code?: string }
    return match.statuses.includes(response.status) && (!match.codes || match.codes.includes(response.code ?? ""))
  }
  if (match.kind === "asr_error") return match.codes.includes((event as { code?: string }).code ?? "")
  const close = event as { code: number; reason: string }
  return match.codes.includes(close.code) && (!match.reasons || match.reasons.includes(close.reason))
}

const specific = (match: TransportMatch) => match.kind === "tts_response" && !!match.codes

/**
 * The widget transports' `classify`: the failure of the first state-table row
 * that matches. A match that also names a body code is tried before matches
 * on the status alone, so a 400 that says the model is unavailable is not read
 * as an ordinary failed sentence.
 */
export function classifyWidgetVoiceEvent(event: VoiceTransportEvent): VoiceAudioFailure | undefined {
  const first = (pick: (match: TransportMatch) => boolean) =>
    WIDGET_VOICE_CALL_STATES.find(row => row.match?.some(match => pick(match) && matches(match, event)))
  return (first(specific) ?? first(match => !specific(match)))?.failure
}

/** Why a widget call stopped, beyond the transport's own failures. */
type WidgetVoiceStop = VoiceAudioFailure | "start_busy" | "microphone" | "send_failed" | "disabled"

type WidgetVoiceMessage = "unavailable" | "busy" | "temporarilyUnavailable" | "sessionTimeout" | "connectionEnded" | "tryLater" | "microphoneDenied" | "sendFailed"

const STOPS: Record<WidgetVoiceStop, { message: WidgetVoiceMessage; retryable: boolean }> = {
  voice_unavailable: { message: "unavailable", retryable: false },
  authentication_required: { message: "unavailable", retryable: false },
  voice_busy: { message: "busy", retryable: true },
  voice_retry_later: { message: "temporarilyUnavailable", retryable: true },
  session_timeout: { message: "sessionTimeout", retryable: true },
  connection_failed: { message: "connectionEnded", retryable: true },
  start_busy: { message: "tryLater", retryable: true },
  microphone: { message: "microphoneDenied", retryable: true },
  send_failed: { message: "sendFailed", retryable: true },
  disabled: { message: "connectionEnded", retryable: true },
}

const MICROPHONE_ERRORS = new Set(["NotAllowedError", "NotFoundError", "NotReadableError", "SecurityError", "OverconstrainedError"])
const microphoneFailed = (error: unknown) =>
  typeof navigator.mediaDevices?.getUserMedia !== "function"
  || (typeof error === "object" && error !== null && MICROPHONE_ERRORS.has(String((error as { name?: unknown }).name)))

export interface WidgetVoiceCallOptions {
  /** Addresses and credentials of the page's own speech endpoints. */
  transport: VoiceAudioTransport
  taskId: number | null
  messages: VoiceReplyMessage[]
  busy: boolean
  waiting: boolean
  disabled: boolean
  onSend: (text: string, clientMessageId: string) => Promise<void>
  onInterrupt: () => boolean | void
  open: boolean
  onOpenChange: (open: boolean) => void
}

/**
 * A voice call in an embedded chat. Unlike the signed-in call it reads no
 * login, models or settings: the page supplies the transport, and every
 * failure is shown with widget copy that never points at an account.
 */
export function useWidgetVoiceCall({ transport, taskId, messages, busy, waiting, disabled, onSend, onInterrupt, open, onOpenChange }: WidgetVoiceCallOptions): VoiceCall {
  const { t } = useI18n()
  const [phase, setPhase] = useState<VoicePhase>("idle")
  const [caption, setCaption] = useState("")
  const [stop, setStop] = useState<WidgetVoiceStop>()
  const [playbackFailed, setPlaybackFailed] = useState(false)
  const [muted, setMuted] = useState(false)
  const [serverVadOnly, setServerVadOnly] = useState(false)
  const audio = useRef<VoiceAudio | null>(null)
  const turns = useRef<VoiceTurns | null>(null)
  const generation = useRef(0)
  const baseline = useRef(new Set<string>())
  const live = useRef({ transport, messages, busy, onSend, onInterrupt })
  live.current = { transport, messages, busy, onSend, onInterrupt }
  const previousTaskId = useRef(taskId)

  const unlocked = useRef<AudioContext | null>(null)
  const unlock = () => {
    if (unlocked.current || typeof AudioContext === "undefined") return
    try {
      const context = new AudioContext()
      void context.resume().catch(() => {})
      unlocked.current = context
    } catch { /* The call creates its own and reports the failure. */ }
  }
  const discardUnlocked = () => {
    const context = unlocked.current
    unlocked.current = null
    if (context && context.state !== "closed") void context.close().catch(() => {})
  }

  const release = () => {
    generation.current += 1
    turns.current?.stop(); turns.current = null
    audio.current?.close(); audio.current = null
  }
  const end = () => {
    release()
    setPhase("idle"); setMuted(false); setCaption(""); setStop(undefined); setPlaybackFailed(false); setServerVadOnly(false)
  }
  const fail = (reason: WidgetVoiceStop) => { release(); setStop(reason); setPlaybackFailed(false); setCaption(""); setPhase("error") }

  // A call belongs to one conversation: a task id appearing for the first
  // message keeps it, any other change (a new conversation) hangs up.
  useEffect(() => {
    const previous = previousTaskId.current
    previousTaskId.current = taskId
    if (previous === taskId || (previous === null && taskId !== null)) return
    end()
    onOpenChange(false)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskId])

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => () => { release(); discardUnlocked() }, [])
  useEffect(() => {
    // Opening is the user's click: unlock audio now, before anything is awaited.
    if (open) unlock()
    else { end(); discardUnlocked() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  useEffect(() => {
    if (disabled && turns.current) fail("disabled")
    turns.current?.updateBusy(busy)
    // Recheck every render, as the signed-in call does.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  })

  useEffect(() => {
    for (const message of messages) {
      if (baseline.current.has(message.id) || message.role !== "assistant" || !message.isResult || message.status === "failed" || message.status === "interrupted") continue
      const text = message.rawContent ?? (typeof message.content === "string" ? message.content : "")
      turns.current?.reply({ id: message.id, text, complete: message.status === "completed" || waiting || !busy })
    }
  }, [messages, busy, waiting])

  const start = () => {
    if (audio.current) return
    const context = unlocked.current ?? undefined
    unlocked.current = null
    end()
    const epoch = generation.current
    const current = () => epoch === generation.current
    setPhase("connecting")
    // The microphone is requested before the recognition socket opens, so a
    // refused microphone sends nothing to the backend.
    const call = createWidgetVoiceAudio(live.current.transport, {
      context,
      activity: speaking => { if (current()) turns.current?.activity(speaking) },
      transcript: (type, text, id) => { if (current()) turns.current?.transcript(type, text, id) },
      error: reason => { if (current()) fail(reason ?? "connection_failed") },
      vadFallback: () => { if (current()) setServerVadOnly(true) },
    })
    audio.current = call
    turns.current = new VoiceTurns({
      send: async (text) => {
        baseline.current = new Set(live.current.messages.map(message => message.id))
        if (current()) setPlaybackFailed(false)
        await live.current.onSend(text, generateClientMessageId())
        if (current()) setCaption(shown => (shown && text.endsWith(shown) ? "" : shown))
      },
      interrupt: () => live.current.onInterrupt(),
      stopAudio: () => call.stopAudio(),
      speak: text => call.speak(text),
      phase: next => { if (current()) setPhase(next) },
      caption: text => { if (current()) setCaption(text) },
      // The widget never shows the send error's own text.
      error: () => { if (current()) fail("send_failed") },
      playbackError: error => {
        if (!current()) return
        if (error instanceof VoiceAudioError) fail(error.code)
        else setPlaybackFailed(true)
      },
    }, { maxChunk: 1000 })
    turns.current.updateBusy(live.current.busy)
    void (async () => {
      try {
        await call.start()
        if (current()) setPhase(turns.current?.muted ? "muted" : "listening")
      } catch (error) {
        if (!current()) return
        fail(error instanceof VoiceAudioError ? error.code : microphoneFailed(error) ? "microphone" : "connection_failed")
      }
    })()
  }

  useEffect(() => {
    if (!open || phase !== "idle" || audio.current) return
    if (disabled) { onOpenChange(false); return }
    if (busy) { setStop("start_busy"); setPhase("error"); return }
    start()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, phase, busy, disabled])

  const retry = () => { unlock(); end() }
  const toggleMute = () => {
    const next = !muted
    setMuted(next)
    if (audio.current) audio.current.muted = next
    turns.current?.mute(next)
  }
  const close = () => { end(); onOpenChange(false) }
  const controls = useRef({ retry, close, toggleMute })
  controls.current = { retry, close, toggleMute }
  const stableRetry = useCallback(() => controls.current.retry(), [])
  const stableEnd = useCallback(() => controls.current.close(), [])
  const stableToggleMute = useCallback(() => controls.current.toggleMute(), [])
  const level = useCallback((source: VoiceLevelSource) => audio.current?.level(source) ?? 0, [])

  const active = phase !== "idle" && phase !== "error"
  const failed = phase === "error" ? STOPS[stop ?? "connection_failed"] : undefined
  const alert = failed && t(`voiceConversation.widget.${failed.message}`)
  const status = t(`voiceConversation.states.${phase === "error" ? "stopped" : phase === "idle" ? "connecting" : phase}`)
  const note = [
    playbackFailed ? t("voiceConversation.widget.playbackFailed") : "",
    serverVadOnly ? t("voiceConversation.serverVadOnly") : "",
  ].filter(Boolean).join(" ") || undefined
  const retryable = Boolean(failed?.retryable)
  const retryDisabled = busy || disabled

  return useMemo<VoiceCall>(() => ({
    open, phase, active, caption, muted, status, alert, note, settingsLink: false, retryable, retryDisabled,
    level, retry: stableRetry, end: stableEnd, toggleMute: stableToggleMute,
  }), [open, phase, active, caption, muted, status, alert, note, retryable, retryDisabled, level, stableRetry, stableEnd, stableToggleMute])
}
