"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useAuth } from "@/contexts/auth-context"
import { useI18n } from "@/contexts/i18n-context"
import { generateClientMessageId } from "@/lib/utils"
import { useVoiceModels, voiceModelReason } from "@/hooks/use-voice-models"
import { useVoicePreferences } from "@/hooks/use-voice-preferences"
import { VoiceAudio, VoiceAudioError, type VoiceAudioFailure, type VoiceLevelSource } from "@/lib/voice-audio"
import { VoiceTurns, type VoicePhase } from "@/lib/voice-conversation"

export interface VoiceReplyMessage {
  id: string
  role: string
  content: unknown
  rawContent?: string
  isResult?: boolean
  status?: string
}

export interface VoiceCallOptions {
  taskId: number | null
  messages: VoiceReplyMessage[]
  busy: boolean
  waiting: boolean
  disabled: boolean
  onSend: (text: string, clientMessageId: string) => Promise<void>
  onInterrupt: () => boolean | void
  open: boolean
  onOpenChange: (open: boolean) => void
  allowTaskAdoption?: boolean
  /**
   * Load the voice models and saved settings before a call is asked for. The
   * audio context can only start inside the click that opens the call, so the
   * call must be able to begin without waiting on the network.
   */
  preload?: boolean
}

/** Why a call that was asked for could not begin. */
type Blocked = "models" | "selection" | "preferences"

/** Everything the composer needs to show and control a voice call. */
export interface VoiceCall {
  /** The user asked for a call and it has not been ended or dismissed. */
  open: boolean
  phase: VoicePhase
  /** Microphone is live (connecting, listening, thinking, speaking or muted). */
  active: boolean
  /** What the user is saying, as it is recognized, until it has been sent. */
  caption: string
  muted: boolean
  /** Short status for the controls, already translated. */
  status: string
  /** What went wrong, already translated; undefined when nothing did. */
  alert?: string
  /** A hint that is not an error, already translated. */
  note?: string
  /** The problem is in the saved voice settings, so link to them. */
  settingsLink: boolean
  /** Loudness 0..1 of the microphone or of the assistant's voice. */
  level: (source: VoiceLevelSource) => number
  /** Trying again could help (a settings change would be needed otherwise). */
  retryable: boolean
  /** Trying again has to wait, because a run is still in progress. */
  retryDisabled: boolean
  /** Start again after a failure. */
  retry: () => void
  end: () => void
  toggleMute: () => void
}

export function useVoiceCall({ taskId, messages, busy, waiting, disabled, onSend, onInterrupt, open, onOpenChange, allowTaskAdoption = false, preload = false }: VoiceCallOptions): VoiceCall {
  const { t } = useI18n()
  const auth = useAuth()
  // Bumped by "try again" so a failed models or settings lookup is repeated.
  const [refresh, setRefresh] = useState(false)
  const models = useVoiceModels(open || preload, refresh)
  const saved = useVoicePreferences(open || preload, refresh)
  const { asr: liveAsr, tts: liveTts, loading } = models
  const modelReason = voiceModelReason(models)
  const [phase, setPhase] = useState<VoicePhase>("idle")
  const [caption, setCaption] = useState("")
  const [sendFailure, setSendFailure] = useState("")
  const [audioFailure, setAudioFailure] = useState<VoiceAudioFailure>()
  const [blocked, setBlocked] = useState<Blocked>()
  const [playbackFailed, setPlaybackFailed] = useState(false)
  const [muted, setMuted] = useState(false)
  const [serverVadOnly, setServerVadOnly] = useState(false)
  const audio = useRef<VoiceAudio | null>(null)
  const turns = useRef<VoiceTurns | null>(null)
  const generation = useRef(0)
  const baseline = useRef(new Set<string>())
  const live = useRef({ messages, busy, waiting, onSend, onInterrupt })
  live.current = { messages, busy, waiting, onSend, onInterrupt }
  const identity = useRef({ taskId, sessionId: auth.session?.sessionId, userId: auth.user?.id })

  const release = () => {
    generation.current += 1
    turns.current?.stop(); turns.current = null
    audio.current?.close(); audio.current = null
  }
  const end = () => {
    release()
    setPhase("idle"); setMuted(false); setCaption(""); setSendFailure(""); setAudioFailure(undefined); setBlocked(undefined); setPlaybackFailed(false); setServerVadOnly(false)
  }
  const fail = (reason?: VoiceAudioFailure) => { release(); setAudioFailure(reason); setPlaybackFailed(false); setCaption(""); setPhase("error") }

  useEffect(() => {
    const previous = identity.current
    identity.current = { taskId, sessionId: auth.session?.sessionId, userId: auth.user?.id }
    const sameLogin = previous.sessionId === auth.session?.sessionId && previous.userId === auth.user?.id
    if (sameLogin && (previous.taskId === taskId || (allowTaskAdoption && previous.taskId === null && taskId !== null))) return
    end()
    onOpenChange(false)
    // A call belongs to one task and one login, including during late permission prompts.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskId, auth.session?.sessionId, auth.user?.id])

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => () => { release() }, [])
  useEffect(() => {
    // Closing always resets: a failed call has no audio left to release but
    // still holds its error, which must not greet the next call.
    if (!open) end()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const asrId = saved.preferences.asr_model_id || liveAsr[0]?.model_id || ""
  const ttsId = saved.preferences.tts_model_id || liveTts[0]?.model_id || ""
  const selectionUnavailable = !liveAsr.some(model => model.model_id === asrId) || !liveTts.some(model => model.model_id === ttsId)
  const canStart = !modelReason && !loading && !saved.loading && !saved.error && !selectionUnavailable && !busy && !disabled
  const active = phase !== "idle" && phase !== "error"

  useEffect(() => {
    if (disabled && turns.current) { release(); setPhase("error") }
    turns.current?.updateBusy(busy)
    // Recheck every render: delivery can become a running task without changing
    // the aggregate busy flag. A pending barge-in must reach that new run too.
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
    if (audio.current || !auth.token || !canStart) return
    end()
    const epoch = generation.current
    const current = () => epoch === generation.current
    setPhase("connecting")
    const call = new VoiceAudio({
      asr: asrId, tts: ttsId, voice: saved.preferences.voice || undefined, token: auth.token, sessionId: auth.session?.sessionId ?? null,
      vadSilenceThreshold: liveAsr.find(model => model.model_id === asrId)?.model_provider?.trim().toLowerCase() === "elevenlabs" ? 0.5 : undefined,
      activity: speaking => { if (current()) turns.current?.activity(speaking) },
      transcript: (type, text, id) => { if (current()) turns.current?.transcript(type, text, id) },
      error: reason => { if (current()) fail(reason) },
      vadFallback: () => { if (current()) setServerVadOnly(true) },
    })
    audio.current = call
    turns.current = new VoiceTurns({
      send: async (text) => {
        baseline.current = new Set(live.current.messages.map(message => message.id))
        if (current()) setPlaybackFailed(false)
        try {
          await live.current.onSend(text, generateClientMessageId())
          // The words now live in the conversation; stop showing them in the input.
          if (current()) setCaption("")
        } catch (error) {
          // The normal task-delivery path supplies a client-safe API error.
          // Keep its explanation separate from microphone/ASR failures.
          if (current()) setSendFailure(error instanceof Error ? error.message : t("voiceConversation.sendFailed"))
          throw error
        }
      },
      interrupt: () => live.current.onInterrupt(),
      stopAudio: () => call.stopAudio(),
      speak: text => call.speak(text),
      phase: next => { if (current()) setPhase(next) },
      caption: text => { if (current()) setCaption(text) },
      error: () => { if (current()) fail() },
      playbackError: error => {
        if (!current()) return
        if (error instanceof VoiceAudioError) fail(error.code)
        else setPlaybackFailed(true)
      },
    })
    turns.current.updateBusy(live.current.busy)
    void (async () => {
      try {
        await call.start()
        if (current()) setPhase(turns.current?.muted ? "muted" : "listening")
      } catch (error) { if (current()) fail(error instanceof VoiceAudioError ? error.code : undefined) }
    })()
  }

  // The button that opens a call is the click that starts it: no confirmation
  // step. Starting must follow the click without waiting on a network round
  // trip (the audio context needs the click's user activation), which is why
  // the models and settings are loaded ahead of time (`preload`) and the call
  // begins as soon as everything it depends on is ready.
  useEffect(() => {
    if (!open) return
    if (phase !== "idle" || audio.current) return
    if (!auth.token) { setAudioFailure("authentication_required"); setPhase("error"); return }
    if (loading || saved.loading) return
    if (modelReason) { setBlocked("models"); setPhase("error"); return }
    if (saved.error) { setBlocked("preferences"); setPhase("error"); return }
    if (selectionUnavailable) { setBlocked("selection"); setPhase("error"); return }
    // Another run is in progress, or voice is switched off: nothing to start.
    if (!canStart) { onOpenChange(false); return }
    start()
    // `start` and `onOpenChange` are recreated every render and read current state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, phase, auth.token, loading, saved.loading, saved.error, modelReason, selectionUnavailable, canStart])

  const retry = () => {
    end()
    // A failed models or settings lookup is looked up again; the effect above
    // then starts the call on the next render.
    if (blocked) setRefresh(value => !value)
  }
  const toggleMute = () => {
    const next = !muted
    setMuted(next)
    if (audio.current) audio.current.muted = next
    turns.current?.mute(next)
  }
  const close = () => { end(); onOpenChange(false) }
  // The controls close over this render's state; hand out stable wrappers so
  // the call object only changes when something visible does.
  const controls = useRef({ retry, close, toggleMute })
  controls.current = { retry, close, toggleMute }
  const stableRetry = useCallback(() => controls.current.retry(), [])
  const stableEnd = useCallback(() => controls.current.close(), [])
  const stableToggleMute = useCallback(() => controls.current.toggleMute(), [])
  const level = useCallback((source: VoiceLevelSource) => audio.current?.level(source) ?? 0, [])

  const failureText = (() => {
    if (phase !== "error") return undefined
    if (sendFailure) return sendFailure
    if (blocked === "preferences") return t("settings.voice.loadFailed")
    if (blocked === "selection") return t("settings.voice.selectionUnavailable")
    if (blocked === "models" && modelReason) return t(`voiceConversation.${modelReason}`)
    if (audioFailure === "session_timeout") return t("voiceConversation.sessionTimeout")
    if (audioFailure === "authentication_required") return t("voiceConversation.authenticationRequired")
    return t("voiceConversation.states.error")
  })()
  const alert = failureText ?? (playbackFailed ? t("voiceConversation.playbackFailed") : undefined)
  // A call that is open but has not begun connecting yet reads as connecting.
  const status = t(`voiceConversation.states.${phase === "error" ? "stopped" : phase === "idle" ? "connecting" : phase}`)
  const note = serverVadOnly ? t("voiceConversation.serverVadOnly") : undefined
  const settingsLink = phase === "error" && (blocked === "preferences" || blocked === "selection" || blocked === "models")
  // Missing or unsuitable models only change in settings, so trying again can
  // not help; everything else (a dropped connection, a failed lookup) can.
  const retryable = phase === "error" && !(blocked === "selection" || (blocked === "models" && modelReason !== "modelsUnavailable" && modelReason !== "checkingModels"))
  const retryDisabled = busy || disabled

  return useMemo<VoiceCall>(() => ({
    open, phase, active, caption, muted, status, alert, note, settingsLink, retryable, retryDisabled,
    level, retry: stableRetry, end: stableEnd, toggleMute: stableToggleMute,
  }), [open, phase, active, caption, muted, status, alert, note, settingsLink, retryable, retryDisabled, level, stableRetry, stableEnd, stableToggleMute])
}
