"use client"

import React, { useEffect, useRef, useState } from "react"
import Link from "next/link"
import { Headphones, Mic, MicOff, PhoneOff } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { useAuth } from "@/contexts/auth-context"
import { useI18n } from "@/contexts/i18n-context"
import { generateClientMessageId } from "@/lib/utils"
import { useVoiceModels, voiceModelReason } from "@/hooks/use-voice-models"
import { useVoicePreferences } from "@/hooks/use-voice-preferences"
import { VoiceAudio, VoiceAudioError, type VoiceAudioFailure } from "@/lib/voice-audio"
import { VoiceTurns, type VoicePhase } from "@/lib/voice-conversation"

interface ReplyMessage {
  id: string
  role: string
  content: unknown
  rawContent?: string
  isResult?: boolean
  status?: string
}

export function VoiceConversation({ taskId, messages, busy, waiting, disabled, onSend, onInterrupt, open, onOpenChange, allowTaskAdoption = false }: {
  taskId: number | null
  messages: ReplyMessage[]
  busy: boolean
  waiting: boolean
  disabled: boolean
  onSend: (text: string, clientMessageId: string) => Promise<void>
  onInterrupt: () => boolean | void
  open: boolean
  onOpenChange: (open: boolean) => void
  allowTaskAdoption?: boolean
}) {
  const { t } = useI18n()
  const auth = useAuth()
  const models = useVoiceModels(open)
  const saved = useVoicePreferences(open)
  const { asr: liveAsr, tts: liveTts, loading } = models
  const modelReason = voiceModelReason(models)
  const [phase, setPhase] = useState<VoicePhase>("idle")
  const [caption, setCaption] = useState("")
  const [sendFailure, setSendFailure] = useState("")
  const [audioFailure, setAudioFailure] = useState<VoiceAudioFailure>()
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
  const end = () => { release(); setPhase("idle"); setMuted(false); setCaption(""); setSendFailure(""); setAudioFailure(undefined); setPlaybackFailed(false); setServerVadOnly(false) }
  const fail = (reason?: VoiceAudioFailure) => { release(); setAudioFailure(reason); setPlaybackFailed(false); setPhase("error") }

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
    if (!open && audio.current) end()
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

  const start = async () => {
    if (audio.current || !auth.token || !canStart) return
    end()
    const epoch = generation.current
    const current = () => epoch === generation.current
    setPhase("connecting")
    const call = new VoiceAudio({
      asr: asrId, tts: ttsId, voice: saved.preferences.voice || undefined, token: auth.token, sessionId: auth.session?.sessionId ?? null,
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
    try {
      await call.start()
      if (current()) setPhase(turns.current?.muted ? "muted" : "listening")
    } catch (error) { if (current()) fail(error instanceof VoiceAudioError ? error.code : undefined) }
  }

  if (!auth.token) return null
  return (
    <Dialog open={open} onOpenChange={value => { if (!value) end(); onOpenChange(value) }}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{t("voiceConversation.title")}</DialogTitle>
          <DialogDescription>{t("voiceConversation.description")}</DialogDescription>
        </DialogHeader>
        <div className="flex flex-col items-center gap-5 py-6">
          <div aria-hidden="true" className={`flex h-28 w-28 items-center justify-center rounded-full bg-primary/10 text-primary ${active && !muted ? "motion-safe:animate-pulse" : ""}`}>
            {muted ? <MicOff className="h-10 w-10" /> : <Headphones className="h-10 w-10" />}
          </div>
          <p role="status" className="text-sm font-medium">{t(phase === "error" && sendFailure ? "voiceConversation.sendFailed" : phase === "error" && audioFailure === "session_timeout" ? "voiceConversation.sessionTimeout" : phase === "error" && audioFailure === "authentication_required" ? "voiceConversation.authenticationRequired" : `voiceConversation.states.${phase}`)}</p>
          {phase === "error" && sendFailure && <p role="alert" className="text-center text-sm text-muted-foreground">{sendFailure}</p>}
          {playbackFailed && <p role="alert" className="text-center text-sm text-muted-foreground">{t("voiceConversation.playbackFailed")}</p>}
          {serverVadOnly && <p className="text-center text-xs text-muted-foreground">{t("voiceConversation.serverVadOnly")}</p>}
          <p aria-live="polite" className="min-h-12 max-h-40 overflow-auto text-center text-sm text-muted-foreground">{caption}</p>
        </div>
        {!active ? <div className="space-y-4">
          {modelReason && <p className="text-sm text-muted-foreground">{t(`voiceConversation.${modelReason}`)}</p>}
          {saved.error && <p role="alert" className="text-sm text-muted-foreground">{t("settings.voice.loadFailed")}</p>}
          {!modelReason && !saved.loading && selectionUnavailable && <p role="alert" className="text-sm text-muted-foreground">{t("settings.voice.selectionUnavailable")}</p>}
          <Button className="w-full" disabled={!canStart} onClick={() => void start()}>{t("voiceConversation.start")}</Button>
          <Link className="block text-center text-sm text-muted-foreground underline underline-offset-4" href="?settings=voice" onClick={() => { end(); onOpenChange(false) }}>{t("settings.voice.openSettings")}</Link>
        </div> : <div className="flex justify-center gap-4">
          <Button variant="outline" aria-label={muted ? t("voiceConversation.unmute") : t("voiceConversation.mute")} aria-pressed={muted} onClick={() => {
            const next = !muted; setMuted(next)
            if (audio.current) audio.current.muted = next
            turns.current?.mute(next)
          }}>{muted ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}</Button>
          <Button variant="destructive" aria-label={t("voiceConversation.end")} onClick={end}><PhoneOff className="h-5 w-5" /></Button>
        </div>}
      </DialogContent>
    </Dialog>
  )
}
