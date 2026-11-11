"use client"

import React, { useCallback, useEffect, useRef } from "react"
import Link from "next/link"
import { Mic, MicOff } from "lucide-react"

import { useI18n } from "@/contexts/i18n-context"
import type { VoiceCall } from "@/hooks/use-voice-call"
import type { VoicePhase } from "@/lib/voice-conversation"
import { cn } from "@/lib/utils"

/** "Talk" affordance: five bars of different heights, like a voice waveform. */
export function VoiceWaveIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      aria-hidden="true"
      className={className}
    >
      <path d="M4 10v4" />
      <path d="M8 6.5v11" />
      <path d="M12 3.5v17" />
      <path d="M16 7.5v9" />
      <path d="M20 10.5v3" />
    </svg>
  )
}

/** The assistant's voice drives the meters while it speaks, otherwise the mic. */
const levelSource = (phase: VoicePhase) => (phase === "speaking" ? "output" : "mic")

const BAR_WIDTH = 3
const BAR_GAP = 3
const SAMPLE_EVERY_MS = 45
const COLOUR_REFRESH_MS = 400

/**
 * Scrolling level meter. History runs left to right (newest on the right):
 * quiet stretches are dots, speech is vertical bars.
 */
export function VoiceWaveform({ call, className }: { call: VoiceCall; className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const phase = useRef(call.phase)
  phase.current = call.phase
  const { level } = call

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext("2d")
    if (!context) return

    let frame = 0
    let lastSample = 0
    let colourReadAt = -Infinity
    let colour = ""
    let history: number[] = []
    let width = 0
    let height = 0

    const resize = () => {
      const ratio = window.devicePixelRatio || 1
      width = canvas.clientWidth
      height = canvas.clientHeight
      canvas.width = Math.max(1, Math.round(width * ratio))
      canvas.height = Math.max(1, Math.round(height * ratio))
      context.setTransform(ratio, 0, 0, ratio, 0, 0)
    }
    resize()
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(resize)
    observer?.observe(canvas)

    const draw = (now: number) => {
      frame = window.requestAnimationFrame(draw)
      const slots = Math.max(1, Math.floor(width / (BAR_WIDTH + BAR_GAP)))
      if (now - lastSample >= SAMPLE_EVERY_MS) {
        lastSample = now
        history.push(level(levelSource(phase.current)))
        if (history.length > slots) history = history.slice(history.length - slots)
      }

      context.clearRect(0, 0, width, height)
      // A style read forces layout, so take the colour a few times a second
      // (often enough to follow the fade between the two phases' colours and a
      // theme change) rather than on every frame.
      if (now - colourReadAt > COLOUR_REFRESH_MS) { colour = getComputedStyle(canvas).color; colourReadAt = now }
      context.fillStyle = colour
      const mid = height / 2
      const offset = slots - history.length
      for (let i = 0; i < slots; i += 1) {
        const value = i < offset ? 0 : history[i - offset]
        const x = i * (BAR_WIDTH + BAR_GAP)
        if (value < 0.04) {
          context.globalAlpha = 0.3
          context.beginPath()
          context.arc(x + BAR_WIDTH / 2, mid, 1.2, 0, Math.PI * 2)
          context.fill()
        } else {
          context.globalAlpha = 1
          const barHeight = Math.max(4, Math.min(1, value) * height)
          context.beginPath()
          // Older browsers (Safari before 16) have no roundRect.
          if (context.roundRect) context.roundRect(x, mid - barHeight / 2, BAR_WIDTH, barHeight, BAR_WIDTH / 2)
          else context.rect(x, mid - barHeight / 2, BAR_WIDTH, barHeight)
          context.fill()
        }
      }
      context.globalAlpha = 1
    }
    frame = window.requestAnimationFrame(draw)

    return () => {
      window.cancelAnimationFrame(frame)
      observer?.disconnect()
    }
  }, [level])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className={cn(
        "h-8 min-w-0 flex-1 transition-colors duration-300",
        call.phase === "speaking" ? "text-primary" : "text-foreground",
        className
      )}
    />
  )
}

/**
 * The composer while a voice call is open: what you are saying (or what went
 * wrong) in the input, and below it the mute button, waveform, status and an
 * End button. It replaces the text editor and toolbar for the length of the call.
 */
export function VoiceCallBar({ call, compact = false, onEnd }: {
  call: VoiceCall
  compact?: boolean
  /** The user ended the call from the bar (End or Escape); runs just before it closes. */
  onEnd?: () => void
}) {
  const { t } = useI18n()
  const failed = call.phase === "error"
  const shown = call.alert ?? (call.caption || undefined)
  const root = useRef<HTMLDivElement>(null)
  const onEndRef = useRef(onEnd)
  onEndRef.current = onEnd
  const { end } = call
  const endCall = useCallback(() => { onEndRef.current?.(); end() }, [end])

  // Buttons come and go as the call fails and recovers (Try again, mute). When
  // the one that has focus goes, focus would drop to the page: keep it here.
  const firstRender = useRef(true)
  useEffect(() => {
    if (firstRender.current) { firstRender.current = false; return }
    const active = document.activeElement
    if (!active || active === document.body) root.current?.focus({ preventScroll: true })
  }, [failed])

  // The call is no longer a dialog, so give back what a dialog gave: focus moves
  // into it (the button that opened it is gone), and Escape ends it.
  useEffect(() => {
    const active = document.activeElement
    const form = root.current?.closest("form")
    if (!active || active === document.body || form?.contains(active)) root.current?.focus({ preventScroll: true })
  }, [])
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented || event.isComposing) return
      // The call is not modal, and other parts of the page (the progress panel,
      // the user menu, dialogs, fields) close on Escape without marking it
      // handled. Only end the call when Escape is pressed in the composer.
      const composer = root.current?.closest("form") ?? root.current
      if (!(event.target instanceof Node) || !composer?.contains(event.target)) return
      endCall()
    }
    document.addEventListener("keydown", onKeyDown)
    return () => document.removeEventListener("keydown", onKeyDown)
  }, [endCall])

  const text = shown || call.note || call.settingsLink ? (
    <div className={cn("min-w-0 text-left text-[15px]", compact && "flex-1")}>
      {shown && (
        <p
          role={call.alert ? "alert" : undefined}
          aria-live={call.alert ? undefined : "polite"}
          className={cn(
            call.alert ? "text-muted-foreground" : "text-foreground",
            compact ? "truncate" : "whitespace-pre-wrap break-words"
          )}
          title={shown}
          data-testid={call.alert ? "voice-alert" : "voice-caption"}
        >
          {shown}
        </p>
      )}
      {call.settingsLink && (
        <Link
          href="?settings=voice"
          onClick={call.end}
          className="mt-1 inline-block text-sm text-muted-foreground underline underline-offset-4"
        >
          {t("settings.voice.openSettings")}
        </Link>
      )}
      {/* Worth knowing, but the call goes on: shown beside what is being said. */}
      {call.note && (
        <p
          className={cn("text-xs text-muted-foreground", compact ? "truncate" : "mt-1")}
          title={call.note}
          data-testid="voice-note"
        >
          {call.note}
        </p>
      )}
    </div>
  ) : null

  const controls = (
    <>
      {!failed && (
        <button
          type="button"
          onClick={call.toggleMute}
          aria-pressed={call.muted}
          aria-label={call.muted ? t("voiceConversation.unmute") : t("voiceConversation.mute")}
          title={call.muted ? t("voiceConversation.unmute") : t("voiceConversation.mute")}
          className={cn(
            "flex h-9 w-9 shrink-0 items-center justify-center rounded-full transition-colors",
            call.muted
              ? "bg-destructive/10 text-destructive hover:bg-destructive/15"
              : "bg-secondary text-muted-foreground hover:text-foreground"
          )}
        >
          {call.muted ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
        </button>
      )}

      {failed ? (
        <div className="min-w-0 flex-1" />
      ) : (
        // In the one-row layout the text takes the left of the bar.
        <VoiceWaveform call={call} className={compact && shown ? "w-28 flex-none sm:w-40" : undefined} />
      )}

      <span
        role="status"
        aria-live="polite"
        className="shrink-0 text-xs font-medium text-muted-foreground"
      >
        {call.status}
      </span>

      {failed && call.retryable && (
        <button
          type="button"
          onClick={call.retry}
          disabled={call.retryDisabled}
          className="h-9 shrink-0 rounded-full border px-4 text-sm font-medium transition-colors hover:bg-secondary disabled:pointer-events-none disabled:opacity-50"
        >
          {t("voiceConversation.retry")}
        </button>
      )}
      <button
        type="button"
        onClick={endCall}
        aria-label={t("voiceConversation.end")}
        className="h-9 shrink-0 rounded-full bg-foreground px-4 text-sm font-semibold text-background transition-opacity hover:opacity-85"
      >
        {t("voiceConversation.endShort")}
      </button>
    </>
  )

  if (compact) {
    return (
      <div
        ref={root}
        role="group"
        aria-label={t("voiceConversation.title")}
        tabIndex={-1}
        className="flex min-h-[48px] w-full items-center gap-3 px-3 py-2 outline-none"
        data-voice-input="false"
        data-testid="voice-call-bar"
      >
        {text}
        {controls}
      </div>
    )
  }

  return (
    <div
      ref={root}
      role="group"
      aria-label={t("voiceConversation.title")}
      tabIndex={-1}
      className="flex w-full flex-col outline-none"
      data-voice-input="false"
      data-testid="voice-call-bar"
    >
      <div className="flex min-h-[64px] items-start px-4 pt-3.5">{text}</div>
      <div className="flex items-center gap-3 px-3 pb-3">{controls}</div>
    </div>
  )
}

// Seconds for the light/shade edge to swing out and back once; 9 unless listed.
const ORB_SPIN_SECONDS: Partial<Record<VoicePhase, number>> = { thinking: 5, speaking: 6.5 }

/**
 * Blue ball lit by a large white disc: the edge between light and shade sweeps
 * round the ball and drifts in and out, with a blue halo around it. The ball
 * swells with the voice.
 */
export function VoiceOrb({ call, className }: { call: VoiceCall; className?: string }) {
  const orbRef = useRef<HTMLDivElement>(null)
  const phase = useRef(call.phase)
  phase.current = call.phase
  const { level } = call

  useEffect(() => {
    // The swell is motion too: leave it out when the user asks for less.
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return
    let frame = 0
    const update = () => {
      frame = window.requestAnimationFrame(update)
      const node = orbRef.current
      if (!node) return
      node.style.transform = `scale(${1 + level(levelSource(phase.current)) * 0.12})`
    }
    frame = window.requestAnimationFrame(update)
    return () => window.cancelAnimationFrame(frame)
  }, [level])

  return (
    <div
      ref={orbRef}
      aria-hidden="true"
      data-phase={call.phase}
      className={cn("relative h-24 w-24 shrink-0 transition-transform duration-100", className)}
      style={{ ["--orb-spin" as string]: `${ORB_SPIN_SECONDS[call.phase] ?? 9}s` }}
    >
      {/* Halo */}
      <div className="absolute -inset-[24%] rounded-full bg-[radial-gradient(circle,rgba(76,96,244,0.5)_36%,rgba(76,96,244,0)_72%)] blur-[14px]" />

      {/* Ball: the shaded side, a little lighter where the light meets it */}
      <div
        className="absolute inset-0 overflow-hidden rounded-full"
        style={{
          background: "radial-gradient(circle at 50% 50%, #5767e8 0%, #3f4dd6 58%, #2c37b7 100%)",
        }}
      >
        {/* Swings the lit disc round the ball. The disc is far larger than the
            ball and sits mostly outside it, so only its curved edge crosses. */}
        <div className="absolute inset-0 animate-[voice-orb-swing_var(--orb-spin)_infinite] motion-reduce:animate-none">
          <div className="absolute -left-[143%] -top-[60%] h-[220%] w-[220%] animate-[voice-orb-drift_6.1s_infinite] motion-reduce:animate-none">
            <div
              className="h-full w-full rounded-full"
              style={{
                background:
                  "radial-gradient(circle closest-side at 50% 50%, #ffffff 0%, #f6f7ff 62%, rgba(214,220,255,0.88) 76%, rgba(190,199,255,0.45) 89%, rgba(190,199,255,0) 100%)",
              }}
            />
          </div>
        </div>

        {/* Fixed sheen so the ball always reads as a lit sphere */}
        <div
          className="absolute inset-0 rounded-full"
          style={{
            background:
              "radial-gradient(circle at 30% 24%, rgba(255,255,255,0.55) 0%, rgba(255,255,255,0) 46%)",
          }}
        />
      </div>
    </div>
  )
}
