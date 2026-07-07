"use client"

import React, { useCallback, useEffect, useRef, useState } from "react"
import { Loader2, Mic, Square } from "lucide-react"

import { toast } from "@/components/ui/sonner"
import { useI18n } from "@/contexts/i18n-context"
import {
  apiRequest,
  getApiErrorMessage,
  isJsonRecord,
  parseApiResponse,
} from "@/lib/api-wrapper"
import { cn, getApiUrl, getUploadApiUrl } from "@/lib/utils"

export type VoiceTarget = HTMLInputElement | HTMLTextAreaElement | HTMLElement
export type VoiceStatus = "idle" | "recording" | "transcribing"

interface UseVoiceInputControlsOptions {
  autoRefresh?: boolean
}

const TEXT_INPUT_TYPES = new Set(["", "email", "search", "tel", "text", "url"])
const SENSITIVE_FIELD_PATTERN =
  /(api[_-]?key|authorization|bearer|card[_-]?number|cardnumber|client[_-]?id|client[_-]?secret|credit[_-]?card|csc|cvc|cvv|one[_-]?time|otp|passcode|password|pin|security[_-]?code|securitycode|secret|social[_-]?security|ssn|token)/i
const SENSITIVE_AUTOCOMPLETE_TOKENS = new Set([
  "cc-additional-name",
  "cc-csc",
  "cc-exp",
  "cc-exp-month",
  "cc-exp-year",
  "cc-family-name",
  "cc-given-name",
  "cc-name",
  "cc-number",
  "cc-type",
  "current-password",
  "new-password",
  "one-time-code",
])

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function isElementHidden(element: HTMLElement): boolean {
  const rect = element.getBoundingClientRect()
  return rect.width <= 0 || rect.height <= 0
}

function getPositionAnchor(target: VoiceTarget): HTMLElement {
  const explicitAnchor = target.closest<HTMLElement>("[data-voice-input-anchor]")
  if (explicitAnchor && !isElementHidden(explicitAnchor)) {
    return explicitAnchor
  }

  const canUseEnclosingAnchor =
    target.isContentEditable || target instanceof HTMLTextAreaElement
  if (!canUseEnclosingAnchor) {
    return target
  }

  const targetRect = target.getBoundingClientRect()
  let current = target.parentElement
  while (current && current !== document.body) {
    if (!isElementHidden(current)) {
      const rect = current.getBoundingClientRect()
      const containsTarget =
        rect.left <= targetRect.left &&
        rect.right >= targetRect.right &&
        rect.top <= targetRect.top &&
        rect.bottom >= targetRect.bottom
      const expandsTarget =
        rect.width >= targetRect.width + 48 ||
        rect.height >= targetRect.height + 24
      if (containsTarget && expandsTarget) {
        return current
      }
    }
    current = current.parentElement
  }

  return target
}

function fieldFingerprint(element: HTMLElement): string {
  return [
    element.getAttribute("aria-label"),
    element.getAttribute("id"),
    element.getAttribute("name"),
    element.getAttribute("placeholder"),
  ]
    .filter(Boolean)
    .join(" ")
}

function hasSensitiveAutocomplete(element: HTMLElement): boolean {
  const autocomplete = element.getAttribute("autocomplete")
  if (!autocomplete) return false
  return autocomplete
    .toLowerCase()
    .split(/\s+/)
    .some((token) => SENSITIVE_AUTOCOMPLETE_TOKENS.has(token))
}

function isVoiceEligibleTarget(target: EventTarget | null): target is VoiceTarget {
  if (!(target instanceof HTMLElement)) return false
  if (target.closest("[data-voice-input-root]")) return false
  if (target.closest("[data-voice-input='false']")) return false
  if (target.getAttribute("data-voice-input") !== "true") return false
  if (target.getAttribute("aria-readonly") === "true") return false
  if (SENSITIVE_FIELD_PATTERN.test(fieldFingerprint(target))) return false
  if (hasSensitiveAutocomplete(target)) return false

  if (target instanceof HTMLInputElement) {
    return (
      TEXT_INPUT_TYPES.has(target.type) &&
      !target.disabled &&
      !target.readOnly &&
      !isElementHidden(target)
    )
  }

  if (target instanceof HTMLTextAreaElement) {
    return !target.disabled && !target.readOnly && !isElementHidden(target)
  }

  return (
    target.isContentEditable &&
    !isElementHidden(target)
  )
}

function isInsideActiveVoiceArea(
  eventTarget: EventTarget | null,
  currentTarget: VoiceTarget | null
): boolean {
  if (!(eventTarget instanceof Element) || !currentTarget) return false
  if (eventTarget.closest("[data-voice-input-root]")) return true

  const anchor = getPositionAnchor(currentTarget)
  return currentTarget.contains(eventTarget) || anchor.contains(eventTarget)
}

function dispatchInputEvents(target: HTMLElement, data: string): void {
  try {
    target.dispatchEvent(
      new InputEvent("input", {
        bubbles: true,
        data,
        inputType: "insertText",
      })
    )
  } catch {
    target.dispatchEvent(new Event("input", { bubbles: true }))
  }
  target.dispatchEvent(new Event("change", { bubbles: true }))
}

function setNativeValue(
  target: HTMLInputElement | HTMLTextAreaElement,
  value: string
): void {
  const prototype =
    target instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype
  const descriptor = Object.getOwnPropertyDescriptor(prototype, "value")
  descriptor?.set?.call(target, value)
}

function withWordBoundary(
  existing: string,
  start: number,
  end: number,
  rawText: string
): string {
  const text = rawText.trim()
  if (!text) return ""

  const before = existing.slice(0, start)
  const after = existing.slice(end)
  const needsLeadingSpace =
    before.length > 0 && !/\s$/.test(before) && !/^[,.;:!?，。！？、]/.test(text)
  const needsTrailingSpace =
    after.length > 0 && !/^\s/.test(after) && !/[([{（【「『]$/.test(text)

  return `${needsLeadingSpace ? " " : ""}${text}${needsTrailingSpace ? " " : ""}`
}

function insertTranscribedText(target: VoiceTarget, rawText: string): void {
  const text = rawText.trim()
  if (!text) return

  target.focus()
  if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) {
    const value = target.value
    const start = target.selectionStart ?? value.length
    const end = target.selectionEnd ?? start
    const insertion = withWordBoundary(value, start, end, text)
    const nextValue = `${value.slice(0, start)}${insertion}${value.slice(end)}`
    setNativeValue(target, nextValue)
    const cursor = start + insertion.length
    target.setSelectionRange(cursor, cursor)
    dispatchInputEvents(target, insertion)
    return
  }

  const selection = window.getSelection()
  const currentText = target.textContent || ""
  const insertion =
    currentText.length > 0 && !/\s$/.test(currentText) ? ` ${text}` : text

  if (
    selection &&
    selection.rangeCount > 0 &&
    selection.anchorNode &&
    target.contains(selection.anchorNode)
  ) {
    document.execCommand("insertText", false, insertion)
  } else {
    target.appendChild(document.createTextNode(insertion))
  }
  dispatchInputEvents(target, insertion)
}

function chooseMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
  ]
  return candidates.find((type) => MediaRecorder.isTypeSupported(type))
}

function extensionForMimeType(mimeType: string): string {
  const normalized = mimeType.toLowerCase()
  if (normalized.includes("mp4")) return "m4a"
  if (normalized.includes("ogg")) return "ogg"
  if (normalized.includes("wav")) return "wav"
  return "webm"
}

export function useVoiceInputControls(
  { autoRefresh = true }: UseVoiceInputControlsOptions = {}
) {
  const { t } = useI18n()
  const [hasAsrModel, setHasAsrModel] = useState(false)
  const [status, setStatus] = useState<VoiceStatus>("idle")
  const targetRef = useRef<VoiceTarget | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const recordingTargetRef = useRef<VoiceTarget | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<BlobPart[]>([])
  const lastAvailabilityFetchRef = useRef(0)

  const refreshAvailability = useCallback(async () => {
    lastAvailabilityFetchRef.current = Date.now()
    try {
      const response = await apiRequest(
        `${getApiUrl()}/api/models/?category=speech&limit=1000`
      )
      if (!response.ok) {
        setHasAsrModel(false)
        return
      }
      const models = await response.json()
      const available =
        Array.isArray(models) &&
        models.some((model) => {
          const abilities = Array.isArray(model?.abilities) ? model.abilities : []
          return abilities.map(String).includes("asr")
        })
      setHasAsrModel(available)
    } catch {
      setHasAsrModel(false)
    }
  }, [])

  useEffect(() => {
    if (autoRefresh) {
      refreshAvailability()
    }
  }, [autoRefresh, refreshAvailability])

  const refreshAvailabilityIfStale = useCallback(() => {
    if (Date.now() - lastAvailabilityFetchRef.current > 30_000) {
      refreshAvailability()
    }
  }, [refreshAvailability])

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
  }, [])

  const transcribeBlob = useCallback(
    async (blob: Blob, target: VoiceTarget | null) => {
      if (!target) return
      if (blob.size === 0) {
        toast.error(t("voiceInput.errors.emptyAudio"))
        return
      }

      setStatus("transcribing")
      try {
        const mimeType = blob.type || "audio/webm"
        const extension = extensionForMimeType(mimeType)
        const file = new File([blob], `voice-input.${extension}`, {
          type: mimeType,
        })
        const formData = new FormData()
        formData.append("file", file)

        const response = await apiRequest(
          `${getUploadApiUrl()}/api/models/speech/transcribe`,
          {
            method: "POST",
            body: formData,
          }
        )
        const parsed = await parseApiResponse(response)
        if (!response.ok) {
          throw new Error(
            getApiErrorMessage(response, parsed, t("voiceInput.errors.failed"))
          )
        }

        const text =
          isJsonRecord(parsed.data) && typeof parsed.data.text === "string"
            ? parsed.data.text
            : ""
        if (!text.trim()) {
          toast.error(t("voiceInput.errors.noText"))
          return
        }
        insertTranscribedText(target, text)
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : t("voiceInput.errors.failed")
        )
      } finally {
        setStatus("idle")
      }
    },
    [t]
  )

  const startRecording = useCallback(
    async (target?: VoiceTarget | null) => {
      const recordingTarget = target ?? targetRef.current
      if (!recordingTarget || status !== "idle") return
      if (
        !navigator.mediaDevices?.getUserMedia ||
        typeof MediaRecorder === "undefined"
      ) {
        toast.error(t("voiceInput.errors.unsupported"))
        return
      }

      targetRef.current = recordingTarget

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        streamRef.current = stream
        chunksRef.current = []
        recordingTargetRef.current = recordingTarget
        const mimeType = chooseMimeType()
        const recorder = new MediaRecorder(
          stream,
          mimeType ? { mimeType } : undefined
        )
        recorderRef.current = recorder

        recorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            chunksRef.current.push(event.data)
          }
        }
        recorder.onstop = () => {
          const blob = new Blob(chunksRef.current, {
            type: recorder.mimeType || mimeType || "audio/webm",
          })
          chunksRef.current = []
          recorderRef.current = null
          const target = recordingTargetRef.current
          recordingTargetRef.current = null
          stopStream()
          transcribeBlob(blob, target)
        }
        recorder.onerror = () => {
          stopStream()
          recorderRef.current = null
          recordingTargetRef.current = null
          chunksRef.current = []
          setStatus("idle")
          toast.error(t("voiceInput.errors.failed"))
        }

        recorder.start()
        setStatus("recording")
      } catch (error) {
        stopStream()
        recordingTargetRef.current = null
        setStatus("idle")
        toast.error(
          error instanceof DOMException && error.name === "NotAllowedError"
            ? t("voiceInput.errors.permissionDenied")
            : t("voiceInput.errors.failed")
        )
      }
    },
    [status, stopStream, t, transcribeBlob]
  )

  const stopRecording = useCallback(() => {
    const recorder = recorderRef.current
    if (recorder && recorder.state !== "inactive") {
      recorder.stop()
    }
  }, [])

  useEffect(() => {
    return () => {
      if (recorderRef.current?.state !== "inactive") {
        recorderRef.current?.stop()
      }
      recordingTargetRef.current = null
      stopStream()
    }
  }, [stopStream])

  return {
    hasAsrModel,
    refreshAvailability,
    refreshAvailabilityIfStale,
    startRecording,
    status,
    stopRecording,
  }
}

export function VoiceInputController() {
  const { t } = useI18n()
  const {
    hasAsrModel,
    refreshAvailabilityIfStale,
    startRecording,
    status,
    stopRecording,
  } = useVoiceInputControls({ autoRefresh: false })
  const [activeTarget, setActiveTarget] = useState<VoiceTarget | null>(null)
  const [position, setPosition] = useState<{ top: number; left: number } | null>(
    null
  )
  const targetRef = useRef<VoiceTarget | null>(null)

  const updatePosition = useCallback((target = targetRef.current) => {
    if (!target || isElementHidden(target)) {
      setPosition(null)
      return
    }

    const anchor = getPositionAnchor(target)
    const rect = anchor.getBoundingClientRect()
    if (
      rect.bottom < 0 ||
      rect.top > window.innerHeight ||
      rect.right < 0 ||
      rect.left > window.innerWidth
    ) {
      setPosition(null)
      return
    }

    const buttonSize = 32
    const viewportPadding = 8
    const anchorInset = 8
    const actionRowRightInset = 72
    const actionRowBottomInset = 24
    const useActionRowPosition = anchor !== target && rect.height >= 88
    const regularTop =
      rect.height > buttonSize + anchorInset * 2
        ? rect.top + anchorInset
        : rect.top + (rect.height - buttonSize) / 2
    const anchorTop = useActionRowPosition
      ? rect.bottom - buttonSize - actionRowBottomInset
      : regularTop
    const outsideLeft = rect.right + 6
    const insideLeft =
      rect.right -
      buttonSize -
      (useActionRowPosition ? actionRowRightInset : anchorInset)
    const hasOutsideSpace =
      anchor === target &&
      outsideLeft + buttonSize <= window.innerWidth - viewportPadding
    const left = clamp(
      hasOutsideSpace ? outsideLeft : insideLeft,
      viewportPadding,
      window.innerWidth - buttonSize - viewportPadding
    )
    const top = clamp(
      anchorTop,
      viewportPadding,
      window.innerHeight - buttonSize - viewportPadding
    )
    setPosition({ top, left })
  }, [])

  const activateTarget = useCallback(
    (target: VoiceTarget) => {
      if (status !== "idle") return

      refreshAvailabilityIfStale()

      targetRef.current = target
      setActiveTarget(target)
      updatePosition(target)
    },
    [refreshAvailabilityIfStale, status, updatePosition]
  )

  const clearTarget = useCallback(() => {
    if (status !== "idle") return
    setActiveTarget(null)
    targetRef.current = null
    setPosition(null)
  }, [status])

  useEffect(() => {
    const focusedTarget = document.activeElement
    if (isVoiceEligibleTarget(focusedTarget)) {
      activateTarget(focusedTarget)
    }

    const handleFocusIn = (event: FocusEvent) => {
      const target = event.target
      if (!isVoiceEligibleTarget(target)) {
        if (!isInsideActiveVoiceArea(target, targetRef.current)) clearTarget()
        return
      }
      activateTarget(target)
    }

    const handlePointerOver = (event: PointerEvent) => {
      const target = event.target
      if (isVoiceEligibleTarget(target)) {
        activateTarget(target)
        return
      }
      if (isInsideActiveVoiceArea(target, targetRef.current)) return
      if (document.activeElement === targetRef.current) return
      clearTarget()
    }

    document.addEventListener("focusin", handleFocusIn)
    document.addEventListener("pointerover", handlePointerOver)
    return () => {
      document.removeEventListener("focusin", handleFocusIn)
      document.removeEventListener("pointerover", handlePointerOver)
    }
  }, [activateTarget, clearTarget])

  useEffect(() => {
    let frameId: number | null = null
    const reposition = () => {
      if (frameId !== null) return
      frameId = window.requestAnimationFrame(() => {
        frameId = null
        updatePosition()
      })
    }
    window.addEventListener("resize", reposition)
    document.addEventListener("scroll", reposition, true)
    return () => {
      window.removeEventListener("resize", reposition)
      document.removeEventListener("scroll", reposition, true)
      if (frameId !== null) {
        window.cancelAnimationFrame(frameId)
      }
    }
  }, [updatePosition])

  const visible = hasAsrModel && !!activeTarget && !!position

  if (!visible || !position) {
    return null
  }

  const label =
    status === "recording"
      ? t("voiceInput.stop")
      : status === "transcribing"
        ? t("voiceInput.transcribing")
        : t("voiceInput.start")

  return (
    <button
      type="button"
      data-voice-input-root
      aria-label={label}
      title={label}
      className={cn(
        "fixed z-[70] inline-flex h-8 w-8 items-center justify-center rounded-full border shadow-md transition-all",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        status === "recording"
          ? "border-red-400 bg-red-500 text-white hover:bg-red-600"
          : "border-border bg-background text-muted-foreground hover:bg-accent hover:text-foreground",
        status === "transcribing" && "cursor-wait opacity-80"
      )}
      style={{ top: position.top, left: position.left }}
      disabled={status === "transcribing"}
      onMouseDown={(event) => event.preventDefault()}
      onClick={() => {
        if (status === "recording") {
          stopRecording()
        } else if (status === "idle") {
          startRecording(activeTarget)
        }
      }}
    >
      {status === "recording" ? (
        <Square className="h-3.5 w-3.5 fill-current" />
      ) : status === "transcribing" ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : (
        <Mic className="h-4 w-4" />
      )}
    </button>
  )
}
