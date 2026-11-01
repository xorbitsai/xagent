"use client"

import { useEffect, useMemo, useState } from "react"
import { useAuth } from "@/contexts/auth-context"
import { apiRequest } from "@/lib/api-wrapper"
import { parseModelList, type ModelWithAccess } from "@/lib/models"
import { getApiUrl } from "@/lib/utils"

export function voiceModelOptions(models: ModelWithAccess[]) {
  const speech = models.filter(model => model.is_active && model.category === "speech" && model.model_provider.trim().toLowerCase() === "elevenlabs")
  return {
    // This adapter provides streaming ASR and server-side VAD together.
    asr: speech.filter(model => model.model_name === "scribe_v2_realtime" && model.abilities?.includes("asr")),
    tts: speech.filter(model => model.abilities?.includes("tts")),
  }
}

export type VoiceModelReason = "checkingModels" | "modelsUnavailable" | "needsAsrAndTts" | "needsAsr" | "needsTts" | "needsLlm"
export function voiceModelReason(state: { asr: ModelWithAccess[]; tts: ModelWithAccess[]; loading: boolean; error: boolean }, llmAvailable = true): VoiceModelReason | undefined {
  if (state.loading) return "checkingModels"
  if (state.error) return "modelsUnavailable"
  if (!llmAvailable) return "needsLlm"
  if (!state.asr.length && !state.tts.length) return "needsAsrAndTts"
  if (!state.asr.length) return "needsAsr"
  if (!state.tts.length) return "needsTts"
}

export function useVoiceModels(enabled: boolean, refreshKey = false) {
  const auth = useAuth()
  const owner = useMemo(() => ({}), [enabled, auth.token, auth.user?.id, auth.session?.sessionId])
  const [state, setState] = useState<{ owner: object | null; models: ModelWithAccess[]; loading: boolean; error: boolean }>({ owner: null, models: [], loading: true, error: false })
  useEffect(() => {
    if (!enabled || !auth.token) return
    let active = true
    let request = 0
    const load = async () => {
      const current = ++request
      setState(previous => previous.owner === owner
        ? { ...previous, loading: previous.loading || previous.error, error: false }
        : { owner, models: [], loading: true, error: false })
      try {
        const response = await apiRequest(`${getApiUrl()}/api/models/?category=speech&limit=1000`)
        if (!response.ok) throw new Error("Models unavailable")
        const models = parseModelList(await response.json())
        if (!models) throw new Error("Invalid model list")
        if (active && request === current) setState({ owner, models, loading: false, error: false })
      } catch {
        if (active && request === current) setState(previous => ({ owner, models: previous.owner === owner ? previous.models : [], loading: false, error: true }))
      }
    }
    void load()
    const refresh = () => { if (document.visibilityState === "visible") void load() }
    window.addEventListener("focus", refresh)
    document.addEventListener("visibilitychange", refresh)
    return () => {
      active = false
      window.removeEventListener("focus", refresh)
      document.removeEventListener("visibilitychange", refresh)
    }
  }, [enabled, auth.token, owner, refreshKey])
  const current = enabled && Boolean(auth.token) && state.owner === owner
  return {
    ...voiceModelOptions(current ? state.models : []),
    loading: enabled && Boolean(auth.token) && (!current || state.loading),
    error: current && state.error,
  }
}
