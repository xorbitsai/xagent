"use client"

import { useEffect, useMemo, useState } from "react"
import { useAuth } from "@/contexts/auth-context"
import { apiRequest } from "@/lib/api-wrapper"
import { parseModelList, type ModelWithAccess } from "@/lib/models"
import { getApiUrl } from "@/lib/utils"

export type VoiceCapabilities = Record<string, { asr?: boolean; tts?: boolean }>

export function voiceModelOptions(models: ModelWithAccess[], capabilities: VoiceCapabilities = {}) {
  const speech = models.filter(model => model.is_active && model.category === "speech")
  return {
    // This adapter provides streaming ASR and server-side VAD together.
    asr: speech.filter(model => model.abilities?.includes("asr") && ((model.model_provider.trim().toLowerCase() === "elevenlabs" && model.model_name === "scribe_v2_realtime") || capabilities[model.model_id]?.asr === true)),
    tts: speech.filter(model => model.abilities?.includes("tts") && (model.model_provider.trim().toLowerCase() === "elevenlabs" || capabilities[model.model_id]?.tts === true)),
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
  const cache = useMemo(() => new Map<string, { value: boolean; checked: number }>(), [owner])
  const [state, setState] = useState<{ owner: object | null; models: ModelWithAccess[]; capabilities?: VoiceCapabilities; loading: boolean; error: boolean }>({ owner: null, models: [], loading: true, error: false })
  useEffect(() => {
    if (!enabled || !auth.token) return
    let active = true
    let inFlight: Promise<void> | undefined
    const performLoad = async () => {
      setState(previous => previous.owner === owner
        ? { ...previous, loading: previous.loading || previous.error, error: false }
        : { owner, models: [], loading: true, error: false })
      try {
        const response = await apiRequest(`${getApiUrl()}/api/models/?category=speech&limit=1000`)
        if (!response.ok) throw new Error("Models unavailable")
        const models = parseModelList(await response.json())
        if (!models) throw new Error("Invalid model list")
        const capabilities: VoiceCapabilities = {}
        const jobs: (() => Promise<void>)[] = []
        const keys = new Set<string>()
        let probeFailed = false
        const publish = (checking: boolean) => {
          const options = voiceModelOptions(models, capabilities)
          const ready = options.asr.length > 0 && options.tts.length > 0
          if (active) setState({ owner, models, capabilities: { ...capabilities }, loading: checking && !ready, error: !checking && probeFailed && !ready })
        }
        for (const model of models.filter(model => model.is_active && model.category === "speech" && model.model_provider.trim().toLowerCase() === "xinference")) {
          for (const ability of ["asr", "tts"] as const) {
            if (!model.abilities?.includes(ability)) continue
            const key = JSON.stringify([model.model_id, model.model_provider, model.model_name, model.base_url, model.updated_at, ability])
            keys.add(key)
            const cached = cache.get(key)
            if (cached) capabilities[model.model_id] = { ...capabilities[model.model_id], [ability]: cached.value }
            if (cached && Date.now() - cached.checked < 30000) continue
            jobs.push(async () => {
              try {
                const response = await apiRequest(`${getApiUrl()}/api/models/speech/capabilities?model_id=${encodeURIComponent(model.model_id)}&ability=${ability}`)
                if (!response.ok) throw new Error("Capabilities unavailable")
                const result = await response.json()
                if (!result || typeof result !== "object" || Array.isArray(result)) throw new Error("Invalid capabilities")
                const value = result.streaming === true && (ability !== "asr" || result.server_vad === true)
                if (!active) return
                cache.set(key, { value, checked: Date.now() })
                capabilities[model.model_id] = { ...capabilities[model.model_id], [ability]: value }
                publish(true)
              } catch { probeFailed = true }
            })
          }
        }
        for (const key of cache.keys()) if (!keys.has(key)) cache.delete(key)
        publish(jobs.length > 0)
        await Promise.all(jobs.map(probe => probe()))
        publish(false)
      } catch {
        if (active) setState(previous => ({ owner, models: previous.owner === owner ? previous.models : [], capabilities: previous.owner === owner ? previous.capabilities : {}, loading: false, error: true }))
      }
    }
    const load = () => {
      if (!inFlight) inFlight = performLoad().finally(() => { inFlight = undefined })
      return inFlight
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
  }, [enabled, auth.token, owner, refreshKey, cache])
  const current = enabled && Boolean(auth.token) && state.owner === owner
  return {
    ...voiceModelOptions(current ? state.models : [], current ? state.capabilities : {}),
    loading: enabled && Boolean(auth.token) && (!current || state.loading),
    error: current && state.error,
  }
}
