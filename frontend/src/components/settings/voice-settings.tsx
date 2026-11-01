"use client"

import React, { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { useI18n } from "@/contexts/i18n-context"
import { useVoiceModels, voiceModelReason } from "@/hooks/use-voice-models"
import { useVoicePreferences } from "@/hooks/use-voice-preferences"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { updateUserPreferences, type VoiceConversationPreferences } from "@/lib/user-preferences"
import type { ModelWithAccess } from "@/lib/models"

type Voice = { voice_id: string; name: string }

export function VoiceSettings() {
  const { t } = useI18n()
  const models = useVoiceModels(true)
  const saved = useVoicePreferences(true)
  const [draft, setDraft] = useState<VoiceConversationPreferences>({})
  const [voices, setVoices] = useState<{ model: string; items: Voice[]; error: boolean } | null>(null)
  const [saving, setSaving] = useState(false)
  const [status, setStatus] = useState("")
  useEffect(() => { if (!saved.loading && !saved.error) setDraft(saved.preferences) }, [saved.preferences, saved.loading, saved.error])
  const ttsId = draft.tts_model_id || models.tts[0]?.model_id || ""
  const availableTts = models.tts.some(model => model.model_id === ttsId)
  useEffect(() => {
    if (!ttsId || !availableTts) return
    let active = true
    setVoices(null)
    void (async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/models/speech/voices?model_id=${encodeURIComponent(ttsId)}`)
        if (!response.ok) throw new Error()
        const items: unknown = await response.json()
        if (!Array.isArray(items) || items.some(item => typeof item?.voice_id !== "string" || typeof item?.name !== "string")) throw new Error()
        if (active) setVoices({ model: ttsId, items, error: false })
      } catch { if (active) setVoices({ model: ttsId, items: [], error: true }) }
    })()
    return () => { active = false }
  }, [ttsId, availableTts])
  const ready = !saved.loading && !saved.error && !models.loading && !models.error
  const available = (items: ModelWithAccess[], id?: string | null) => !id || items.some(model => model.model_id === id)
  const validModels = available(models.asr, draft.asr_model_id) && available(models.tts, draft.tts_model_id)
  const reason = voiceModelReason(models)
  const currentVoices = voices?.model === ttsId ? voices : null
  const savedVoice = draft.tts_model_id === saved.preferences.tts_model_id && draft.voice === saved.preferences.voice
  const validVoice = !draft.voice || Boolean(currentVoices?.items.some(voice => voice.voice_id === draft.voice)) || Boolean(currentVoices?.error && savedVoice)
  const change = (patch: VoiceConversationPreferences) => { setDraft(previous => ({ ...previous, ...patch })); setStatus("") }
  const modelSelect = (kind: "asr" | "tts", label: string) => {
    const field = kind === "asr" ? "asr_model_id" : "tts_model_id"
    const value = draft[field] || ""
    return <label className="block space-y-2 text-sm font-medium">
      <span>{label}</span>
      <select className="w-full rounded-md border bg-background p-2 font-normal" aria-label={label} value={value} disabled={!ready || saving} onChange={event => change({ [field]: event.target.value || null, ...(kind === "tts" ? { voice: null } : {}) })}>
        <option value="">{t("settings.voice.automatic")}</option>
        {!available(models[kind], value) && <option value={value} disabled>{t("settings.voice.selectionUnavailable")}</option>}
        {models[kind].map(model => <option key={model.id} value={model.model_id}>{model.model_name} · {model.model_id}</option>)}
      </select>
    </label>
  }
  async function save() {
    if (!ready || !validModels || !validVoice || saving) return
    setSaving(true); setStatus("")
    const result = await updateUserPreferences({ voice_conversation: {
      asr_model_id: draft.asr_model_id || null,
      tts_model_id: draft.tts_model_id || null,
      voice: draft.voice || null,
    } })
    setStatus(t(result.ok ? "settings.voice.saved" : "settings.voice.saveFailed"))
    setSaving(false)
  }
  return <div className="space-y-6">
    <p className="text-sm text-muted-foreground">{t("settings.voice.description")}</p>
    {saved.error && <p role="alert" className="text-sm text-destructive">{t("settings.voice.loadFailed")}</p>}
    {reason && <p role="status" className="text-sm text-muted-foreground">{t(`voiceConversation.${reason}`)}</p>}
    {modelSelect("asr", t("voiceConversation.transcriptionModel"))}
    {modelSelect("tts", t("voiceConversation.speechModel"))}
    <label className="block space-y-2 text-sm font-medium">
      <span>{t("settings.voice.speaker")}</span>
      <select aria-label={t("settings.voice.speaker")} className="w-full rounded-md border bg-background p-2 font-normal" value={draft.voice || ""} disabled={!ready || saving || !availableTts} onChange={event => change({ voice: event.target.value || null, ...(event.target.value ? { tts_model_id: ttsId } : {}) })}>
        <option value="">{t("settings.voice.modelDefault")}</option>
        {draft.voice && !currentVoices?.items.some(voice => voice.voice_id === draft.voice) && <option value={draft.voice} disabled>{t(currentVoices?.error && savedVoice ? "settings.voice.savedVoice" : "settings.voice.selectionUnavailable")}</option>}
        {currentVoices?.items.map(voice => <option key={voice.voice_id} value={voice.voice_id}>{voice.name}</option>)}
      </select>
    </label>
    {availableTts && !currentVoices && <p role="status" className="text-sm text-muted-foreground">{t("settings.voice.loadingVoices")}</p>}
    {currentVoices?.error && <p role="alert" className="text-sm text-muted-foreground">{t("settings.voice.voicesFailed")}</p>}
    <Button disabled={!ready || !validModels || !validVoice || saving} onClick={() => void save()}>{t(saving ? "common.loading" : "common.save")}</Button>
    {status && <p role="status" className="text-sm">{status}</p>}
  </div>
}
