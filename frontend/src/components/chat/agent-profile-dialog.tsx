"use client"

import React, { useEffect, useRef, useState, type ReactNode } from "react"
import { Loader2 } from "lucide-react"
import { PersonaAvatar } from "@/components/templates/persona-avatar"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog"
import { toast } from "@/components/ui/sonner"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
import { canEditAgent } from "@/lib/agent-ui-access"
import { parseAgentUpdateError } from "@/lib/agent-widget-config"
import { getApiUrl } from "@/lib/utils"

interface Profile {
  id: number
  name: string
  description?: string | null
  instructions?: string | null
  status: string
  readonly?: boolean
  can_edit?: boolean
}

interface AgentProfileDialogProps {
  agent: Profile
  avatar: string | null
  onClose: () => void
  onSaved: (updated: Record<string, unknown>) => void
  children?: ReactNode
}

/** Mounted per opening so Cancel discards drafts and reopening reads fresh data. */
export function AgentProfileDialog({ agent, avatar, onClose, onSaved, children }: AgentProfileDialogProps) {
  const { t } = useI18n()
  const [profile, setProfile] = useState(agent)
  const [draft, setDraft] = useState({ name: "", description: "", instructions: "" })
  const [loading, setLoading] = useState(true)
  const [loaded, setLoaded] = useState(false)
  const [retry, setRetry] = useState(0)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState("")
  const mounted = useRef(false)
  const savingRef = useRef(false)
  const canEdit = loaded && canEditAgent(profile)
  const instructionsChanged = draft.instructions !== (profile.instructions ?? "")
  const invalidInstructions = instructionsChanged && !draft.instructions.trim()
  const dirty = draft.name !== profile.name
    || draft.description !== (profile.description ?? "")
    || instructionsChanged
  const statusLabel = profile.status === "published"
    ? t("builds.list.status.published")
    : profile.status === "draft" ? t("builds.list.status.draft")
      : profile.status === "archived" ? t("builds.list.status.archived") : profile.status

  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false }
  }, [])

  useEffect(() => {
    let current = true
    setLoading(true)
    setError("")
    void (async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/agents/${agent.id}`)
        if (!response.ok) throw new Error(t("agentProfile.loadFailed"))
        const data = await response.json()
        if (data.id !== agent.id || typeof data.name !== "string") throw new Error(t("agentProfile.loadFailed"))
        if (!current) return
        setProfile(data)
        setDraft({ name: data.name, description: data.description ?? "", instructions: data.instructions ?? "" })
        setLoaded(true)
      } catch {
        if (current) setError(t("agentProfile.loadFailed"))
      } finally {
        if (current) setLoading(false)
      }
    })()
    return () => { current = false }
  }, [agent.id, retry, t])

  const close = () => {
    if (!savingRef.current) onClose()
  }

  const save = async () => {
    if (!canEdit || !dirty || savingRef.current) return
    if (!draft.name.trim()) {
      setError(t("builds.editor.validation.nameRequired"))
      return
    }
    if (invalidInstructions) {
      setError(t("builds.editor.validation.instructionsRequired"))
      return
    }
    savingRef.current = true
    setSaving(true)
    setError("")
    try {
      const response = await apiRequest(`${getApiUrl()}/api/agents/${agent.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: draft.name.trim(),
          description: draft.description,
          // Omit untouched rules so metadata edits preserve null/empty values
          // (and do not overwrite instructions changed elsewhere).
          ...(instructionsChanged ? { instructions: draft.instructions } : {}),
        }),
      })
      if (!response.ok) throw await parseAgentUpdateError(response, t("agentProfile.saveFailed"))
      const updated = await response.json()
      if (updated.id !== agent.id || typeof updated.name !== "string") throw new Error(t("agentProfile.saveFailed"))
      if (!mounted.current) return
      onSaved(updated)
      toast.success(t("agentProfile.saved"))
      onClose()
    } catch (cause) {
      if (mounted.current) setError(cause instanceof Error ? cause.message : t("agentProfile.saveFailed"))
    } finally {
      savingRef.current = false
      if (mounted.current) setSaving(false)
    }
  }

  const inputClass = "h-10 w-full rounded-xl border border-border bg-background px-3 text-sm font-normal outline-none focus:border-primary focus:ring-2 focus:ring-primary/15 read-only:bg-muted/30 disabled:opacity-60"

  return (
    <Dialog open onOpenChange={(open) => { if (!open) close() }}>
      <DialogContent
        className="flex max-h-[90dvh] w-[min(500px,calc(100vw-24px))] max-w-none flex-col gap-0 overflow-hidden rounded-[20px] p-0"
        showCloseButton={!saving}
      >
        <div className="flex shrink-0 items-center gap-3 border-b border-border px-5 py-4 pr-12">
          <PersonaAvatar persona={{ name: profile.name, avatar }} sizeClassName="h-10 w-10" textClassName="text-sm" decorative />
          <div>
            <DialogTitle className="text-lg font-semibold">{t("agentProfile.title", { name: profile.name })}</DialogTitle>
            <DialogDescription className="text-xs">{t("agentProfile.subtitle")}</DialogDescription>
          </div>
        </div>
        <form className="flex min-h-0 flex-col" onSubmit={(event) => { event.preventDefault(); void save() }}>
          <div className="space-y-4 overflow-y-auto px-5 py-4">
            {loading ? <p role="status" className="text-sm text-muted-foreground">{t("common.loading")}</p> : loaded && <>
              {!canEdit && <p className="text-sm text-muted-foreground">{t("agentProfile.readOnly")}</p>}
              <div className="grid grid-cols-2 gap-3">
                <label className="space-y-1.5 text-xs font-semibold">{t("agentProfile.name")}
                  <input value={draft.name} onChange={(event) => setDraft({ ...draft, name: event.target.value })} readOnly={!canEdit} disabled={saving} maxLength={200} required className={inputClass} />
                </label>
                <label className="space-y-1.5 text-xs font-semibold">{t("agentProfile.status")}
                  <input value={statusLabel} readOnly className={inputClass} />
                </label>
              </div>
              <label className="block space-y-1.5 text-xs font-semibold">{t("agentProfile.description")}
                <textarea value={draft.description} onChange={(event) => setDraft({ ...draft, description: event.target.value })} readOnly={!canEdit} disabled={saving} rows={3} className={`${inputClass} h-auto resize-y py-2 leading-relaxed`} />
              </label>
              <label className="block space-y-1.5 text-xs font-semibold">{t("agentProfile.rules")}
                <textarea value={draft.instructions} onChange={(event) => setDraft({ ...draft, instructions: event.target.value })} readOnly={!canEdit} disabled={saving} rows={5} required={instructionsChanged} className={`${inputClass} h-auto resize-y py-2 leading-relaxed`} />
              </label>
            </>}
            {error && <div role="alert" className="text-sm text-destructive">{error}{!loaded && !loading && <Button type="button" variant="outline" className="ml-2" onClick={() => setRetry((value) => value + 1)}>{t("common.retry")}</Button>}</div>}
            <fieldset disabled={saving} className="space-y-4 disabled:opacity-60">{children}</fieldset>
          </div>
          <div className="flex shrink-0 justify-end gap-2 border-t border-border px-5 py-3">
            <Button type="button" variant="outline" onClick={close} disabled={saving}>{t("common.cancel")}</Button>
            {canEdit && <Button type="submit" disabled={saving || !dirty || !draft.name.trim() || invalidInstructions}>
              {saving && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {saving ? t("agentProfile.saving") : t("common.save")}
            </Button>}
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}
