"use client"

import { useCallback, useEffect, useState } from "react"
import { PlusCircle, X } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { MultiSelect } from "@/components/ui/multi-select"
import {
  SelectRadix,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { toast } from "@/components/ui/sonner"
import { useAuth } from "@/contexts/auth-context"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"

interface SshBinding {
  public_id: string
  target_public_id: string | null
  target_alias: string | null
  tool_alias: string
  capabilities: string[]
  approval_policy: string
}

interface TargetOption {
  public_id: string
  alias: string
  display_name: string | null
  hostname: string
}

const CAPABILITIES = ["execute", "upload", "download"] as const
const APPROVAL_POLICIES = ["always", "risk_based", "not_required"] as const

interface AgentSshBindingsProps {
  agentId?: string
  readOnly?: boolean
  /** Reports the current binding count so the editor can auto-enable the
   *  "ssh" tool category when the agent has at least one bound target. */
  onCount?: (count: number) => void
}

/**
 * SSH target bindings for an agent. Bindings are a per-agent sub-resource
 * (POST/GET/DELETE /api/agents/{id}/ssh-targets), managed inline rather than
 * folded into the agent save — so they need an existing agent.
 */
export function AgentSshBindings({ agentId, readOnly = false, onCount }: AgentSshBindingsProps) {
  const { t } = useI18n()
  const [bindings, setBindings] = useState<SshBinding[]>([])
  const [targets, setTargets] = useState<TargetOption[]>([])
  const [dialogOpen, setDialogOpen] = useState(false)
  const [targetPublicId, setTargetPublicId] = useState("")
  const [capabilities, setCapabilities] = useState<string[]>(["execute"])
  const [approvalPolicy, setApprovalPolicy] = useState<string>("always")
  const [submitting, setSubmitting] = useState(false)
  const { inTeam } = useAuth()

  const load = useCallback(async () => {
    if (!agentId) return
    try {
      const res = await apiRequest(`${getApiUrl()}/api/agents/${agentId}/ssh-targets`)
      if (!res.ok) throw new Error(await res.text())
      const data: SshBinding[] = await res.json()
      setBindings(data)
      onCount?.(data.length)
    } catch {
      // Surface the failure instead of silently treating it as "no bindings":
      // saving in this state would drop the "ssh" tool category even though
      // bindings exist server-side, since onCount never fires.
      toast.error(t("ssh.bindings.loadFailed"))
    }
  }, [agentId, onCount, t])

  useEffect(() => {
    load()
  }, [load])

  useEffect(() => {
    if (!dialogOpen) return
    ;(async () => {
      try {
        const res = await apiRequest(`${getApiUrl()}/api/ssh/targets?scope=user`)
        if (!res.ok) throw new Error(await res.text())
        setTargets(await res.json())
      } catch {
        // Surface the failure instead of leaving a silently-empty dropdown,
        // mirroring the bindings-load() toast in this same file (m5).
        toast.error(t("ssh.bindings.targetsLoadFailed"))
      }
    })()
  }, [dialogOpen, t])

  function resetForm() {
    setTargetPublicId("")
    setCapabilities(["execute"])
    setApprovalPolicy("always")
  }

  // The agent-facing tool alias is derived from the target's own alias so the
  // user doesn't have to invent one. If this agent already bound another
  // target with that alias (possible across owner scopes), suffix it to keep
  // the per-agent alias unique.
  function deriveToolAlias(): string {
    const target = targets.find((tg) => tg.public_id === targetPublicId)
    const base = (target?.alias || targetPublicId).trim()
    const taken = new Set(bindings.map((b) => b.tool_alias))
    let alias = base
    for (let i = 2; taken.has(alias); i++) alias = `${base}-${i}`
    return alias
  }

  async function submit() {
    if (submitting || !agentId) return
    setSubmitting(true)
    try {
      const res = await apiRequest(`${getApiUrl()}/api/agents/${agentId}/ssh-targets`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_public_id: targetPublicId,
          tool_alias: deriveToolAlias(),
          capabilities,
          approval_policy: approvalPolicy,
        }),
      })
      if (!res.ok) throw new Error(await res.text())
      toast.success(t("ssh.bindings.created"))
      setDialogOpen(false)
      resetForm()
      await load()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setSubmitting(false)
    }
  }

  async function remove(binding: SshBinding) {
    if (readOnly || !agentId) return
    if (!window.confirm(t("ssh.bindings.confirmDelete"))) return
    try {
      const res = await apiRequest(
        `${getApiUrl()}/api/agents/${agentId}/ssh-targets/${binding.public_id}`,
        { method: "DELETE" },
      )
      if (!res.ok) throw new Error(await res.text())
      toast.success(t("ssh.bindings.deleted"))
      await load()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    }
  }

  if (!inTeam) return null

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label>{t("ssh.bindings.label")}</Label>
        {agentId && !readOnly && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => setDialogOpen(true)}
            className="h-7 border-dashed border-primary/45 bg-primary/5 px-2 text-xs text-primary hover:border-primary hover:bg-primary/10"
          >
            <PlusCircle className="mr-1 h-3.5 w-3.5" />
            {t("ssh.bindings.add")}
          </Button>
        )}
      </div>
      {!agentId ? (
        <p className="text-xs text-muted-foreground">{t("ssh.bindings.saveFirst")}</p>
      ) : bindings.length === 0 ? (
        <p className="text-xs text-muted-foreground">{t("ssh.bindings.empty")}</p>
      ) : (
        <div className="flex flex-col gap-2">
          {bindings.map((b) => (
            <div key={b.public_id} className="flex items-center gap-3 rounded-md border p-2">
              <div className="min-w-0">
                {/* tool_alias is auto-derived from the target alias, so showing
                    both is redundant — show the single alias the user configured. */}
                <div className="truncate text-sm font-medium">
                  {b.target_alias || b.tool_alias}
                </div>
              </div>
              <div className="flex flex-wrap gap-1">
                {b.capabilities.map((c) => (
                  <Badge key={c} variant="outline">
                    {c}
                  </Badge>
                ))}
                <Badge>{b.approval_policy}</Badge>
              </div>
              {!readOnly && (
                <div className="ml-auto">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-8 w-8 p-0 text-red-500 hover:bg-red-50 hover:text-red-600"
                    onClick={() => remove(b)}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <Dialog
        open={dialogOpen}
        onOpenChange={(next) => {
          if (submitting) return
          if (!next) resetForm()
          setDialogOpen(next)
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("ssh.bindings.dialog.title")}</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>{t("ssh.bindings.dialog.targetLabel")}</Label>
              <SelectRadix value={targetPublicId} onValueChange={setTargetPublicId}>
                <SelectTrigger>
                  <SelectValue placeholder={t("ssh.bindings.dialog.targetPlaceholder")} />
                </SelectTrigger>
                <SelectContent>
                  {targets.map((tg) => (
                    <SelectItem key={tg.public_id} value={tg.public_id}>
                      {(tg.display_name || tg.alias) + " (" + tg.hostname + ")"}
                    </SelectItem>
                  ))}
                </SelectContent>
              </SelectRadix>
            </div>
            <div className="space-y-2">
              <Label>{t("ssh.bindings.dialog.capabilitiesLabel")}</Label>
              <MultiSelect
                values={capabilities}
                onValuesChange={setCapabilities}
                options={CAPABILITIES.map((c) => ({ value: c, label: c }))}
                placeholder={t("ssh.bindings.dialog.capabilitiesPlaceholder")}
              />
            </div>
            <div className="space-y-2">
              <Label>{t("ssh.bindings.dialog.approvalLabel")}</Label>
              <SelectRadix value={approvalPolicy} onValueChange={setApprovalPolicy}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {APPROVAL_POLICIES.map((p) => (
                    <SelectItem key={p} value={p}>
                      {t(`ssh.bindings.approval.${p}`)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </SelectRadix>
            </div>
          </div>
          <DialogFooter>
            <Button
              onClick={submit}
              disabled={submitting || !targetPublicId || capabilities.length === 0}
            >
              {t("ssh.bindings.dialog.submit")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
