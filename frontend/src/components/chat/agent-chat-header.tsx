"use client"

import React, { useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { useRouter } from "next/navigation"
import {
  ArrowLeft,
  CalendarClock,
  Code2,
  MoreHorizontal,
  Rocket,
  Settings2,
  User,
} from "lucide-react"

import { AgentTriggersDialog } from "@/components/build/agent-triggers-dialog"
import { DeployAgentDialog, type Agent as DeployAgent } from "@/components/build/deploy-agent-dialog"
import { PersonaAvatar } from "@/components/templates/persona-avatar"
import { AgentProfileDialog } from "@/components/chat/agent-profile-dialog"
import { useI18n } from "@/contexts/i18n-context"
import { useApp } from "@/contexts/app-context-chat"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl, resolveAgentLogoUrl } from "@/lib/utils"

interface AgentDetails extends DeployAgent {
  instructions?: string | null
  execution_mode?: string | null
  suggested_prompts?: string[]
  title?: string | null
}

interface AgentChatHeaderProps {
  agentId: number
  name?: string | null
  description?: string | null
  logoUrl?: string | null
  showBack?: boolean
  rightSlot?: ReactNode
}

function normalizeAgent(value: unknown, agentId: number): AgentDetails {
  const candidate = value && typeof value === "object"
    ? value as Record<string, unknown>
    : {}
  return {
    id: Number(candidate.id) || agentId,
    name: typeof candidate.name === "string" ? candidate.name : "Agent",
    description: typeof candidate.description === "string" ? candidate.description : "",
    logo_url: typeof candidate.logo_url === "string" ? candidate.logo_url : null,
    template_id: typeof candidate.template_id === "string" ? candidate.template_id : null,
    status: typeof candidate.status === "string" ? candidate.status : "draft",
    created_at: typeof candidate.created_at === "string" ? candidate.created_at : "",
    updated_at: typeof candidate.updated_at === "string" ? candidate.updated_at : "",
    widget_enabled: Boolean(candidate.widget_enabled),
    allowed_domains: Array.isArray(candidate.allowed_domains)
      ? candidate.allowed_domains.filter((domain): domain is string => typeof domain === "string")
      : [],
    access: typeof candidate.access === "string" ? candidate.access : undefined,
    readonly: typeof candidate.readonly === "boolean" ? candidate.readonly : undefined,
    can_edit: typeof candidate.can_edit === "boolean" ? candidate.can_edit : undefined,
    can_publish: typeof candidate.can_publish === "boolean" ? candidate.can_publish : undefined,
    can_delete: typeof candidate.can_delete === "boolean" ? candidate.can_delete : undefined,
    share_enabled: typeof candidate.share_enabled === "boolean" ? candidate.share_enabled : undefined,
    share_updated_at: typeof candidate.share_updated_at === "string" ? candidate.share_updated_at : null,
    instructions: typeof candidate.instructions === "string" ? candidate.instructions : null,
    execution_mode: typeof candidate.execution_mode === "string" ? candidate.execution_mode : null,
    suggested_prompts: Array.isArray(candidate.suggested_prompts)
      ? candidate.suggested_prompts.filter((prompt): prompt is string => typeof prompt === "string")
      : [],
    title: typeof candidate.title === "string" ? candidate.title : null,
  }
}

function AgentChatHeaderContent({
  agentId,
  name,
  description,
  logoUrl,
  showBack = false,
  rightSlot,
}: AgentChatHeaderProps) {
  const { t } = useI18n()
  const router = useRouter()
  const { dispatch } = useApp()
  const detailVersion = useRef(0)
  const [agent, setAgent] = useState<AgentDetails>(() => normalizeAgent({ name, description, logo_url: logoUrl }, agentId))
  const [menuOpen, setMenuOpen] = useState(false)
  const [profileOpen, setProfileOpen] = useState(false)
  const [deployOpen, setDeployOpen] = useState(false)
  const [routinesOpen, setRoutinesOpen] = useState(false)

  useEffect(() => {
    let cancelled = false
    const version = detailVersion.current
    apiRequest(`${getApiUrl()}/api/agents/${agentId}`)
      .then(async (response) => response.ok ? response.json() : null)
      .then((data) => { if (!cancelled && version === detailVersion.current && data) setAgent(normalizeAgent(data, agentId)) })
      .catch(() => undefined)
    return () => { cancelled = true }
  }, [agentId])

  const avatar = useMemo(() => resolveAgentLogoUrl(agent.logo_url ?? null, getApiUrl()), [agent.logo_url])
  const statusLabel = t(agent.status === "published" ? "workspace.ready" : "workspace.draft")

  const openAction = (action: () => void) => {
    setMenuOpen(false)
    action()
  }

  return (
    <>
      <header className="flex h-[62px] shrink-0 items-center gap-3 border-b border-border bg-background/95 px-4 backdrop-blur">
        {showBack && (
          <button onClick={() => router.push("/task")} aria-label={t("workspace.backToTask")} className="grid h-8 w-8 place-items-center rounded-full text-muted-foreground hover:bg-muted hover:text-foreground">
            <ArrowLeft className="h-4 w-4" />
          </button>
        )}
        <PersonaAvatar persona={{ name: agent.name, avatar }} sizeClassName="h-8 w-8" textClassName="text-xs" decorative />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <b className="truncate text-[13px] font-semibold">{agent.name}</b>
            <span className="text-[10px] text-muted-foreground">{statusLabel}</span>
          </div>
          <p className="truncate text-[10.5px] text-muted-foreground">{agent.description || t("workspace.aiTeammate")}</p>
        </div>
        <button onClick={() => setRoutinesOpen(true)} className="inline-flex h-8 items-center gap-1.5 rounded-full border border-border px-3 text-[11.5px] font-medium hover:bg-muted">
          <CalendarClock className="h-3.5 w-3.5" /> {t("workspace.routines")}
        </button>
        {rightSlot}
        <Popover open={menuOpen} onOpenChange={setMenuOpen}>
          <PopoverTrigger asChild>
            <button aria-label={t("workspace.moreOptions", { name: agent.name })} className="grid h-8 w-8 place-items-center rounded-full text-muted-foreground hover:bg-muted hover:text-foreground"><MoreHorizontal className="h-4 w-4" /></button>
          </PopoverTrigger>
          <PopoverContent align="end" sideOffset={7} className="w-56 rounded-xl p-1.5 shadow-xl">
            <button onClick={() => openAction(() => setProfileOpen(true))} className="flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-sm hover:bg-accent"><User className="h-4 w-4 text-muted-foreground" /> {t("workspace.profile")}</button>
            <button onClick={() => openAction(() => router.push(`/build/${agent.id}`))} className="flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-sm hover:bg-accent"><Settings2 className="h-4 w-4 text-muted-foreground" /> {t("workspace.builderSettings")}</button>
            <button onClick={() => openAction(() => setDeployOpen(true))} className="flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-sm hover:bg-accent"><Rocket className="h-4 w-4 text-muted-foreground" /> {t("workspace.deploy")}</button>
          </PopoverContent>
        </Popover>
      </header>

      {profileOpen && (
        <AgentProfileDialog
          agent={agent}
          avatar={avatar}
          onClose={() => setProfileOpen(false)}
          onSaved={(updated) => {
            detailVersion.current += 1
            setAgent(normalizeAgent(updated, agentId))
            dispatch({ type: "TRIGGER_TASK_UPDATE" })
          }}
        >
          <button type="button" onClick={() => { setProfileOpen(false); setRoutinesOpen(true) }} className="flex w-full items-center gap-3 rounded-xl border border-border p-3 text-left hover:bg-muted"><CalendarClock className="h-4 w-4 text-[#2536e0]" /><span className="flex-1"><b className="block text-sm font-semibold">{t("workspace.routines")}</b><small className="text-xs text-muted-foreground">{t("workspace.routinesHint")}</small></span><ArrowLeft className="h-4 w-4 rotate-180 text-muted-foreground" /></button>
          <button type="button" onClick={() => router.push(`/build/${agent.id}`)} className="flex w-full items-center gap-3 rounded-xl border border-border p-3 text-left hover:bg-muted"><Code2 className="h-4 w-4 text-[#2536e0]" /><span className="flex-1"><b className="block text-sm font-semibold">{t("workspace.builderSettings")}</b><small className="text-xs text-muted-foreground">{t("workspace.builderHint")}</small></span><ArrowLeft className="h-4 w-4 rotate-180 text-muted-foreground" /></button>
        </AgentProfileDialog>
      )}

      {deployOpen && (
        <DeployAgentDialog
          deployAgent={agent}
          onClose={() => setDeployOpen(false)}
          onUpdate={(updated) => setAgent((current) => ({ ...current, ...updated }))}
          onManageApiKey={() => router.push(`/api-keys?agent=${agent.id}`)}
          onManageTriggers={() => { setDeployOpen(false); setRoutinesOpen(true) }}
        />
      )}
      <AgentTriggersDialog agentId={agent.id} agentName={agent.name} open={routinesOpen} onOpenChange={setRoutinesOpen} />
    </>
  )
}

export function AgentChatHeader(props: AgentChatHeaderProps) {
  return <AgentChatHeaderContent key={props.agentId} {...props} />
}
