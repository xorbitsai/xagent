"use client"

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useRouter } from "next/navigation"
import {
  AlertCircle,
  CalendarClock,
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Info,
  Loader2,
  Play,
  Plus,
  RefreshCcw,
  RotateCcw,
  Trash2,
  Webhook,
  Zap,
} from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { toast } from "@/components/ui/sonner"
import { useI18n } from "@/contexts/i18n-context"
import {
  AgentTrigger,
  AgentTriggerRun,
  AgentTriggerType,
  createAgentTrigger,
  deleteAgentTrigger,
  listAgentTriggerRuns,
  listAgentTriggers,
  testAgentTrigger,
  updateAgentTrigger,
} from "@/lib/agent-triggers-api"
import { copyToClipboard } from "@/lib/clipboard"
import { cn, getApiUrl } from "@/lib/utils"

interface AgentTriggersDialogProps {
  agentId: number | null
  agentName?: string
  open: boolean
  onOpenChange: (open: boolean) => void
  onChanged?: () => void
  initialType?: AgentTriggerType | null
}

interface TriggerFormState {
  type: AgentTriggerType
  name: string
  enabled: boolean
  intervalSeconds: string
  nextRunAt: string
  secret: string
  promptTemplate: string
}

const TRIGGER_TYPES: AgentTriggerType[] = ["webhook", "scheduled"]
const DEFAULT_TEST_PAYLOAD = "{\n  \"message\": \"test trigger\"\n}"

function emptyForm(type: AgentTriggerType = "webhook"): TriggerFormState {
  return {
    type,
    name: "",
    enabled: true,
    intervalSeconds: "3600",
    nextRunAt: "",
    secret: "",
    promptTemplate: "",
  }
}

function defaultConfigForType(type: AgentTriggerType): Record<string, unknown> {
  return type === "scheduled" ? { interval_seconds: 3600 } : {}
}

function formatDateTime(value: string | null): string {
  if (!value) return "-"
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

function toDateTimeLocal(value: string | null): string {
  if (!value) return ""
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ""
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60_000)
  return local.toISOString().slice(0, 16)
}

function configNumber(config: Record<string, unknown>, key: string): string {
  const value = config[key]
  return typeof value === "number" || typeof value === "string" ? String(value) : ""
}

function formFromTrigger(trigger: AgentTrigger): TriggerFormState {
  return {
    type: trigger.type,
    name: trigger.name,
    enabled: trigger.enabled,
    intervalSeconds:
      trigger.type === "scheduled"
        ? configNumber(trigger.config, "interval_seconds") || "3600"
        : "3600",
    nextRunAt:
      trigger.type === "scheduled" && typeof trigger.config.next_run_at === "string"
        ? toDateTimeLocal(trigger.config.next_run_at)
        : "",
    secret: "",
    promptTemplate: trigger.prompt_template ?? "",
  }
}

function webhookUrl(trigger: AgentTrigger | null): string {
  if (!trigger?.webhook_token) return ""
  return `${getApiUrl()}/api/triggers/webhook/${trigger.webhook_token}`
}

function runStatusClass(status: string): string {
  if (status === "completed") return "text-emerald-600"
  if (status === "failed") return "text-destructive"
  if (status === "running") return "text-blue-600"
  return "text-muted-foreground"
}

function newestFirst(a: AgentTrigger, b: AgentTrigger): number {
  return b.id - a.id
}

function isValidAgentId(agentId: number | null): agentId is number {
  return typeof agentId === "number" && Number.isFinite(agentId)
}

export function AgentTriggersDialog({
  agentId,
  agentName,
  open,
  onOpenChange,
  onChanged,
  initialType = null,
}: AgentTriggersDialogProps) {
  const { t } = useI18n()
  const router = useRouter()
  const [triggers, setTriggers] = useState<AgentTrigger[]>([])
  const [activeType, setActiveTypeState] = useState<AgentTriggerType | null>(null)
  const [selectedTriggerId, setSelectedTriggerIdState] = useState<number | null>(null)
  const [runs, setRuns] = useState<AgentTriggerRun[]>([])
  const [loading, setLoading] = useState(false)
  const [runsLoading, setRunsLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [busyType, setBusyType] = useState<AgentTriggerType | null>(null)
  const [form, setForm] = useState<TriggerFormState>(emptyForm)
  const [testPayload, setTestPayload] = useState(DEFAULT_TEST_PAYLOAD)
  const [sourceEventId, setSourceEventId] = useState("")
  const [secretReveal, setSecretReveal] = useState<string | null>(null)
  const [copied, setCopied] = useState<string | null>(null)
  const [deleteConfirmId, setDeleteConfirmId] = useState<number | null>(null)
  const selectedTriggerIdRef = useRef<number | null>(null)
  const activeTypeRef = useRef<AgentTriggerType | null>(null)

  const setSelectedTriggerId = useCallback((id: number | null) => {
    selectedTriggerIdRef.current = id
    setSelectedTriggerIdState(id)
  }, [])

  const setActiveType = useCallback((type: AgentTriggerType | null) => {
    activeTypeRef.current = type
    setActiveTypeState(type)
  }, [])

  const triggerGroups = useMemo(() => {
    return TRIGGER_TYPES.reduce<Record<AgentTriggerType, AgentTrigger[]>>(
      (acc, type) => {
        acc[type] = triggers.filter((trigger) => trigger.type === type).sort(newestFirst)
        return acc
      },
      { webhook: [], scheduled: [] },
    )
  }, [triggers])

  const activeTypeTriggers = activeType ? triggerGroups[activeType] : []
  const selectedTrigger = useMemo(() => {
    if (!activeType) return null
    return (
      activeTypeTriggers.find((trigger) => trigger.id === selectedTriggerId) ??
      activeTypeTriggers[0] ??
      null
    )
  }, [activeType, activeTypeTriggers, selectedTriggerId])

  const selectedWebhookUrl = webhookUrl(selectedTrigger)

  const defaultNameForType = useCallback((type: AgentTriggerType) => {
    return type === "webhook"
      ? t("triggers.defaults.webhookName")
      : t("triggers.defaults.scheduledName")
  }, [t])

  const loadTriggers = useCallback(async (preferredTriggerId?: number | null) => {
    if (!isValidAgentId(agentId)) return
    setLoading(true)
    try {
      const data = await listAgentTriggers(agentId)
      setTriggers(data)

      const currentSelectedId = preferredTriggerId ?? selectedTriggerIdRef.current
      if (currentSelectedId && data.some((trigger) => trigger.id === currentSelectedId)) {
        setSelectedTriggerId(currentSelectedId)
      } else if (activeTypeRef.current) {
        const next = data
          .filter((trigger) => trigger.type === activeTypeRef.current)
          .sort(newestFirst)[0]
        setSelectedTriggerId(next?.id ?? null)
      }
    } catch (err) {
      console.error(err)
      toast.error(err instanceof Error ? err.message : t("triggers.messages.loadFailed"))
    } finally {
      setLoading(false)
    }
  }, [agentId, setSelectedTriggerId, t])

  const loadRuns = useCallback(async () => {
    if (!isValidAgentId(agentId) || !selectedTrigger) {
      setRuns([])
      return
    }
    setRunsLoading(true)
    try {
      setRuns(await listAgentTriggerRuns(agentId, selectedTrigger.id))
    } catch (err) {
      console.error(err)
      toast.error(err instanceof Error ? err.message : t("triggers.messages.runsLoadFailed"))
    } finally {
      setRunsLoading(false)
    }
  }, [agentId, selectedTrigger, t])

  useEffect(() => {
    if (!open) return
    setActiveType(initialType)
    setSelectedTriggerId(null)
    setSecretReveal(null)
    setCopied(null)
    setDeleteConfirmId(null)
    setRuns([])
    if (initialType) {
      setForm(emptyForm(initialType))
    }
    void loadTriggers(null)
  }, [initialType, loadTriggers, open, setActiveType, setSelectedTriggerId])

  useEffect(() => {
    if (!open || !activeType) return
    if (selectedTrigger) {
      setForm(formFromTrigger(selectedTrigger))
      void loadRuns()
    } else {
      setForm(emptyForm(activeType))
      setRuns([])
    }
  }, [activeType, loadRuns, open, selectedTrigger])

  const openType = (type: AgentTriggerType) => {
    const primary =
      triggerGroups[type].find((trigger) => trigger.enabled) ??
      triggerGroups[type][0] ??
      null
    setActiveType(type)
    setSelectedTriggerId(primary?.id ?? null)
    setSecretReveal(null)
    setDeleteConfirmId(null)
    setForm(primary ? formFromTrigger(primary) : emptyForm(type))
  }

  const beginCreateForType = (type: AgentTriggerType) => {
    setActiveType(type)
    setSelectedTriggerId(null)
    setSecretReveal(null)
    setDeleteConfirmId(null)
    setForm(emptyForm(type))
    setRuns([])
  }

  const beginEdit = (trigger: AgentTrigger) => {
    setActiveType(trigger.type)
    setSelectedTriggerId(trigger.id)
    setSecretReveal(null)
    setDeleteConfirmId(null)
    setForm(formFromTrigger(trigger))
  }

  const setFormValue = <K extends keyof TriggerFormState>(
    key: K,
    value: TriggerFormState[K],
  ) => {
    setForm((current) => ({ ...current, [key]: value }))
  }

  const buildConfig = (): Record<string, unknown> => {
    if (form.type === "webhook") return {}

    const config: Record<string, unknown> = {}
    const intervalValue = form.intervalSeconds.trim()
    if (intervalValue) {
      const interval = Number(intervalValue)
      if (!Number.isInteger(interval) || interval <= 0) {
        throw new Error(t("triggers.validation.interval"))
      }
      config.interval_seconds = interval
    }

    if (form.nextRunAt.trim()) {
      const next = new Date(form.nextRunAt)
      if (Number.isNaN(next.getTime())) {
        throw new Error(t("triggers.validation.nextRunAt"))
      }
      config.next_run_at = next.toISOString()
    }

    if (!config.interval_seconds && !config.next_run_at) {
      throw new Error(t("triggers.validation.scheduleRequired"))
    }
    return config
  }

  const buildPayload = () => {
    const name = form.name.trim() || defaultNameForType(form.type)
    if (name.length > 200) {
      throw new Error(t("triggers.validation.nameLength"))
    }

    return {
      type: form.type,
      name,
      enabled: form.enabled,
      config: buildConfig(),
      prompt_template: form.promptTemplate.trim() ? form.promptTemplate : null,
      secret: form.type === "webhook" && form.secret.trim() ? form.secret.trim() : null,
    }
  }

  const notifyChanged = () => {
    onChanged?.()
  }

  const createDefaultTrigger = async (type: AgentTriggerType, enabled: boolean) => {
    if (!isValidAgentId(agentId)) return null
    const saved = await createAgentTrigger(agentId, {
      type,
      name: defaultNameForType(type),
      enabled,
      config: defaultConfigForType(type),
      prompt_template: null,
      secret: null,
    })
    setSecretReveal(saved.webhook_secret ?? null)
    notifyChanged()
    return saved
  }

  const handleTypeToggle = async (type: AgentTriggerType, checked: boolean) => {
    if (!isValidAgentId(agentId)) return
    const typeTriggers = triggerGroups[type]
    const primary =
      typeTriggers.find((trigger) => trigger.enabled) ??
      typeTriggers[0] ??
      null
    setBusyType(type)
    try {
      let preferredId: number | null = primary?.id ?? null
      if (checked) {
        if (primary) {
          const updated = await updateAgentTrigger(agentId, primary.id, { enabled: true })
          preferredId = updated.id
        } else {
          const created = await createDefaultTrigger(type, true)
          preferredId = created?.id ?? null
          if (created) {
            setActiveType(type)
            setSelectedTriggerId(created.id)
            setForm(formFromTrigger(created))
          }
        }
      } else {
        const enabledTriggers = typeTriggers.filter((trigger) => trigger.enabled)
        await Promise.all(
          enabledTriggers.map((trigger) =>
            updateAgentTrigger(agentId, trigger.id, { enabled: false }),
          ),
        )
      }
      await loadTriggers(preferredId)
      notifyChanged()
      toast.success(checked ? t("triggers.messages.enabled") : t("triggers.messages.disabled"))
    } catch (err) {
      console.error(err)
      toast.error(err instanceof Error ? err.message : t("triggers.messages.saveFailed"))
    } finally {
      setBusyType(null)
    }
  }

  const handleSubmit = async () => {
    if (!isValidAgentId(agentId)) return
    let payload
    try {
      payload = buildPayload()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("triggers.messages.saveFailed"))
      return
    }

    setBusy(true)
    try {
      const saved = selectedTrigger
        ? await updateAgentTrigger(agentId, selectedTrigger.id, payload)
        : await createAgentTrigger(agentId, payload)
      setSecretReveal(saved.webhook_secret ?? null)
      setSelectedTriggerId(saved.id)
      setForm(formFromTrigger(saved))
      await loadTriggers(saved.id)
      notifyChanged()
      toast.success(selectedTrigger ? t("triggers.messages.updated") : t("triggers.messages.created"))
    } catch (err) {
      console.error(err)
      toast.error(err instanceof Error ? err.message : t("triggers.messages.saveFailed"))
    } finally {
      setBusy(false)
    }
  }

  const handleRotateSecret = async () => {
    if (!isValidAgentId(agentId) || !selectedTrigger || selectedTrigger.type !== "webhook") return
    setBusy(true)
    try {
      const updated = await updateAgentTrigger(agentId, selectedTrigger.id, {
        rotate_secret: true,
      })
      setSecretReveal(updated.webhook_secret ?? null)
      await loadTriggers(updated.id)
      notifyChanged()
      toast.success(t("triggers.messages.secretRotated"))
    } catch (err) {
      console.error(err)
      toast.error(err instanceof Error ? err.message : t("triggers.messages.secretRotateFailed"))
    } finally {
      setBusy(false)
    }
  }

  const handleDelete = async (trigger: AgentTrigger) => {
    if (!isValidAgentId(agentId)) return
    if (deleteConfirmId !== trigger.id) {
      setDeleteConfirmId(trigger.id)
      return
    }
    setBusy(true)
    try {
      await deleteAgentTrigger(agentId, trigger.id)
      setSelectedTriggerId(null)
      setRuns([])
      setSecretReveal(null)
      setDeleteConfirmId(null)
      await loadTriggers(null)
      notifyChanged()
      toast.success(t("triggers.messages.deleted"))
    } catch (err) {
      console.error(err)
      toast.error(err instanceof Error ? err.message : t("triggers.messages.deleteFailed"))
    } finally {
      setBusy(false)
    }
  }

  const handleTest = async () => {
    if (!isValidAgentId(agentId) || !selectedTrigger) return
    let payload: Record<string, unknown>
    try {
      const parsed = JSON.parse(testPayload || "{}")
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error(t("triggers.validation.testPayload"))
      }
      payload = parsed as Record<string, unknown>
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("triggers.validation.testPayload"))
      return
    }

    setBusy(true)
    try {
      const result = await testAgentTrigger(agentId, selectedTrigger.id, {
        payload,
        source_event_id: sourceEventId.trim() || null,
      })
      await loadRuns()
      toast.success(
        result.duplicate
          ? t("triggers.messages.testDuplicate")
          : t("triggers.messages.testStarted"),
      )
    } catch (err) {
      console.error(err)
      toast.error(err instanceof Error ? err.message : t("triggers.messages.testFailed"))
    } finally {
      setBusy(false)
    }
  }

  const handleCopy = async (id: string, value: string) => {
    if (!value) return
    if (await copyToClipboard(value)) {
      setCopied(id)
      toast.success(t("common.copied"))
      window.setTimeout(() => setCopied(null), 2000)
    } else {
      toast.error(t("triggers.messages.copyFailed"))
    }
  }

  const closeDialog = (nextOpen: boolean) => {
    if (!nextOpen) {
      setActiveType(null)
      setSecretReveal(null)
      setCopied(null)
      setDeleteConfirmId(null)
    }
    onOpenChange(nextOpen)
  }

  const renderTypeIcon = (type: AgentTriggerType, className?: string) => {
    return type === "webhook" ? (
      <Webhook className={className} />
    ) : (
      <CalendarClock className={className} />
    )
  }

  const renderTypeCard = (type: AgentTriggerType) => {
    const typeTriggers = triggerGroups[type]
    const enabledCount = typeTriggers.filter((trigger) => trigger.enabled).length
    const hasTriggers = typeTriggers.length > 0
    const isEnabled = enabledCount > 0
    const iconClass =
      type === "webhook"
        ? "bg-fuchsia-50 text-fuchsia-600 dark:bg-fuchsia-950/40 dark:text-fuchsia-300"
        : "bg-amber-50 text-amber-600 dark:bg-amber-950/40 dark:text-amber-300"

    return (
      <div
        key={type}
        className={cn(
          "group flex w-full items-center gap-4 rounded-lg border bg-background p-4 text-left transition-colors",
          "hover:border-primary/50 hover:bg-muted/30",
          isEnabled && "border-primary/40 bg-primary/[0.03]",
        )}
      >
        <button
          type="button"
          className="flex min-w-0 flex-1 items-center gap-4 text-left"
          onClick={() => openType(type)}
        >
          <div className={cn("flex h-10 w-10 shrink-0 items-center justify-center rounded-lg", iconClass)}>
            {renderTypeIcon(type, "h-5 w-5")}
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <div className="truncate text-sm font-semibold">{t(`triggers.cards.${type}.title`)}</div>
              {hasTriggers && (
                <Badge variant={isEnabled ? "default" : "secondary"} className="h-5 px-1.5 text-[10px]">
                  {isEnabled
                    ? t("triggers.cards.activeCount", { count: enabledCount })
                    : t("triggers.status.disabled")}
                </Badge>
              )}
            </div>
            <div className="mt-1 line-clamp-2 text-xs text-muted-foreground">
              {t(`triggers.cards.${type}.description`)}
            </div>
          </div>
        </button>
        <div className="flex items-center gap-3">
          <Switch
            checked={isEnabled}
            disabled={busyType === type || !isValidAgentId(agentId)}
            onCheckedChange={(checked) => void handleTypeToggle(type, checked)}
          />
          <button
            type="button"
            className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            onClick={() => openType(type)}
            aria-label={t(`triggers.cards.${type}.title`)}
          >
            <ChevronRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
          </button>
        </div>
      </div>
    )
  }

  const renderOverview = () => (
    <div className="space-y-4">
      <Alert className="border-primary/20 bg-primary/5 text-primary">
        <Info className="h-4 w-4" />
        <AlertDescription className="text-sm text-foreground">
          {t("triggers.overview.info")}
        </AlertDescription>
      </Alert>

      <div className="space-y-3">
        {loading ? (
          <div className="flex items-center justify-center rounded-lg border border-dashed py-16 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
        ) : (
          TRIGGER_TYPES.map(renderTypeCard)
        )}
      </div>
    </div>
  )

  const renderTriggerPicker = () => {
    if (!activeType || activeTypeTriggers.length === 0) return null
    return (
      <div className="flex flex-wrap items-center gap-2 border-b pb-4">
        {activeTypeTriggers.map((trigger) => (
          <button
            key={trigger.id}
            type="button"
            className={cn(
              "inline-flex max-w-full items-center gap-2 rounded-md border px-3 py-2 text-sm transition-colors",
              selectedTrigger?.id === trigger.id
                ? "border-primary bg-primary/5 text-foreground"
                : "bg-background text-muted-foreground hover:text-foreground",
            )}
            onClick={() => beginEdit(trigger)}
          >
            <span className={cn("h-2 w-2 rounded-full", trigger.enabled ? "bg-emerald-500" : "bg-muted-foreground/40")} />
            <span className="truncate">{trigger.name}</span>
          </button>
        ))}
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => beginCreateForType(activeType)}
        >
          <Plus className="mr-1.5 h-4 w-4" />
          {t("triggers.actions.addAnother")}
        </Button>
      </div>
    )
  }

  const renderDetail = () => {
    if (!activeType) return null
    const isNew = !selectedTrigger

    return (
      <div className="space-y-5">
        <div className="flex items-center justify-between gap-3 border-b pb-4">
          <div className="flex min-w-0 items-center gap-3">
            <Button variant="ghost" size="sm" className="-ml-2" onClick={() => setActiveType(null)}>
              <ChevronLeft className="mr-1 h-4 w-4" />
              {t("common.back")}
            </Button>
            <div className="flex min-w-0 items-center gap-2 border-l pl-3">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-muted">
                {renderTypeIcon(activeType, "h-4 w-4")}
              </div>
              <div className="truncate text-sm font-semibold">
                {t(`triggers.cards.${activeType}.title`)}
              </div>
            </div>
          </div>
          <Switch
            checked={form.enabled}
            onCheckedChange={(checked) => setFormValue("enabled", checked)}
            disabled={busy}
          />
        </div>

        {renderTriggerPicker()}

        <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_220px]">
          <div className="space-y-2">
            <Label htmlFor="trigger-name">{t("triggers.form.name")}</Label>
            <Input
              id="trigger-name"
              value={form.name}
              maxLength={200}
              onChange={(event) => setFormValue("name", event.target.value)}
              placeholder={defaultNameForType(activeType)}
            />
          </div>

          {activeType === "scheduled" ? (
            <div className="space-y-2">
              <Label htmlFor="trigger-interval">{t("triggers.form.intervalSeconds")}</Label>
              <Input
                id="trigger-interval"
                type="number"
                min={1}
                value={form.intervalSeconds}
                onChange={(event) => setFormValue("intervalSeconds", event.target.value)}
              />
            </div>
          ) : (
            <div className="space-y-2">
              <Label htmlFor="trigger-secret">{t("triggers.form.secret")}</Label>
              <Input
                id="trigger-secret"
                type="password"
                value={form.secret}
                onChange={(event) => setFormValue("secret", event.target.value)}
                placeholder={isNew ? t("triggers.form.secretPlaceholder") : t("triggers.form.secretEditPlaceholder")}
              />
            </div>
          )}
        </div>

        {activeType === "scheduled" && (
          <div className="space-y-2">
            <Label htmlFor="trigger-next-run">{t("triggers.form.nextRunAt")}</Label>
            <Input
              id="trigger-next-run"
              type="datetime-local"
              value={form.nextRunAt}
              onChange={(event) => setFormValue("nextRunAt", event.target.value)}
            />
          </div>
        )}

        <div className="space-y-2">
          <Label htmlFor="trigger-prompt">{t("triggers.form.promptTemplate")}</Label>
          <Textarea
            id="trigger-prompt"
            value={form.promptTemplate}
            onChange={(event) => setFormValue("promptTemplate", event.target.value)}
            placeholder={t("triggers.form.promptPlaceholder")}
            className="min-h-[112px]"
          />
        </div>

        {secretReveal && (
          <Alert className="border-amber-300 bg-amber-50 text-amber-950 dark:bg-amber-950/30 dark:text-amber-100">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>{t("triggers.secret.title")}</AlertTitle>
            <AlertDescription>
              <div className="mt-2 flex gap-2">
                <code className="min-w-0 flex-1 break-all rounded bg-background/70 px-2 py-1.5 text-xs">
                  {secretReveal}
                </code>
                <Button
                  size="icon"
                  variant="secondary"
                  onClick={() => void handleCopy("secret", secretReveal)}
                  title={t("common.copy")}
                >
                  {copied === "secret" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
            </AlertDescription>
          </Alert>
        )}

        {selectedTrigger?.type === "webhook" && (
          <section className="space-y-3 rounded-lg border bg-muted/20 p-4">
            <div className="text-sm font-medium">{t("triggers.webhook.title")}</div>
            <div className="flex gap-2">
              <Input readOnly value={selectedWebhookUrl} className="font-mono text-xs" />
              <Button
                variant="secondary"
                onClick={() => void handleCopy("webhook-url", selectedWebhookUrl)}
              >
                {copied === "webhook-url" ? <Check className="mr-2 h-4 w-4" /> : <Copy className="mr-2 h-4 w-4" />}
                {t("common.copy")}
              </Button>
            </div>
            <div className="text-xs text-muted-foreground">
              {t("triggers.webhook.secretHeader")}
            </div>
          </section>
        )}

        <div className="flex flex-wrap items-center justify-between gap-2 border-t pt-4">
          <div className="flex flex-wrap gap-2">
            <Button onClick={handleSubmit} disabled={busy || !isValidAgentId(agentId)}>
              {busy && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {isNew ? t("triggers.actions.enable") : t("triggers.actions.save")}
            </Button>
            {selectedTrigger && (
              <Button variant="outline" onClick={handleTest} disabled={busy}>
                <Play className="mr-2 h-4 w-4" />
                {t("triggers.actions.test")}
              </Button>
            )}
            {selectedTrigger?.type === "webhook" && (
              <Button variant="outline" onClick={handleRotateSecret} disabled={busy}>
                <RotateCcw className="mr-2 h-4 w-4" />
                {t("triggers.actions.rotateSecret")}
              </Button>
            )}
          </div>
          {selectedTrigger && (
            <Button
              variant={deleteConfirmId === selectedTrigger.id ? "destructive" : "ghost"}
              className={cn(deleteConfirmId !== selectedTrigger.id && "text-destructive hover:text-destructive")}
              onClick={() => void handleDelete(selectedTrigger)}
              disabled={busy}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              {deleteConfirmId === selectedTrigger.id
                ? t("triggers.actions.confirmDelete")
                : t("triggers.actions.delete")}
            </Button>
          )}
        </div>

        {selectedTrigger && (
          <section className="space-y-3 rounded-lg border p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-medium">{t("triggers.runs.title")}</h3>
                <p className="text-xs text-muted-foreground">
                  {selectedTrigger.next_run_at
                    ? `${t("triggers.runs.nextRun")}: ${formatDateTime(selectedTrigger.next_run_at)}`
                    : t("triggers.runs.noNextRun")}
                </p>
              </div>
              <Button variant="outline" size="sm" onClick={() => void loadRuns()} disabled={runsLoading}>
                <RefreshCcw className={cn("mr-2 h-4 w-4", runsLoading && "animate-spin")} />
                {t("common.refresh")}
              </Button>
            </div>

            <div className="space-y-2">
              {runsLoading ? (
                <div className="py-6 text-center text-sm text-muted-foreground">{t("common.loading")}</div>
              ) : runs.length === 0 ? (
                <div className="py-6 text-center text-sm text-muted-foreground">{t("triggers.runs.empty")}</div>
              ) : (
                runs.slice(0, 5).map((run) => (
                  <div key={run.id} className="flex items-center gap-3 rounded-md bg-muted/40 px-3 py-2 text-sm">
                    <span className={cn("font-medium", runStatusClass(run.status))}>
                      {t(`triggers.runStatus.${run.status}`)}
                    </span>
                    <span className="min-w-0 flex-1 truncate text-muted-foreground">
                      {run.source_event_id || run.idempotency_key}
                    </span>
                    {run.task_id ? (
                      <Button
                        variant="link"
                        className="h-auto p-0 text-xs"
                        onClick={() => router.push(`/task/${run.task_id}`)}
                      >
                        #{run.task_id}
                      </Button>
                    ) : (
                      <span className="text-xs text-muted-foreground">-</span>
                    )}
                  </div>
                ))
              )}
            </div>
          </section>
        )}

        {selectedTrigger && (
          <section className="space-y-3 rounded-lg border p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-medium">{t("triggers.test.title")}</h3>
                <p className="text-xs text-muted-foreground">{t("triggers.test.subtitle")}</p>
              </div>
            </div>
            <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_200px]">
              <Textarea
                value={testPayload}
                onChange={(event) => setTestPayload(event.target.value)}
                className="min-h-[112px] font-mono text-xs"
              />
              <div className="space-y-2">
                <Label htmlFor="trigger-source-event">{t("triggers.test.sourceEventId")}</Label>
                <Input
                  id="trigger-source-event"
                  value={sourceEventId}
                  onChange={(event) => setSourceEventId(event.target.value)}
                  placeholder={t("triggers.test.sourceEventPlaceholder")}
                />
              </div>
            </div>
          </section>
        )}
      </div>
    )
  }

  return (
    <Dialog open={open} onOpenChange={closeDialog}>
      <DialogContent
        aria-describedby="agent-triggers-dialog-description"
        className="flex max-h-[88vh] w-[calc(100vw-2rem)] max-w-none flex-col overflow-hidden p-0 sm:max-w-[680px]"
      >
        <DialogHeader className="border-b px-5 py-4 pr-12">
          <DialogTitle className="flex items-center gap-2 text-base">
            <Zap className="h-4 w-4 text-primary" />
            {t("triggers.title")}
          </DialogTitle>
          <DialogDescription id="agent-triggers-dialog-description">
            {agentName ? `${agentName} · ${t("triggers.subtitle")}` : t("triggers.subtitle")}
          </DialogDescription>
        </DialogHeader>

        <div className="min-h-0 flex-1 overflow-y-auto p-5">
          {activeType ? renderDetail() : renderOverview()}
        </div>
      </DialogContent>
    </Dialog>
  )
}
