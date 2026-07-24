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
  Mail,
  Play,
  Plus,
  RefreshCcw,
  RotateCcw,
  Webhook,
  X,
  Zap,
} from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Select } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { toast } from "@/components/ui/sonner"
import { useI18n } from "@/contexts/i18n-context"
import {
  AgentTrigger,
  AgentTriggerRun,
  AgentTriggerType,
  GmailAccount,
  StagedTrigger,
  TriggerOwnerRef,
  createOwnerTrigger,
  deleteOwnerTrigger,
  disableOwnerTriggersOfType,
  listGmailAccounts,
  listOwnerTriggerRuns,
  listOwnerTriggers,
  mergeUpdatedTriggers,
  stagedToPseudoTrigger,
  testOwnerTrigger,
  updateOwnerTrigger,
} from "@/lib/agent-triggers-api"
import { copyToClipboard } from "@/lib/clipboard"
import { cn, getApiUrl } from "@/lib/utils"

interface GmailConnectionState {
  isConnected: boolean
  connectedAccount?: string | null
}

interface AgentTriggersDialogProps {
  agentId: number | null
  agentName?: string
  // Workforce triggers (#950): when set, all live CRUD targets this owner
  // instead of the agent. Takes precedence over agentId; agentId is still
  // used for the agent-creation staging flow (workforces never stage).
  owner?: TriggerOwnerRef | null
  open: boolean
  onOpenChange: (open: boolean) => void
  onChanged?: () => void
  initialType?: AgentTriggerType | null
  gmailConnection?: GmailConnectionState | null
  onConnectGmail?: () => void
  // Creation flow (#928): when the agent does not exist yet the parent owns a
  // list of staged triggers. All create/update/delete operations mutate that
  // list instead of calling the API; the builder posts the staged triggers
  // right after the agent is created.
  staged?: { triggers: StagedTrigger[]; onChange: (next: StagedTrigger[]) => void } | null
}

interface TriggerFormState {
  type: AgentTriggerType
  name: string
  enabled: boolean
  intervalSeconds: string
  nextRunAt: string
  secret: string
  promptTemplate: string
  watchLabel: string
  senderFilter: string
  subjectKeyword: string
  oauthAccountId: string
}

const TRIGGER_TYPES: AgentTriggerType[] = ["webhook", "scheduled", "gmail"]
// Field labels follow the design refresh: small, semibold, muted.
const FIELD_LABEL_CLASS = "text-xs font-semibold text-muted-foreground"
const DEFAULT_TEST_PAYLOAD = "{\n  \"message\": \"test trigger\"\n}"

// True when `form` has no pending edits beyond `enabled` — i.e. reversing a
// not-yet-submitted enable intent (the only way a fresh creation form gets
// marked dirty without the user touching any other field) leaves nothing to
// commit.
function formMatchesEmptyIgnoringEnabled(
  form: TriggerFormState,
  type: AgentTriggerType,
): boolean {
  const empty = emptyForm(type)
  return (Object.keys(empty) as Array<keyof TriggerFormState>).every(
    (key) => key === "enabled" || form[key] === empty[key],
  )
}

function emptyForm(type: AgentTriggerType = "webhook"): TriggerFormState {
  return {
    type,
    name: "",
    // New triggers start disabled so the detail switch matches the overview
    // switch (which is off while no trigger of the type exists). Quick-toggle
    // creation from the overview passes enabled=true explicitly.
    enabled: false,
    intervalSeconds: "3600",
    nextRunAt: "",
    secret: "",
    promptTemplate: "",
    watchLabel: "INBOX",
    senderFilter: "",
    subjectKeyword: "",
    oauthAccountId: "",
  }
}

function defaultConfigForType(type: AgentTriggerType): Record<string, unknown> {
  if (type === "scheduled") return { interval_seconds: 3600 }
  if (type === "gmail") return { watch_label: "INBOX" }
  return {}
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

function configString(config: Record<string, unknown>, key: string): string {
  const value = config[key]
  return typeof value === "string" ? value : ""
}

function configId(config: Record<string, unknown>, key: string): string {
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
    watchLabel:
      trigger.type === "gmail" ? configString(trigger.config, "watch_label") || "INBOX" : "INBOX",
    senderFilter: trigger.type === "gmail" ? configString(trigger.config, "sender_filter") : "",
    subjectKeyword: trigger.type === "gmail" ? configString(trigger.config, "subject_keyword") : "",
    oauthAccountId:
      trigger.type === "gmail" ? configId(trigger.config, "oauth_account_id") : "",
  }
}

function webhookUrl(trigger: AgentTrigger | null): string {
  if (!trigger?.callback_id) return ""
  return `${getApiUrl()}/api/triggers/callback/webhook/${trigger.callback_id}`
}

function runStatusClass(status: string): string {
  if (status === "completed") return "text-emerald-600"
  if (status === "failed") return "text-destructive"
  if (status === "running") return "text-blue-600"
  return "text-muted-foreground"
}

function newestFirst(a: AgentTrigger, b: AgentTrigger): number {
  // Compare by magnitude: real ids are positive and grow with creation order,
  // staged pseudo ids are negative and shrink (-1, -2, …). Plain b - a would
  // invert the order for staged triggers; the two id spaces never mix in one
  // list.
  return Math.abs(b.id) - Math.abs(a.id)
}

// After deleting the selected trigger, the next one to show — same type,
// newest first — or null if none remain. Shared by handleDelete's staged and
// live branches, which otherwise differ only in whether the list still needs
// stagedToPseudoTrigger mapping before this runs.
function pickNextAfterDelete(remaining: AgentTrigger[], type: AgentTriggerType): number | null {
  return remaining.filter((item) => item.type === type).sort(newestFirst)[0]?.id ?? null
}

function isValidAgentId(agentId: number | null): agentId is number {
  return typeof agentId === "number" && Number.isFinite(agentId)
}

export function AgentTriggersDialog({
  agentId,
  agentName,
  owner = null,
  open,
  onOpenChange,
  onChanged,
  initialType = null,
  gmailConnection = null,
  onConnectGmail,
  staged = null,
}: AgentTriggersDialogProps) {
  const { t } = useI18n()
  const router = useRouter()
  const [liveTriggers, setLiveTriggers] = useState<AgentTrigger[]>([])
  const [activeType, setActiveTypeState] = useState<AgentTriggerType | null>(null)
  const [selectedTriggerId, setSelectedTriggerIdState] = useState<number | null>(null)
  const [runs, setRuns] = useState<AgentTriggerRun[]>([])
  const [loading, setLoading] = useState(false)
  const [runsLoading, setRunsLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  // A Set, not a scalar: two overview switches toggled back-to-back must each
  // keep their own guard, or one PATCH resolving would re-enable the other
  // type's still-in-flight switch (matches agent-builder.tsx's summary cards).
  const [busyTypes, setBusyTypes] = useState<ReadonlySet<AgentTriggerType>>(new Set())
  const [form, setForm] = useState<TriggerFormState>(emptyForm)
  // True when the form holds field edits that have not been persisted yet.
  // There is no explicit Save button: pending edits are committed on Done,
  // Back, and when switching to another trigger. The enabled switch persists
  // itself immediately and never marks the form dirty.
  const [formDirty, setFormDirty] = useState(false)
  const [testPayload, setTestPayload] = useState(DEFAULT_TEST_PAYLOAD)
  const [sourceEventId, setSourceEventId] = useState("")
  const [secretReveal, setSecretReveal] = useState<string | null>(null)
  const [copied, setCopied] = useState<string | null>(null)
  const [deleteConfirmId, setDeleteConfirmId] = useState<number | null>(null)
  const [gmailAccounts, setGmailAccounts] = useState<GmailAccount[] | null>(null)
  const [gmailAccountsLoading, setGmailAccountsLoading] = useState(false)
  const selectedTriggerIdRef = useRef<number | null>(null)
  const activeTypeRef = useRef<AgentTriggerType | null>(null)
  // Identity of the trigger the form was last synced from ("type:id", or
  // "type:new" for a creation form). The form-sync effect only resyncs when
  // this changes, so list refreshes that merely replace the selected
  // trigger's object identity never wipe unsaved field edits. Navigation
  // functions that set the form explicitly stamp the key themselves.
  const syncedFormKeyRef = useRef<string | null>(null)
  const formKeyFor = (type: AgentTriggerType, triggerId: number | null) =>
    `${type}:${triggerId ?? "new"}`

  const stagedTriggersProp = staged?.triggers ?? null
  const isStaging = !isValidAgentId(agentId) && stagedTriggersProp !== null
  // The live-CRUD target. Explicit owner (e.g. a workforce) wins; otherwise
  // fall back to the agent. Memoized on the owner's primitive fields (not the
  // `owner` object identity) so an inline object literal from the caller still
  // yields a referentially stable value safe for effect/callback deps.
  //
  // ASSUMPTION: callers pass owner by value (kind + id), never relying on a
  // stable object reference. If a future caller memoizes `owner` and expects
  // identity-based change detection, revisit these deps — an owner whose
  // primitives are unchanged but whose reference changed will NOT re-run this.
  const resolvedOwner = useMemo<TriggerOwnerRef | null>(() => {
    if (owner) return owner
    if (isValidAgentId(agentId)) return { kind: "agent", id: agentId }
    return null
  }, [owner?.kind, owner?.id, agentId]) // eslint-disable-line react-hooks/exhaustive-deps
  const canOperate = resolvedOwner !== null || isStaging

  // Ref mirror so the dialog-open effect can pick a default selection without
  // re-running whenever the staged list changes.
  const stagedTriggersRef = useRef<StagedTrigger[] | null>(null)
  stagedTriggersRef.current = stagedTriggersProp

  // Staged clientIds are stable negative numbers so they can serve as pseudo
  // AgentTrigger ids without ever colliding with a real server id.
  const nextStagedClientId = () =>
    (stagedTriggersProp ?? []).reduce((min, item) => Math.min(min, item.clientId), 0) - 1

  // Staging mode: derive the trigger list from the parent-owned staged
  // triggers so grouping/selection/form logic works unchanged. Derivation (not
  // a state mirror) keeps `triggers` in sync within the same render — a
  // useEffect mirror lags one render behind, which briefly resolved
  // selectedTrigger to null after a save and reset the form.
  const triggers = useMemo(
    () =>
      isStaging && stagedTriggersProp
        ? stagedTriggersProp.map(stagedToPseudoTrigger)
        : liveTriggers,
    [isStaging, stagedTriggersProp, liveTriggers],
  )

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
      { webhook: [], scheduled: [], gmail: [] },
    )
  }, [triggers])

  const activeTypeTriggers = activeType ? triggerGroups[activeType] : []
  // A null selectedTriggerId means "creating a new trigger" (e.g. via the Add
  // button), so it must NOT fall back to an existing trigger — that would make
  // handleSubmit overwrite it. Every browse flow selects an id explicitly
  // (openType, beginEdit, loadTriggers, the open effect).
  const selectedTrigger = useMemo(() => {
    if (!activeType || selectedTriggerId === null) return null
    return activeTypeTriggers.find((trigger) => trigger.id === selectedTriggerId) ?? null
  }, [activeType, activeTypeTriggers, selectedTriggerId])

  const selectedWebhookUrl = webhookUrl(selectedTrigger)

  const defaultNameForType = useCallback((type: AgentTriggerType) => {
    if (type === "webhook") return t("triggers.defaults.webhookName")
    if (type === "gmail") return t("triggers.defaults.gmailName")
    return t("triggers.defaults.scheduledName")
  }, [t])

  const loadTriggers = useCallback(async (preferredTriggerId?: number | null) => {
    if (!resolvedOwner) return
    setLoading(true)
    try {
      const data = await listOwnerTriggers(resolvedOwner)
      setLiveTriggers(data)

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
  }, [resolvedOwner, setSelectedTriggerId, t])

  // Takes the target trigger explicitly so navigation handlers can load runs
  // for a selection they just made (before the state round-trips).
  const loadRunsFor = useCallback(async (trigger: AgentTrigger | null) => {
    if (!resolvedOwner || !trigger || trigger.id < 0) {
      setRuns([])
      return
    }
    setRunsLoading(true)
    try {
      setRuns(await listOwnerTriggerRuns(resolvedOwner, trigger.id))
    } catch (err) {
      console.error(err)
      toast.error(err instanceof Error ? err.message : t("triggers.messages.runsLoadFailed"))
    } finally {
      setRunsLoading(false)
    }
  }, [resolvedOwner, t])

  const loadRuns = useCallback(
    () => loadRunsFor(selectedTrigger),
    [loadRunsFor, selectedTrigger],
  )

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
      // Live mode gets its default selection from loadTriggers below; in
      // staging mode it early-returns, so pick the primary staged trigger of
      // the requested type here (selectedTrigger no longer falls back to the
      // first trigger when nothing is selected).
      if (isStaging) {
        const typeStaged = (stagedTriggersRef.current ?? []).filter(
          (item) => item.type === initialType,
        )
        const primary = typeStaged.find((item) => item.enabled) ?? typeStaged[0]
        if (primary) setSelectedTriggerId(primary.clientId)
      }
    }
    void loadTriggers(null)
  }, [initialType, isStaging, loadTriggers, open, setActiveType, setSelectedTriggerId])

  useEffect(() => {
    if (!open) return
    let cancelled = false
    setGmailAccountsLoading(true)
    listGmailAccounts()
      .then((accounts) => {
        if (!cancelled) setGmailAccounts(accounts)
      })
      .catch((err) => {
        console.error(err)
        if (!cancelled) setGmailAccounts([])
      })
      .finally(() => {
        if (!cancelled) setGmailAccountsLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [open])

  // With exactly one connected account, bind new Gmail triggers to it. With
  // several accounts the user must pick explicitly so the wrong mailbox is
  // never chosen silently.
  useEffect(() => {
    if (!open || activeType !== "gmail" || selectedTrigger) return
    if (gmailAccounts?.length === 1) {
      const onlyAccountId = String(gmailAccounts[0].id)
      setForm((current) =>
        current.oauthAccountId ? current : { ...current, oauthAccountId: onlyAccountId },
      )
    }
  }, [activeType, gmailAccounts, open, selectedTrigger])

  useEffect(() => {
    if (!open || !activeType) {
      syncedFormKeyRef.current = null
      return
    }
    const key = formKeyFor(activeType, selectedTrigger?.id ?? null)
    if (syncedFormKeyRef.current === key) return
    syncedFormKeyRef.current = key
    if (selectedTrigger) {
      setForm(formFromTrigger(selectedTrigger))
      setFormDirty(false)
      void loadRunsFor(selectedTrigger)
    } else {
      setForm(emptyForm(activeType))
      setFormDirty(false)
      setRuns([])
    }
  }, [activeType, loadRunsFor, open, selectedTrigger])

  // Navigation helpers set the form synchronously (no flash of stale values)
  // and stamp syncedFormKeyRef so the form-sync effect treats the target as
  // already synced. secretReveal deliberately survives navigation: it is a
  // one-time value the user must copy, so only closing the dialog or deleting
  // a trigger clears it.
  const openType = (type: AgentTriggerType) => {
    const primary =
      triggerGroups[type].find((trigger) => trigger.enabled) ??
      triggerGroups[type][0] ??
      null
    syncedFormKeyRef.current = formKeyFor(type, primary?.id ?? null)
    setActiveType(type)
    setSelectedTriggerId(primary?.id ?? null)
    setDeleteConfirmId(null)
    setForm(primary ? formFromTrigger(primary) : emptyForm(type))
    setFormDirty(false)
    if (primary) {
      void loadRunsFor(primary)
    } else {
      setRuns([])
    }
  }

  const beginCreateForType = (type: AgentTriggerType, initial?: Partial<TriggerFormState>) => {
    syncedFormKeyRef.current = formKeyFor(type, null)
    setActiveType(type)
    setSelectedTriggerId(null)
    setDeleteConfirmId(null)
    setForm({ ...emptyForm(type), ...initial })
    // Preset values carry real user intent (e.g. the Gmail quick toggle's
    // enabled=true); mark them dirty so Done/Back attempts the creation
    // instead of silently dropping them.
    setFormDirty(Boolean(initial))
    setRuns([])
  }

  const beginEdit = (trigger: AgentTrigger) => {
    syncedFormKeyRef.current = formKeyFor(trigger.type, trigger.id)
    setActiveType(trigger.type)
    setSelectedTriggerId(trigger.id)
    setDeleteConfirmId(null)
    setForm(formFromTrigger(trigger))
    setFormDirty(false)
    void loadRunsFor(trigger)
  }

  const setFormValue = <K extends keyof TriggerFormState>(
    key: K,
    value: TriggerFormState[K],
  ) => {
    setForm((current) => ({ ...current, [key]: value }))
    // The enabled switch persists immediately (handleDetailToggle); every
    // other field is a pending edit committed on Done/Back/selection change.
    if (key !== "enabled") setFormDirty(true)
  }

  const buildConfig = (): Record<string, unknown> => {
    if (form.type === "webhook") return {}

    if (form.type === "gmail") {
      const watchLabel = form.watchLabel.trim()
      if (!watchLabel) {
        throw new Error(t("triggers.validation.watchLabel"))
      }
      const accountId = Number(form.oauthAccountId)
      if (!form.oauthAccountId.trim() || !Number.isInteger(accountId)) {
        throw new Error(t("triggers.validation.gmailAccount"))
      }
      const config: Record<string, unknown> = {
        watch_label: watchLabel,
        oauth_account_id: accountId,
      }
      const senderFilter = form.senderFilter.trim()
      const subjectKeyword = form.subjectKeyword.trim()
      if (senderFilter) config.sender_filter = senderFilter
      if (subjectKeyword) config.subject_keyword = subjectKeyword
      return config
    }

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
    const config = defaultConfigForType(type)
    if (type === "gmail" && gmailAccounts?.length === 1) {
      config.oauth_account_id = gmailAccounts[0].id
    }
    if (isStaging && staged) {
      const stagedTrigger: StagedTrigger = {
        clientId: nextStagedClientId(),
        type,
        name: defaultNameForType(type),
        enabled,
        config,
        prompt_template: null,
        secret: null,
      }
      staged.onChange([...staged.triggers, stagedTrigger])
      notifyChanged()
      return stagedToPseudoTrigger(stagedTrigger)
    }
    if (!resolvedOwner) return null
    const saved = await createOwnerTrigger(resolvedOwner, {
      type,
      name: defaultNameForType(type),
      enabled,
      config,
      prompt_template: null,
      secret: null,
    })
    setSecretReveal(saved.webhook_secret ?? null)
    notifyChanged()
    return saved
  }

  const handleTypeToggle = async (type: AgentTriggerType, checked: boolean) => {
    if (!canOperate) return
    const typeTriggers = triggerGroups[type]
    const primary =
      typeTriggers.find((trigger) => trigger.enabled) ??
      typeTriggers[0] ??
      null
    // A quick toggle must never guess which mailbox to bind: without exactly
    // one connected Gmail account, open the form for an explicit choice.
    if (checked && type === "gmail" && !triggerGroups.gmail.length) {
      const accountCount = gmailAccounts?.length ?? 0
      if (accountCount !== 1) {
        // The user asked to enable; carry that intent into the creation form
        // as a dirty preset, so Done/Back attempts the creation and surfaces
        // the missing-account validation instead of silently dropping it.
        beginCreateForType("gmail", { enabled: true })
        toast.info(
          accountCount === 0
            ? t("triggers.gmail.notConnectedDescription")
            : t("triggers.validation.gmailAccount"),
        )
        return
      }
    }
    setBusyTypes((current) => new Set(current).add(type))
    try {
      // Toggling from the overview never navigates into the config view; it
      // only flips (or creates) the trigger and stays on the list.
      if (isStaging && staged) {
        if (checked) {
          if (primary) {
            staged.onChange(
              staged.triggers.map((item) =>
                item.clientId === primary.id ? { ...item, enabled: true } : item,
              ),
            )
          } else {
            await createDefaultTrigger(type, true)
          }
        } else {
          staged.onChange(
            staged.triggers.map((item) =>
              item.type === type ? { ...item, enabled: false } : item,
            ),
          )
        }
        notifyChanged()
        toast.success(checked ? t("triggers.messages.enabled") : t("triggers.messages.disabled"))
        return
      }
      if (!resolvedOwner) return
      if (checked) {
        if (primary) {
          const updated = await updateOwnerTrigger(resolvedOwner, primary.id, { enabled: true })
          setLiveTriggers((current) =>
            current.map((item) => (item.id === updated.id ? updated : item)),
          )
        } else {
          const created = await createDefaultTrigger(type, true)
          if (created) {
            setLiveTriggers((current) => [...current, created])
          }
        }
      } else {
        const updatedList = await disableOwnerTriggersOfType(resolvedOwner, typeTriggers, type)
        setLiveTriggers((current) => mergeUpdatedTriggers(current, updatedList))
      }
      notifyChanged()
      toast.success(checked ? t("triggers.messages.enabled") : t("triggers.messages.disabled"))
    } catch (err) {
      console.error(err)
      toast.error(err instanceof Error ? err.message : t("triggers.messages.saveFailed"))
      // A batch disable is not atomic: some triggers may already be disabled
      // server-side, so resync rather than trusting the local list.
      if (resolvedOwner) void loadTriggers(selectedTriggerIdRef.current)
    } finally {
      setBusyTypes((current) => {
        const next = new Set(current)
        next.delete(type)
        return next
      })
    }
  }

  // The enabled switch in the detail view applies immediately — no Save
  // needed. Toggling on in the creation state creates the trigger right away
  // from the current form values (delegated to handleSubmit so the two create
  // paths cannot drift), mirroring the overview quick toggle.
  const handleDetailToggle = async (checked: boolean) => {
    setFormValue("enabled", checked)
    if (!canOperate) return

    // Identity of the form this toggle started on. If the user navigates
    // away (Back/pill switch/Add) before an in-flight request settles, the
    // failure handlers below check this before touching `form` — otherwise a
    // late rejection would silently mutate whatever unrelated form is now on
    // screen.
    const formKeyAtStart = syncedFormKeyRef.current

    if (!selectedTrigger) {
      if (!checked) {
        // Reversing a not-yet-submitted enable intent (the Gmail quick
        // toggle's only way to mark a fresh creation form dirty). If nothing
        // else was edited, there is nothing left to commit — clear the flag
        // so Done/Back don't attempt a phantom, unvalidated create.
        if (activeType && formMatchesEmptyIgnoringEnabled(form, activeType)) {
          setFormDirty(false)
        }
        return
      }
      const result = await handleSubmit(true)
      if (!result.ok && syncedFormKeyRef.current === formKeyAtStart) {
        setForm((current) => ({ ...current, enabled: false }))
      }
      return
    }

    // Existing trigger: a minimal enabled-only update that leaves any other
    // unsaved field edits (and formDirty) untouched. The form-sync effect
    // keys on the trigger id, so the list patch below cannot wipe them.
    if (isStaging && staged) {
      staged.onChange(
        staged.triggers.map((item) =>
          item.clientId === selectedTrigger.id ? { ...item, enabled: checked } : item,
        ),
      )
      notifyChanged()
      toast.success(checked ? t("triggers.messages.enabled") : t("triggers.messages.disabled"))
      return
    }

    if (!resolvedOwner) return
    setBusy(true)
    try {
      const updated = await updateOwnerTrigger(resolvedOwner, selectedTrigger.id, { enabled: checked })
      // Patch from the response, not a hand-set `enabled` — a scheduled
      // trigger's next_run_at/last_run_at can change server-side on
      // enable/disable, matching the overview toggle's own patch.
      setLiveTriggers((current) =>
        current.map((item) => (item.id === updated.id ? updated : item)),
      )
      // Reconcile the switch itself from the response too, in case a future
      // backend rule ever returns an `enabled` that differs from what was
      // requested — only while still on the same form (see the guard above).
      if (syncedFormKeyRef.current === formKeyAtStart) {
        setForm((current) => ({ ...current, enabled: updated.enabled }))
      }
      notifyChanged()
      toast.success(checked ? t("triggers.messages.enabled") : t("triggers.messages.disabled"))
    } catch (err) {
      console.error(err)
      if (syncedFormKeyRef.current === formKeyAtStart) {
        setForm((current) => ({ ...current, enabled: !checked }))
      }
      toast.error(err instanceof Error ? err.message : t("triggers.messages.saveFailed"))
    } finally {
      setBusy(false)
    }
  }

  interface SubmitResult {
    ok: boolean
    // One-time webhook secret generated by this submit, if any. Callers that
    // close the dialog check it so the reveal alert is seen before closing.
    secret: string | null
  }

  // Persists the current form (update selected / create new). `enabled`
  // lets callers force the field on top of the form state (the detail switch
  // passes enabled=true when creating). Returns success plus any freshly
  // generated webhook secret so commit-on-navigation callers can stay put on
  // failure or on a pending secret reveal.
  const handleSubmit = async (enabled?: boolean): Promise<SubmitResult> => {
    if (!canOperate) return { ok: false, secret: null }
    let payload
    try {
      payload = { ...buildPayload(), ...(enabled !== undefined && { enabled }) }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("triggers.messages.saveFailed"))
      return { ok: false, secret: null }
    }

    if (isStaging && staged) {
      if (selectedTrigger) {
        staged.onChange(
          staged.triggers.map((item) =>
            item.clientId === selectedTrigger.id
              ? {
                  ...item,
                  name: payload.name,
                  enabled: payload.enabled,
                  config: payload.config,
                  prompt_template: payload.prompt_template,
                  // Like the live edit flow, a blank secret keeps the current one.
                  secret: payload.secret ?? item.secret,
                }
              : item,
          ),
        )
        setForm((current) => ({ ...current, enabled: payload.enabled }))
      } else {
        const clientId = nextStagedClientId()
        staged.onChange([
          ...staged.triggers,
          {
            clientId,
            type: payload.type,
            name: payload.name,
            enabled: payload.enabled,
            config: payload.config,
            prompt_template: payload.prompt_template,
            secret: payload.secret,
          },
        ])
        // No key stamp here: when the staged list round-trips through the
        // parent, the form-sync effect re-syncs from the normalized pseudo
        // trigger (default name etc.). The merge below keeps the switch
        // correct in the meantime.
        setSelectedTriggerId(clientId)
        setForm((current) => ({ ...current, enabled: payload.enabled }))
      }
      setFormDirty(false)
      notifyChanged()
      toast.success(t("triggers.messages.staged"))
      return { ok: true, secret: null }
    }

    if (!resolvedOwner) return { ok: false, secret: null }
    setBusy(true)
    try {
      const saved = selectedTrigger
        ? await updateOwnerTrigger(resolvedOwner, selectedTrigger.id, payload)
        : await createOwnerTrigger(resolvedOwner, payload)
      const revealedSecret = saved.webhook_secret ?? null
      setSecretReveal(revealedSecret)
      // Patch the local list from the response instead of refetching it; the
      // form is synced here, so stamp the key to keep the effect from
      // re-syncing (and re-fetching runs) for the same trigger.
      syncedFormKeyRef.current = formKeyFor(saved.type, saved.id)
      setSelectedTriggerId(saved.id)
      setForm(formFromTrigger(saved))
      setFormDirty(false)
      setLiveTriggers((current) =>
        selectedTrigger
          ? current.map((item) => (item.id === saved.id ? saved : item))
          : [...current, saved],
      )
      notifyChanged()
      toast.success(selectedTrigger ? t("triggers.messages.updated") : t("triggers.messages.created"))
      return { ok: true, secret: revealedSecret }
    } catch (err) {
      console.error(err)
      toast.error(err instanceof Error ? err.message : t("triggers.messages.saveFailed"))
      return { ok: false, secret: null }
    } finally {
      setBusy(false)
    }
  }

  // There is no Save button: pending field edits are committed when leaving
  // the form (Done, Back, switching pills, Add, or dismissing the dialog).
  // Callers keep the user on the current form when ok is false.
  const commitPendingEdits = async (): Promise<SubmitResult> => {
    if (!activeType || !formDirty || !canOperate) return { ok: true, secret: null }
    return handleSubmit()
  }

  // A pending secret blocks close whether it was just generated by this very
  // commit (`result.secret` — `secretReveal` itself is a stale closure value
  // mid-await here, since the setState that wrote it happened inside the same
  // call) or was already sitting in state from an earlier action, like the
  // detail switch's own create (a fresh closure per render sees that fine).
  // The alert's explicit dismiss (which clears secretReveal) is what lets a
  // later Done/dismissal actually close.
  const secretPending = (result: SubmitResult) => Boolean(result.secret || secretReveal)

  const handleDone = async () => {
    const result = await commitPendingEdits()
    if (!result.ok) return
    if (secretPending(result)) return
    closeDialog(false)
  }

  const handleBack = async () => {
    const result = await commitPendingEdits()
    if (!result.ok || secretPending(result)) return
    setActiveType(null)
  }

  const handleSelectTrigger = async (trigger: AgentTrigger) => {
    if (trigger.id === selectedTriggerId) return
    const result = await commitPendingEdits()
    if (!result.ok || secretPending(result)) return
    beginEdit(trigger)
  }

  const handleAddAnother = async (type: AgentTriggerType) => {
    const result = await commitPendingEdits()
    if (!result.ok || secretPending(result)) return
    beginCreateForType(type)
  }

  const handleRotateSecret = async () => {
    if (!resolvedOwner || !selectedTrigger || selectedTrigger.type !== "webhook") return
    setBusy(true)
    try {
      const updated = await updateOwnerTrigger(resolvedOwner, selectedTrigger.id, {
        rotate_secret: true,
      })
      setSecretReveal(updated.webhook_secret ?? null)
      setLiveTriggers((current) =>
        current.map((item) => (item.id === updated.id ? updated : item)),
      )
      notifyChanged()
      toast.success(t("triggers.messages.secretRotated"))
    } catch (err) {
      console.error(err)
      toast.error(err instanceof Error ? err.message : t("triggers.messages.secretRotateFailed"))
    } finally {
      setBusy(false)
    }
  }

  // Confirmation happens in the pill's popover; by the time this runs the
  // user has already clicked the destructive button there.
  const handleDelete = async (trigger: AgentTrigger) => {
    if (!canOperate) return
    if (isStaging && staged) {
      const remaining = staged.triggers.filter((item) => item.clientId !== trigger.id)
      staged.onChange(remaining)
      // Deleting a pill other than the selected one must not reset the form
      // being edited; when the selected one goes, fall back to the next
      // trigger of the type (newest first), mirroring the live flow.
      if (selectedTriggerIdRef.current === trigger.id) {
        setSelectedTriggerId(pickNextAfterDelete(remaining.map(stagedToPseudoTrigger), trigger.type))
        setRuns([])
      }
      setDeleteConfirmId(null)
      notifyChanged()
      toast.success(t("triggers.messages.deleted"))
      return
    }
    if (!resolvedOwner) return
    setBusy(true)
    try {
      await deleteOwnerTrigger(resolvedOwner, trigger.id)
      const remaining = liveTriggers.filter((item) => item.id !== trigger.id)
      setLiveTriggers(remaining)
      // Deleting a pill other than the selected one must not reset the form
      // being edited; when the selected one goes, fall back to the next
      // trigger of the type (newest first), like the staging branch.
      if (selectedTriggerIdRef.current === trigger.id) {
        setSelectedTriggerId(pickNextAfterDelete(remaining, trigger.type))
        setRuns([])
      }
      setSecretReveal(null)
      setDeleteConfirmId(null)
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
    if (!resolvedOwner || !selectedTrigger) return
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
      const result = await testOwnerTrigger(resolvedOwner, selectedTrigger.id, {
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

  // Dismissal (Esc, overlay click, header X) commits pending edits like every
  // other exit path. A failed commit is surfaced via its toast but does not
  // trap the user — dismissal still closes rather than forcing them to fix
  // an invalid form just to leave. A freshly generated webhook secret is the
  // one thing dismissal must not drop: unlike a form validation error, it
  // already exists server-side and is unrecoverable once unseen, so — like
  // handleDone — closing waits until it has been shown.
  const handleDismiss = (nextOpen: boolean) => {
    if (nextOpen) {
      onOpenChange(true)
      return
    }
    void (async () => {
      const result = await commitPendingEdits()
      if (secretPending(result)) return
      closeDialog(false)
    })()
  }

  const renderTypeIcon = (type: AgentTriggerType, className?: string) => {
    if (type === "webhook") return <Webhook className={className} />
    if (type === "gmail") return <Mail className={className} />
    return <CalendarClock className={className} />
  }

  const typeIconClass = (type: AgentTriggerType) =>
    type === "webhook"
      ? "bg-fuchsia-50 text-fuchsia-600 dark:bg-fuchsia-950/40 dark:text-fuchsia-300"
      : type === "gmail"
        ? "bg-rose-50 text-rose-600 dark:bg-rose-950/40 dark:text-rose-300"
        : "bg-amber-50 text-amber-600 dark:bg-amber-950/40 dark:text-amber-300"

  const renderTypeCard = (type: AgentTriggerType) => {
    const typeTriggers = triggerGroups[type]
    const enabledCount = typeTriggers.filter((trigger) => trigger.enabled).length
    const hasTriggers = typeTriggers.length > 0
    const isEnabled = enabledCount > 0

    return (
      <div
        key={type}
        className={cn(
          "group flex w-full items-center gap-3 rounded-[10px] border bg-background px-4 py-3 text-left transition-colors",
          "hover:border-primary/50",
          isEnabled && "border-primary/40",
        )}
      >
        <button
          type="button"
          className="flex min-w-0 flex-1 items-center gap-3 text-left"
          onClick={() => openType(type)}
        >
          <div className={cn("flex h-[34px] w-[34px] shrink-0 items-center justify-center rounded-lg", typeIconClass(type))}>
            {renderTypeIcon(type, "h-4 w-4")}
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <div className="truncate text-[13px] font-semibold">{t(`triggers.cards.${type}.title`)}</div>
              {hasTriggers && (
                <Badge variant={isEnabled ? "default" : "secondary"} className="h-5 px-1.5 text-[10px]">
                  {isEnabled
                    ? t("triggers.cards.activeCount", { count: enabledCount })
                    : t("triggers.status.disabled")}
                </Badge>
              )}
            </div>
            <div className="mt-0.5 line-clamp-2 text-xs text-muted-foreground">
              {t(`triggers.cards.${type}.description`)}
            </div>
          </div>
        </button>
        <div className="flex items-center gap-2.5">
          <Switch
            checked={isEnabled}
            disabled={busyTypes.has(type) || !canOperate}
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

  // One-time reveal for a freshly generated webhook secret. Rendered on both
  // the overview (quick-toggle creation stays there) and the detail view.
  // The explicit dismiss is what lets Done/Back/dismissal close the dialog
  // afterward — see handleDone/handleDismiss, which refuse to close while
  // secretReveal is still set so the secret is never lost unseen.
  const renderSecretReveal = () =>
    secretReveal && (
      <Alert className="relative border-amber-300 bg-amber-50 text-amber-950 dark:bg-amber-950/30 dark:text-amber-100">
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
        <button
          type="button"
          className="absolute right-3 top-3 rounded p-0.5 text-amber-700 hover:bg-amber-100 dark:text-amber-200 dark:hover:bg-amber-900/40"
          onClick={() => setSecretReveal(null)}
          aria-label={t("triggers.secret.dismiss")}
          title={t("triggers.secret.dismiss")}
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </Alert>
    )

  const renderOverview = () => (
    <div className="space-y-2.5">
      {renderSecretReveal()}
      {loading ? (
        <div className="flex items-center justify-center rounded-lg border border-dashed py-16 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
        </div>
      ) : (
        TRIGGER_TYPES.map(renderTypeCard)
      )}
    </div>
  )

  const renderTriggerPicker = () => {
    if (!activeType || activeTypeTriggers.length === 0) return null
    return (
      <div className="flex flex-wrap items-center gap-2 border-b pb-4">
        {activeTypeTriggers.map((trigger) => (
          <div
            key={trigger.id}
            className={cn(
              "inline-flex max-w-full items-center rounded-full border text-xs transition-colors",
              selectedTrigger?.id === trigger.id
                ? "border-primary bg-primary/10 font-medium text-primary"
                : "bg-background text-muted-foreground hover:border-primary/50 hover:text-foreground",
            )}
          >
            <button
              type="button"
              className="flex min-w-0 items-center gap-1.5 py-1.5 pl-3 pr-1"
              onClick={() => void handleSelectTrigger(trigger)}
              disabled={busy}
            >
              <span className={cn("h-2 w-2 rounded-full", trigger.enabled ? "bg-emerald-500" : "bg-muted-foreground/40")} />
              <span className="truncate">{trigger.name}</span>
            </button>
            <Popover
              open={deleteConfirmId === trigger.id}
              onOpenChange={(nextOpen) => setDeleteConfirmId(nextOpen ? trigger.id : null)}
            >
              <PopoverTrigger asChild>
                <button
                  type="button"
                  aria-label={t("triggers.actions.delete")}
                  title={t("triggers.actions.delete")}
                  className={cn(
                    "mr-1 rounded-full p-1 transition-colors",
                    deleteConfirmId === trigger.id
                      ? "bg-destructive/10 text-destructive"
                      : "text-muted-foreground/60 hover:bg-muted hover:text-destructive",
                  )}
                  disabled={busy}
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </PopoverTrigger>
              <PopoverContent align="end" className="w-auto max-w-64 p-3">
                <div className="text-sm">{t("triggers.deleteConfirm")}</div>
                <div className="mt-2.5 flex justify-end gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => setDeleteConfirmId(null)}
                  >
                    {t("common.cancel")}
                  </Button>
                  <Button
                    type="button"
                    variant="destructive"
                    size="sm"
                    disabled={busy}
                    onClick={() => void handleDelete(trigger)}
                  >
                    {t("triggers.actions.confirmDelete")}
                  </Button>
                </div>
              </PopoverContent>
            </Popover>
          </div>
        ))}
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => void handleAddAnother(activeType)}
          disabled={busy}
          className="h-7 rounded-full border-dashed border-primary/45 bg-primary/5 text-xs text-primary hover:border-primary hover:bg-primary/10"
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
    // Whether leaving the secret field blank keeps an existing secret: true
    // for live triggers (the server holds one) and for staged triggers that
    // stored a user-provided secret; a staged trigger without one gets a
    // generated secret when the agent is created.
    const blankSecretKeepsCurrent = isStaging
      ? Boolean(
          selectedTrigger &&
            stagedTriggersProp?.find((item) => item.clientId === selectedTrigger.id)?.secret,
        )
      : !isNew

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between gap-3 border-b pb-4">
          <div className="flex min-w-0 items-center gap-2.5">
            <Button
              variant="ghost"
              size="sm"
              className="-ml-2 h-8 px-2 text-muted-foreground hover:text-foreground"
              onClick={() => void handleBack()}
              disabled={busy}
            >
              <ChevronLeft className="mr-1 h-4 w-4" />
              {t("common.back")}
            </Button>
            <div className={cn("flex h-6 w-6 shrink-0 items-center justify-center rounded-md", typeIconClass(activeType))}>
              {renderTypeIcon(activeType, "h-3.5 w-3.5")}
            </div>
            <div className="truncate text-sm font-bold">
              {t(`triggers.cards.${activeType}.title`)}
            </div>
          </div>
          <Switch
            checked={form.enabled}
            onCheckedChange={(checked) => void handleDetailToggle(checked)}
            disabled={busy}
          />
        </div>

        {renderTriggerPicker()}

        <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_220px]">
          <div className="space-y-2">
            <Label className={FIELD_LABEL_CLASS} htmlFor="trigger-name">{t("triggers.form.name")}</Label>
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
              <Label className={FIELD_LABEL_CLASS} htmlFor="trigger-interval">{t("triggers.form.intervalSeconds")}</Label>
              <Input
                id="trigger-interval"
                type="number"
                min={1}
                value={form.intervalSeconds}
                onChange={(event) => setFormValue("intervalSeconds", event.target.value)}
              />
            </div>
          ) : activeType === "gmail" ? (
            <div className="space-y-2">
              <Label className={FIELD_LABEL_CLASS} htmlFor="trigger-watch-label">{t("triggers.form.watchLabel")}</Label>
              <Input
                id="trigger-watch-label"
                value={form.watchLabel}
                onChange={(event) => setFormValue("watchLabel", event.target.value)}
                placeholder={t("triggers.form.watchLabelPlaceholder")}
              />
              <p className="text-xs text-muted-foreground">{t("triggers.form.watchLabelHelp")}</p>
            </div>
          ) : (
            <div className="space-y-2">
              <Label className={FIELD_LABEL_CLASS} htmlFor="trigger-secret">{t("triggers.form.secret")}</Label>
              <Input
                id="trigger-secret"
                type="password"
                value={form.secret}
                onChange={(event) => setFormValue("secret", event.target.value)}
                placeholder={
                  blankSecretKeepsCurrent
                    ? t("triggers.form.secretEditPlaceholder")
                    : t("triggers.form.secretPlaceholder")
                }
              />
            </div>
          )}
        </div>

        {activeType === "scheduled" && (
          <div className="space-y-2">
            <Label className={FIELD_LABEL_CLASS} htmlFor="trigger-next-run">{t("triggers.form.nextRunAt")}</Label>
            <Input
              id="trigger-next-run"
              type="datetime-local"
              value={form.nextRunAt}
              onChange={(event) => setFormValue("nextRunAt", event.target.value)}
            />
          </div>
        )}

        {activeType === "gmail" && (
          <div className="space-y-2">
            <Label className={FIELD_LABEL_CLASS} id="trigger-gmail-account-label">{t("triggers.form.gmailAccount")}</Label>
            <div aria-labelledby="trigger-gmail-account-label">
              <Select
                value={form.oauthAccountId || undefined}
                onValueChange={(value) => setFormValue("oauthAccountId", value)}
                options={(gmailAccounts ?? []).map((account) => ({
                  value: String(account.id),
                  label: account.email || `#${account.id}`,
                }))}
                placeholder={
                  gmailAccountsLoading
                    ? t("common.loading")
                    : gmailAccounts && gmailAccounts.length === 0
                      ? t("triggers.gmail.noAccounts")
                      : t("triggers.form.gmailAccountPlaceholder")
                }
                disabled={gmailAccountsLoading || (gmailAccounts?.length ?? 0) === 0}
              />
            </div>
            {form.oauthAccountId &&
              gmailAccounts &&
              !gmailAccounts.some(
                (account) => String(account.id) === form.oauthAccountId,
              ) && (
                <p className="text-xs text-destructive">
                  {t("triggers.gmail.accountMissing")}
                </p>
              )}
            <p className="text-xs text-muted-foreground">
              {t("triggers.form.gmailAccountHelp")}
            </p>
          </div>
        )}

        {activeType === "gmail" && (
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label className={FIELD_LABEL_CLASS} htmlFor="trigger-sender-filter">{t("triggers.form.senderFilter")}</Label>
              <Input
                id="trigger-sender-filter"
                value={form.senderFilter}
                onChange={(event) => setFormValue("senderFilter", event.target.value)}
                placeholder={t("triggers.form.senderFilterPlaceholder")}
              />
            </div>
            <div className="space-y-2">
              <Label className={FIELD_LABEL_CLASS} htmlFor="trigger-subject-keyword">{t("triggers.form.subjectKeyword")}</Label>
              <Input
                id="trigger-subject-keyword"
                value={form.subjectKeyword}
                onChange={(event) => setFormValue("subjectKeyword", event.target.value)}
                placeholder={t("triggers.form.subjectKeywordPlaceholder")}
              />
            </div>
          </div>
        )}

        {activeType === "gmail" && (
          <Alert
            className={cn(
              gmailConnection?.isConnected
                ? "border-emerald-200 bg-emerald-50 text-emerald-950 dark:border-emerald-900/60 dark:bg-emerald-950/30 dark:text-emerald-100"
                : "border-amber-200 bg-amber-50 text-amber-950 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-100",
            )}
          >
            <Mail className="h-4 w-4" />
            <AlertTitle>
              {gmailConnection?.isConnected
                ? t("triggers.gmail.connected")
                : t("triggers.gmail.notConnected")}
            </AlertTitle>
            <AlertDescription>
              <div className="mt-1 flex flex-wrap items-center justify-between gap-3 text-sm">
                <span>
                  {gmailConnection?.isConnected
                    ? gmailConnection.connectedAccount || t("triggers.gmail.connectedDescription")
                    : t("triggers.gmail.notConnectedDescription")}
                </span>
                {!gmailConnection?.isConnected && onConnectGmail && (
                  <Button
                    type="button"
                    size="sm"
                    variant="secondary"
                    onClick={onConnectGmail}
                  >
                    {t("triggers.gmail.connect")}
                  </Button>
                )}
              </div>
            </AlertDescription>
          </Alert>
        )}

        <div className="space-y-2">
          <Label className={FIELD_LABEL_CLASS} htmlFor="trigger-prompt">{t("triggers.form.promptTemplate")}</Label>
          <Textarea
            id="trigger-prompt"
            value={form.promptTemplate}
            onChange={(event) => setFormValue("promptTemplate", event.target.value)}
            placeholder={t("triggers.form.promptPlaceholder")}
            className="min-h-[112px]"
          />
        </div>

        {renderSecretReveal()}

        {isStaging && activeType === "webhook" && (
          <Alert className="border-primary/20 bg-primary/5">
            <Info className="h-4 w-4" />
            <AlertDescription className="text-sm text-foreground">
              {t("triggers.staging.webhookPending")}
            </AlertDescription>
          </Alert>
        )}

        {!isStaging && selectedTrigger?.type === "webhook" && (
          <section className="space-y-1.5">
            <div className={FIELD_LABEL_CLASS}>{t("triggers.webhook.title")}</div>
            <div className="flex items-center gap-2 rounded-lg border bg-muted/50 px-3 py-2">
              <code className="min-w-0 flex-1 truncate font-mono text-xs text-muted-foreground">
                {selectedWebhookUrl}
              </code>
              <button
                type="button"
                className="shrink-0 rounded p-1 text-muted-foreground transition-colors hover:text-primary"
                onClick={() => void handleCopy("webhook-url", selectedWebhookUrl)}
                aria-label={t("common.copy")}
                title={t("common.copy")}
              >
                {copied === "webhook-url" ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
              </button>
            </div>
            <p className="text-[11px] leading-relaxed text-muted-foreground">
              {t("triggers.webhook.secretHeader")}
            </p>
          </section>
        )}

        {!isStaging && selectedTrigger && (
          <div className="flex flex-wrap items-center gap-2 border-t pt-4">
            <Button variant="outline" onClick={handleTest} disabled={busy}>
              <Play className="mr-2 h-4 w-4" />
              {t("triggers.actions.test")}
            </Button>
            {selectedTrigger.type === "webhook" && (
              <Button variant="outline" onClick={handleRotateSecret} disabled={busy}>
                <RotateCcw className="mr-2 h-4 w-4" />
                {t("triggers.actions.rotateSecret")}
              </Button>
            )}
          </div>
        )}

        {!isStaging && selectedTrigger && (
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

        {!isStaging && selectedTrigger && (
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
                <Label className={FIELD_LABEL_CLASS} htmlFor="trigger-source-event">{t("triggers.test.sourceEventId")}</Label>
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
    <Dialog open={open} onOpenChange={handleDismiss}>
      <DialogContent
        aria-describedby="agent-triggers-dialog-description"
        className="flex max-h-[88vh] w-[calc(100vw-2rem)] max-w-none flex-col overflow-hidden rounded-2xl p-0 sm:max-w-[560px]"
      >
        <DialogHeader className="border-b px-5 py-4 pr-12">
          <DialogTitle className="flex items-center gap-2 text-[15px] font-bold">
            <Zap className="h-4 w-4 text-primary" />
            {t("triggers.title")}
          </DialogTitle>
          <DialogDescription id="agent-triggers-dialog-description" className="text-xs">
            {agentName ? `${agentName} · ${t("triggers.subtitle")}` : t("triggers.subtitle")}
          </DialogDescription>
        </DialogHeader>

        <div className="min-h-0 flex-1 overflow-y-auto p-5 pt-4">
          <Alert className="mb-4 rounded-lg border-primary/25 bg-primary/[0.07] px-3.5 py-2.5 text-primary">
            <Info className="h-4 w-4" />
            <AlertDescription className="text-xs leading-relaxed text-muted-foreground">
              {t(isStaging ? "triggers.staging.info" : "triggers.overview.info")}
            </AlertDescription>
          </Alert>
          {activeType ? renderDetail() : renderOverview()}
        </div>

        <DialogFooter className="border-t px-5 py-3.5">
          <Button onClick={() => void handleDone()} disabled={busy}>
            {busy && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {t("common.done")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
