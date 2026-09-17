"use client"

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useRouter } from "next/navigation"
import {
  AlertCircle,
  CalendarCheck,
  CalendarClock,
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Info,
  Loader2,
  Mail,
  Pencil,
  Play,
  Plus,
  RefreshCcw,
  RotateCcw,
  Trash2,
  Wand2,
  Webhook,
  X,
  Zap,
} from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
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
import {
  RECURRENCE_TYPES,
  ScheduleFields,
  localIsoDate,
  localTimeOfDay,
  scheduleFieldsDefaults,
  summarizeSchedule,
  zonedIsoDate,
  zonedTimeOfDay,
  type ScheduleCustomUnit,
  type ScheduleFieldsValue,
  type ScheduleRecurrence,
} from "./agent-triggers-schedule-fields"

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

interface TriggerFormState extends ScheduleFieldsValue {
  type: AgentTriggerType
  name: string
  enabled: boolean
  secret: string
  promptTemplate: string
  watchLabel: string
  senderFilter: string
  subjectKeyword: string
  oauthAccountId: string
}

const TRIGGER_TYPES: AgentTriggerType[] = ["webhook", "scheduled", "gmail"]
// The plain "YYYY-MM-DD" shape buildConfig sends for daily/weekly/monthly's
// start_at — zone-agnostic on purpose (see buildConfig / scheduleFieldsFromConfig).
const CALENDAR_DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/
// Field labels follow the design refresh: small, semibold, muted.
const FIELD_LABEL_CLASS = "text-xs font-semibold text-muted-foreground"
// Sent as-is by the one-click "Test trigger" button; the backend stamps a
// random idempotency key for test runs, so every click starts a fresh run.
const TEST_RUN_PAYLOAD: Record<string, unknown> = { message: "test trigger" }

// Same palette as the reference design's Gmail account avatars; the account
// id picks a stable color.
const GMAIL_AVATAR_COLORS = [
  "hsl(5 75% 50%)",
  "hsl(217 91% 55%)",
  "hsl(142 60% 40%)",
  "hsl(280 70% 55%)",
  "hsl(38 90% 45%)",
]

function gmailAvatarColor(id: number | string): string {
  return GMAIL_AVATAR_COLORS[Math.abs(Number(id) || 0) % GMAIL_AVATAR_COLORS.length]
}

function gmailAvatarInitials(email: string | null, id: number | string): string {
  return (email || String(id)).slice(0, 2).toUpperCase()
}

// Client-side mirror of the backend's render_trigger_prompt (triggers.py),
// used to test STAGED triggers: the agent doesn't exist server-side yet, so
// the rendered prompt — exactly what a real firing would send — is run
// through the builder's live preview instead.
function renderStagedTestPrompt(
  type: AgentTriggerType,
  name: string,
  promptTemplate: string | null,
  payload: Record<string, unknown>,
): string {
  const payloadJson = JSON.stringify(payload, null, 2)
  const template = (promptTemplate ?? "").trim()
  if (template) {
    return template
      .replaceAll("{{payload}}", payloadJson)
      .replaceAll("{{trigger_type}}", type)
      .replaceAll("{{source_event_id}}", "")
      .replaceAll("{{test}}", "true")
  }
  return (
    `Handle this test ${type} trigger event.\n\n` +
    `Trigger: ${name}\n` +
    `Source event ID: none\n\n` +
    `Event payload:\n${payloadJson}`
  )
}

const ADD_ANOTHER_KEYS: Record<AgentTriggerType, string> = {
  webhook: "triggers.actions.addAnotherWebhook",
  scheduled: "triggers.actions.addAnotherSchedule",
  gmail: "triggers.actions.addAnotherGmail",
}

const EDITOR_TITLE_KEYS: Record<AgentTriggerType, { create: string; edit: string }> = {
  webhook: { create: "triggers.editor.webhookNew", edit: "triggers.editor.webhookEdit" },
  scheduled: { create: "triggers.editor.scheduledNew", edit: "triggers.editor.scheduledEdit" },
  gmail: { create: "triggers.editor.gmailNew", edit: "triggers.editor.gmailEdit" },
}

// The backend treats "*" and "all" as match-anything watch labels
// (gmail_triggers.py); the UI expresses that state as a blank field.
function displayWatchLabel(raw: string): string {
  const value = raw.trim()
  return value === "*" || value.toLowerCase() === "all" ? "" : value
}

// gmail_triggers.py resolves BOTH an absent `watch_label` key AND a
// present-but-empty string to "inbox" (`config.get("watch_label") or ""`,
// then `... or "inbox"` once stripped) — an empty string is exactly as
// falsy as a missing key. "*" / "all" are the only real wildcard sentinels.
// A config missing the key entirely is a legacy row from before this field
// existed; a present "" is presumably from the same era via some other
// write path. Either way the backend treats it as INBOX-only, so the editor
// must too — showing it as an indistinguishable blank field (this UI's
// wildcard display) would silently widen it to match everything the moment
// the user opens and saves it.
function gmailFormWatchLabel(config: Record<string, unknown>): string {
  const raw = configString(config, "watch_label")
  if (!("watch_label" in config) || !raw.trim()) return "INBOX"
  return displayWatchLabel(raw)
}

function emptyForm(type: AgentTriggerType = "webhook"): TriggerFormState {
  return {
    type,
    name: "",
    // Creation flows (type switch, empty-state CTA, "Add another") pass
    // enabled=true explicitly so a freshly saved trigger is live right away,
    // matching the reference design.
    enabled: false,
    secret: "",
    promptTemplate: "",
    // Blank = watch all incoming emails (saved as the "*" sentinel).
    watchLabel: "",
    senderFilter: "",
    subjectKeyword: "",
    oauthAccountId: "",
    ...scheduleFieldsDefaults(),
  }
}

function unitSecondsFor(unit: ScheduleCustomUnit): number {
  if (unit === "days") return 86400
  if (unit === "hours") return 3600
  return 60
}

/** Parses an hourly/custom schedule's anchor from the editor's startDate/
 * timeOfDay fields into a concrete instant, or null when there's nothing
 * parseable (blank date, or an unparsable combination). Only hourly/custom
 * use this — daily/weekly/monthly send a bare calendar date the backend
 * combines with its own timezone instead. Shared by scheduleWillFireImmediately
 * and buildConfig (PR #1051 review, N10 cleanup) — they used to each
 * independently recompute `new Date(...)` with the same NaN-guard; each call
 * site still decides what null MEANS for its own purposes (a silent
 * "don't warn" here, a thrown validation error in buildConfig). */
function parseHourlyCustomAnchor(startDate: string, timeOfDay: string): Date | null {
  if (!startDate) return null
  const anchor = new Date(`${startDate}T${timeOfDay || "00:00"}:00`)
  return Number.isNaN(anchor.getTime()) ? null : anchor
}

/** True when any field feeding into a schedule's actual fire-time computation
 * differs between two ScheduleFieldsValue snapshots. Used to tell a genuine
 * schedule edit apart from an unrelated field change (rename, prompt tweak) —
 * mirrors the backend's _schedule_signature comparison closely enough for
 * this UI-only warning's purposes (see scheduleWillFireImmediately); exact
 * backend normalization details don't need to be replicated bit-for-bit. */
function scheduleRelevantFieldsChanged(a: ScheduleFieldsValue, b: ScheduleFieldsValue): boolean {
  return (
    a.recurrence !== b.recurrence ||
    a.startDate !== b.startDate ||
    a.timeOfDay !== b.timeOfDay ||
    a.timezone !== b.timezone ||
    a.dayOfMonth !== b.dayOfMonth ||
    a.customAmount !== b.customAmount ||
    a.customUnit !== b.customUnit ||
    a.weekdays.length !== b.weekdays.length ||
    a.weekdays.some((day) => !b.weekdays.includes(day))
  )
}

/** True when an hourly/custom schedule's picked start instant has already
 * passed AND that instant is actually about to be (re)computed as a fresh
 * anchor. UI-only signal (PR #1051 review, N2): the backend's documented
 * cron semantics deliberately fire such a schedule on the very next scan
 * tick rather than skipping ahead — this just lets the editor warn about
 * that up front instead of surprising the user after Save. daily/weekly/
 * monthly never "catch up" this way (the backend always computes a future
 * occurrence for those), so this only applies to hourly/custom.
 *
 * `baseline` is the schedule fields as loaded from the trigger being edited
 * (null when creating a brand-new trigger, which always gets a fresh anchor
 * on save). Without checking it against a real edit, EVERY reopen of an
 * already-fired hourly/custom trigger would warn: scheduleFieldsFromConfig
 * derives startDate/timeOfDay from the trigger's stored config, which is
 * frozen at creation time and never rewritten by the scan loop (only the
 * next_run_at DB COLUMN is advanced) — so it reads as "in the past" on every
 * edit, even though _apply_trigger_updates's own "did the schedule actually
 * change" check (_schedule_signature) sees no diff and will NOT recompute
 * next_run_at on an unrelated-field Save (PR #1051 review, N2 follow-up). */
function scheduleWillFireImmediately(
  form: ScheduleFieldsValue,
  baseline: ScheduleFieldsValue | null,
): boolean {
  if (form.recurrence !== "hourly" && form.recurrence !== "custom") return false
  if (baseline && !scheduleRelevantFieldsChanged(form, baseline)) return false
  const anchor = parseHourlyCustomAnchor(form.startDate, form.timeOfDay)
  if (!anchor) return false
  return anchor.getTime() <= Date.now()
}

function inferCustomAmountUnit(intervalSeconds: number): { amount: string; unit: ScheduleCustomUnit } {
  if (intervalSeconds >= 86400 && intervalSeconds % 86400 === 0) {
    return { amount: String(intervalSeconds / 86400), unit: "days" }
  }
  if (intervalSeconds >= 3600 && intervalSeconds % 3600 === 0) {
    return { amount: String(intervalSeconds / 3600), unit: "hours" }
  }
  return { amount: String(Math.max(1, Math.round(intervalSeconds / 60))), unit: "minutes" }
}

/** Reconstruct the schedule-editor fields from a stored scheduled-trigger
 * config, including a best-effort mapping for legacy configs saved before
 * `recurrence` existed (flat interval_seconds/next_run_at only). */
function scheduleFieldsFromConfig(config: Record<string, unknown>): ScheduleFieldsValue {
  const defaults = scheduleFieldsDefaults()
  const recurrenceRaw = configString(config, "recurrence")
  const isCalendarRecurrence =
    recurrenceRaw === "daily" || recurrenceRaw === "weekly" || recurrenceRaw === "monthly"
  const configuredTimezone = configString(config, "timezone") || "UTC"

  const startAtRaw = configString(config, "start_at") || configString(config, "next_run_at")
  let startDate = ""
  let anchorTimeOfDay: string | null = null
  if (CALENDAR_DATE_PATTERN.test(startAtRaw)) {
    // The plain "YYYY-MM-DD" shape buildConfig now sends for daily/weekly/
    // monthly (see buildConfig) — already zone-agnostic, nothing to convert.
    startDate = startAtRaw
  } else if (startAtRaw) {
    // A full ISO instant: either the flat hourly/custom `next_run_at` (no
    // zone-bound civil time — rendering it in the CURRENT browser's zone is
    // self-consistent, since buildConfig reconstructs it the same way), or
    // legacy daily/weekly/monthly data predating the plain-date format
    // above. The latter must render in the trigger's OWN stored zone, not
    // the editing browser's — otherwise the displayed date can be off by a
    // day from whichever zone the schedule actually runs in.
    const anchorDate = new Date(startAtRaw)
    if (!Number.isNaN(anchorDate.getTime())) {
      if (isCalendarRecurrence) {
        startDate = zonedIsoDate(anchorDate, configuredTimezone)
        anchorTimeOfDay = zonedTimeOfDay(anchorDate, configuredTimezone)
      } else {
        startDate = localIsoDate(anchorDate)
        anchorTimeOfDay = localTimeOfDay(anchorDate)
      }
    }
  }
  if (!startDate) {
    // No stored start_at/next_run_at at all: the backend genuinely allows
    // an unset start_at for daily/weekly/monthly (it's optional at the
    // schema level), but buildConfig requires a startDate to save at all.
    // Without this fallback, a calendar trigger created via the API with no
    // start_at would reconstruct as permanently uneditable in this dialog —
    // default to today. For a calendar recurrence, "today" must be computed
    // in the trigger's OWN configured zone (configuredTimezone, already used
    // a few lines above for the same reason) rather than defaults.startDate
    // (the browser's local today) — otherwise an incidental Save from a
    // browser whose local date disagrees with the trigger's zone silently
    // sets start_at to the wrong calendar day. hourly/custom has no zone
    // concept at all, so the browser-local default is correct there, same
    // as a brand-new trigger draft (scheduleFieldsDefaults).
    startDate = isCalendarRecurrence
      ? zonedIsoDate(new Date(), configuredTimezone)
      : defaults.startDate
  }
  // A stored time_of_day is authoritative when present; otherwise fall back
  // to the REAL scheduled time carried by the anchor timestamp itself
  // (legacy configs, or any config saved before a recurrence used
  // time_of_day) rather than a hardcoded default — the latter would show a
  // fabricated time and silently move the schedule there on the next Save.
  const timeOfDay = configString(config, "time_of_day") || anchorTimeOfDay || defaults.timeOfDay

  if ((RECURRENCE_TYPES as string[]).includes(recurrenceRaw)) {
    const recurrence = recurrenceRaw as ScheduleRecurrence
    const weekdaysRaw = config.weekdays
    const weekdays = Array.isArray(weekdaysRaw)
      ? weekdaysRaw.map((day) => Number(day)).filter((day) => Number.isInteger(day))
      : defaults.weekdays
    const dayOfMonthRaw = config.day_of_month
    const dayOfMonth =
      typeof dayOfMonthRaw === "number" && Number.isInteger(dayOfMonthRaw)
        ? dayOfMonthRaw
        : defaults.dayOfMonth
    let customAmount = defaults.customAmount
    let customUnit = defaults.customUnit
    if (recurrence === "custom") {
      const interval = Number(configScalar(config, "interval_seconds"))
      if (Number.isFinite(interval) && interval > 0) {
        const inferred = inferCustomAmountUnit(interval)
        customAmount = inferred.amount
        customUnit = inferred.unit
      }
    }
    return {
      recurrence,
      timeOfDay,
      weekdays: weekdays.length ? weekdays : defaults.weekdays,
      dayOfMonth,
      customAmount,
      customUnit,
      startDate,
      // The zone this config is ACTUALLY interpreted in server-side (see
      // _schedule_tzinfo: absent means UTC) — not the editing browser's
      // current zone. Preserving this (rather than re-deriving it fresh on
      // every Save) is what stops an incidental machine/zone difference
      // from silently relocating an already-armed schedule.
      //
      // hourly/custom configs can never carry a stored timezone (the
      // backend schema rejects it for those recurrences — it's a flat
      // interval with no civil time-of-day for a zone to apply to), so
      // `configString(...) || "UTC"` would always hit the "UTC" fallback
      // for them, permanently overwriting whatever zone was showing before
      // a chip switch. Fall back to the browser's own zone instead, same as
      // the legacy/no-recurrence branch below and scheduleFieldsDefaults.
      timezone:
        recurrence === "hourly" || recurrence === "custom"
          ? configString(config, "timezone") || defaults.timezone
          : configString(config, "timezone") || "UTC",
    }
  }

  // Legacy config saved before `recurrence` existed: only interval_seconds /
  // next_run_at are present. `timeOfDay` still carries the anchor-derived
  // real time (not defaults.timeOfDay's hardcoded "09:00") from above.
  const rawInterval = configScalar(config, "interval_seconds")
  if (!rawInterval) {
    // No interval_seconds at all (only next_run_at) is a deliberate
    // backend-supported one-shot: it fires once and the scan loop disables
    // it (see test_scheduled_scan_disables_one_shot_trigger) — there was
    // never a "recurrence" to infer. Defaulting this to "hourly" would
    // claim a recurring cadence the config doesn't have, and a no-op Save
    // would then write interval_seconds: 3600, silently and irreversibly
    // turning a one-time job into a perpetual one. "custom" at least makes
    // no false claim about the cadence; it does NOT by itself preserve the
    // one-shot semantics on Save (this UI has no representation for "no
    // interval" — a real fix needs a dedicated one-time recurrence option).
    return { ...defaults, recurrence: "custom", timeOfDay, startDate }
  }
  const interval = Number(rawInterval) || 3600
  if (interval === 3600) {
    return { ...defaults, recurrence: "hourly", timeOfDay, startDate }
  }
  if (interval === 86400) {
    return { ...defaults, recurrence: "daily", timeOfDay, startDate }
  }
  const { amount, unit } = inferCustomAmountUnit(interval)
  return {
    ...defaults,
    recurrence: "custom",
    customAmount: amount,
    customUnit: unit,
    timeOfDay,
    startDate,
  }
}

function formatDateTime(value: string | null): string {
  if (!value) return "-"
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

/** Stringify a config field that may arrive as a number or a string
 * (interval_seconds, oauth_account_id, ...); anything else (missing,
 * object, boolean) is "not present". */
function configScalar(config: Record<string, unknown>, key: string): string {
  const value = config[key]
  return typeof value === "number" || typeof value === "string" ? String(value) : ""
}

function configString(config: Record<string, unknown>, key: string): string {
  const value = config[key]
  return typeof value === "string" ? value : ""
}

function formFromTrigger(trigger: AgentTrigger): TriggerFormState {
  const scheduleFields =
    trigger.type === "scheduled" ? scheduleFieldsFromConfig(trigger.config) : scheduleFieldsDefaults()
  return {
    type: trigger.type,
    name: trigger.name,
    enabled: trigger.enabled,
    secret: "",
    promptTemplate: trigger.prompt_template ?? "",
    // The "*"/"all" match-anything sentinels render as a blank field, whose
    // placeholder explains that blank watches every incoming email — but a
    // config missing the key entirely (legacy) renders as "INBOX", matching
    // what the backend actually still does with it (see gmailFormWatchLabel).
    watchLabel: trigger.type === "gmail" ? gmailFormWatchLabel(trigger.config) : "",
    senderFilter: trigger.type === "gmail" ? configString(trigger.config, "sender_filter") : "",
    subjectKeyword: trigger.type === "gmail" ? configString(trigger.config, "subject_keyword") : "",
    oauthAccountId:
      trigger.type === "gmail" ? configScalar(trigger.config, "oauth_account_id") : "",
    ...scheduleFields,
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
  const { t, locale } = useI18n()
  const router = useRouter()
  const [liveTriggers, setLiveTriggers] = useState<AgentTrigger[]>([])
  const [activeType, setActiveTypeState] = useState<AgentTriggerType | null>(null)
  // Detail sub-mode: false = the type's trigger list (cards / empty state),
  // true = the editor form for selectedTriggerId (null = creating a new one).
  const [editing, setEditing] = useState(false)
  const [selectedTriggerId, setSelectedTriggerIdState] = useState<number | null>(null)
  const [runs, setRuns] = useState<AgentTriggerRun[]>([])
  const [loading, setLoading] = useState(false)
  const [runsLoading, setRunsLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  // A Set, not a scalar: two overview switches toggled back-to-back must each
  // keep their own guard, or one PATCH resolving would re-enable the other
  // type's still-in-flight switch (matches agent-builder.tsx's summary cards).
  const [busyTypes, setBusyTypes] = useState<ReadonlySet<AgentTriggerType>>(new Set())
  // Local draft: field edits (other than the header/overview switch, which
  // persists immediately) only reach the server when Save is pressed —
  // Cancel/Back/Done/switching selection all just discard this draft.
  const [form, setForm] = useState<TriggerFormState>(emptyForm)
  // True while a one-click test run is being started; drives the button's
  // "Running test…" state without blocking the rest of the editor.
  const [testing, setTesting] = useState(false)
  // Staging-mode test result: the agent doesn't exist server-side yet, so a
  // test renders the trigger prompt locally — exactly what a real firing
  // would send to the agent — and shows it as a run row inside the editor.
  const [stagedTestRun, setStagedTestRun] = useState<{ id: string; prompt: string } | null>(null)
  // True right after "Generate secret" filled the secret field — shows the
  // copy-it-now hint until the user types their own value.
  const [secretGenerated, setSecretGenerated] = useState(false)
  const [secretReveal, setSecretReveal] = useState<string | null>(null)
  const [copied, setCopied] = useState<string | null>(null)
  const [deleteConfirmId, setDeleteConfirmId] = useState<number | null>(null)
  const [gmailFiltersOpen, setGmailFiltersOpen] = useState(false)
  const [gmailAccounts, setGmailAccounts] = useState<GmailAccount[] | null>(null)
  // Set by "Change account" (clearing oauthAccountId is an explicit user
  // choice); makes the auto-bind effect below not overwrite it should
  // gmailAccounts ever become reactive to something other than the current
  // one-shot per-open fetch. Reset alongside the other per-view state.
  const gmailAccountClearedByUserRef = useRef(false)
  const [gmailAccountsLoading, setGmailAccountsLoading] = useState(false)
  const selectedTriggerIdRef = useRef<number | null>(null)
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

  const activeTypeTriggers = useMemo(
    () => (activeType ? triggerGroups[activeType] : []),
    [activeType, triggerGroups],
  )
  // A null selectedTriggerId means "creating a new trigger" (e.g. via the Add
  // button), so it must NOT fall back to an existing trigger — that would make
  // handleSubmit overwrite it. Every browse flow selects an id explicitly
  // (openType, beginEdit, loadTriggers, the open effect).
  const selectedTrigger = useMemo(() => {
    if (!activeType || selectedTriggerId === null) return null
    return activeTypeTriggers.find((trigger) => trigger.id === selectedTriggerId) ?? null
  }, [activeType, activeTypeTriggers, selectedTriggerId])

  // The schedule fields as originally loaded from the trigger being edited
  // (null for a brand-new draft) — the baseline scheduleWillFireImmediately
  // compares the live form against to tell a genuine schedule edit apart
  // from an unrelated field change (PR #1051 review, N2 follow-up).
  const loadedScheduleFields = useMemo(
    () =>
      selectedTrigger && selectedTrigger.type === "scheduled"
        ? scheduleFieldsFromConfig(selectedTrigger.config)
        : null,
    [selectedTrigger],
  )

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

      // Selection only matters while the editor is open. Keep it when the
      // trigger still exists; if it vanished server-side, fall back to the
      // type's list view. A null selection (creating a new draft) is left
      // untouched so a background resync never kicks the user out of it.
      const currentSelectedId = preferredTriggerId ?? selectedTriggerIdRef.current
      if (currentSelectedId !== null && currentSelectedId !== undefined) {
        if (data.some((trigger) => trigger.id === currentSelectedId)) {
          setSelectedTriggerId(currentSelectedId)
        } else {
          setSelectedTriggerId(null)
          setEditing(false)
        }
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
    // Opening (optionally straight into a type via initialType) always lands
    // on a list: the overview, or the type's trigger list. The editor is only
    // entered through an explicit action (edit / add / first toggle-on).
    setActiveType(initialType)
    setEditing(false)
    setSelectedTriggerId(null)
    setSecretReveal(null)
    setCopied(null)
    setDeleteConfirmId(null)
    setStagedTestRun(null)
    setSecretGenerated(false)
    setGmailFiltersOpen(false)
    gmailAccountClearedByUserRef.current = false
    setRuns([])
    if (initialType) setForm(emptyForm(initialType))
    void loadTriggers(null)
  }, [initialType, loadTriggers, open, setActiveType, setSelectedTriggerId])

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
  // never chosen silently. The setForm call is idempotent (a no-op once
  // oauthAccountId is already set), and gmailAccounts is only ever replaced
  // once per dialog-open (see the fetch effect above, gated on `[open]`
  // alone) — so re-running this on every unrelated dependency change within
  // the same open dialog never clobbers a since-picked selection; only an
  // explicit "Change account" (gmailAccountClearedByUserRef) needs its own
  // guard.
  useEffect(() => {
    if (!open || !editing || activeType !== "gmail" || selectedTrigger) return
    if (gmailAccountClearedByUserRef.current) return
    if (gmailAccounts?.length === 1) {
      const onlyAccountId = String(gmailAccounts[0].id)
      setForm((current) =>
        current.oauthAccountId ? current : { ...current, oauthAccountId: onlyAccountId },
      )
    }
  }, [activeType, editing, gmailAccounts, open, selectedTrigger])

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
      void loadRunsFor(selectedTrigger)
    } else {
      setForm(emptyForm(activeType))
      setRuns([])
    }
  }, [activeType, loadRunsFor, open, selectedTrigger])

  // Navigation helpers set the form synchronously (no flash of stale values)
  // and stamp syncedFormKeyRef so the form-sync effect treats the target as
  // already synced. secretReveal deliberately survives navigation: it is a
  // one-time value the user must copy, so only closing the dialog or deleting
  // a trigger clears it.

  // Opens a type's trigger list (cards, or the empty state when none exist).
  // Shared state reset behind every navigation inside the dialog: any new
  // per-view state (confirm popovers, staged test previews, disclosure
  // open/closed, ...) is reset HERE, once, instead of in four copies.
  const showDetailView = (
    type: AgentTriggerType,
    options: { editing: boolean; trigger?: AgentTrigger | null; form: TriggerFormState },
  ) => {
    const trigger = options.trigger ?? null
    syncedFormKeyRef.current = formKeyFor(type, trigger?.id ?? null)
    setActiveType(type)
    setEditing(options.editing)
    setSelectedTriggerId(trigger?.id ?? null)
    setDeleteConfirmId(null)
    setStagedTestRun(null)
    setSecretGenerated(false)
    setGmailFiltersOpen(false)
    gmailAccountClearedByUserRef.current = false
    setForm(options.form)
    if (trigger) {
      void loadRunsFor(trigger)
    } else {
      setRuns([])
    }
  }

  const openType = (type: AgentTriggerType) => {
    showDetailView(type, { editing: false, form: emptyForm(type) })
  }

  const beginCreateForType = (type: AgentTriggerType, initial?: Partial<TriggerFormState>) => {
    showDetailView(type, { editing: true, form: { ...emptyForm(type), ...initial } })
  }

  const beginEdit = (trigger: AgentTrigger) => {
    showDetailView(trigger.type, { editing: true, trigger, form: formFromTrigger(trigger) })
  }

  // Cancel/after-save: leave the editor and land back on the type's list,
  // discarding any draft (Save is what persists).
  const closeEditor = () => {
    if (activeType) openType(activeType)
  }

  const setFormValue = <K extends keyof TriggerFormState>(
    key: K,
    value: TriggerFormState[K],
  ) => {
    setForm((current) => ({ ...current, [key]: value }))
  }

  const buildConfig = (sourceForm: TriggerFormState = form): Record<string, unknown> => {
    if (sourceForm.type === "webhook") return {}

    if (sourceForm.type === "gmail") {
      // Optional, per the reference design: blank watches all incoming
      // emails ("*" matches any label).
      const watchLabel = sourceForm.watchLabel.trim() || "*"
      const accountId = Number(sourceForm.oauthAccountId)
      if (!sourceForm.oauthAccountId.trim() || !Number.isInteger(accountId)) {
        throw new Error(t("triggers.validation.gmailAccount"))
      }
      const config: Record<string, unknown> = {
        watch_label: watchLabel,
        oauth_account_id: accountId,
      }
      const senderFilter = sourceForm.senderFilter.trim()
      const subjectKeyword = sourceForm.subjectKeyword.trim()
      if (senderFilter) config.sender_filter = senderFilter
      if (subjectKeyword) config.subject_keyword = subjectKeyword
      return config
    }

    // The start date is required: for every recurrence the picked time only
    // reaches the backend through this anchor, so a missing date would
    // silently ignore the chosen time.
    if (!sourceForm.startDate) {
      throw new Error(t("triggers.validation.startDate"))
    }

    if (
      sourceForm.recurrence === "weekly" ||
      sourceForm.recurrence === "monthly" ||
      sourceForm.recurrence === "daily"
    ) {
      // Sent as a bare "YYYY-MM-DD" date — zone-agnostic on purpose. The
      // backend combines it with time_of_day/timezone itself (the same
      // _localize the actual occurrence math uses); materializing a UTC
      // instant here via `new Date(...)` would parse the wall-clock fields
      // in THIS BROWSER's zone, which silently disagrees with the trigger's
      // stored `timezone` whenever they differ (e.g. editing from a
      // different machine than the trigger was created on) — the date
      // shifts by up to a day, or the schedule starts a full cycle late.
      if (!CALENDAR_DATE_PATTERN.test(sourceForm.startDate)) {
        throw new Error(t("triggers.validation.nextRunAt"))
      }
      const config: Record<string, unknown> = {
        recurrence: sourceForm.recurrence,
        time_of_day: sourceForm.timeOfDay || "00:00",
        // The user's local wall-clock — the backend needs the zone to
        // compute occurrences correctly. Sent from form state (populated
        // once, from the trigger's stored zone or the browser's own for a
        // new trigger — see scheduleFieldsFromConfig / scheduleFieldsDefaults),
        // NOT re-derived here on every Save: doing that would silently
        // relocate an already-armed schedule if the user happens to edit it
        // from a different machine/zone than it was created in.
        timezone: sourceForm.timezone || Intl.DateTimeFormat().resolvedOptions().timeZone,
        start_at: sourceForm.startDate,
      }
      if (sourceForm.recurrence === "weekly") {
        if (!sourceForm.weekdays.length) {
          throw new Error(t("triggers.validation.scheduleRequired"))
        }
        config.weekdays = sourceForm.weekdays
      } else if (sourceForm.recurrence === "monthly") {
        config.day_of_month = sourceForm.dayOfMonth
      }
      return config
    }

    // hourly/custom: no civil time-of-day for a zone to apply to (see the
    // `recurrence` field's docstring in schemas.py) — the anchor really is
    // just an absolute instant, so browser-local parsing is fine, and
    // self-consistent on every re-save since the editor always displays
    // this anchor back in the CURRENT browser's zone too (scheduleFieldsFromConfig).
    const anchor = parseHourlyCustomAnchor(sourceForm.startDate, sourceForm.timeOfDay)
    if (!anchor) {
      throw new Error(t("triggers.validation.nextRunAt"))
    }
    let intervalSeconds = 3600
    if (sourceForm.recurrence === "custom") {
      const amount = Number(sourceForm.customAmount)
      if (!Number.isInteger(amount) || amount <= 0) {
        throw new Error(t("triggers.validation.interval"))
      }
      intervalSeconds = amount * unitSecondsFor(sourceForm.customUnit)
    }
    return {
      recurrence: sourceForm.recurrence,
      interval_seconds: intervalSeconds,
      next_run_at: anchor.toISOString(),
    }
  }

  const buildPayload = (sourceForm: TriggerFormState = form) => {
    // Gmail triggers have no name field in the editor (the bound account IS
    // the identity, like the reference design), so a fresh/never-customized
    // trigger's name always tracks the bound account's email — including
    // across a rebind, so the card doesn't keep showing an account it no
    // longer watches. Only names that are STILL exactly the account the
    // trigger was previously bound to count as "never customized" — a
    // deliberately renamed trigger (e.g. "Support inbox") keeps its name
    // when rebound, same as it would for any other unrelated edit.
    const boundEmail =
      sourceForm.type === "gmail"
        ? ((gmailAccounts ?? []).find(
            (account) => String(account.id) === sourceForm.oauthAccountId,
          )?.email ?? null)
        : null
    const previouslyBoundEmail =
      sourceForm.type === "gmail" && selectedTrigger?.type === "gmail"
        ? ((gmailAccounts ?? []).find(
            (account) =>
              String(account.id) === configScalar(selectedTrigger.config, "oauth_account_id"),
          )?.email ?? null)
        : null
    const wasAutoNamed =
      sourceForm.type === "gmail" &&
      // A brand-new gmail draft has nothing to preserve — always derive.
      (!selectedTrigger ||
        // An existing trigger's old account is no longer resolvable (e.g.
        // disconnected) — can't tell whether the current name was ever
        // auto-derived, so the safer default is to leave a possible
        // customization alone rather than risk clobbering it.
        (previouslyBoundEmail !== null && sourceForm.name.trim() === previouslyBoundEmail))
    const name = wasAutoNamed
      ? boundEmail || defaultNameForType(sourceForm.type)
      : sourceForm.name.trim() || boundEmail || defaultNameForType(sourceForm.type)
    if (name.length > 200) {
      throw new Error(t("triggers.validation.nameLength"))
    }

    return {
      type: sourceForm.type,
      name,
      // For an existing trigger, prefer the live server-known `enabled`
      // over the form's captured snapshot: the type-level master switch and
      // the per-card switch both PATCH `enabled` directly while an editor
      // for that same trigger may already be open, and the form-sync effect
      // is gated on a same-trigger-id key so it does not re-run just
      // because `enabled` changed externally. Without this, editing an
      // unrelated field after toggling the switch off (with the editor
      // still open) would resend the stale `enabled: true` on Save and
      // silently re-enable it — the editor has no `enabled` control of its
      // own to make the discrepancy visible. A brand-new draft (no
      // selectedTrigger) has no live value to prefer, so it keeps the
      // form's own value, set explicitly by beginCreateForType.
      enabled: selectedTrigger ? selectedTrigger.enabled : sourceForm.enabled,
      config: buildConfig(sourceForm),
      prompt_template: sourceForm.promptTemplate.trim() ? sourceForm.promptTemplate : null,
      secret: sourceForm.type === "webhook" && sourceForm.secret.trim() ? sourceForm.secret.trim() : null,
    }
  }

  const notifyChanged = () => {
    onChanged?.()
  }

  // The type-level switch (overview card and detail header). Turning it on
  // with no trigger of the type yet opens a fresh draft editor — nothing is
  // created until Save (per the design refresh). With existing triggers it
  // enables the primary one / disables them all, without navigating.
  const handleTypeToggle = async (type: AgentTriggerType, checked: boolean) => {
    if (!canOperate) return
    const typeTriggers = triggerGroups[type]
    if (checked && typeTriggers.length === 0) {
      // Already composing a draft of this type: the switch is a no-op
      // rather than a form reset — saving the draft is what turns it on.
      if (editing && activeType === type && selectedTriggerId === null) return
      if (type === "gmail") {
        // Accounts still loading: don't guess between "connect" and "draft".
        if (gmailAccountsLoading || gmailAccounts === null) return
        if (gmailAccounts.length === 0) {
          // Nothing to enable yet — land on the "connect Gmail" empty state
          // (switch left off) instead of a form with no account to bind to.
          openType("gmail")
          return
        }
      }
      // First toggle-on goes straight into the new-trigger editor. The draft
      // saves as enabled, which is what flips this switch on for real.
      beginCreateForType(type, { enabled: true })
      return
    }
    // Prefer whichever trigger of this type is currently open in the editor
    // (even if it's disabled) over the "first enabled, else first in list"
    // heuristic — otherwise turning the type-level switch on while a
    // different, currently-disabled trigger is open enables the wrong one,
    // leaving the one the user is actually looking at still off (PR #1051
    // review, N9; pre-existing, not introduced by this PR).
    const openTriggerId = editing && activeType === type ? selectedTriggerId : null
    const openTrigger =
      openTriggerId !== null
        ? typeTriggers.find((trigger) => trigger.id === openTriggerId)
        : undefined
    const primary =
      openTrigger ??
      typeTriggers.find((trigger) => trigger.enabled) ??
      typeTriggers[0] ??
      null
    setBusyTypes((current) => new Set(current).add(type))
    try {
      if (isStaging && staged) {
        if (checked) {
          staged.onChange(
            staged.triggers.map((item) =>
              item.clientId === primary!.id ? { ...item, enabled: true } : item,
            ),
          )
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
        const updated = await updateOwnerTrigger(resolvedOwner, primary!.id, { enabled: true })
        setLiveTriggers((current) =>
          current.map((item) => (item.id === updated.id ? updated : item)),
        )
        notifyChanged()
        toast.success(t("triggers.messages.enabled"))
      } else {
        const updatedList = await disableOwnerTriggersOfType(resolvedOwner, typeTriggers, type)
        setLiveTriggers((current) => mergeUpdatedTriggers(current, updatedList))
        notifyChanged()
        toast.success(t("triggers.messages.disabled"))
      }
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

  // Per-card switch in the type's trigger list: a minimal enabled-only
  // update for that one trigger. The type-level switch is derived (any
  // enabled trigger), so it follows automatically.
  const handleItemToggle = async (trigger: AgentTrigger, checked: boolean) => {
    if (!canOperate) return
    if (isStaging && staged) {
      staged.onChange(
        staged.triggers.map((item) =>
          item.clientId === trigger.id ? { ...item, enabled: checked } : item,
        ),
      )
      notifyChanged()
      toast.success(checked ? t("triggers.messages.enabled") : t("triggers.messages.disabled"))
      return
    }
    if (!resolvedOwner) return
    setBusy(true)
    try {
      const updated = await updateOwnerTrigger(resolvedOwner, trigger.id, { enabled: checked })
      // Patch from the response, not a hand-set `enabled` — a scheduled
      // trigger's next_run_at/last_run_at can change server-side on
      // enable/disable.
      setLiveTriggers((current) =>
        current.map((item) => (item.id === updated.id ? updated : item)),
      )
      notifyChanged()
      toast.success(checked ? t("triggers.messages.enabled") : t("triggers.messages.disabled"))
    } catch (err) {
      console.error(err)
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
    // The persisted trigger (live mode only) — lets save-then-act callers
    // like the Test button keep working with the fresh id.
    saved: AgentTrigger | null
  }

  // Persists the current form (update selected / create new). Saving
  // normally lands back on the type's trigger list; `keepEditing` keeps the
  // editor open on the saved trigger instead (used by save-then-test).
  const handleSubmit = async (options?: { keepEditing?: boolean }): Promise<SubmitResult> => {
    if (!canOperate) return { ok: false, secret: null, saved: null }
    let payload
    try {
      payload = buildPayload()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("triggers.messages.saveFailed"))
      return { ok: false, secret: null, saved: null }
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
      }
      notifyChanged()
      toast.success(t("triggers.messages.staged"))
      // Saving lands back on the type's trigger list, where the new/updated
      // card is visible.
      closeEditor()
      return { ok: true, secret: null, saved: null }
    }

    if (!resolvedOwner) return { ok: false, secret: null, saved: null }
    setBusy(true)
    try {
      const saved = selectedTrigger
        ? await updateOwnerTrigger(resolvedOwner, selectedTrigger.id, payload)
        : await createOwnerTrigger(resolvedOwner, payload)
      const revealedSecret = saved.webhook_secret ?? null
      // A freshly generated secret is rendered right where the user lands
      // next — inside the editor (keepEditing) or above the list — and
      // blocks closing the dialog until dismissed.
      setSecretReveal(revealedSecret)
      setLiveTriggers((current) =>
        selectedTrigger
          ? current.map((item) => (item.id === saved.id ? saved : item))
          : [...current, saved],
      )
      notifyChanged()
      toast.success(selectedTrigger ? t("triggers.messages.updated") : t("triggers.messages.created"))
      if (options?.keepEditing) {
        // Stay in the editor, now bound to the persisted trigger. Stamp the
        // form key so the sync effect doesn't wipe the (just-saved) fields.
        syncedFormKeyRef.current = formKeyFor(saved.type, saved.id)
        setSelectedTriggerId(saved.id)
        setForm(formFromTrigger(saved))
      } else {
        closeEditor()
      }
      return { ok: true, secret: revealedSecret, saved }
    } catch (err) {
      console.error(err)
      toast.error(err instanceof Error ? err.message : t("triggers.messages.saveFailed"))
      return { ok: false, secret: null, saved: null }
    } finally {
      setBusy(false)
    }
  }

  // Field edits are a local draft — only Save persists them. Back, Cancel,
  // switching pills, and Add all just discard the draft and navigate; nothing
  // to await, nothing to lose (a fresh secret is separately protected below,
  // since it's already unrecoverable server-side once unseen).
  const handleSave = () => {
    void handleSubmit()
  }

  const handleDone = () => {
    if (secretReveal) return
    closeDialog(false)
  }

  // Header Back always returns to the overview (type list), abandoning any
  // open draft — like the reference design.
  const handleBack = () => {
    setActiveType(null)
    setEditing(false)
  }

  // The editor's inline "Cancel" button: back to the type's trigger list.
  const handleCancel = closeEditor

  const handleAddAnother = (type: AgentTriggerType) => {
    beginCreateForType(type, { enabled: true })
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

  // Confirmation happens in the card's popover; by the time this runs the
  // user has already clicked the destructive button there.
  const handleDelete = async (trigger: AgentTrigger) => {
    if (!canOperate) return
    if (isStaging && staged) {
      staged.onChange(staged.triggers.filter((item) => item.clientId !== trigger.id))
      if (selectedTriggerIdRef.current === trigger.id) {
        setSelectedTriggerId(null)
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
      setLiveTriggers((current) => current.filter((item) => item.id !== trigger.id))
      if (selectedTriggerIdRef.current === trigger.id) {
        setSelectedTriggerId(null)
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

  // One-click test, per the reference design: no payload editing — the
  // backend runs the trigger with a sample payload and the run shows up in
  // Recent runs right below. A trigger can only run once it exists
  // server-side, so testing an unsaved draft saves it first (staying in the
  // editor) and then fires the test — no extra step for the user.
  //
  // Staged triggers (agent not created yet) have nothing server-side to run,
  // so the test happens right here in the editor: the trigger prompt is
  // rendered locally with the sample payload — exactly what a real firing
  // will send to the agent — and shown as a run row below.
  const handleTest = async () => {
    if (isStaging) {
      let payload
      try {
        payload = buildPayload()
      } catch (err) {
        toast.error(err instanceof Error ? err.message : t("triggers.messages.saveFailed"))
        return
      }
      setStagedTestRun({
        id: `trigger-run:test:draft:${Math.random().toString(36).slice(2, 12)}`,
        prompt: renderStagedTestPrompt(
          payload.type,
          payload.name,
          payload.prompt_template,
          TEST_RUN_PAYLOAD,
        ),
      })
      return
    }
    if (!resolvedOwner) return
    setTesting(true)
    try {
      // Always save first — new draft or unsaved edits to an existing
      // trigger alike — so the test exercises exactly what is on screen.
      const submit = await handleSubmit({ keepEditing: true })
      // Validation/save failures already toasted inside handleSubmit.
      if (!submit.ok || !submit.saved) return
      const target = submit.saved
      const result = await testOwnerTrigger(resolvedOwner, target.id, {
        payload: TEST_RUN_PAYLOAD,
        source_event_id: null,
      })
      // Fetch runs for the explicit target: right after a save-then-test the
      // selectedTrigger state may not have committed yet.
      await loadRunsFor(target)
      toast.success(
        result.duplicate
          ? t("triggers.messages.testDuplicate")
          : t("triggers.messages.testStarted"),
      )
    } catch (err) {
      console.error(err)
      toast.error(err instanceof Error ? err.message : t("triggers.messages.testFailed"))
    } finally {
      setTesting(false)
    }
  }

  // Client-side secret generation, like the reference design: fills the
  // field with a random whsec_ value the user can copy before saving.
  const handleGenerateSecret = () => {
    // 64-character alphabet: 256 % 64 === 0, so byte % length is unbiased.
    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    const bytes = new Uint8Array(32)
    crypto.getRandomValues(bytes)
    const token = Array.from(bytes, (byte) => chars[byte % chars.length]).join("")
    setFormValue("secret", `whsec_${token}`)
    setSecretGenerated(true)
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

  // Dismissal (Esc, overlay click, header X) discards any unsaved draft like
  // every other exit path. A freshly generated webhook secret is the one
  // thing dismissal must not drop: unlike a form draft, it already exists
  // server-side and is unrecoverable once unseen, so — like handleDone —
  // closing waits until it has been shown (dismissed via its own X).
  const handleDismiss = (nextOpen: boolean) => {
    if (nextOpen) {
      onOpenChange(true)
      return
    }
    if (secretReveal) return
    closeDialog(false)
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

  // One-line summary under a card's name: the webhook URL snippet, the
  // schedule recurrence, or the Gmail account/label being watched.
  const triggerSummary = (trigger: AgentTrigger): string => {
    if (trigger.type === "webhook") {
      if (trigger.callback_id) return `…/webhook/${trigger.callback_id.slice(0, 12)}…`
      // Staged webhooks have no endpoint yet — it's minted with the agent.
      return t("triggers.staging.webhookPending")
    }
    if (trigger.type === "scheduled") {
      return summarizeSchedule(scheduleFieldsFromConfig(trigger.config), t, locale)
    }
    const accountId = configScalar(trigger.config, "oauth_account_id")
    const email = gmailAccounts?.find((account) => String(account.id) === accountId)?.email
    // "*"/"all" are the backend's match-anything sentinels; a MISSING label
    // is INBOX (a legacy default the backend still applies), not wildcard —
    // see gmailFormWatchLabel. Anything else, including an explicit INBOX,
    // is a real label filter.
    const watchLabel = gmailFormWatchLabel(trigger.config)
    const labelPart = watchLabel || t("triggers.item.gmailAllEmails")
    return [email, labelPart].filter(Boolean).join(" · ")
  }

  // Manage-list card: per-trigger enable switch, edit, and delete (with its
  // confirmation popover) — the reference design's list rows.
  const renderTriggerCard = (trigger: AgentTrigger) => (
    <div
      key={trigger.id}
      className={cn(
        "flex items-center gap-3 rounded-[10px] border bg-background px-3.5 py-3 transition-opacity",
        !trigger.enabled && "opacity-60",
      )}
    >
      <div
        className={cn(
          "flex h-[34px] w-[34px] shrink-0 items-center justify-center rounded-lg",
          typeIconClass(trigger.type),
        )}
      >
        {renderTypeIcon(trigger.type, "h-4 w-4")}
      </div>
      <div className="min-w-0 flex-1">
        <div className="truncate text-[13px] font-semibold">{trigger.name}</div>
        <div className="truncate text-xs text-muted-foreground">{triggerSummary(trigger)}</div>
        {trigger.provisioning_status === "failed" && trigger.provisioning_error && (
          // Surfaces the backend's provisioning_status/provisioning_error
          // (PR #1051 review, N6) — otherwise a trigger that silently
          // stopped firing (e.g. scan_due_scheduled_triggers disabling it
          // after a recompute failure) shows no visible signal beyond the
          // generic enabled/disabled state. `title` carries the full message
          // as a native tooltip, same pattern as the edit/delete buttons below.
          <div
            className="mt-0.5 flex items-center gap-1 truncate text-xs text-destructive"
            title={trigger.provisioning_error}
          >
            <AlertCircle className="h-3 w-3 shrink-0" />
            <span className="truncate">{trigger.provisioning_error}</span>
          </div>
        )}
      </div>
      <Switch
        checked={trigger.enabled}
        disabled={busy || busyTypes.has(trigger.type) || !canOperate}
        onCheckedChange={(checked) => void handleItemToggle(trigger, checked)}
      />
      <button
        type="button"
        className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        onClick={() => beginEdit(trigger)}
        aria-label={t("triggers.actions.edit")}
        title={t("triggers.actions.edit")}
        disabled={busy}
      >
        <Pencil className="h-3.5 w-3.5" />
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
              "rounded-md p-1.5 transition-colors",
              deleteConfirmId === trigger.id
                ? "bg-destructive/10 text-destructive"
                : "text-muted-foreground hover:bg-muted hover:text-destructive",
            )}
            disabled={busy}
          >
            <Trash2 className="h-3.5 w-3.5" />
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
  )

  // The type's manage list: empty state when nothing exists yet, otherwise
  // the cards plus an "Add another …" button.
  const renderTriggerList = () => {
    if (!activeType) return null
    return (
      <div className="space-y-3">
        {renderSecretReveal()}
        {activeTypeTriggers.length === 0 ? (
          renderEmptyState(activeType)
        ) : (
          <>
            <div className="space-y-2.5">{activeTypeTriggers.map(renderTriggerCard)}</div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => handleAddAnother(activeType)}
              disabled={busy || !canOperate}
              className="h-8 rounded-full border-dashed border-primary/45 bg-primary/5 text-xs text-primary hover:border-primary hover:bg-primary/10"
            >
              <Plus className="mr-1.5 h-4 w-4" />
              {t(ADD_ANOTHER_KEYS[activeType] as never)}
            </Button>
          </>
        )}
        {activeType === "gmail" && activeTypeTriggers.length > 0 && renderGmailConnectionAlert()}
      </div>
    )
  }

  // Shown inside the type's list while it has no trigger yet. The CTA opens
  // the new-trigger editor — except Gmail with no connected account, whose
  // missing prerequisite is the connection itself.
  const renderEmptyState = (type: AgentTriggerType) => {
    const needsGmailConnect = type === "gmail" && (gmailAccounts?.length ?? 0) === 0
    const handleCta = () => {
      if (needsGmailConnect) {
        onConnectGmail?.()
        return
      }
      handleAddAnother(type)
    }
    const ctaLabel = needsGmailConnect
      ? t("triggers.cards.gmail.empty.cta")
      : type === "gmail"
        ? t("triggers.cards.gmail.addTrigger")
        : t(`triggers.cards.${type}.empty.cta`)
    return (
      <div className="flex flex-col items-center gap-3 rounded-lg border border-dashed py-12 text-center">
        <div className={cn("flex h-12 w-12 items-center justify-center rounded-full", typeIconClass(type))}>
          {renderTypeIcon(type, "h-6 w-6")}
        </div>
        <div className="space-y-1 px-6">
          <p className="text-sm font-semibold">{t(`triggers.cards.${type}.empty.title`)}</p>
          <p className="text-xs text-muted-foreground">{t(`triggers.cards.${type}.empty.description`)}</p>
        </div>
        <Button
          type="button"
          size="sm"
          onClick={handleCta}
          disabled={type === "gmail" && gmailAccountsLoading}
        >
          <Plus className="mr-1.5 h-4 w-4" />
          {ctaLabel}
        </Button>
      </div>
    )
  }

  // The prompt-template field, shared by all three editors; only the
  // label/placeholder (and gmail's help line) differ per type.
  const renderPromptField = (labelKey: string, placeholderKey: string, helpKey?: string) => (
    <div className="space-y-2">
      <Label className={FIELD_LABEL_CLASS} htmlFor="trigger-prompt">
        {t(labelKey as never)}
      </Label>
      <Textarea
        id="trigger-prompt"
        value={form.promptTemplate}
        onChange={(event) => setFormValue("promptTemplate", event.target.value)}
        placeholder={t(placeholderKey as never)}
        className="min-h-[74px]"
      />
      {helpKey && <p className="text-xs text-muted-foreground">{t(helpKey as never)}</p>}
    </div>
  )

  // Shown only while Gmail is NOT connected — a connected integration needs
  // no banner (the bound account is visible on the cards/editor already).
  const renderGmailConnectionAlert = () => {
    if (gmailConnection?.isConnected) return null
    return (
      <Alert className="border-amber-200 bg-amber-50 text-amber-950 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-100">
        <Mail className="h-4 w-4" />
        <AlertTitle>{t("triggers.gmail.notConnected")}</AlertTitle>
        <AlertDescription>
          <div className="mt-1 flex flex-wrap items-center justify-between gap-3 text-sm">
            <span>{t("triggers.gmail.notConnectedDescription")}</span>
            {onConnectGmail && (
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
    )
  }

  const renderDetail = () => {
    if (!activeType) return null
    // The header switch is type-level and derived: on while any trigger of
    // the type is enabled. Gmail with nothing to draft (no connected account)
    // keeps it inert — the connect CTA is right below.
    const anyEnabled = activeTypeTriggers.some((trigger) => trigger.enabled)
    const masterSwitchDisabled =
      busy ||
      busyTypes.has(activeType) ||
      !canOperate ||
      (activeType === "gmail" &&
        activeTypeTriggers.length === 0 &&
        (gmailAccountsLoading || (gmailAccounts?.length ?? 0) === 0))

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between gap-3 border-b pb-4">
          <div className="flex min-w-0 items-center gap-2.5">
            <Button
              variant="ghost"
              size="sm"
              className="-ml-2 h-8 px-2 text-muted-foreground hover:text-foreground"
              onClick={handleBack}
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
            checked={anyEnabled}
            onCheckedChange={(checked) => void handleTypeToggle(activeType, checked)}
            disabled={masterSwitchDisabled}
          />
        </div>

        {editing ? renderEditor() : renderTriggerList()}
      </div>
    )
  }

  // The add/edit form for one trigger of the active type. Cancel/Save land
  // back on the type's manage list.
  const renderEditor = () => {
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
    const saveLabelKey =
      activeType === "webhook"
        ? "triggers.actions.saveWebhook"
        : activeType === "scheduled"
          ? "triggers.actions.saveSchedule"
          : "triggers.actions.saveSettings"
    // The account this Gmail trigger is bound to (drafts bind on selection).
    // While unbound — or bound to a since-disconnected account — the editor
    // shows the account picker instead of the avatar header.
    const boundGmailAccount =
      activeType === "gmail" && form.oauthAccountId
        ? ((gmailAccounts ?? []).find(
            (account) => String(account.id) === form.oauthAccountId,
          ) ?? null)
        : null

    return (
      <div className="space-y-4">
        {activeType === "gmail" && boundGmailAccount ? (
          // Bound-account header, like the reference design: the editor is
          // "this account's settings", not an anonymous form.
          <div className="flex items-center gap-2.5">
            <div
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-bold text-white"
              style={{ background: gmailAvatarColor(boundGmailAccount.id) }}
            >
              {gmailAvatarInitials(boundGmailAccount.email, boundGmailAccount.id)}
            </div>
            <div className="min-w-0 flex-1 truncate text-sm font-bold">
              {boundGmailAccount.email || `#${boundGmailAccount.id}`}
            </div>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-7 shrink-0 px-2 text-xs text-muted-foreground hover:text-foreground"
              onClick={() => {
                gmailAccountClearedByUserRef.current = true
                setFormValue("oauthAccountId", "")
              }}
            >
              {t("triggers.gmail.changeAccount")}
            </Button>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-sm font-semibold">
            {renderTypeIcon(activeType, "h-4 w-4 text-muted-foreground")}
            {t(EDITOR_TITLE_KEYS[activeType][isNew ? "create" : "edit"] as never)}
          </div>
        )}

        {activeType !== "gmail" && (
          <div className="space-y-2">
            <Label className={FIELD_LABEL_CLASS} htmlFor="trigger-name">
              {activeType === "scheduled" ? t("triggers.schedule.nameLabel") : t("triggers.form.name")}
            </Label>
            <Input
              id="trigger-name"
              value={form.name}
              maxLength={200}
              onChange={(event) => setFormValue("name", event.target.value)}
              placeholder={
                activeType === "scheduled"
                  ? t("triggers.form.scheduleNamePlaceholder")
                  : t("triggers.form.webhookNamePlaceholder")
              }
            />
          </div>
        )}

        {activeType === "webhook" && (
          <>
            {renderPromptField("triggers.form.webhookPrompt", "triggers.form.webhookPromptPlaceholder")}

            <div className="space-y-2">
              <Label className={FIELD_LABEL_CLASS} htmlFor="trigger-secret">{t("triggers.form.secret")}</Label>
              <div className="flex gap-2">
                <Input
                  id="trigger-secret"
                  value={form.secret}
                  onChange={(event) => {
                    setSecretGenerated(false)
                    setFormValue("secret", event.target.value)
                  }}
                  placeholder={
                    blankSecretKeepsCurrent
                      ? t("triggers.form.secretEditPlaceholder")
                      : t("triggers.form.secretPlaceholder")
                  }
                  className="flex-1 font-mono"
                />
                <Button type="button" variant="outline" className="shrink-0" onClick={handleGenerateSecret}>
                  <Wand2 className="mr-2 h-4 w-4" />
                  {t("triggers.form.generateSecret")}
                </Button>
              </div>
              {secretGenerated && (
                <p className="text-xs text-amber-600 dark:text-amber-400">
                  {t("triggers.form.secretGeneratedHint")}
                </p>
              )}
            </div>

            {!isStaging && selectedTrigger?.type === "webhook" ? (
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
            ) : (
              <Alert className="border-primary/20 bg-primary/5">
                <Info className="h-4 w-4" />
                <AlertDescription className="text-sm text-foreground">
                  {isStaging ? t("triggers.staging.webhookPending") : t("triggers.webhook.pendingSave")}
                </AlertDescription>
              </Alert>
            )}
          </>
        )}

        {activeType === "scheduled" && (
          <>
            <ScheduleFields
              value={form}
              // ScheduleFieldsValue's keys are a subset of TriggerFormState's
              // with matching types, but TS can't prove that across two
              // differently-constrained generics — this adapter is a thin,
              // known-safe pass-through.
              onChange={(key, value) => setFormValue(key, value as TriggerFormState[typeof key])}
              t={t}
              locale={locale}
            />

            {renderPromptField("triggers.form.schedulePrompt", "triggers.form.schedulePromptPlaceholder")}

            <div className="flex items-center gap-2 rounded-lg border border-primary/20 bg-primary/5 px-3 py-2.5 text-sm">
              <CalendarCheck className="h-4 w-4 shrink-0 text-primary" />
              <span>{summarizeSchedule(form, t, locale)}</span>
            </div>
            {scheduleWillFireImmediately(form, loadedScheduleFields) && (
              <p className="text-xs text-amber-600 dark:text-amber-400">
                {t("triggers.schedule.runsImmediatelyHint")}
              </p>
            )}
          </>
        )}

        {activeType === "gmail" && (
          <>
            {!boundGmailAccount && (
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

            <Collapsible open={gmailFiltersOpen} onOpenChange={setGmailFiltersOpen}>
              <CollapsibleTrigger asChild>
                <button
                  type="button"
                  className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground transition-colors hover:text-foreground"
                >
                  <ChevronRight className={cn("h-3.5 w-3.5 transition-transform", gmailFiltersOpen && "rotate-90")} />
                  {t("triggers.gmail.optionalFilters")}
                </button>
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-3 grid gap-4 sm:grid-cols-2">
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
              </CollapsibleContent>
            </Collapsible>

            {renderPromptField(
              "triggers.form.gmailPrompt",
              "triggers.form.gmailPromptPlaceholder",
              "triggers.form.gmailPromptHelp",
            )}

            {renderGmailConnectionAlert()}
          </>
        )}

        {renderSecretReveal()}

        <div className="flex flex-wrap items-center justify-between gap-2 border-t pt-4">
          <div className="flex flex-wrap items-center gap-2">
            {/* Always usable, like the reference design (whose schedule
                editor has no test button). A live unsaved draft is saved
                automatically before the test fires; a staged trigger (no
                agent yet) renders its test locally instead. */}
            {activeType !== "scheduled" && (
              <Button
                variant="outline"
                onClick={handleTest}
                disabled={busy || testing}
              >
                {testing ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Play className="mr-2 h-4 w-4" />
                )}
                {testing ? t("triggers.test.running") : t("triggers.actions.test")}
              </Button>
            )}
            {!isStaging && selectedTrigger?.type === "webhook" && (
              <Button variant="outline" onClick={handleRotateSecret} disabled={busy}>
                <RotateCcw className="mr-2 h-4 w-4" />
                {t("triggers.actions.rotateSecret")}
              </Button>
            )}
          </div>
          <div className="ml-auto flex items-center gap-2">
            <Button type="button" variant="outline" onClick={handleCancel} disabled={busy}>
              {t("common.cancel")}
            </Button>
            <Button
              type="button"
              onClick={handleSave}
              // Also gated on busyTypes: the type-level header switch above
              // PATCHes `enabled` asynchronously and doesn't update
              // selectedTrigger.enabled until that resolves — Save reads
              // selectedTrigger.enabled synchronously (see buildPayload), so
              // clicking Save while that PATCH is still in flight would race
              // it with a stale enabled value.
              disabled={busy || busyTypes.has(activeType)}
            >
              {busy && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {t(saveLabelKey as never)}
            </Button>
          </div>
        </div>

        {isStaging && stagedTestRun && (
          <section className="space-y-3 rounded-lg border p-4">
            <div>
              <h3 className="text-sm font-medium">{t("triggers.runs.title")}</h3>
              <p className="text-xs text-muted-foreground">
                {t("triggers.test.stagedPreviewNote")}
              </p>
            </div>
            <div className="flex items-center gap-3 rounded-md bg-muted/40 px-3 py-2 text-sm">
              <span className={cn("font-medium", runStatusClass("completed"))}>
                {t("triggers.runStatus.completed")}
              </span>
              <span className="min-w-0 flex-1 truncate font-mono text-xs text-muted-foreground">
                {stagedTestRun.id}
              </span>
            </div>
            <div className="space-y-1.5">
              <div className={FIELD_LABEL_CLASS}>{t("triggers.test.stagedPromptLabel")}</div>
              <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded-md bg-muted/50 px-3 py-2 font-mono text-xs text-muted-foreground">
                {stagedTestRun.prompt}
              </pre>
            </div>
          </section>
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
          {activeType ? renderDetail() : renderOverview()}
        </div>

        <DialogFooter className="border-t px-5 py-3.5">
          <Button onClick={handleDone} disabled={busy}>
            {busy && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {t("common.done")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
