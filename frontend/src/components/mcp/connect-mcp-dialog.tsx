import React, { useState } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { SearchInput } from "@/components/ui/search-input"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { getApiUrl } from "@/lib/utils"
import {
  Loader2,
  LayoutTemplate,
  Link2,
  Globe,
  Home,
  CheckCircle2,
  LayoutGrid,
  Users,
  MessageSquare,
  LifeBuoy,
  Megaphone,
  Calendar,
  CreditCard,
  BarChart3,
  Plug,
  Zap,
  Settings,
  Trash2,
  Plus,
} from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { useAuth } from "@/contexts/auth-context"
import { useMcpApps } from "@/contexts/mcp-apps-context"
import { apiRequest } from "@/lib/api-wrapper"
import { toast } from "@/components/ui/sonner"
import { Input } from "@/components/ui/input"
import { Select } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { useEffect } from "react"
import { sanitizeAppIntegrations } from "@/lib/team-sharing-sanitizers"

import {
  isValidMcpName,
  parseMcpOAuthErrorMessage,
  MCP_OAUTH_POPUP_WINDOW_NAME,
  type McpOAuthConnectResponse,
  buildCustomApiPayload,
  buildMcpServerPayload,
  customApiDetailToEditState,
  mcpServerDetailToEditState,
  parseCustomApiDetail,
  parseMcpServerDetail,
  type CustomApiDetail,
  type McpServerDetail,
} from "@/lib/mcp-utils"

// Matches the backend mask; a masked value submitted unchanged keeps the stored secret.
const MASKED_SECRET_VALUE = "********"

// Upper bound on a catalog connect POST (connectCatalogApp below). Without
// this, a hung request left catalogConnectsInFlight stuck above zero forever
// — silently blocking every future dialog close with no explanation
// (round-6 MAJOR-1).
const CATALOG_CONNECT_TIMEOUT_MS = 30_000

// A connector's (type, numeric id) for the /api/connectors sharing endpoints.
// Only connected connectors carry a numeric server_id; catalog entries without
// one can't be shared/statused, so return null.
const connectorRef = (app: unknown) =>
  app !== null &&
  typeof app === "object" &&
  Number.isInteger((app as { server_id?: unknown }).server_id)
    ? {
        type:
          (app as { transport?: unknown }).transport === "custom_api" ? "custom_api" : "mcp",
        id: (app as { server_id: number }).server_id,
      }
    : null

export type { AppIntegration } from "./types"
import type { AppIntegration } from "./types"

import { OfficialMcpSettingsDialog } from "./official-mcp-settings-dialog"
import { CustomApiForm, MCPServerFormData } from "./custom-api-form"
import { CustomMcpForm } from "./custom-mcp-form"
import {
  getRuntimeConfigError,
  type RuntimeConfigErrorKey,
} from "./runtime-inputs-form"

interface ConnectMcpDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onConnectCustom?: () => void
  selectedMcpServers?: string[]
  onConnectSelected?: (selectedApps: string[]) => void
  customContent?: React.ReactNode
  onSuccess?: () => void
}

export function ConnectMcpDialog({
  open,
  onOpenChange,
  onConnectCustom,
  selectedMcpServers = [],
  onConnectSelected,
  customContent,
  onSuccess
}: ConnectMcpDialogProps) {
  const { t } = useI18n()
  const { token, inTeam } = useAuth()
  const { apps: officialApps } = useMcpApps()
  const [searchQuery, setSearchQuery] = useState("")
  const [debouncedSearch, setDebouncedSearch] = useState("")
  // Per-app-id in-flight set, not a single shared slot: every connect
  // mechanism (keyless, key-based, mcp_oauth, builtin_oauth) reads/clears
  // this by app id, and a single `string | null` slot let one app's cleanup
  // clobber another app's in-flight state whenever two connects overlapped
  // across different mechanisms — reported three times (round 3 MINOR-6,
  // round 4, round 5 M1) because per-call-site match-guards on a shared slot
  // don't compose; this retires all of them at once.
  const [loadingApps, setLoadingApps] = useState<Set<string>>(new Set())
  const markAppLoading = (appId: string) => {
    setLoadingApps(prev => (prev.has(appId) ? prev : new Set(prev).add(appId)))
  }
  const clearAppLoading = (appId: string) => {
    setLoadingApps(prev => {
      if (!prev.has(appId)) return prev
      const next = new Set(prev)
      next.delete(appId)
      return next
    })
  }
  const [isLoadingApps, setIsLoadingApps] = useState(false)
  const [activeCategory, setActiveCategory] = useState("All")
  const [activeLocation, setActiveLocation] = useState("remote")
  const [activeStatus, setActiveStatus] = useState("all")
  const [apps, setApps] = useState<AppIntegration[]>([])
  const [selectedApp, setSelectedApp] = useState<AppIntegration | null>(null)
  // Key-based (non-oauth) catalog connect: only the required secret(s) are editable.
  const [connectingKeyApp, setConnectingKeyApp] = useState<AppIntegration | null>(null)
  const [keyEnvValues, setKeyEnvValues] = useState<Record<string, string>>({})
  const [keyEnvSource, setKeyEnvSource] = useState<"own" | "shared" | "platform">("own")
  const [isConnectingKey, setIsConnectingKey] = useState(false)
  // Number of catalog connect POSTs (key-based or keyless) currently in
  // flight — deliberately narrower than loadingApps, which also stays set for
  // the whole builtin-OAuth popup wait (up to 5 minutes): the main dialog's
  // close guard reads this, and must not lock the dialog for that long.
  // A counter, not a boolean: with overlapping requests a boolean is cleared
  // by the first finally while the second is still pending, silently
  // reopening the guard window.
  const [catalogConnectsInFlight, setCatalogConnectsInFlight] = useState(0)
  // Ownership choice at connect/create; "team" triggers a share call after success.
  const [shareChoice, setShareChoice] = useState<"private" | "team">("private")
  const [localSelectedServers, setLocalSelectedServers] = useState<string[]>([])
  const [activeTab, setActiveTab] = useState("library")
  const [editingCustomServerId, setEditingCustomServerId] = useState<number | null>(null)
  const [customApiEditBaseline, setCustomApiEditBaseline] = useState<CustomApiDetail | null>(null)
  const [mcpEditBaseline, setMcpEditBaseline] = useState<McpServerDetail | null>(null)
  const connectorEditRequestRef = React.useRef(0)
  // Popup-closed poll intervals for in-flight mcp_oauth/builtin_oauth
  // connects, so they can be cleared on unmount or dialog-close instead of
  // leaking (N3: the poll otherwise has no unmount cleanup, unlike
  // custom-mcp-form's equivalent).
  const mcpOauthPollTimersRef = React.useRef<Set<number>>(new Set())
  // The builtin_oauth flow's postMessage listener, tracked the same way so a
  // dialog-close or unmount removes it too — otherwise a genuinely
  // successful auth completed after the user closed the dialog would still
  // fire onSuccess/auto-select via a listener nothing else was going to
  // remove (its own interval, the only thing that used to remove it, is
  // itself cleared by the same close/unmount cleanup).
  const mcpOauthMessageListenersRef = React.useRef<Set<(event: MessageEvent) => void>>(new Set())
  // F5: this component instance is commonly kept mounted by the parent
  // across dialog open/close (only the Radix Dialog's own content
  // unmounts), so the unmount-only cleanup above never fires on an ordinary
  // close. Checked after every await in handleConnectMcpOAuthApp so an
  // in-flight connect POST that resolves after the dialog (or the whole
  // component) is gone doesn't register a poll nothing will ever clear.
  const isMountedRef = React.useRef(true)

  // Custom MCP Server state
  const [isSavingCustom, setIsSavingCustom] = useState(false)
  const [customApiEnv, setCustomApiEnv] = useState<{ key: string, value: string }[]>([{ key: "", value: "" }])
  const [mcpFormData, setMcpFormData] = useState<MCPServerFormData>({
    name: "",
    transport: "stdio",
    description: "",
    config: {} as Record<string, any>
  })
  const [runtimeValidationError, setRuntimeValidationError] = useState<RuntimeConfigErrorKey | null>(null)

  const isAppConnected = (app: AppIntegration) => Boolean(app.is_connected)

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(searchQuery), 300)
    return () => clearTimeout(timer)
  }, [searchQuery])

  const loadApps = async () => {
    setIsLoadingApps(true)
    try {
      const params = new URLSearchParams()
      if (debouncedSearch) params.append("search", debouncedSearch)
      if (activeCategory && activeCategory !== "All") params.append("category", activeCategory)
      if (activeLocation) params.append("location", activeLocation)
      if (activeStatus === "verified") params.append("status", "verified")

      const response = await apiRequest(`${getApiUrl()}/api/mcp/apps?${params.toString()}`)
      if (response.ok) {
        const data = sanitizeAppIntegrations(await response.json())
        setApps(data)
        void loadSharingStatus(data)
      }
    } catch (error) {
      console.error("Failed to load apps:", error)
    } finally {
      setIsLoadingApps(false)
    }
  }

  // Batch-fetch team-sharing status for connected connectors and merge it into
  // the apps list so cards/settings can show Shared/Private/Needs-config.
  const loadSharingStatus = async (list: AppIntegration[]) => {
    // /api/connectors/status is an overlay-only route; standalone has no team
    // sharing, so skip it entirely when the user is not in a team.
    if (!inTeam) return
    const refs = list.map(connectorRef).filter((r): r is { type: string; id: number } => r !== null)
    if (refs.length === 0) return
    try {
      const response = await apiRequest(`${getApiUrl()}/api/connectors/status`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refs }),
      })
      if (!response.ok) return
      const status: Record<string, { shared: boolean; is_owner: boolean; needs_config: boolean }> =
        await response.json()
      setApps((prev) =>
        prev.map((app) => {
          const ref = connectorRef(app)
          const s = ref ? status[`${ref.type}:${ref.id}`] : undefined
          return s ? { ...app, shared: s.shared, is_owner: s.is_owner, needs_config: s.needs_config } : app
        }),
      )
    } catch (error) {
      console.error("Failed to load connector sharing status:", error)
    }
  }

  // Share a just-connected/created connector with the team. Returns false (and
  // toasts) when the caller isn't the owner/admin (403), so callers don't fail
  // silently — the connector simply stays private.
  const shareConnector = async (ref: { type: string; id: number }): Promise<boolean> => {
    try {
      // Round-7: callers await this from inside connectCatalogApp's success
      // branch, ahead of the finally that clears catalogConnectsInFlight — an
      // unbounded hang here would wedge the dialog's close guard open forever,
      // one layer below the timeout that already bounds the connect POST
      // itself. Same bound, so a hang here fails exactly as visibly.
      const response = await apiRequest(`${getApiUrl()}/api/connectors/${ref.type}/${ref.id}/share`, {
        method: "POST",
        signal: AbortSignal.timeout(CATALOG_CONNECT_TIMEOUT_MS),
      })
      if (response.ok) return true
      toast.error(
        response.status === 403
          ? t("tools.mcp.sharing.onlyOwnerOrAdmin")
          : t("tools.mcp.alerts.saveFailed"),
      )
      return false
    } catch (error) {
      console.error("Failed to share connector:", error)
      // Round-8 N4: a share timeout must not claim the *connection* timed
      // out — by the time this runs, the connect POST already succeeded and
      // its success toast already fired. Telling the user to "try again"
      // would prompt them to retry an already-established connection.
      const timedOut = error instanceof DOMException && error.name === "TimeoutError"
      toast.error(timedOut ? t('tools.mcp.alerts.shareTimedOut') : t("tools.mcp.alerts.saveFailed"))
      return false
    }
  }

  // Shared by the dialog-close branch below and the unmount cleanup effect:
  // stop any in-flight OAuth popup poll and remove the builtin_oauth flow's
  // postMessage listener (F6) — leaving the listener attached would let a
  // genuinely successful auth, completed after the user closed the dialog,
  // still fire onSuccess/auto-select via a listener the (now-cleared)
  // interval was the only thing that used to remove.
  const clearMcpOauthPollState = () => {
    mcpOauthPollTimersRef.current.forEach((timer) => window.clearInterval(timer))
    mcpOauthPollTimersRef.current.clear()
    mcpOauthMessageListenersRef.current.forEach((listener) =>
      window.removeEventListener('message', listener)
    )
    mcpOauthMessageListenersRef.current.clear()
    // This teardown abandons every timer/listener tracked above regardless
    // of which app(s) they belonged to, so it clears every in-flight app,
    // not just one.
    setLoadingApps(new Set())
  }

  useEffect(() => {
    if (open) {
      setMcpFormData({
        name: "",
        transport: "stdio",
        description: "",
        config: {},
        user_env: {},
        can_edit_global: true
      })
      setLocalSelectedServers(selectedMcpServers || [])
      setActiveTab("library")
      setEditingCustomServerId(null)
      setCustomApiEditBaseline(null)
      setMcpEditBaseline(null)
      setRuntimeValidationError(null)
      setShareChoice("private")
    } else {
      connectorEditRequestRef.current += 1
      // F6: closing the dialog (not just unmounting it) must also stop any
      // in-flight OAuth popup poll — otherwise it can later fire
      // onSuccess/auto-select against a closed dialog, and leaves a stale
      // spinner on the card if the dialog is reopened.
      clearMcpOauthPollState()
    }
  }, [open, t, selectedMcpServers])

  useEffect(() => () => {
    connectorEditRequestRef.current += 1
  }, [])

  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
    }
  }, [])

  useEffect(() => () => {
    clearMcpOauthPollState()
  }, [])

  useEffect(() => {
    if (open) {
      loadApps()
    }
  }, [open, debouncedSearch, activeCategory, activeLocation, activeStatus])

  const openCustomApiEditor = async (serverId: number): Promise<boolean> => {
    const requestId = connectorEditRequestRef.current + 1
    connectorEditRequestRef.current = requestId

    try {
      const response = await apiRequest(`${getApiUrl()}/api/custom-apis/${serverId}`)
      if (!response.ok) throw new Error(`Custom API detail request failed (${response.status})`)

      const detail = parseCustomApiDetail(await response.json())
      if (detail.id !== serverId) throw new Error("Custom API detail response ID mismatch")
      if (requestId !== connectorEditRequestRef.current || !open) return false

      const editState = customApiDetailToEditState(detail)
      setSelectedApp(null)
      setEditingCustomServerId(detail.id)
      setCustomApiEditBaseline(detail)
      setMcpEditBaseline(null)
      setRuntimeValidationError(null)
      setMcpFormData(editState.formData)
      setCustomApiEnv(editState.env)
      setActiveTab("custom_api")
      return true
    } catch (error) {
      if (requestId !== connectorEditRequestRef.current || !open) return false
      console.error("Failed to load Custom API detail:", error)
      toast.error(t('tools.mcp.dialog.customApiDetailFetchError'))
      return false
    }
  }

  const openMcpServerEditor = async (serverId: number): Promise<boolean> => {
    const requestId = connectorEditRequestRef.current + 1
    connectorEditRequestRef.current = requestId

    try {
      const response = await apiRequest(`${getApiUrl()}/api/mcp/servers/${serverId}`)
      if (!response.ok) throw new Error(`MCP server detail request failed (${response.status})`)

      const detail = parseMcpServerDetail(await response.json())
      if (detail.id !== serverId) throw new Error("MCP server detail response ID mismatch")
      if (requestId !== connectorEditRequestRef.current || !open) return false

      const editState = mcpServerDetailToEditState(detail)
      setSelectedApp(null)
      setEditingCustomServerId(detail.id)
      setCustomApiEditBaseline(null)
      setMcpEditBaseline(detail)
      setRuntimeValidationError(null)
      setMcpFormData(editState.formData)
      setActiveTab("custom")
      return true
    } catch (error) {
      if (requestId !== connectorEditRequestRef.current || !open) return false
      console.error("Failed to load MCP server detail:", error)
      toast.error(t('tools.mcp.dialog.mcpDetailFetchError'))
      return false
    }
  }

  // Shared close path for every "close this dialog" action — the Dialog's
  // own Escape/outside-click dismissal AND every Cancel/footer button, which
  // previously called the raw onOpenChange(false) prop directly and so
  // bypassed the in-flight guard entirely (round-6 MAJOR-1): starting a
  // catalog connect on the Library tab, then switching to the Custom API/MCP
  // tab and hitting Cancel (or successfully saving), closed the dialog out
  // from under a still-pending POST. Anything that wants to close this
  // dialog must call this, not the raw onOpenChange prop.
  const requestClose = () => {
    // Don't let a close action go through while a catalog connect POST is in
    // flight, or its success/error toast fires after the user already
    // believes they dismissed it. Scoped to the POST itself (not
    // loadingApps, which also spans OAuth popup waits).
    if (catalogConnectsInFlight > 0) {
      // Round-9: Escape, outside-click, and the header X all route through
      // this same function (see the Dialog's onOpenChange below), so a
      // blocked attempt previously no-op'd with zero feedback for up to the
      // full connect timeout — the only in-flight signal is a small per-card
      // spinner that's easy to miss. Surface it explicitly instead.
      toast.warning(t('tools.mcp.alerts.closeBlockedWhileConnecting'))
      return
    }
    connectorEditRequestRef.current += 1
    setCustomApiEditBaseline(null)
    setMcpEditBaseline(null)
    onOpenChange(false)
    setRuntimeValidationError(null)
  }

  const handleSaveCustomMcp = async () => {
    if (!mcpFormData.name.trim()) {
      toast.error(t('tools.mcp.alerts.nameRequired'))
      return
    }

    if (!isValidMcpName(mcpFormData.name)) {
      toast.error(t('tools.mcp.alerts.nameInvalidFormat') || "Name can only contain letters, numbers, hyphens and underscores");
      return;
    }

    const formPayload = { ...mcpFormData };
    let payload: object = formPayload;
    let url = "";
    const method = editingCustomServerId ? 'PUT' : 'POST';
    const connectorType = formPayload.transport === "custom_api" ? "custom_api" : "mcp";
    const runtimeError = runtimeValidationError || getRuntimeConfigError(formPayload, connectorType);
    if (runtimeError) {
      toast.error(t(runtimeError));
      return;
    }

    if (formPayload.transport === "custom_api") {
      if (!mcpFormData.url?.trim()) {
        toast.error(t('tools.mcp.alerts.urlRequired'));
        return;
      }

      if (editingCustomServerId && !customApiEditBaseline) {
        toast.error(t('tools.mcp.dialog.customApiDetailFetchError'))
        return
      }
      const buildResult = buildCustomApiPayload(
        payload,
        customApiEnv,
        editingCustomServerId ? customApiEditBaseline ?? undefined : undefined,
      );
      if (!buildResult.isValid) {
        toast.error(t(buildResult.errorKey || 'tools.mcp.alerts.atLeastOneSecret'));
        return;
      }
      payload = buildResult.payload;

      url = editingCustomServerId
        ? `${getApiUrl()}/api/custom-apis/${editingCustomServerId}`
        : `${getApiUrl()}/api/custom-apis`;

      setIsSavingCustom(true)
      try {
        const response = await apiRequest(url, {
          method,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        })
        await handleSaveResponse(response);
      } catch (error) {
        console.error("Failed to save custom API:", error)
        toast.error(t('tools.mcp.alerts.saveFailed'))
        setIsSavingCustom(false)
      }
      return;
    }

    // Regular MCP logic
    if (editingCustomServerId && !mcpEditBaseline) {
      toast.error(t('tools.mcp.dialog.mcpDetailFetchError'))
      return
    }
    payload = buildMcpServerPayload(
      formPayload,
      editingCustomServerId ? mcpEditBaseline ?? undefined : undefined,
    )
    setIsSavingCustom(true)
    try {
      url = editingCustomServerId
        ? `${getApiUrl()}/api/mcp/servers/${editingCustomServerId}`
        : `${getApiUrl()}/api/mcp/servers`

      const response = await apiRequest(url, {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      })
      await handleSaveResponse(response);
    } catch (error) {
      console.error("Failed to save custom MCP server:", error)
      toast.error(t('tools.mcp.alerts.saveFailed'))
      setIsSavingCustom(false)
    }
  }

  const handleSaveResponse = async (response: any) => {
    if (response.ok) {
      toast.success(t('tools.mcp.buttons.save'))
      // Newly created self-owned connector + "Share with team" chosen: share it.
      // Skip on edit (the connector already exists and keeps its sharing state).
      if (!editingCustomServerId && shareChoice === "team") {
        try {
          const created = await response.json()
          const type = mcpFormData.transport === "custom_api" ? "custom_api" : "mcp"
          if (Number.isInteger(created?.id)) await shareConnector({ type, id: created.id })
        } catch (error) {
          console.error("Failed to read created connector for sharing:", error)
        }
      }
      if (onSuccess) onSuccess()
      loadApps()

      // If in select mode (agent builder), switch to local tab and select the new server
      if (isSelectMode) {
        if (!editingCustomServerId) {
          const newServerName = mcpFormData.name;
          setLocalSelectedServers(prev => prev.includes(newServerName) ? prev : [...prev, newServerName]);
          setActiveLocation("local");
        }
        setActiveTab("library");
      } else {
        // If in standalone tools page, just close the dialog
        requestClose();
      }

      setEditingCustomServerId(null)
      setCustomApiEditBaseline(null)
      setMcpEditBaseline(null)
      setMcpFormData({ name: "", transport: "stdio", description: "", config: {}, user_env: {}, can_edit_global: true })
    } else {
      const error = await response.json()
      toast.error(error.detail || t('tools.mcp.alerts.saveFailed'))
    }
    setIsSavingCustom(false)
  }

  const isSelectMode = !!onConnectSelected;

  // Key-based (non-oauth) catalog app: collect only the required secret(s), then
  // POST to the connect endpoint (command/args/description come from the catalog,
  // not the user). Users can never edit the shared server config this way.
  const openKeyConnect = (app: AppIntegration) => {
    const required = app.launch_config?.required_env || []
    const initial: Record<string, string> = {}
    // Pre-fill masked when the user already has a key, so submitting without
    // retyping preserves it (the backend restores masked values) instead of
    // silently clearing it.
    required.forEach((k) => { initial[k] = app.user_env_configured ? MASKED_SECRET_VALUE : "" })
    setKeyEnvValues(initial)
    // Default the source selector to the user's current pick — but only if that
    // source is still available (a stored "shared"/"platform" can go away when
    // its key is removed, and its radio would then not render). Else fall back to
    // whichever option is usable, preferring "own" when they already have a key.
    const defaultSource: "own" | "shared" | "platform" =
      (app.env_source === "shared" && app.shared_env_available ? "shared" : null)
        || (app.env_source === "platform" && app.platform_env_available ? "platform" : null)
        || (app.env_source === "own" ? "own" : null)
        || (app.user_env_configured ? "own" : null)
        || (app.shared_env_available ? "shared" : null)
        || (app.platform_env_available ? "platform" : null)
        || "own"
    setKeyEnvSource(defaultSource)
    setShareChoice("private")
    setConnectingKeyApp(app)
  }

  // Shared POST to /apps/{id}/connect for both catalog-connect paths (key-based
  // and keyless): same request shape, same success/error/loading handling.
  // Callers own their own loading-state setter and any path-specific
  // post-success work (closing their own dialog state, sharing with team) via
  // the onSuccess callback, which receives the raw response so it can still
  // read the connected server's id when needed.
  const connectCatalogApp = async (
    app: AppIntegration,
    body: Record<string, unknown>,
    options: {
      autoSelect: boolean
      setLoading: (loading: boolean) => void
      onSuccess?: (response: Response) => Promise<void> | void
    }
  ) => {
    options.setLoading(true)
    setCatalogConnectsInFlight(count => count + 1)
    try {
      // Bounded so a hung request can't wedge the dialog's close guard
      // (round-6 MAJOR-1) indefinitely — apiRequest forwards this signal
      // straight through to fetch, no shared-wrapper change needed.
      const response = await apiRequest(`${getApiUrl()}/api/mcp/apps/${app.id}/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(CATALOG_CONNECT_TIMEOUT_MS),
      })
      if (response.ok) {
        toast.success(t('tools.mcp.dialog.connectSuccess', { name: app.name }))
        // Path-specific follow-up runs before the loadApps() refresh below so
        // its effects (e.g. the key path's team share) are already reflected
        // in the reloaded list.
        if (options.onSuccess) await options.onSuccess(response)
        if (options.autoSelect && onConnectSelected) {
          setLocalSelectedServers(prev => prev.includes(app.name) ? prev : [...prev, app.name])
        }
        if (onSuccess) onSuccess()
        loadApps()
        // Same clobber risk as loadingApps above: selectedApp is shared
        // across the whole catalog too. If the user closed this app's
        // settings and opened a different app while this request was still
        // in flight, selectedApp now points at that other app — clearing it
        // unconditionally would close whatever the user is currently
        // looking at instead of just this (already-closed) dialog.
        setSelectedApp(current => current?.id === app.id ? null : current)
      } else {
        const error = await response.json()
        toast.error(error.detail || t('tools.mcp.alerts.saveFailed'))
      }
    } catch (error) {
      console.error("Failed to connect app:", error)
      const timedOut = error instanceof DOMException && error.name === "TimeoutError"
      toast.error(timedOut ? t('tools.mcp.alerts.connectTimedOut') : t('tools.mcp.alerts.saveFailed'))
    } finally {
      options.setLoading(false)
      setCatalogConnectsInFlight(count => count - 1)
    }
  }

  const submitKeyConnect = async (autoSelect: boolean) => {
    if (!connectingKeyApp) return
    const app = connectingKeyApp
    // Only the "own" source sends a per-user key. For shared/platform we omit
    // env entirely (undefined drops from the JSON) so the backend leaves the
    // stored own key untouched — an empty {} would clear it, forcing re-entry
    // when switching back to "own".
    const env = keyEnvSource === "own" ? keyEnvValues : undefined
    await connectCatalogApp(app, { env, env_source: keyEnvSource }, {
      autoSelect,
      setLoading: setIsConnectingKey,
      onSuccess: async (response) => {
        // "Share with team" chosen: catalog connect users are usually not the
        // owner, so shareConnector toasts a clear 403 message and stays private.
        if (shareChoice === "team") {
          try {
            const connected = await response.json()
            if (Number.isInteger(connected?.id)) await shareConnector({ type: "mcp", id: connected.id })
          } catch (error) {
            console.error("Failed to read connected app for sharing:", error)
          }
        }
        setConnectingKeyApp(null)
      },
    })
  }

  // Keyless catalog app (e.g. Chrome): no secrets to collect, so skip the key
  // dialog entirely and POST straight to the connect endpoint. is_active is
  // sent explicitly (not omitted) so re-connecting after a dormant
  // association (is_active=false) reactivates it instead of silently staying
  // disconnected — the backend only flips is_active when told to.
  // The ref guard (round-8 N6) covers the one window the disabled buttons
  // can't: loadingApps is React state, so disabled={} lags a commit cycle,
  // and a double-click landing in the same tick would fire two POSTs. The
  // backend recovers idempotently, so this only saves a wasted duplicate
  // request — a ref is synchronous, closing the window outright.
  const keylessConnectsRef = React.useRef<Set<string>>(new Set())
  const submitKeylessConnect = async (app: AppIntegration, autoSelect: boolean) => {
    if (keylessConnectsRef.current.has(app.id)) return
    keylessConnectsRef.current.add(app.id)
    try {
      await connectCatalogApp(app, { is_active: true }, {
        autoSelect,
        setLoading: (loading) => (loading ? markAppLoading(app.id) : clearAppLoading(app.id)),
      })
    } finally {
      keylessConnectsRef.current.delete(app.id)
    }
  }

  // Remote-MCP OAuth catalog app (e.g. Granola): the backend ensures the
  // shared server row + this user's association, runs Dynamic Client
  // Registration when the provider has no static client, and returns the
  // authorization URL. Mirrors custom-mcp-form's handleConnectMcpOAuth,
  // but keyed by catalog app_id instead of a server id.
  const handleConnectMcpOAuthApp = async (app: AppIntegration, autoSelect: boolean) => {
    markAppLoading(app.id)
    // Open the popup synchronously on the click, before any await — popup
    // blockers reject windows opened outside direct user-gesture handling.
    // The window-features string matters: without it, browsers open a full
    // new tab instead of the small centered popup the builtin OAuth flow uses.
    const width = 600
    const height = 700
    const left = window.screenX + (window.outerWidth - width) / 2
    const top = window.screenY + (window.outerHeight - height) / 2
    const popup = window.open(
      "about:blank",
      MCP_OAUTH_POPUP_WINDOW_NAME,
      `width=${width},height=${height},left=${left},top=${top},scrollbars=yes`,
    )
    if (!popup) {
      toast.error("Popup blocked. Please allow popups for this site to connect.")
      clearAppLoading(app.id)
      return
    }
    popup.opener = null
    try {
      const response = await apiRequest(`${getApiUrl()}/api/mcp/apps/${app.id}/oauth/connect`, {
        method: "POST",
        credentials: "include",
        headers: {
          "Accept": "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ redirect_after: "/tools?tab=mcp" }),
      })
      if (!response.ok) {
        popup.close()
        const message = await parseMcpOAuthErrorMessage(response, t('tools.mcp.dialog.oauthConnectFailed'))
        toast.error(message)
        clearAppLoading(app.id)
        return
      }
      const data = await response.json() as McpOAuthConnectResponse
      if (!data.authorization_url) {
        popup.close()
        toast.error(t('tools.mcp.dialog.oauthConnectFailed'))
        clearAppLoading(app.id)
        return
      }
      popup.location.href = data.authorization_url
    } catch (error) {
      console.error("Failed to start MCP OAuth connect:", error)
      popup.close()
      toast.error(t('tools.mcp.dialog.oauthConnectFailed'))
      clearAppLoading(app.id)
      return
    }
    // F5: if the dialog closed (or this component unmounted) while the
    // connect POST was in flight, don't register a poll after the cleanup
    // effects have already run — it would never get cleared. The popup
    // itself is left navigating to the authorization URL either way; the
    // user can still complete auth in it, and the next apps refresh (e.g.
    // reopening the dialog) will reflect it.
    if (!isMountedRef.current) return

    // The popup's opener link is severed (popup.opener = null), so unlike the
    // builtin flow there is no postMessage channel — and a closed popup can
    // mean success OR a cancelled/denied/failed authorization (the error
    // redirect leaves the popup open for the user to read, then they close it
    // by hand). So on close, ask the backend which one actually happened and
    // gate the success actions on the app really being connected; is_connected
    // for mcp_oauth apps requires a completed grant, not just the association.
    const startedAt = Date.now()
    const maxWaitMs = 5 * 60 * 1000
    const checkPopup = window.setInterval(() => {
      const expired = Date.now() - startedAt >= maxWaitMs
      if (!popup.closed && !expired) return
      window.clearInterval(checkPopup)
      mcpOauthPollTimersRef.current.delete(checkPopup)
      clearAppLoading(app.id)
      if (!popup.closed) {
        // Timed out with the popup still open: stop the spinner. If the user
        // eventually finishes, the next apps refresh shows the connection.
        return
      }
      void (async () => {
        let connected = false
        try {
          const response = await apiRequest(`${getApiUrl()}/api/mcp/apps?location=remote`)
          if (response.ok) {
            const data = sanitizeAppIntegrations(await response.json())
            connected = data.some(
              (candidate) => candidate.id === app.id && candidate.is_connected,
            )
          }
        } catch (error) {
          console.error("Failed to refresh apps after the OAuth popup closed:", error)
        }
        loadApps()
        if (!connected) return
        if (onSuccess) onSuccess()
        if (autoSelect && onConnectSelected) {
          setLocalSelectedServers(prev => prev.includes(app.name) ? prev : [...prev, app.name])
        }
        // Same clobber class as connectCatalogApp above: only close this
        // app's settings dialog if the user hasn't since opened another one.
        setSelectedApp(current => current?.id === app.id ? null : current)
      })()
    }, 500)
    mcpOauthPollTimersRef.current.add(checkPopup)
  }

  const handleConnectApp = (app: AppIntegration, autoSelect: boolean = false) => {
    if (app.auth_type !== "builtin_oauth") {
      // Key-based catalog app: collect the key; keyless app: connect directly;
      // remote-MCP OAuth app: start the per-user OAuth (DCR) flow. Anything
      // else is a mis-authored entry.
      if (app.auth_type === "api_key") {
        openKeyConnect(app);
      } else if (app.auth_type === "keyless") {
        void submitKeylessConnect(app, autoSelect);
      } else if (app.auth_type === "mcp_oauth") {
        handleConnectMcpOAuthApp(app, autoSelect);
      } else {
        toast.error(t('tools.mcp.alerts.notConfigured'));
      }
      return;
    }

    const provider = app.provider;
    if (!provider) {
      // Mis-authored OAuth entry: transport says oauth but no provider to
      // build the auth URL. Fail clearly instead of opening a broken popup.
      toast.error(t('tools.mcp.alerts.providerNotDefined'));
      return;
    }

    markAppLoading(app.id)
    // Open OAuth in a popup window to handle the callback smoothly
    const width = 600;
    const height = 700;
    const left = window.screenX + (window.outerWidth - width) / 2;
    const top = window.screenY + (window.outerHeight - height) / 2;

    const authUrl = `${getApiUrl()}/api/auth/${provider}/login?token=${token || ''}&app_id=${app.id}&redirect=${encodeURIComponent(window.location.href)}`;
    const popup = window.open(
      authUrl,
      `${provider} OAuth`,
      `width=${width},height=${height},left=${left},top=${top},scrollbars=yes`
    );

    if (!popup) {
      toast.error("Popup blocked. Please allow popups for this site to connect.");
      clearAppLoading(app.id);
      return;
    }

    // Listen for the postMessage from the popup
    const handleMessage = (event: MessageEvent) => {
      if (event.data?.type === 'oauth-success') {
        clearAppLoading(app.id)
        window.removeEventListener('message', handleMessage)
        mcpOauthMessageListenersRef.current.delete(handleMessage)
        window.clearInterval(checkPopup)
        mcpOauthPollTimersRef.current.delete(checkPopup)

        loadApps();
        if (onSuccess) onSuccess();

        if (autoSelect && onConnectSelected) {
          // If it was just connected, it is not selected yet, so add it to local selection
          setLocalSelectedServers(prev => prev.includes(app.name) ? prev : [...prev, app.name]);
        }

        // Same clobber class as connectCatalogApp above: only close this
        // app's settings dialog if the user hasn't since opened another one.
        setSelectedApp(current => current?.id === app.id ? null : current);
      }
    };

    window.addEventListener('message', handleMessage);
    mcpOauthMessageListenersRef.current.add(handleMessage)

    // Fallback: check if popup was closed without success message, or give up
    // after a timeout (N3: this poll previously had neither an unmount-safe
    // cleanup nor a timeout cap, unlike the mcp_oauth handler above and
    // custom-mcp-form.tsx's equivalent flow).
    const startedAt = Date.now()
    const maxWaitMs = 5 * 60 * 1000
    const checkPopup: number = window.setInterval(() => {
      const expired = Date.now() - startedAt >= maxWaitMs
      if (!popup?.closed && !expired) return
      window.clearInterval(checkPopup);
      mcpOauthPollTimersRef.current.delete(checkPopup)
      window.removeEventListener('message', handleMessage);
      mcpOauthMessageListenersRef.current.delete(handleMessage)
      clearAppLoading(app.id);
    }, 500);
    mcpOauthPollTimersRef.current.add(checkPopup)
  }

  const handleCardClick = (app: AppIntegration, isGloballyConnected: boolean) => {
    if (isSelectMode && isGloballyConnected) {
      setLocalSelectedServers(prev =>
        prev.includes(app.name)
          ? prev.filter(name => name !== app.name)
          : [...prev, app.name]
      );
    } else {
      setSelectedApp(app);
    }
  }

  // Ownership radio shared by the three connect/create flows (key connect,
  // custom MCP, custom API). Hidden on edit — sharing is managed in settings.
  const ownershipRadio = editingCustomServerId || !inTeam ? null : (
    <div className="space-y-1.5">
      <Label>{t('tools.mcp.dialog.ownership.label')}</Label>
      <RadioGroup value={shareChoice} onValueChange={(v) => setShareChoice(v as "private" | "team")}>
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="private" id="ownership-private" />
          <Label htmlFor="ownership-private" className="font-normal cursor-pointer">
            {t('tools.mcp.dialog.ownership.private')}
          </Label>
        </div>
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="team" id="ownership-team" />
          <Label htmlFor="ownership-team" className="font-normal cursor-pointer">
            {t('tools.mcp.dialog.ownership.team')}
          </Label>
        </div>
      </RadioGroup>
    </div>
  )

  const selectedRemoteCount = localSelectedServers.filter(name =>
    officialApps.some(app => app.name.toLowerCase() === name.toLowerCase() || app.id.toLowerCase() === name.toLowerCase())
  ).length;
  const selectedLocalCount = localSelectedServers.length - selectedRemoteCount;

  return (
    <>
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        // Radix's own Escape/outside-click dismissal — route it through the
        // same requestClose() every explicit close action uses, so this is
        // the only place the guard logic lives.
        if (!nextOpen) {
          requestClose()
          return
        }
        onOpenChange(nextOpen)
      }}
    >
      <DialogContent className="sm:max-w-5xl md:max-w-6xl w-[95vw] h-[85vh] flex flex-col p-0 overflow-hidden gap-0 bg-slate-50">
        <DialogHeader className="px-6 py-4 border-b bg-white shrink-0 pr-10">
          <DialogTitle className="text-xl flex items-center gap-2 font-bold text-left">
            <Plug className="h-5 w-5 text-blue-600 shrink-0" /> {t('tools.mcp.dialog.connector')}
          </DialogTitle>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col overflow-hidden bg-white">
          <div className="px-6 border-b shrink-0 bg-white overflow-x-auto overflow-y-hidden">
            <TabsList className="bg-transparent h-14 p-0 border-b-0 space-x-6 min-w-max">
              <TabsTrigger
                value="library"
                className="data-[state=active]:bg-transparent data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-blue-600 data-[state=active]:text-blue-600 rounded-none h-full px-0 font-semibold flex items-center gap-2"
              >
                <LayoutTemplate className="h-4 w-4" /> {t('tools.mcp.dialog.browseLibrary')}
              </TabsTrigger>
              <TabsTrigger
                value="custom_api"
                className="data-[state=active]:bg-transparent data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-blue-600 data-[state=active]:text-blue-600 rounded-none h-full px-0 font-semibold flex items-center gap-2 text-slate-500"
                onClick={() => {
                  connectorEditRequestRef.current += 1
                  setEditingCustomServerId(null)
                  setCustomApiEditBaseline(null)
                  setMcpEditBaseline(null)
                  setRuntimeValidationError(null)
                  setMcpFormData({
                    name: "",
                    transport: "custom_api",
                    description: "",
                    config: { env: {} }
                  })
                  setCustomApiEnv([{ key: "", value: "" }])
                }}
              >
                <Globe className="h-4 w-4" /> {t('tools.mcp.dialog.customApi')}
              </TabsTrigger>
              <TabsTrigger
                value="custom"
                className="data-[state=active]:bg-transparent data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-blue-600 data-[state=active]:text-blue-600 rounded-none h-full px-0 font-semibold flex items-center gap-2 text-slate-500"
                onClick={(e) => {
                  connectorEditRequestRef.current += 1
                  if (onConnectCustom) {
                    e.preventDefault()
                    onConnectCustom()
                  } else {
                    setEditingCustomServerId(null)
                    setCustomApiEditBaseline(null)
                    setMcpEditBaseline(null)
                    setRuntimeValidationError(null)
                    setMcpFormData({
                      name: "",
                      transport: "stdio",
                      description: "",
                      config: {},
                      user_env: {},
                      can_edit_global: true
                    })
                  }
                }}
              >
                <Link2 className="h-4 w-4" /> {t('tools.mcp.dialog.customMcp')}
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="library" className="flex-1 overflow-hidden m-0 flex flex-col md:flex-row bg-slate-50/50">
            {/* Sidebar */}
            <div className="w-full md:w-56 shrink-0 border-r bg-slate-50/30 overflow-y-auto hidden md:block">
              <div className="p-4 space-y-6">
                <div>
                  <h4 className="text-xs font-bold tracking-wider text-slate-500 uppercase mb-3 px-2">{t('tools.mcp.dialog.location')}</h4>
                  <div className="space-y-1">
                    <button
                      className={`w-full flex items-center justify-between px-2 py-1.5 text-sm font-medium rounded-md ${activeLocation === 'remote' ? 'bg-blue-100/50 text-blue-700' : 'text-slate-600 hover:bg-slate-100'}`}
                      onClick={() => setActiveLocation('remote')}
                    >
                      <div className="flex items-center gap-3">
                        <Globe className="h-4 w-4" /> {t('tools.mcp.dialog.remote')}
                      </div>
                      {isSelectMode && selectedRemoteCount > 0 && (
                        <Badge variant="secondary" className="h-5 px-1.5 min-w-5 flex items-center justify-center bg-blue-100 text-blue-700 border-none">{selectedRemoteCount}</Badge>
                      )}
                    </button>
                    <button
                      className={`w-full flex items-center justify-between px-2 py-1.5 text-sm font-medium rounded-md ${activeLocation === 'local' ? 'bg-blue-100/50 text-blue-700' : 'text-slate-600 hover:bg-slate-100'}`}
                      onClick={() => setActiveLocation('local')}
                    >
                      <div className="flex items-center gap-3">
                        <Home className="h-4 w-4" /> {t('tools.mcp.dialog.local')}
                      </div>
                      {isSelectMode && selectedLocalCount > 0 && (
                        <Badge variant="secondary" className="h-5 px-1.5 min-w-5 flex items-center justify-center bg-blue-100 text-blue-700 border-none">{selectedLocalCount}</Badge>
                      )}
                    </button>
                  </div>
                </div>

                <div>
                  <h4 className="text-xs font-bold tracking-wider text-slate-500 uppercase mb-3 px-2">{t('tools.mcp.dialog.status')}</h4>
                  <div className="space-y-1">
                    <button
                      className={`w-full flex items-center gap-3 px-2 py-1.5 text-sm font-medium rounded-md ${activeStatus === 'verified' ? 'bg-blue-100/50 text-blue-700' : 'text-slate-600 hover:bg-slate-100'}`}
                      onClick={() => setActiveStatus(activeStatus === 'verified' ? 'all' : 'verified')}
                    >
                      <CheckCircle2 className="h-4 w-4" /> {t('tools.mcp.dialog.verified')}
                    </button>
                  </div>
                </div>

                <div>
                  <h4 className="text-xs font-bold tracking-wider text-slate-500 uppercase mb-3 px-2">{t('tools.mcp.dialog.categories')}</h4>
                  <div className="space-y-1">
                    <button
                      className={`w-full flex items-center gap-3 px-2 py-1.5 text-sm font-medium rounded-md ${activeCategory === 'All' ? 'bg-blue-100/50 text-blue-700' : 'text-slate-600 hover:bg-slate-100'}`}
                      onClick={() => setActiveCategory('All')}
                    >
                      <LayoutGrid className="h-4 w-4" /> {t('tools.mcp.dialog.all')}
                    </button>
                    <button
                      className={`w-full flex items-center gap-3 px-2 py-1.5 text-sm font-medium rounded-md ${activeCategory === 'CRM' ? 'bg-blue-100/50 text-blue-700' : 'text-slate-600 hover:bg-slate-100'}`}
                      onClick={() => setActiveCategory('CRM')}
                    >
                      <Users className="h-4 w-4" /> CRM
                    </button>
                    <button
                      className={`w-full flex items-center gap-3 px-2 py-1.5 text-sm font-medium rounded-md ${activeCategory === 'Communication' ? 'bg-blue-100/50 text-blue-700' : 'text-slate-600 hover:bg-slate-100'}`}
                      onClick={() => setActiveCategory('Communication')}
                    >
                      <MessageSquare className="h-4 w-4" /> Communication
                    </button>
                    <button
                      className={`w-full flex items-center gap-3 px-2 py-1.5 text-sm font-medium rounded-md ${activeCategory === 'Support' ? 'bg-blue-100/50 text-blue-700' : 'text-slate-600 hover:bg-slate-100'}`}
                      onClick={() => setActiveCategory('Support')}
                    >
                      <LifeBuoy className="h-4 w-4" /> Support
                    </button>
                    <button
                      className={`w-full flex items-center gap-3 px-2 py-1.5 text-sm font-medium rounded-md ${activeCategory === 'Marketing' ? 'bg-blue-100/50 text-blue-700' : 'text-slate-600 hover:bg-slate-100'}`}
                      onClick={() => setActiveCategory('Marketing')}
                    >
                      <Megaphone className="h-4 w-4" /> Marketing
                    </button>
                    <button
                      className={`w-full flex items-center gap-3 px-2 py-1.5 text-sm font-medium rounded-md ${activeCategory === 'Scheduling' ? 'bg-blue-100/50 text-blue-700' : 'text-slate-600 hover:bg-slate-100'}`}
                      onClick={() => setActiveCategory('Scheduling')}
                    >
                      <Calendar className="h-4 w-4" /> Scheduling
                    </button>
                    <button
                      className={`w-full flex items-center gap-3 px-2 py-1.5 text-sm font-medium rounded-md ${activeCategory === 'Payments' ? 'bg-blue-100/50 text-blue-700' : 'text-slate-600 hover:bg-slate-100'}`}
                      onClick={() => setActiveCategory('Payments')}
                    >
                      <CreditCard className="h-4 w-4" /> Payments
                    </button>
                    <button
                      className={`w-full flex items-center gap-3 px-2 py-1.5 text-sm font-medium rounded-md ${activeCategory === 'Analytics' ? 'bg-blue-100/50 text-blue-700' : 'text-slate-600 hover:bg-slate-100'}`}
                      onClick={() => setActiveCategory('Analytics')}
                    >
                      <BarChart3 className="h-4 w-4" /> Analytics
                    </button>
                    <button
                      className={`w-full flex items-center gap-3 px-2 py-1.5 text-sm font-medium rounded-md ${activeCategory === 'Operations' ? 'bg-blue-100/50 text-blue-700' : 'text-slate-600 hover:bg-slate-100'}`}
                      onClick={() => setActiveCategory('Operations')}
                    >
                      <Settings className="h-4 w-4" /> Operations
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Main Content Area */}
            <div className="flex-1 flex flex-col overflow-hidden bg-white">
              <div className="p-6 pb-2 shrink-0">
                <div className="mb-4 flex md:hidden items-center gap-2">
                  <button
                    className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 text-sm font-medium rounded-md border transition-colors ${activeLocation === 'remote'
                      ? 'bg-blue-50 text-blue-700 border-blue-200'
                      : 'bg-background text-slate-600 border-slate-200 hover:bg-slate-50'}`}
                    onClick={() => setActiveLocation('remote')}
                  >
                    <Globe className="h-4 w-4" />
                    <span>{t('tools.mcp.dialog.remote')}</span>
                    {isSelectMode && selectedRemoteCount > 0 && (
                      <Badge variant="secondary" className="h-5 px-1.5 min-w-5 flex items-center justify-center bg-blue-100 text-blue-700 border-none">
                        {selectedRemoteCount}
                      </Badge>
                    )}
                  </button>
                  <button
                    className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 text-sm font-medium rounded-md border transition-colors ${activeLocation === 'local'
                      ? 'bg-blue-50 text-blue-700 border-blue-200'
                      : 'bg-background text-slate-600 border-slate-200 hover:bg-slate-50'}`}
                    onClick={() => setActiveLocation('local')}
                  >
                    <Home className="h-4 w-4" />
                    <span>{t('tools.mcp.dialog.local')}</span>
                    {isSelectMode && selectedLocalCount > 0 && (
                      <Badge variant="secondary" className="h-5 px-1.5 min-w-5 flex items-center justify-center bg-blue-100 text-blue-700 border-none">
                        {selectedLocalCount}
                      </Badge>
                    )}
                  </button>
                </div>
                <SearchInput
                  placeholder={t('tools.mcp.dialog.searchPlaceholder')}
                  value={searchQuery}
                  onChange={setSearchQuery}
                  className="w-full max-w-full bg-slate-50/50"
                />
                <div className="mt-4 text-sm text-slate-500 font-medium flex items-center h-5">
                  {isLoadingApps ? (
                    <div className="h-4 bg-slate-200 rounded animate-pulse w-24" />
                  ) : (
                    t('tools.mcp.dialog.serversFound', { count: apps.length })
                  )}
                </div>
              </div>

              <div className="flex-1 overflow-y-auto p-6 pt-4">
                {isLoadingApps ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Array.from({ length: 6 }).map((_, i) => (
                      <Card key={i} className="p-[0] shadow-sm border-slate-200">
                        <CardContent className="p-5 flex flex-col h-full">
                          <div className="flex items-start gap-3 mb-3">
                            <div className="w-10 h-10 rounded-md bg-slate-200 animate-pulse shrink-0" />
                            <div className="flex-1 min-w-0 space-y-2 py-1">
                              <div className="h-4 bg-slate-200 rounded animate-pulse w-3/4" />
                              <div className="h-3 bg-slate-200 rounded animate-pulse w-1/2" />
                            </div>
                          </div>
                          <div className="space-y-2 mb-4 mt-2">
                            <div className="h-3 bg-slate-200 rounded animate-pulse w-full" />
                            <div className="h-3 bg-slate-200 rounded animate-pulse w-5/6" />
                          </div>
                          <div className="flex items-center justify-between mt-auto pt-2">
                            <div className="h-5 w-16 bg-slate-200 rounded animate-pulse" />
                            <div className="h-5 w-12 bg-slate-200 rounded animate-pulse" />
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                ) : apps.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-slate-500 py-12">
                    <LayoutGrid className="h-12 w-12 mb-4 text-slate-200" />
                    <p>{t('tools.mcp.dialog.noServersFound')}</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {apps.map(app => {
                      const isGloballyConnected = isAppConnected(app)
                      const isSelected = localSelectedServers.includes(app.id) || localSelectedServers.includes(app.name)
                      const isLoading = loadingApps.has(app.id)
                      return (
                        <Card key={app.id} className={`p-[0] cursor-pointer transition-colors shadow-sm relative ${isSelectMode && isSelected ? 'border-blue-500 bg-blue-50/30 ring-1 ring-blue-500' : 'hover:border-slate-300 border-slate-200'}`} onClick={() => handleCardClick(app, isGloballyConnected)}>
                          {isGloballyConnected && (
                            <div className="absolute top-4 right-4 text-green-500">
                              <CheckCircle2 className="h-5 w-5 fill-green-100" />
                            </div>
                          )}
                          <CardContent className="p-5 flex flex-col h-full">
                            <div className="flex items-start gap-3 mb-3">
                              {app.icon ? (
                                <img
                                  src={app.icon}
                                  alt={app.name}
                                  className="w-10 h-10 rounded-md object-contain bg-white p-1 border shadow-sm shrink-0"
                                  onError={(e) => {
                                    (e.target as HTMLImageElement).src = `https://ui-avatars.com/api/?name=${encodeURIComponent(app.name)}&background=random&color=fff&size=128`
                                  }}
                                />
                              ) : (
                                <div className="w-10 h-10 rounded-md bg-blue-50 text-blue-600 border shadow-sm flex items-center justify-center font-bold text-lg shrink-0">
                                  {app.name.charAt(0).toUpperCase()}
                                </div>
                              )}
                              <div className="flex-1 min-w-0">
                                <h3 className="font-bold text-base text-slate-900 truncate">{app.name}</h3>
                                <p className="text-xs text-slate-500 truncate">{app.id}</p>
                              </div>
                            </div>
                            <p className="text-sm text-slate-600 line-clamp-2 flex-1 mb-4 leading-relaxed">
                              {app.description}
                            </p>
                            <div className="flex items-center justify-between mt-auto">
                              <div className="flex items-center gap-2">
                                <Badge variant="secondary" className="bg-slate-100 text-slate-600 font-medium px-2 py-0.5 rounded-md border border-slate-200 shadow-none">
                                  {app.is_local ? <Home className="h-3 w-3 mr-1.5 text-slate-400" /> : <Globe className="h-3 w-3 mr-1.5 text-slate-400" />}
                                  {app.is_local ? t('tools.mcp.dialog.local') : t('tools.mcp.dialog.remote')}
                                </Badge>
                                {inTeam && isGloballyConnected && connectorRef(app) && (
                                  <Badge variant="secondary" className={`font-medium px-2 py-0.5 rounded-md border shadow-none ${app.shared ? 'bg-blue-50 text-blue-700 border-blue-200' : 'bg-slate-100 text-slate-600 border-slate-200'}`}>
                                    {!app.shared
                                      ? t('tools.mcp.sharing.private')
                                      : app.is_owner
                                        ? t('tools.mcp.sharing.shared')
                                        : t('tools.mcp.sharing.teamTool')}
                                  </Badge>
                                )}
                                {app.needs_config && (
                                  <Badge variant="secondary" className="font-medium px-2 py-0.5 rounded-md border border-amber-200 bg-amber-50 text-amber-700 shadow-none">
                                    {t('tools.mcp.sharing.needsConfig')}
                                  </Badge>
                                )}
                              </div>
                              <div className="flex items-center gap-2">
                                {isLoading ? (
                                  <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                                ) : isGloballyConnected && (
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    className="h-7 text-xs text-slate-600 hover:text-slate-900 px-2 bg-slate-100 hover:bg-slate-200"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setSelectedApp(app);
                                    }}
                                  >
                                    <Settings className="h-3 w-3 mr-1" /> {t('tools.mcp.dialog.configure')}
                                  </Button>
                                )}
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      )
                    })}
                  </div>
                )}
              </div>
            </div>
          </TabsContent>
          <TabsContent value="custom_api" className="flex-1 overflow-y-auto p-6 m-0 bg-slate-50/50">
            <div className="max-w-2xl mx-auto w-full">
              <div className="mb-6">
                <h2 className="text-xl font-bold">{editingCustomServerId ? t('tools.mcp.dialog.editCustomApi') : t('tools.mcp.dialog.addCustomApi')}</h2>
                <p className="text-sm text-slate-500 mt-1">{t('tools.mcp.dialog.customApiDescription')}</p>
              </div>

              <div className="space-y-4">
                <CustomApiForm
                  key={editingCustomServerId || 'new'}
                  mcpFormData={mcpFormData}
                  setMcpFormData={setMcpFormData}
                  customApiEnv={customApiEnv}
                  setCustomApiEnv={setCustomApiEnv}
                  onRuntimeValidationErrorChange={setRuntimeValidationError}
                  originalEnvObj={
                    editingCustomServerId ? customApiEditBaseline?.env ?? {} : {}
                  }
                />
                {ownershipRadio}
              </div>

              <div className="flex justify-end gap-3 mt-8 pt-4 border-t">
                {/* Round-8 N2: requestClose silently refuses while a catalog
                    connect is in flight — disable the trigger so the blocked
                    state is visible instead of the button appearing dead. */}
                <Button variant="outline" onClick={requestClose} disabled={catalogConnectsInFlight > 0}>
                  {t('tools.mcp.buttons.cancel')}
                </Button>
                <Button
                  onClick={handleSaveCustomMcp}
                  // Round-9: Save was missed when the Cancel/footer-Connect
                  // buttons got this treatment in round 8 -- reachable by
                  // starting a keyless connect on the Library tab, switching
                  // to this tab (no tab-switch guard exists), and saving:
                  // the save succeeds and toasts, but requestClose() no-ops
                  // while catalogConnectsInFlight > 0, leaving the dialog
                  // silently stuck open with no explanation.
                  disabled={
                    isSavingCustom ||
                    catalogConnectsInFlight > 0 ||
                    !mcpFormData.name?.trim() ||
                    !mcpFormData.url?.trim() ||
                    (customApiEnv.length > 0 && customApiEnv.some(env => env.key.trim() && !env.value.trim()))
                  }
                >
                  {isSavingCustom && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                  {t('tools.mcp.buttons.save')}
                </Button>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="custom" className="flex-1 overflow-y-auto p-6 m-0 bg-slate-50/50">
            {customContent ? customContent : (
              <div className="max-w-2xl mx-auto w-full">
                <div className="mb-6">
                  <h2 className="text-xl font-bold">{editingCustomServerId ? t('tools.mcp.dialog.editTitle') : t('tools.mcp.dialog.addTitle')}</h2>
                  <p className="text-sm text-slate-500 mt-1">{t('tools.mcp.dialog.description')}</p>
                </div>

                <div className="space-y-4">
                  <CustomMcpForm
                    key={editingCustomServerId || 'new'}
                    mcpFormData={mcpFormData}
                    setMcpFormData={setMcpFormData}
                    serverId={editingCustomServerId}
                    onOAuthStatusChange={loadApps}
                    onRuntimeValidationErrorChange={setRuntimeValidationError}
                  />
                  {ownershipRadio}
                </div>
                <div className="flex justify-end gap-3 mt-8">
                  {/* Round-8 N2: same rationale as the Custom API tab's
                      Cancel above. */}
                  <Button variant="outline" onClick={requestClose} disabled={catalogConnectsInFlight > 0}>
                    {t('tools.mcp.buttons.cancel')}
                  </Button>
                  {/* Round-9: same rationale as the Custom API tab's Save. */}
                  <Button onClick={handleSaveCustomMcp} disabled={isSavingCustom || catalogConnectsInFlight > 0}>
                    {isSavingCustom && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                    {t('tools.mcp.buttons.save')}
                  </Button>
                </div>
              </div>
            )}
          </TabsContent>
        </Tabs>

        {/* Footer Actions */}
        {isSelectMode && activeTab === "library" && (
          <div className="p-4 border-t bg-slate-50/80 flex items-center justify-between shrink-0 mt-auto">
            <div className="flex items-center gap-4">
              {localSelectedServers.length > 0 && (
                <div className="flex items-center gap-2 bg-blue-100 text-blue-700 px-3 py-1.5 rounded-md font-medium text-sm">
                  <CheckCircle2 className="h-4 w-4" /> {t('tools.mcp.dialog.selected', { count: localSelectedServers.length })}
                </div>
              )}
            </div>
            <Button
              className="font-medium bg-blue-600 hover:bg-blue-700 text-white shadow-sm px-6"
              // Round-7: onConnectSelected commits localSelectedServers to the
              // parent as a one-shot snapshot, with no later reconciliation.
              // If a catalog connect is still in flight, that snapshot can be
              // stale (missing the app the in-flight connect will add via
              // autoSelect once it succeeds) — and requestClose()'s own guard
              // wouldn't even close the dialog after committing it. Disable
              // the trigger instead of letting it commit early; same signal
              // requestClose already gates on.
              disabled={catalogConnectsInFlight > 0}
              onClick={() => {
                if (onConnectSelected) {
                  onConnectSelected(localSelectedServers);
                }
                requestClose();
              }}
            >
              <Zap className="h-4 w-4 mr-2" /> {t('tools.mcp.dialog.connect')}
            </Button>
          </div>
        )}
      </DialogContent>

      {/* App Details Sub-Dialog */}
      <OfficialMcpSettingsDialog
        open={!!selectedApp}
        onOpenChange={(nextOpen) => {
          // Deliberately NOT gated on catalogConnectsInFlight, unlike the
          // parent dialog's requestClose (round-8 N5): closing this per-app
          // sub-dialog mid-connect is tolerated by design — in-flight state
          // is tracked per app id, completion clears selectedApp only if it
          // still points at the same app, and the cross-app regression
          // tests pin exactly this sequence. Blocking it would trap the
          // user inside the sub-dialog for the full request bound instead.
          if (!nextOpen) {
            connectorEditRequestRef.current += 1
            setSelectedApp(null)
          }
        }}
        app={selectedApp}
        isGloballyConnected={selectedApp ? isAppConnected(selectedApp) : false}
        isConnecting={!!selectedApp && loadingApps.has(selectedApp.id)}
        onSuccess={() => {
          if (onSuccess) onSuccess();
          loadApps();
        }}
        onDisconnect={(disconnectedApp) => {
          setLocalSelectedServers(prev => {
            const newSelection = prev.filter(name =>
              name.toLowerCase() !== disconnectedApp.name.toLowerCase() &&
              name.toLowerCase() !== disconnectedApp.id.toLowerCase()
            );
            // Use setTimeout to move the parent state update out of the render cycle
            // This prevents React "setState in render" warning and potential crashes
            if (onConnectSelected) {
              setTimeout(() => onConnectSelected(newSelection), 0);
            }
            return newSelection;
          });
        }}
        onConnectStart={(appToConnect) => handleConnectApp(appToConnect, isSelectMode)}
        onManageKey={(appToManage) => {
          setSelectedApp(null);
          openKeyConnect(appToManage);
        }}
        onConfigure={(appToConfigure) => {
          if (!appToConfigure.is_custom || !Number.isInteger(appToConfigure.server_id)) {
            return
          }
          if (appToConfigure.transport === "custom_api") {
            void openCustomApiEditor(appToConfigure.server_id as number)
          } else {
            void openMcpServerEditor(appToConfigure.server_id as number)
          }
        }}
      />
    </Dialog>

    {/* Key-based (non-oauth) connect: only the required secret(s) are editable. */}
    <Dialog open={!!connectingKeyApp} onOpenChange={(o) => { if (!o && !isConnectingKey) setConnectingKeyApp(null) }}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {connectingKeyApp?.icon && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={connectingKeyApp?.icon} alt="" className="h-5 w-5" />
            )}
            {t('tools.mcp.dialog.connect')} {connectingKeyApp?.name}
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-4 py-2">
          {/* Source selector: only show options that are actually usable. */}
          <RadioGroup
            value={keyEnvSource}
            onValueChange={(v) => setKeyEnvSource(v as "own" | "shared" | "platform")}
          >
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="own" id="env-source-own" />
              <Label htmlFor="env-source-own" className="font-normal cursor-pointer">
                {t('tools.mcp.dialog.envSource.own')}
              </Label>
            </div>
            {connectingKeyApp?.shared_env_available && (
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="shared" id="env-source-shared" />
                <Label htmlFor="env-source-shared" className="font-normal cursor-pointer">
                  {t('tools.mcp.dialog.envSource.shared')}
                </Label>
              </div>
            )}
            {connectingKeyApp?.platform_env_available && (
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="platform" id="env-source-platform" />
                <Label htmlFor="env-source-platform" className="font-normal cursor-pointer">
                  {t('tools.mcp.dialog.envSource.platform')}
                </Label>
              </div>
            )}
          </RadioGroup>

          {keyEnvSource === "own" && (
            <>
              {(connectingKeyApp?.launch_config?.required_env || []).map((k) => (
                <div key={k} className="space-y-1.5">
                  <Label htmlFor={`key-${k}`}>{k}</Label>
                  <Input
                    id={`key-${k}`}
                    type="password"
                    autoComplete="off"
                    value={keyEnvValues[k] || ""}
                    onFocus={(e) => {
                      // Select the mask so typing replaces it, but clicking/tabbing away
                      // keeps it — submitting the mask unchanged preserves the stored key.
                      if (keyEnvValues[k] === MASKED_SECRET_VALUE) {
                        e.currentTarget.select()
                      }
                    }}
                    onChange={(e) => setKeyEnvValues(prev => ({ ...prev, [k]: e.target.value }))}
                  />
                </div>
              ))}
              <p className="text-xs text-slate-500">{t('tools.mcp.dialog.apiKeyOptionalHint')}</p>
            </>
          )}
          {ownershipRadio}
        </div>
        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={() => setConnectingKeyApp(null)} disabled={isConnectingKey}>
            {t('common.cancel')}
          </Button>
          <Button onClick={() => submitKeyConnect(isSelectMode)} disabled={isConnectingKey}>
            {isConnectingKey ? t('tools.mcp.dialog.connecting') : t('tools.mcp.dialog.connect')}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
    </>
  )
}
