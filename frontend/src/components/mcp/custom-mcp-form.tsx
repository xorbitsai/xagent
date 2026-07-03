import React, { useCallback, useEffect, useRef, useState } from "react"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select-radix"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { CheckCircle2, ChevronDown, ChevronRight, ExternalLink, Info, Loader2, Plus, RefreshCw, Search, ShieldCheck, Trash2 } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { toast } from "@/components/ui/sonner"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { MCPServerFormData } from "./custom-api-form"

interface CustomMcpFormProps {
  mcpFormData: MCPServerFormData
  setMcpFormData: React.Dispatch<React.SetStateAction<MCPServerFormData>>
  serverId?: number | null
  onOAuthStatusChange?: () => void
}

interface McpOAuthGrantStatus {
  id: number
  resource_owner_key: string
  issuer: string
  resource: string
  scope: string
  token_type: string
  status: string
  expires_at?: string | null
}

interface McpOAuthStatus {
  server_id: number
  auth_type?: string | null
  resource?: string | null
  issuer?: string | null
  scope?: string | null
  grants: McpOAuthGrantStatus[]
}

interface McpOAuthDiscoveryResponse {
  resource: string
  issuer: string
  scopes: string[]
}

interface McpOAuthConnectResponse {
  authorization_url: string
}

const MASKED_SECRET_VALUE = "********"
const HTTP_MCP_OAUTH_TRANSPORTS = new Set(["streamable_http", "sse", "websocket"])

export function isHttpMcpOAuthTransport(transport: string): boolean {
  return HTTP_MCP_OAUTH_TRANSPORTS.has(transport)
}

async function parseMcpOAuthErrorMessage(response: Response, fallback: string): Promise<string> {
  try {
    const payload = await response.json()
    if (typeof payload?.detail === "string") return payload.detail
    if (payload?.detail?.message) return payload.detail.message
    if (payload?.detail?.code) return payload.detail.code
  } catch {
    // Keep the provided fallback.
  }
  return fallback
}

export function CustomMcpForm({
  mcpFormData,
  setMcpFormData,
  serverId,
  onOAuthStatusChange
}: CustomMcpFormProps) {
  const { t } = useI18n()
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false)
  const [oauthStatus, setOauthStatus] = useState<McpOAuthStatus | null>(null)
  const [oauthStatusLoading, setOauthStatusLoading] = useState(false)
  const [oauthAction, setOauthAction] = useState<string | null>(null)
  const isMountedRef = useRef(false)
  const pollingTimeoutRef = useRef<number | null>(null)
  const editedSecretFieldsRef = useRef<Set<string>>(new Set())
  const previousServerIdRef = useRef<number | null | undefined>(serverId)

  // Default to sse if not set
  const transport = mcpFormData.transport || "sse"
  const authConfig = mcpFormData.config?.auth
  const authType = authConfig?.type || "none"
  const isHttpMcpTransport = isHttpMcpOAuthTransport(transport)
  const isMcpOAuth = authType === "mcp_oauth"
  const bearerTokenValue = authConfig?.bearer_token
  const apiKeyValue = authConfig?.api_key_value
  const clientSecretValue = authConfig?.client_secret

  const clearOAuthPolling = useCallback(() => {
    if (pollingTimeoutRef.current !== null && typeof window !== "undefined") {
      window.clearTimeout(pollingTimeoutRef.current)
      pollingTimeoutRef.current = null
    }
  }, [])

  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
      clearOAuthPolling()
    }
  }, [clearOAuthPolling])

  const updateConfig = (key: string, value: unknown) => {
    setMcpFormData((prev: MCPServerFormData) => ({
      ...prev,
      config: { ...prev.config, [key]: value }
    }))
  }

  const updateAuth = (key: string, value: unknown) => {
    setMcpFormData((prev: MCPServerFormData) => ({
      ...prev,
      config: {
        ...prev.config,
        auth: { ...(prev.config?.auth || {}), [key]: value }
      }
    }))
  }

  const updateSecretAuth = (key: string, value: string) => {
    editedSecretFieldsRef.current.add(key)
    updateAuth(key, value)
  }

  const focusSecretAuth = (key: string, value: unknown) => {
    if (value === MASKED_SECRET_VALUE) {
      editedSecretFieldsRef.current.delete(key)
      updateAuth(key, "")
    }
  }

  const blurSecretAuth = (
    key: string,
    value: unknown,
    originalMaskedValue: string | undefined,
  ) => {
    if (
      (value == null || value === "") &&
      originalMaskedValue &&
      !editedSecretFieldsRef.current.has(key)
    ) {
      updateAuth(key, MASKED_SECRET_VALUE)
    }
  }

  const updateAuthType = (value: string) => {
    setMcpFormData((prev: MCPServerFormData) => ({
      ...prev,
      config: {
        ...prev.config,
        auth: { type: value }
      }
    }))
  }

  const updateAuthFields = (fields: Record<string, unknown>) => {
    setMcpFormData((prev: MCPServerFormData) => ({
      ...prev,
      config: {
        ...prev.config,
        auth: { ...(prev.config?.auth || {}), ...fields }
      }
    }))
  }

  const buildOAuthRequestBody = (includeRedirectAfter = false) => {
    const body: Record<string, string | undefined> = {}
    if (includeRedirectAfter) {
      body.redirect_after = typeof window !== "undefined"
        ? `${window.location.pathname}${window.location.search}`
        : undefined
    }
    return body
  }

  const loadOAuthStatus = useCallback(async () => {
    if (!serverId || !isMcpOAuth) {
      if (isMountedRef.current) setOauthStatus(null)
      return
    }
    if (isMountedRef.current) setOauthStatusLoading(true)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/mcp/${serverId}/oauth/status`)
      if (!isMountedRef.current) return
      if (response.ok) {
        const status = await response.json()
        if (!isMountedRef.current) return
        setOauthStatus(status)
      } else {
        const message = await parseMcpOAuthErrorMessage(response, t('tools.mcp.dialog.oauthStatusFailed'))
        if (!isMountedRef.current) return
        toast.error(message)
      }
    } catch (error) {
      console.error("Failed to load MCP OAuth status:", error)
      if (isMountedRef.current) toast.error(t('tools.mcp.dialog.oauthStatusFailed'))
    } finally {
      if (isMountedRef.current) setOauthStatusLoading(false)
    }
  }, [serverId, isMcpOAuth, t])

  useEffect(() => {
    loadOAuthStatus()
  }, [loadOAuthStatus])

  const handleDiscoverMcpOAuth = async () => {
    if (!serverId) {
      toast.error(t('tools.mcp.dialog.oauthSaveBeforeConnect'))
      return
    }
    setOauthAction("discover")
    try {
      const response = await apiRequest(`${getApiUrl()}/api/mcp/${serverId}/oauth/discover`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildOAuthRequestBody())
      })
      if (!isMountedRef.current) return
      if (!response.ok) {
        const message = await parseMcpOAuthErrorMessage(response, t('tools.mcp.dialog.oauthDiscoveryFailed'))
        if (!isMountedRef.current) return
        toast.error(message)
        return
      }
      const discovery = await response.json() as McpOAuthDiscoveryResponse
      if (!isMountedRef.current) return
      updateAuthFields({
        resource: discovery.resource,
        issuer: discovery.issuer,
        scope: discovery.scopes?.join(" ") || mcpFormData.config?.auth?.scope || ""
      })
      toast.success(t('tools.mcp.dialog.oauthDiscoverySuccess'))
    } catch (error) {
      console.error("Failed to discover MCP OAuth metadata:", error)
      if (isMountedRef.current) toast.error(t('tools.mcp.dialog.oauthDiscoveryFailed'))
    } finally {
      if (isMountedRef.current) setOauthAction(null)
    }
  }

  const handleConnectMcpOAuth = async () => {
    if (!serverId) {
      toast.error(t('tools.mcp.dialog.oauthSaveBeforeConnect'))
      return
    }
    setOauthAction("connect")
    let popup: Window | null = null
    try {
      clearOAuthPolling()
      if (typeof window !== "undefined") {
        popup = window.open("about:blank", "_blank")
        if (!popup) {
          toast.error(t('tools.mcp.dialog.oauthConnectFailed'))
          return
        }
        popup.opener = null
      }
      const response = await apiRequest(`${getApiUrl()}/api/mcp/${serverId}/oauth/connect`, {
        method: "POST",
        headers: {
          "Accept": "application/json",
          "Content-Type": "application/json"
        },
        body: JSON.stringify(buildOAuthRequestBody(true))
      })
      if (!isMountedRef.current) {
        if (popup) popup.close()
        return
      }
      if (!response.ok) {
        if (popup) popup.close()
        const message = await parseMcpOAuthErrorMessage(response, t('tools.mcp.dialog.oauthConnectFailed'))
        if (!isMountedRef.current) return
        toast.error(message)
        return
      }
      const data = await response.json() as McpOAuthConnectResponse
      if (!isMountedRef.current) {
        if (popup) popup.close()
        return
      }
      if (!data.authorization_url) {
        if (popup) popup.close()
        toast.error(t('tools.mcp.dialog.oauthConnectFailed'))
        return
      }
      if (popup) {
        popup.location.href = data.authorization_url
      } else {
        popup = window.open(data.authorization_url, "_blank", "noopener,noreferrer")
      }
      const startedAt = Date.now()
      const maxWaitMs = 5 * 60 * 1000
      const poll = async () => {
        if (!isMountedRef.current) return
        const expired = Date.now() - startedAt >= maxWaitMs
        const stillOpen = popup && !popup.closed
        if (stillOpen && !expired) {
          await loadOAuthStatus()
          if (!isMountedRef.current) return
          pollingTimeoutRef.current = window.setTimeout(poll, 3000)
          return
        }
        clearOAuthPolling()
        await loadOAuthStatus()
        if (!isMountedRef.current) return
        onOAuthStatusChange?.()
      }
      pollingTimeoutRef.current = window.setTimeout(poll, 3000)
    } catch (error) {
      if (popup) popup.close()
      console.error("Failed to start MCP OAuth authorization:", error)
      if (isMountedRef.current) toast.error(t('tools.mcp.dialog.oauthConnectFailed'))
    } finally {
      if (isMountedRef.current) setOauthAction(null)
    }
  }

  const handleDeleteGrant = async (grantId: number) => {
    if (!serverId) return
    setOauthAction(`delete-${grantId}`)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/mcp/${serverId}/oauth/grants/${grantId}`, {
        method: "DELETE"
      })
      if (!isMountedRef.current) return
      if (!response.ok) {
        const message = await parseMcpOAuthErrorMessage(response, t('tools.mcp.dialog.oauthDisconnectFailed'))
        if (!isMountedRef.current) return
        toast.error(message)
        return
      }
      toast.success(t('tools.mcp.dialog.oauthDisconnectSuccess'))
      await loadOAuthStatus()
      if (!isMountedRef.current) return
      onOAuthStatusChange?.()
    } catch (error) {
      console.error("Failed to delete MCP OAuth grant:", error)
      if (isMountedRef.current) toast.error(t('tools.mcp.dialog.oauthDisconnectFailed'))
    } finally {
      if (isMountedRef.current) setOauthAction(null)
    }
  }

  // Handle headers state (array of {key, value} for the UI, but config expects an object)
  const headersObj = mcpFormData.config?.headers || {}
  const [headersList, setHeadersList] = useState<{ key: string, value: string }[]>(
    Object.keys(headersObj).length > 0
      ? Object.entries(headersObj).map(([k, v]) => ({ key: k, value: String(v) }))
      : []
  )

  // Handle stdio env vars state (array of {key, value} for the UI, but config expects an object)
  const envObj = mcpFormData.config?.env || {}
  const [envList, setEnvList] = useState<{ key: string, value: string }[]>(
    Object.keys(envObj).length > 0
      ? Object.entries(envObj).map(([k, v]) => ({ key: k, value: String(v) }))
      : []
  )

  const syncEnv = (newList: { key: string, value: string }[]) => {
    setEnvList(newList)
    const newEnvObj: Record<string, string> = {}
    newList.forEach(e => {
      if (e.key.trim()) newEnvObj[e.key.trim()] = e.value
    })
    updateConfig("env", Object.keys(newEnvObj).length > 0 ? newEnvObj : {})
  }

  // New servers default to owner (editable global); existing servers use the flag.
  const canEditGlobal = mcpFormData.can_edit_global ?? true

  // Per-user env overrides (top-level user_env, merged over global env at runtime).
  const userEnvObj = mcpFormData.user_env || {}
  const [userEnvList, setUserEnvList] = useState<{ key: string, value: string }[]>(
    Object.keys(userEnvObj).length > 0
      ? Object.entries(userEnvObj).map(([k, v]) => ({ key: k, value: String(v) }))
      : []
  )

  const syncUserEnv = (newList: { key: string, value: string }[]) => {
    setUserEnvList(newList)
    const obj: Record<string, string> = {}
    newList.forEach(e => {
      if (e.key.trim()) obj[e.key.trim()] = e.value
    })
    setMcpFormData((prev: MCPServerFormData) => ({ ...prev, user_env: obj }))
  }

  // Track original masked values to restore them on blur if empty
  const [originalAuth, setOriginalAuth] = useState<{
    bearer_token?: string;
    api_key_value?: string;
    client_secret?: string;
  }>({
    bearer_token: bearerTokenValue === MASKED_SECRET_VALUE ? MASKED_SECRET_VALUE : undefined,
    api_key_value: apiKeyValue === MASKED_SECRET_VALUE ? MASKED_SECRET_VALUE : undefined,
    client_secret: clientSecretValue === MASKED_SECRET_VALUE ? MASKED_SECRET_VALUE : undefined,
  })

  useEffect(() => {
    const serverChanged = previousServerIdRef.current !== serverId
    if (serverChanged) {
      editedSecretFieldsRef.current.clear()
      previousServerIdRef.current = serverId
    }
    const nextOriginalAuth = {
      bearer_token: bearerTokenValue === MASKED_SECRET_VALUE ? MASKED_SECRET_VALUE : undefined,
      api_key_value: apiKeyValue === MASKED_SECRET_VALUE ? MASKED_SECRET_VALUE : undefined,
      client_secret: clientSecretValue === MASKED_SECRET_VALUE ? MASKED_SECRET_VALUE : undefined,
    }
    if (nextOriginalAuth.bearer_token) {
      editedSecretFieldsRef.current.delete("bearer_token")
    }
    if (nextOriginalAuth.api_key_value) {
      editedSecretFieldsRef.current.delete("api_key_value")
    }
    if (nextOriginalAuth.client_secret) {
      editedSecretFieldsRef.current.delete("client_secret")
    }
    setOriginalAuth((prev) => ({
      bearer_token: serverChanged ? nextOriginalAuth.bearer_token : nextOriginalAuth.bearer_token || prev.bearer_token,
      api_key_value: serverChanged ? nextOriginalAuth.api_key_value : nextOriginalAuth.api_key_value || prev.api_key_value,
      client_secret: serverChanged ? nextOriginalAuth.client_secret : nextOriginalAuth.client_secret || prev.client_secret,
    }))
  }, [
    serverId,
    bearerTokenValue,
    apiKeyValue,
    clientSecretValue,
  ])

  const syncHeaders = (newList: { key: string, value: string }[]) => {
    setHeadersList(newList)
    const newHeadersObj: Record<string, string> = {}
    newList.forEach(h => {
      if (h.key.trim()) newHeadersObj[h.key.trim()] = h.value.trim()
    })
    updateConfig("headers", Object.keys(newHeadersObj).length > 0 ? newHeadersObj : {})
  }

  const renderEnvRows = (
    list: { key: string, value: string }[],
    sync: (l: { key: string, value: string }[]) => void,
    readOnly: boolean
  ) => (
    <>
      {list.length === 0 ? (
        <p className="text-sm text-slate-500">{t('tools.mcp.dialog.noEnvVariables')}</p>
      ) : (
        <div className="space-y-2">
          {list.map((e, i) => (
            <div key={i} className="flex gap-2 items-center">
              <Input
                placeholder={t('tools.mcp.dialog.envKeyPlaceholder')}
                value={e.key}
                disabled={readOnly}
                onChange={(ev) => {
                  const newList = [...list]
                  newList[i] = { ...newList[i], key: ev.target.value }
                  sync(newList)
                }}
                className="flex-1"
              />
              <Input
                type="password"
                placeholder={t('tools.mcp.dialog.envValuePlaceholder')}
                value={e.value}
                disabled={readOnly}
                // Select-all on focus so typing replaces the mask; we never write
                // an empty intermediate value, so an early submit can't wipe the
                // stored secret. If the user appends instead, strip the leading mask.
                onFocus={(ev) => ev.target.select()}
                onChange={(ev) => {
                  let value = ev.target.value
                  if (e.value === "********" && value !== "********" && value.startsWith("********")) {
                    value = value.slice("********".length)
                  }
                  const newList = [...list]
                  newList[i] = { ...newList[i], value }
                  sync(newList)
                }}
                className="flex-1"
              />
              {!readOnly && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => {
                    const newList = [...list]
                    newList.splice(i, 1)
                    sync(newList)
                  }}
                  className="text-red-500 hover:text-red-700"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              )}
            </div>
          ))}
        </div>
      )}
      {!readOnly && (
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="w-full border-dashed text-blue-600 border-blue-200 hover:bg-blue-50 hover:text-blue-700"
          onClick={() => sync([...list, { key: "", value: "" }])}
        >
          <Plus className="h-4 w-4 mr-2" /> {t('tools.mcp.dialog.addEnvVariable')}
        </Button>
      )}
    </>
  )

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="name">{t('tools.mcp.form.nameLabel')}</Label>
        <Input
          id="name"
          value={mcpFormData.name || ""}
          disabled={!canEditGlobal}
          onChange={(e) => setMcpFormData((prev: MCPServerFormData) => ({ ...prev, name: e.target.value }))}
          placeholder={t('tools.mcp.form.namePlaceholder')}
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="description">{t('tools.mcp.form.descriptionLabel')}</Label>
        <textarea
          id="description"
          className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
          value={mcpFormData.description || ""}
          disabled={!canEditGlobal}
          onChange={(e) => setMcpFormData((prev: MCPServerFormData) => ({ ...prev, description: e.target.value }))}
          placeholder={t('tools.mcp.form.descriptionPlaceholder')}
        />
      </div>
      <div className="space-y-2">
        <Label>{t('tools.mcp.dialog.transport')}</Label>
        <div className="flex bg-slate-100 p-1 rounded-md flex-wrap gap-1">
          {(["streamable_http", "sse", "stdio", "websocket"] as const).map((t) => (
            <button
              key={t}
              type="button"
              disabled={!canEditGlobal}
              className={`flex-1 py-1.5 text-sm font-medium rounded-md transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${transport === t ? "bg-blue-600 text-white shadow" : "text-slate-600 hover:text-slate-900 hover:bg-slate-200"}`}
              onClick={() => setMcpFormData((prev: MCPServerFormData) => ({ ...prev, transport: t }))}
            >
              {t === "sse" ? "SSE" : t === "streamable_http" ? "HTTP" : t === "stdio" ? "STDIO" : "WebSocket"}
            </button>
          ))}
        </div>
      </div>

      {transport === "stdio" ? (
        <>
          <div className="space-y-2">
            <Label htmlFor="command">{t('tools.mcp.dialog.command')}</Label>
            <Input
              id="command"
              value={mcpFormData.config?.command || ""}
              disabled={!canEditGlobal}
              onChange={(e) => updateConfig("command", e.target.value)}
              placeholder={t('tools.mcp.dialog.commandPlaceholder')}
            />
            <Alert className="border-amber-200 bg-amber-50 text-amber-900">
              <Info className="h-4 w-4 text-amber-700" />
              <AlertDescription className="text-amber-800">
                {t('tools.mcp.form.stdioSandboxHint')}
              </AlertDescription>
            </Alert>
          </div>
          <div className="space-y-2">
            <Label htmlFor="args">{t('tools.mcp.dialog.arguments')}</Label>
            <Input
              id="args"
              value={Array.isArray(mcpFormData.config?.args) ? mcpFormData.config.args.join(" ") : (mcpFormData.config?.args || "")}
              disabled={!canEditGlobal}
              onChange={(e) => {
                // Split by space for simple arg passing (in a real app, might want a better parser)
                const argsArr = e.target.value.split(" ").filter(Boolean)
                updateConfig("args", argsArr)
              }}
              placeholder={t('tools.mcp.dialog.argumentsPlaceholder')}
            />
          </div>
          {/* Per-user env overrides: each user's private values, merged over the global env */}
          <div className="space-y-3">
            <div>
              <Label className="text-sm font-semibold">{t('tools.mcp.dialog.userEnvVariables')}</Label>
              <p className="text-xs text-slate-500">{t('tools.mcp.dialog.userEnvVariablesDesc')}</p>
            </div>
            {renderEnvRows(userEnvList, syncUserEnv, false)}
          </div>

          {/* Global env: shared fallback default, editable only by owner/admin */}
          <div className="space-y-3">
            <div>
              <Label className="text-sm font-semibold">{t('tools.mcp.dialog.envVariables')}</Label>
              <p className="text-xs text-slate-500">
                {canEditGlobal
                  ? t('tools.mcp.dialog.globalEnvVariablesDesc')
                  : t('tools.mcp.dialog.globalEnvVariablesReadonlyDesc')}
              </p>
            </div>
            {renderEnvRows(envList, syncEnv, !canEditGlobal)}
          </div>
        </>
      ) : (
        <>
          <div className="space-y-2">
            <Label htmlFor="url">{t('tools.mcp.dialog.url')}</Label>
            <Input
              id="url"
              value={mcpFormData.config?.url || ""}
              disabled={!canEditGlobal}
              onChange={(e) => updateConfig("url", e.target.value)}
              placeholder={transport === "websocket" ? "wss://mcp.example.com/ws" : transport === "streamable_http" ? "https://mcp.example.com/mcp" : "https://mcp.example.com/sse"}
            />
          </div>

          {/* Auth config is shared/global; non-owners see it read-only. The OAuth
              status/actions below stay enabled so they can connect their own account. */}
          <fieldset disabled={!canEditGlobal} className="contents">
          <div className="space-y-2">
            <Label className="flex items-center gap-1">
              {t('tools.mcp.dialog.authentication')} <span className="text-slate-400 text-xs">(?)</span>
            </Label>
            <Select
              value={authType}
              onValueChange={(val) => updateAuthType(val)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">{t('tools.mcp.dialog.authTypes.none')}</SelectItem>
                <SelectItem value="bearer">{t('tools.mcp.dialog.authTypes.bearer')}</SelectItem>
                <SelectItem value="api_key">{t('tools.mcp.dialog.authTypes.apiKey')}</SelectItem>
                <SelectItem value="oauth2">{t('tools.mcp.dialog.authTypes.oauth2')}</SelectItem>
                {(isHttpMcpTransport || isMcpOAuth) && (
                  <SelectItem value="mcp_oauth">{t('tools.mcp.dialog.authTypes.mcpOAuth')}</SelectItem>
                )}
              </SelectContent>
            </Select>
          </div>

          {mcpFormData.config?.auth?.type === "bearer" && (
            <div className="space-y-2">
              <Label htmlFor="bearer_token">{t('tools.mcp.dialog.token')}</Label>
              <Input
                id="bearer_token"
                type="password"
                value={mcpFormData.config?.auth?.bearer_token || ""}
                onChange={(e) => updateSecretAuth("bearer_token", e.target.value)}
                onFocus={() => {
                  focusSecretAuth("bearer_token", mcpFormData.config?.auth?.bearer_token)
                }}
                onBlur={() => {
                  blurSecretAuth("bearer_token", mcpFormData.config?.auth?.bearer_token, originalAuth.bearer_token)
                }}
                placeholder={t('tools.mcp.dialog.tokenPlaceholder')}
              />
            </div>
          )}

          {mcpFormData.config?.auth?.type === "api_key" && (
            <>
              <div className="space-y-2">
                <Label htmlFor="api_key_name">{t('tools.mcp.dialog.headerName')}</Label>
                <Input
                  id="api_key_name"
                  value={mcpFormData.config?.auth?.api_key_name || ""}
                  onChange={(e) => updateAuth("api_key_name", e.target.value)}
                  placeholder={t('tools.mcp.dialog.headerNamePlaceholder')}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="api_key_value">{t('tools.mcp.dialog.apiKey')}</Label>
                <Input
                  id="api_key_value"
                  type="password"
                  value={mcpFormData.config?.auth?.api_key_value || ""}
                  onChange={(e) => updateSecretAuth("api_key_value", e.target.value)}
                  onFocus={() => {
                    focusSecretAuth("api_key_value", mcpFormData.config?.auth?.api_key_value)
                  }}
                  onBlur={() => {
                    blurSecretAuth("api_key_value", mcpFormData.config?.auth?.api_key_value, originalAuth.api_key_value)
                  }}
                  placeholder={t('tools.mcp.dialog.apiKeyPlaceholder')}
                />
              </div>
            </>
          )}

          {mcpFormData.config?.auth?.type === "oauth2" && (
            <>
              <div className="space-y-2">
                <Label htmlFor="client_id">{t('tools.mcp.dialog.clientId')}</Label>
                <Input
                  id="client_id"
                  value={mcpFormData.config?.auth?.client_id || ""}
                  onChange={(e) => updateAuth("client_id", e.target.value)}
                  placeholder={t('tools.mcp.dialog.clientIdPlaceholder')}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="client_secret">{t('tools.mcp.dialog.clientSecret')}</Label>
                <Input
                  id="client_secret"
                  type="password"
                  value={mcpFormData.config?.auth?.client_secret || ""}
                  onChange={(e) => updateSecretAuth("client_secret", e.target.value)}
                  onFocus={() => {
                    focusSecretAuth("client_secret", mcpFormData.config?.auth?.client_secret)
                  }}
                  onBlur={() => {
                    blurSecretAuth("client_secret", mcpFormData.config?.auth?.client_secret, originalAuth.client_secret)
                  }}
                  placeholder={t('tools.mcp.dialog.clientSecretPlaceholder')}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="token_url">{t('tools.mcp.dialog.tokenUrl')}</Label>
                <Input
                  id="token_url"
                  value={mcpFormData.config?.auth?.token_url || ""}
                  onChange={(e) => updateAuth("token_url", e.target.value)}
                  placeholder={t('tools.mcp.dialog.tokenUrlPlaceholder')}
                />
              </div>
            </>
          )}
          </fieldset>

          {isMcpOAuth && (
            <>
              <fieldset disabled={!canEditGlobal} className="contents">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label htmlFor="mcp_oauth_resource">{t('tools.mcp.dialog.oauthResource')}</Label>
                  <Input
                    id="mcp_oauth_resource"
                    value={mcpFormData.config?.auth?.resource || ""}
                    onChange={(e) => updateAuth("resource", e.target.value)}
                    placeholder="https://mcp.example.com/mcp"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="mcp_oauth_issuer">{t('tools.mcp.dialog.oauthIssuer')}</Label>
                  <Input
                    id="mcp_oauth_issuer"
                    value={mcpFormData.config?.auth?.issuer || ""}
                    onChange={(e) => updateAuth("issuer", e.target.value)}
                    placeholder="https://auth.example.com"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="mcp_oauth_scope">{t('tools.mcp.dialog.oauthScope')}</Label>
                <Input
                  id="mcp_oauth_scope"
                  value={mcpFormData.config?.auth?.scope || ""}
                  onChange={(e) => updateAuth("scope", e.target.value)}
                  placeholder="records.read records.write"
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label htmlFor="mcp_oauth_client_id">{t('tools.mcp.dialog.clientId')}</Label>
                  <Input
                    id="mcp_oauth_client_id"
                    value={mcpFormData.config?.auth?.client_id || ""}
                    onChange={(e) => updateAuth("client_id", e.target.value)}
                    placeholder={t('tools.mcp.dialog.clientIdPlaceholder')}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="mcp_oauth_client_secret">{t('tools.mcp.dialog.clientSecret')}</Label>
                  <Input
                    id="mcp_oauth_client_secret"
                    type="password"
                    value={mcpFormData.config?.auth?.client_secret || ""}
                    onChange={(e) => updateSecretAuth("client_secret", e.target.value)}
                    onFocus={() => {
                      focusSecretAuth("client_secret", mcpFormData.config?.auth?.client_secret)
                    }}
                    onBlur={() => {
                      blurSecretAuth("client_secret", mcpFormData.config?.auth?.client_secret, originalAuth.client_secret)
                    }}
                    placeholder={t('tools.mcp.dialog.clientSecretPlaceholder')}
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label htmlFor="mcp_oauth_resource_metadata">{t('tools.mcp.dialog.oauthResourceMetadataUrl')}</Label>
                  <Input
                    id="mcp_oauth_resource_metadata"
                    value={mcpFormData.config?.auth?.resource_metadata_url || ""}
                    onChange={(e) => updateAuth("resource_metadata_url", e.target.value)}
                    placeholder="https://mcp.example.com/.well-known/oauth-protected-resource"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="mcp_oauth_redirect_uri">{t('tools.mcp.dialog.oauthRedirectUri')}</Label>
                  <Input
                    id="mcp_oauth_redirect_uri"
                    value={mcpFormData.config?.auth?.redirect_uri || ""}
                    onChange={(e) => updateAuth("redirect_uri", e.target.value)}
                    placeholder="https://xagent.example.com/api/mcp/oauth/callback"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label>{t('tools.mcp.dialog.oauthTokenEndpointAuthMethod')}</Label>
                <Select
                  value={mcpFormData.config?.auth?.token_endpoint_auth_method || "none"}
                  onValueChange={(val) => updateAuth("token_endpoint_auth_method", val)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">none</SelectItem>
                    <SelectItem value="client_secret_post">client_secret_post</SelectItem>
                    <SelectItem value="client_secret_basic">client_secret_basic</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              </fieldset>

              <div className="rounded-md border border-slate-200 bg-slate-50 p-3 space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2 min-w-0">
                    <ShieldCheck className="h-4 w-4 text-slate-600 shrink-0" />
                    <span className="text-sm font-medium text-slate-800 truncate">
                      {t('tools.mcp.dialog.oauthStatus')}
                    </span>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={loadOAuthStatus}
                    disabled={!serverId || oauthStatusLoading}
                    title={t('tools.mcp.dialog.oauthRefreshStatus')}
                  >
                    <RefreshCw className={`h-4 w-4 ${oauthStatusLoading ? "animate-spin" : ""}`} />
                  </Button>
                </div>

                {!serverId ? (
                  <p className="text-sm text-slate-500">{t('tools.mcp.dialog.oauthSaveBeforeConnect')}</p>
                ) : oauthStatus?.grants?.length ? (
                  <div className="space-y-2">
                    {oauthStatus.grants.map((grant) => (
                      <div key={grant.id} className="flex items-start justify-between gap-3 rounded-md border border-slate-200 bg-white px-3 py-2">
                        <div className="min-w-0 space-y-1">
                          <div className="flex items-center gap-2 text-sm font-medium text-slate-800">
                            <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0" />
                            <span className="truncate">{grant.resource_owner_key}</span>
                          </div>
                          <p className="text-xs text-slate-500 truncate">{grant.scope || t('tools.mcp.dialog.oauthNoScope')}</p>
                        </div>
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          onClick={() => handleDeleteGrant(grant.id)}
                          disabled={oauthAction === `delete-${grant.id}`}
                          title={t('tools.mcp.dialog.oauthDisconnectGrant')}
                          className="shrink-0 text-red-500 hover:text-red-700"
                        >
                          {oauthAction === `delete-${grant.id}` ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Trash2 className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-slate-500">{t('tools.mcp.dialog.oauthNoGrants')}</p>
                )}

                <div className="flex flex-col sm:flex-row gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={handleDiscoverMcpOAuth}
                    disabled={!serverId || oauthAction === "discover"}
                    className="flex-1"
                  >
                    {oauthAction === "discover" ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Search className="h-4 w-4 mr-2" />}
                    {t('tools.mcp.dialog.oauthDiscover')}
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    onClick={handleConnectMcpOAuth}
                    disabled={!serverId || oauthAction === "connect"}
                    className="flex-1"
                  >
                    {oauthAction === "connect" ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <ExternalLink className="h-4 w-4 mr-2" />}
                    {oauthStatus?.grants?.length ? t('tools.mcp.dialog.oauthReconnect') : t('tools.mcp.dialog.oauthConnect')}
                  </Button>
                </div>
              </div>
            </>
          )}
        </>
      )}

      <Collapsible open={isAdvancedOpen} onOpenChange={setIsAdvancedOpen} className="w-full space-y-2">
        <CollapsibleTrigger asChild>
          <Button variant="ghost" className="flex w-full items-center justify-start p-0 h-auto font-medium text-slate-700 hover:text-slate-900 hover:bg-transparent">
            {isAdvancedOpen ? <ChevronDown className="h-4 w-4 mr-2" /> : <ChevronRight className="h-4 w-4 mr-2" />}
            {t('tools.mcp.dialog.advancedOptions')}
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="space-y-4 pt-2 border-l-2 border-slate-100 pl-4 ml-2">
          <div className="space-y-3">
            <div>
              <Label className="text-sm font-semibold">{t('tools.mcp.dialog.customHeaders')}</Label>
              <p className="text-xs text-slate-500">{t('tools.mcp.dialog.customHeadersDesc')}</p>
            </div>

            {headersList.length === 0 ? (
              <p className="text-sm text-slate-500">{t('tools.mcp.dialog.noCustomHeaders')}</p>
            ) : (
              <div className="space-y-2">
                {headersList.map((h, i) => (
                  <div key={i} className="flex gap-2 items-center">
                    <Input
                      placeholder={t('tools.mcp.dialog.headerKeyPlaceholder')}
                      value={h.key}
                      onChange={(e) => {
                        const newList = [...headersList]
                        newList[i].key = e.target.value
                        syncHeaders(newList)
                      }}
                      className="flex-1"
                    />
                    <Input
                      placeholder={t('tools.mcp.dialog.headerValuePlaceholder')}
                      value={h.value}
                      onChange={(e) => {
                        const newList = [...headersList]
                        newList[i].value = e.target.value
                        syncHeaders(newList)
                      }}
                      className="flex-1"
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => {
                        const newList = [...headersList]
                        newList.splice(i, 1)
                        syncHeaders(newList)
                      }}
                      className="text-red-500 hover:text-red-700"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}

            <Button
              type="button"
              variant="outline"
              size="sm"
              className="w-full border-dashed text-blue-600 border-blue-200 hover:bg-blue-50 hover:text-blue-700"
              onClick={() => syncHeaders([...headersList, { key: "", value: "" }])}
            >
              <Plus className="h-4 w-4 mr-2" /> {t('tools.mcp.dialog.addHeader')}
            </Button>
          </div>

          <div className="space-y-2">
            <Label htmlFor="timeout">{t('tools.mcp.dialog.timeout')}</Label>
            <div className="flex items-center gap-2">
              <Input
                id="timeout"
                type="number"
                value={mcpFormData.config?.timeout || 30}
                onChange={(e) => updateConfig("timeout", Number(e.target.value))}
                className="w-full"
              />
              <span className="text-sm text-slate-500">{t('tools.mcp.dialog.timeoutUnit')}</span>
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  )
}
