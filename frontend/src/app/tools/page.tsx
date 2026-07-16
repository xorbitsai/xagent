"use client"

import { Suspense, useState, useEffect, useRef } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { Select } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import {
  Server,
  Plus,
  Wrench,
  Flame,
  Globe,
  Hash,
  Code,
  FileText,
  Book,
  Loader2,
  Mic,
  Video,
  Database,
  Trash2,
  Search,
} from "lucide-react"
import { getApiUrl, cn } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { ConnectMcpDialog, AppIntegration } from "@/components/mcp/connect-mcp-dialog"
import { OfficialMcpSettingsDialog } from "@/components/mcp/official-mcp-settings-dialog"
import { CustomApiForm, MCPServerFormData } from "@/components/mcp/custom-api-form"
import { CustomMcpForm } from "@/components/mcp/custom-mcp-form"
import {
  getRuntimeConfigError,
  type RuntimeConfigErrorKey,
} from "@/components/mcp/runtime-inputs-form"
import { useI18n } from "@/contexts/i18n-context"
import { useAuth } from "@/contexts/auth-context"
import { useMcpApps } from "@/contexts/mcp-apps-context"
import { toast } from "@/components/ui/sonner"
import {
  isValidMcpName,
  buildCustomApiPayload,
  buildMcpServerPayload,
  customApiDetailToEditState,
  mcpServerDetailToEditState,
  parseCustomApiDetail,
  parseMcpServerDetail,
  type CustomApiDetail,
  type McpServerDetail,
} from "@/lib/mcp-utils"

interface Tool {
  name: string
  description: string
  type: 'builtin' | 'mcp' | 'image' | 'vision' | 'audio' | 'video'
  category: string
  display_category?: string  // Add display_category field
  enabled: boolean
  requires_configuration?: boolean
  status?: string
  status_reason?: string
  config?: Record<string, any>
  source?: string
  usage_count?: number
}

export interface MCPServer {
  id: number
  user_id: number
  name: string
  transport: string
  description?: string
  config: Record<string, any>
  is_active: boolean
  is_default: boolean
  transport_display: string
  created_at: string
  updated_at: string
  connected_account?: string
  app_id?: string
  provider?: string
  user_env?: Record<string, string>
  can_edit_global?: boolean
  runtime_input_schema?: Record<string, any> | null
  runtime_bindings?: Record<string, any>[] | null
  allow_delegated_authorization?: boolean
}

interface ConfigurableToolField {
  label: string
  required: boolean
  secret: boolean
  source: 'db' | 'env' | 'none'
  is_configured: boolean
  masked: string
}

interface ConfigurableTool {
  tool_name: string
  display_name?: string
  configured: boolean
  fields: Record<string, ConfigurableToolField>
}

interface SqlConnectionItem {
  name: string
  source: 'db' | 'env' | 'none'
  masked: string
}

type SqlDbType = 'postgresql' | 'mysql' | 'mariadb' | 'mssql' | 'sqlite'

const DEFAULT_PORTS: Record<Exclude<SqlDbType, 'sqlite'>, string> = {
  postgresql: '5432',
  mysql: '3306',
  mariadb: '3306',
  mssql: '1433',
}

function ToolsPageContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [tools, setTools] = useState<Tool[]>([])
  const [mcpServers, setMcpServers] = useState<MCPServer[]>([])
  const [configurableTools, setConfigurableTools] = useState<ConfigurableTool[]>([])
  const [sqlConnections, setSqlConnections] = useState<SqlConnectionItem[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isConnectMcpOpen, setIsConnectMcpOpen] = useState(false)
  const [isOfficialAppDialogOpen, setIsOfficialAppDialogOpen] = useState(false)
  const [editingOfficialApp, setEditingOfficialApp] = useState<AppIntegration | null>(null)
  const [isMcpDialogOpen, setIsMcpDialogOpen] = useState(false)
  const [customApiEnv, setCustomApiEnv] = useState<{ key: string, value: string }[]>([{ key: "", value: "" }])
  const [isCredentialDialogOpen, setIsCredentialDialogOpen] = useState(false)
  const [editingConfigTool, setEditingConfigTool] = useState<ConfigurableTool | null>(null)
  const [credentialValues, setCredentialValues] = useState<Record<string, string>>({})
  const [isSavingCredentials, setIsSavingCredentials] = useState(false)
  const [pendingToolToggles, setPendingToolToggles] = useState<Record<string, boolean>>({})
  const [pendingSqlDeletes, setPendingSqlDeletes] = useState<Record<string, boolean>>({})
  const [isSqlManagerOpen, setIsSqlManagerOpen] = useState(false)
  const [sqlFormName, setSqlFormName] = useState("")
  const [sqlFormType, setSqlFormType] = useState<SqlDbType>('postgresql')
  const [sqlFormHost, setSqlFormHost] = useState("")
  const [sqlFormPort, setSqlFormPort] = useState(DEFAULT_PORTS.postgresql)
  const [sqlFormDatabase, setSqlFormDatabase] = useState("")
  const [sqlFormUsername, setSqlFormUsername] = useState("")
  const [sqlFormPassword, setSqlFormPassword] = useState("")
  const [sqlFormParams, setSqlFormParams] = useState("")
  const [sqlFormSqlitePath, setSqlFormSqlitePath] = useState("")
  const [isSavingSql, setIsSavingSql] = useState(false)
  const [editingServer, setEditingServer] = useState<MCPServer | null>(null)
  const [customApiEditBaseline, setCustomApiEditBaseline] = useState<CustomApiDetail | null>(null)
  const [mcpEditBaseline, setMcpEditBaseline] = useState<McpServerDetail | null>(null)
  const connectorEditRequestRef = useRef(0)
  const [searchQuery, setSearchQuery] = useState("")
  const [activeTab, setActiveTab] = useState<string>("all")
  const [mcpFormData, setMcpFormData] = useState<MCPServerFormData>({
    name: "",
    transport: "stdio",
    description: "",
    config: {},
    url: "",
    method: "GET",
    headers: {}
  })
  const [runtimeValidationError, setRuntimeValidationError] = useState<RuntimeConfigErrorKey | null>(null)

  const { t, tDynamic } = useI18n()
  const { user } = useAuth()
  const { getAppIcon } = useMcpApps()
  const isAdmin = Boolean(user?.is_admin)

  useEffect(() => {
    const oauthErrorMessage = searchParams.get("mcp_oauth_error_message")
    if (!oauthErrorMessage) return

    toast.error(oauthErrorMessage)
    const nextParams = new URLSearchParams(searchParams.toString())
    nextParams.delete("mcp_oauth_error")
    nextParams.delete("mcp_oauth_error_message")
    const nextQuery = nextParams.toString()
    router.replace(nextQuery ? `/tools?${nextQuery}` : "/tools", { scroll: false })
  }, [router, searchParams])

  useEffect(() => {
    loadTools()
    loadMCPServers()
  }, [])

  useEffect(() => {
    if (!user) {
      setConfigurableTools([])
      setSqlConnections([])
      return
    }

    void loadSqlConnections()
    if (!isAdmin) {
      setConfigurableTools([])
      return
    }

    void loadConfigurableTools()
  }, [isAdmin, user])

  const loadTools = async () => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/tools/available`)
      if (!response.ok) throw new Error("Failed to load tools")
      const data = await response.json()

      const transformedTools: Tool[] = data.tools.map((tool: any) => ({
        name: tool.name,
        description: tool.description,
        type: tool.type,
        category: tool.category,
        display_category: tool.display_category,  // Read display_category from API
        enabled: tool.enabled,
        requires_configuration: Boolean(tool.requires_configuration),
        status: tool.status,
        status_reason: tool.status_reason,
        config: tool.config,
        source: tool.type === 'basic' || tool.type === 'file' || tool.type === 'knowledge' ? 'builtin' : undefined,
        usage_count: tool.usage_count || 0
      }))

      setTools(transformedTools)
    } catch (error) {
      console.error("Failed to load tools:", error)
      setTools([])
    }
  }


  const loadMCPServers = async () => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/mcp/servers`)
      if (response.ok) {
        const servers = await response.json()
        setMcpServers(servers)
      }
    } catch (error) {
      console.error("Failed to load MCP servers:", error)
    }
  }

  const loadConfigurableTools = async () => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/tools/configurable`)
      if (!response.ok) {
        setConfigurableTools([])
        return
      }

      const data = await response.json()
      setConfigurableTools(data.tools || [])
    } catch (error) {
      console.error("Failed to load configurable tools:", error)
      setConfigurableTools([])
    }
  }

  const loadSqlConnections = async () => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/tools/sql-connections`)
      if (!response.ok) {
        setSqlConnections([])
        return
      }

      const data = await response.json()
      setSqlConnections(data.connections || [])
    } catch (error) {
      console.error("Failed to load SQL connections:", error)
      setSqlConnections([])
    }
  }

  const getCredentialStatusLabel = (source: 'db' | 'env' | 'none') => {
    if (source === 'db') return t('tools.credentials.status.db')
    if (source === 'env') return t('tools.credentials.status.env')
    return t('tools.credentials.status.none')
  }

  const openCredentialDialog = (toolName: string) => {
    const tool = configurableTools.find((item) => item.tool_name === toolName)
    if (!tool) return

    setEditingConfigTool(tool)
    setCredentialValues({})
    setIsCredentialDialogOpen(true)
  }

  const resetSqlForm = () => {
    setSqlFormName("")
    setSqlFormType('postgresql')
    setSqlFormHost("")
    setSqlFormPort(DEFAULT_PORTS.postgresql)
    setSqlFormDatabase("")
    setSqlFormUsername("")
    setSqlFormPassword("")
    setSqlFormParams("")
    setSqlFormSqlitePath("")
  }

  const openSqlManager = () => {
    resetSqlForm()
    setIsSqlManagerOpen(true)
  }

  const handleSaveCredentials = async () => {
    if (!editingConfigTool) return

    const payload: Record<string, { value: string }> = {}
    Object.entries(credentialValues).forEach(([fieldName, value]) => {
      const normalized = value.trim()
      if (normalized) payload[fieldName] = { value: normalized }
    })

    if (Object.keys(payload).length === 0) {
      toast.error(t('tools.credentials.validation.required'))
      return
    }

    setIsSavingCredentials(true)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/tools/${editingConfigTool.tool_name}/credentials`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ credentials: payload }),
      })

      if (!response.ok) {
        const err = await response.json()
        toast.error(err.detail || t('tools.credentials.saveFailed'))
        return
      }

      await loadConfigurableTools()
      await loadTools()
      setIsCredentialDialogOpen(false)
      toast.success(t('tools.credentials.saveSuccess'))
    } catch (error) {
      console.error('Failed to save credentials:', error)
      toast.error(t('tools.credentials.saveFailed'))
    } finally {
      setIsSavingCredentials(false)
    }
  }

  const handleSaveSqlConnection = async () => {
    const name = sqlFormName.trim()
    if (!name) {
      toast.error(t('tools.database.validation.required'))
      return
    }

    let connectionUrl = ''
    if (sqlFormType === 'sqlite') {
      const sqlitePath = sqlFormSqlitePath.trim()
      if (!sqlitePath) {
        toast.error(t('tools.database.validation.sqlitePathRequired'))
        return
      }
      connectionUrl = `sqlite:///${sqlitePath}`
    } else {
      const host = sqlFormHost.trim()
      const port = sqlFormPort.trim() || DEFAULT_PORTS[sqlFormType]
      const database = sqlFormDatabase.trim()
      const username = sqlFormUsername.trim()
      const password = sqlFormPassword.trim()
      const params = sqlFormParams.trim()

      if (!host || !database || !username) {
        toast.error(t('tools.database.validation.required'))
        return
      }

      const encodedUser = encodeURIComponent(username)
      const encodedPass = password ? `:${encodeURIComponent(password)}` : ''
      const auth = `${encodedUser}${encodedPass}@`
      const query = params ? `?${params.replace(/^\?/, '')}` : ''

      connectionUrl = `${sqlFormType}://${auth}${host}:${port}/${database}${query}`
    }

    setIsSavingSql(true)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/tools/sql-connections/${encodeURIComponent(name)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ connection_url: connectionUrl }),
      })

      if (!response.ok) {
        const err = await response.json()
        toast.error(err.detail || t('tools.database.saveFailed'))
        return
      }

      resetSqlForm()
      await loadSqlConnections()
      toast.success(t('tools.database.saveSuccess'))
    } catch (error) {
      console.error('Failed to save SQL connection:', error)
      toast.error(t('tools.database.saveFailed'))
    } finally {
      setIsSavingSql(false)
    }
  }

  const handleDeleteSqlConnection = async (name: string) => {
    if (pendingSqlDeletes[name]) return
    if (!confirm(t('tools.database.deleteConfirm', { name }))) return

    setPendingSqlDeletes((prev) => ({ ...prev, [name]: true }))
    try {
      const response = await apiRequest(`${getApiUrl()}/api/tools/sql-connections/${encodeURIComponent(name)}`, {
        method: 'DELETE',
      })

      if (!response.ok) {
        const err = await response.json()
        toast.error(err.detail || t('tools.database.deleteFailed'))
        return
      }

      await loadSqlConnections()
      toast.success(t('tools.database.deleteSuccess'))
    } catch (error) {
      console.error('Failed to delete SQL connection:', error)
      toast.error(t('tools.database.deleteFailed'))
    } finally {
      setPendingSqlDeletes((prev) => ({ ...prev, [name]: false }))
    }
  }

  const openCustomApiEditor = async (server: MCPServer): Promise<boolean> => {
    const requestId = connectorEditRequestRef.current + 1
    connectorEditRequestRef.current = requestId

    try {
      const response = await apiRequest(`${getApiUrl()}/api/custom-apis/${server.id}`)
      if (!response.ok) throw new Error(`Custom API detail request failed (${response.status})`)

      const detail = parseCustomApiDetail(await response.json())
      if (detail.id !== server.id) throw new Error("Custom API detail response ID mismatch")
      if (requestId !== connectorEditRequestRef.current) return false

      const editState = customApiDetailToEditState(detail)
      setEditingServer(server)
      setCustomApiEditBaseline(detail)
      setMcpEditBaseline(null)
      setRuntimeValidationError(null)
      setMcpFormData(editState.formData)
      setCustomApiEnv(editState.env)
      setIsMcpDialogOpen(true)
      return true
    } catch (error) {
      if (requestId !== connectorEditRequestRef.current) return false
      console.error("Failed to load Custom API detail:", error)
      toast.error(t('tools.mcp.dialog.customApiDetailFetchError'))
      return false
    }
  }

  const openMcpServerEditor = async (server: MCPServer): Promise<boolean> => {
    const requestId = connectorEditRequestRef.current + 1
    connectorEditRequestRef.current = requestId

    try {
      const response = await apiRequest(`${getApiUrl()}/api/mcp/servers/${server.id}`)
      if (!response.ok) throw new Error(`MCP server detail request failed (${response.status})`)

      const detail = parseMcpServerDetail(await response.json())
      if (detail.id !== server.id) throw new Error("MCP server detail response ID mismatch")
      if (requestId !== connectorEditRequestRef.current) return false

      const editState = mcpServerDetailToEditState(detail)
      setEditingServer(server)
      setCustomApiEditBaseline(null)
      setMcpEditBaseline(detail)
      setRuntimeValidationError(null)
      setMcpFormData(editState.formData)
      setIsMcpDialogOpen(true)
      return true
    } catch (error) {
      if (requestId !== connectorEditRequestRef.current) return false
      console.error("Failed to load MCP server detail:", error)
      toast.error(t('tools.mcp.dialog.mcpDetailFetchError'))
      return false
    }
  }

  useEffect(() => () => {
    connectorEditRequestRef.current += 1
  }, [])

  const handleEditMcpServer = async (server: MCPServer) => {
    // Check if this is an official integration (from library)
    const isOfficial = server.transport === 'oauth'

    if (isOfficial) {
      connectorEditRequestRef.current += 1
      const isGoogle = server.name.toLowerCase().includes('google') || server.name.toLowerCase() === 'gmail'
      const provider = server.provider || (isGoogle ? 'google' : 'linkedin')

      // Use app_id from backend if available, fallback to basic logic
      const appId = server.app_id || server.name.toLowerCase().replace(/\s+/g, '-')

      // We need to fetch the icon or use a generic one
      let icon = getAppIcon(server.name) || "";

      // Create an AppIntegration-like object for the dialog
      setEditingOfficialApp({
        id: appId, // Store the app ID for OAuth flow
        server_id: server.id, // Store the actual server ID for disconnect
        name: server.name,
        description: server.description || "",
        icon: icon,
        is_connected: true,
        provider: provider,
        connected_account: server.connected_account,
        is_custom: false,
        // This reconstruction path is only reached for oauth servers (gated
        // above); set auth_type explicitly so the settings dialog's isKeyBased
        // check stays correct if this path is ever reused for other transports.
        auth_type: "builtin_oauth"
      })
      setIsOfficialAppDialogOpen(true)
    } else if (server.transport === "custom_api") {
      await openCustomApiEditor(server)
    } else {
      await openMcpServerEditor(server)
    }
  }

  const handleSaveMcpServer = async () => {
    if (!mcpFormData.name.trim()) {
      toast.error(t('tools.mcp.alerts.nameRequired'))
      return
    }

    if (!isValidMcpName(mcpFormData.name)) {
      toast.error(t('tools.mcp.alerts.nameInvalidFormat') || "Name can only contain letters, numbers, hyphens and underscores");
      return;
    }

    let payload: any = { ...mcpFormData }
    const connectorType = payload.transport === "custom_api" ? "custom_api" : "mcp"
    const runtimeError = runtimeValidationError || getRuntimeConfigError(payload, connectorType)
    if (runtimeError) {
      toast.error(t(runtimeError))
      return
    }

    if (payload.transport === "custom_api") {
      if (!mcpFormData.url?.trim()) {
        toast.error(t('tools.mcp.alerts.urlRequired'));
        return;
      }
      if (editingServer && !customApiEditBaseline) {
        toast.error(t('tools.mcp.dialog.customApiDetailFetchError'))
        return
      }
      const buildResult = buildCustomApiPayload(
        payload,
        customApiEnv,
        editingServer ? customApiEditBaseline ?? undefined : undefined,
      );
      if (!buildResult.isValid) {
        toast.error(t(buildResult.errorKey || 'tools.mcp.alerts.atLeastOneSecret') || "At least one valid secret is required");
        return;
      }
      payload = buildResult.payload;
    } else {
      if (editingServer && !mcpEditBaseline) {
        toast.error(t('tools.mcp.dialog.mcpDetailFetchError'))
        return
      }
      payload = buildMcpServerPayload(
        payload,
        editingServer ? mcpEditBaseline ?? undefined : undefined,
      )
    }

    setIsLoading(true)
    try {
      const url = mcpFormData.transport === 'custom_api'
        ? (editingServer
          ? `${getApiUrl()}/api/custom-apis/${editingServer.id}`
          : `${getApiUrl()}/api/custom-apis`)
        : (editingServer
          ? `${getApiUrl()}/api/mcp/servers/${editingServer.id}`
          : `${getApiUrl()}/api/mcp/servers`);
      const method = editingServer ? 'PUT' : 'POST'
      const response = await apiRequest(url, {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      })
      if (response.ok) {
        await loadMCPServers()
        setIsMcpDialogOpen(false)
      } else {
        const error = await response.json()
        toast.error(error.detail || t('tools.mcp.alerts.saveFailed'))
      }
    } catch (error) {
      console.error("Failed to save MCP server:", error)
      toast.error(t('tools.mcp.alerts.saveFailed'))
    } finally {
      setIsLoading(false)
    }
  }

  const handleToggleToolEnabled = async (tool: Tool) => {
    if (pendingToolToggles[tool.name]) return

    setPendingToolToggles((prev) => ({ ...prev, [tool.name]: true }))
    try {
      const response = await apiRequest(`${getApiUrl()}/api/tools/${tool.name}/enabled`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !tool.enabled }),
      })
      if (!response.ok) {
        const err = await response.json()
        toast.error(err.detail || t('tools.policy.toggleFailed'))
        return
      }
      await loadTools()
      toast.success(t(tool.enabled ? 'tools.policy.toggleSuccessDisabled' : 'tools.policy.toggleSuccessEnabled'))
    } catch (error) {
      console.error("Failed to toggle tool enabled:", error)
      toast.error(t('tools.policy.toggleFailed'))
    } finally {
      setPendingToolToggles((prev) => ({ ...prev, [tool.name]: false }))
    }
  }


  const getCategoryLabel = (category: string) => {
    if (!category) return ""

    // Special cases for correct capitalization
    const categoryDisplayMap: Record<string, string> = {
      ppt: "PPT",
      pptx: "PPTX",
      ai: "AI",
      api: "API",
      llm: "LLM",
      ai2: "AI2",
    }

    // Check special cases first
    if (categoryDisplayMap[category]) {
      const key = `tools.categories.${category}`
      return tDynamic(key, categoryDisplayMap[category])
    }

    // Try translation
    const key = `tools.categories.${category}`
    const fallback = category.charAt(0).toUpperCase() + category.slice(1).replace(/_/g, ' ')
    return tDynamic(key, fallback)
  }

  const getToolIcon = (name: string, type: string, category?: string) => {
    const lowerName = name.toLowerCase()
    const lowerCategory = (category || "").toLowerCase()
    if (type === 'mcp') {
      const appIcon = getAppIcon(name)
      if (appIcon) {
        return <img src={appIcon} alt={name} className="h-6 w-6 rounded-sm object-contain" />
      }
      return <Server className="h-6 w-6 text-green-600" />
    }

    if (lowerName.includes('firecrawl')) return <Flame className="h-6 w-6 text-orange-500" />
    if (lowerName.includes('google')) return <Globe className="h-6 w-6 text-blue-500" />
    if (lowerName.includes('slack')) return <Hash className="h-6 w-6 text-purple-500" />

    if (lowerCategory === 'browser') return <Globe className="h-6 w-6 text-blue-500" />
    if (lowerCategory === 'file') return <FileText className="h-6 w-6 text-amber-500" />
    if (lowerCategory === 'knowledge') return <Book className="h-6 w-6 text-indigo-500" />
    if (lowerCategory === 'audio') return <Mic className="h-6 w-6 text-green-500" />
    if (lowerCategory === 'video') return <Video className="h-6 w-6 text-rose-500" />
    if (type === 'builtin' || lowerCategory === 'basic') return <Wrench className="h-6 w-6 text-slate-500" />

    return <Code className="h-6 w-6 text-slate-500" />
  }

  const getBadgeLabel = (tool: Tool) => {
    // Use display_category if available, otherwise fallback to category
    if (tool.display_category) return tool.display_category  // Already formatted correctly (PPT not Ppt)
    if (tool.category) return getCategoryLabel(tool.category)  // Fallback to translation
    return t('tools.badges.types.tool')
  }

  // Get unique categories
  const categories = Array.from(new Set(
    tools
      .map(t => t.category)
      .filter(Boolean)
      .filter(c => c !== 'mcp')
  )).sort()

  const filteredTools = tools.filter(t => {
    const matchesSearch = t.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      t.description.toLowerCase().includes(searchQuery.toLowerCase())

    // For 'all' tab, show everything (except MCP type which are handled separately if we want,
    // but here we filter 'mcp' type out from tools array usually, let's check filteredApiTools logic)
    // Actually, let's just filter based on category match

    const matchesTab = activeTab === 'all' || t.category === activeTab

    // Exclude MCP type tools from this list if they are handled by mcpServers,
    // but if tools[] contains valid tools we should show them.
    // Previous code excluded type === 'mcp' in filteredApiTools.

    return t.type !== 'mcp' && matchesSearch && matchesTab
  })

  const filteredMcpServers = mcpServers.filter(s =>
    s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (s.description || "").toLowerCase().includes(searchQuery.toLowerCase())
  )

  const configurableToolByName = configurableTools.reduce<Record<string, ConfigurableTool>>((acc, tool) => {
    acc[tool.tool_name] = tool
    return acc
  }, {})

  const getConfigToolNameForRuntimeTool = (tool: Tool): string | null => {
    if (tool.name === 'zhipu_web_search') return 'zhipu_web_search'
    if (tool.name === 'web_search') {
      const description = tool.description.toLowerCase()
      if (description.includes('tavily')) return 'tavily_web_search'
      return 'web_search'
    }

    return configurableToolByName[tool.name] ? tool.name : null
  }

  const runtimeConfigToolNames = new Set(
    tools
      .map((tool) => getConfigToolNameForRuntimeTool(tool))
      .filter((toolName): toolName is string => Boolean(toolName))
  )

  const filteredSearchProviderTools = configurableTools.filter((tool) => {
    const searchLower = searchQuery.toLowerCase()
    const matchesSearch =
      !searchLower ||
      tool.tool_name.toLowerCase().includes(searchLower) ||
      (tool.display_name || tool.tool_name).toLowerCase().includes(searchLower) ||
      Object.values(tool.fields).some((field) => field.label.toLowerCase().includes(searchLower))

    return !runtimeConfigToolNames.has(tool.tool_name) && matchesSearch
  })

  const getConfigurableToolDescription = (tool: ConfigurableTool) => {
    const toolName = tool.tool_name
    if (toolName === 'zhipu_web_search') {
      return 'Configure Zhipu Web Search credentials to enable this provider in the runtime tool list.'
    }
    if (toolName === 'tavily_web_search') {
      return 'Configure Tavily credentials to enable web search without Google setup.'
    }
    if (toolName === 'web_search') {
      return 'Configure Google Search credentials to enable the web search runtime tool.'
    }
    return t('tools.credentials.setup.description')
  }

  const ConfigurableToolCard = ({ tool }: { tool: ConfigurableTool }) => {
    return (
      <div className="flex min-w-0 flex-col overflow-hidden rounded-[14px] border border-border bg-card p-5">
        <div className="mb-3 flex items-start justify-between">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-muted/50">
            <Globe className="h-5 w-5 text-slate-500" />
          </div>
          <span className={tool.configured
            ? "rounded-full bg-[rgba(28,202,91,0.12)] px-2 py-0.5 text-[10px] font-semibold text-[rgb(21,157,71)]"
            : "rounded-full bg-muted px-2 py-0.5 text-[10px] font-semibold text-muted-foreground"}>
            {tool.configured ? t('tools.credentials.configured') : t('tools.credentials.notConfigured')}
          </span>
        </div>
        <div className="mb-1 text-[9.5px] font-bold uppercase tracking-[0.07em] text-muted-foreground">Basic</div>
        <h3 className="mb-2 text-[14px] font-bold tracking-[-0.02em] text-foreground">{tool.display_name || tool.tool_name}</h3>
        <p className="mb-4 flex-1 text-[12px] leading-[1.5] text-muted-foreground line-clamp-3">{getConfigurableToolDescription(tool)}</p>
        <div className="border-t border-border pt-3">
          <button
            className="w-full rounded-lg border border-[rgba(60,131,246,0.28)] px-[14px] py-[6px] text-[11px] font-semibold text-[rgb(60,131,246)] hover:bg-[rgba(60,131,246,0.06)] transition-colors"
            onClick={() => openCredentialDialog(tool.tool_name)}
          >
            {t('tools.credentials.configure')}
          </button>
        </div>
      </div>
    )
  }

  const ToolCard = ({ tool }: { tool: Tool }) => {
    const label = getBadgeLabel(tool)
    const icon = getToolIcon(tool.name, tool.type, tool.category)
    const configToolName = getConfigToolNameForRuntimeTool(tool)
    const configurableTool = configToolName ? configurableToolByName[configToolName] : undefined
    const canConfigureCredentials = isAdmin && Boolean(configurableTool)
    const canManageSqlConnections = Boolean(user) && tool.category === 'database' && Boolean(tool.requires_configuration)
    const hasSecondaryAction = canConfigureCredentials || canManageSqlConnections
    const isTogglePending = Boolean(pendingToolToggles[tool.name])
    const configButtonLabel = canConfigureCredentials
      ? t('tools.credentials.configure')
      : t('tools.database.manageConnections')

    return (
      <div className="flex min-w-0 flex-col overflow-hidden rounded-[14px] border border-border bg-card p-5">
        <div className="mb-3 flex items-start justify-between">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-muted/50">
            {icon}
          </div>
          <span className={tool.enabled
            ? "rounded-full bg-[rgba(28,202,91,0.12)] px-2 py-0.5 text-[10px] font-semibold text-[rgb(21,157,71)]"
            : "rounded-full bg-muted px-2 py-0.5 text-[10px] font-semibold text-muted-foreground"}>
            {tool.enabled ? t('tools.policy.enabled') : t('tools.policy.disabled')}
          </span>
        </div>
        <div className="mb-1 truncate text-[9.5px] font-bold uppercase tracking-[0.07em] text-muted-foreground">{label}</div>
        <h3 className="mb-2 truncate text-[14px] font-bold tracking-[-0.02em] text-foreground">{tool.name}</h3>
        <p className="mb-4 flex-1 text-[12px] leading-[1.5] text-muted-foreground line-clamp-3 break-words">{tool.description}</p>
        <div className="flex flex-wrap items-center gap-1 mb-3">
          {configurableTool && (
            <span className={configurableTool.configured
              ? "rounded-full bg-[rgba(28,202,91,0.12)] px-2 py-0.5 text-[10px] font-semibold text-[rgb(21,157,71)]"
              : "rounded-full bg-muted px-2 py-0.5 text-[10px] font-semibold text-muted-foreground"}>
              {configurableTool.configured ? t('tools.credentials.configured') : t('tools.credentials.notConfigured')}
            </span>
          )}
          {canManageSqlConnections && (
            <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] font-semibold text-muted-foreground">
              {sqlConnections.length} {t('tools.database.connectionBadge')}
            </span>
          )}
          <span className="ml-auto text-[10px] text-muted-foreground">{t('tools.list.usedByAgents', { count: tool.usage_count || 0 })}</span>
        </div>
        <div className="flex gap-2 border-t border-border pt-3">
          {hasSecondaryAction && (
            <button
              className="flex-1 rounded-lg border border-[rgba(60,131,246,0.28)] px-[14px] py-[6px] text-[11px] font-semibold text-[rgb(60,131,246)] hover:bg-[rgba(60,131,246,0.06)] transition-colors"
              onClick={() => {
                if (canConfigureCredentials && configToolName) { openCredentialDialog(configToolName); return }
                openSqlManager()
              }}
            >
              {configButtonLabel}
            </button>
          )}
          {isAdmin && (
            <button
              className={`rounded-lg border border-[rgba(60,131,246,0.28)] px-[14px] py-[6px] text-[11px] font-semibold text-[rgb(60,131,246)] hover:bg-[rgba(60,131,246,0.06)] transition-colors disabled:opacity-50 ${hasSecondaryAction ? 'flex-1' : 'w-full'}`}
              onClick={() => handleToggleToolEnabled(tool)}
              disabled={isTogglePending}
            >
              {isTogglePending && <Loader2 className="mr-1 inline h-3 w-3 animate-spin" />}
              {tool.enabled ? t('tools.policy.disableAction') : t('tools.policy.enableAction')}
            </button>
          )}
        </div>
      </div>
    )
  }

  const MCPServerCard = ({ server }: { server: MCPServer }) => {
    const icon = getToolIcon(server.name, 'mcp', 'mcp')
    return (
      <div
        className="flex min-w-0 cursor-pointer flex-col overflow-hidden rounded-[14px] border border-border bg-card p-5 transition-all hover:border-primary/40 hover:shadow-sm"
        onClick={() => handleEditMcpServer(server)}
      >
        <div className="mb-3 flex items-start justify-between">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-muted/50">
            {icon}
          </div>
          <span className="rounded-full bg-[rgba(28,202,91,0.12)] px-2 py-0.5 text-[10px] font-semibold text-[rgb(21,157,71)]">
            {t('tools.mcp.badge')}
          </span>
        </div>
        <div className="mb-1 truncate text-[9.5px] font-bold uppercase tracking-[0.07em] text-muted-foreground">
          {server.transport}
        </div>
        <h3 className="mb-2 truncate text-[14px] font-bold tracking-[-0.02em] text-foreground">{server.name}</h3>
        <p className="flex-1 break-words text-[12px] leading-[1.5] text-muted-foreground line-clamp-3">
          {server.description || t('tools.list.noDescription')}
        </p>
      </div>
    )
  }

  const allTabs = [
    { id: 'all', label: t('tools.tabs.all') },
    ...categories.map(cat => ({ id: cat, label: getCategoryLabel(cat) })),
    { id: 'mcp', label: t('tools.tabs.connectors') },
  ]

  const totalCount = filteredTools.length + (activeTab === 'all' || activeTab === 'mcp' ? filteredMcpServers.length : 0) + (isAdmin && (activeTab === 'all' || activeTab === 'basic') ? filteredSearchProviderTools.length : 0)

  return (
    <div className="flex h-full flex-col overflow-y-auto p-[48px_52px_72px]">
      {/* Hero — bleeds to edges */}
      <div className="m-[-48px_-52px_36px] flex flex-col items-center gap-[14px] border-b border-border bg-background p-[48px_64px_44px] text-center">
        <div>
          <div className="mb-1 text-[30px] font-extrabold tracking-[-0.04em] text-foreground">
            {t('tools.header.title')}
          </div>
          <div className="text-[13px] tracking-[0.01em] text-muted-foreground">
            {t('tools.header.description')}
          </div>
        </div>

        {/* Pill search bar */}
        <div className="flex w-full max-w-[620px]">
          <div className="flex w-full items-center gap-3 rounded-full border-[1.5px] border-border bg-background px-5 py-[13px] shadow-[0_1px_3px_rgba(0,0,0,0.06)]">
            <Search className="h-[18px] w-[18px] flex-shrink-0 text-muted-foreground" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={t('tools.list.searchPlaceholder')}
              className="w-full border-none bg-transparent text-sm text-foreground outline-none"
            />
          </div>
        </div>
      </div>

      {/* Pill tabs row */}
      <div className="mb-7 grid grid-cols-[auto_1fr_auto] items-start gap-3">
        <Button
          className="h-[30px] shrink-0 whitespace-nowrap rounded-lg bg-primary px-3 text-xs text-primary-foreground hover:bg-primary/90"
          size="sm"
          onClick={() => setIsConnectMcpOpen(true)}
        >
          <Plus className="h-3.5 w-3.5 mr-1" />
          {t('tools.mcp.addConnector')}
        </Button>

        <div role="tablist" aria-label={t('tools.header.title')} className="flex flex-wrap items-center justify-center gap-1.5">
          {allTabs.map((tab) => {
            const isActive = activeTab === tab.id
            return (
              <button
                key={tab.id}
                role="tab"
                aria-selected={isActive}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "rounded-full px-4 py-1.5 text-xs transition-all duration-150",
                  isActive
                    ? "border-none bg-[linear-gradient(135deg,rgb(48,64,207),rgb(60,131,246))] font-semibold text-white"
                    : "border border-[rgba(60,131,246,0.16)] bg-transparent font-medium text-muted-foreground"
                )}
              >
                {tab.label}
              </button>
            )
          })}
        </div>

        <span className="shrink-0 whitespace-nowrap rounded-full border border-[rgba(60,131,246,0.18)] bg-[rgba(60,131,246,0.08)] px-[10px] py-[3px] text-[11px] font-semibold text-[rgb(60,131,246)]">
          {totalCount === 1
            ? t('tools.tabs.countOne', { count: totalCount })
            : t('tools.tabs.countOther', { count: totalCount })}
        </span>
      </div>

      {/* Add connector / edit dialogs */}
      <Dialog
        open={isMcpDialogOpen}
        onOpenChange={(nextOpen) => {
          setIsMcpDialogOpen(nextOpen)
          if (!nextOpen) {
            connectorEditRequestRef.current += 1
            setCustomApiEditBaseline(null)
            setMcpEditBaseline(null)
            setRuntimeValidationError(null)
          }
        }}
      >
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              {editingServer ? mcpFormData.transport === 'custom_api' ? t('tools.mcp.dialog.editCustomApi') : t('tools.mcp.dialog.editTitle') : mcpFormData.transport === 'custom_api' ? t('tools.mcp.dialog.addCustomApi') : t('tools.mcp.dialog.addTitle')}
            </DialogTitle>
            <DialogDescription>
              {mcpFormData.transport === 'custom_api' ? t('tools.mcp.dialog.customApiDescription') : t('tools.mcp.dialog.description')}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            {mcpFormData.transport === 'custom_api' ? (
              <CustomApiForm
                key={editingServer?.id || 'new'}
                mcpFormData={mcpFormData}
                setMcpFormData={setMcpFormData}
                customApiEnv={customApiEnv}
                setCustomApiEnv={setCustomApiEnv}
                onRuntimeValidationErrorChange={setRuntimeValidationError}
                originalEnvObj={customApiEditBaseline?.env ?? {}}
              />
            ) : (
              <CustomMcpForm
                mcpFormData={mcpFormData}
                setMcpFormData={setMcpFormData}
                serverId={editingServer?.id}
                onOAuthStatusChange={loadMCPServers}
                onRuntimeValidationErrorChange={setRuntimeValidationError}
              />
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsMcpDialogOpen(false)}>
              {t('tools.mcp.buttons.cancel')}
            </Button>
            <Button
              onClick={handleSaveMcpServer}
              disabled={
                isLoading ||
                !mcpFormData.name.trim() ||
                (mcpFormData.transport === 'custom_api' && customApiEnv.length > 0 && customApiEnv.some(env => !env.key.trim() || !env.value.trim()))
              }
            >
              {isLoading && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              {t('tools.mcp.buttons.save')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Tools grid */}
      <div className="grid grid-cols-1 gap-3.5 md:grid-cols-2 xl:grid-cols-4">
        {isAdmin && (activeTab === 'all' || activeTab === 'basic') && filteredSearchProviderTools.map((tool) => (
          <ConfigurableToolCard key={`config-${tool.tool_name}`} tool={tool} />
        ))}

        {activeTab !== 'mcp' && filteredTools.map(tool => (
          <ToolCard key={`${tool.category}-${tool.name}`} tool={tool} />
        ))}

        {(activeTab === 'all' || activeTab === 'mcp') && filteredMcpServers.map(server => (
          <MCPServerCard key={`mcp-${server.id}`} server={server} />
        ))}

        {(activeTab !== 'mcp' && filteredTools.length === 0 &&
          filteredSearchProviderTools.length === 0 &&
          ((activeTab !== 'all' && activeTab !== 'mcp') || (activeTab === 'all' && filteredMcpServers.length === 0)) &&
          (activeTab !== 'mcp' || filteredMcpServers.length === 0)) && (
            <div className="col-span-full flex justify-center">
              <EmptyState />
            </div>
          )}

        {activeTab === 'mcp' && filteredMcpServers.length === 0 && (
          <div className="col-span-full flex justify-center">
            <EmptyState />
          </div>
        )}
      </div>

      <Dialog open={isSqlManagerOpen} onOpenChange={setIsSqlManagerOpen}>
        <DialogContent className="max-w-4xl max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{t('tools.database.dialog.title')}</DialogTitle>
            <DialogDescription>{t('tools.database.dialog.description')}</DialogDescription>
          </DialogHeader>

          <div className="space-y-6">
            <div className="space-y-4">
              <h3 className="font-medium">{t('tools.database.existingConnections')}</h3>
              {sqlConnections.length === 0 ? (
                <div className="flex justify-center py-4">
                  <ConfigEmptyState
                    title={t('tools.database.empty.title')}
                    description={t('tools.database.empty.description')}
                  />
                </div>
              ) : (
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  {sqlConnections.map((item) => {
                    const isDeleting = Boolean(pendingSqlDeletes[item.name])

                    return (
                      <Card key={item.name} className="group border-border/60">
                        <CardContent className="p-5">
                          <div className="mb-4 flex items-start justify-between gap-3">
                            <div className="flex items-start gap-3 min-w-0">
                              <div className="mt-0.5 rounded-lg bg-muted/60 p-2.5 h-fit">
                                <Database className="h-5 w-5 text-slate-600" />
                              </div>
                              <div className="min-w-0">
                                <h3 className="truncate font-semibold text-base text-foreground">{item.name}</h3>
                                <div className="mt-1 flex flex-wrap items-center gap-2">
                                  <Badge variant="outline" className="text-[11px]">
                                    {t('tools.database.connectionBadge')}
                                  </Badge>
                                  <Badge variant={item.source === 'db' ? 'secondary' : 'outline'} className="text-[11px]">
                                    {t(`tools.credentials.status.${item.source}`)}
                                  </Badge>
                                </div>
                              </div>
                            </div>

                            {item.source === 'db' ? (
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 opacity-80 group-hover:opacity-100"
                                onClick={() => handleDeleteSqlConnection(item.name)}
                                disabled={isDeleting}
                              >
                                {isDeleting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                              </Button>
                            ) : null}
                          </div>

                          <div className="space-y-2">
                            <p className="text-xs font-medium text-muted-foreground">{t('tools.database.maskedValue')}</p>
                            <div className="rounded-md border border-border/70 bg-muted/30 px-3 py-2">
                              <p className="break-all text-xs leading-relaxed text-foreground/80">{item.masked || '--'}</p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    )
                  })}
                </div>
              )}
            </div>

            <div className="space-y-4 rounded-lg border border-border/70 p-4">
              <h3 className="font-medium">{t('tools.database.addConnection')}</h3>

              <div className="space-y-2">
                <Label htmlFor="sql-conn-name">{t('tools.database.connectionName')}</Label>
                <Input
                  id="sql-conn-name"
                  value={sqlFormName}
                  onChange={(e) => setSqlFormName(e.target.value)}
                  placeholder={t('tools.database.connectionNamePlaceholder')}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="sql-conn-type">{t('tools.database.dbType')}</Label>
                <Select
                  value={sqlFormType}
                  onValueChange={(value: string) => {
                    const typed = value as SqlDbType
                    setSqlFormType(typed)
                    if (typed !== 'sqlite') {
                      setSqlFormPort(DEFAULT_PORTS[typed])
                    }
                  }}
                  options={[
                    { value: 'postgresql', label: t('tools.database.types.postgresql') },
                    { value: 'mysql', label: t('tools.database.types.mysql') },
                    { value: 'mariadb', label: t('tools.database.types.mariadb') },
                    { value: 'mssql', label: t('tools.database.types.mssql') },
                    { value: 'sqlite', label: t('tools.database.types.sqlite') },
                  ]}
                  placeholder={t('tools.database.dbType')}
                />
              </div>

              {sqlFormType === 'sqlite' ? (
                <div className="space-y-2">
                  <Label htmlFor="sql-conn-sqlite-path">{t('tools.database.sqlitePath')}</Label>
                  <Input
                    id="sql-conn-sqlite-path"
                    value={sqlFormSqlitePath}
                    onChange={(e) => setSqlFormSqlitePath(e.target.value)}
                    placeholder={t('tools.database.sqlitePathPlaceholder')}
                  />
                </div>
              ) : (
                <>
                  <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label htmlFor="sql-conn-host">{t('tools.database.host')}</Label>
                      <Input
                        id="sql-conn-host"
                        value={sqlFormHost}
                        onChange={(e) => setSqlFormHost(e.target.value)}
                        placeholder={t('tools.database.hostPlaceholder')}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="sql-conn-port">{t('tools.database.port')}</Label>
                      <Input
                        id="sql-conn-port"
                        value={sqlFormPort}
                        onChange={(e) => setSqlFormPort(e.target.value)}
                        placeholder={t('tools.database.portPlaceholder')}
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="sql-conn-database">{t('tools.database.databaseName')}</Label>
                    <Input
                      id="sql-conn-database"
                      value={sqlFormDatabase}
                      onChange={(e) => setSqlFormDatabase(e.target.value)}
                      placeholder={t('tools.database.databaseNamePlaceholder')}
                    />
                  </div>

                  <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label htmlFor="sql-conn-username">{t('tools.database.username')}</Label>
                      <Input
                        id="sql-conn-username"
                        value={sqlFormUsername}
                        onChange={(e) => setSqlFormUsername(e.target.value)}
                        placeholder={t('tools.database.usernamePlaceholder')}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="sql-conn-password">{t('tools.database.password')}</Label>
                      <Input
                        id="sql-conn-password"
                        type="password"
                        value={sqlFormPassword}
                        onChange={(e) => setSqlFormPassword(e.target.value)}
                        placeholder={t('tools.database.passwordPlaceholder')}
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="sql-conn-params">{t('tools.database.params')}</Label>
                    <Input
                      id="sql-conn-params"
                      value={sqlFormParams}
                      onChange={(e) => setSqlFormParams(e.target.value)}
                      placeholder={t('tools.database.paramsPlaceholder')}
                    />
                  </div>
                </>
              )}
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsSqlManagerOpen(false)}>
              {t('tools.mcp.buttons.cancel')}
            </Button>
            <Button onClick={handleSaveSqlConnection} disabled={isSavingSql}>
              {isSavingSql && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {t('tools.database.save')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={isCredentialDialogOpen} onOpenChange={setIsCredentialDialogOpen}>
        <DialogContent className="max-w-xl">
          <DialogHeader>
            <DialogTitle>{t('tools.credentials.dialog.title')}</DialogTitle>
            <DialogDescription>
              {editingConfigTool
                ? t('tools.credentials.dialog.description', {
                  tool: editingConfigTool.display_name || editingConfigTool.tool_name,
                })
                : ''}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            {editingConfigTool &&
              Object.entries(editingConfigTool.fields).map(([fieldName, field]) => (
                <div key={fieldName} className="space-y-2">
                  <Label htmlFor={`cred-${fieldName}`}>
                    {field.label}
                    {field.required ? ' *' : ''}
                  </Label>
                  <Input
                    id={`cred-${fieldName}`}
                    type={field.secret ? 'password' : 'text'}
                    value={credentialValues[fieldName] || ''}
                    placeholder={field.masked || getCredentialStatusLabel(field.source)}
                    onChange={(e) =>
                      setCredentialValues((prev) => ({ ...prev, [fieldName]: e.target.value }))
                    }
                  />
                  <p className="text-xs text-muted-foreground">
                    {t('tools.credentials.currentSource')}: {getCredentialStatusLabel(field.source)}
                  </p>
                </div>
              ))}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsCredentialDialogOpen(false)}>
              {t('tools.mcp.buttons.cancel')}
            </Button>
            <Button onClick={handleSaveCredentials} disabled={isSavingCredentials}>
              {isSavingCredentials && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {t('tools.credentials.save')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <ConnectMcpDialog
        open={isConnectMcpOpen}
        onOpenChange={setIsConnectMcpOpen}
        selectedMcpServers={[]} // No pre-selection logic on tools page
        onSuccess={loadMCPServers}
      />
      <OfficialMcpSettingsDialog
        open={isOfficialAppDialogOpen}
        onOpenChange={(nextOpen) => {
          if (!nextOpen) connectorEditRequestRef.current += 1
          setIsOfficialAppDialogOpen(nextOpen)
        }}
        app={editingOfficialApp}
        isGloballyConnected={true} // In tools page, official apps are always already connected
        onSuccess={loadMCPServers}
      />
    </div>
  )
}

export default function ToolsPage() {
  return (
    <Suspense fallback={null}>
      <ToolsPageContent />
    </Suspense>
  )
}

function EmptyState() {
  const { t } = useI18n()
  return (
    <div className="mx-auto w-full max-w-2xl min-h-[220px] flex flex-col items-center justify-center text-center py-16 text-muted-foreground border border-dashed rounded-lg">
      <Wrench className="h-10 w-10 mx-auto mb-4 opacity-50" />
      <div className="font-medium mb-1">{t('tools.list.empty.title')}</div>
      <div className="text-sm">{t('tools.list.empty.description')}</div>
    </div>
  )
}

function ConfigEmptyState({
  title,
  description,
}: {
  title: string
  description: string
}) {
  return (
    <div className="mx-auto w-full max-w-2xl min-h-[180px] flex flex-col items-center justify-center text-center py-10 text-muted-foreground border border-dashed rounded-lg">
      <Wrench className="h-8 w-8 mx-auto mb-3 opacity-50" />
      <div className="font-medium mb-1">{title}</div>
      <div className="text-sm">{description}</div>
    </div>
  )
}
