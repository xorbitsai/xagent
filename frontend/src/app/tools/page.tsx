"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { SearchInput } from "@/components/ui/search-input"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
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
  Database,
  Trash2,
} from "lucide-react"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { ConnectMcpDialog, AppIntegration } from "@/components/mcp/connect-mcp-dialog"
import { OfficialMcpSettingsDialog } from "@/components/mcp/official-mcp-settings-dialog"
import { useI18n } from "@/contexts/i18n-context"
import { useAuth } from "@/contexts/auth-context"
import { useMcpApps } from "@/contexts/mcp-apps-context"
import { toast } from "sonner"

interface Tool {
  name: string
  description: string
  type: 'builtin' | 'mcp' | 'image' | 'vision' | 'audio'
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
}

interface TransportConfig {
  value: string
  label: string
  description: string
  fields: Array<{
    name: string
    label: string
    type: 'text' | 'number' | 'textarea' | 'select'
    required: boolean
    placeholder?: string
    options?: Array<{ value: string; label: string }>
  }>
}

interface MCPServerFormData {
  name: string
  transport: string
  description: string
  config: Record<string, any>
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

export default function ToolsPage() {
  const [tools, setTools] = useState<Tool[]>([])
  const [mcpServers, setMcpServers] = useState<MCPServer[]>([])
  const [transports, setTransports] = useState<TransportConfig[]>([])
  const [configurableTools, setConfigurableTools] = useState<ConfigurableTool[]>([])
  const [sqlConnections, setSqlConnections] = useState<SqlConnectionItem[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isConnectMcpOpen, setIsConnectMcpOpen] = useState(false)
  const [isOfficialAppDialogOpen, setIsOfficialAppDialogOpen] = useState(false)
  const [editingOfficialApp, setEditingOfficialApp] = useState<AppIntegration | null>(null)
  const [isMcpDialogOpen, setIsMcpDialogOpen] = useState(false)
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
  const [searchQuery, setSearchQuery] = useState("")
  const [activeTab, setActiveTab] = useState<string>("all")
  const [mcpFormData, setMcpFormData] = useState<MCPServerFormData>({
    name: "",
    transport: "stdio",
    description: "",
    config: {}
  })

  const { t } = useI18n()
  const { user } = useAuth()
  const { getAppIcon } = useMcpApps()
  const isAdmin = Boolean(user?.is_admin)

  useEffect(() => {
    loadTools()
    loadMCPServers()
    loadTransports()
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

  const loadTransports = async () => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/mcp/transports`)
      if (response.ok) {
        const data = await response.json()
        // Transform API data to match expected format
        const transformedTransports = (data.transports || []).map((transport: any) => ({
          value: transport.id,
          label: transport.name,
          description: transport.description,
          fields: (transport.config_fields || []).map((field: any) => ({
            name: field.name,
            label: field.description,
            type: field.type === 'string' ? 'text' : field.type === 'array' ? 'textarea' : field.type,
            required: field.required,
            placeholder: t('tools.mcp.form.fieldPlaceholderPrefix', { field: field.description })
          }))
        }))
        setTransports(transformedTransports)
      } else {
        // Fallback transports if API fails
        setTransports([
          { value: "stdio", label: t('tools.mcp.transports.stdio.label'), description: t('tools.mcp.transports.stdio.description'), fields: [] },
          { value: "sse", label: t('tools.mcp.transports.sse.label'), description: t('tools.mcp.transports.sse.description'), fields: [] },
          { value: "websocket", label: t('tools.mcp.transports.websocket.label'), description: t('tools.mcp.transports.websocket.description'), fields: [] },
          { value: "streamable_http", label: t('tools.mcp.transports.streamable_http.label'), description: t('tools.mcp.transports.streamable_http.description'), fields: [] }
        ])
      }
    } catch (error) {
      console.error("Failed to load transports:", error)
      // Fallback transports if API fails
      setTransports([
        { value: "stdio", label: t('tools.mcp.transports.stdio.label'), description: t('tools.mcp.transports.stdio.description'), fields: [] },
        { value: "sse", label: t('tools.mcp.transports.sse.label'), description: t('tools.mcp.transports.sse.description'), fields: [] },
        { value: "websocket", label: t('tools.mcp.transports.websocket.label'), description: t('tools.mcp.transports.websocket.description'), fields: [] },
        { value: "streamable_http", label: t('tools.mcp.transports.streamable_http.label'), description: t('tools.mcp.transports.streamable_http.description'), fields: [] }
      ])
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

  const handleEditMcpServer = (server: MCPServer) => {
    // Check if this is an official integration (from library)
    const isOfficial = server.transport === 'oauth'

    if (isOfficial) {
      // Use provider from backend if available, fallback to basic logic
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
        is_custom: false
      })
      setIsOfficialAppDialogOpen(true)
    } else {
      // For custom servers, show the detail dialog first
      setEditingOfficialApp({
        id: server.id.toString(),
        server_id: server.id,
        name: server.name,
        description: server.description || "",
        icon: "", // Use generic icon
        is_connected: true,
        provider: "custom",
        is_custom: true,
        server: server
      })
      setIsOfficialAppDialogOpen(true)
    }
  }

  const handleSaveMcpServer = async () => {
    if (!mcpFormData.name.trim()) {
      toast.error(t('tools.mcp.alerts.nameRequired'))
      return
    }
    setIsLoading(true)
    try {
      const url = editingServer
        ? `${getApiUrl()}/api/mcp/servers/${editingServer.id}`
        : `${getApiUrl()}/api/mcp/servers`
      const method = editingServer ? 'PUT' : 'POST'
      const response = await apiRequest(url, {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(mcpFormData)
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
      const translated = t(key)
      // If translation exists and is not just the key itself, use it
      if (translated !== key) {
        return translated
      }
      // Otherwise use the special case mapping
      return categoryDisplayMap[category]
    }

    // Try translation
    const key = `tools.categories.${category}`
    const translated = t(key)
    if (translated === key) {
      // Fallback: capitalize and replace underscores
      return category.charAt(0).toUpperCase() + category.slice(1).replace(/_/g, ' ')
    }
    return translated
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
    if (type === 'builtin' || lowerCategory === 'basic') return <Wrench className="h-6 w-6 text-slate-500" />

    return <Code className="h-6 w-6 text-slate-500" />
  }

  const getBadgeInfo = (tool: Tool) => {
    // Use display_category if available, otherwise fallback to category
    const categoryDisplay = tool.display_category
      ? tool.display_category  // Already formatted correctly (PPT not Ppt)
      : tool.category
        ? getCategoryLabel(tool.category)  // Fallback to translation
        : t('tools.badges.types.tool');

    return { label: categoryDisplay, variant: "secondary" as const }
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
      <Card className="hover:shadow-md transition-all duration-300 border-border/50 hover:border-primary hover:-translate-y-1">
        <CardContent className="p-6">
          <div className="flex items-start justify-between mb-4">
            <div className="flex gap-4">
              <div className="mt-1 bg-muted/50 p-3 rounded-lg h-fit">
                <Globe className="h-6 w-6 text-slate-500" />
              </div>
              <div>
                <h3 className="font-semibold text-base mb-1">{tool.display_name || tool.tool_name}</h3>
                <Badge variant="secondary" className="font-normal text-xs bg-muted text-muted-foreground hover:bg-muted">
                  Basic
                </Badge>
              </div>
            </div>
          </div>

          <p className="text-sm text-muted-foreground mb-6 line-clamp-2 h-10">
            {getConfigurableToolDescription(tool)}
          </p>

          <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <Badge variant={tool.configured ? 'secondary' : 'outline'}>
              {tool.configured ? t('tools.credentials.configured') : t('tools.credentials.notConfigured')}
            </Badge>
          </div>

          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              className="w-full"
              onClick={() => openCredentialDialog(tool.tool_name)}
            >
              {t('tools.credentials.configure')}
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  const ToolCard = ({ tool }: { tool: Tool }) => {
    const { label, variant } = getBadgeInfo(tool)
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
      <Card className="hover:shadow-md transition-all duration-300 border-border/50 hover:border-primary hover:-translate-y-1">
        <CardContent className="p-6">
          <div className="flex items-start justify-between mb-4">
            <div className="flex gap-4">
              <div className="mt-1 bg-muted/50 p-3 rounded-lg h-fit">
                {icon}
              </div>
              <div>
                <h3 className="font-semibold text-base mb-1">{tool.name}</h3>
                <Badge variant={variant} className="font-normal text-xs bg-muted text-muted-foreground hover:bg-muted">
                  {label}
                </Badge>
              </div>
            </div>
          </div>

          <p className="text-sm text-muted-foreground mb-6 line-clamp-2 h-10">
            {tool.description}
          </p>

          <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <Badge variant={tool.enabled ? 'secondary' : 'outline'}>
              {tool.enabled ? t('tools.policy.enabled') : t('tools.policy.disabled')}
            </Badge>
            {configurableTool && (
              <Badge variant={configurableTool.configured ? 'secondary' : 'outline'}>
                {configurableTool.configured
                  ? t('tools.credentials.configured')
                  : t('tools.credentials.notConfigured')}
              </Badge>
            )}
            {canManageSqlConnections && (
              <Badge variant="outline">
                {`${sqlConnections.length} ${t('tools.database.connectionBadge')}`}
              </Badge>
            )}
            <span className="ml-auto">{t('tools.list.usedByAgents', { count: tool.usage_count || 0 })}</span>
          </div>

          <div className="flex gap-2">
            {hasSecondaryAction && (
              <Button
                variant="outline"
                size="sm"
                className="flex-1"
                onClick={() => {
                  if (canConfigureCredentials && configToolName) {
                    openCredentialDialog(configToolName)
                    return
                  }
                  openSqlManager()
                }}
              >
                {configButtonLabel}
              </Button>
            )}
            <Button
              variant="outline"
              size="sm"
              className={hasSecondaryAction ? 'flex-1' : 'w-full'}
              onClick={() => handleToggleToolEnabled(tool)}
              disabled={isTogglePending}
            >
              {isTogglePending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {tool.enabled ? t('tools.policy.disableAction') : t('tools.policy.enableAction')}
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  const MCPServerCard = ({ server }: { server: MCPServer }) => {
    return (
      <Card className="hover:shadow-md cursor-pointer transition-all duration-300 border-border/50 hover:border-primary hover:-translate-y-1" onClick={() => handleEditMcpServer(server)}>
        <CardContent className="p-6">
          <div className="flex items-start justify-between mb-4">
            <div className="flex gap-4">
              <div className="mt-1 bg-muted/50 p-3 rounded-lg h-fit">
                {getToolIcon(server.name, 'mcp', 'mcp')}
              </div>
              <div>
                <h3 className="font-semibold text-base mb-1">{server.name}</h3>
                <Badge variant="secondary" className="font-normal text-xs bg-muted text-muted-foreground hover:bg-muted">
                  {t('tools.mcp.badge')}
                </Badge>
              </div>
            </div>
          </div>

          <p className="text-sm text-muted-foreground mb-6 line-clamp-2 h-10">
            {server.description || t('tools.list.noDescription')}
          </p>

          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span className="capitalize">{server.transport} {t('tools.list.transport')}</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="w-full p-6 space-y-8 overflow-y-auto h-[calc(100vh-2rem)]">
      {/* Header */}
      <div className="flex justify-between items-center mb-8">
        <div className="space-y-1">
          <h1 className="text-3xl font-bold mb-1">{t('tools.header.title')}</h1>
          <p className="text-muted-foreground">{t('tools.header.description')}</p>
        </div>
        <div className="flex items-center gap-3">
          <SearchInput
            placeholder={t('tools.list.searchPlaceholder')}
            value={searchQuery}
            onChange={setSearchQuery}
            className="w-64 bg-background"
          />
          <Button className="bg-primary text-primary-foreground hover:bg-primary/90" onClick={() => setIsConnectMcpOpen(true)}>
            <Plus className="h-4 w-4 mr-2" />
            {t('tools.mcp.addServer')}
          </Button>
          <Dialog open={isMcpDialogOpen} onOpenChange={setIsMcpDialogOpen}>
            <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle>
                  {editingServer ? t('tools.mcp.dialog.editTitle') : t('tools.mcp.dialog.addTitle')}
                </DialogTitle>
                <DialogDescription>
                  {t('tools.mcp.dialog.description')}
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="name">{t('tools.mcp.form.nameLabel')}</Label>
                  <Input
                    id="name"
                    value={mcpFormData.name}
                    onChange={(e) => setMcpFormData(prev => ({ ...prev, name: e.target.value }))}
                    placeholder={t('tools.mcp.form.namePlaceholder')}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="transport">{t('tools.mcp.form.transportLabel')}</Label>
                  <Select
                    value={mcpFormData.transport}
                    onValueChange={(value: string) => setMcpFormData(prev => ({ ...prev, transport: value }))}
                    options={transports}
                    placeholder={t('tools.mcp.form.transportPlaceholder')}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="description">{t('tools.mcp.form.descriptionLabel')}</Label>
                  <Textarea
                    id="description"
                    value={mcpFormData.description}
                    onChange={(e) => setMcpFormData(prev => ({ ...prev, description: e.target.value }))}
                    placeholder={t('tools.mcp.form.descriptionPlaceholder')}
                    rows={3}
                  />
                </div>
                {(() => {
                  const selectedTransport = transports.find(t => t.value === mcpFormData.transport);
                  return selectedTransport?.fields?.map((field) => (
                    <div key={field.name} className="space-y-2">
                      <Label htmlFor={field.name}>{field.label} {field.required && "*"}</Label>
                      {field.type === 'textarea' ? (
                        <Textarea
                          id={field.name}
                          value={mcpFormData.config[field.name] || ''}
                          onChange={(e) => setMcpFormData(prev => ({
                            ...prev,
                            config: { ...prev.config, [field.name]: e.target.value }
                          }))}
                          placeholder={field.placeholder}
                          rows={3}
                        />
                      ) : field.type === 'select' ? (
                        <Select
                          value={mcpFormData.config[field.name] || ''}
                          onValueChange={(value: string) => setMcpFormData(prev => ({
                            ...prev,
                            config: { ...prev.config, [field.name]: value }
                          }))}
                          options={field.options || []}
                          placeholder={field.placeholder}
                        />
                      ) : (
                        <Input
                          id={field.name}
                          type={field.type === 'number' ? 'number' : 'text'}
                          value={mcpFormData.config[field.name] || ''}
                          onChange={(e) => setMcpFormData(prev => ({
                            ...prev,
                            config: { ...prev.config, [field.name]: field.type === 'number' ? Number(e.target.value) : e.target.value }
                          }))}
                          placeholder={field.placeholder}
                        />
                      )}
                    </div>
                  ));
                })()}
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setIsMcpDialogOpen(false)}>
                  {t('tools.mcp.buttons.cancel')}
                </Button>
                <Button onClick={handleSaveMcpServer} disabled={isLoading}>
                  {isLoading && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                  {t('tools.mcp.buttons.save')}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="w-full justify-start bg-transparent p-0 h-auto border-b border-border/80 rounded-none flex overflow-x-auto">
          <div className="flex space-x-4">
            <TabsTrigger
              value="all"
              className="data-[state=active]:text-primary font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary"
            >
              {t('tools.tabs.all')}
            </TabsTrigger>
            {categories.map(category => (
              <TabsTrigger
                key={category}
                value={category}
                className="data-[state=active]:text-primary font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary"
              >
                {getCategoryLabel(category)}
              </TabsTrigger>
            ))}
            <TabsTrigger
              value="mcp"
              className="data-[state=active]:text-primary font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary"
            >
              {t('tools.tabs.mcp')}
            </TabsTrigger>
          </div>
        </TabsList>

        <div className="mt-6">
          <TabsContent value={activeTab} className="m-0">
            <div className="w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {isAdmin && (activeTab === 'all' || activeTab === 'basic') && filteredSearchProviderTools.map((tool) => (
                <ConfigurableToolCard key={`config-${tool.tool_name}`} tool={tool} />
              ))}

              {/* Show tools matching the tab */}
              {activeTab !== 'mcp' && filteredTools.map(tool => (
                <ToolCard key={`${tool.category}-${tool.name}`} tool={tool} />
              ))}

              {/* Show MCP servers only in 'all' or 'mcp' tab */}
              {(activeTab === 'all' || activeTab === 'mcp') && filteredMcpServers.map(server => (
                <MCPServerCard key={`mcp-${server.id}`} server={server} />
              ))}

              {/* Empty State */}
              {(activeTab !== 'mcp' && filteredTools.length === 0 &&
                filteredSearchProviderTools.length === 0 &&
                ((activeTab !== 'all' && activeTab !== 'mcp') || (activeTab === 'all' && filteredMcpServers.length === 0)) &&
                (activeTab !== 'mcp' || filteredMcpServers.length === 0)) && (
                  <div className="col-span-full flex justify-center">
                    <EmptyState />
                  </div>
                )}

              {/* Special case: Tab is MCP and no servers */}
              {activeTab === 'mcp' && filteredMcpServers.length === 0 && (
                <div className="col-span-full flex justify-center">
                  <EmptyState />
                </div>
              )}
            </div>
          </TabsContent>
        </div>
      </Tabs>

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
        globalMcpServers={mcpServers}
        selectedMcpServers={[]} // No pre-selection logic on tools page
        onSuccess={loadMCPServers}
      />
      <OfficialMcpSettingsDialog
        open={isOfficialAppDialogOpen}
        onOpenChange={setIsOfficialAppDialogOpen}
        app={editingOfficialApp}
        isGloballyConnected={true} // In tools page, official apps are always already connected
        onSuccess={loadMCPServers}
        onConfigure={(app) => {
          if (app.is_custom && app.server) {
            setIsOfficialAppDialogOpen(false);
            setEditingServer(app.server);
            setMcpFormData({
              name: app.server.name,
              transport: app.server.transport,
              description: app.server.description || "",
              config: app.server.config
            });
            setIsMcpDialogOpen(true);
          }
        }}
      />
    </div>
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
