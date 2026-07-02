"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { SearchInput } from "@/components/ui/search-input"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
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
} from "lucide-react"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { ConnectMcpDialog, AppIntegration } from "@/components/mcp/connect-mcp-dialog"
import { OfficialMcpSettingsDialog } from "@/components/mcp/official-mcp-settings-dialog"
import { CustomApiForm, MCPServerFormData } from "@/components/mcp/custom-api-form"
import { CustomMcpForm } from "@/components/mcp/custom-mcp-form"
import { useI18n } from "@/contexts/i18n-context"
import { useAuth } from "@/contexts/auth-context"
import { useMcpApps } from "@/contexts/mcp-apps-context"
import { toast } from "@/components/ui/sonner"
import { isValidMcpName, buildCustomApiPayload } from "@/lib/mcp-utils"

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

interface ConfigurableToolField {
  label: string
  required: boolean
  secret: boolean
  source: string
  is_configured: boolean
  masked: string
}

interface ConfigurableTool {
  tool_name: string
  display_name?: string
  configured: boolean
  fields: Record<string, ConfigurableToolField>
}

export default function ToolsPage() {
  const [tools, setTools] = useState<Tool[]>([])
  const [mcpServers, setMcpServers] = useState<MCPServer[]>([])
  const [transports, setTransports] = useState<TransportConfig[]>([])
  const [configurableTools, setConfigurableTools] = useState<ConfigurableTool[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isConnectMcpOpen, setIsConnectMcpOpen] = useState(false)
  const [isOfficialAppDialogOpen, setIsOfficialAppDialogOpen] = useState(false)
  const [editingOfficialApp, setEditingOfficialApp] = useState<AppIntegration | null>(null)
  const [isMcpDialogOpen, setIsMcpDialogOpen] = useState(false)
  const [customApiEnv, setCustomApiEnv] = useState<{ key: string, value: string }[]>([{ key: "", value: "" }])
  const [pendingToolToggles, setPendingToolToggles] = useState<Record<string, boolean>>({})
  const [editingServer, setEditingServer] = useState<MCPServer | null>(null)
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
      return
    }

    void loadConfigurableTools()
  }, [user])

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
      const response = await apiRequest(`${getApiUrl()}/api/tool-credentials?scope=user`)
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

  const openCredentialDialog = (toolName: string) => {
    window.location.href = `/tool-credentials?tool=${encodeURIComponent(toolName)}`
  }

  const handleEditMcpServer = (server: MCPServer) => {
    // Check if this is an official integration (from library)
    const isOfficial = server.transport === 'oauth'

    if (isOfficial) {
      const isGoogle = server.name.toLowerCase().includes('google') || server.name.toLowerCase() === 'gmail'
      const provider = server.provider || (isGoogle ? 'google' : 'linkedin')

      // Use app_id from backend if available, fallback to basic logic
      const appId = server.app_id || server.name.toLowerCase().replace(/\s+/g, '-')

      // We need to fetch the icon or use a generic one
      const icon = getAppIcon(server.name) || "";

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
    } else if (server.transport === "custom_api") {
      setEditingServer(server)
      setMcpFormData({
        name: server.name,
        transport: server.transport,
        description: server.description || "",
        url: server.config?.url || "",
        method: server.config?.method || "GET",
        headers: server.config?.headers || {},
        body: server.config?.body || "",
        config: server.config || {}
      })

      const configObj = server.config || {};
      const envObj = configObj.env || {};
      const envList = Object.entries(envObj).map(([k, v]) => ({ key: k, value: v as string }));
      if (envList.length === 0) {
        envList.push({ key: "", value: "" });
      }
      setCustomApiEnv(envList);

      setIsMcpDialogOpen(true)
    } else {
      setEditingServer(server)
      setMcpFormData({
        name: server.name,
        transport: server.transport,
        description: server.description || "",
        config: server.config || {}
      })
      setIsMcpDialogOpen(true)
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
    if (payload.transport === "custom_api") {
      if (!mcpFormData.url?.trim()) {
        toast.error(t('tools.mcp.alerts.urlRequired'));
        return;
      }
      const buildResult = buildCustomApiPayload(payload, customApiEnv);
      if (!buildResult.isValid) {
        toast.error(t(buildResult.errorKey || 'tools.mcp.alerts.atLeastOneSecret') || "At least one valid secret is required");
        return;
      }
      payload = buildResult.payload;
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

  const getConfigurableToolLabel = (tool: ConfigurableTool) => {
    const labelKey = `tools.credentials.toolNames.${tool.tool_name}`
    const localizedLabel = t(labelKey)
    return localizedLabel !== labelKey ? localizedLabel : tool.display_name || tool.tool_name
  }

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
      getConfigurableToolLabel(tool).toLowerCase().includes(searchLower) ||
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
    const toolLabel = getConfigurableToolLabel(tool)

    return (
      <Card className="hover:shadow-md transition-all duration-300 border-border/50 hover:border-primary hover:-translate-y-1">
        <CardContent className="p-6">
          <div className="flex items-start justify-between mb-4">
            <div className="flex gap-4">
              <div className="mt-1 bg-muted/50 p-3 rounded-lg h-fit">
                <Globe className="h-6 w-6 text-slate-500" />
              </div>
              <div>
                <h3 className="font-semibold text-base mb-1">{toolLabel}</h3>
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
    const canConfigureCredentials = Boolean(user) && Boolean(configurableTool)
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
                {t('tools.database.connectionBadge')}
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
                  window.location.href = '/tool-credentials?tool=sql_query'
                }}
              >
                {configButtonLabel}
              </Button>
            )}
            {isAdmin && (
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
            )}
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
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-8">
        <div className="space-y-1 w-full sm:w-auto">
          <h1 className="text-2xl sm:text-3xl font-bold mb-1">{t('tools.header.title')}</h1>
          <p className="text-sm sm:text-base text-muted-foreground">{t('tools.header.description')}</p>
        </div>
        <div className="flex items-center gap-3 w-full sm:w-auto">
          <SearchInput
            placeholder={t('tools.list.searchPlaceholder')}
            value={searchQuery}
            onChange={setSearchQuery}
            className="w-full bg-background"
            containerClassName="flex-1 sm:w-64"
          />
          <Button className="bg-primary text-primary-foreground hover:bg-primary/90 shrink-0" onClick={() => setIsConnectMcpOpen(true)}>
            <Plus className="h-4 w-4 mr-1 sm:mr-2" />
            <span className="hidden sm:inline">{t('tools.mcp.addConnector')}</span>
            <span className="sm:hidden">{t('common.add') || t('tools.mcp.addConnector')}</span>
          </Button>
          <Dialog open={isMcpDialogOpen} onOpenChange={setIsMcpDialogOpen}>
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
                  <>
                    <CustomApiForm
                      key={editingServer?.id || 'new'}
                      mcpFormData={mcpFormData}
                      setMcpFormData={setMcpFormData}
                      customApiEnv={customApiEnv}
                      setCustomApiEnv={setCustomApiEnv}
                      originalEnvObj={(() => {
                        let originalEnvObj: Record<string, any> = {};
                        if (editingServer?.config?.env) {
                          if (typeof editingServer.config.env === 'string') {
                            try {
                              originalEnvObj = JSON.parse(editingServer.config.env);
                            } catch (e) {
                              console.error("Failed to parse env:", e);
                              originalEnvObj = {};
                            }
                          } else {
                            originalEnvObj = editingServer.config.env;
                          }
                        }
                        return originalEnvObj;
                      })()}
                    />
                  </>
                ) : (
                  <>
                    <CustomMcpForm
                      mcpFormData={mcpFormData}
                      setMcpFormData={setMcpFormData}
                      transports={transports}
                    />
                  </>
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
              {t('tools.tabs.connectors')}
            </TabsTrigger>
          </div>
        </TabsList>

        <div className="mt-6">
          <TabsContent value={activeTab} className="m-0">
            <div className="w-full grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
              {user && (activeTab === 'all' || activeTab === 'basic') && filteredSearchProviderTools.map((tool) => (
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

            if (app.server.transport === "custom_api") {
              const configObj = app.server.config || {};
              const envObj = configObj.env || {};
              const envList = typeof envObj === 'object' && !Array.isArray(envObj)
                ? Object.entries(envObj).map(([k, v]) => ({ key: k, value: v as string }))
                : [];
              if (envList.length === 0) {
                envList.push({ key: "", value: "" });
              }
              setCustomApiEnv(envList);

              setMcpFormData({
                name: app.server.name,
                transport: "custom_api",
                description: app.server.description || "",
                url: configObj.url || "",
                method: configObj.method || "GET",
                headers: configObj.headers || {},
                config: configObj
              });
            } else {
              setMcpFormData({
                name: app.server.name,
                transport: app.server.transport,
                description: app.server.description || "",
                config: app.server.config || {}
              });
            }
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
