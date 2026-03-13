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
  Search,
  Wrench,
  Flame,
  Globe,
  Hash,
  Code,
  FileText,
  Book,
  Loader2,
  Mic
} from "lucide-react"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"

interface Tool {
  name: string
  description: string
  type: 'builtin' | 'mcp' | 'image' | 'vision' | 'audio'
  category: string
  display_category?: string  // Add display_category field
  enabled: boolean
  status?: string
  status_reason?: string
  config?: Record<string, any>
  source?: string
  usage_count?: number
}

interface MCPServer {
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

export default function ToolsPage() {
  const [tools, setTools] = useState<Tool[]>([])
  const [mcpServers, setMcpServers] = useState<MCPServer[]>([])
  const [transports, setTransports] = useState<TransportConfig[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isMcpDialogOpen, setIsMcpDialogOpen] = useState(false)
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

  useEffect(() => {
    loadTools()
    loadMCPServers()
    loadTransports()
  }, [])

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

  const handleAddMcpServer = () => {
    setEditingServer(null)
    setMcpFormData({
      name: "",
      transport: "stdio",
      description: "",
      config: {}
    })
    setIsMcpDialogOpen(true)
  }

  const handleEditMcpServer = (server: MCPServer) => {
    setEditingServer(server)
    setMcpFormData({
      name: server.name,
      transport: server.transport,
      description: server.description || "",
      config: server.config
    })
    setIsMcpDialogOpen(true)
  }

  const handleSaveMcpServer = async () => {
    if (!mcpFormData.name.trim()) {
      alert(t('tools.mcp.alerts.nameRequired'))
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
        alert(error.detail || t('tools.mcp.alerts.saveFailed'))
      }
    } catch (error) {
      console.error("Failed to save MCP server:", error)
      alert(t('tools.mcp.alerts.saveFailed'))
    } finally {
      setIsLoading(false)
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

    if (lowerName.includes('firecrawl')) return <Flame className="h-6 w-6 text-orange-500" />
    if (lowerName.includes('google')) return <Globe className="h-6 w-6 text-blue-500" />
    if (lowerName.includes('slack')) return <Hash className="h-6 w-6 text-purple-500" />

    if (lowerCategory === 'browser') return <Globe className="h-6 w-6 text-blue-500" />
    if (lowerCategory === 'file') return <FileText className="h-6 w-6 text-amber-500" />
    if (lowerCategory === 'knowledge') return <Book className="h-6 w-6 text-indigo-500" />
    if (lowerCategory === 'audio') return <Mic className="h-6 w-6 text-green-500" />

    if (type === 'mcp') return <Server className="h-6 w-6 text-green-600" />
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
  const categories = Array.from(new Set(tools.map(t => t.category).filter(Boolean))).sort()

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

  const ToolCard = ({ tool }: { tool: Tool }) => {
    const { label, variant } = getBadgeInfo(tool)
    const icon = getToolIcon(tool.name, tool.type, tool.category)

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

          <div className="flex items-center justify-end text-xs text-muted-foreground">
            <span>{t('tools.list.usedByAgents', { count: tool.usage_count || 0 })}</span>
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
          <Dialog open={isMcpDialogOpen} onOpenChange={setIsMcpDialogOpen}>
            <DialogTrigger asChild>
              <Button className="bg-primary text-primary-foreground hover:bg-primary/90" onClick={handleAddMcpServer}>
                <Plus className="h-4 w-4 mr-2" />
                {t('tools.mcp.addServer')}
              </Button>
            </DialogTrigger>
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
        <TabsList className="w-full justify-start bg-transparent p-0 h-auto border-b border-border/80 rounded-none flex">
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
              {/* Show tools matching the tab */}
              {activeTab !== 'mcp' && filteredTools.map(tool => (
                <ToolCard key={`${tool.category}-${tool.name}`} tool={tool} />
              ))}

              {/* Show MCP servers only in 'all' or 'mcp' tab */}
              {(activeTab === 'all' || activeTab === 'mcp') && filteredMcpServers.map(server => (
                <MCPServerCard key={`mcp-${server.id}`} server={server} />
              ))}
            </div>

            {/* Empty State */}
            {((activeTab !== 'mcp' && filteredTools.length === 0) &&
              ((activeTab !== 'all' && activeTab !== 'mcp') || (activeTab === 'all' && filteredMcpServers.length === 0)) &&
              (activeTab !== 'mcp' || filteredMcpServers.length === 0)) && (
              <EmptyState />
            )}

            {/* Special case: Tab is MCP and no servers */}
            {activeTab === 'mcp' && filteredMcpServers.length === 0 && (
              <EmptyState />
            )}
          </TabsContent>
        </div>
      </Tabs>
    </div>
  )
}

function EmptyState() {
  const { t } = useI18n()
  return (
    <div className="text-center py-16 text-muted-foreground border border-dashed rounded-lg">
      <Wrench className="h-10 w-10 mx-auto mb-4 opacity-50" />
      <div className="font-medium mb-1">{t('tools.list.empty.title')}</div>
      <div className="text-sm">{t('tools.list.empty.description')}</div>
    </div>
  )
}
