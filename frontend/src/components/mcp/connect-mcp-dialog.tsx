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
import { toast } from "sonner"
import { Input } from "@/components/ui/input"
import { Select } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { useEffect } from "react"
import { MCPServer } from "@/app/tools/page"

import { isValidMcpName, buildCustomApiPayload } from "@/lib/mcp-utils"

export interface AppIntegration {
  id: string
  name: string
  description: string
  icon: string
  is_connected?: boolean
  users?: string
  provider?: string
  category?: string
  is_local?: boolean
  server_id?: number
  transport?: string
  connected_account?: string
  is_custom?: boolean
  server?: MCPServer
}

import { OfficialMcpSettingsDialog } from "./official-mcp-settings-dialog"
import { CustomApiForm, MCPServerFormData } from "./custom-api-form"
import { CustomMcpForm } from "./custom-mcp-form"

interface ConnectMcpDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onConnectCustom?: () => void
  globalMcpServers?: any[]
  selectedMcpServers?: string[]
  onConnectSelected?: (selectedApps: string[]) => void
  customContent?: React.ReactNode
  onSuccess?: () => void
}

export function ConnectMcpDialog({
  open,
  onOpenChange,
  onConnectCustom,
  globalMcpServers = [],
  selectedMcpServers = [],
  onConnectSelected,
  customContent,
  onSuccess
}: ConnectMcpDialogProps) {
  const { t } = useI18n()
  const { token } = useAuth()
  const { apps: officialApps } = useMcpApps()
  const [searchQuery, setSearchQuery] = useState("")
  const [debouncedSearch, setDebouncedSearch] = useState("")
  const [loadingApp, setLoadingApp] = useState<string | null>(null)
  const [isLoadingApps, setIsLoadingApps] = useState(false)
  const [activeCategory, setActiveCategory] = useState("All")
  const [activeLocation, setActiveLocation] = useState("remote")
  const [activeStatus, setActiveStatus] = useState("all")
  const [apps, setApps] = useState<AppIntegration[]>([])
  const [selectedApp, setSelectedApp] = useState<AppIntegration | null>(null)
  const [localSelectedServers, setLocalSelectedServers] = useState<string[]>([])
  const [activeTab, setActiveTab] = useState("library")
  const [editingCustomServerId, setEditingCustomServerId] = useState<number | null>(null)

  // Custom MCP Server state
  const [isSavingCustom, setIsSavingCustom] = useState(false)
  const [customApiEnv, setCustomApiEnv] = useState<{ key: string, value: string }[]>([{ key: "", value: "" }])
  const [transports, setTransports] = useState<any[]>([])
  const [mcpFormData, setMcpFormData] = useState<MCPServerFormData>({
    name: "",
    transport: "stdio",
    description: "",
    config: {} as Record<string, any>
  })

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
        const data = await response.json()
        setApps(data || [])
      }
    } catch (error) {
      console.error("Failed to load apps:", error)
    } finally {
      setIsLoadingApps(false)
    }
  }

  useEffect(() => {
    const loadTransports = async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/mcp/transports`)
        if (response.ok) {
          const data = await response.json()
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
        }
      } catch (error) {
        console.error("Failed to load transports:", error)
      }
    }

    if (open) {
      loadTransports()
      setMcpFormData({
        name: "",
        transport: "stdio",
        description: "",
        config: {}
      })
      setLocalSelectedServers(selectedMcpServers || [])
      setActiveTab("library")
      setEditingCustomServerId(null)
    }
  }, [open, t, selectedMcpServers])

  useEffect(() => {
    if (open) {
      loadApps()
    }
  }, [open, debouncedSearch, activeCategory, activeLocation, activeStatus])

  const handleSaveCustomMcp = async () => {
    if (!mcpFormData.name.trim()) {
      toast.error(t('tools.mcp.alerts.nameRequired'))
      return
    }

    if (!isValidMcpName(mcpFormData.name)) {
      toast.error(t('tools.mcp.alerts.nameInvalidFormat') || "Name can only contain letters, numbers, hyphens and underscores");
      return;
    }

    let payload = { ...mcpFormData };
    let url = "";
    let method = editingCustomServerId ? 'PUT' : 'POST';

    if (payload.transport === "custom_api") {
      const buildResult = buildCustomApiPayload(payload, customApiEnv);
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
        onOpenChange(false);
      }

      setEditingCustomServerId(null)
      setMcpFormData({ name: "", transport: "stdio", description: "", config: {} })
    } else {
      const error = await response.json()
      toast.error(error.detail || t('tools.mcp.alerts.saveFailed'))
    }
    setIsSavingCustom(false)
  }

  const isSelectMode = !!onConnectSelected;

  const handleConnectApp = (app: AppIntegration, autoSelect: boolean = false) => {
    const provider = app.provider;
    if (!provider) {
      toast.error("Error: App provider is not defined");
      return;
    }

    setLoadingApp(app.id)
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
      setLoadingApp(null);
      return;
    }

    // Listen for the postMessage from the popup
    const handleMessage = (event: MessageEvent) => {
      if (event.data?.type === 'oauth-success') {
        setLoadingApp(null)
        window.removeEventListener('message', handleMessage)

        loadApps();
        if (onSuccess) onSuccess();

        if (autoSelect && onConnectSelected) {
          // If it was just connected, it is not selected yet, so add it to local selection
          setLocalSelectedServers(prev => prev.includes(app.name) ? prev : [...prev, app.name]);
        }

        setSelectedApp(null);
      }
    };

    window.addEventListener('message', handleMessage);

    // Fallback: check if popup was closed without success message
    const checkPopup = setInterval(() => {
      if (popup?.closed) {
        clearInterval(checkPopup);
        window.removeEventListener('message', handleMessage);
        setLoadingApp(null);
      }
    }, 500);
  }

  const handleDisconnectApp = async (app: AppIntegration) => {
    // Determine if it's a custom API or an MCP server
    const isCustomApi = app.transport === 'custom_api' || app.is_custom;
    const server = globalMcpServers.find(s =>
      (s.name.toLowerCase() === app.id.toLowerCase() || s.name.toLowerCase() === app.name.toLowerCase()) &&
      (isCustomApi ? s.transport === 'custom_api' : s.transport !== 'custom_api')
    );

    // For custom APIs, we might not have them in globalMcpServers since that fetches from /api/mcp/servers
    // We should use app.server_id if available
    const serverId = server ? server.id : app.server_id;

    if (serverId) {
      try {
        const endpoint = isCustomApi
          ? `${getApiUrl()}/api/custom-apis/${serverId}`
          : `${getApiUrl()}/api/mcp/servers/${serverId}`;

        const response = await apiRequest(endpoint, {
          method: 'DELETE'
        });
        if (response.ok) {
          toast.success(t('tools.mcp.alerts.deleteSuccess') || "Disconnected successfully");
          if (onSuccess) onSuccess();
          setSelectedApp(null);
          // Reload apps to refresh the is_connected state visually
          loadApps();
        } else {
          const err = await response.json();
          toast.error(err.detail || "Failed to disconnect");
        }
      } catch (e) {
        console.error(e);
        toast.error("Failed to disconnect");
      }
    }
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

  const selectedRemoteCount = localSelectedServers.filter(name =>
    officialApps.some(app => app.name.toLowerCase() === name.toLowerCase() || app.id.toLowerCase() === name.toLowerCase())
  ).length;
  const selectedLocalCount = localSelectedServers.length - selectedRemoteCount;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-5xl md:max-w-6xl w-[95vw] h-[85vh] flex flex-col p-0 overflow-hidden gap-0 bg-slate-50">
        <DialogHeader className="px-6 py-4 border-b bg-white shrink-0">
          <DialogTitle className="text-xl flex items-center gap-2 font-bold">
            <Plug className="h-5 w-5 text-blue-600" /> {t('tools.mcp.dialog.connector')}
          </DialogTitle>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col overflow-hidden bg-white">
          <div className="px-6 border-b shrink-0 bg-white">
            <TabsList className="bg-transparent h-14 p-0 border-b-0 space-x-6">
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
                  setEditingCustomServerId(null)
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
                  if (onConnectCustom) {
                    e.preventDefault()
                    onConnectCustom()
                  } else {
                    setEditingCustomServerId(null)
                    setMcpFormData({
                      name: "",
                      transport: "stdio",
                      description: "",
                      config: {}
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
                  </div>
                </div>
              </div>
            </div>

            {/* Main Content Area */}
            <div className="flex-1 flex flex-col overflow-hidden bg-white">
              <div className="p-6 pb-2 shrink-0">
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
                      const isGloballyConnected = app.is_connected || globalMcpServers.some(s => s.name.toLowerCase() === app.id.toLowerCase() || s.name.toLowerCase() === app.name.toLowerCase())
                      const isSelected = localSelectedServers.includes(app.id) || localSelectedServers.includes(app.name)
                      const isLoading = loadingApp === app.id
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
                  originalEnvObj={
                    editingCustomServerId
                      ? globalMcpServers.find(s => s.id === editingCustomServerId && s.transport === "custom_api")?.config?.env || {}
                      : {}
                  }
                />
              </div>

              <div className="flex justify-end gap-3 mt-8 pt-4 border-t">
                <Button variant="outline" onClick={() => onOpenChange(false)}>
                  {t('tools.mcp.buttons.cancel')}
                </Button>
                <Button
                  onClick={handleSaveCustomMcp}
                  disabled={
                    isSavingCustom ||
                    !mcpFormData.name.trim() ||
                    (customApiEnv.length > 0 && customApiEnv.some(env => !env.key.trim() || !env.value.trim()))
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
                    mcpFormData={mcpFormData}
                    setMcpFormData={setMcpFormData}
                    transports={transports}
                  />
                </div>
                <div className="flex justify-end gap-3 mt-8">
                  <Button variant="outline" onClick={() => onOpenChange(false)}>
                    {t('tools.mcp.buttons.cancel')}
                  </Button>
                  <Button onClick={handleSaveCustomMcp} disabled={isSavingCustom}>
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
              onClick={() => {
                if (onConnectSelected) {
                  onConnectSelected(localSelectedServers);
                }
                onOpenChange(false);
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
        onOpenChange={(open) => !open && setSelectedApp(null)}
        app={(() => {
          if (!selectedApp) return null;
          // Find the actual server from globalMcpServers to get the real numeric ID
          const isCustomApi = selectedApp.transport === 'custom_api' || selectedApp.is_custom;
          const server = globalMcpServers.find(s =>
            (s.name.toLowerCase() === selectedApp.id.toLowerCase() || s.name.toLowerCase() === selectedApp.name.toLowerCase()) &&
            (isCustomApi ? s.transport === 'custom_api' : s.transport !== 'custom_api')
          );

          if (server) {
            // Merge the server ID into the app object so the child dialog can use it for deletion
            return {
              ...selectedApp,
              server_id: server.id,
              server: server,
              is_custom: server.transport !== 'oauth'
            };
          }
          return selectedApp;
        })()}
        isGloballyConnected={selectedApp ? (selectedApp.is_connected || globalMcpServers.some(s => s.name.toLowerCase() === selectedApp.id.toLowerCase() || s.name.toLowerCase() === selectedApp.name.toLowerCase())) : false}
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
        onConfigure={(appToConfigure) => {
          if (appToConfigure.is_custom && appToConfigure.server) {
            setSelectedApp(null);
            setEditingCustomServerId(appToConfigure.server.id);
            setMcpFormData({
              name: appToConfigure.server.name,
              transport: appToConfigure.server.transport,
              description: appToConfigure.server.description || "",
              config: appToConfigure.server.config || {}
            });
            if (appToConfigure.server.transport === "custom_api") {
              const configObj = appToConfigure.server.config || {};
              const envObj = configObj.env || {};
              const envList = Object.entries(envObj).map(([k, v]) => ({ key: k, value: v as string }));
              if (envList.length === 0) {
                envList.push({ key: "", value: "" });
              }
              setCustomApiEnv(envList);

              // Map url, method, headers to top level for form component since custom-api-form expects them there
              setMcpFormData({
                name: appToConfigure.server.name,
                transport: "custom_api",
                description: appToConfigure.server.description || "",
                url: configObj.url || "",
                method: configObj.method || "GET",
                headers: configObj.headers || {},
                config: configObj
              });
            } else {
              setMcpFormData({
                name: appToConfigure.server.name,
                transport: appToConfigure.server.transport,
                description: appToConfigure.server.description || "",
                config: appToConfigure.server.config || {}
              });
            }
            setActiveTab(appToConfigure.server.transport === "custom_api" ? "custom_api" : "custom");
          }
        }}
      />
    </Dialog>
  )
}
