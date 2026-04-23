"use client"

import { useState, useEffect } from "react"
import { Plus, Trash2, Edit2, Search } from "lucide-react"
import { useAuth } from "@/contexts/auth-context"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { toast } from "sonner"

import { useI18n } from "@/contexts/i18n-context"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select-radix"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogDescription } from "@/components/ui/dialog"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Stepper } from "@/components/ui/stepper"

interface OAuthProvider {
  id: number
  provider_name: string
  name: string
  client_id: string
  client_secret: string
  auth_url: string
  token_url: string
  redirect_uri?: string | null
  userinfo_url: string | null
  user_id_path: string | null
  email_path: string | null
  default_scopes: string[] | null
}

interface PublicMCPApp {
  id: number
  app_id: string
  name: string
  description: string | null
  icon: string | null
  transport: string
  provider_name: string | null
  category: string | null
  oauth_scopes: string[] | null
  launch_config: any | null
}

export default function AdminMcpPage() {
  const { user } = useAuth()
  const { t } = useI18n()
  const [providers, setProviders] = useState<OAuthProvider[]>([])
  const [apps, setApps] = useState<PublicMCPApp[]>([])
  const [loading, setLoading] = useState(true)
  const [appSearchQuery, setAppSearchQuery] = useState("")

  // Modals state
  const [isAddModalOpen, setIsAddModalOpen] = useState(false)
  const [step, setStep] = useState(1) // 1: Provider, 2: App Details

  // Form states
  const [selectedProviderName, setSelectedProviderName] = useState<string>("none")
  const [isCreatingProvider, setIsCreatingProvider] = useState(false)
  const [isStandaloneProvider, setIsStandaloneProvider] = useState(false)

  const [editingProviderId, setEditingProviderId] = useState<number | null>(null)
  const [editingAppId, setEditingAppId] = useState<number | null>(null)

  // New Provider Form
  const [newProvider, setNewProvider] = useState<Partial<OAuthProvider>>({
    provider_name: "", name: "", client_id: "", client_secret: "",
    auth_url: "", token_url: "", redirect_uri: "", userinfo_url: "", user_id_path: "id", email_path: "email", default_scopes: []
  })

  // New App Form
  const [newApp, setNewApp] = useState<Partial<PublicMCPApp>>({
    app_id: "", name: "", description: "", icon: "", transport: "oauth",
    category: "Communication", oauth_scopes: [], launch_config: "{}"
  })

  useEffect(() => {
    if (user?.is_admin) {
      fetchData()
    }
  }, [user])

  const normalizedAppSearchQuery = appSearchQuery.trim().toLowerCase()
  const filteredApps = normalizedAppSearchQuery
    ? apps.filter((a) => {
      const haystack = [
        a.app_id,
        a.name,
        a.provider_name ?? "",
        a.transport,
        a.category ?? "",
      ]
        .join(" ")
        .toLowerCase()
      return haystack.includes(normalizedAppSearchQuery)
    })
    : apps

  const fetchData = async () => {
    try {
      setLoading(true)
      const [providersRes, appsRes] = await Promise.all([
        apiRequest(`${getApiUrl()}/api/admin/mcp/providers`),
        apiRequest(`${getApiUrl()}/api/admin/mcp/apps`)
      ])

      if (providersRes.ok) setProviders(await providersRes.json())
      if (appsRes.ok) setApps(await appsRes.json())
    } catch (err) {
      toast.error("Failed to load data")
    } finally {
      setLoading(false)
    }
  }

  const handleEditProvider = (p: OAuthProvider) => {
    setEditingProviderId(p.id)
    setNewProvider({
      ...p,
      default_scopes: p.default_scopes ? p.default_scopes.join(", ") : "" as any
    })
    setIsAddModalOpen(true)
    setStep(1)
    setIsCreatingProvider(true)
    setIsStandaloneProvider(true)
  }

  const handleEditApp = (a: PublicMCPApp) => {
    setEditingAppId(a.id)
    setNewApp({
      ...a,
      oauth_scopes: a.oauth_scopes ? a.oauth_scopes.join(", ") : "" as any,
      launch_config: typeof a.launch_config === 'string' ? a.launch_config : JSON.stringify(a.launch_config, null, 2)
    })
    setSelectedProviderName(a.provider_name || "none")
    setIsAddModalOpen(true)
    setStep(2)
    setIsStandaloneProvider(false)
  }

  const handleDeleteProvider = async (id: number) => {
    if (!confirm(t("adminMcp.providers.deleteConfirm"))) return
    try {
      const res = await apiRequest(`${getApiUrl()}/api/admin/mcp/providers/${id}`, { method: 'DELETE' })
      if (res.ok) {
        toast.success(t("adminMcp.providers.deleteSuccess"))
        fetchData()
      } else {
        toast.error(t("adminMcp.providers.deleteFailed"))
      }
    } catch (err) {
      toast.error(t("adminMcp.providers.deleteFailed"))
    }
  }

  const handleDeleteApp = async (id: number) => {
    if (!confirm(t("adminMcp.apps.deleteConfirm"))) return
    try {
      const res = await apiRequest(`${getApiUrl()}/api/admin/mcp/apps/${id}`, { method: 'DELETE' })
      if (res.ok) {
        toast.success(t("adminMcp.apps.deleteSuccess"))
        fetchData()
      } else {
        toast.error(t("adminMcp.apps.deleteFailed"))
      }
    } catch (err) {
      toast.error(t("adminMcp.apps.deleteFailed"))
    }
  }

  const handleNextStep1 = async () => {
    if (isCreatingProvider) {
      const isEdit = editingProviderId !== null
      const url = isEdit
        ? `${getApiUrl()}/api/admin/mcp/providers/${editingProviderId}`
        : `${getApiUrl()}/api/admin/mcp/providers`
      const method = isEdit ? 'PUT' : 'POST'

      try {
        const providerPayload: any = {
          ...newProvider,
          default_scopes: typeof newProvider.default_scopes === 'string'
            ? (newProvider.default_scopes as string).split(',').map(s => s.trim()).filter(Boolean)
            : newProvider.default_scopes
        }

        if (isEdit) {
          if (providerPayload.client_id === "********" || providerPayload.client_id == null) {
            delete providerPayload.client_id
          }
          if (providerPayload.client_secret === "********" || providerPayload.client_secret == null) {
            delete providerPayload.client_secret
          }
        }

        const res = await apiRequest(url, {
          method,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(providerPayload)
        })
        if (res.ok) {
          const saved = await res.json()
          toast.success(isEdit ? (t("adminMcp.providers.saveSuccess")) : (t("adminMcp.providers.saveSuccess")))

          if (isEdit || isStandaloneProvider) {
            setIsAddModalOpen(false)
            fetchData()
          } else {
            setProviders([...providers, saved])
            setSelectedProviderName(saved.provider_name)
            setIsCreatingProvider(false)
            setStep(2)
          }
        } else {
          toast.error(isEdit ? (t("adminMcp.providers.saveFailed")) : (t("adminMcp.providers.saveFailed")))
        }
      } catch (err) {
        toast.error(t("adminMcp.providers.saveFailed"))
      }
    } else {
      setStep(2)
    }
  }

  const handleNextStep2 = async () => {
    try {
      let parsedConfig = {}
      try {
        if (typeof newApp.launch_config === 'string') {
          const raw = newApp.launch_config.trim()
          parsedConfig = raw === "" ? {} : JSON.parse(raw)
        } else {
          parsedConfig = newApp.launch_config
        }
      } catch (e) {
        toast.error(t("adminMcp.apps.form.invalidJson"))
        return
      }

      const appData = {
        ...newApp,
        provider_name: selectedProviderName === "none" ? null : selectedProviderName,
        launch_config: parsedConfig,
        oauth_scopes: typeof newApp.oauth_scopes === 'string'
          ? (newApp.oauth_scopes as string).split(',').map(s => s.trim()).filter(Boolean)
          : newApp.oauth_scopes
      }

      const isEdit = editingAppId !== null
      const url = isEdit
        ? `${getApiUrl()}/api/admin/mcp/apps/${editingAppId}`
        : `${getApiUrl()}/api/admin/mcp/apps`
      const method = isEdit ? 'PUT' : 'POST'

      const res = await apiRequest(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(appData)
      })

      if (res.ok) {
        toast.success(isEdit ? (t("adminMcp.apps.saveSuccess")) : (t("adminMcp.apps.saveSuccess")))
        fetchData()
        setIsAddModalOpen(false)
      } else {
        toast.error(isEdit ? (t("adminMcp.apps.saveFailed")) : (t("adminMcp.apps.saveFailed")))
      }
    } catch (err) {
      toast.error(t("adminMcp.apps.saveFailed"))
    }
  }

  // Effect to reset state when modal closes
  useEffect(() => {
    if (!isAddModalOpen) {
      const timeoutId = window.setTimeout(() => {
        setStep(1)
        setEditingProviderId(null)
        setEditingAppId(null)
        setIsCreatingProvider(false)
        setNewProvider({ provider_name: "", name: "", client_id: "", client_secret: "", auth_url: "", token_url: "", redirect_uri: "", userinfo_url: "", user_id_path: "id", email_path: "email", default_scopes: [] })
        setNewApp({ app_id: "", name: "", description: "", icon: "", transport: "oauth", category: "Communication", oauth_scopes: [], launch_config: "{}" })
      }, 300)
      return () => window.clearTimeout(timeoutId)
    }
  }, [isAddModalOpen])

  if (!user?.is_admin) {
    return <div className="p-8 text-center text-muted-foreground">{t("adminMcp.adminRequired")}</div>
  }

  return (
    <div className="flex flex-col h-screen bg-background">
      <div className="flex justify-between items-center p-8">
        <div>
          <h1 className="text-2xl font-bold tracking-tight mb-1">{t("adminMcp.pageTitle")}</h1>
          <p className="text-sm text-muted-foreground">{t("adminMcp.pageDescription")}</p>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-12 items-start px-8">
        <div className="md:col-span-4">
          <Card className="shadow-sm">
            <CardHeader className="pb-3 border-b flex flex-row items-center justify-between space-y-0">
              <div>
                <CardTitle className="text-lg font-semibold flex items-center gap-2">
                  {t("adminMcp.providers.title")} <Badge variant="secondary" className="rounded-full px-2 py-0 bg-blue-50 text-blue-500 font-normal">{providers.length}</Badge>
                </CardTitle>
                <p className="text-xs text-muted-foreground mt-1 font-normal">{t("adminMcp.providers.description")}</p>
              </div>
              <Button size="sm" className="bg-blue-500 hover:bg-blue-600 h-8" onClick={() => { setIsAddModalOpen(true); setStep(1); setIsCreatingProvider(true); setIsStandaloneProvider(true) }}>
                <Plus className="w-4 h-4 mr-1" /> {t("adminMcp.providers.add")}
              </Button>
            </CardHeader>
            <CardContent className="p-0">
              <div className="divide-y">
                {providers.map(p => {
                  const linkedApps = apps.filter(a => a.provider_name === p.provider_name).length
                  return (
                    <div key={p.id} className="p-4 flex items-center justify-between group hover:bg-slate-50 transition-colors">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-blue-500 text-white flex items-center justify-center font-bold text-lg">
                          {p.name.charAt(0).toUpperCase()}
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-sm">{p.name}</span>
                            <Badge variant="outline" className="text-[10px] font-normal text-muted-foreground bg-slate-50">{p.provider_name}</Badge>
                          </div>
                          <p className="text-xs text-muted-foreground mt-0.5">{linkedApps !== 1 ? t("adminMcp.providers.linkedAppsPlural", { count: linkedApps }) : t("adminMcp.providers.linkedApps", { count: linkedApps })}</p>
                        </div>
                      </div>
                      <div className="flex justify-end opacity-0 group-hover:opacity-100 transition-opacity">
                        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => handleEditProvider(p)}>
                          <Edit2 className="w-4 h-4 text-muted-foreground hover:text-blue-500" />
                        </Button>
                        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => handleDeleteProvider(p.id)}>
                          <Trash2 className="w-4 h-4 text-muted-foreground hover:text-destructive" />
                        </Button>
                      </div>
                    </div>
                  )
                })}
                {providers.length === 0 && !loading && (
                  <div className="p-8 text-center text-muted-foreground text-sm">{t("adminMcp.providers.noData")}</div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="md:col-span-8">
          <Card className="shadow-sm">
            <CardHeader className="pb-3 border-b flex flex-row items-center justify-between space-y-0">
              <div>
                <CardTitle className="text-lg font-semibold flex items-center gap-2">
                  {t("adminMcp.apps.title")} <Badge variant="secondary" className="rounded-full px-2 py-0 bg-blue-50 text-blue-500 font-normal">{filteredApps.length}</Badge>
                </CardTitle>
                <p className="text-xs text-muted-foreground mt-1 font-normal">{t("adminMcp.apps.description")}</p>
              </div>
              <div className="flex items-center gap-3">
                <div className="relative w-64">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    placeholder={t("adminMcp.apps.searchPlaceholder")}
                    className="pl-9 h-9"
                    value={appSearchQuery}
                    onChange={(e) => setAppSearchQuery(e.target.value)}
                  />
                </div>
                <Button size="sm" className="bg-blue-500 hover:bg-blue-600 h-9" onClick={() => { setIsAddModalOpen(true); setStep(1); setIsCreatingProvider(false); setIsStandaloneProvider(false) }}>
                  <Plus className="w-4 h-4 mr-1" /> {t("adminMcp.addApp")}
                </Button>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <Table>
                <TableHeader>
                  <TableRow className="hover:bg-transparent">
                    <TableHead className="text-xs font-semibold text-muted-foreground">{t("adminMcp.apps.columns.appId")}</TableHead>
                    <TableHead className="text-xs font-semibold text-muted-foreground">{t("adminMcp.apps.columns.provider")}</TableHead>
                    <TableHead className="text-xs font-semibold text-muted-foreground">{t("adminMcp.apps.columns.transport")}</TableHead>
                    <TableHead className="w-[100px]"></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredApps.map(a => (
                    <TableRow key={a.id} className="group">
                      <TableCell className="font-semibold text-sm">{a.app_id}</TableCell>
                      <TableCell>
                        {a.provider_name ? <Badge variant="outline" className="font-normal text-muted-foreground bg-slate-50">{a.provider_name}</Badge> : "-"}
                      </TableCell>
                      <TableCell>
                        <Badge variant="secondary" className="bg-blue-50 text-blue-500 font-normal hover:bg-blue-50">{a.transport}</Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex justify-end opacity-0 group-hover:opacity-100 transition-opacity">
                          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => handleEditApp(a)}>
                            <Edit2 className="w-4 h-4 text-muted-foreground hover:text-blue-500" />
                          </Button>
                          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => handleDeleteApp(a.id)}>
                            <Trash2 className="w-4 h-4 text-muted-foreground hover:text-destructive" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                  {filteredApps.length === 0 && !loading && (
                    <TableRow>
                      <TableCell colSpan={4} className="text-center text-muted-foreground h-32">{t("adminMcp.apps.noData")}</TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </div>
      </div>

      <Dialog open={isAddModalOpen} onOpenChange={setIsAddModalOpen}>
        <DialogContent className="sm:max-w-[600px] max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              {isStandaloneProvider ? (
                editingProviderId ? t("adminMcp.providers.edit") : t("adminMcp.providers.new")
              ) : (
                <>
                  {step === 1 && (t("adminMcp.modal.step1"))}
                  {step === 2 && (t("adminMcp.modal.step2"))}
                </>
              )}
            </DialogTitle>
            <DialogDescription>
              {isStandaloneProvider ? (
                t("adminMcp.providers.description")
              ) : (
                <>
                  {step === 1 && (t("adminMcp.modal.step1Desc"))}
                  {step === 2 && (t("adminMcp.modal.step2Desc"))}
                </>
              )}
            </DialogDescription>
          </DialogHeader>

          {isStandaloneProvider ? (
            <>
              {step === 1 && (
                <div className="space-y-4 py-4">
                  {!isCreatingProvider && !isStandaloneProvider ? (
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <Label>{t("adminMcp.providers.form.selectLabel")}</Label>
                        <Select value={selectedProviderName} onValueChange={setSelectedProviderName}>
                          <SelectTrigger>
                            <SelectValue placeholder={t("adminMcp.providers.form.selectPlaceholder")} />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="none">{t("adminMcp.providers.form.noOAuth")}</SelectItem>
                            {providers.map(p => (
                              <SelectItem key={p.provider_name} value={p.provider_name}>{p.name} ({p.provider_name})</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="pt-4 border-t flex flex-col items-center">
                        <span className="text-sm text-muted-foreground mb-4">{t("adminMcp.providers.form.orCreateNew")}</span>
                        <Button variant="outline" onClick={() => setIsCreatingProvider(true)}>
                          <Plus className="w-4 h-4 mr-2" /> {t("adminMcp.providers.form.addNew")}
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="grid gap-4 py-2">
                      {!isStandaloneProvider && (
                        <div className="flex justify-between items-center mb-2">
                          <h3 className="font-medium">{editingProviderId ? (t("adminMcp.providers.edit")) : (t("adminMcp.providers.new"))}</h3>
                        </div>
                      )}
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label>{t("adminMcp.providers.form.providerId")}</Label>
                          <Input value={newProvider.provider_name} onChange={e => setNewProvider({ ...newProvider, provider_name: e.target.value })} />
                        </div>
                        <div className="space-y-2">
                          <Label>{t("adminMcp.providers.form.displayName")}</Label>
                          <Input value={newProvider.name} onChange={e => setNewProvider({ ...newProvider, name: e.target.value })} />
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label>{t("adminMcp.providers.form.clientId")}</Label>
                          <Input value={newProvider.client_id} onChange={e => setNewProvider({ ...newProvider, client_id: e.target.value })} />
                        </div>
                        <div className="space-y-2">
                          <Label>{t("adminMcp.providers.form.clientSecret")}</Label>
                          <Input type="password" value={newProvider.client_secret} onChange={e => setNewProvider({ ...newProvider, client_secret: e.target.value })} />
                        </div>
                      </div>
                      <div className="space-y-2">
                        <Label>{t("adminMcp.providers.form.authUrl")}</Label>
                        <Input value={newProvider.auth_url} onChange={e => setNewProvider({ ...newProvider, auth_url: e.target.value })} />
                      </div>
                      <div className="space-y-2">
                        <Label>{t("adminMcp.providers.form.tokenUrl")}</Label>
                        <Input value={newProvider.token_url} onChange={e => setNewProvider({ ...newProvider, token_url: e.target.value })} />
                      </div>
                      <div className="space-y-2">
                        <Label>{t("adminMcp.providers.form.redirectUri")}</Label>
                        <Input placeholder={t("adminMcp.providers.form.redirectUriPlaceholder")} value={newProvider.redirect_uri || ""} onChange={e => setNewProvider({ ...newProvider, redirect_uri: e.target.value })} />
                      </div>
                      <div className="space-y-2">
                        <Label>{t("adminMcp.providers.form.userinfoUrl")}</Label>
                        <Input value={newProvider.userinfo_url || ""} onChange={e => setNewProvider({ ...newProvider, userinfo_url: e.target.value })} />
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label>{t("adminMcp.providers.form.userIdPath")}</Label>
                          <Input value={newProvider.user_id_path || "id"} onChange={e => setNewProvider({ ...newProvider, user_id_path: e.target.value })} />
                        </div>
                        <div className="space-y-2">
                          <Label>{t("adminMcp.providers.form.emailPath")}</Label>
                          <Input value={newProvider.email_path || "email"} onChange={e => setNewProvider({ ...newProvider, email_path: e.target.value })} />
                        </div>
                      </div>
                      <div className="space-y-2">
                        <Label>{t("adminMcp.providers.form.defaultScopes")}</Label>
                        <Input value={(newProvider.default_scopes as any) || ""} onChange={e => setNewProvider({ ...newProvider, default_scopes: e.target.value as any })} />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          ) : (
            <Stepper
              steps={[
                {
                  label: t("adminMcp.modal.step1"),
                  content: (
                    <div className="space-y-4 py-4">
                      {!isCreatingProvider && !isStandaloneProvider ? (
                        <div className="space-y-4">
                          <div className="space-y-2">
                            <Label>{t("adminMcp.providers.form.selectLabel")}</Label>
                            <Select value={selectedProviderName} onValueChange={setSelectedProviderName}>
                              <SelectTrigger>
                                <SelectValue placeholder={t("adminMcp.providers.form.selectPlaceholder")} />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="none">{t("adminMcp.providers.form.noOAuth")}</SelectItem>
                                {providers.map(p => (
                                  <SelectItem key={p.provider_name} value={p.provider_name}>{p.name} ({p.provider_name})</SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>
                          <div className="pt-4 border-t flex flex-col items-center">
                            <span className="text-sm text-muted-foreground mb-4">{t("adminMcp.providers.form.orCreateNew")}</span>
                            <Button variant="outline" onClick={() => setIsCreatingProvider(true)}>
                              <Plus className="w-4 h-4 mr-2" /> {t("adminMcp.providers.form.addNew")}
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <div className="grid gap-4 py-2">
                          {!isStandaloneProvider && (
                            <div className="flex justify-between items-center mb-2">
                              <h3 className="font-medium">{editingProviderId ? (t("adminMcp.providers.edit")) : (t("adminMcp.providers.new"))}</h3>
                            </div>
                          )}
                          <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2">
                              <Label>{t("adminMcp.providers.form.providerId")}</Label>
                              <Input value={newProvider.provider_name} onChange={e => setNewProvider({ ...newProvider, provider_name: e.target.value })} />
                            </div>
                            <div className="space-y-2">
                              <Label>{t("adminMcp.providers.form.displayName")}</Label>
                              <Input value={newProvider.name} onChange={e => setNewProvider({ ...newProvider, name: e.target.value })} />
                            </div>
                          </div>
                          <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2">
                              <Label>{t("adminMcp.providers.form.clientId")}</Label>
                              <Input value={newProvider.client_id} onChange={e => setNewProvider({ ...newProvider, client_id: e.target.value })} />
                            </div>
                            <div className="space-y-2">
                              <Label>{t("adminMcp.providers.form.clientSecret")}</Label>
                              <Input type="password" value={newProvider.client_secret} onChange={e => setNewProvider({ ...newProvider, client_secret: e.target.value })} />
                            </div>
                          </div>
                          <div className="space-y-2">
                            <Label>{t("adminMcp.providers.form.authUrl")}</Label>
                            <Input value={newProvider.auth_url} onChange={e => setNewProvider({ ...newProvider, auth_url: e.target.value })} />
                          </div>
                          <div className="space-y-2">
                            <Label>{t("adminMcp.providers.form.tokenUrl")}</Label>
                            <Input value={newProvider.token_url} onChange={e => setNewProvider({ ...newProvider, token_url: e.target.value })} />
                          </div>
                          <div className="space-y-2">
                            <Label>{t("adminMcp.providers.form.redirectUri")}</Label>
                            <Input placeholder={t("adminMcp.providers.form.redirectUriPlaceholder")} value={newProvider.redirect_uri || ""} onChange={e => setNewProvider({ ...newProvider, redirect_uri: e.target.value })} />
                          </div>
                          <div className="space-y-2">
                            <Label>{t("adminMcp.providers.form.userinfoUrl")}</Label>
                            <Input value={newProvider.userinfo_url || ""} onChange={e => setNewProvider({ ...newProvider, userinfo_url: e.target.value })} />
                          </div>
                          <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2">
                              <Label>{t("adminMcp.providers.form.userIdPath")}</Label>
                              <Input value={newProvider.user_id_path || "id"} onChange={e => setNewProvider({ ...newProvider, user_id_path: e.target.value })} />
                            </div>
                            <div className="space-y-2">
                              <Label>{t("adminMcp.providers.form.emailPath")}</Label>
                              <Input value={newProvider.email_path || "email"} onChange={e => setNewProvider({ ...newProvider, email_path: e.target.value })} />
                            </div>
                          </div>
                          <div className="space-y-2">
                            <Label>{t("adminMcp.providers.form.defaultScopes")}</Label>
                            <Input value={(newProvider.default_scopes as any) || ""} onChange={e => setNewProvider({ ...newProvider, default_scopes: e.target.value as any })} />
                          </div>
                        </div>
                      )}
                    </div>
                  )
                },
                {
                  label: t("adminMcp.modal.step2"),
                  content: (
                    <div className="grid gap-4 py-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label>{t("adminMcp.apps.form.appId")}</Label>
                          <Input value={newApp.app_id} onChange={e => setNewApp({ ...newApp, app_id: e.target.value })} />
                        </div>
                        <div className="space-y-2">
                          <Label>{t("adminMcp.apps.form.displayName")}</Label>
                          <Input value={newApp.name} onChange={e => setNewApp({ ...newApp, name: e.target.value })} />
                        </div>
                      </div>
                      <div className="space-y-2">
                        <Label>{t("adminMcp.apps.form.description")}</Label>
                        <Input value={newApp.description || ""} onChange={e => setNewApp({ ...newApp, description: e.target.value })} />
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label>{t("adminMcp.apps.form.iconUrl")}</Label>
                          <Input value={newApp.icon || ""} onChange={e => setNewApp({ ...newApp, icon: e.target.value })} />
                        </div>
                        <div className="space-y-2">
                          <Label>{t("adminMcp.apps.form.category")}</Label>
                          <Select value={newApp.category || "Other"} onValueChange={v => setNewApp({ ...newApp, category: v })}>
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="CRM">CRM</SelectItem>
                              <SelectItem value="Communication">Communication</SelectItem>
                              <SelectItem value="Support">Support</SelectItem>
                              <SelectItem value="Marketing">Marketing</SelectItem>
                              <SelectItem value="Scheduling">Scheduling</SelectItem>
                              <SelectItem value="Payments">Payments</SelectItem>
                              <SelectItem value="Analytics">Analytics</SelectItem>
                              <SelectItem value="Other">Other</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <Label>{t("adminMcp.apps.form.transport")}</Label>
                        <Select value={newApp.transport} onValueChange={v => setNewApp({ ...newApp, transport: v })}>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="oauth">OAuth</SelectItem>
                            <SelectItem value="stdio">Stdio</SelectItem>
                            <SelectItem value="sse">SSE</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      {newApp.transport === "oauth" && (
                        <div className="space-y-2">
                          <Label>{t("adminMcp.apps.form.oauthScopes")}</Label>
                          <Input value={(newApp.oauth_scopes as any) || ""} onChange={e => setNewApp({ ...newApp, oauth_scopes: e.target.value as any })} />
                        </div>
                      )}
                      <div className="space-y-2">
                        <Label>{t("adminMcp.apps.form.launchConfig")}</Label>
                        <p className="text-xs text-muted-foreground">Example: {`{"command": "uv", "args": ["run", "python", "-m", "xagent.web.tools.mcp.gmail"], "env_mapping": {"GOOGLE_ACCESS_TOKEN": "access_token"}}`}</p>
                        <Textarea
                          value={typeof newApp.launch_config === 'string' ? newApp.launch_config : JSON.stringify(newApp.launch_config, null, 2)}
                          onChange={e => setNewApp({ ...newApp, launch_config: e.target.value })}
                          className="font-mono text-xs"
                          rows={6}
                        />
                      </div>
                    </div>
                  )
                }
              ]}
              currentStep={step}
              className="mt-4"
            />
          )}

          <DialogFooter>
            {step === 1 && (
              <>
                {isCreatingProvider && !isStandaloneProvider && (
                  <Button variant="outline" onClick={() => setIsCreatingProvider(false)}>{t("common.cancel")}</Button>
                )}
                <Button onClick={handleNextStep1} disabled={isCreatingProvider && (!newProvider.provider_name || !newProvider.client_id)}>
                  {editingProviderId ? (t("adminMcp.modal.saveProvider")) : (t("adminMcp.modal.next"))}
                </Button>
              </>
            )}
            {step === 2 && (
              <>
                <Button variant="outline" onClick={() => setStep(1)}>{t("adminMcp.modal.back")}</Button>
                <Button onClick={handleNextStep2} disabled={!newApp.app_id || !newApp.name}>{t("adminMcp.modal.saveApp")}</Button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
