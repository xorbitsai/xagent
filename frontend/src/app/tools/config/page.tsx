"use client"

import { useEffect, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"
import { Database, KeyRound, Loader2, PlusCircle, Search, Trash2, Wrench } from "lucide-react"

interface ConfigurableToolField {
  label: string
  required: boolean
  secret: boolean
  source: "db" | "env" | "none"
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
  source: "db" | "env" | "none"
  masked: string
}

type SqlDbType = "postgresql" | "mysql" | "mariadb" | "mssql" | "sqlite"

const DEFAULT_PORTS: Record<Exclude<SqlDbType, "sqlite">, string> = {
  postgresql: "5432",
  mysql: "3306",
  mariadb: "3306",
  mssql: "1433",
}

export default function ToolsConfigPage() {
  const { t } = useI18n()
  const [searchQuery, setSearchQuery] = useState("")
  const [activeTab, setActiveTab] = useState<"web_search" | "db_connections">("web_search")

  const [configurableTools, setConfigurableTools] = useState<ConfigurableTool[]>([])
  const [sqlConnections, setSqlConnections] = useState<SqlConnectionItem[]>([])

  const [isCredentialDialogOpen, setIsCredentialDialogOpen] = useState(false)
  const [editingConfigTool, setEditingConfigTool] = useState<ConfigurableTool | null>(null)
  const [credentialValues, setCredentialValues] = useState<Record<string, string>>({})
  const [isSavingCredentials, setIsSavingCredentials] = useState(false)

  const [isSqlDialogOpen, setIsSqlDialogOpen] = useState(false)
  const [sqlFormName, setSqlFormName] = useState("")
  const [sqlFormType, setSqlFormType] = useState<SqlDbType>("postgresql")
  const [sqlFormHost, setSqlFormHost] = useState("")
  const [sqlFormPort, setSqlFormPort] = useState(DEFAULT_PORTS.postgresql)
  const [sqlFormDatabase, setSqlFormDatabase] = useState("")
  const [sqlFormUsername, setSqlFormUsername] = useState("")
  const [sqlFormPassword, setSqlFormPassword] = useState("")
  const [sqlFormParams, setSqlFormParams] = useState("")
  const [sqlFormSqlitePath, setSqlFormSqlitePath] = useState("")
  const [isSavingSql, setIsSavingSql] = useState(false)

  useEffect(() => {
    void Promise.all([loadConfigurableTools(), loadSqlConnections()])
  }, [])

  const loadConfigurableTools = async () => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/tools/configurable`)
      if (!response.ok) {
        setConfigurableTools([])
        return
      }
      const data = await response.json()
      setConfigurableTools(data.tools || [])
    } catch {
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
    } catch {
      setSqlConnections([])
    }
  }

  const getCredentialStatusLabel = (source: "db" | "env" | "none") => {
    if (source === "db") return t("tools.credentials.status.db")
    if (source === "env") return t("tools.credentials.status.env")
    return t("tools.credentials.status.none")
  }

  const openCredentialDialog = (tool: ConfigurableTool) => {
    setEditingConfigTool(tool)
    setCredentialValues({})
    setIsCredentialDialogOpen(true)
  }

  const handleSaveCredentials = async () => {
    if (!editingConfigTool) return
    const payload: Record<string, { value: string }> = {}
    Object.entries(credentialValues).forEach(([fieldName, value]) => {
      const normalized = value.trim()
      if (normalized) payload[fieldName] = { value: normalized }
    })

    if (Object.keys(payload).length === 0) {
      setIsCredentialDialogOpen(false)
      return
    }

    setIsSavingCredentials(true)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/tools/${editingConfigTool.tool_name}/credentials`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ credentials: payload }),
      })
      if (!response.ok) {
        const err = await response.json()
        alert(err.detail || t("tools.credentials.saveFailed"))
        return
      }
      await loadConfigurableTools()
      setIsCredentialDialogOpen(false)
    } catch {
      alert(t("tools.credentials.saveFailed"))
    } finally {
      setIsSavingCredentials(false)
    }
  }

  const handleSaveSqlConnection = async () => {
    const name = sqlFormName.trim()
    if (!name) {
      alert(t("tools.database.validation.required"))
      return
    }

    let connectionUrl = ""
    if (sqlFormType === "sqlite") {
      const sqlitePath = sqlFormSqlitePath.trim()
      if (!sqlitePath) {
        alert(t("tools.database.validation.sqlitePathRequired"))
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
        alert(t("tools.database.validation.required"))
        return
      }

      const encodedUser = encodeURIComponent(username)
      const encodedPass = password ? `:${encodeURIComponent(password)}` : ""
      const auth = `${encodedUser}${encodedPass}@`
      const query = params ? `?${params.replace(/^\?/, "")}` : ""

      connectionUrl = `${sqlFormType}://${auth}${host}:${port}/${database}${query}`
    }

    setIsSavingSql(true)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/tools/sql-connections/${encodeURIComponent(name)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ connection_url: connectionUrl }),
      })
      if (!response.ok) {
        const err = await response.json()
        alert(err.detail || t("tools.database.saveFailed"))
        return
      }
      setIsSqlDialogOpen(false)
      setSqlFormName("")
      setSqlFormType("postgresql")
      setSqlFormHost("")
      setSqlFormPort(DEFAULT_PORTS.postgresql)
      setSqlFormDatabase("")
      setSqlFormUsername("")
      setSqlFormPassword("")
      setSqlFormParams("")
      setSqlFormSqlitePath("")
      await loadSqlConnections()
    } catch {
      alert(t("tools.database.saveFailed"))
    } finally {
      setIsSavingSql(false)
    }
  }

  const handleDeleteSqlConnection = async (name: string) => {
    if (!confirm(t("tools.database.deleteConfirm", { name }))) return
    try {
      const response = await apiRequest(`${getApiUrl()}/api/tools/sql-connections/${encodeURIComponent(name)}`, {
        method: "DELETE",
      })
      if (!response.ok) {
        const err = await response.json()
        alert(err.detail || t("tools.database.deleteFailed"))
        return
      }
      await loadSqlConnections()
    } catch {
      alert(t("tools.database.deleteFailed"))
    }
  }

  const filteredConfigurableTools = configurableTools.filter((tool) => {
    const title = (tool.display_name || tool.tool_name).toLowerCase()
    return title.includes(searchQuery.toLowerCase())
  })

  const filteredSqlConnections = sqlConnections.filter((item) =>
    item.name.toLowerCase().includes(searchQuery.toLowerCase())
  )

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold mb-1">{t("tools.configHeader.title")}</h1>
          <p className="text-muted-foreground">{t("tools.configHeader.description")}</p>
        </div>

        <div className="w-72 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder={t("tools.list.searchPlaceholder")}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
          />
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as "web_search" | "db_connections")} className="w-full">
        <TabsList className="w-full justify-start bg-transparent p-0 h-auto border-b border-border/80 rounded-none flex">
          <div className="flex space-x-4">
            <TabsTrigger value="web_search" className="data-[state=active]:text-primary font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary">
              {t("tools.tabs.webSearch")}
            </TabsTrigger>
            <TabsTrigger value="db_connections" className="data-[state=active]:text-primary font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary">
              {t("tools.tabs.databaseConnections")}
            </TabsTrigger>
          </div>
        </TabsList>

        <div className="mt-6">
          <TabsContent value="web_search" className="m-0 w-full">
            {filteredConfigurableTools.length === 0 ? (
              <div className="w-full flex justify-center py-6">
                <EmptyState
                  title={t("tools.credentials.empty.title")}
                  description={t("tools.credentials.empty.description")}
                />
              </div>
            ) : (
              <div className="w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredConfigurableTools.map((tool) => (
                  <Card key={tool.tool_name} className="hover:shadow-md transition-all duration-300 border-border/50 hover:border-primary hover:-translate-y-1">
                    <CardContent className="p-6">
                      <div className="mb-4 flex items-start justify-between">
                        <div className="flex gap-4">
                          <div className="mt-1 bg-muted/50 p-3 rounded-lg h-fit">
                            <KeyRound className="h-6 w-6 text-slate-500" />
                          </div>
                          <div>
                            <h3 className="font-semibold text-base mb-1">{tool.display_name || tool.tool_name}</h3>
                            <Badge variant={tool.configured ? "secondary" : "outline"}>
                              {tool.configured ? t("tools.credentials.configured") : t("tools.credentials.notConfigured")}
                            </Badge>
                          </div>
                        </div>
                      </div>

                      <div className="space-y-2 text-xs text-muted-foreground mb-4">
                        {Object.entries(tool.fields).map(([fieldName, field]) => (
                          <div key={fieldName} className="flex items-center justify-between gap-2">
                            <span>{field.label}</span>
                            <span className="truncate">{field.masked || getCredentialStatusLabel(field.source)}</span>
                          </div>
                        ))}
                      </div>

                      <Button variant="outline" size="sm" className="w-full" onClick={() => openCredentialDialog(tool)}>
                        {t("tools.credentials.configure")}
                      </Button>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="db_connections" className="m-0 w-full">
            <div className="w-full space-y-4">
              <div className="flex justify-end">
                <Button
                  onClick={() => {
                    setSqlFormType("postgresql")
                    setSqlFormHost("")
                    setSqlFormPort(DEFAULT_PORTS.postgresql)
                    setSqlFormDatabase("")
                    setSqlFormUsername("")
                    setSqlFormPassword("")
                    setSqlFormParams("")
                    setSqlFormSqlitePath("")
                    setIsSqlDialogOpen(true)
                  }}
                >
                  <PlusCircle className="mr-2 h-4 w-4" />
                  {t("tools.database.addConnection")}
                </Button>
              </div>

              {filteredSqlConnections.length === 0 ? (
                <div className="w-full flex justify-center py-6">
                  <EmptyState
                    title={t("tools.database.empty.title")}
                    description={t("tools.database.empty.description")}
                  />
                </div>
              ) : (
                <div className="w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {filteredSqlConnections.map((item) => (
                    <Card key={item.name} className="group hover:shadow-lg transition-all duration-300 border-border/60 hover:border-primary/50 hover:-translate-y-1 overflow-hidden">
                      <CardContent className="p-0">
                        <div className="p-5">
                          <div className="mb-4 flex items-start justify-between gap-3">
                            <div className="flex items-start gap-3 min-w-0">
                              <div className="mt-0.5 bg-muted/60 p-2.5 rounded-lg h-fit">
                                <Database className="h-5 w-5 text-slate-600" />
                              </div>
                              <div className="min-w-0">
                                <h3 className="truncate font-semibold text-base text-foreground">{item.name}</h3>
                                <div className="mt-1 flex flex-wrap items-center gap-2">
                                  <Badge variant="outline" className="text-[11px]">
                                    {t("tools.database.connectionBadge")}
                                  </Badge>
                                  <Badge variant={item.source === "db" ? "secondary" : "outline"} className="text-[11px]">
                                    {t(`tools.credentials.status.${item.source}`)}
                                  </Badge>
                                </div>
                              </div>
                            </div>

                            {item.source === "db" ? (
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 opacity-80 group-hover:opacity-100"
                                onClick={() => handleDeleteSqlConnection(item.name)}
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            ) : null}
                          </div>

                          <div className="space-y-2">
                            <p className="text-xs font-medium text-muted-foreground">{t("tools.database.maskedValue")}</p>
                            <div className="rounded-md border border-border/70 bg-muted/30 px-3 py-2">
                              <p className="text-xs text-foreground/80 break-all leading-relaxed">{item.masked || "--"}</p>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </div>
          </TabsContent>
        </div>
      </Tabs>

      <Dialog open={isSqlDialogOpen} onOpenChange={setIsSqlDialogOpen}>
        <DialogContent className="max-w-xl">
          <DialogHeader>
            <DialogTitle>{t("tools.database.dialog.title")}</DialogTitle>
            <DialogDescription>{t("tools.database.dialog.description")}</DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="sql-conn-name">{t("tools.database.connectionName")}</Label>
              <Input
                id="sql-conn-name"
                value={sqlFormName}
                onChange={(e) => setSqlFormName(e.target.value)}
                placeholder={t("tools.database.connectionNamePlaceholder")}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="sql-conn-type">{t("tools.database.dbType")}</Label>
              <Select
                value={sqlFormType}
                onValueChange={(value: string) => {
                  const typed = value as SqlDbType
                  setSqlFormType(typed)
                  if (typed !== "sqlite") {
                    setSqlFormPort(DEFAULT_PORTS[typed])
                  }
                }}
                options={[
                  { value: "postgresql", label: t("tools.database.types.postgresql") },
                  { value: "mysql", label: t("tools.database.types.mysql") },
                  { value: "mariadb", label: t("tools.database.types.mariadb") },
                  { value: "mssql", label: t("tools.database.types.mssql") },
                  { value: "sqlite", label: t("tools.database.types.sqlite") },
                ]}
                placeholder={t("tools.database.dbType")}
              />
            </div>

            {sqlFormType === "sqlite" ? (
              <div className="space-y-2">
                <Label htmlFor="sql-conn-sqlite-path">{t("tools.database.sqlitePath")}</Label>
                <Input
                  id="sql-conn-sqlite-path"
                  value={sqlFormSqlitePath}
                  onChange={(e) => setSqlFormSqlitePath(e.target.value)}
                  placeholder={t("tools.database.sqlitePathPlaceholder")}
                />
              </div>
            ) : (
              <>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="sql-conn-host">{t("tools.database.host")}</Label>
                    <Input
                      id="sql-conn-host"
                      value={sqlFormHost}
                      onChange={(e) => setSqlFormHost(e.target.value)}
                      placeholder={t("tools.database.hostPlaceholder")}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="sql-conn-port">{t("tools.database.port")}</Label>
                    <Input
                      id="sql-conn-port"
                      value={sqlFormPort}
                      onChange={(e) => setSqlFormPort(e.target.value)}
                      placeholder={t("tools.database.portPlaceholder")}
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="sql-conn-database">{t("tools.database.databaseName")}</Label>
                  <Input
                    id="sql-conn-database"
                    value={sqlFormDatabase}
                    onChange={(e) => setSqlFormDatabase(e.target.value)}
                    placeholder={t("tools.database.databaseNamePlaceholder")}
                  />
                </div>

                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="sql-conn-username">{t("tools.database.username")}</Label>
                    <Input
                      id="sql-conn-username"
                      value={sqlFormUsername}
                      onChange={(e) => setSqlFormUsername(e.target.value)}
                      placeholder={t("tools.database.usernamePlaceholder")}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="sql-conn-password">{t("tools.database.password")}</Label>
                    <Input
                      id="sql-conn-password"
                      type="password"
                      value={sqlFormPassword}
                      onChange={(e) => setSqlFormPassword(e.target.value)}
                      placeholder={t("tools.database.passwordPlaceholder")}
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="sql-conn-params">{t("tools.database.params")}</Label>
                  <Input
                    id="sql-conn-params"
                    value={sqlFormParams}
                    onChange={(e) => setSqlFormParams(e.target.value)}
                    placeholder={t("tools.database.paramsPlaceholder")}
                  />
                </div>
              </>
            )}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsSqlDialogOpen(false)}>
              {t("tools.mcp.buttons.cancel")}
            </Button>
            <Button onClick={handleSaveSqlConnection} disabled={isSavingSql}>
              {isSavingSql && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {t("tools.database.save")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={isCredentialDialogOpen} onOpenChange={setIsCredentialDialogOpen}>
        <DialogContent className="max-w-xl">
          <DialogHeader>
            <DialogTitle>{t("tools.credentials.dialog.title")}</DialogTitle>
            <DialogDescription>
              {editingConfigTool
                ? t("tools.credentials.dialog.description", {
                    tool: editingConfigTool.display_name || editingConfigTool.tool_name,
                  })
                : ""}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            {editingConfigTool &&
              Object.entries(editingConfigTool.fields).map(([fieldName, field]) => (
                <div key={fieldName} className="space-y-2">
                  <Label htmlFor={`cred-${fieldName}`}>
                    {field.label}
                    {field.required ? " *" : ""}
                  </Label>
                  <Input
                    id={`cred-${fieldName}`}
                    type={field.secret ? "password" : "text"}
                    value={credentialValues[fieldName] || ""}
                    placeholder={field.masked || getCredentialStatusLabel(field.source)}
                    onChange={(e) =>
                      setCredentialValues((prev) => ({ ...prev, [fieldName]: e.target.value }))
                    }
                  />
                  <p className="text-xs text-muted-foreground">
                    {t("tools.credentials.currentSource")}: {getCredentialStatusLabel(field.source)}
                  </p>
                </div>
              ))}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsCredentialDialogOpen(false)}>
              {t("tools.mcp.buttons.cancel")}
            </Button>
            <Button onClick={handleSaveCredentials} disabled={isSavingCredentials}>
              {isSavingCredentials && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {t("tools.credentials.save")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

function EmptyState({
  title,
  description,
}: {
  title: string
  description: string
}) {
  return (
    <div className="mx-auto w-[760px] max-w-[calc(100%-2rem)] min-h-[240px] flex flex-col items-center justify-center text-center py-16 text-muted-foreground border border-dashed rounded-lg">
      <Wrench className="h-10 w-10 mx-auto mb-4 opacity-50" />
      <div className="font-medium mb-1">{title}</div>
      <div className="text-sm">{description}</div>
    </div>
  )
}
