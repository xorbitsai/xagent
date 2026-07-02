"use client"

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Database, Globe, Loader2, Trash2 } from "lucide-react"

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
import { toast } from "@/components/ui/sonner"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { useI18n } from "@/contexts/i18n-context"

type CredentialSource = string
type CredentialScope = "user" | "instance"

interface ConfigurableToolField {
  label: string
  required: boolean
  secret: boolean
  source: CredentialSource
  is_configured: boolean
  masked: string
}

interface ConfigurableTool {
  tool_name: string
  display_name?: string
  configured: boolean
  fields: Record<string, ConfigurableToolField>
}

type SqlDbType = "postgresql" | "mysql" | "mariadb" | "mssql" | "sqlite"

const DEFAULT_PORTS: Record<Exclude<SqlDbType, "sqlite">, string> = {
  postgresql: "5432",
  mysql: "3306",
  mariadb: "3306",
  mssql: "1433",
}

const SOURCE_DISPLAY_ORDER: CredentialSource[] = ["user", "instance", "env"]

export function ToolCredentialsPanel({
  scope,
  initialToolName,
  onAvailabilityChange,
  showTitle = true,
  endpointBase,
  credentialScopeKey,
  sourceLabels = {},
}: {
  scope: CredentialScope
  initialToolName?: string | null
  onAvailabilityChange?: (available: boolean) => void
  showTitle?: boolean
  endpointBase?: string
  credentialScopeKey?: string
  sourceLabels?: Record<string, string>
}) {
  const { t } = useI18n()
  const [tools, setTools] = useState<ConfigurableTool[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [editingTool, setEditingTool] = useState<ConfigurableTool | null>(null)
  const [values, setValues] = useState<Record<string, string>>({})
  const [isSaving, setIsSaving] = useState(false)
  const [pendingDeletes, setPendingDeletes] = useState<Record<string, boolean>>({})
  const [isDeletingToolCredentials, setIsDeletingToolCredentials] = useState(false)
  const autoOpenedTool = useRef<string | null>(null)

  const [sqlName, setSqlName] = useState("")
  const [sqlType, setSqlType] = useState<SqlDbType>("postgresql")
  const [sqlHost, setSqlHost] = useState("")
  const [sqlPort, setSqlPort] = useState(DEFAULT_PORTS.postgresql)
  const [sqlDatabase, setSqlDatabase] = useState("")
  const [sqlUsername, setSqlUsername] = useState("")
  const [sqlPassword, setSqlPassword] = useState("")
  const [sqlParams, setSqlParams] = useState("")
  const [sqlitePath, setSqlitePath] = useState("")
  const [isSavingSql, setIsSavingSql] = useState(false)

  const selectedTool = useMemo(
    () =>
      editingTool
        ? tools.find((tool) => tool.tool_name === editingTool.tool_name) || editingTool
        : null,
    [editingTool, tools],
  )
  const isEditingSql = selectedTool?.tool_name === "sql_query"
  const currentScopeFieldNames = useMemo(
    () =>
      selectedTool
        ? Object.entries(selectedTool.fields)
            .filter(([, field]) => field.source === (credentialScopeKey || scope))
            .map(([fieldName]) => fieldName)
        : [],
    [credentialScopeKey, scope, selectedTool],
  )
  const requestBase = endpointBase
    ? `${getApiUrl()}${endpointBase}`
    : `${getApiUrl()}/api/tool-credentials`
  const requestSuffix = endpointBase ? "" : `?scope=${scope}`
  const toolCredentialUrl = useCallback((toolName?: string, fieldName?: string) => {
    const path = [toolName, fieldName]
      .filter(Boolean)
      .map((part) => encodeURIComponent(part as string))
      .join("/")
    return `${requestBase}${path ? `/${path}` : ""}${requestSuffix}`
  }, [requestBase, requestSuffix])

  const loadCredentials = useCallback(async () => {
    setIsLoading(true)
    try {
      const response = await apiRequest(toolCredentialUrl())
      if (!response.ok) {
        setTools([])
        onAvailabilityChange?.(false)
        return
      }
      const data = await response.json()
      setTools(data.tools || [])
      onAvailabilityChange?.(true)
    } catch (error) {
      console.error("Failed to load tool credentials:", error)
      setTools([])
    } finally {
      setIsLoading(false)
    }
  }, [onAvailabilityChange, toolCredentialUrl])

  useEffect(() => {
    void loadCredentials()
  }, [loadCredentials])

  useEffect(() => {
    if (!initialToolName) return
    const autoOpenKey = `${scope}:${initialToolName}`
    if (autoOpenedTool.current === autoOpenKey) return
    const tool = tools.find((item) => item.tool_name === initialToolName)
    if (!tool) return
    autoOpenedTool.current = autoOpenKey
    setEditingTool(tool)
    setValues({})
  }, [initialToolName, scope, tools])

  const getSourceLabel = (source: CredentialSource) => {
    if (sourceLabels[source]) return sourceLabels[source]
    if (source === "user") return t("tools.credentials.status.user")
    if (source === "instance") return t("tools.credentials.status.instance")
    if (source === "env") return t("tools.credentials.status.env")
    if (source === "none") return t("tools.credentials.status.none")
    const customLabelKey = `tools.credentials.status.${source}`
    const customLabel = t(customLabelKey)
    if (customLabel !== customLabelKey) return customLabel
    return t("tools.credentials.status.db")
  }

  const getToolLabel = useCallback((tool: ConfigurableTool) => {
    const labelKey = `tools.credentials.toolNames.${tool.tool_name}`
    const localizedLabel = t(labelKey)
    return localizedLabel !== labelKey ? localizedLabel : tool.display_name || tool.tool_name
  }, [t])

  const getToolConfiguredSources = (tool: ConfigurableTool) => {
    const configuredSources = new Set(
      Object.values(tool.fields)
        .filter((field) => field.is_configured && field.source !== "none")
        .map((field) => field.source),
    )
    const ordered = SOURCE_DISPLAY_ORDER.filter((source) => configuredSources.has(source))
    const custom = [...configuredSources].filter((source) => !SOURCE_DISPLAY_ORDER.includes(source))
    return [...ordered, ...custom]
  }

  const resetSqlForm = () => {
    setSqlName("")
    setSqlType("postgresql")
    setSqlHost("")
    setSqlPort(DEFAULT_PORTS.postgresql)
    setSqlDatabase("")
    setSqlUsername("")
    setSqlPassword("")
    setSqlParams("")
    setSqlitePath("")
  }

  async function saveProviderCredentials() {
    if (!editingTool) return

    const payload: Record<string, { value: string }> = {}
    Object.entries(values).forEach(([fieldName, value]) => {
      const normalized = value.trim()
      if (normalized) payload[fieldName] = { value: normalized }
    })
    if (Object.keys(payload).length === 0) {
      toast.error(t("tools.credentials.validation.required"))
      return
    }

    setIsSaving(true)
    try {
      const response = await apiRequest(
        toolCredentialUrl(editingTool.tool_name),
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ credentials: payload }),
        },
      )
      if (!response.ok) {
        const err = await response.json()
        toast.error(err.detail || t("tools.credentials.saveFailed"))
        return
      }
      await loadCredentials()
      setEditingTool(null)
      toast.success(t("tools.credentials.saveSuccess"))
    } catch (error) {
      console.error("Failed to save credentials:", error)
      toast.error(t("tools.credentials.saveFailed"))
    } finally {
      setIsSaving(false)
    }
  }

  async function saveSqlConnection() {
    const name = sqlName.trim()
    if (!name) {
      toast.error(t("tools.database.validation.required"))
      return
    }

    let connectionUrl = ""
    if (sqlType === "sqlite") {
      const path = sqlitePath.trim()
      if (!path) {
        toast.error(t("tools.database.validation.sqlitePathRequired"))
        return
      }
      connectionUrl = `sqlite:///${path}`
    } else {
      const host = sqlHost.trim()
      const port = sqlPort.trim() || DEFAULT_PORTS[sqlType]
      const database = sqlDatabase.trim()
      const username = sqlUsername.trim()
      const password = sqlPassword.trim()
      const params = sqlParams.trim()

      if (!host || !database || !username) {
        toast.error(t("tools.database.validation.required"))
        return
      }

      const encodedUser = encodeURIComponent(username)
      const encodedPass = password ? `:${encodeURIComponent(password)}` : ""
      const query = params ? `?${params.replace(/^\?/, "")}` : ""
      connectionUrl = `${sqlType}://${encodedUser}${encodedPass}@${host}:${port}/${database}${query}`
    }

    setIsSavingSql(true)
    try {
      const response = await apiRequest(
        toolCredentialUrl("sql_query"),
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ credentials: { [name]: { value: connectionUrl } } }),
        },
      )
      if (!response.ok) {
        const err = await response.json()
        toast.error(err.detail || t("tools.database.saveFailed"))
        return
      }
      resetSqlForm()
      await loadCredentials()
      setEditingTool(null)
      toast.success(t("tools.database.saveSuccess"))
    } catch (error) {
      console.error("Failed to save SQL connection:", error)
      toast.error(t("tools.database.saveFailed"))
    } finally {
      setIsSavingSql(false)
    }
  }

  async function deleteSqlConnection(name: string) {
    if (pendingDeletes[name]) return
    if (!confirm(t("tools.database.deleteConfirm", { name }))) return

    setPendingDeletes((prev) => ({ ...prev, [name]: true }))
    try {
      const response = await apiRequest(
        toolCredentialUrl("sql_query", name),
        { method: "DELETE" },
      )
      if (!response.ok) {
        const err = await response.json()
        toast.error(err.detail || t("tools.database.deleteFailed"))
        return
      }
      await loadCredentials()
      toast.success(t("tools.database.deleteSuccess"))
    } catch (error) {
      console.error("Failed to delete SQL connection:", error)
      toast.error(t("tools.database.deleteFailed"))
    } finally {
      setPendingDeletes((prev) => ({ ...prev, [name]: false }))
    }
  }

  async function deleteProviderCredential(toolName: string, fieldName: string) {
    const deleteKey = `${toolName}:${fieldName}`
    if (pendingDeletes[deleteKey]) return
    if (!confirm(t("tools.credentials.deleteConfirm", { field: fieldName }))) return

    setPendingDeletes((prev) => ({ ...prev, [deleteKey]: true }))
    try {
      const response = await apiRequest(
        toolCredentialUrl(toolName, fieldName),
        { method: "DELETE" },
      )
      if (!response.ok) {
        const err = await response.json()
        toast.error(err.detail || t("tools.credentials.deleteFailed"))
        return
      }
      await loadCredentials()
      setEditingTool(null)
      toast.success(t("tools.credentials.deleteSuccess"))
    } catch (error) {
      console.error("Failed to delete credential:", error)
      toast.error(t("tools.credentials.deleteFailed"))
    } finally {
      setPendingDeletes((prev) => ({ ...prev, [deleteKey]: false }))
    }
  }

  async function deleteAllToolCredentials() {
    if (
      !selectedTool ||
      currentScopeFieldNames.length === 0 ||
      isDeletingToolCredentials
    ) {
      return
    }

    const toolLabel = getToolLabel(selectedTool)
    if (!confirm(t("tools.credentials.deleteAllConfirm", { tool: toolLabel }))) return

    setIsDeletingToolCredentials(true)
    try {
      const responses = await Promise.all(
        currentScopeFieldNames.map((fieldName) =>
          apiRequest(
            toolCredentialUrl(selectedTool.tool_name, fieldName),
            { method: "DELETE" },
          ),
        ),
      )
      const failedResponse = responses.find((response) => !response.ok)
      if (failedResponse) {
        const err = await failedResponse.json()
        toast.error(err.detail || t("tools.credentials.deleteAllFailed"))
        return
      }
      await loadCredentials()
      setEditingTool(null)
      toast.success(t("tools.credentials.deleteAllSuccess"))
    } catch (error) {
      console.error("Failed to delete tool credentials:", error)
      toast.error(t("tools.credentials.deleteAllFailed"))
    } finally {
      setIsDeletingToolCredentials(false)
    }
  }

  return (
    <div className="space-y-6">
      {showTitle && (
        <div>
          <h2 className="text-xl font-semibold">{t("tools.configHeader.title")}</h2>
          <p className="text-sm text-muted-foreground">{t("tools.configHeader.description")}</p>
        </div>
      )}

      {isLoading ? (
        <div className="flex justify-center py-10">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
          {tools.map((tool) => {
            const toolLabel = getToolLabel(tool)
            return (
              <Card key={tool.tool_name} className="border-border/60">
                <CardContent className="p-5">
                  <div className="mb-4 flex items-start gap-3">
                    <div className="rounded-lg bg-muted/60 p-2.5">
                      {tool.tool_name === "sql_query" ? (
                        <Database className="h-5 w-5 text-slate-600" />
                      ) : (
                        <Globe className="h-5 w-5 text-slate-600" />
                      )}
                    </div>
                    <div className="min-w-0">
                      <h3 className="truncate font-semibold">{toolLabel}</h3>
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {getToolConfiguredSources(tool).length > 0 ? (
                          getToolConfiguredSources(tool).map((source) => (
                            <Badge
                              key={source}
                              variant={source === "env" ? "outline" : "secondary"}
                            >
                              {getSourceLabel(source)}
                            </Badge>
                          ))
                        ) : (
                          <Badge variant="outline">{getSourceLabel("none")}</Badge>
                        )}
                      </div>
                    </div>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full"
                    onClick={() => {
                      setEditingTool(tool)
                      setValues({})
                      if (tool.tool_name === "sql_query") resetSqlForm()
                    }}
                  >
                    {t("tools.credentials.configure")}
                  </Button>
                </CardContent>
              </Card>
            )
          })}
        </div>
      )}

      <Dialog open={Boolean(editingTool)} onOpenChange={(open) => !open && setEditingTool(null)}>
        <DialogContent className="max-w-xl">
          <DialogHeader>
            <DialogTitle>{t("tools.credentials.dialog.title")}</DialogTitle>
            <DialogDescription>
              {selectedTool
                ? t("tools.credentials.dialog.description", {
                    tool: getToolLabel(selectedTool),
                  })
                : ""}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            {selectedTool && isEditingSql ? (
              <div className="space-y-5">
                {Object.entries(selectedTool.fields).length > 0 && (
                  <div className="grid grid-cols-1 gap-4">
                    {Object.entries(selectedTool.fields).map(([name, field]) => {
                      const isDeleting = Boolean(pendingDeletes[name])
                      return (
                        <div key={name} className="rounded-md border border-border/70 p-4">
                          <div className="mb-3 flex items-start justify-between gap-3">
                            <div className="flex min-w-0 items-start gap-3">
                              <Database className="mt-1 h-5 w-5 shrink-0 text-slate-600" />
                              <div className="min-w-0">
                                <h4 className="truncate font-medium">{name}</h4>
                                <Badge variant={field.source === "env" ? "outline" : "secondary"} className="mt-1">
                                  {getSourceLabel(field.source)}
                                </Badge>
                              </div>
                            </div>
                            {field.source === scope && (
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8"
                                onClick={() => deleteSqlConnection(name)}
                                disabled={isDeleting}
                              >
                                {isDeleting ? (
                                  <Loader2 className="h-4 w-4 animate-spin" />
                                ) : (
                                  <Trash2 className="h-4 w-4" />
                                )}
                              </Button>
                            )}
                          </div>
                          <p className="break-all text-xs text-muted-foreground">{field.masked || "--"}</p>
                        </div>
                      )
                    })}
                  </div>
                )}

                <div className="grid grid-cols-1 gap-4">
                  <div className="space-y-2">
                    <Label>{t("tools.database.connectionName")}</Label>
                    <Input value={sqlName} onChange={(e) => setSqlName(e.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <Label>{t("tools.database.dbType")}</Label>
                    <Select
                      value={sqlType}
                      onValueChange={(value) => {
                        const nextType = value as SqlDbType
                        setSqlType(nextType)
                        if (nextType !== "sqlite") setSqlPort(DEFAULT_PORTS[nextType])
                      }}
                      options={[
                        { value: "postgresql", label: t("tools.database.types.postgresql") },
                        { value: "mysql", label: t("tools.database.types.mysql") },
                        { value: "mariadb", label: t("tools.database.types.mariadb") },
                        { value: "mssql", label: t("tools.database.types.mssql") },
                        { value: "sqlite", label: t("tools.database.types.sqlite") },
                      ]}
                    />
                  </div>
                  {sqlType === "sqlite" ? (
                    <div className="space-y-2">
                      <Label>{t("tools.database.sqlitePath")}</Label>
                      <Input value={sqlitePath} onChange={(e) => setSqlitePath(e.target.value)} />
                    </div>
                  ) : (
                    <>
                      <div className="space-y-2">
                        <Label>{t("tools.database.host")}</Label>
                        <Input value={sqlHost} onChange={(e) => setSqlHost(e.target.value)} />
                      </div>
                      <div className="space-y-2">
                        <Label>{t("tools.database.port")}</Label>
                        <Input value={sqlPort} onChange={(e) => setSqlPort(e.target.value)} />
                      </div>
                      <div className="space-y-2">
                        <Label>{t("tools.database.databaseName")}</Label>
                        <Input value={sqlDatabase} onChange={(e) => setSqlDatabase(e.target.value)} />
                      </div>
                      <div className="space-y-2">
                        <Label>{t("tools.database.username")}</Label>
                        <Input value={sqlUsername} onChange={(e) => setSqlUsername(e.target.value)} />
                      </div>
                      <div className="space-y-2">
                        <Label>{t("tools.database.password")}</Label>
                        <Input
                          type="password"
                          value={sqlPassword}
                          onChange={(e) => setSqlPassword(e.target.value)}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label>{t("tools.database.params")}</Label>
                        <Input value={sqlParams} onChange={(e) => setSqlParams(e.target.value)} />
                      </div>
                    </>
                  )}
                </div>
              </div>
            ) : (
              selectedTool &&
              Object.entries(selectedTool.fields).map(([fieldName, field]) => {
                const deleteKey = `${selectedTool.tool_name}:${fieldName}`
                const isDeleting = Boolean(pendingDeletes[deleteKey])
                return (
                  <div key={fieldName} className="space-y-2">
                    <div className="flex items-center justify-between gap-3">
                      <Label htmlFor={`cred-${scope}-${fieldName}`}>
                        {field.label}
                        {field.required ? " *" : ""}
                      </Label>
                      {field.source === scope && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          onClick={() => deleteProviderCredential(selectedTool.tool_name, fieldName)}
                          disabled={isDeleting}
                        >
                          {isDeleting ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Trash2 className="h-4 w-4" />
                          )}
                        </Button>
                      )}
                    </div>
                    <Input
                      id={`cred-${scope}-${fieldName}`}
                      type={field.secret ? "password" : "text"}
                      value={values[fieldName] || ""}
                      placeholder={field.masked || getSourceLabel(field.source)}
                      onChange={(event) =>
                        setValues((prev) => ({ ...prev, [fieldName]: event.target.value }))
                      }
                    />
                    <p className="text-xs text-muted-foreground">
                      {t("tools.credentials.currentSource")}: {getSourceLabel(field.source)}
                    </p>
                  </div>
                )
              })
            )}
          </div>
          <DialogFooter>
            <div className="flex w-full flex-col-reverse gap-2 sm:flex-row sm:items-center sm:justify-between">
              {currentScopeFieldNames.length > 0 ? (
                <Button
                  variant="destructive"
                  onClick={deleteAllToolCredentials}
                  disabled={isDeletingToolCredentials || isSavingSql || isSaving}
                >
                  {isDeletingToolCredentials ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                  {t("tools.credentials.deleteAll")}
                </Button>
              ) : (
                <div />
              )}
              <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
                <Button variant="outline" onClick={() => setEditingTool(null)}>
                  {t("tools.mcp.buttons.cancel")}
                </Button>
                <Button
                  onClick={isEditingSql ? saveSqlConnection : saveProviderCredentials}
                  disabled={isEditingSql ? isSavingSql : isSaving}
                >
                  {(isEditingSql ? isSavingSql : isSaving) && (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  )}
                  {isEditingSql ? t("tools.database.save") : t("tools.credentials.save")}
                </Button>
              </div>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
