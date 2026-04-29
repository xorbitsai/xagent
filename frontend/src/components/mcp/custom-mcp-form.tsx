import React, { useState } from "react"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select-radix"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { ChevronDown, ChevronRight, Plus, Trash2 } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { MCPServerFormData } from "./custom-api-form"

interface CustomMcpFormProps {
  mcpFormData: MCPServerFormData
  setMcpFormData: React.Dispatch<React.SetStateAction<MCPServerFormData>>
  transports: any[]
}

export function CustomMcpForm({
  mcpFormData,
  setMcpFormData,
  transports
}: CustomMcpFormProps) {
  const { t } = useI18n()
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false)

  // Default to sse if not set
  const transport = mcpFormData.transport || "sse"

  const updateConfig = (key: string, value: any) => {
    setMcpFormData((prev: MCPServerFormData) => ({
      ...prev,
      config: { ...prev.config, [key]: value }
    }))
  }

  const updateAuth = (key: string, value: any) => {
    setMcpFormData((prev: MCPServerFormData) => ({
      ...prev,
      config: {
        ...prev.config,
        auth: { ...(prev.config?.auth || {}), [key]: value }
      }
    }))
  }

  // Handle headers state (array of {key, value} for the UI, but config expects an object)
  const headersObj = mcpFormData.config?.headers || {}
  const [headersList, setHeadersList] = useState<{ key: string, value: string }[]>(
    Object.keys(headersObj).length > 0
      ? Object.entries(headersObj).map(([k, v]) => ({ key: k, value: String(v) }))
      : []
  )

  // Track original masked values to restore them on blur if empty
  const [originalAuth, setOriginalAuth] = useState<{
    bearer_token?: string;
    api_key_value?: string;
    client_secret?: string;
  }>({
    bearer_token: mcpFormData.config?.auth?.bearer_token === '********' ? '********' : undefined,
    api_key_value: mcpFormData.config?.auth?.api_key_value === '********' ? '********' : undefined,
    client_secret: mcpFormData.config?.auth?.client_secret === '********' ? '********' : undefined,
  })

  const syncHeaders = (newList: { key: string, value: string }[]) => {
    setHeadersList(newList)
    const newHeadersObj: Record<string, string> = {}
    newList.forEach(h => {
      if (h.key.trim()) newHeadersObj[h.key.trim()] = h.value.trim()
    })
    updateConfig("headers", Object.keys(newHeadersObj).length > 0 ? newHeadersObj : undefined)
  }

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="name">{t('tools.mcp.form.nameLabel')}</Label>
        <Input
          id="name"
          value={mcpFormData.name || ""}
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
          onChange={(e) => setMcpFormData((prev: MCPServerFormData) => ({ ...prev, description: e.target.value }))}
          placeholder={t('tools.mcp.form.descriptionPlaceholder')}
        />
      </div>

      <div className="space-y-2">
        <Label>{t('tools.mcp.dialog.transport')}</Label>
        <div className="flex bg-slate-100 p-1 rounded-md">
          <button
            type="button"
            className={`flex-1 py-1.5 text-sm font-medium rounded-md transition-colors ${transport === "sse" ? "bg-blue-600 text-white shadow" : "text-slate-600 hover:text-slate-900 hover:bg-slate-200"}`}
            onClick={() => setMcpFormData((prev: MCPServerFormData) => ({ ...prev, transport: "sse" }))}
          >
            SSE / HTTP
          </button>
          <button
            type="button"
            className={`flex-1 py-1.5 text-sm font-medium rounded-md transition-colors ${transport === "stdio" ? "bg-blue-600 text-white shadow" : "text-slate-600 hover:text-slate-900 hover:bg-slate-200"}`}
            onClick={() => setMcpFormData((prev: MCPServerFormData) => ({ ...prev, transport: "stdio" }))}
          >
            STDIO
          </button>
          <button
            type="button"
            className={`flex-1 py-1.5 text-sm font-medium rounded-md transition-colors ${transport === "websocket" ? "bg-blue-600 text-white shadow" : "text-slate-600 hover:text-slate-900 hover:bg-slate-200"}`}
            onClick={() => setMcpFormData((prev: MCPServerFormData) => ({ ...prev, transport: "websocket" }))}
          >
            WebSocket
          </button>
        </div>
      </div>

      {transport === "stdio" ? (
        <>
          <div className="space-y-2">
            <Label htmlFor="command">{t('tools.mcp.dialog.command')}</Label>
            <Input
              id="command"
              value={mcpFormData.config?.command || ""}
              onChange={(e) => updateConfig("command", e.target.value)}
              placeholder={t('tools.mcp.dialog.commandPlaceholder')}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="args">{t('tools.mcp.dialog.arguments')}</Label>
            <Input
              id="args"
              value={Array.isArray(mcpFormData.config?.args) ? mcpFormData.config.args.join(" ") : (mcpFormData.config?.args || "")}
              onChange={(e) => {
                // Split by space for simple arg passing (in a real app, might want a better parser)
                const argsArr = e.target.value.split(" ").filter(Boolean)
                updateConfig("args", argsArr)
              }}
              placeholder={t('tools.mcp.dialog.argumentsPlaceholder')}
            />
          </div>
        </>
      ) : (
        <>
          <div className="space-y-2">
            <Label htmlFor="url">{t('tools.mcp.dialog.url')}</Label>
            <Input
              id="url"
              value={mcpFormData.config?.url || ""}
              onChange={(e) => updateConfig("url", e.target.value)}
              placeholder={transport === "websocket" ? "wss://mcp.example.com/ws" : "https://mcp.example.com"}
            />
          </div>

          <div className="space-y-2">
            <Label className="flex items-center gap-1">
              {t('tools.mcp.dialog.authentication')} <span className="text-slate-400 text-xs">(?)</span>
            </Label>
            <Select
              value={mcpFormData.config?.auth?.type || "none"}
              onValueChange={(val) => updateAuth("type", val)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">{t('tools.mcp.dialog.authTypes.none')}</SelectItem>
                <SelectItem value="bearer">{t('tools.mcp.dialog.authTypes.bearer')}</SelectItem>
                <SelectItem value="api_key">{t('tools.mcp.dialog.authTypes.apiKey')}</SelectItem>
                <SelectItem value="oauth2">{t('tools.mcp.dialog.authTypes.oauth2')}</SelectItem>
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
                onChange={(e) => updateAuth("bearer_token", e.target.value)}
                onFocus={() => {
                  if (mcpFormData.config?.auth?.bearer_token === "********") {
                    updateAuth("bearer_token", "")
                  }
                }}
                onBlur={() => {
                  if ((!mcpFormData.config?.auth?.bearer_token || mcpFormData.config.auth.bearer_token === "") && originalAuth.bearer_token) {
                    updateAuth("bearer_token", "********")
                  }
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
                  onChange={(e) => updateAuth("api_key_value", e.target.value)}
                  onFocus={() => {
                    if (mcpFormData.config?.auth?.api_key_value === "********") {
                      updateAuth("api_key_value", "")
                    }
                  }}
                  onBlur={() => {
                    if ((!mcpFormData.config?.auth?.api_key_value || mcpFormData.config.auth.api_key_value === "") && originalAuth.api_key_value) {
                      updateAuth("api_key_value", "********")
                    }
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
                  onChange={(e) => updateAuth("client_secret", e.target.value)}
                  onFocus={() => {
                    if (mcpFormData.config?.auth?.client_secret === "********") {
                      updateAuth("client_secret", "")
                    }
                  }}
                  onBlur={() => {
                    if ((!mcpFormData.config?.auth?.client_secret || mcpFormData.config.auth.client_secret === "") && originalAuth.client_secret) {
                      updateAuth("client_secret", "********")
                    }
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
