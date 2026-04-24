import React, { useEffect, useState } from "react"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { useI18n } from "@/contexts/i18n-context"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select-radix"
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible"
import { ChevronDown, ChevronRight, Plus, Trash2 } from "lucide-react"

export interface MCPServerFormData {
    name: string;
    transport: string;
    description: string;
    url?: string;
    method?: string;
    headers?: Record<string, string>;
    config?: Record<string, any>;
    [key: string]: any;
}

interface CustomApiFormProps {
    mcpFormData: MCPServerFormData
    setMcpFormData: React.Dispatch<React.SetStateAction<MCPServerFormData>>
    customApiEnv: { key: string, value: string }[]
    setCustomApiEnv: React.Dispatch<React.SetStateAction<{ key: string, value: string }[]>>
    originalEnvObj?: Record<string, any>
}

export function CustomApiForm({
    mcpFormData,
    setMcpFormData,
    customApiEnv,
    setCustomApiEnv,
    originalEnvObj = {}
}: CustomApiFormProps) {
    const { t } = useI18n()

    // Determine initial auth state based on existing headers
    const initialAuthInfo = React.useMemo(() => {
        let aType: "none" | "bearer" | "api_key" | "basic" = "none";
        let aHeaderName = "";
        let aSecret = "";
        let bUsername = "";
        let bPassword = "";
        let cHeaders: { key: string, value: string }[] = [];

        if (mcpFormData.headers) {
            let authFound = false;
            if (mcpFormData.headers["Authorization"] === "Bearer $BEARER_TOKEN") {
                aType = "bearer";
                aHeaderName = "Authorization";
                authFound = true;
                const tokenEnv = customApiEnv.find(e => e.key === "BEARER_TOKEN");
                if (tokenEnv) aSecret = tokenEnv.value;
            } else if (mcpFormData.headers["Authorization"] === "Basic $BASIC_AUTH") {
                aType = "basic";
                aHeaderName = "Authorization";
                authFound = true;
                const authEnv = customApiEnv.find(e => e.key === "BASIC_AUTH");
                if (authEnv) aSecret = authEnv.value;
            } else {
                for (const [hName, hVal] of Object.entries(mcpFormData.headers)) {
                    if (hVal === "$API_KEY") {
                        aType = "api_key";
                        aHeaderName = hName;
                        authFound = true;
                        const keyEnv = customApiEnv.find(e => e.key === "API_KEY");
                        if (keyEnv) aSecret = keyEnv.value;
                        break;
                    }
                }
            }

            for (const [k, v] of Object.entries(mcpFormData.headers)) {
                if (authFound && k === aHeaderName) continue;
                cHeaders.push({ key: k, value: String(v) });
            }
        }
        return { aType, aHeaderName, aSecret, bUsername, bPassword, cHeaders };
    }, []); // Only run once on mount

    const [authType, setAuthType] = useState<"none" | "bearer" | "api_key" | "basic">(initialAuthInfo.aType)
    const [authHeaderName, setAuthHeaderName] = useState(initialAuthInfo.aHeaderName)
    const [authSecret, setAuthSecret] = useState(initialAuthInfo.aSecret)
    const [basicUsername, setBasicUsername] = useState(initialAuthInfo.bUsername)
    const [basicPassword, setBasicPassword] = useState(initialAuthInfo.bPassword)

    const [isAdvancedOpen, setIsAdvancedOpen] = useState(false)
    const [customHeaders, setCustomHeaders] = useState<{ key: string, value: string }[]>(initialAuthInfo.cHeaders)

    useEffect(() => {
        if (!mcpFormData.method) {
            setMcpFormData((prev: MCPServerFormData) => ({ ...prev, method: "GET" }))
        }
    }, [])

    // Sync auth state to headers and env
    useEffect(() => {
        let newHeaders: Record<string, string> = {}
        let newEnv: { key: string, value: string }[] = []

        if (authType === "bearer") {
            newHeaders = { "Authorization": "Bearer $BEARER_TOKEN" }
            newEnv = [{ key: "BEARER_TOKEN", value: authSecret }]
        } else if (authType === "api_key") {
            if (authHeaderName) {
                newHeaders = { [authHeaderName]: "$API_KEY" }
            }
            newEnv = [{ key: "API_KEY", value: authSecret }]
        } else if (authType === "basic") {
            newHeaders = { "Authorization": "Basic $BASIC_AUTH" }
            // For basic auth, we combine username:password and base64 encode it
            // However, since it might be masked as ********, we just store it directly
            const combined = authSecret === "********" ? authSecret : btoa(`${basicUsername}:${basicPassword}`)
            newEnv = [{ key: "BASIC_AUTH", value: combined }]
        }

        // Add custom headers
        customHeaders.forEach(h => {
            if (h.key.trim()) {
                newHeaders[h.key.trim()] = h.value.trim()
            }
        })

        setMcpFormData((prev: MCPServerFormData) => ({ ...prev, headers: newHeaders }))
        // Only set env if auth type is not none
        if (authType !== "none") {
            setCustomApiEnv(newEnv)
        } else {
            // Keep any existing custom env vars if auth is none, or just clear them
            setCustomApiEnv([])
        }
    }, [authType, authHeaderName, authSecret, basicUsername, basicPassword, customHeaders, setMcpFormData, setCustomApiEnv])

    const methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]

    return (
        <div className="space-y-4">
            <div className="space-y-2">
                <Label htmlFor="api_name">{t('tools.mcp.dialog.customApiName')} <span className="text-red-500">*</span></Label>
                <Input
                    id="api_name"
                    value={mcpFormData.name || ""}
                    onChange={(e) => setMcpFormData((prev: MCPServerFormData) => ({ ...prev, name: e.target.value }))}
                />
            </div>

            <div className="space-y-2">
                <Label htmlFor="api_url">{t('tools.mcp.dialog.endpointUrl')}</Label>
                <Input
                    id="api_url"
                    value={mcpFormData.url || ""}
                    onChange={(e) => setMcpFormData((prev: MCPServerFormData) => ({ ...prev, url: e.target.value }))}
                />
            </div>

            <div className="space-y-2">
                <Label>{t('tools.mcp.dialog.method')}</Label>
                <div className="flex bg-slate-100 p-1 rounded-md">
                    {methods.map(method => (
                        <button
                            key={method}
                            type="button"
                            className={`flex-1 py-1.5 text-sm font-medium rounded-md transition-colors ${(mcpFormData.method || "GET") === method
                                ? "bg-blue-600 text-white shadow"
                                : "text-slate-600 hover:text-slate-900 hover:bg-slate-200"
                                }`}
                            onClick={() => setMcpFormData((prev: MCPServerFormData) => ({ ...prev, method }))}
                        >
                            {method}
                        </button>
                    ))}
                </div>
            </div>

            <div className="space-y-2">
                <Label>{t('tools.mcp.dialog.authentication')}</Label>
                <Select
                    value={authType}
                    onValueChange={(val: "none" | "bearer" | "api_key" | "basic") => {
                        setAuthType(val)
                        setAuthSecret("") // Reset secret when changing auth type
                    }}
                >
                    <SelectTrigger>
                        <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                        <SelectItem value="none">{t('tools.mcp.dialog.authTypes.none')}</SelectItem>
                        <SelectItem value="bearer">{t('tools.mcp.dialog.authTypes.bearer')}</SelectItem>
                        <SelectItem value="api_key">{t('tools.mcp.dialog.authTypes.apiKey')}</SelectItem>
                        <SelectItem value="basic">{t('tools.mcp.dialog.authTypes.basic')}</SelectItem>
                    </SelectContent>
                </Select>
            </div>

            {authType === "api_key" && (
                <>
                    <div className="space-y-2">
                        <Label htmlFor="header_name">{t('tools.mcp.dialog.headerName')}</Label>
                        <Input
                            id="header_name"
                            value={authHeaderName}
                            onChange={(e) => setAuthHeaderName(e.target.value)}
                            placeholder={t('tools.mcp.dialog.headerNamePlaceholder')}
                        />
                    </div>
                    <div className="space-y-2">
                        <Label htmlFor="api_key_val">{t('tools.mcp.dialog.apiKey')}</Label>
                        <Input
                            id="api_key_val"
                            type="password"
                            value={authSecret}
                            onChange={(e) => setAuthSecret(e.target.value)}
                            placeholder={t('tools.mcp.dialog.apiKeyPlaceholder')}
                            onFocus={() => {
                                if (authSecret === "********") {
                                    setAuthSecret("")
                                }
                            }}
                            onBlur={() => {
                                if (authSecret === "" && originalEnvObj["API_KEY"]) {
                                    setAuthSecret("********")
                                }
                            }}
                        />
                    </div>
                </>
            )}

            {authType === "bearer" && (
                <div className="space-y-2">
                    <Label htmlFor="bearer_token">{t('tools.mcp.dialog.token')}</Label>
                    <Input
                        id="bearer_token"
                        type="password"
                        value={authSecret}
                        onChange={(e) => setAuthSecret(e.target.value)}
                        placeholder={t('tools.mcp.dialog.tokenPlaceholder')}
                        onFocus={() => {
                            if (authSecret === "********") {
                                setAuthSecret("")
                            }
                        }}
                        onBlur={() => {
                            if (authSecret === "" && originalEnvObj["BEARER_TOKEN"]) {
                                setAuthSecret("********")
                            }
                        }}
                    />
                </div>
            )}

            {authType === "basic" && (
                <>
                    <div className="space-y-2">
                        <Label htmlFor="basic_user">{t('tools.mcp.dialog.username')}</Label>
                        <Input
                            id="basic_user"
                            value={basicUsername}
                            onChange={(e) => {
                                setBasicUsername(e.target.value)
                                if (authSecret === "********") setAuthSecret("") // Force recompute
                            }}
                            placeholder={t('tools.mcp.dialog.usernamePlaceholder')}
                        />
                    </div>
                    <div className="space-y-2">
                        <Label htmlFor="basic_pass">{t('tools.mcp.dialog.password')}</Label>
                        <Input
                            id="basic_pass"
                            type="password"
                            value={basicPassword}
                            onChange={(e) => {
                                setBasicPassword(e.target.value)
                                if (authSecret === "********") setAuthSecret("") // Force recompute
                            }}
                            placeholder={t('tools.mcp.dialog.passwordPlaceholder')}
                        />
                    </div>
                    {authSecret === "********" && (
                        <div className="text-xs text-slate-500">
                            {t('tools.mcp.dialog.basicAuthNote')}
                        </div>
                    )}
                </>
            )}

            <Collapsible open={isAdvancedOpen} onOpenChange={setIsAdvancedOpen} className="w-full space-y-2 pt-4">
                <CollapsibleTrigger asChild>
                    <Button variant="ghost" className="flex w-full items-center justify-between p-4 h-auto font-medium text-slate-700 bg-slate-50 border hover:text-slate-900 hover:bg-slate-100">
                        <div className="flex items-center">
                            <svg className="w-4 h-4 mr-2 text-slate-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 20h9"></path><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"></path></svg>
                            {t('tools.mcp.dialog.advancedOptions')}
                        </div>
                        {isAdvancedOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                    </Button>
                </CollapsibleTrigger>
                <CollapsibleContent className="space-y-4 p-4 border border-t-0 bg-slate-50/50 rounded-b-md -mt-3">
                    <div className="space-y-3">
                        <div>
                            <Label className="text-sm font-semibold">{t('tools.mcp.dialog.customHeaders')}</Label>
                            <p className="text-xs text-slate-500">{t('tools.mcp.dialog.customHeadersDesc')}</p>
                        </div>

                        {customHeaders.length === 0 ? (
                            <p className="text-sm text-slate-500">{t('tools.mcp.dialog.noCustomHeaders')}</p>
                        ) : (
                            <div className="space-y-2">
                                {customHeaders.map((h, i) => (
                                    <div key={i} className="flex gap-2 items-center">
                                        <Input
                                            placeholder={t('tools.mcp.dialog.headerKeyPlaceholder')}
                                            value={h.key}
                                            onChange={(e) => {
                                                const newList = [...customHeaders]
                                                newList[i].key = e.target.value
                                                setCustomHeaders(newList)
                                            }}
                                            className="flex-1"
                                        />
                                        <span className="text-slate-400">:</span>
                                        <Input
                                            placeholder={t('tools.mcp.dialog.headerValuePlaceholder')}
                                            value={h.value}
                                            onChange={(e) => {
                                                const newList = [...customHeaders]
                                                newList[i].value = e.target.value
                                                setCustomHeaders(newList)
                                            }}
                                            className="flex-1"
                                        />
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            onClick={() => {
                                                const newList = [...customHeaders]
                                                newList.splice(i, 1)
                                                setCustomHeaders(newList)
                                            }}
                                            className="text-slate-400 hover:text-red-500"
                                        >
                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
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
                            onClick={() => setCustomHeaders([...customHeaders, { key: "", value: "" }])}
                        >
                            <Plus className="h-4 w-4 mr-2" /> {t('tools.mcp.dialog.addHeader')}
                        </Button>
                    </div>
                </CollapsibleContent>
            </Collapsible>
        </div>
    )
}
