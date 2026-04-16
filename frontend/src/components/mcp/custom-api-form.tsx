import React from "react"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Trash2, Plus } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"

interface CustomApiFormProps {
    mcpFormData: any
    setMcpFormData: React.Dispatch<React.SetStateAction<any>>
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

    return (
        <>
            <div className="space-y-2">
                <Label htmlFor="api_name">{t('tools.mcp.dialog.customApiName')} <span className="text-red-500">*</span></Label>
                <Input
                    id="api_name"
                    value={mcpFormData.name}
                    onChange={(e) => setMcpFormData((prev: any) => ({ ...prev, name: e.target.value }))}
                    placeholder={t('tools.mcp.dialog.customApiNamePlaceholder')}
                />
            </div>

            <div className="space-y-2">
                <Label htmlFor="api_description">{t('tools.mcp.dialog.customApiNote')}</Label>
                <Textarea
                    id="api_description"
                    value={mcpFormData.description}
                    onChange={(e) => setMcpFormData((prev: any) => ({ ...prev, description: e.target.value }))}
                    placeholder={t('tools.mcp.dialog.customApiNotePlaceholder')}
                    rows={4}
                />
            </div>

            <div className="space-y-2 pt-2">
                <Label>{t('tools.mcp.dialog.customApiSecrets')} <span className="text-red-500">*</span></Label>
                <div className="space-y-3">
                    {customApiEnv.map((env, index) => (
                        <div key={index} className="flex items-start gap-2 bg-white p-3 border rounded-md shadow-sm">
                            <div className="flex-1 space-y-2">
                                <div>
                                    <Label className="text-xs text-slate-500 mb-1 block">{t('tools.mcp.dialog.customApiSecretName')}</Label>
                                    <Input
                                        value={env.key}
                                        onChange={(e) => {
                                            const newEnv = [...customApiEnv];
                                            newEnv[index].key = e.target.value;
                                            setCustomApiEnv(newEnv);
                                        }}
                                        placeholder="SOME_UNIQUE_KEY_NAME"
                                    />
                                </div>
                                <div>
                                    <Label className="text-xs text-slate-500 mb-1 block">{t('tools.mcp.dialog.customApiSecretValue')}</Label>
                                    <Textarea
                                        value={env.value}
                                        onChange={(e) => {
                                            const newEnv = [...customApiEnv];
                                            newEnv[index].value = e.target.value;
                                            setCustomApiEnv(newEnv);
                                        }}
                                        onFocus={() => {
                                            if (env.value === "********") {
                                                const newEnv = [...customApiEnv];
                                                newEnv[index].value = "";
                                                setCustomApiEnv(newEnv);
                                            }
                                        }}
                                        onBlur={() => {
                                            if (env.value === "") {
                                                // If the key exists in the original data and had a value, restore the mask
                                                if (originalEnvObj[env.key]) {
                                                    const newEnv = [...customApiEnv];
                                                    newEnv[index].value = "********";
                                                    setCustomApiEnv(newEnv);
                                                }
                                            }
                                        }}
                                        placeholder={t('tools.mcp.dialog.customApiSecretValuePlaceholder')}
                                        rows={2}
                                        className="font-mono text-sm"
                                    />
                                </div>
                            </div>
                            <Button
                                variant="ghost"
                                size="icon"
                                className="mt-6 text-red-500 hover:text-red-700 hover:bg-red-50"
                                disabled={customApiEnv.length <= 1}
                                onClick={() => {
                                    if (customApiEnv.length > 1) {
                                        const newEnv = [...customApiEnv];
                                        newEnv.splice(index, 1);
                                        setCustomApiEnv(newEnv);
                                    }
                                }}
                            >
                                <Trash2 className="h-4 w-4" />
                            </Button>
                        </div>
                    ))}

                    <Button
                        variant="outline"
                        className="w-full mt-2 border-dashed"
                        onClick={() => {
                            setCustomApiEnv([...customApiEnv, { key: "", value: "" }]);
                        }}
                    >
                        <Plus className="h-4 w-4 mr-2" /> {t('tools.mcp.dialog.customApiAddSecret')}
                    </Button>
                </div>
            </div>
        </>
    )
}
