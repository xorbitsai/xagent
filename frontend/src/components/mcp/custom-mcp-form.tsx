import React from "react"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Select } from "@/components/ui/select"
import { useI18n } from "@/contexts/i18n-context"

interface CustomMcpFormProps {
  mcpFormData: any
  setMcpFormData: React.Dispatch<React.SetStateAction<any>>
  transports: any[]
}

export function CustomMcpForm({
  mcpFormData,
  setMcpFormData,
  transports
}: CustomMcpFormProps) {
  const { t } = useI18n()

  return (
    <>
      <div className="space-y-2">
        <Label htmlFor="name">{t('tools.mcp.form.nameLabel')}</Label>
        <Input
          id="name"
          value={mcpFormData.name}
          onChange={(e) => setMcpFormData((prev: any) => ({ ...prev, name: e.target.value }))}
          placeholder={t('tools.mcp.form.namePlaceholder')}
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="transport">{t('tools.mcp.form.transportLabel')}</Label>
        <Select
          value={mcpFormData.transport}
          onValueChange={(value: string) => setMcpFormData((prev: any) => ({ ...prev, transport: value }))}
          options={transports}
          placeholder={t('tools.mcp.form.transportPlaceholder')}
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="description">{t('tools.mcp.form.descriptionLabel')}</Label>
        <Textarea
          id="description"
          value={mcpFormData.description}
          onChange={(e) => setMcpFormData((prev: any) => ({ ...prev, description: e.target.value }))}
          placeholder={t('tools.mcp.form.descriptionPlaceholder')}
          rows={3}
        />
      </div>
      {(() => {
        const selectedTransport = transports.find(t => t.value === mcpFormData.transport);
        return selectedTransport?.fields?.map((field: any) => (
          <div key={field.name} className="space-y-2">
            <Label htmlFor={field.name}>{field.label} {field.required && "*"}</Label>
            {field.type === 'textarea' ? (
              <Textarea
                id={field.name}
                value={mcpFormData.config[field.name] || ''}
                onChange={(e) => setMcpFormData((prev: any) => ({
                  ...prev,
                  config: { ...prev.config, [field.name]: e.target.value }
                }))}
                placeholder={field.placeholder}
                rows={3}
              />
            ) : field.type === 'select' ? (
              <Select
                value={mcpFormData.config[field.name] || ''}
                onValueChange={(value: string) => setMcpFormData((prev: any) => ({
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
                onChange={(e) => setMcpFormData((prev: any) => ({
                  ...prev,
                  config: { ...prev.config, [field.name]: field.type === 'number' ? Number(e.target.value) : e.target.value }
                }))}
                placeholder={field.placeholder}
              />
            )}
          </div>
        ));
      })()}
    </>
  )
}
