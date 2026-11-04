import React, { useCallback, useEffect, useRef, useState } from "react"
import { AlertTriangle, Plus, Trash2 } from "lucide-react"

import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select-radix"
import { Switch } from "@/components/ui/switch"
import { useI18n } from "@/contexts/i18n-context"
import type { TranslationKey } from "@/i18n/translations"

import type { MCPServerFormData } from "./custom-api-form"

type ConnectorType = "mcp" | "custom_api"
type RuntimeInputType = "context" | "secrets" | "auth_selector"
type RuntimeValueType = "string" | "object"
type RuntimeTargetType =
  | "mcp_meta"
  | "tool_arguments"
  | "transport_headers"
  | "headers"
  | "body_field"

interface RuntimeInputRow {
  inputType: RuntimeInputType
  key: string
  valueType: RuntimeValueType
  required: boolean
}

interface EditableRuntimeInputRow extends RuntimeInputRow {
  id: string
}

export interface RuntimeBindingRow {
  sourceType: RuntimeInputType
  sourceKey: string
  targetType: RuntimeTargetType
  targetKey: string
}

interface EditableRuntimeBindingRow extends RuntimeBindingRow {
  id: string
}

interface RuntimeBindingConfig {
  source: {
    input_type: RuntimeInputType
    key: string
  }
  target:
    | {
        target_type: Exclude<RuntimeTargetType, "body_field">
        key: string
      }
    | {
        target_type: "body_field"
        path: string
      }
}

interface RuntimeInputsFormProps {
  connectorType: ConnectorType
  formData: MCPServerFormData
  setFormData: React.Dispatch<React.SetStateAction<MCPServerFormData>>
  onValidationErrorChange?: (error: RuntimeConfigErrorKey | null) => void
  disabled?: boolean
}

export type RuntimeConfigErrorKey = Extract<
  TranslationKey,
  `tools.mcp.runtime.errors.${string}`
>

const SOURCE_KEY_RE = /[^A-Za-z0-9_-]/g

const inputTypeOptions: Record<ConnectorType, RuntimeInputType[]> = {
  mcp: ["context", "secrets", "auth_selector"],
  custom_api: ["context", "secrets"],
}

const bindingSourceOptions: Record<ConnectorType, RuntimeInputType[]> = {
  mcp: ["context", "secrets"],
  custom_api: ["context", "secrets"],
}

const targetOptions: Record<ConnectorType, RuntimeTargetType[]> = {
  mcp: ["mcp_meta", "tool_arguments", "transport_headers"],
  custom_api: ["headers", "body_field"],
}

function sanitizeRuntimeKey(value: string): string {
  return value.replace(SOURCE_KEY_RE, "")
}

function runtimeInputsFromSchema(schema: unknown): RuntimeInputRow[] {
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) return []
  const result: RuntimeInputRow[] = []
  for (const inputType of ["context", "secrets", "auth_selector"] as const) {
    const section = (schema as Record<string, unknown>)[inputType]
    if (!section || typeof section !== "object" || Array.isArray(section)) continue
    for (const [key, declaration] of Object.entries(section)) {
      const config =
        declaration && typeof declaration === "object" && !Array.isArray(declaration)
          ? (declaration as Record<string, unknown>)
          : {}
      result.push({
        inputType,
        key,
        valueType: config.type === "object" ? "object" : "string",
        required: Boolean(config.required),
      })
    }
  }
  return result
}

function schemaFromInputs(rows: RuntimeInputRow[]): Record<string, unknown> | null {
  const schema: Record<string, Record<string, { type: RuntimeValueType; required: boolean }>> = {}
  for (const row of rows) {
    const key = sanitizeRuntimeKey(row.key.trim())
    if (!key) continue
    schema[row.inputType] = schema[row.inputType] || {}
    schema[row.inputType][key] = {
      type: row.valueType,
      required: row.required,
    }
  }
  return Object.keys(schema).length > 0 ? schema : null
}

export function runtimeBindingsFromConfig(value: unknown): RuntimeBindingRow[] {
  if (!Array.isArray(value)) return []
  return value
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
    .map((item) => {
      const source =
        item.source && typeof item.source === "object"
          ? (item.source as Record<string, unknown>)
          : {}
      const target =
        item.target && typeof item.target === "object"
          ? (item.target as Record<string, unknown>)
          : {}
      return {
        sourceType:
          source.input_type === "secrets" || source.input_type === "auth_selector"
            ? source.input_type
            : "context",
        sourceKey: typeof source.key === "string" ? source.key : "",
        targetType:
          typeof target.target_type === "string"
            ? (target.target_type as RuntimeTargetType)
            : "mcp_meta",
        targetKey:
          typeof target.path === "string"
            ? target.path
            : typeof target.key === "string"
              ? target.key
              : "",
      }
    })
}

function bindingsToConfig(rows: RuntimeBindingRow[]): RuntimeBindingConfig[] | null {
  const bindings = rows
    .reduce<RuntimeBindingConfig[]>((acc, row) => {
      const sourceKey = sanitizeRuntimeKey(row.sourceKey.trim())
      const targetKey =
        row.targetType === "body_field"
          ? row.targetKey
              .split(".")
              .map((part) => sanitizeRuntimeKey(part.trim()))
              .filter(Boolean)
              .join(".")
          : row.targetKey.trim()
      if (!sourceKey && !targetKey) return acc
      acc.push({
        source: { input_type: row.sourceType, key: sourceKey },
        target:
          row.targetType === "body_field"
            ? { target_type: row.targetType, path: targetKey }
            : { target_type: row.targetType, key: targetKey },
      })
      return acc
    }, [])
  return bindings.length > 0 ? bindings : null
}

function withInputIds(
  rows: RuntimeInputRow[],
  nextId: () => string,
): EditableRuntimeInputRow[] {
  return rows.map((row) => ({ ...row, id: nextId() }))
}

function withBindingIds(
  rows: RuntimeBindingRow[],
  nextId: () => string,
): EditableRuntimeBindingRow[] {
  return rows.map((row) => ({ ...row, id: nextId() }))
}

function nextKey(rows: RuntimeInputRow[], base: string): string {
  const used = new Set(rows.map((row) => row.key))
  if (!used.has(base)) return base
  let index = 1
  while (used.has(`${base}_${index}`)) index += 1
  return `${base}_${index}`
}

function defaultTarget(connectorType: ConnectorType): RuntimeTargetType {
  return connectorType === "mcp" ? "mcp_meta" : "headers"
}

function targetNeedsPath(targetType: RuntimeTargetType): boolean {
  return targetType === "body_field"
}

function duplicateRuntimeInputError(
  inputs: RuntimeInputRow[],
): RuntimeConfigErrorKey | null {
  const seen = new Set<string>()
  for (const input of inputs) {
    const key = sanitizeRuntimeKey(input.key.trim())
    if (!key) continue
    const scopedKey = `${input.inputType}:${key}`
    if (seen.has(scopedKey)) return "tools.mcp.runtime.errors.duplicateInput"
    seen.add(scopedKey)
  }
  return null
}

function bindingError(
  binding: RuntimeBindingRow,
  inputs: RuntimeInputRow[],
  connectorType: ConnectorType,
  delegatedEnabled: boolean,
): RuntimeConfigErrorKey | null {
  if (!binding.targetKey.trim()) return "tools.mcp.runtime.errors.targetMissing"
  const source = inputs.find(
    (row) => row.inputType === binding.sourceType && row.key === binding.sourceKey,
  )
  if (!source) return "tools.mcp.runtime.errors.sourceMissing"
  if (!inputTypeOptions[connectorType].includes(source.inputType)) {
    return "tools.mcp.runtime.errors.sourceUnsupported"
  }
  if (!targetOptions[connectorType].includes(binding.targetType)) {
    return "tools.mcp.runtime.errors.targetUnsupported"
  }
  if (source.inputType === "auth_selector") {
    return "tools.mcp.runtime.errors.authSelectorBinding"
  }
  if (
    source.inputType === "secrets" &&
    !["headers", "transport_headers"].includes(binding.targetType)
  ) {
    return "tools.mcp.runtime.errors.secretTarget"
  }
  if (source.inputType === "context") {
    const allowedTargets =
      connectorType === "mcp"
        ? ["mcp_meta", "tool_arguments"]
        : ["headers", "body_field"]
    if (!allowedTargets.includes(binding.targetType)) {
      return "tools.mcp.runtime.errors.contextTarget"
    }
  }
  if (
    source.valueType === "object" &&
    ["headers", "transport_headers"].includes(binding.targetType)
  ) {
    return "tools.mcp.runtime.errors.objectHeader"
  }
  if (
    ["headers", "transport_headers"].includes(binding.targetType) &&
    binding.targetKey.trim().toLowerCase() === "authorization" &&
    !delegatedEnabled
  ) {
    return "tools.mcp.runtime.errors.authorizationRequiresDelegated"
  }
  return null
}

export function getRuntimeConfigError(
  formData: MCPServerFormData,
  connectorType: ConnectorType,
): RuntimeConfigErrorKey | null {
  const inputs = runtimeInputsFromSchema(formData.runtime_input_schema)
  const bindings = runtimeBindingsFromConfig(formData.runtime_bindings)
  const delegatedEnabled = Boolean(formData.allow_delegated_authorization)
  return (
    duplicateRuntimeInputError(inputs) ||
    bindings
      .map((binding) => bindingError(binding, inputs, connectorType, delegatedEnabled))
      .find((error): error is RuntimeConfigErrorKey => Boolean(error)) ||
    null
  )
}

export function RuntimeInputsForm({
  connectorType,
  formData,
  setFormData,
  onValidationErrorChange,
  disabled = false,
}: RuntimeInputsFormProps) {
  const { t } = useI18n()
  const delegatedEnabled = Boolean(formData.allow_delegated_authorization)
  const nextIdRef = useRef(0)
  const nextRowId = useCallback(() => `runtime-row-${nextIdRef.current++}`, [])
  const lastSchemaRef = useRef(formData.runtime_input_schema)
  const lastBindingsRef = useRef(formData.runtime_bindings)
  const [inputs, setInputs] = useState<EditableRuntimeInputRow[]>(() =>
    withInputIds(runtimeInputsFromSchema(formData.runtime_input_schema), nextRowId),
  )
  const [bindings, setBindings] = useState<EditableRuntimeBindingRow[]>(() =>
    withBindingIds(runtimeBindingsFromConfig(formData.runtime_bindings), nextRowId),
  )

  useEffect(() => {
    if (formData.runtime_input_schema === lastSchemaRef.current) return
    lastSchemaRef.current = formData.runtime_input_schema
    setInputs(
      withInputIds(runtimeInputsFromSchema(formData.runtime_input_schema), nextRowId),
    )
  }, [formData.runtime_input_schema, nextRowId])

  useEffect(() => {
    if (formData.runtime_bindings === lastBindingsRef.current) return
    lastBindingsRef.current = formData.runtime_bindings
    setBindings(
      withBindingIds(runtimeBindingsFromConfig(formData.runtime_bindings), nextRowId),
    )
  }, [formData.runtime_bindings, nextRowId])

  const updateInputs = (nextInputs: EditableRuntimeInputRow[]) => {
    const nextSchema = schemaFromInputs(nextInputs)
    lastSchemaRef.current = nextSchema
    setInputs(nextInputs)
    setFormData((prev) => ({
      ...prev,
      runtime_input_schema: nextSchema,
    }))
  }

  const updateBindings = (nextBindings: EditableRuntimeBindingRow[]) => {
    const nextConfig = bindingsToConfig(nextBindings)
    lastBindingsRef.current = nextConfig
    setBindings(nextBindings)
    setFormData((prev) => ({
      ...prev,
      runtime_bindings: nextConfig,
    }))
  }

  const updateDelegated = (checked: boolean) => {
    setFormData((prev) => ({
      ...prev,
      allow_delegated_authorization: checked,
    }))
  }

  const inputKeysByType = new Map<RuntimeInputType, string[]>()
  for (const row of inputs) {
    const existing = inputKeysByType.get(row.inputType) || []
    inputKeysByType.set(row.inputType, [...existing, row.key])
  }

  const hasAuthorizationBinding = bindings.some(
    (binding) =>
      ["headers", "transport_headers"].includes(binding.targetType) &&
      binding.targetKey.trim().toLowerCase() === "authorization",
  )
  const bindingErrors = bindings
    .map((binding) => bindingError(binding, inputs, connectorType, delegatedEnabled))
    .filter((error): error is RuntimeConfigErrorKey => Boolean(error))
  const duplicateInputError = duplicateRuntimeInputError(inputs)
  const localValidationError = duplicateInputError || bindingErrors[0] || null
  const hasToolArgumentBinding = bindings.some(
    (binding) => binding.targetType === "tool_arguments",
  )

  useEffect(() => {
    onValidationErrorChange?.(localValidationError)
  }, [localValidationError, onValidationErrorChange])

  return (
    <div className="space-y-4 rounded-md border bg-slate-50/60 p-4">
      <div className="flex items-center justify-between gap-3">
        <Label className="text-sm font-semibold">
          {t("tools.mcp.runtime.title")}
        </Label>
        <div className="flex items-center gap-2">
          <Label htmlFor="runtime-delegated-auth" className="text-xs text-slate-600">
            {t("tools.mcp.runtime.delegatedAuthorization")}
          </Label>
          <Switch
            id="runtime-delegated-auth"
            checked={delegatedEnabled}
            disabled={disabled}
            onCheckedChange={updateDelegated}
          />
        </div>
      </div>

      {hasAuthorizationBinding && !delegatedEnabled && (
        <Alert className="border-amber-200 bg-amber-50 text-amber-900">
          <AlertTriangle className="h-4 w-4 text-amber-700" />
          <AlertDescription className="text-amber-800">
            {t("tools.mcp.runtime.authorizationRequiresDelegated")}
          </AlertDescription>
        </Alert>
      )}

      {localValidationError && (
        <Alert className="border-red-200 bg-red-50 text-red-900">
          <AlertTriangle className="h-4 w-4 text-red-700" />
          <AlertDescription className="text-red-800">
            {t(localValidationError)}
          </AlertDescription>
        </Alert>
      )}

      {hasToolArgumentBinding && (
        <Alert className="border-blue-200 bg-blue-50 text-blue-900">
          <AlertTriangle className="h-4 w-4 text-blue-700" />
          <AlertDescription className="text-blue-800">
            {t("tools.mcp.runtime.toolArgumentsHidden")}
          </AlertDescription>
        </Alert>
      )}

      <div className="space-y-2">
        <div className="flex items-center justify-between gap-3">
          <Label className="text-xs font-semibold uppercase text-slate-500">
            {t("tools.mcp.runtime.inputs")}
          </Label>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={disabled}
            onClick={() =>
              updateInputs([
                ...inputs,
                {
                  id: nextRowId(),
                  inputType: "context",
                  key: nextKey(inputs, "account_id"),
                  valueType: "string",
                  required: false,
                },
              ])
            }
          >
            <Plus className="h-4 w-4 mr-2" />
            {t("tools.mcp.runtime.addInput")}
          </Button>
        </div>

        {inputs.length === 0 ? (
          <div className="rounded-md border border-dashed bg-white px-3 py-3 text-sm text-slate-500">
            {t("tools.mcp.runtime.noInputs")}
          </div>
        ) : (
          <div className="space-y-2">
            {inputs.map((row, index) => (
              <div key={row.id} className="grid grid-cols-12 gap-2 rounded-md border bg-white p-2">
                <Select
                  value={row.inputType}
                  disabled={disabled}
                  onValueChange={(value) => {
                    const inputType = value as RuntimeInputType
                    const next = [...inputs]
                    next[index] = { ...row, inputType }
                    updateInputs(next)
                  }}
                >
                  <SelectTrigger className="col-span-12 md:col-span-3">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {inputTypeOptions[connectorType].map((option) => (
                      <SelectItem key={option} value={option}>
                        {t(`tools.mcp.runtime.inputTypes.${option}`)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                <Input
                  className="col-span-12 md:col-span-3"
                  value={row.key}
                  disabled={disabled}
                  placeholder={t("tools.mcp.runtime.key")}
                  onChange={(event) => {
                    const next = [...inputs]
                    next[index] = {
                      ...row,
                      key: sanitizeRuntimeKey(event.target.value),
                    }
                    updateInputs(next)
                  }}
                />

                <Select
                  value={row.valueType}
                  disabled={disabled}
                  onValueChange={(value) => {
                    const next = [...inputs]
                    next[index] = { ...row, valueType: value as RuntimeValueType }
                    updateInputs(next)
                  }}
                >
                  <SelectTrigger className="col-span-6 md:col-span-2">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="string">{t("tools.mcp.runtime.valueTypes.string")}</SelectItem>
                    <SelectItem value="object">{t("tools.mcp.runtime.valueTypes.object")}</SelectItem>
                  </SelectContent>
                </Select>

                <label className="col-span-4 md:col-span-2 flex items-center gap-2 text-sm text-slate-600">
                  <input
                    type="checkbox"
                    checked={row.required}
                    disabled={disabled}
                    onChange={(event) => {
                      const next = [...inputs]
                      next[index] = { ...row, required: event.target.checked }
                      updateInputs(next)
                    }}
                  />
                  {t("tools.mcp.runtime.required")}
                </label>

                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  disabled={disabled}
                  className="col-span-2 md:col-span-2 justify-self-end text-slate-400 hover:text-red-600"
                  onClick={() => updateInputs(inputs.filter((_, itemIndex) => itemIndex !== index))}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between gap-3">
          <Label className="text-xs font-semibold uppercase text-slate-500">
            {t("tools.mcp.runtime.bindings")}
          </Label>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={disabled || inputs.length === 0}
            onClick={() => {
              const source = inputs.find((row) => row.inputType !== "auth_selector")
              updateBindings([
                ...bindings,
                {
                  id: nextRowId(),
                  sourceType: source?.inputType || "context",
                  sourceKey: source?.key || "account_id",
                  targetType: defaultTarget(connectorType),
                  targetKey: "account_id",
                },
              ])
            }}
          >
            <Plus className="h-4 w-4 mr-2" />
            {t("tools.mcp.runtime.addBinding")}
          </Button>
        </div>

        {bindings.length === 0 ? (
          <div className="rounded-md border border-dashed bg-white px-3 py-3 text-sm text-slate-500">
            {t("tools.mcp.runtime.noBindings")}
          </div>
        ) : (
          <div className="space-y-2">
            {bindings.map((row, index) => {
              const keys = inputKeysByType.get(row.sourceType) || []
              return (
                <div key={row.id} className="grid grid-cols-12 gap-2 rounded-md border bg-white p-2">
                  <Select
                    value={row.sourceType}
                    disabled={disabled}
                    onValueChange={(value) => {
                      const sourceType = value as RuntimeInputType
                      const next = [...bindings]
                      next[index] = {
                        ...row,
                        sourceType,
                        sourceKey: inputKeysByType.get(sourceType)?.[0] || "",
                      }
                      updateBindings(next)
                    }}
                  >
                    <SelectTrigger className="col-span-12 md:col-span-3">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {bindingSourceOptions[connectorType].map((option) => (
                        <SelectItem key={option} value={option}>
                          {t(`tools.mcp.runtime.inputTypes.${option}`)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>

                  <Select
                    value={row.sourceKey || "__empty__"}
                    disabled={disabled || keys.length === 0}
                    onValueChange={(value) => {
                      const next = [...bindings]
                      next[index] = { ...row, sourceKey: value === "__empty__" ? "" : value }
                      updateBindings(next)
                    }}
                  >
                    <SelectTrigger className="col-span-12 md:col-span-3">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {keys.length === 0 ? (
                        <SelectItem value="__empty__">{t("tools.mcp.runtime.noSourceKeys")}</SelectItem>
                      ) : (
                        keys.map((key) => (
                          <SelectItem key={key} value={key}>
                            {key}
                          </SelectItem>
                        ))
                      )}
                    </SelectContent>
                  </Select>

                  <Select
                    value={row.targetType}
                    disabled={disabled}
                    onValueChange={(value) => {
                      const targetType = value as RuntimeTargetType
                      const next = [...bindings]
                      next[index] = {
                        ...row,
                        targetType,
                        targetKey: targetNeedsPath(targetType) ? "account.id" : row.targetKey,
                      }
                      updateBindings(next)
                    }}
                  >
                    <SelectTrigger className="col-span-12 md:col-span-3">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {targetOptions[connectorType].map((option) => (
                        <SelectItem key={option} value={option}>
                          {t(`tools.mcp.runtime.targetTypes.${option}`)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>

                  <Input
                    className="col-span-10 md:col-span-2"
                    value={row.targetKey}
                    disabled={disabled}
                    placeholder={
                      targetNeedsPath(row.targetType)
                        ? t("tools.mcp.runtime.path")
                        : t("tools.mcp.runtime.key")
                    }
                    onChange={(event) => {
                      const next = [...bindings]
                      next[index] = { ...row, targetKey: event.target.value }
                      updateBindings(next)
                    }}
                  />

                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    disabled={disabled}
                    className="col-span-2 md:col-span-1 justify-self-end text-slate-400 hover:text-red-600"
                    onClick={() => updateBindings(bindings.filter((_, itemIndex) => itemIndex !== index))}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
