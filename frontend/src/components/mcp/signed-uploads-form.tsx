import React, { useCallback, useEffect, useRef, useState } from "react"
import { AlertTriangle, Plus, Trash2 } from "lucide-react"

import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Textarea } from "@/components/ui/textarea"
import { useI18n } from "@/contexts/i18n-context"

import type { MCPServerFormData } from "./custom-api-form"

interface SignedUploadContentTypeRow {
  id: string
  extension: string
  contentType: string
}

/**
 * How the host uses one entry. "replacement" uploads the file before a call to
 * the tool that takes it and replaces that tool's file argument; "issuer" runs
 * the issuing tool for the model and returns only the connector's file ID;
 * "link" replaces a workspace image in a URL argument with a short-lived
 * download link. "unrecognized" is an entry that carries a key the form does
 * not know: it is shown read-only and written back exactly as it was stored.
 */
export type SignedUploadMode = "replacement" | "issuer" | "link" | "unrecognized"

// An entry is read-only only when it carries a key the form does not know.
// Exactly the issuer keys is an issuer entry and exactly the link keys is a
// link entry; any other entry whose keys are all replacement keys (a complete
// one, or one with keys missing) is a replacement entry.
const ISSUER_ENTRY_KEYS = [
  "issuing_tool_name",
  "allowed_upload_hosts",
  "upload_path_prefix",
  "content_types",
] as const
const REPLACEMENT_ENTRY_KEYS: ReadonlySet<string> = new Set([
  "tool_name",
  "file_argument",
  ...ISSUER_ENTRY_KEYS,
])
const LINK_ENTRY_KEYS = ["tool_name", "link_argument"] as const

export interface SignedUploadRow {
  id: string
  mode: SignedUploadMode
  // toolName is empty in an issuer row. fileArgument is empty in an issuer and
  // a link row; linkArgument is empty in every row but a link row; the issuer
  // and file fields are empty in a link row.
  toolName: string
  fileArgument: string
  linkArgument: string
  // The stored entry of an unrecognized row, which is the only row that has one.
  raw?: unknown
  issuingToolName: string
  hostsText: string // one hostname per line, exactly as typed
  uploadPathPrefix: string
  contentTypes: SignedUploadContentTypeRow[]
}

export interface SignedUploadsFormProps {
  formData: MCPServerFormData
  setFormData: React.Dispatch<React.SetStateAction<MCPServerFormData>>
  disabled?: boolean
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function textOrEmpty(value: unknown): string {
  return typeof value === "string" ? value : ""
}

// The title of an unrecognized entry: its tool_name, else its
// issuing_tool_name, when that is a string that is not blank. Empty when it
// has neither, and the row then shows a generic title.
function unrecognizedTitle(raw: unknown): string {
  if (!isRecord(raw)) return ""
  for (const key of ["tool_name", "issuing_tool_name"]) {
    const value = raw[key]
    if (typeof value === "string" && value.trim() !== "") return value.trim()
  }
  return ""
}

function hasExactKeys(item: Record<string, unknown>, expected: readonly string[]): boolean {
  const keys = Object.keys(item)
  return keys.length === expected.length && expected.every((key) => keys.includes(key))
}

function entryMode(entry: unknown): SignedUploadMode {
  // An entry that is not an object reads as an empty replacement row, which
  // is dropped on save.
  if (!isRecord(entry)) return "replacement"
  if (hasExactKeys(entry, ISSUER_ENTRY_KEYS)) return "issuer"
  if (hasExactKeys(entry, LINK_ENTRY_KEYS)) return "link"
  const keys = Object.keys(entry)
  return keys.every((key) => REPLACEMENT_ENTRY_KEYS.has(key))
    ? "replacement"
    : "unrecognized"
}

/**
 * Turn a stored signed_uploads value into editable rows, showing as much of a
 * stored entry as possible: a missing or mistyped field reads as empty and an
 * entry that is not an object reads as an empty row. An entry with exactly the
 * issuer keys reads as an issuer row, exactly the link keys as a link row, and
 * any other entry whose keys are all replacement keys as a replacement row. An
 * entry is read-only only when it has a key the form does not know: it reads
 * as an unrecognized row that keeps a copy of the stored value.
 */
export function signedUploadRowsFromValue(
  value: unknown,
  nextId: () => string,
): SignedUploadRow[] {
  if (!Array.isArray(value)) return []
  return value.map((entry) => {
    const mode = entryMode(entry)
    if (mode === "unrecognized") {
      return {
        id: nextId(),
        mode,
        toolName: "",
        fileArgument: "",
        linkArgument: "",
        issuingToolName: "",
        hostsText: "",
        uploadPathPrefix: "",
        contentTypes: [],
        raw: structuredClone(entry),
      }
    }
    const item = isRecord(entry) ? entry : {}
    const hosts = Array.isArray(item.allowed_upload_hosts)
      ? item.allowed_upload_hosts.filter((host): host is string => typeof host === "string")
      : []
    const contentTypes = isRecord(item.content_types)
      ? Object.entries(item.content_types)
        .filter((pair): pair is [string, string] => typeof pair[1] === "string")
        .map(([extension, contentType]) => ({ id: nextId(), extension, contentType }))
      : []
    return {
      id: nextId(),
      mode,
      toolName: textOrEmpty(item.tool_name),
      fileArgument: textOrEmpty(item.file_argument),
      linkArgument: textOrEmpty(item.link_argument),
      issuingToolName: textOrEmpty(item.issuing_tool_name),
      hostsText: hosts.join("\n"),
      uploadPathPrefix: textOrEmpty(item.upload_path_prefix),
      contentTypes,
    }
  })
}

/**
 * Turn editable rows into the signed_uploads request value. Names and the path
 * prefix are trimmed, host lines are trimmed and blank ones dropped, file type
 * rows with both cells blank are dropped, and a row with nothing left is
 * dropped. An issuer row writes only the issuer keys, a replacement row writes
 * all six keys and a link row writes its two keys. An unrecognized row writes
 * its stored entry unchanged. Every other rule is left to the server.
 */
export function signedUploadsFromRows(rows: SignedUploadRow[]): Record<string, unknown>[] {
  const entries: Record<string, unknown>[] = []
  for (const row of rows) {
    if (row.mode === "unrecognized") {
      entries.push(row.raw as Record<string, unknown>)
      continue
    }
    const toolName = row.toolName.trim()
    if (row.mode === "link") {
      const linkArgument = row.linkArgument.trim()
      if (toolName || linkArgument) {
        entries.push({ tool_name: toolName, link_argument: linkArgument })
      }
      continue
    }
    const fileArgument = row.fileArgument.trim()
    const issuingToolName = row.issuingToolName.trim()
    const uploadPathPrefix = row.uploadPathPrefix.trim()
    const hosts = row.hostsText
      .split("\n")
      .map((host) => host.trim())
      .filter((host) => host !== "")
    const contentTypePairs = row.contentTypes
      .map((item): [string, string] => [item.extension.trim(), item.contentType.trim()])
      .filter(([extension, contentType]) => extension !== "" || contentType !== "")
    if (
      !toolName
      && !fileArgument
      && !issuingToolName
      && !uploadPathPrefix
      && hosts.length === 0
      && contentTypePairs.length === 0
    ) {
      continue
    }
    if (row.mode === "issuer") {
      entries.push({
        issuing_tool_name: issuingToolName,
        allowed_upload_hosts: hosts,
        upload_path_prefix: uploadPathPrefix,
        content_types: Object.fromEntries(contentTypePairs),
      })
      continue
    }
    entries.push({
      tool_name: toolName,
      file_argument: fileArgument,
      issuing_tool_name: issuingToolName,
      allowed_upload_hosts: hosts,
      upload_path_prefix: uploadPathPrefix,
      content_types: Object.fromEntries(contentTypePairs),
    })
  }
  return entries
}

interface RowFieldsProps {
  row: SignedUploadRow
  fieldId: (field: string) => string
  update: (patch: Partial<SignedUploadRow>) => void
}

// The inputs of a link row: the tool that takes the link and the argument
// that receives it.
function LinkFields({ row, fieldId, update }: RowFieldsProps) {
  const { t } = useI18n()
  return (
    <>
      <div className="space-y-2">
        <Label htmlFor={fieldId("toolName")}>
          {t("tools.mcp.signedUploads.linkToolName")}
        </Label>
        <Input
          id={fieldId("toolName")}
          value={row.toolName}
          onChange={(event) => update({ toolName: event.target.value })}
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor={fieldId("linkArgument")}>
          {t("tools.mcp.signedUploads.linkArgument")}
        </Label>
        <Input
          id={fieldId("linkArgument")}
          value={row.linkArgument}
          onChange={(event) =>
            update({ linkArgument: event.target.value })}
        />
      </div>
      <p className="text-xs text-slate-500">
        {t("tools.mcp.signedUploads.linkNamesNotCheckedHint")}
      </p>
    </>
  )
}

// The inputs of a replacement or an issuer row. Only a replacement row has
// the tool that takes the file and its file argument; both have the issuing
// tool and the upload rules.
function UploadFields({
  row,
  fieldId,
  update,
  nextRowId,
}: RowFieldsProps & { nextRowId: () => string }) {
  const { t } = useI18n()
  return (
    <>
      {row.mode === "replacement" && (
        <>
          <div className="space-y-2">
            <Label htmlFor={fieldId("toolName")}>
              {t("tools.mcp.signedUploads.toolName")}
            </Label>
            <Input
              id={fieldId("toolName")}
              value={row.toolName}
              onChange={(event) => update({ toolName: event.target.value })}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor={fieldId("fileArgument")}>
              {t("tools.mcp.signedUploads.fileArgument")}
            </Label>
            <Input
              id={fieldId("fileArgument")}
              value={row.fileArgument}
              onChange={(event) =>
                update({ fileArgument: event.target.value })}
            />
          </div>
        </>
      )}
      <div className="space-y-2">
        <Label htmlFor={fieldId("issuingToolName")}>
          {t("tools.mcp.signedUploads.issuingToolName")}
        </Label>
        <Input
          id={fieldId("issuingToolName")}
          value={row.issuingToolName}
          onChange={(event) =>
            update({ issuingToolName: event.target.value })}
        />
      </div>
      <p className="text-xs text-slate-500">
        {t("tools.mcp.signedUploads.toolNamesHint")}
      </p>
      <p className="text-xs text-slate-500">
        {t("tools.mcp.signedUploads.namesNotCheckedHint")}
      </p>

      <div className="space-y-2">
        <Label htmlFor={fieldId("allowedUploadHosts")}>
          {t("tools.mcp.signedUploads.allowedUploadHosts")}
        </Label>
        <Textarea
          id={fieldId("allowedUploadHosts")}
          value={row.hostsText}
          placeholder="uploads.example.test"
          onChange={(event) => update({ hostsText: event.target.value })}
        />
        <p className="text-xs text-slate-500">
          {t("tools.mcp.signedUploads.allowedUploadHostsHint")}
        </p>
      </div>

      <div className="space-y-2">
        <Label htmlFor={fieldId("uploadPathPrefix")}>
          {t("tools.mcp.signedUploads.uploadPathPrefix")}
        </Label>
        <Input
          id={fieldId("uploadPathPrefix")}
          value={row.uploadPathPrefix}
          placeholder="/uploads/"
          onChange={(event) =>
            update({ uploadPathPrefix: event.target.value })}
        />
      </div>

      <div className="space-y-2">
        <Label>{t("tools.mcp.signedUploads.contentTypes")}</Label>
        {row.contentTypes.map((item) => (
          <div key={item.id} className="flex items-center gap-2">
            <Input
              aria-label={t("tools.mcp.signedUploads.extension")}
              value={item.extension}
              placeholder=".pdf"
              className="flex-1"
              onChange={(event) =>
                update({
                  contentTypes: row.contentTypes.map((other) =>
                    other.id === item.id
                      ? { ...other, extension: event.target.value }
                      : other),
                })}
            />
            <Input
              aria-label={t("tools.mcp.signedUploads.contentType")}
              value={item.contentType}
              placeholder="application/pdf"
              className="flex-1"
              onChange={(event) =>
                update({
                  contentTypes: row.contentTypes.map((other) =>
                    other.id === item.id
                      ? { ...other, contentType: event.target.value }
                      : other),
                })}
            />
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="text-red-500 hover:text-red-700"
              aria-label={t("tools.mcp.signedUploads.removeContentType")}
              title={t("tools.mcp.signedUploads.removeContentType")}
              onClick={() =>
                update({
                  contentTypes: row.contentTypes.filter((other) => other.id !== item.id),
                })}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        ))}
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() =>
            update({
              contentTypes: [
                ...row.contentTypes,
                { id: nextRowId(), extension: "", contentType: "" },
              ],
            })}
        >
          <Plus className="h-4 w-4 mr-2" />
          {t("tools.mcp.signedUploads.addContentType")}
        </Button>
      </div>
    </>
  )
}

/**
 * The "File uploads" section of the custom MCP form. It reads and writes
 * formData.signed_uploads and shows formData.signed_uploads_error. Mounting
 * and rendering never write to the form: only an edit does, so a stored value
 * that cannot be used stays exactly as it was until the owner changes or
 * clears it.
 */
export function SignedUploadsForm({
  formData,
  setFormData,
  disabled = false,
}: SignedUploadsFormProps) {
  const { t } = useI18n()
  const nextIdRef = useRef(0)
  const nextRowId = useCallback(() => `signed-upload-row-${nextIdRef.current++}`, [])
  const lastValueRef = useRef(formData.signed_uploads)
  const [rows, setRows] = useState<SignedUploadRow[]>(() =>
    signedUploadRowsFromValue(formData.signed_uploads, nextRowId),
  )

  useEffect(() => {
    if (formData.signed_uploads === lastValueRef.current) return
    lastValueRef.current = formData.signed_uploads
    setRows(signedUploadRowsFromValue(formData.signed_uploads, nextRowId))
  }, [formData.signed_uploads, nextRowId])

  const updateRows = (nextRows: SignedUploadRow[]) => {
    const next = signedUploadsFromRows(nextRows)
    setRows(nextRows)
    lastValueRef.current = next
    setFormData((prev) => ({ ...prev, signed_uploads: next }))
  }

  const updateRow = (id: string, patch: Partial<SignedUploadRow>) => {
    updateRows(rows.map((row) => (row.id === id ? { ...row, ...patch } : row)))
  }

  // Switching mode drops the fields the new mode does not have. The tool name
  // is kept between replacement and link rows, which both have one.
  const updateMode = (row: SignedUploadRow, mode: SignedUploadMode) => {
    if (mode === row.mode) return
    if (mode === "link") {
      updateRow(row.id, {
        mode,
        fileArgument: "",
        issuingToolName: "",
        hostsText: "",
        uploadPathPrefix: "",
        contentTypes: [],
      })
    } else if (mode === "issuer") {
      updateRow(row.id, { mode, toolName: "", fileArgument: "", linkArgument: "" })
    } else {
      updateRow(row.id, { mode, linkArgument: "" })
    }
  }

  const clearSettings = () => {
    const next: Record<string, unknown>[] = []
    setRows([])
    lastValueRef.current = next
    setFormData((prev) => ({ ...prev, signed_uploads: next }))
  }

  const settingsError = formData.signed_uploads_error
  const hasSettingsError = typeof settingsError === "string"
  const isEmptyList = Array.isArray(formData.signed_uploads) && formData.signed_uploads.length === 0

  return (
    <fieldset disabled={disabled} className="contents">
      <div className="space-y-4 rounded-md border bg-slate-50/60 p-4">
        <div className="space-y-1">
          <Label className="text-sm font-semibold">{t("tools.mcp.signedUploads.title")}</Label>
          <p className="text-xs text-slate-500">{t("tools.mcp.signedUploads.description")}</p>
        </div>

        {hasSettingsError && !isEmptyList && (
          <Alert className="border-amber-200 bg-amber-50 text-amber-900">
            <AlertTriangle className="h-4 w-4 text-amber-700" />
            <AlertDescription className="flex items-center justify-between gap-3 text-amber-800">
              <span>{t("tools.mcp.signedUploads.invalid", { error: settingsError })}</span>
              <Button type="button" variant="outline" size="sm" onClick={clearSettings}>
                {t("common.clear")}
              </Button>
            </AlertDescription>
          </Alert>
        )}

        {hasSettingsError && isEmptyList && (
          <p className="text-sm text-slate-600">{t("tools.mcp.signedUploads.willBeCleared")}</p>
        )}

        {rows.length === 0 ? (
          <div className="rounded-md border border-dashed bg-white px-3 py-3 text-sm text-slate-500">
            {t("tools.mcp.signedUploads.empty")}
          </div>
        ) : (
          <div className="space-y-3">
            {rows.map((row) => {
              const fieldId = (field: string) => `signed-upload-${row.id}-${field}`
              const update = (patch: Partial<SignedUploadRow>) => updateRow(row.id, patch)
              return (
                <div key={row.id} className="space-y-3 rounded-md border bg-white p-3">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-medium text-slate-700">
                      {row.mode === "unrecognized"
                        ? unrecognizedTitle(row.raw)
                          || t("tools.mcp.signedUploads.unrecognizedEntryTitle")
                        : (row.mode === "issuer" ? row.issuingToolName : row.toolName).trim()
                          || t("tools.mcp.signedUploads.newTool")}
                    </span>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="text-red-500 hover:text-red-700"
                      aria-label={t("tools.mcp.signedUploads.removeTool")}
                      title={t("tools.mcp.signedUploads.removeTool")}
                      onClick={() => updateRows(rows.filter((item) => item.id !== row.id))}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>

                  {row.mode === "unrecognized" ? (
                    <div className="space-y-2">
                      <p className="text-xs text-slate-500">
                        {t("tools.mcp.signedUploads.unrecognizedEntry")}
                      </p>
                      <Textarea
                        readOnly
                        aria-label={t("tools.mcp.signedUploads.unrecognizedEntryValue")}
                        className="font-mono text-xs"
                        value={JSON.stringify(row.raw, null, 2)}
                      />
                    </div>
                  ) : (
                    <>
                      <div className="space-y-2">
                        <Label id={fieldId("mode")}>{t("tools.mcp.signedUploads.mode")}</Label>
                        <RadioGroup
                          aria-labelledby={fieldId("mode")}
                          value={row.mode}
                          disabled={disabled}
                          onValueChange={(value) =>
                            updateMode(
                              row,
                              value === "issuer" || value === "link" ? value : "replacement",
                            )}
                        >
                          <div className="flex items-center space-x-2">
                            <RadioGroupItem value="replacement" id={fieldId("modeReplacement")} />
                            <Label
                              htmlFor={fieldId("modeReplacement")}
                              className="font-normal cursor-pointer"
                            >
                              {t("tools.mcp.signedUploads.modeReplacement")}
                            </Label>
                          </div>
                          <div className="flex items-center space-x-2">
                            <RadioGroupItem value="issuer" id={fieldId("modeIssuer")} />
                            <Label htmlFor={fieldId("modeIssuer")} className="font-normal cursor-pointer">
                              {t("tools.mcp.signedUploads.modeIssuer")}
                            </Label>
                          </div>
                          <div className="flex items-center space-x-2">
                            <RadioGroupItem value="link" id={fieldId("modeLink")} />
                            <Label htmlFor={fieldId("modeLink")} className="font-normal cursor-pointer">
                              {t("tools.mcp.signedUploads.modeLink")}
                            </Label>
                          </div>
                        </RadioGroup>
                        <p className="text-xs text-slate-500">
                          {row.mode === "issuer"
                            ? t("tools.mcp.signedUploads.modeIssuerHint")
                            : row.mode === "link"
                              ? t("tools.mcp.signedUploads.modeLinkHint")
                              : t("tools.mcp.signedUploads.modeReplacementHint")}
                        </p>
                      </div>
                      {row.mode === "link" ? (
                        <LinkFields row={row} fieldId={fieldId} update={update} />
                      ) : (
                        <UploadFields
                          row={row}
                          fieldId={fieldId}
                          update={update}
                          nextRowId={nextRowId}
                        />
                      )}
                    </>
                  )}
                </div>
              )
            })}
          </div>
        )}

        <Button
          type="button"
          variant="outline"
          size="sm"
          className="w-full border-dashed text-blue-600 border-blue-200 hover:bg-blue-50 hover:text-blue-700"
          onClick={() =>
            updateRows([
              ...rows,
              {
                id: nextRowId(),
                mode: "replacement",
                toolName: "",
                fileArgument: "",
                linkArgument: "",
                issuingToolName: "",
                hostsText: "",
                uploadPathPrefix: "",
                contentTypes: [{ id: nextRowId(), extension: "", contentType: "" }],
              },
            ])}
        >
          <Plus className="h-4 w-4 mr-2" />
          {t("tools.mcp.signedUploads.addTool")}
        </Button>
      </div>
    </fieldset>
  )
}
