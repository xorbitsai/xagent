"use client"

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { AlertTriangle, Check, Copy, KeyRound, Plus, Search, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { toast } from "@/components/ui/sonner"
import { useI18n } from "@/contexts/i18n-context"
import { copyToClipboard } from "@/lib/clipboard"
import {
  PersonalApiKeyCreated,
  PersonalApiKeyListItem,
  createPersonalApiKey,
  listPersonalApiKeys,
  revokePersonalApiKey,
} from "@/lib/personal-api-keys-api"

function statusPillClass(status: PersonalApiKeyListItem["status"]): string {
  switch (status) {
    case "active":
      return "bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-400"
    case "expired":
      return "bg-amber-100 text-amber-700 dark:bg-amber-900/20 dark:text-amber-400"
    default:
      return "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"
  }
}

export interface PersonalApiKeysPanelProps {
  active: boolean
}

export function PersonalApiKeysPanel({ active }: PersonalApiKeysPanelProps) {
  const { t } = useI18n()
  const [keys, setKeys] = useState<PersonalApiKeyListItem[]>([])
  const [canManageOthers, setCanManageOthers] = useState(false)
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [reveal, setReveal] = useState<PersonalApiKeyCreated | null>(null)
  const [copied, setCopied] = useState(false)
  const [confirmKey, setConfirmKey] = useState<PersonalApiKeyListItem | null>(null)
  const [revoking, setRevoking] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const listGeneration = useRef(0)
  const hasActivated = useRef(false)

  const loadKeys = useCallback(async () => {
    const generation = ++listGeneration.current
    setLoading(true)
    try {
      const response = await listPersonalApiKeys()
      if (generation !== listGeneration.current) return
      setKeys(response.items)
      setCanManageOthers(response.can_manage_others)
    } catch (error) {
      if (generation !== listGeneration.current) return
      console.error(error)
      toast.error(t("personalApiKeys.messages.loadFailed") || "Failed to load personal API keys")
    } finally {
      if (generation === listGeneration.current) setLoading(false)
    }
  }, [t])

  useEffect(() => {
    if (!active || hasActivated.current) return
    hasActivated.current = true
    loadKeys()
  }, [active, loadKeys])

  const handleCreate = async () => {
    setCreating(true)
    try {
      const created = await createPersonalApiKey()
      setReveal(created)
      toast.success(t("personalApiKeys.messages.created") || "Personal API key created")
      loadKeys()
    } catch (error) {
      console.error(error)
      toast.error(t("personalApiKeys.messages.createFailed") || "Failed to create personal API key")
    } finally {
      setCreating(false)
    }
  }

  const handleCopyReveal = async () => {
    if (!reveal) return
    if (await copyToClipboard(reveal.full_key)) {
      setCopied(true)
      toast.success(t("personalApiKeys.messages.copied") || "Copied to clipboard")
      setTimeout(() => setCopied(false), 2000)
    } else {
      toast.error(t("personalApiKeys.messages.copyFailed") || "Failed to copy to clipboard")
    }
  }

  const handleRevoke = async () => {
    if (!confirmKey) return
    setRevoking(true)
    try {
      await revokePersonalApiKey(confirmKey.id)
      setConfirmKey(null)
      toast.success(t("personalApiKeys.messages.revoked") || "Personal API key revoked")
      loadKeys()
    } catch (error) {
      console.error(error)
      toast.error(t("personalApiKeys.messages.revokeFailed") || "Failed to revoke personal API key")
    } finally {
      setRevoking(false)
    }
  }

  const normalizedQuery = searchQuery.trim().toLowerCase()
  const filteredKeys = useMemo(
    () =>
      keys.filter(
        (k) =>
          !normalizedQuery ||
          [k.masked_key, k.key_prefix, k.owner.username, k.owner.email ?? ""]
            .join(" ")
            .toLowerCase()
            .includes(normalizedQuery)
      ),
    [keys, normalizedQuery]
  )

  const activeCount = useMemo(() => keys.filter((k) => k.status === "active").length, [keys])

  const formatDate = (value: string) => new Date(value).toLocaleDateString()
  const revokeDescription = confirmKey
    ? canManageOthers
      ? t("personalApiKeys.confirm.revokeOtherDescription", { owner: confirmKey.owner.username }) ||
        `Revoke this personal key for ${confirmKey.owner.username}?`
      : t("personalApiKeys.confirm.revokeOwnDescription") || "Revoking immediately invalidates this key."
    : ""

  return (
    <div className="px-6 md:px-8 pb-8 mt-6">
      <div className="grid gap-4 grid-cols-2 lg:grid-cols-4 mb-6">
        <Card>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              {t("personalApiKeys.stats.totalKeys") || "Total Keys"}
            </p>
            <p className="text-2xl font-bold mt-1">{loading ? "—" : keys.length}</p>
            <p className="text-xs text-muted-foreground mt-1">
              {t("personalApiKeys.stats.totalKeysHint") || "personal keys"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              {t("personalApiKeys.stats.activeKeys") || "Active Keys"}
            </p>
            <p className="text-2xl font-bold mt-1">{loading ? "—" : activeCount}</p>
            <p className="text-xs text-muted-foreground mt-1">
              {t("personalApiKeys.stats.activeKeysHint") || "accepting requests"}
            </p>
          </CardContent>
        </Card>
      </div>

      <Card className="shadow-sm">
        <CardHeader className="pb-3 border-b flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 space-y-0">
          <div>
            <h2 className="text-lg font-semibold">{t("personalApiKeys.title") || "Personal Keys"}</h2>
            <p className="text-sm text-muted-foreground mt-1">
              {t("personalApiKeys.description") || "Manage your personal SDK and REST API keys."}
            </p>
          </div>
          <div className="flex items-center gap-2 w-full sm:w-auto">
            <div className="relative w-full sm:w-64">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder={t("personalApiKeys.searchPlaceholder") || "Search keys or owners..."}
                className="pl-9 h-9"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <Button onClick={handleCreate} disabled={creating} className="shrink-0">
              <Plus className="w-4 h-4 mr-1" />
              {canManageOthers
                ? t("personalApiKeys.createForMe") || "Create Personal Key for Me"
                : t("personalApiKeys.create") || "Create Personal Key"}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="text-xs font-semibold text-muted-foreground">
                  {t("personalApiKeys.columns.key") || "Secret Key"}
                </TableHead>
                {canManageOthers && (
                  <TableHead className="text-xs font-semibold text-muted-foreground">
                    {t("personalApiKeys.columns.owner") || "Owner"}
                  </TableHead>
                )}
                <TableHead className="text-xs font-semibold text-muted-foreground">
                  {t("personalApiKeys.columns.status") || "Status"}
                </TableHead>
                <TableHead className="text-xs font-semibold text-muted-foreground">
                  {t("personalApiKeys.columns.expires") || "Expiry"}
                </TableHead>
                <TableHead className="text-xs font-semibold text-muted-foreground">
                  {t("personalApiKeys.columns.created") || "Created"}
                </TableHead>
                <TableHead className="w-[100px]" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredKeys.map((key) => (
                <TableRow key={key.id} className={key.status === "active" ? "" : "opacity-50"}>
                  <TableCell>
                    <span className="inline-flex items-center gap-1.5 font-mono text-xs text-muted-foreground">
                      <KeyRound className="w-3.5 h-3.5 shrink-0" />
                      {key.masked_key}
                    </span>
                  </TableCell>
                  {canManageOthers && <TableCell className="text-sm">{key.owner.username}</TableCell>}
                  <TableCell>
                    <span
                      className={`inline-flex text-[11px] px-2 py-0.5 rounded-full capitalize font-medium ${statusPillClass(key.status)}`}
                    >
                      {t(`personalApiKeys.status.${key.status}`) || key.status}
                    </span>
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {key.expires_at
                      ? formatDate(key.expires_at)
                      : t("personalApiKeys.neverExpires") || "Never"}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">{formatDate(key.created_at)}</TableCell>
                  <TableCell className="text-right">
                    {key.status === "active" && <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-destructive hover:text-destructive"
                      onClick={() => setConfirmKey(key)}
                      title={t("personalApiKeys.actions.revoke") || "Revoke"}
                      aria-label={t("personalApiKeys.actions.revoke") || "Revoke"}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>}
                  </TableCell>
                </TableRow>
              ))}
              {filteredKeys.length === 0 && !loading && (
                <TableRow>
                  <TableCell colSpan={canManageOthers ? 6 : 5} className="text-center text-muted-foreground h-32">
                    {keys.length === 0
                      ? t("personalApiKeys.noData") || "No personal API keys yet."
                      : t("personalApiKeys.noResults") || "No keys match your search."}
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Dialog open={reveal !== null} onOpenChange={(open) => !open && setReveal(null)}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <KeyRound className="h-5 w-5" />
              {t("personalApiKeys.reveal.title") || "Personal API Key Created"}
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-2 rounded-md border border-amber-300 bg-amber-50 dark:bg-amber-950/30 p-3">
            <DialogDescription className="flex items-center gap-2 text-sm font-medium text-amber-700 dark:text-amber-400">
              <AlertTriangle className="h-4 w-4" />
              {t("personalApiKeys.reveal.warning") || "Copy this key now — it is shown only once."}
            </DialogDescription>
            <div className="flex items-center gap-2">
              <code className="flex-1 break-all rounded bg-muted px-2 py-1.5 text-xs font-mono">{reveal?.full_key}</code>
              <Button
                size="icon"
                variant="secondary"
                onClick={handleCopyReveal}
                title={t("personalApiKeys.actions.copy") || "Copy personal API key"}
                aria-label={t("personalApiKeys.actions.copy") || "Copy personal API key"}
              >
                {copied ? <Check className="h-4 w-4 text-green-500" /> : <Copy className="h-4 w-4" />}
              </Button>
            </div>
          </div>
          <DialogFooter>
            <Button onClick={() => setReveal(null)}>{t("common.done") || "Done"}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        isOpen={confirmKey !== null}
        onOpenChange={(open) => !open && setConfirmKey(null)}
        onConfirm={handleRevoke}
        isLoading={revoking}
        title={t("personalApiKeys.confirm.revokeTitle") || "Revoke personal API key?"}
        description={revokeDescription}
        confirmText={t("personalApiKeys.actions.revoke") || "Revoke"}
      />
    </div>
  )
}
