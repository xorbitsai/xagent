"use client"

import React, { useCallback, useEffect, useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import {
  AlertTriangle,
  Check,
  Copy,
  KeyRound,
  Pause,
  Play,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  X,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { PageHeader } from "@/components/ui/page-header"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select-radix"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"
import { toast } from "@/components/ui/sonner"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { copyToClipboard } from "@/lib/clipboard"
import {
  AgentApiKeyCreated,
  AgentApiKeyListItem,
  AgentApiKeyStats,
  createAgentApiKey,
  deleteAgentApiKey,
  getAgentApiKeyStats,
  listAgentApiKeys,
  pauseAgentApiKey,
  regenerateAgentApiKey,
  resumeAgentApiKey,
} from "@/lib/agent-api-keys-api"
import { PersonalApiKeysPanel } from "./personal-api-keys-panel"

interface AgentOption {
  id: number
  name: string
}

type ConfirmAction =
  | { type: "regenerate"; key: AgentApiKeyListItem }
  | { type: "delete"; key: AgentApiKeyListItem }

function statusPillClass(status: AgentApiKeyListItem["status"]): string {
  switch (status) {
    case "active":
      return "bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-400"
    case "paused":
      return "bg-amber-100 text-amber-700 dark:bg-amber-900/20 dark:text-amber-400"
    default:
      return "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"
  }
}

export function ApiKeysPage() {
  const { t } = useI18n()
  const router = useRouter()
  const searchParams = useSearchParams()

  const [keys, setKeys] = useState<AgentApiKeyListItem[]>([])
  const [stats, setStats] = useState<AgentApiKeyStats | null>(null)
  const [agents, setAgents] = useState<AgentOption[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState("")
  // Exact agent_id filter from the ?agent=<id> jump-link, kept separate
  // from the free-text searchQuery -- matching by agent *name* substring
  // would wrongly mix in another agent whose name happens to contain
  // this one's (e.g. "Bot" vs. "Support Bot").
  const [agentFilterId, setAgentFilterId] = useState<number | null>(null)
  const [busyKeyId, setBusyKeyId] = useState<number | null>(null)

  const [isCreateOpen, setIsCreateOpen] = useState(false)
  const [createAgentId, setCreateAgentId] = useState<string>("")
  const [createLabel, setCreateLabel] = useState("")
  const [creating, setCreating] = useState(false)

  const [reveal, setReveal] = useState<AgentApiKeyCreated | null>(null)
  const [copied, setCopied] = useState(false)

  const [confirmAction, setConfirmAction] = useState<ConfirmAction | null>(null)
  const [confirmBusy, setConfirmBusy] = useState(false)

  const activeTab = searchParams.get("agent")
    ? "agent"
    : searchParams.get("tab") === "personal"
      ? "personal"
      : "agent"

  const handleTabChange = (tab: string) => {
    const params = new URLSearchParams(searchParams.toString())
    if (tab === "personal") {
      setAgentFilterId(null)
      params.delete("agent")
      params.set("tab", "personal")
    } else {
      params.delete("tab")
    }
    const query = params.toString()
    router.replace(query ? `/api-keys?${query}` : "/api-keys")
  }

  const fetchAll = useCallback(async () => {
    setLoading(true)
    try {
      const [keyList, statsResult] = await Promise.all([
        listAgentApiKeys(),
        getAgentApiKeyStats(),
      ])
      setKeys(keyList)
      setStats(statsResult)
    } catch (err) {
      console.error(err)
      toast.error(t("apiKeysPage.messages.loadFailed") || "Failed to load API keys")
    } finally {
      setLoading(false)
    }
  }, [t])

  const fetchAgents = useCallback(async () => {
    try {
      const res = await apiRequest(`${getApiUrl()}/api/agents`)
      if (res.ok) {
        const data = await res.json()
        setAgents(data.map((a: { id: number; name: string }) => ({ id: a.id, name: a.name })))
      }
    } catch (err) {
      console.error(err)
    }
  }, [])

  useEffect(() => {
    if (activeTab !== "agent") return
    fetchAll()
    fetchAgents()
  }, [activeTab, fetchAll, fetchAgents])

  // Jump-link from an agent card / deploy dialog: pre-filter to that agent
  // by id (exact), independent of whether the agents list has loaded yet.
  useEffect(() => {
    const agentId = searchParams.get("agent")
    const parsed = agentId ? Number(agentId) : NaN
    if (!Number.isNaN(parsed)) setAgentFilterId(parsed)
    else setAgentFilterId(null)
  }, [searchParams])

  const agentFilterName = agentFilterId
    ? agents.find((a) => a.id === agentFilterId)?.name
    : undefined

  // Also strips ?agent=<id> from the URL -- otherwise it lingers and
  // openCreateDialog (which reads searchParams directly) would keep
  // preselecting the agent the user just cleared the filter for.
  const clearAgentFilter = () => {
    setAgentFilterId(null)
    router.replace("/api-keys")
  }

  // A key binds to an agent or a workforce; surface whichever owner name
  // is set (the other is null on the wire).
  const ownerName = (k: AgentApiKeyListItem): string =>
    (k.owner_type === "workforce" ? k.workforce_name : k.agent_name) ?? ""

  const normalizedQuery = searchQuery.trim().toLowerCase()
  const filteredKeys = keys
    .filter((k) => agentFilterId === null || k.agent_id === agentFilterId)
    .filter(
      (k) =>
        !normalizedQuery ||
        [k.label ?? "", ownerName(k), k.key_prefix]
          .join(" ")
          .toLowerCase()
          .includes(normalizedQuery)
    )

  const openCreateDialog = () => {
    const preselected = searchParams.get("agent")
    setCreateAgentId(preselected && agents.some((a) => String(a.id) === preselected) ? preselected : "")
    setCreateLabel("")
    setIsCreateOpen(true)
  }

  const handleCreate = async () => {
    if (!createAgentId) return
    setCreating(true)
    try {
      const result = await createAgentApiKey(Number(createAgentId), createLabel.trim() || null)
      setIsCreateOpen(false)
      setReveal(result)
      toast.success(t("apiKeysPage.messages.created") || "API key created")
      fetchAll()
    } catch (err) {
      console.error(err)
      toast.error(t("apiKeysPage.messages.createFailed") || "Failed to create API key")
    } finally {
      setCreating(false)
    }
  }

  const handleTogglePause = async (key: AgentApiKeyListItem) => {
    setBusyKeyId(key.id)
    try {
      if (key.status === "paused") {
        await resumeAgentApiKey(key.id)
        toast.success(t("apiKeysPage.messages.resumed") || "API key resumed")
      } else {
        await pauseAgentApiKey(key.id)
        toast.success(t("apiKeysPage.messages.paused") || "API key paused")
      }
      fetchAll()
    } catch (err) {
      console.error(err)
      toast.error(t("apiKeysPage.messages.actionFailed") || "Action failed")
    } finally {
      setBusyKeyId(null)
    }
  }

  const handleConfirm = async () => {
    if (!confirmAction) return
    setConfirmBusy(true)
    try {
      if (confirmAction.type === "regenerate") {
        const result = await regenerateAgentApiKey(confirmAction.key.id)
        setReveal(result)
        toast.success(t("apiKeysPage.messages.regenerated") || "API key regenerated")
      } else {
        await deleteAgentApiKey(confirmAction.key.id)
        toast.success(t("apiKeysPage.messages.deleted") || "API key deleted")
      }
      setConfirmAction(null)
      fetchAll()
    } catch (err) {
      console.error(err)
      toast.error(t("apiKeysPage.messages.actionFailed") || "Action failed")
    } finally {
      setConfirmBusy(false)
    }
  }

  const handleCopyReveal = async () => {
    if (!reveal) return
    if (await copyToClipboard(reveal.full_key)) {
      setCopied(true)
      toast.success(t("apiKeysPage.messages.copied") || "Copied to clipboard")
      setTimeout(() => setCopied(false), 2000)
    } else {
      toast.error(t("apiKeysPage.messages.copyFailed") || "Failed to copy to clipboard")
    }
  }

  const formatDate = (value: string | null) =>
    value ? new Date(value).toLocaleString() : "—"

  return (
    <div className="flex h-full flex-col bg-background overflow-auto">
      <PageHeader
        title={t("apiKeysPage.title") || "API Keys"}
        description={t("apiKeysPage.subtitle") || "Manage API keys for programmatic access to your agents."}
      />

      <Tabs value={activeTab} onValueChange={handleTabChange} className="mt-6">
        <TabsList className="!flex bg-transparent h-auto p-0 gap-6 justify-start border-b w-full rounded-none px-6 md:px-8">
          <TabsTrigger
            value="agent"
            className="!flex-none !w-auto rounded-none border-0 border-b-2 border-transparent px-1 py-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground shadow-none hover:text-foreground data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:text-primary data-[state=active]:shadow-none"
          >
            {t("apiKeysPage.tabs.agent") || "Agent Keys"}
          </TabsTrigger>
          <TabsTrigger
            value="personal"
            className="!flex-none !w-auto rounded-none border-0 border-b-2 border-transparent px-1 py-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground shadow-none hover:text-foreground data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:text-primary data-[state=active]:shadow-none"
          >
            {t("apiKeysPage.tabs.personal") || "Personal Keys"}
          </TabsTrigger>
        </TabsList>
        <TabsContent value="agent" className="mt-0 flex-col">

      <div className="grid gap-4 grid-cols-2 lg:grid-cols-4 px-6 md:px-8 mt-6">
        <Card>
          <CardContent>
            <p className="text-xs text-muted-foreground">{t("apiKeysPage.stats.totalKeys") || "Total Keys"}</p>
            <p className="text-2xl font-bold mt-1">{stats?.total_keys ?? "—"}</p>
            <p className="text-xs text-muted-foreground mt-1">
              {t("apiKeysPage.stats.totalKeysHint") || "across all agents"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <p className="text-xs text-muted-foreground">{t("apiKeysPage.stats.activeKeys") || "Active Keys"}</p>
            <p className="text-2xl font-bold mt-1">{stats?.active_keys ?? "—"}</p>
            <p className="text-xs text-muted-foreground mt-1">
              {t("apiKeysPage.stats.activeKeysHint") || "accepting requests"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <p className="text-xs text-muted-foreground">{t("apiKeysPage.stats.callsThisMonth") || "Calls This Month"}</p>
            <p className="text-2xl font-bold mt-1">{stats?.calls_this_month ?? "—"}</p>
            <p className="text-xs text-muted-foreground mt-1">
              {t("apiKeysPage.stats.callsThisMonthHint") || "includes now-inactive keys"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <p className="text-xs text-muted-foreground">{t("apiKeysPage.stats.lastApiCall") || "Last API Call"}</p>
            <p className="text-2xl font-bold mt-1">
              {stats?.last_api_call ? new Date(stats.last_api_call).toLocaleDateString() : "—"}
            </p>
            {!stats?.last_api_call && (
              <p className="text-xs text-muted-foreground mt-1">
                {t("apiKeysPage.stats.noActivity") || "No activity yet"}
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="px-6 md:px-8 pb-8 mt-6">
        <Card className="shadow-sm">
          <CardHeader className="pb-3 border-b flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 space-y-0">
            <h2 className="text-lg font-semibold">{t("apiKeysPage.allKeys") || "All Keys"}</h2>
            <div className="flex items-center gap-2 w-full sm:w-auto">
              {agentFilterId !== null && (
                <span className="inline-flex items-center gap-1.5 text-xs bg-indigo-50 text-indigo-700 dark:bg-indigo-900/20 dark:text-indigo-300 rounded-full px-2.5 py-1 shrink-0">
                  {agentFilterName || t("apiKeysPage.filteredByAgent") || "Filtered agent"}
                  <button
                    type="button"
                    onClick={clearAgentFilter}
                    title={t("apiKeysPage.clearFilter") || "Clear filter"}
                  >
                    <X className="w-3 h-3" />
                  </button>
                </span>
              )}
              <div className="relative w-full sm:w-64">
                <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder={t("apiKeysPage.searchPlaceholder") || "Search keys or agents..."}
                  className="pl-9 h-9"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
              <Button onClick={openCreateDialog} className="shrink-0">
                <Plus className="w-4 h-4 mr-1" />
                {t("apiKeysPage.newKey") || "New API Key"}
              </Button>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead className="text-xs font-semibold text-muted-foreground">
                    {t("apiKeysPage.columns.label") || "Key Label"}
                  </TableHead>
                  <TableHead className="text-xs font-semibold text-muted-foreground">
                    {t("apiKeysPage.columns.agent") || "Agent"}
                  </TableHead>
                  <TableHead className="text-xs font-semibold text-muted-foreground">
                    {t("apiKeysPage.columns.secretKey") || "Secret Key"}
                  </TableHead>
                  <TableHead className="text-xs font-semibold text-muted-foreground">
                    {t("apiKeysPage.columns.status") || "Status"}
                  </TableHead>
                  <TableHead className="text-xs font-semibold text-muted-foreground">
                    {t("apiKeysPage.columns.lastUsed") || "Last Used"}
                  </TableHead>
                  <TableHead className="text-xs font-semibold text-muted-foreground">
                    {t("apiKeysPage.columns.created") || "Created"}
                  </TableHead>
                  <TableHead className="w-[140px]" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredKeys.map((key) => (
                  <TableRow key={key.id} className={key.status === "revoked" ? "opacity-50" : ""}>
                    <TableCell className="font-medium text-sm">
                      {key.label || <span className="text-muted-foreground">—</span>}
                    </TableCell>
                    <TableCell>
                      <span className="inline-flex items-center gap-1.5 text-sm">
                        <span
                          className={`h-2 w-2 rounded-full ${
                            key.owner_type === "workforce"
                              ? "bg-emerald-500"
                              : "bg-indigo-500"
                          }`}
                        />
                        {ownerName(key)}
                      </span>
                    </TableCell>
                    <TableCell>
                      <span className="font-mono text-xs text-muted-foreground">
                        {key.masked_key}
                      </span>
                    </TableCell>
                    <TableCell>
                      <span
                        className={`inline-flex text-[11px] px-2 py-0.5 rounded-full capitalize font-medium ${statusPillClass(key.status)}`}
                      >
                        {t(`apiKeysPage.status.${key.status}`) || key.status}
                      </span>
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {key.last_used_at ? formatDate(key.last_used_at) : t("apiKeysPage.never") || "Never"}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {new Date(key.created_at).toLocaleDateString()}
                    </TableCell>
                    <TableCell className="text-right">
                      {key.status !== "revoked" && (
                        <div className="flex justify-end gap-1">
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8"
                            disabled={busyKeyId === key.id}
                            onClick={() => handleTogglePause(key)}
                            title={
                              key.status === "paused"
                                ? t("apiKeysPage.actions.resume") || "Resume"
                                : t("apiKeysPage.actions.pause") || "Pause"
                            }
                          >
                            {key.status === "paused" ? (
                              <Play className="w-4 h-4" />
                            ) : (
                              <Pause className="w-4 h-4" />
                            )}
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => setConfirmAction({ type: "regenerate", key })}
                            title={t("apiKeysPage.actions.regenerate") || "Regenerate"}
                          >
                            <RefreshCw className="w-4 h-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-destructive hover:text-destructive"
                            onClick={() => setConfirmAction({ type: "delete", key })}
                            title={t("apiKeysPage.actions.delete") || "Delete"}
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
                {filteredKeys.length === 0 && !loading && (
                  <TableRow>
                    <TableCell colSpan={7} className="text-center text-muted-foreground h-32">
                      {t("apiKeysPage.noData") || "No API keys yet."}
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>

      {/* Create dialog */}
      <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
        <DialogContent className="sm:max-w-[440px]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <KeyRound className="h-5 w-5" />
              {t("apiKeysPage.newKey") || "New API Key"}
            </DialogTitle>
            <DialogDescription>
              {t("apiKeysPage.newKeyDescription") || "Create a new SDK / REST API credential for an agent."}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <Label>{t("apiKeysPage.form.agent") || "Agent"}</Label>
              <Select value={createAgentId} onValueChange={setCreateAgentId}>
                <SelectTrigger>
                  <SelectValue placeholder={t("apiKeysPage.form.selectAgent") || "Select an agent"} />
                </SelectTrigger>
                <SelectContent>
                  {agents.map((agent) => (
                    <SelectItem key={agent.id} value={String(agent.id)}>
                      {agent.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>{t("apiKeysPage.form.label") || "Label"}</Label>
              <Input
                placeholder={t("apiKeysPage.form.labelPlaceholder") || "e.g. Production"}
                value={createLabel}
                onChange={(e) => setCreateLabel(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsCreateOpen(false)} disabled={creating}>
              {t("common.cancel") || "Cancel"}
            </Button>
            <Button onClick={handleCreate} disabled={!createAgentId || creating}>
              {t("apiKeysPage.form.create") || "Create"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* One-time plaintext reveal */}
      <Dialog open={reveal !== null} onOpenChange={(open) => !open && setReveal(null)}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <KeyRound className="h-5 w-5" />
              {t("apiKeysPage.reveal.title") || "API Key Created"}
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-2 rounded-md border border-amber-300 bg-amber-50 dark:bg-amber-950/30 p-3">
            <div className="flex items-center gap-2 text-sm font-medium text-amber-700 dark:text-amber-400">
              <AlertTriangle className="h-4 w-4" />
              {t("apiKeysPage.reveal.warning") || "Copy this key now — it is shown only once."}
            </div>
            <div className="flex items-center gap-2">
              <code className="flex-1 break-all rounded bg-muted px-2 py-1.5 text-xs font-mono">
                {reveal?.full_key}
              </code>
              <Button size="icon" variant="secondary" onClick={handleCopyReveal}>
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
        isOpen={confirmAction !== null}
        onOpenChange={(open) => !open && setConfirmAction(null)}
        onConfirm={handleConfirm}
        isLoading={confirmBusy}
        title={
          confirmAction?.type === "regenerate"
            ? t("apiKeysPage.confirm.regenerateTitle") || "Regenerate API key?"
            : t("apiKeysPage.confirm.deleteTitle") || "Delete API key?"
        }
        description={
          confirmAction?.type === "regenerate"
            ? t("apiKeysPage.confirm.regenerateDescription") ||
              "Regenerating immediately invalidates the current secret. Any app using it will stop working until updated."
            : t("apiKeysPage.confirm.deleteDescription") ||
              "Deleting immediately invalidates this key. Any app using it will stop working."
        }
        confirmText={
          confirmAction?.type === "regenerate"
            ? t("apiKeysPage.actions.regenerate") || "Regenerate"
            : t("apiKeysPage.actions.delete") || "Delete"
        }
      />
        </TabsContent>
        <TabsContent value="personal" className="mt-0 flex-col data-[state=inactive]:hidden" forceMount>
          <PersonalApiKeysPanel active={activeTab === "personal"} />
        </TabsContent>
      </Tabs>
    </div>
  )
}
