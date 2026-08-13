"use client"

import Link from "next/link"
import React, { useCallback, useEffect, useState } from "react"
import { ChevronLeft, ChevronRight, Play, Plus, Users, Zap, GitBranch, ShieldCheck, Pencil, Rocket, Trash2, ArchiveRestore, Archive, Globe, MoreVertical } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { SearchInput } from "@/components/ui/search-input"
import { PageHeader } from "@/components/ui/page-header"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"
import { useI18n } from "@/contexts/i18n-context"
import { useRouter } from "next/navigation"
import {
  archiveWorkforce,
  deleteWorkforcePermanently,
  listWorkforces,
  publishWorkforce,
  unarchiveWorkforce,
  unpublishWorkforce,
} from "@/lib/workforces-api"
import { formatTime } from "@/lib/time-utils"
import type { WorkforceListItem } from "@/types/workforce"
import { getDeployDisabledReason, getRunDisabledReason } from "./workforce-ui-state"
import { FeatureEmptyState } from "@/components/ui/feature-empty-state"
import { toast } from "sonner"
import { WorkforceCreateView } from "@/components/workforce/workforce-create-view"
import {
  WorkforceStatusBadge,
  WorkforceDeployHubDialog,
  DeployWorkforceDialog,
  WorkforceShareDialog,
  WorkforceWidgetDialog,
} from "@/components/workforce"
import { AgentTriggersDialog } from "@/components/build/agent-triggers-dialog"

type DeployView = "options" | "embed" | "api" | "share"

export default function WorkforcesPage() {
  const { t } = useI18n()
  const router = useRouter()
  const [items, setItems] = useState<WorkforceListItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState("")
  const [page, setPage] = useState(1)
  const [pages, setPages] = useState(1)
  const [total, setTotal] = useState(0)
  const pageSize = 10
  const [view, setView] = useState<"list" | "create">("list")
  const hasActiveSearch = search.trim().length > 0

  const [deployItem, setDeployItem] = useState<WorkforceListItem | null>(null)
  const [deployView, setDeployView] = useState<DeployView | null>(null)
  const [triggersItem, setTriggersItem] = useState<WorkforceListItem | null>(null)
  const [deleteItem, setDeleteItem] = useState<WorkforceListItem | null>(null)
  const [deleting, setDeleting] = useState(false)
  // Single slot, not one flag per action: publish/unpublish/archive/
  // unarchive/delete on the same card must not overlap -- with separate
  // flags, publish being in flight left archive/unarchive/delete on that
  // same card still clickable, so the outcome depended on whichever
  // request the backend happened to finish first.
  const [busyItemId, setBusyItemId] = useState<number | null>(null)
  // The three-dot menu is uncontrolled-by-default in Radix (stays open
  // after an item click), but the Publish/Unpublish row is conditionally
  // rendered on item.status -- so once an action's load() resolves and the
  // list re-renders with a new status, the menu item list at the same
  // on-screen position can be a different action than what was there when
  // the menu opened. Tracked as a single id (not per-item local state)
  // since only one card's menu is ever meaningfully open at a time; closed
  // eagerly on every action click below rather than waiting for load() to
  // settle.
  const [openMenuId, setOpenMenuId] = useState<number | null>(null)

  const closeDeploy = () => {
    setDeployItem(null)
    setDeployView(null)
  }

  const load = useCallback(async (nextPage: number, nextSearch: string) => {
    try {
      setLoading(true)
      setError(null)
      const data = await listWorkforces({ page: nextPage, size: pageSize, search: nextSearch })
      setItems(data.items)
      setPages(data.pages)
      setTotal(data.total)
    } catch (err) {
      const nextError = err instanceof Error ? err.message : t("workforces.errors.loadList")
      setError(nextError)
      toast.error(nextError)
      setItems([])
    } finally {
      setLoading(false)
    }
  }, [t])

  useEffect(() => {
    void load(page, search)
  }, [load, page, search])

  // Shared shape for every immediate (non-confirm-dialog) menu action:
  // close the menu, claim the busy slot, call the API, toast, reload, clear
  // the slot. Delete doesn't fit -- it opens a confirm dialog instead of
  // calling the API directly -- so handleDelete below stays separate but
  // still claims/releases the same busyItemId slot.
  const runMenuAction = (
    item: WorkforceListItem,
    apiCall: (id: number | string) => Promise<unknown>,
    successKey: "workforces.messages.published" | "workforces.messages.unpublished" | "workforces.messages.archived" | "workforces.messages.unarchived",
    errorKey: "workforces.errors.publish" | "workforces.errors.unpublish" | "workforces.errors.archive" | "workforces.errors.unarchive",
  ) => async () => {
    setOpenMenuId(null)
    try {
      setBusyItemId(item.id)
      await apiCall(item.id)
      toast.success(t(successKey))
      void load(page, search)
    } catch (err) {
      const nextError = err instanceof Error ? err.message : t(errorKey)
      toast.error(nextError)
    } finally {
      setBusyItemId(null)
    }
  }

  const handlePublish = (item: WorkforceListItem) =>
    runMenuAction(item, publishWorkforce, "workforces.messages.published", "workforces.errors.publish")()
  const handleUnpublish = (item: WorkforceListItem) =>
    runMenuAction(item, unpublishWorkforce, "workforces.messages.unpublished", "workforces.errors.unpublish")()
  const handleArchive = (item: WorkforceListItem) =>
    runMenuAction(item, archiveWorkforce, "workforces.messages.archived", "workforces.errors.archive")()
  const handleUnarchive = (item: WorkforceListItem) =>
    runMenuAction(item, unarchiveWorkforce, "workforces.messages.unarchived", "workforces.errors.unarchive")()

  const handleDelete = async () => {
    if (!deleteItem) return
    try {
      setDeleting(true)
      setBusyItemId(deleteItem.id)
      await deleteWorkforcePermanently(deleteItem.id)
      toast.success(t("workforces.messages.deleted"))
      setDeleteItem(null)
      if (items.length === 1 && page > 1) {
        // Deleting the last card of the last page would otherwise reload an
        // out-of-range page: the backend returns zero items for it, which
        // renders as the "no workforces" empty state with the pagination
        // controls hidden (pages shrank to exclude the stale page value).
        // Stepping back re-triggers the load effect with a valid page.
        setPage(page - 1)
      } else {
        void load(page, search)
      }
    } catch (err) {
      const nextError = err instanceof Error ? err.message : t("workforces.errors.delete")
      toast.error(nextError)
    } finally {
      setDeleting(false)
      setBusyItemId(null)
    }
  }

  if (view === "create") {
    return (
      <div className="h-full overflow-y-auto">
        <WorkforceCreateView
          onBack={() => setView("list")}
          onCreated={(workforce) => {
            router.push(`/workforces/${workforce.id}?view=canvas`)
          }}
        />
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto">
      {/* Header — build-page style */}
      <PageHeader
        title={t("workforces.list.title")}
        description={t("workforces.list.description")}
        actions={
          <>
            <SearchInput
              placeholder={t("workforces.list.searchPlaceholder")}
              value={search}
              onChange={(value) => {
                setSearch(value)
                setPage(1)
              }}
              containerClassName="flex-1 sm:w-64"
            />
            <Button onClick={() => setView("create")} className="shrink-0 rounded-lg">
              <Plus className="mr-2 h-4 w-4" />
              {t("workforces.actions.new")}
            </Button>
          </>
        }
      />

      <div className="mx-auto flex w-full flex-col gap-6 px-6 py-6 md:px-8">

        {loading ? <div className="p-8 text-muted-foreground">{t("workforces.loading.list")}</div> : null}
        {error ? <div className="p-8 text-red-500">{error}</div> : null}

        {!loading && !error ? (
          items.length === 0 ? (
            hasActiveSearch ? (
              <Card className="border-dashed">
                <CardContent className="flex flex-col items-center gap-4 p-12 text-center">
                  <div className="text-lg font-medium">{t("workforces.list.noResultsTitle")}</div>
                  <p className="max-w-xl text-sm text-muted-foreground">
                    {t("workforces.list.noResults")}
                  </p>
                </CardContent>
              </Card>
            ) : (
              <FeatureEmptyState
                icon={Users}
                title={t("workforces.emptyState.title")}
                description={t("workforces.emptyState.description")}
                features={[
                  {
                    icon: GitBranch,
                    title: t("workforces.emptyState.features.managerAgent.title"),
                    description: t("workforces.emptyState.features.managerAgent.description")
                  },
                  {
                    icon: Zap,
                    title: t("workforces.emptyState.features.subAgents.title"),
                    description: t("workforces.emptyState.features.subAgents.description")
                  },
                  {
                    icon: Play,
                    title: t("workforces.emptyState.features.parallelExecution.title"),
                    description: t("workforces.emptyState.features.parallelExecution.description")
                  },
                  {
                    icon: ShieldCheck,
                    title: t("workforces.emptyState.features.approvalGates.title"),
                    description: t("workforces.emptyState.features.approvalGates.description")
                  }
                ]}
                actionLabel={t("workforces.emptyState.action")}
                onAction={() => setView("create")}
                className="h-full mt-4"
              />
            )
          ) : (
            <>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {items.map((item) => {
                  const runDisabledReason = getRunDisabledReason(item.status, t)
                  const deployDisabledReason = getDeployDisabledReason(item.status, t)
                  return (
                    <Card key={item.id} className="relative overflow-hidden flex flex-col h-full hover:shadow-md transition-shadow">
                      <CardContent className="flex flex-col h-full">
                        <div className="flex items-start gap-3 mb-4">
                          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-blue-600 text-white">
                            <Users className="h-5 w-5" />
                          </div>
                          <div className="flex-1 min-w-0 pr-6">
                            <Link
                              href={`/workforces/${item.id}`}
                              className="text-base font-semibold truncate hover:underline block"
                            >
                              {item.name}
                            </Link>
                            <div className="mt-1">
                              <WorkforceStatusBadge status={item.status} />
                            </div>
                            <div className="text-xs text-muted-foreground truncate mt-1">
                              {t("workforces.list.manager", { name: item.manager?.name })}
                            </div>
                          </div>
                        </div>

                        <div className="absolute right-3 top-3">
                          <Popover
                            open={openMenuId === item.id}
                            onOpenChange={(open) => setOpenMenuId(open ? item.id : null)}
                          >
                            <PopoverTrigger asChild>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 text-muted-foreground hover:text-foreground"
                                aria-label={t("workforces.actions.moreActions")}
                                disabled={busyItemId === item.id}
                              >
                                <MoreVertical className="h-4 w-4" />
                              </Button>
                            </PopoverTrigger>
                            <PopoverContent align="end" className="w-40 p-1">
                              <div className="flex flex-col">
                                {item.status !== "archived" && (
                                  <>
                                    {item.status === "active" ? (
                                      <Button
                                        variant="ghost"
                                        className="justify-start px-2 py-1.5 h-auto font-normal text-sm"
                                        disabled={busyItemId === item.id}
                                        onClick={() => handleUnpublish(item)}
                                      >
                                        <Globe className="mr-2 h-4 w-4" />
                                        {t("workforces.actions.unpublish")}
                                      </Button>
                                    ) : (
                                      <Button
                                        variant="ghost"
                                        className="justify-start px-2 py-1.5 h-auto font-normal text-sm"
                                        disabled={busyItemId === item.id}
                                        onClick={() => handlePublish(item)}
                                      >
                                        <Globe className="mr-2 h-4 w-4" />
                                        {t("workforces.actions.publish")}
                                      </Button>
                                    )}
                                    <div className="h-px bg-border my-1 mx-1" />
                                  </>
                                )}
                                {item.status === "archived" ? (
                                  <Button
                                    variant="ghost"
                                    className="justify-start px-2 py-1.5 h-auto font-normal text-sm"
                                    disabled={busyItemId === item.id}
                                    onClick={() => handleUnarchive(item)}
                                  >
                                    <ArchiveRestore className="mr-2 h-4 w-4" />
                                    {t("workforces.actions.unarchive")}
                                  </Button>
                                ) : (
                                  <Button
                                    variant="ghost"
                                    className="justify-start px-2 py-1.5 h-auto font-normal text-sm"
                                    disabled={busyItemId === item.id}
                                    onClick={() => handleArchive(item)}
                                  >
                                    <Archive className="mr-2 h-4 w-4" />
                                    {t("workforces.actions.archive")}
                                  </Button>
                                )}
                                <div className="h-px bg-border my-1 mx-1" />
                                <Button
                                  variant="ghost"
                                  className="justify-start px-2 py-1.5 h-auto font-normal text-sm text-destructive hover:text-destructive hover:bg-destructive/10"
                                  disabled={busyItemId === item.id}
                                  onClick={() => {
                                    setOpenMenuId(null)
                                    setDeleteItem(item)
                                  }}
                                >
                                  <Trash2 className="mr-2 h-4 w-4" />
                                  {t("workforces.actions.delete")}
                                </Button>
                              </div>
                            </PopoverContent>
                          </Popover>
                        </div>

                        <div className="flex-1">
                          <p className="text-sm text-muted-foreground line-clamp-2 mb-4">
                            {item.description || t("workforces.common.noDescription")}
                          </p>

                          <div className="flex items-center gap-2 mb-4">
                            <div className="flex -space-x-2">
                              {Array.from({ length: Math.min(item.worker_count, 4) }).map((_, i) => (
                                <div key={i} className="flex h-6 w-6 items-center justify-center rounded-full border-2 border-background bg-blue-600 text-[10px] font-medium text-white">
                                  {String.fromCharCode(65 + i)}
                                </div>
                              ))}
                            </div>
                            <span className="text-xs text-muted-foreground">
                              {t("workforces.list.workers", { count: item.worker_count })}
                            </span>
                          </div>
                        </div>

                        <div className="mt-auto pt-4 border-t flex items-center justify-between">
                          <div className="text-xs text-muted-foreground">
                            {item.last_run?.created_at ? (
                              <span>{t("workforces.list.lastRunTime")} {formatTime(item.last_run.created_at, 'datetime')}</span>
                            ) : (
                              t("workforces.list.noRuns")
                            )}
                          </div>
                          <div className="flex items-center gap-2">
                            {runDisabledReason ? (
                              <Button size="sm" className="h-8 bg-blue-600 hover:bg-blue-700 text-white rounded-md px-3" disabled>
                                <Play className="mr-1.5 h-3.5 w-3.5 fill-current" />
                                {t("workforces.actions.run")}
                              </Button>
                            ) : (
                              <Button size="sm" className="h-8 bg-blue-600 hover:bg-blue-700 text-white rounded-md px-3" asChild>
                                <Link href={`/workforces/${item.id}/run`}>
                                  <Play className="mr-1.5 h-3.5 w-3.5 fill-current" />
                                  {t("workforces.actions.run")}
                                </Link>
                              </Button>
                            )}
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-8 w-8 rounded-md p-0"
                              title={deployDisabledReason || t("workforces.actions.deploy")}
                              disabled={Boolean(deployDisabledReason)}
                              onClick={() => {
                                setDeployItem(item)
                                setDeployView("options")
                              }}
                            >
                              <Rocket className="h-3.5 w-3.5" />
                            </Button>
                            <Button size="sm" variant="outline" className="h-8 rounded-md px-3" asChild>
                              <Link href={`/workforces/${item.id}`}>
                                <Pencil className="mr-1.5 h-3.5 w-3.5" />
                                {t("workforces.actions.edit")}
                              </Link>
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )
                })}
              </div>

              {pages > 1 ? (
                <div className="flex items-center justify-between">
                  <div className="text-sm text-muted-foreground">
                    {t("workforces.pagination.showing", {
                      start: (page - 1) * pageSize + 1,
                      end: Math.min(page * pageSize, total),
                      total,
                    })}
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage((current) => current - 1)}
                      disabled={page <= 1}
                    >
                      <ChevronLeft className="mr-1 h-4 w-4" />
                      {t("workforces.pagination.prev")}
                    </Button>
                    <span className="text-sm text-muted-foreground">
                      {t("workforces.pagination.page", { page, pages })}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage((current) => current + 1)}
                      disabled={page >= pages}
                    >
                      {t("workforces.pagination.next")}
                      <ChevronRight className="ml-1 h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ) : null}
            </>
          )
        ) : null}
      </div>

      <WorkforceDeployHubDialog
        open={!!deployItem && deployView === "options"}
        onClose={closeDeploy}
        workforceName={deployItem?.name ?? ""}
        onSelectEmbed={() => setDeployView("embed")}
        onSelectApi={() => setDeployView("api")}
        onSelectShare={() => setDeployView("share")}
        onSelectWebhook={() => {
          setTriggersItem(deployItem)
          closeDeploy()
        }}
      />
      {deployItem && (
        <DeployWorkforceDialog
          open={deployView === "api"}
          workforceId={deployItem.id}
          workforceName={deployItem.name}
          onClose={() => setDeployView("options")}
        />
      )}
      <WorkforceWidgetDialog
        workforce={deployView === "embed" ? deployItem : null}
        open={deployView === "embed"}
        onClose={() => setDeployView("options")}
      />
      <WorkforceShareDialog
        workforce={deployView === "share" ? deployItem : null}
        open={deployView === "share"}
        onClose={() => setDeployView("options")}
      />
      <AgentTriggersDialog
        agentId={null}
        owner={triggersItem ? { kind: "workforce", id: triggersItem.id } : null}
        agentName={triggersItem?.name}
        open={!!triggersItem}
        onOpenChange={(open) => { if (!open) setTriggersItem(null) }}
      />
      <ConfirmDialog
        isOpen={!!deleteItem}
        onOpenChange={(open) => { if (!open) setDeleteItem(null) }}
        onConfirm={handleDelete}
        isLoading={deleting}
        title={t("workforces.delete.confirmTitle")}
        description={t("workforces.delete.confirmDescription", { name: deleteItem?.name ?? "" })}
        confirmText={t("workforces.delete.confirmAction")}
      />
    </div>
  )
}
