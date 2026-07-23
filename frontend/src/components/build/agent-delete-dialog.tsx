"use client"

import React, { useEffect, useState } from "react"
import Link from "next/link"
import {
  AlertTriangle,
  CheckCircle2,
  ExternalLink,
  Info,
  Loader2,
  Trash2,
} from "lucide-react"

import { WorkforceStatusBadge } from "@/components/workforce"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { useI18n } from "@/contexts/i18n-context"
import { cn } from "@/lib/utils"
import type {
  AgentDeleteConflictDetail,
  AgentDeleteWorkforceReference,
} from "@/lib/agent-delete"

export type AgentDeletePendingAction =
  | { kind: "delete" }
  | { kind: "discard"; workforceId: number }
  | null

interface AgentDeleteDialogProps {
  target: { id: number; name: string } | null
  conflict: AgentDeleteConflictDetail | null
  pendingAction: AgentDeletePendingAction
  onOpenChange: (open: boolean) => void
  onConfirmDelete: () => void
  onDiscardWorkforce: (reference: AgentDeleteWorkforceReference) => void
}

export function AgentDeleteDialog({
  target,
  conflict,
  pendingAction,
  onOpenChange,
  onConfirmDelete,
  onDiscardWorkforce,
}: AgentDeleteDialogProps) {
  const { t } = useI18n()
  const [confirmDiscardId, setConfirmDiscardId] = useState<number | null>(null)
  const isPending = pendingAction !== null

  useEffect(() => {
    setConfirmDiscardId(null)
  }, [target?.id, conflict])

  const handleOpenChange = (open: boolean) => {
    if (!isPending) {
      onOpenChange(open)
    }
  }

  const handleDelete = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault()
    onConfirmDelete()
  }

  return (
    <AlertDialog open={target !== null} onOpenChange={handleOpenChange}>
      <AlertDialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-2xl">
        <AlertDialogHeader className="space-y-3">
          <div className="flex flex-col items-center gap-3 sm:flex-row">
            <div
              className={cn(
                "flex h-10 w-10 shrink-0 items-center justify-center rounded-full",
                conflict
                  ? "bg-amber-500/15 text-amber-600 dark:text-amber-400"
                  : "bg-destructive/10 text-destructive",
              )}
            >
              {conflict ? (
                <AlertTriangle className="h-5 w-5" aria-hidden="true" />
              ) : (
                <Trash2 className="h-5 w-5" aria-hidden="true" />
              )}
            </div>
            <AlertDialogTitle>
              {t(
                conflict
                  ? "builds.list.deleteDialog.blockedTitle"
                  : "builds.list.deleteDialog.title",
              )}
            </AlertDialogTitle>
          </div>
          <AlertDialogDescription asChild>
            <div className="space-y-4 text-left">
              <p>
                {target
                  ? t(
                      conflict
                        ? "builds.list.deleteDialog.blockedDescription"
                        : "builds.list.deleteDialog.description",
                      { name: target.name },
                    )
                  : null}
              </p>

              {conflict ? (
                <>
                  {conflict.references.length > 0 ? (
                    <ul className="space-y-3" aria-label={t("builds.list.deleteDialog.referencesLabel")}>
                      {conflict.references.map((reference) => {
                        const isDiscardPending =
                          pendingAction?.kind === "discard" &&
                          pendingAction.workforceId === reference.workforce_id
                        const isConfirmingDiscard =
                          confirmDiscardId === reference.workforce_id

                        return (
                          <li
                            key={reference.workforce_id}
                            className="overflow-hidden rounded-lg border shadow-sm"
                          >
                            <div className="flex flex-wrap items-center gap-2 p-4">
                              <Link
                                href={`/workforces/${reference.workforce_id}`}
                                target="_blank"
                                rel="noreferrer"
                                aria-label={t("builds.list.deleteDialog.openWorkforce", {
                                  name: reference.name,
                                })}
                                className="inline-flex items-center gap-1 font-semibold text-primary hover:underline"
                              >
                                {reference.name}
                                <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
                              </Link>
                              <WorkforceStatusBadge status={reference.status} />
                              {reference.roles.map((role) => (
                                <Badge key={role} variant="outline">
                                  {t(`builds.list.deleteDialog.roles.${role}`)}
                                </Badge>
                              ))}
                              {!reference.can_edit ? (
                                <Badge variant="secondary">
                                  {t("workforces.actions.readOnly")}
                                </Badge>
                              ) : null}
                            </div>

                            {reference.can_discard ? (
                              <div className="flex flex-wrap items-center gap-2 border-t bg-muted/40 px-4 py-2.5">
                                <Button
                                  type="button"
                                  variant={isConfirmingDiscard ? "destructive" : "outline"}
                                  size="sm"
                                  disabled={isPending}
                                  onClick={() => {
                                    if (!isConfirmingDiscard) {
                                      setConfirmDiscardId(reference.workforce_id)
                                      return
                                    }
                                    setConfirmDiscardId(null)
                                    onDiscardWorkforce(reference)
                                  }}
                                >
                                  {isDiscardPending ? (
                                    <Loader2 className="animate-spin" aria-hidden="true" />
                                  ) : null}
                                  <span>
                                    {t(
                                      isConfirmingDiscard
                                        ? "builds.list.deleteDialog.confirmDiscardDraft"
                                        : "builds.list.deleteDialog.discardDraft",
                                      { name: reference.name },
                                    )}
                                  </span>
                                </Button>
                              </div>
                            ) : null}
                          </li>
                        )
                      })}
                    </ul>
                  ) : null}

                  {conflict.has_hidden_references ? (
                    <p
                      className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-amber-700 dark:text-amber-400"
                      role="note"
                    >
                      <Info className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                      <span>{t("builds.list.deleteDialog.hiddenReferences")}</span>
                    </p>
                  ) : null}

                  {conflict.references.length === 0 &&
                  !conflict.has_hidden_references ? (
                    <p
                      className="flex items-start gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-3 text-emerald-700 dark:text-emerald-400"
                      aria-live="polite"
                    >
                      <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                      <span>{t("builds.list.deleteDialog.readyToRetry")}</span>
                    </p>
                  ) : null}
                </>
              ) : null}
            </div>
          </AlertDialogDescription>
        </AlertDialogHeader>

        <AlertDialogFooter>
          <AlertDialogCancel disabled={isPending}>
            {t("common.cancel")}
          </AlertDialogCancel>
          <AlertDialogAction
            disabled={isPending}
            className="bg-destructive text-white hover:bg-destructive/90"
            onClick={handleDelete}
          >
            {pendingAction?.kind === "delete" ? (
              <Loader2 className="animate-spin" aria-hidden="true" />
            ) : null}
            <span>
              {t(
                conflict
                  ? "builds.list.deleteDialog.retryDelete"
                  : "builds.list.deleteDialog.confirm",
              )}
            </span>
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
