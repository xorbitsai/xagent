"use client"

import React, { useEffect, useMemo, useState } from "react"
import { Check, Copy, KeyRound, Loader2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { toast } from "@/components/ui/sonner"
import { useI18n } from "@/contexts/i18n-context"
import { copyToClipboard } from "@/lib/clipboard"
import { getApiSnippetTarget } from "@/lib/api-snippet-base-url"
import {
  formatWorkforceApiSnippets,
  type ApiSnippetTab,
} from "@/lib/api-snippet-format"
import type { ApiSnippetTarget } from "@/lib/api-snippet-target"
import {
  createWorkforceApiKey,
  listAgentApiKeys,
  type AgentApiKeyListItem,
} from "@/lib/agent-api-keys-api"

interface DeployWorkforceDialogProps {
  open: boolean
  workforceId: number
  workforceName: string
  onClose: () => void
}

/**
 * REST API / SDK deployment surface for a workforce (#949).
 *
 * Self-contained (unlike the agent deploy dialog, which delegates key
 * management to a shared page): shows the run snippet and manages this
 * workforce's ``xag_*`` keys inline -- list, create (one-shot reveal).
 * The multi-turn / polling flow reuses ``/v1/chat/tasks/{id}`` and is
 * documented in the snippet itself.
 */
export function DeployWorkforceDialog({
  open,
  workforceId,
  workforceName,
  onClose,
}: DeployWorkforceDialogProps) {
  const { t } = useI18n()
  const [apiTab, setApiTab] = useState<ApiSnippetTab>("curl")
  const [copiedSnippet, setCopiedSnippet] = useState(false)
  const [apiTarget, setApiTarget] = useState<ApiSnippetTarget>({ baseUrl: "" })

  const [keys, setKeys] = useState<AgentApiKeyListItem[]>([])
  const [loadingKeys, setLoadingKeys] = useState(false)
  const [creating, setCreating] = useState(false)
  const [newLabel, setNewLabel] = useState("")
  const [revealedKey, setRevealedKey] = useState<string | null>(null)
  const [copiedKey, setCopiedKey] = useState(false)

  useEffect(() => {
    if (open) setApiTarget(getApiSnippetTarget())
  }, [open])

  useEffect(() => {
    if (!open) {
      // Never keep a one-shot plaintext secret around after close.
      setRevealedKey(null)
      setNewLabel("")
      return
    }
    let cancelled = false
    setLoadingKeys(true)
    listAgentApiKeys({ workforceId })
      .then((rows) => {
        if (!cancelled) setKeys(rows)
      })
      .catch(() => {
        if (!cancelled) toast.error(t("apiKeysPage.messages.loadFailed"))
      })
      .finally(() => {
        if (!cancelled) setLoadingKeys(false)
      })
    return () => {
      cancelled = true
    }
  }, [open, workforceId, t])

  const snippets = useMemo(
    () => formatWorkforceApiSnippets(workforceId, apiTarget),
    [workforceId, apiTarget],
  )

  const handleCopySnippet = async () => {
    const ok = await copyToClipboard(snippets[apiTab])
    if (ok) {
      setCopiedSnippet(true)
      setTimeout(() => setCopiedSnippet(false), 1500)
    }
  }

  const handleCopyKey = async () => {
    if (!revealedKey) return
    const ok = await copyToClipboard(revealedKey)
    if (ok) {
      setCopiedKey(true)
      setTimeout(() => setCopiedKey(false), 1500)
    }
  }

  const handleCreateKey = async () => {
    setCreating(true)
    try {
      const created = await createWorkforceApiKey(
        workforceId,
        newLabel.trim() || null,
      )
      setRevealedKey(created.full_key)
      setNewLabel("")
      try {
        const rows = await listAgentApiKeys({ workforceId })
        setKeys(rows)
      } catch {
        // The key already exists and its one-shot secret must remain visible.
        // Report only the secondary refresh failure so users do not retry the
        // creation and accidentally issue duplicate credentials.
        toast.error(t("apiKeysPage.messages.loadFailed"))
      }
    } catch {
      toast.error(t("apiKeysPage.messages.createFailed"))
    } finally {
      setCreating(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <KeyRound className="h-4 w-4" />
            {t("deploy_workforce.title") || "Deploy via REST API / SDK"}
          </DialogTitle>
          <DialogDescription>
            {t("deploy_workforce.desc") ||
              `Create runs on "${workforceName}" with a workforce API key, then poll GET /v1/chat/tasks/{id} for results.`}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-5">
          {/* Snippet tabs */}
          <div>
            <div className="flex gap-1 border-b">
              {(["curl", "python"] as ApiSnippetTab[]).map((tab) => (
                <button
                  key={tab}
                  type="button"
                  onClick={() => setApiTab(tab)}
                  className={`px-3 py-1.5 text-sm font-medium border-b-2 -mb-px transition-colors ${
                    apiTab === tab
                      ? "border-primary text-foreground"
                      : "border-transparent text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {tab === "curl" ? "cURL" : "Python"}
                </button>
              ))}
            </div>
            <div className="mt-3 bg-muted p-4 rounded-md text-xs font-mono relative group">
              <pre className="whitespace-pre-wrap break-all text-muted-foreground max-h-72 overflow-auto">
                {snippets[apiTab]}
              </pre>
              <Button
                variant="secondary"
                size="icon"
                className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
                onClick={handleCopySnippet}
                title={t("deploy_workforce.copy") || "Copy"}
              >
                {copiedSnippet ? (
                  <Check className="h-4 w-4 text-green-500" />
                ) : (
                  <Copy className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>

          {/* One-shot reveal of a freshly created key */}
          {revealedKey && (
            <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 space-y-2">
              <div className="text-sm font-medium">
                {t("deploy_workforce.new_key") ||
                  "Copy this key now — it won't be shown again."}
              </div>
              <div className="flex items-center gap-2">
                <code className="flex-1 font-mono text-xs break-all">
                  {revealedKey}
                </code>
                <Button variant="secondary" size="icon" onClick={handleCopyKey}>
                  {copiedKey ? (
                    <Check className="h-4 w-4 text-green-500" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>
          )}

          {/* Key management */}
          <div className="space-y-2">
            <div className="text-sm font-medium">
              {t("deploy_workforce.keys_title") || "API keys"}
            </div>
            <div className="flex items-center gap-2">
              <Input
                value={newLabel}
                onChange={(e) => setNewLabel(e.target.value)}
                placeholder={
                  t("deploy_workforce.label_placeholder") ||
                  "Label (optional), e.g. CI pipeline"
                }
                maxLength={100}
              />
              <Button onClick={handleCreateKey} disabled={creating}>
                {creating ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  t("deploy_workforce.create_key") || "Create key"
                )}
              </Button>
            </div>

            {loadingKeys ? (
              <div className="text-sm text-muted-foreground py-2">
                <Loader2 className="h-4 w-4 animate-spin inline mr-1" />
                {t("common.loading") || "Loading…"}
              </div>
            ) : keys.length === 0 ? (
              <div className="text-sm text-muted-foreground py-2">
                {t("deploy_workforce.no_keys") || "No API keys yet."}
              </div>
            ) : (
              <ul className="divide-y rounded-md border">
                {keys.map((key) => (
                  <li
                    key={key.id}
                    className="flex items-center justify-between px-3 py-2 text-sm"
                  >
                    <span className="flex items-center gap-2">
                      <span className="font-mono text-xs text-muted-foreground">
                        {key.masked_key}
                      </span>
                      {key.label && (
                        <span className="text-muted-foreground">
                          {key.label}
                        </span>
                      )}
                    </span>
                    <span className="text-[11px] capitalize text-muted-foreground">
                      {key.status}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
