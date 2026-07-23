"use client"

import React, { useState, useEffect, useRef, useMemo, useCallback } from "react"
import { ResizableThreeColumnLayout } from "@/components/layout/resizable-three-column-layout"
import { AgentBuilderChat } from "./agent-builder-chat"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { apiRequest, isJsonRecord } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { isBuiltinModel, hostnameFromUrl } from "@/lib/models"
import { PlusCircle, MessageSquare, Upload, Settings2, Check, Zap, BookOpen, Gauge, Sparkles, Loader2, X, XCircle, Trash2, Bot, Brain, Webhook, CalendarClock, Mail, Eye, Workflow, AlertCircle, Copy } from "lucide-react"
import { ConnectMcpDialog } from "@/components/mcp/connect-mcp-dialog"
import { useI18n } from "@/contexts/i18n-context"
import { useApp } from "@/contexts/app-context-chat"
import { useAuth } from "@/contexts/auth-context"
import { useMcpApps } from "@/contexts/mcp-apps-context"
import { createFileChipHTML } from "@/components/chat/FileChip"
import { MultiSelect } from "@/components/ui/multi-select"
import { useFileMention } from "@/hooks/use-file-mention"
import { FileMentionDropdown } from "@/components/chat/FileMentionDropdown"
import {
  Select,
  SelectRadix,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select"
import {
  InfoTooltip,
} from "@/components/ui/tooltip"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { useRouter, useSearchParams } from "next/navigation"
import { KnowledgeBaseCreationDialog } from "@/components/kb/knowledge-base-creation-dialog"
import { toast } from "@/components/ui/sonner"
import { cn } from "@/lib/utils"
import { getBrandingFromEnv } from "@/lib/branding"
import { findMatchingMcpApp, findMatchingMcpServer, mcpNameMatches } from "@/lib/mcp-lookup"
import { BuildFilePreviewSheet } from "./build-file-preview-sheet"
import { TaskConversationPanel } from "@/components/task/task-conversation-panel"
import { AgentTriggersDialog } from "./agent-triggers-dialog"
import { AgentFlowView } from "./agent-flow-view"
import {
  AgentTrigger,
  AgentTriggerType,
  FailedStagedTrigger,
  StagedTrigger,
  createAgentTrigger,
  createStagedTriggers,
  listAgentTriggers,
  stagedToCreatePayload,
  stagedToPseudoTrigger,
} from "@/lib/agent-triggers-api"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { copyToClipboard } from "@/lib/clipboard"
import {
  sanitizeUnsharedConnectors,
  sanitizeUnsharedKnowledgeBases,
  type UnsharedConnector,
  type UnsharedKnowledgeBase,
} from "@/lib/team-sharing-sanitizers"

interface KnowledgeBase {
  name: string
  [key: string]: any
}

interface Skill {
  name: string
  description?: string
  when_to_use?: string
  tags?: string[]
  [key: string]: any
}

interface Tool {
  name: string
  description: string
  type: string
  category: string
  enabled: boolean
  [key: string]: any
}

interface Model {
  id: number
  model_id: string
  model_name: string
  model_provider: string
  category: string
  base_url?: string
  is_default?: boolean
  is_small_fast?: boolean
  is_visual?: boolean
  is_compact?: boolean
}

interface UserDefaultModel {
  id: number
  config_type: string
  model: {
    id: number
    model_id: string
    model_name: string
    model_provider: string
  }
}

interface AgentModelConfig {
  general: number | null
  small_fast: number | null
  visual: number | null
  compact: number | null
}

interface AgentBuilderProps {
  agentId?: string
}

interface TemplateRequirements {
  requiredSkills: string[]
  requiredToolCategories: string[]
  requiredMcpServers: string[]
  requiresKnowledgeBase: boolean
}

// Categories a user may never assign from the builder, mirroring the
// backend's AGENT_CONFIG_UNASSIGNABLE_CATEGORIES (which also strips them on
// write): `agent` (multi-agent delegation) is configured through Workforce
// instead (issue #802), and `other` is an internal fallback bucket.
const isAssignableToolCategory = (c: string) => c !== 'agent' && c !== 'other'

function readNonEmptyString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null
}

function getAgentUpdateErrorMessage(error: unknown, fallback: string): string {
  if (!isJsonRecord(error)) return fallback

  const detail = error.detail
  const detailMessage = readNonEmptyString(detail)
  if (detailMessage) return detailMessage

  if (isJsonRecord(detail)) {
    const message = readNonEmptyString(detail.message)
    if (message) return message
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        const itemMessage = readNonEmptyString(item)
        if (itemMessage) return itemMessage
        if (!isJsonRecord(item)) return null
        return readNonEmptyString(item.msg) ?? readNonEmptyString(item.message)
      })
      .filter((message): message is string => message !== null)
    if (messages.length > 0) return messages.join("; ")
  }

  return readNonEmptyString(error.message) ?? fallback
}

// One-time reveal of auto-generated webhook secrets. Rendered both inside the
// creation success dialog and inline in the config form (retry path / while
// the dialog is closed); only the inline instance is dismissible.
function WebhookSecretsAlert({
  secrets,
  onDismiss,
}: {
  secrets: { name: string; secret: string }[]
  onDismiss?: () => void
}) {
  const { t } = useI18n()
  if (secrets.length === 0) return null
  return (
    <Alert className="border-amber-300 bg-amber-50 text-amber-950 dark:bg-amber-950/30 dark:text-amber-100">
      <AlertCircle className="h-4 w-4" />
      <AlertTitle className="flex items-center justify-between gap-2">
        {t("triggers.secret.title")}
        {onDismiss && (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-6 w-6 shrink-0"
            onClick={onDismiss}
          >
            <X className="h-3.5 w-3.5" />
          </Button>
        )}
      </AlertTitle>
      <AlertDescription>
        <div className="mt-2 space-y-2">
          {secrets.map((item) => (
            <div key={`${item.name}:${item.secret}`} className="flex items-center gap-2">
              <span className="max-w-[160px] shrink-0 truncate text-xs font-medium">{item.name}</span>
              <code className="min-w-0 flex-1 break-all rounded bg-background/70 px-2 py-1.5 text-xs">
                {item.secret}
              </code>
              <Button
                type="button"
                size="icon"
                variant="secondary"
                className="h-7 w-7 shrink-0"
                onClick={async () => {
                  if (await copyToClipboard(item.secret)) {
                    toast.success(t("common.copied"))
                  }
                }}
                title={t("common.copy")}
              >
                <Copy className="h-3.5 w-3.5" />
              </Button>
            </div>
          ))}
        </div>
      </AlertDescription>
    </Alert>
  )
}

export function AgentBuilder({ agentId }: AgentBuilderProps) {
  const MAX_INSTRUCTIONS_LENGTH = 8192;
  const { state, setTaskId, sendMessage, dispatch, closeFilePreview } = useApp()
  const { t, locale } = useI18n()
  const { apps: officialApps, getAppIcon, refresh: refreshMcpApps } = useMcpApps()
  const { user, inTeam, teamRole } = useAuth()
  // inTeam gates the whole control (standard xagent has no teams);
  // canSetAdminsOnly gates the "admins" option to team admins.
  const canSetAdminsOnly = teamRole === "admin"
  // Set once loadAgent decides to load an owner-scoped MCP list for an admin
  // cross-user view, so the mount-time self-scoped fetch won't clobber it.
  const ownerScopedMcpRef = useRef(false)
  const router = useRouter()
  const searchParams = useSearchParams()
  const templateId = searchParams.get("template")
  const [localAgentId, setLocalAgentId] = useState<string | undefined>(agentId)
  const isEditMode = !!localAgentId
  const branding = getBrandingFromEnv();

  // Config State
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [instructions, setInstructions] = useState("")
  const [executionMode, setExecutionMode] = useState("balanced") // "flash", "balanced", "think"
  const [suggestedPrompts, setSuggestedPrompts] = useState<string[]>([])
  const [visibility, setVisibility] = useState<"team" | "admins">("team")
  // Agent ownership: personal (team_id NULL) is the default; "team" reveals the
  // visibility sub-option and drives promote/demote calls on save.
  const [ownership, setOwnership] = useState<"personal" | "team">("personal")
  const [modelConfig, setModelConfig] = useState<AgentModelConfig>({
    general: null,
    small_fast: null,
    visual: null,
    compact: null,
  })
  const [selectedKbs, setSelectedKbs] = useState<string[]>([])
  const [selectedSkills, setSelectedSkills] = useState<string[]>([])
  const [selectedToolCategories, setSelectedToolCategories] = useState<string[]>([])
  const [selectedMcpServers, setSelectedMcpServers] = useState<string[]>([])
  const [logoFile, setLogoFile] = useState<File | null>(null)
  const [logoUrl, setLogoUrl] = useState<string | null>(null)  // Existing logo URL
  const [isCreating, setIsCreating] = useState(false)
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [loadingAgent, setLoadingAgent] = useState(false)
  const [originalData, setOriginalData] = useState<any>(null)
  const [isKbModalOpen, setIsKbModalOpen] = useState(false)
  const [isModelConfigOpen, setIsModelConfigOpen] = useState(false)
  const [isTriggersDialogOpen, setIsTriggersDialogOpen] = useState(false)
  const [triggerDialogInitialType, setTriggerDialogInitialType] = useState<AgentTriggerType | null>(null)
  const [triggerSummary, setTriggerSummary] = useState<AgentTrigger[]>([])
  // Triggers configured before the agent exists (#928). Posted via the
  // trigger API right after the agent is created, then cleared.
  const [stagedTriggers, setStagedTriggers] = useState<StagedTrigger[]>([])
  // Staged triggers whose POST failed after the agent was created. Their
  // config is kept for retry/discard instead of being silently dropped.
  const [failedStagedTriggers, setFailedStagedTriggers] = useState<FailedStagedTrigger[]>([])
  // Auto-generated webhook secrets from staged-trigger creation; shown once.
  const [createdWebhookSecrets, setCreatedWebhookSecrets] = useState<
    { name: string; secret: string }[]
  >([])
  // Creation succeeded but ownership reconciliation opened the
  // share-connectors dialog instead of the success dialog. The success dialog
  // (which reveals generated webhook secrets) must still be shown once that
  // flow resolves, whichever way it resolves.
  const pendingCreationDialogRef = useRef(false)
  const [showAIAssistant, setShowAIAssistant] = useState(false)
  const [viewMode, setViewMode] = useState<"config" | "flow">("config")
  const [configSynced, setConfigSynced] = useState(false)
  const [notFound, setNotFound] = useState(false)
  // Admin viewing another user's agent: writes are owner-only, so the whole
  // builder is locked and Save/Publish/AI-assistant are hidden (#783 follow-up).
  const [readOnly, setReadOnly] = useState(false)
  const isFirstRender = useRef(true)
  const modelSectionRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false
      return
    }
    setConfigSynced(true)
    const timer = setTimeout(() => setConfigSynced(false), 2000)
    return () => clearTimeout(timer)
  }, [name, description, instructions, executionMode, suggestedPrompts, selectedKbs, selectedSkills, selectedToolCategories, modelConfig])

  // Create Success Dialog State
  const [showSuccessDialog, setShowSuccessDialog] = useState(false)
  const [createdAgent, setCreatedAgent] = useState<any>(null)
  // Connectors referenced by an agent that isn't yet team-shared. Populated from
  // a 422 promote-team response so the user can share them and retry.
  const [unsharedConnectors, setUnsharedConnectors] = useState<UnsharedConnector[]>([])
  const [unsharedKnowledgeBases, setUnsharedKnowledgeBases] = useState<
    UnsharedKnowledgeBase[]
  >([])
  const [isSharingConnectors, setIsSharingConnectors] = useState(false)
  const hasUnresolvedConnectors = unsharedConnectors.some((c) => c.reason === "unresolved")
  const [templateRequirements, setTemplateRequirements] = useState<TemplateRequirements | null>(null)

  // Data State
  const [models, setModels] = useState<Model[]>([])
  const [kbs, setKbs] = useState<KnowledgeBase[]>([])
  const [skills, setSkills] = useState<Skill[]>([])
  const [tools, setTools] = useState<Tool[]>([])
  const [mcpServers, setMcpServers] = useState<any[]>([])
  const [isConnectMcpOpen, setIsConnectMcpOpen] = useState(false)
  const [isInitialDataLoaded, setIsInitialDataLoaded] = useState(false)

  const refreshTriggerSummary = useCallback(async () => {
    if (!localAgentId) {
      setTriggerSummary([])
      return
    }

    try {
      setTriggerSummary(await listAgentTriggers(localAgentId))
    } catch (error) {
      console.error("Failed to load trigger summary:", error)
    }
  }, [localAgentId])

  useEffect(() => {
    void refreshTriggerSummary()
  }, [refreshTriggerSummary])

  // While failed retries (or an unread secret) are on screen, dialog-close
  // handlers skip router.replace to keep this instance mounted; this flag
  // remembers that a navigation is owed once both lists drain.
  const navPendingRef = useRef(false)
  const reconcileDeferredNavigation = (
    nextFailed: FailedStagedTrigger[],
    nextSecrets: { name: string; secret: string }[],
  ) => {
    if (!navPendingRef.current) return
    if (nextFailed.length > 0 || nextSecrets.length > 0) return
    navPendingRef.current = false
    if (createdAgent?.id) {
      router.replace(`/build/${createdAgent.id}`)
    }
  }

  // Retry a staged trigger whose POST failed during agent creation. The agent
  // exists by now, so this goes straight through the live trigger API.
  // Entries are keyed by their stable clientId — a positional index goes
  // stale when another entry is discarded while a retry is in flight.
  const retryFailedTrigger = async (clientId: number) => {
    if (!localAgentId) return
    const entry = failedStagedTriggers.find((item) => item.staged.clientId === clientId)
    if (!entry) return
    try {
      const created = await createAgentTrigger(localAgentId, stagedToCreatePayload(entry.staged))
      const nextFailed = failedStagedTriggers.filter(
        (item) => item.staged.clientId !== clientId,
      )
      let nextSecrets = createdWebhookSecrets
      const secret = created.webhook_secret
      if (secret && !entry.staged.secret) {
        nextSecrets = [...createdWebhookSecrets, { name: entry.staged.name, secret }]
      }
      setFailedStagedTriggers(nextFailed)
      setCreatedWebhookSecrets(nextSecrets)
      void refreshTriggerSummary()
      toast.success(t("triggers.messages.created"))
      reconcileDeferredNavigation(nextFailed, nextSecrets)
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error)
      setFailedStagedTriggers((prev) =>
        prev.map((item) =>
          item.staged.clientId === clientId ? { ...item, error: reason } : item,
        ),
      )
      toast.error(
        t("triggers.messages.stagedCreateFailed", { name: entry.staged.name, error: reason }),
      )
    }
  }

  // Discarding drops the entered config irreversibly, so it uses the same
  // two-click confirm pattern as trigger deletion in the dialog.
  const [discardConfirmClientId, setDiscardConfirmClientId] = useState<number | null>(null)
  const discardFailedTrigger = (clientId: number) => {
    if (discardConfirmClientId !== clientId) {
      setDiscardConfirmClientId(clientId)
      return
    }
    setDiscardConfirmClientId(null)
    const nextFailed = failedStagedTriggers.filter(
      (item) => item.staged.clientId !== clientId,
    )
    setFailedStagedTriggers(nextFailed)
    reconcileDeferredNavigation(nextFailed, createdWebhookSecrets)
  }

  const dismissWebhookSecrets = () => {
    setCreatedWebhookSecrets([])
    reconcileDeferredNavigation(failedStagedTriggers, [])
  }

  // During creation the agent has no server-side triggers yet; the summary
  // sections and flow view render the staged ones instead (#928).
  const effectiveTriggerSummary = useMemo(
    () => (localAgentId ? triggerSummary : stagedTriggers.map(stagedToPseudoTrigger)),
    [localAgentId, triggerSummary, stagedTriggers],
  )

  const triggerStats = useMemo(() => {
    const stats = {
      webhook: { total: 0, enabled: 0 },
      scheduled: { total: 0, enabled: 0 },
      gmail: { total: 0, enabled: 0 },
    }
    effectiveTriggerSummary.forEach((trigger) => {
      if (trigger.type !== "webhook" && trigger.type !== "scheduled" && trigger.type !== "gmail") return
      stats[trigger.type].total += 1
      if (trigger.enabled) {
        stats[trigger.type].enabled += 1
      }
    })
    return stats
  }, [effectiveTriggerSummary])

  const gmailConnection = useMemo(() => {
    const gmailApp = findMatchingMcpApp(officialApps, "gmail")
    return {
      isConnected: Boolean(gmailApp?.is_connected),
      connectedAccount: gmailApp?.connected_account ?? null,
    }
  }, [officialApps])

  // File picker state for Instructions
  const instructionsRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [isInstructionsFocused, setIsInstructionsFocused] = useState(false)
  const lastInstructionsRef = useRef(instructions)
  const normalizeLineBreaks = (value: string) => value.replace(/\r\n|\r|\u2028|\u2029/g, "\n")
  const escapeHtml = (value: string) =>
    value
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")

  const serializeInstructionsContent = (editor: HTMLElement) => {
    const clone = editor.cloneNode(true) as HTMLElement;
    const chips = clone.querySelectorAll('[data-file-path]');

    chips.forEach((chip) => {
      const path = chip.getAttribute('data-file-path');
      const fileId = chip.getAttribute('data-file-id');
      const filename = chip.getAttribute('data-filename') || path?.split('/').pop() || path;
      const id = fileId || path;
      chip.replaceWith(document.createTextNode(`[${filename}](file:${id})`));
    });

    clone.querySelectorAll("br").forEach((lineBreak) => {
      lineBreak.replaceWith(document.createTextNode("\n"));
    });

    clone.querySelectorAll("div, p").forEach((block) => {
      if (block.lastChild?.textContent?.endsWith("\n")) {
        return;
      }
      block.appendChild(document.createTextNode("\n"));
    });

    return normalizeLineBreaks((clone.textContent || ""))
      .replace(/\u200B/g, "")
      .replace(/\n{3,}/g, "\n\n")
      .replace(/\n$/, "");
  };

  const handleInstructionsInput = () => {
    const editor = instructionsRef.current;
    if (!editor) return;

    let text = serializeInstructionsContent(editor);

    if (text.length > MAX_INSTRUCTIONS_LENGTH) {
      text = text.substring(0, MAX_INSTRUCTIONS_LENGTH);

      let html = escapeHtml(text);

      html = html.replace(/\[([^\]]+)\]\(file:(?:\/\/)?([^)]+)\)/g, (match, filename, id) => {
        return createFileChipHTML(id, id, filename);
      });

      html = html.replace(/\n/g, "<br>");
      if (html.endsWith("<br>")) {
        html += "<br>";
      }

      editor.innerHTML = html;

      // Move cursor to the end
      const range = document.createRange();
      const sel = window.getSelection();
      range.selectNodeContents(editor);
      range.collapse(false);
      sel?.removeAllRanges();
      sel?.addRange(range);
    }

    lastInstructionsRef.current = text;
    setInstructions(text);

    fileMention.checkTrigger();
  };

  const fileMention = useFileMention(instructionsRef, containerRef, handleInstructionsInput, t);

  const handleInstructionsPaste = (e: React.ClipboardEvent<HTMLDivElement>) => {
    e.preventDefault();
    const text = normalizeLineBreaks(e.clipboardData.getData("text/plain"));

    const currentLength = lastInstructionsRef.current.length;
    const availableSpace = MAX_INSTRUCTIONS_LENGTH - currentLength;

    if (availableSpace <= 0) {
      return;
    }

    let textToInsert = text;
    if (text.length > availableSpace) {
      textToInsert = text.substring(0, availableSpace);
    }

    const editor = instructionsRef.current;
    if (editor) {
      editor.focus();
      const selection = window.getSelection();
      if (selection && selection.rangeCount > 0) {
        const range = selection.getRangeAt(0);
        range.deleteContents();
        const fragment = document.createDocumentFragment();
        const parts = textToInsert.split("\n");
        parts.forEach((part, index) => {
          fragment.appendChild(document.createTextNode(part));
          if (index < parts.length - 1) {
            fragment.appendChild(document.createElement("br"));
          }
        });
        const lastNode = fragment.lastChild;
        range.insertNode(fragment);
        if (lastNode) {
          const newRange = document.createRange();
          newRange.setStartAfter(lastNode);
          newRange.collapse(true);
          selection.removeAllRanges();
          selection.addRange(newRange);
        }
      } else {
        editor.innerHTML += escapeHtml(textToInsert).replace(/\n/g, "<br>");
      }
    }
    handleInstructionsInput();
  };

  // Handle click on delete button for chips
  useEffect(() => {
    const editor = instructionsRef.current;
    if (!editor) return;

    const handleClick = (e: MouseEvent) => {
      if (readOnly) return;
      const target = e.target as HTMLElement;
      const deleteBtn = target.closest('.file-chip-delete');
      if (deleteBtn) {
        e.preventDefault();
        e.stopPropagation();
        const chip = deleteBtn.closest('[data-file-path]');
        if (chip) {
          chip.remove();
          // Trigger input handling manually
          handleInstructionsInput();
        }
        return;
      }
    };

    editor.addEventListener('click', handleClick);
    return () => editor.removeEventListener('click', handleClick);
  }, [readOnly]);

  // Sync state -> DOM
  useEffect(() => {
    const editor = instructionsRef.current;
    if (!editor) return;

    if (instructions !== lastInstructionsRef.current) {
      if (!instructions) {
        editor.innerHTML = "";
      } else if (document.activeElement !== editor) {
        // Escape HTML to prevent XSS
        let html = escapeHtml(normalizeLineBreaks(instructions));

        // Restore canonical and legacy file links.
        html = html.replace(/\[([^\]]+)\]\(file:(?:\/\/)?([^)]+)\)/g, (match, filename, id) => {
          return createFileChipHTML(id, id, filename);
        });
        html = html.replace(/\n/g, "<br>");
        if (html.endsWith("<br>")) {
          html += "<br>";
        }

        editor.innerHTML = html;
      }
      lastInstructionsRef.current = instructions;
    }
  }, [instructions]);

  const fileInputRef = useRef<HTMLInputElement>(null)
  const previewTaskIdRef = useRef<number | null>(null)

  const resetPreviewSession = useCallback(() => {
    previewTaskIdRef.current = null
    closeFilePreview()
    dispatch({ type: "CLEAR_MESSAGES" })
    dispatch({ type: "SET_TRACE_EVENTS", payload: [] })
    dispatch({ type: "SET_STEPS", payload: [] })
    dispatch({ type: "SET_DAG_EXECUTION", payload: null })
    dispatch({ type: "SET_CURRENT_TASK", payload: null })
    dispatch({ type: "SET_HISTORY_LOADING", payload: false })
    setTaskId(null, { navigate: false })
  }, [closeFilePreview, dispatch, setTaskId])

  const invalidatePreviewTask = useCallback(() => {
    previewTaskIdRef.current = null
  }, [])

  useEffect(() => {
    resetPreviewSession()
    return () => {
      resetPreviewSession()
    }
  }, [resetPreviewSession])

  useEffect(() => {
    if (!previewTaskIdRef.current) {
      return
    }
    invalidatePreviewTask()
  }, [instructions, executionMode, selectedKbs, selectedSkills, selectedToolCategories, selectedMcpServers, modelConfig, invalidatePreviewTask])

  // Fetch Data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [kbRes, skillsRes, toolsRes, modelsRes, userDefaultsRes, mcpRes] = await Promise.all([
          apiRequest(`${getApiUrl()}/api/kb/collections`),
          apiRequest(`${getApiUrl()}/api/skills/`),
          apiRequest(`${getApiUrl()}/api/tools/available`),
          apiRequest(`${getApiUrl()}/api/models/?category=llm`),
          apiRequest(`${getApiUrl()}/api/models/user-default`),
          apiRequest(`${getApiUrl()}/api/mcp/servers`)
        ])

        if (kbRes.ok) {
          const kbData = await kbRes.json()
          setKbs(kbData.collections || [])
        }

        if (skillsRes.ok) {
          const skillsData = await skillsRes.json()
          console.log("Skills API response:", skillsData)
          setSkills(skillsData || [])
        } else {
          console.error("Skills API failed:", skillsRes.status, await skillsRes.text())
        }

        if (toolsRes.ok) {
          const toolsData = await toolsRes.json()
          // Filter only enabled tools
          setTools((toolsData.tools || []).filter((t: Tool) => t.enabled))
        }

        if (mcpRes.ok && !ownerScopedMcpRef.current) {
          // Skip when an admin cross-user view has taken over the MCP list
          // (loadAgent fetches the owner's servers); otherwise this mount fetch
          // could resolve last and clobber the owner list with the admin's own.
          const mcpData = await mcpRes.json()
          // Re-check after the await: loadAgent may have claimed the owner-scoped
          // list during the JSON parse, and we must not clobber it.
          if (!ownerScopedMcpRef.current) setMcpServers(mcpData || [])
        }

        let availableModels: Model[] = []
        if (modelsRes.ok) {
          availableModels = await modelsRes.json()
          setModels(availableModels || [])
        }

        if (userDefaultsRes.ok) {
          const userDefaults = await userDefaultsRes.json()

          // Set model config based on user defaults (only for new agent)
          if (!isEditMode) {
            const config: AgentModelConfig = {
              general: null,
              small_fast: null,
              visual: null,
              compact: null,
            }

            for (const m of userDefaults) {
              if (m.config_type === 'general') config.general = m.model.id
              else if (m.config_type === 'small_fast') config.small_fast = m.model.id
              else if (m.config_type === 'visual') config.visual = m.model.id
              else if (m.config_type === 'compact') config.compact = m.model.id
            }

            // Fallback: If no general model set, pick first available LLM
            if (!config.general && availableModels.length > 0) {
              // models endpoint was called with ?category=llm so these should be LLMs
              const firstLlm = availableModels[0]
              if (firstLlm) {
                config.general = firstLlm.id
              }
            }

            setModelConfig(config)
          }
        }
      } catch (error) {
        console.error("Failed to fetch data:", error)
      } finally {
        setIsInitialDataLoaded(true)
      }
    }

    fetchData()
  }, [])

  const refreshKbs = async () => {
    try {
      const kbRes = await apiRequest(`${getApiUrl()}/api/kb/collections`)
      if (kbRes.ok) {
        const kbData = await kbRes.json()
        setKbs(kbData.collections || [])
      }
    } catch (error) {
      console.error("Failed to refresh KBs:", error)
    }
  }

  // Load agent data in edit mode
  useEffect(() => {
    if (!isEditMode || !localAgentId) return

    // Discard results of a stale load if the effect re-runs (e.g. switching
    // agents) before its async fetches resolve, so a late response can't
    // clobber the current agent's state.
    let active = true

    const loadAgent = async () => {
      try {
        setLoadingAgent(true)
        const response = await apiRequest(`${getApiUrl()}/api/agents/${localAgentId}`)
        if (!active) return
        if (response.ok) {
          const agent = await response.json()
          if (!active) return
          setOriginalData(agent)
          setReadOnly(agent.can_edit === false)
          setName(agent.name || "")
          setDescription(agent.description || "")
          setInstructions(agent.instructions || "")
          setExecutionMode(agent.execution_mode || "balanced")
          setSuggestedPrompts(agent.suggested_prompts || [])
          setVisibility(agent.visibility === "admins" ? "admins" : "team")
          setOwnership(agent.team_id == null ? "personal" : "team")
          setSelectedKbs(agent.knowledge_bases || [])
          setSelectedSkills(agent.skills || [])

          // Legacy agents may still have an unassignable category saved —
          // never show or round-trip it.
          const rawToolCategories = agent.tool_categories || []
          setSelectedToolCategories(rawToolCategories.filter((c: string) => !c.startsWith('mcp:') && isAssignableToolCategory(c)))
          setSelectedMcpServers(rawToolCategories.filter((c: string) => c.startsWith('mcp:')).map((c: string) => c.replace('mcp:', '')))

          // Admin inspecting someone else's agent: the mount-time /api/mcp/servers
          // fetch returned the admin's own servers, so the owner's mcp: entries have
          // no matching row to render. Re-fetch scoped to the owner.
          if (user?.is_admin && agent.user_id != null && String(agent.user_id) !== String(user.id)) {
            ownerScopedMcpRef.current = true
            const ownerMcpRes = await apiRequest(`${getApiUrl()}/api/mcp/servers?user_id=${agent.user_id}`)
            if (!active) return
            if (ownerMcpRes.ok) {
              setMcpServers((await ownerMcpRes.json()) || [])
            }
          } else if (ownerScopedMcpRef.current) {
            // Switching back to a self-owned agent after an admin cross-user view:
            // reset the guard and reload the admin's own servers, since the
            // mount-time fetch runs only once and won't re-populate them.
            ownerScopedMcpRef.current = false
            const selfMcpRes = await apiRequest(`${getApiUrl()}/api/mcp/servers`)
            if (!active) return
            if (selfMcpRes.ok) {
              setMcpServers((await selfMcpRes.json()) || [])
            }
          }

          setLogoUrl(agent.logo_url || null)

          // Load models
          if (agent.models) {
            setModelConfig({
              general: agent.models.general || null,
              small_fast: agent.models.small_fast || null,
              visual: agent.models.visual || null,
              compact: agent.models.compact || null,
            })
          }
        } else if (response.status === 404) {
          setNotFound(true)
        }
      } catch (error) {
        console.error("Failed to load agent:", error)
      } finally {
        if (active) setLoadingAgent(false)
      }
    }

    loadAgent()
    return () => {
      active = false
    }
  }, [isEditMode, localAgentId, user?.id, user?.is_admin])

  // Load template data when template parameter is present
  useEffect(() => {
    if (!templateId || isEditMode) return

    const loadTemplate = async () => {
      try {
        setLoadingAgent(true)
        const response = await apiRequest(
          `${getApiUrl()}/api/templates/${templateId}`
        )
        if (response.ok) {
          const template = await response.json()
          setName(template.name || "")
          setDescription(template.description || "")
          setInstructions(template.agent_config?.instructions || "")
          setExecutionMode(template.agent_config?.execution_mode || "balanced")
          setSelectedSkills(template.agent_config?.skills || [])

          // Separate regular tools from MCP servers
          const allCategories = template.agent_config?.tool_categories || []
          setSelectedToolCategories(allCategories.filter((c: string) => !c.startsWith('mcp:') && isAssignableToolCategory(c)))

          const explicitlyConfiguredMcps = allCategories
            .filter((c: string) => c.startsWith('mcp:'))
            .map((c: string) => c.replace('mcp:', ''))

          // _enrich_template merges connections into tool_categories as mcp: entries, so
          // iterating both explicitlyConfiguredMcps and connections would add each
          // connection-backed server twice (raw name + resolved name). Seed the list with
          // only the explicitly configured MCPs that are NOT covered by connections (e.g.
          // custom MCP servers), then let the connections loop below resolve and add the rest.
          const connectionNames = (template.connections && Array.isArray(template.connections))
            ? template.connections.map((conn: any) => typeof conn === 'string' ? conn : conn.name).filter(Boolean)
            : []

          let connectedMcpApps: string[] = explicitlyConfiguredMcps.filter(
            (mcp: string) => !connectionNames.some((connName: string) => mcpNameMatches(mcp, connName))
          )

          // Use the template's 'connections' to figure out which MCP apps to select
          if (template.connections && Array.isArray(template.connections)) {
            template.connections.forEach((conn: any) => {
              const connName = typeof conn === 'string' ? conn : conn.name;
              if (!connName) return;

              // Find the actual server object to use its exact name, to avoid case mismatches
              const server = findMatchingMcpServer(mcpServers, connName)
              const finalName = server ? server.name : connName;
              if (!connectedMcpApps.some(existing => mcpNameMatches(existing, finalName) || mcpNameMatches(existing, connName))) {
                connectedMcpApps.push(finalName)
              }
            });
          }

          setTemplateRequirements({
            requiredSkills: template.agent_config?.skills || [],
            requiredToolCategories: allCategories.filter((c: string) => !c.startsWith('mcp:')),
            requiredMcpServers: connectedMcpApps,
            requiresKnowledgeBase: allCategories.includes("knowledge"),
          })
          setSelectedMcpServers(connectedMcpApps)
        }
      } catch (error) {
        console.error("Failed to load template:", error)
      } finally {
        setLoadingAgent(false)
      }
    }

    loadTemplate()
  }, [templateId, isEditMode, locale, mcpServers])

  useEffect(() => {
    if (!templateId || isEditMode) {
      setTemplateRequirements(null)
    }
  }, [templateId, isEditMode])

  // Convert kbs to MultiSelect options
  const kbOptions = (Array.isArray(kbs) ? kbs : []).map((kb) => ({
    value: kb.name,
    label: kb.name,
  }))

  // Convert skills to MultiSelect options
  const skillOptions = (Array.isArray(skills) ? skills : []).map((skill) => ({
    value: skill.name,
    label: skill.name,
    description: skill.description || skill.when_to_use || undefined,
  }))

  const modelHost = (model: Model): string => {
    const host = hostnameFromUrl(model.base_url)
    return host ? `${host} (${model.model_provider})` : model.model_provider
  }

  const modelOptions = [
    { value: "", label: "--" },
    ...(Array.isArray(models) ? models : []).map((model) => ({
      value: model.id.toString(),
      label: model.model_name,
      host: modelHost(model),
      description: model.model_id,
      isBuiltin: isBuiltinModel(model),
      isDefault: model.is_default,
      isSmallFast: model.is_small_fast,
      isVisual: model.is_visual,
      isCompact: model.is_compact,
    }))
  ]

  // Group tools by category for category selection. `mcp` has its own
  // server-level selector; unassignable categories never appear.
  const toolCategories = Array.from(
    new Set((Array.isArray(tools) ? tools : []).map(t => t.category).filter(c => c !== 'mcp' && isAssignableToolCategory(c)))
  ).sort()

  const toolCategoryOptions = toolCategories.map(category => {
    const toolsInCategory = (Array.isArray(tools) ? tools : []).filter(t => t.category === category)
    const categoryDesc = getCategoryDescription(category)
    return {
      value: category,
      label: getCategoryLabel(category),
      count: toolsInCategory.length,
      description: (categoryDesc ? `**${categoryDesc}**\n\n` : '') + `${toolsInCategory.map(t => t.name).join(', ')}`
    }
  })

  // Helper function for category descriptions
  function getCategoryDescription(category: string): string {
    const descriptions: Record<string, string> = {
      'basic': t('builds.configForm.tools.categoryDescriptions.basic'),
      'web_search': t('builds.configForm.tools.categoryDescriptions.webSearch'),
      'file': t('builds.configForm.tools.categoryDescriptions.file'),
      'vision': t('builds.configForm.tools.categoryDescriptions.vision'),
      'image': t('builds.configForm.tools.categoryDescriptions.image'),
      'video': t('builds.configForm.tools.categoryDescriptions.video'),
      'knowledge': t('builds.configForm.tools.categoryDescriptions.knowledge'),
      'mcp': t('builds.configForm.tools.categoryDescriptions.mcp'),
      'browser': t('builds.configForm.tools.categoryDescriptions.browser'),
      'ppt': t('builds.configForm.tools.categoryDescriptions.ppt'),
      'office': t('builds.configForm.tools.categoryDescriptions.office'),
      'special_image': t('builds.configForm.tools.categoryDescriptions.specialImage'),
      'database': t('builds.configForm.tools.categoryDescriptions.database'),
      'skill': t('builds.configForm.tools.categoryDescriptions.skill'),
    }
    return descriptions[category] || ""
  }

  // Helper function for category labels
  function getCategoryLabel(category: string): string {
    const labels: Record<string, string> = {
      'basic': t('builds.configForm.tools.categories.basic'),
      'web_search': t('builds.configForm.tools.categories.webSearch'),
      'file': t('builds.configForm.tools.categories.file'),
      'vision': t('builds.configForm.tools.categories.vision'),
      'image': t('builds.configForm.tools.categories.image'),
      'video': t('builds.configForm.tools.categories.video'),
      'knowledge': t('builds.configForm.tools.categories.knowledge'),
      'mcp': t('builds.configForm.tools.categories.mcp'),
      'browser': t('builds.configForm.tools.categories.browser'),
      'ppt': t('builds.configForm.tools.categories.ppt'),
      'office': t('builds.configForm.tools.categories.office'),
      'special_image': t('builds.configForm.tools.categories.specialImage'),
      'database': t('builds.configForm.tools.categories.database'),
      'skill': t('builds.configForm.tools.categories.skill'),
    }
    return labels[category] || category
  }

  const handlePreviewSendMessage = async (content: string, _config?: any, files?: File[]) => {
    try {
      // Check if general model is selected
      if (!modelConfig.general) {
        dispatch({
          type: "ADD_MESSAGE",
          payload: {
            id: `preview-error-${Date.now()}`,
            role: "assistant",
            content: t("builds.preview.errors.noModel"),
            timestamp: Date.now().toString(),
            isResult: true,
          }
        })
        return
      }

      let previewTaskId = previewTaskIdRef.current
      const processedFiles = (files || []).map(f => ({
        file_id: (f as any).file_id,
        name: f.name,
        size: f.size,
        type: f.type || ''
      }))

      let backendMessage = content
      if (!backendMessage.trim() && processedFiles.length > 0) {
        backendMessage = `Uploaded files: ${processedFiles.map(f => f.name).join(', ')}`
      }

      const finalToolCategories = [...selectedToolCategories]
      selectedMcpServers.forEach(server => {
        finalToolCategories.push(`mcp:${server}`)
      })

      if (!previewTaskId) {
        const response = await apiRequest(`${getApiUrl()}/api/chat/task/create`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            title: (backendMessage || "Build preview").slice(0, 80),
            description: backendMessage,
            llm_ids: [
              modelConfig.general ? String(modelConfig.general) : null,
              modelConfig.small_fast ? String(modelConfig.small_fast) : null,
              modelConfig.visual ? String(modelConfig.visual) : null,
              modelConfig.compact ? String(modelConfig.compact) : null,
            ],
            agent_config: {
              instructions,
              knowledge_bases: selectedKbs,
              skills: selectedSkills,
              tool_categories: finalToolCategories,
              is_preview: true,
              preview_agent_id: localAgentId && typeof localAgentId === 'string' ? parseInt(localAgentId) : null,
            },
            execution_mode: executionMode,
            is_visible: false,
          }),
        })

        if (!response.ok) {
          throw new Error(await response.text())
        }

        const taskData = await response.json()
        previewTaskId = Number(taskData.task_id)
        if (!Number.isFinite(previewTaskId)) {
          throw new Error("Preview task creation returned an invalid task id")
        }
        previewTaskIdRef.current = previewTaskId

        // Close any file preview opened from the previous preview task before switching context.
        closeFilePreview()
        setTaskId(previewTaskId, { navigate: false })
        dispatch({
          type: "SET_CURRENT_TASK",
          payload: {
            id: previewTaskId.toString(),
            title: taskData.title,
            description: taskData.description || backendMessage,
            status: taskData.status,
            createdAt: taskData.created_at,
            updatedAt: taskData.updated_at,
            modelId: taskData.model_id,
            smallFastModelId: taskData.small_fast_model_id,
            visualModelId: taskData.visual_model_id,
            compactModelId: taskData.compact_model_id,
            modelName: taskData.model_name || taskData.modelName,
            smallFastModelName: taskData.small_fast_model_name || taskData.smallFastModelName,
            visualModelName: taskData.visual_model_name,
            compactModelName: taskData.compact_model_name,
            executionMode: taskData.execution_mode,
            isDag: taskData.is_dag,
            agentId: taskData.agent_id,
            waitingQuestion: taskData.waiting_question,
            waitingInteractions: taskData.waiting_interactions,
          }
        })
        dispatch({ type: "TRIGGER_TASK_UPDATE" })
      }

      await sendMessage(backendMessage, { force: true, targetTaskId: previewTaskId }, files)
    } catch (error) {
      console.error("Preview failed:", error)
      dispatch({
        type: "ADD_MESSAGE",
        payload: {
          id: `preview-error-${Date.now()}`,
          role: "assistant",
          content: t("builds.preview.errors.requestFailed"),
          timestamp: Date.now().toString(),
          isResult: true,
        }
      })
    }
  }

  const handleLogoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setLogoFile(e.target.files[0])
    }
  }

  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        const result = reader.result as string
        resolve(result)
      }
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  }

  const isDirty = useMemo(() => {
    if (!originalData) return false

    // Helper to normalize arrays for comparison
    const normalize = (arr: any[]) => [...(arr || [])].sort().join(',')

    // Helper to normalize prompts (filter empty)
    const normalizePrompts = (arr: string[]) =>
      [...(arr || [])].filter(p => p.trim()).sort().join(',')

    // Compare basic fields
    if (name !== (originalData.name || "")) return true
    if ((description || "") !== (originalData.description || "")) return true
    if ((instructions || "") !== (originalData.instructions || "")) return true
    if (executionMode !== (originalData.execution_mode || "graph")) return true

    // Compare ownership / visibility
    const originalOwnership = originalData.team_id == null ? "personal" : "team"
    if (ownership !== originalOwnership) return true
    if (ownership === "team" && visibility !== (originalData.visibility === "admins" ? "admins" : "team")) return true

    // Compare logo
    if (logoFile) return true

    // Compare arrays
    if (normalizePrompts(suggestedPrompts) !== normalizePrompts(originalData.suggested_prompts)) return true
    if (normalize(selectedKbs) !== normalize(originalData.knowledge_bases)) return true
    if (normalize(selectedSkills) !== normalize(originalData.skills)) return true

    // Check MCP servers by extracting them from originalData.tool_categories
    const originalMcpServers = (originalData.tool_categories || [])
      .filter((c: string) => c.startsWith('mcp:'))
      .map((c: string) => c.replace('mcp:', ''))
    if (normalize(selectedMcpServers) !== normalize(originalMcpServers)) return true

    // Check non-MCP tool categories
    const nonMcpCategories = selectedToolCategories.filter(c => !c.startsWith('mcp:'))
    const originalNonMcpCategories = (originalData.tool_categories || []).filter((c: string) => !c.startsWith('mcp:'))
    if (normalize(nonMcpCategories) !== normalize(originalNonMcpCategories)) return true

    // Compare models
    const origModels = originalData.models || {}
    if ((modelConfig.general || null) !== (origModels.general || null)) return true
    if ((modelConfig.small_fast || null) !== (origModels.small_fast || null)) return true
    if ((modelConfig.visual || null) !== (origModels.visual || null)) return true
    if ((modelConfig.compact || null) !== (origModels.compact || null)) return true

    return false
  }, [name, description, instructions, executionMode, ownership, visibility, logoFile, suggestedPrompts, selectedKbs, selectedSkills, selectedToolCategories, selectedMcpServers, modelConfig, originalData])

  // After a successful save, align server-side ownership with the chosen control:
  // promote a personal agent to team (with visibility) or demote a team agent back
  // to personal. Returns the ownership fields to fold into originalData, or null if
  // a promote was blocked by unshared connectors (dialog opened for the retry).
  const reconcileOwnership = async (
    agentId: string,
    savedTeamId: unknown,
  ): Promise<{ team_id: number | null; visibility: string } | null> => {
    const wasTeam = savedTeamId != null

    if (ownership === "team" && !wasTeam) {
      const res = await apiRequest(`${getApiUrl()}/api/agents/${agentId}/promote-team`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ visibility }),
      })
      if (res.status === 422) {
        const body = await res.json().catch(() => ({}))
        const connectors = sanitizeUnsharedConnectors(body?.detail?.unshared_connectors)
        const knowledgeBases = sanitizeUnsharedKnowledgeBases(
          body?.detail?.unshared_knowledge_bases,
        )
        if (connectors.length > 0 || knowledgeBases.length > 0) {
          setUnsharedConnectors(connectors)
          setUnsharedKnowledgeBases(knowledgeBases)
          return null
        }
        throw new Error(body?.detail?.message ?? body?.detail ?? t("builds.editor.error.unknown"))
      }
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body?.detail?.message ?? body?.detail ?? t("builds.editor.error.unknown"))
      }
      const agent = await res.json().catch(() => ({}))
      toast.success(t("builds.configForm.promoteTeam.success"))
      return { team_id: agent?.team_id ?? null, visibility: agent?.visibility ?? visibility }
    }

    if (ownership === "personal" && wasTeam) {
      const res = await apiRequest(`${getApiUrl()}/api/agents/${agentId}/demote-personal`, {
        method: "POST",
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body?.detail?.message ?? body?.detail ?? t("builds.editor.error.unknown"))
      }
      return { team_id: null, visibility: "team" }
    }

    // No ownership change; keep saved values.
    return { team_id: (savedTeamId as number | null) ?? null, visibility }
  }

  // Share every listed connector, then retry the promote-team flow.
  const handleShareConnectorsAndContinue = async () => {
    if (!localAgentId) return
    setIsSharingConnectors(true)
    try {
      for (const c of unsharedConnectors) {
        if (c.reason === "unresolved" || c.id == null) continue
        const res = await apiRequest(
          `${getApiUrl()}/api/connectors/${c.type}/${c.id}/share`,
          { method: "POST" },
        )
        if (!res.ok) {
          const body = await res.json().catch(() => ({}))
          throw new Error(body?.detail?.message ?? body?.detail ?? t("builds.editor.error.unknown"))
        }
      }
      for (const kb of unsharedKnowledgeBases) {
        const res = await apiRequest(
          `${getApiUrl()}/api/knowledge-bases/${encodeURIComponent(kb.name)}/promote-team`,
          { method: "POST" },
        )
        if (!res.ok) {
          const body = await res.json().catch(() => ({}))
          throw new Error(body?.detail?.message ?? body?.detail ?? t("builds.editor.error.unknown"))
        }
      }
      setUnsharedConnectors([])
      setUnsharedKnowledgeBases([])
      const result = await reconcileOwnership(localAgentId, null)
      if (result) {
        setOriginalData((current: any) => ({ ...current, ...result }))
        pendingCreationDialogRef.current = false
        setShowSuccessDialog(true)
      }
    } catch (error) {
      console.error("Failed to share connectors:", error)
      toast.error(error instanceof Error ? error.message : t("builds.editor.error.unknown"))
    } finally {
      setIsSharingConnectors(false)
    }
  }

  const handleCancelShareConnectors = () => {
    // User declined to share: stay personal.
    setUnsharedConnectors([])
    setUnsharedKnowledgeBases([])
    setOwnership("personal")
    // If this share flow interrupted agent creation, the success dialog was
    // deferred; show it now so generated webhook secrets are still revealed.
    if (pendingCreationDialogRef.current) {
      pendingCreationDialogRef.current = false
      setShowSuccessDialog(true)
    }
  }

  const handleCreate = async () => {
    // Validation
    if (!name.trim()) {
      toast.error(t("builds.editor.validation.nameRequired"))
      return
    }

    if (!instructions.trim()) {
      toast.error(t("builds.editor.validation.instructionsRequired"))
      return
    }

    if (!modelConfig.general) {
      toast.error(t("builds.editor.validation.modelRequired"))
      return
    }

    let finalToolCategories = [...selectedToolCategories]
    if (selectedKbs.length > 0 && !finalToolCategories.includes("knowledge")) {
      finalToolCategories.push("knowledge")
    }

    // Add selected MCP servers back into tool_categories
    selectedMcpServers.forEach(server => {
      const connectedServer = findMatchingMcpServer(mcpServers, server)
      const connectedApp = findMatchingMcpApp(officialApps, server)
      finalToolCategories.push(`mcp:${connectedServer?.name || connectedApp?.name || server}`)
    })

    setIsCreating(true)

    try {
      // Convert logo to base64 if provided
      let logo_base64: string | undefined
      if (logoFile) {
        logo_base64 = await fileToBase64(logoFile)
      }

      const url = isEditMode && localAgentId
        ? `${getApiUrl()}/api/agents/${localAgentId}`
        : `${getApiUrl()}/api/agents`

      const method = isEditMode ? "PUT" : "POST"

      const response = await apiRequest(url, {
        method,
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: name.trim(),
          description: description.trim() || undefined,
          instructions: instructions.trim() || undefined,
          execution_mode: executionMode,
          suggested_prompts: suggestedPrompts.filter(p => p.trim()),
          models: modelConfig,
          knowledge_bases: selectedKbs,
          skills: selectedSkills,
          tool_categories: finalToolCategories,
          // Only meaningful for team agents; sending it for a personal agent
          // could trip the team-admin visibility guard on an ordinary save.
          // Ownership transitions go through promote-team/demote-personal.
          ...(ownership === "team" ? { visibility } : {}),
          logo_base64,
        }),
      })

      if (response.ok) {
        if (isEditMode) {
          const trimmedName = name.trim()
          const trimmedDesc = description.trim()
          const trimmedInstr = instructions.trim()
          const trimmedPrompts = suggestedPrompts.filter(p => p.trim())

          // Update local state to match saved data
          setName(trimmedName)
          setDescription(trimmedDesc)
          setInstructions(trimmedInstr)
          setSuggestedPrompts(trimmedPrompts)

          // Reconcile team/personal ownership (may open the share-connectors dialog).
          const ownershipResult = localAgentId
            ? await reconcileOwnership(localAgentId, originalData?.team_id)
            : { team_id: originalData?.team_id ?? null, visibility }

          // Update original data to reflect saved state
          setOriginalData({
            ...originalData,
            name: trimmedName,
            description: trimmedDesc || undefined,
            instructions: trimmedInstr || undefined,
            execution_mode: executionMode,
            suggested_prompts: trimmedPrompts,
            models: modelConfig,
            knowledge_bases: selectedKbs,
            skills: selectedSkills,
            tool_categories: finalToolCategories,
            ...(ownershipResult ?? {}),
          })
          setLogoFile(null)
          // Optional: Reload agent to get updated logo URL if needed, but avoiding it keeps it fast
        } else {
          const newAgent = await response.json()
          setCreatedAgent(newAgent)

          // Staged triggers (configured before the agent existed, #928) go
          // through the regular trigger API now that a real agent id exists.
          // Failures don't roll back the agent; failed configs are kept in
          // failedStagedTriggers for retry/discard instead of being dropped.
          if (stagedTriggers.length > 0) {
            const outcome = await createStagedTriggers(newAgent.id, stagedTriggers)
            outcome.failed.forEach((entry) => {
              toast.error(
                t("triggers.messages.stagedCreateFailed", {
                  name: entry.staged.name,
                  error: entry.error,
                }),
              )
            })
            setStagedTriggers([])
            setFailedStagedTriggers(outcome.failed)
            setCreatedWebhookSecrets(outcome.generatedSecrets)
            try {
              setTriggerSummary(await listAgentTriggers(newAgent.id))
            } catch (error) {
              console.error("Failed to refresh trigger summary:", error)
            }
          }

          // Newly created agents are always personal; promote if the user chose Team.
          const ownershipResult = await reconcileOwnership(newAgent.id.toString(), newAgent.team_id)
          setOriginalData({ ...newAgent, ...(ownershipResult ?? {}) })
          // Only confirm success once ownership resolved. If a promote was blocked
          // (the share-connectors dialog opened, ownershipResult is null), don't
          // stack a "created" dialog on top of it — but remember to show it when
          // that flow resolves, or generated webhook secrets would never surface.
          if (ownershipResult) {
            setShowSuccessDialog(true)
          } else {
            pendingCreationDialogRef.current = true
          }
          setLocalAgentId(newAgent.id.toString())

          // Silently update URL to include ID so refreshing works
          // We don't want to trigger a full navigation that might close the dialog or reset state if not handled carefully
          // But since we are setting state, a replace might be fine.
          // Let's use history API to be safe and avoid component remount
          window.history.pushState({}, '', `/build/${newAgent.id}`)

          // Also update internal state so "Edit Mode" logic kicks in effectively if we were to re-render
          // Note: agentId comes from searchParams which won't update until router.push/replace
          // But for the dialog purpose, we have what we need.
        }
      } else {
        const error: unknown = await response.json().catch(() => null)
        toast.error(
          getAgentUpdateErrorMessage(error, t("builds.editor.error.unknown"))
        )
      }
    } catch (error) {
      console.error("Failed to save agent:", error)
      toast.error(t("builds.editor.error.unknown"))
    } finally {
      setIsCreating(false)
    }
  }

  const handlePublish = async () => {
    if (!localAgentId) return

    setLoadingAgent(true)

    try {
      const response = await apiRequest(`${getApiUrl()}/api/agents/${localAgentId}/publish`, {
        method: "POST",
      })

      if (response.ok) {
        setOriginalData({
          ...originalData,
          status: "published",
        })
        toast.success(t("builds.editor.success.published"))
      } else {
        const error = await response.json()
        toast.error(error.detail || t("builds.editor.error.publishFailed"))
      }
    } catch (error) {
      console.error("Failed to publish agent:", error)
      toast.error(t("builds.editor.error.unknown"))
    } finally {
      setLoadingAgent(false)
    }
  }

  const handleUnpublish = async () => {
    if (!localAgentId) return

    setLoadingAgent(true)

    try {
      const response = await apiRequest(`${getApiUrl()}/api/agents/${localAgentId}/unpublish`, {
        method: "POST",
      })

      if (response.ok) {
        setOriginalData({
          ...originalData,
          status: "draft",
        })
        toast.success(t("builds.editor.success.unpublished"))
      } else {
        const error = await response.json()
        toast.error(error.detail || t("builds.editor.error.unpublishFailed"))
      }
    } catch (error) {
      console.error("Failed to unpublish agent:", error)
      toast.error(t("builds.editor.error.unknown"))
    } finally {
      setLoadingAgent(false)
    }
  }

  const handleDialogPublish = async () => {
    if (!createdAgent?.id) return

    setLoadingAgent(true)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/agents/${createdAgent.id}/publish`, {
        method: "POST",
      })

      if (response.ok) {
        toast.success(t("builds.editor.success.published"))
        setShowSuccessDialog(false)
        if (failedStagedTriggers.length === 0) {
          router.replace(`/build/${createdAgent.id}`)
        } else {
          navPendingRef.current = true
        }
      } else {
        const error = await response.json()
        toast.error(error.detail || t("builds.editor.error.publishFailed"))
      }
    } catch (error) {
      console.error("Failed to publish agent:", error)
      toast.error(t("builds.editor.error.unknown"))
    } finally {
      setLoadingAgent(false)
    }
  }

  const handleDialogClose = () => {
    setShowSuccessDialog(false)
    // router.replace remounts the builder and would drop the failed-trigger
    // retry list. The URL was already pushState'd to /build/{id} at creation,
    // so staying on this instance is safe while retries are pending; the
    // deferred navigation runs once the failed list drains.
    if (!createdAgent?.id) return
    if (failedStagedTriggers.length === 0) {
      router.replace(`/build/${createdAgent.id}`)
    } else {
      navPendingRef.current = true
    }
  }

  const handleOptimizeInstructions = async () => {
    if (!instructions.trim()) {
      toast.error(t("builds.editor.validation.instructionsRequired"))
      return
    }

    setIsOptimizing(true)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/agents/optimize-instructions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          instructions,
          model_id: modelConfig.general
        }),
      })

      if (response.ok) {
        const data = await response.json()
        setInstructions(data.optimized_instructions)
        toast.success(t("builds.configForm.instructions.optimizeSuccess"))
      } else {
        const error = await response.json()
        toast.error(error.detail || t("builds.configForm.instructions.optimizeError"))
      }
    } catch (error) {
      console.error("Failed to optimize instructions:", error)
      toast.error(t("builds.configForm.instructions.optimizeError"))
    } finally {
      setIsOptimizing(false)
    }
  }

  const isTemplateEntry = Boolean(templateId) && !isEditMode
  const isTemplateRequirementsPending = isTemplateEntry && (!isInitialDataLoaded || loadingAgent || !templateRequirements)
  const isTemplateBuildFlow = isTemplateEntry && !isTemplateRequirementsPending
  const templateMissingKb = Boolean(
    isTemplateBuildFlow &&
    templateRequirements?.requiresKnowledgeBase &&
    selectedKbs.length === 0
  )
  const templateMissingSkills = Boolean(
    isTemplateBuildFlow &&
    templateRequirements?.requiredSkills.some((skill) => !selectedSkills.includes(skill))
  )
  const templateMissingTools = Boolean(
    isTemplateBuildFlow &&
    templateRequirements?.requiredToolCategories.some((category) => !selectedToolCategories.includes(category))
  )
  const templateMissingMcp = Boolean(
    isTemplateBuildFlow &&
    templateRequirements?.requiredMcpServers.some((serverName) => {
      const isSelected = selectedMcpServers.some((name) => mcpNameMatches(name, serverName))
      const connectedServer = findMatchingMcpServer(mcpServers, serverName)
      const connectedApp = findMatchingMcpApp(officialApps, serverName)
      const isConnected = Boolean(connectedServer || connectedApp?.is_connected)
      return !isSelected || !isConnected
    })
  )
  const useTemplateSpecificHighlights =
    templateMissingKb || templateMissingSkills || templateMissingTools || templateMissingMcp
  const describeStepCompleted = Boolean(name.trim() && description.trim() && instructions.trim())
  const configStepCompleted = isTemplateRequirementsPending
    ? false
    : isTemplateBuildFlow
      ? !templateMissingKb && !templateMissingSkills && !templateMissingTools && !templateMissingMcp
      : (
        selectedKbs.length > 0 ||
        selectedSkills.length > 0 ||
        selectedToolCategories.length > 0 ||
        selectedMcpServers.length > 0
      )
  const previewStepCompleted = state.messages.some((message) => message.role === "user")
  const shouldHighlightConfigStep = !configStepCompleted
  const shouldHighlightKbSection = useTemplateSpecificHighlights ? templateMissingKb : shouldHighlightConfigStep
  const shouldHighlightSkillsSection = useTemplateSpecificHighlights ? templateMissingSkills : shouldHighlightConfigStep
  const shouldHighlightToolsSection = useTemplateSpecificHighlights ? templateMissingTools : shouldHighlightConfigStep
  const shouldHighlightConnectorSection = useTemplateSpecificHighlights ? templateMissingMcp : shouldHighlightConfigStep

  const buildSteps = [
    {
      key: "describe",
      label: t("builds.editor.stepGuide.describe"),
      status: describeStepCompleted ? "complete" : "current" as "complete" | "current" | "upcoming",
    },
    {
      key: "configure",
      label: t("builds.editor.stepGuide.configure"),
      status: configStepCompleted
        ? "complete"
        : describeStepCompleted
          ? "current"
          : "upcoming" as "complete" | "current" | "upcoming",
    },
    {
      key: "preview",
      label: t("builds.editor.stepGuide.preview"),
      status: previewStepCompleted
        ? "complete"
        : describeStepCompleted && configStepCompleted
          ? "current"
          : "upcoming" as "complete" | "current" | "upcoming",
    },
  ]
  const allStepsCompleted = buildSteps.every((step) => step.status === "complete")
  const shouldShowCompletedBanner = allStepsCompleted && !localAgentId

  const getStepStatusClasses = (status: "complete" | "current" | "upcoming") =>
    status === "complete"
      ? "border-green-500 bg-green-500 text-white"
      : status === "current"
        ? "border-primary bg-primary text-primary-foreground shadow-sm"
        : "border-border bg-background text-muted-foreground"

  const getStepLabelClasses = (status: "complete" | "current" | "upcoming") =>
    status === "complete"
      ? "text-green-700 dark:text-green-400"
      : status === "current"
        ? "text-foreground"
        : "text-muted-foreground"

  const getStepConnectorClasses = (status: "complete" | "current" | "upcoming") =>
    status === "complete" ? "bg-green-500" : status === "current" ? "bg-primary/40" : "bg-border"

  const getConfigSectionClasses = (highlight: boolean) =>
    cn(
      "space-y-2 transition-all duration-200",
      highlight && "rounded-xl border border-primary/30 bg-primary/5 p-4 shadow-sm"
    )

  const scrollToModelSection = () => {
    modelSectionRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    })
  }

  const generalModelLabel = useMemo(
    () => models.find((m) => m.id === modelConfig.general)?.model_name || "—",
    [models, modelConfig.general],
  )

  const connectorDisplayNames = useMemo(
    () =>
      selectedMcpServers.map((serverName) => {
        const connectedServer = findMatchingMcpServer(mcpServers, serverName)
        const matchingApp = findMatchingMcpApp(officialApps, serverName)
        return connectedServer?.name || matchingApp?.name || serverName
      }),
    [selectedMcpServers, mcpServers, officialApps],
  )

  const flowTriggerRows = useMemo(
    () =>
      ([
        { type: "webhook", titleKey: "triggers.cards.webhook.title", descKey: "triggers.cards.webhook.description" },
        { type: "scheduled", titleKey: "triggers.cards.scheduled.title", descKey: "triggers.cards.scheduled.description" },
        { type: "gmail", titleKey: "triggers.cards.gmail.title", descKey: "triggers.cards.gmail.description" },
      ] as const)
        .filter((item) => triggerStats[item.type].enabled > 0)
        .map((item) => ({ key: item.type as string, label: t(item.titleKey), description: t(item.descKey) })),
    [triggerStats, t],
  )

  const LeftPanel = (
    <div className="p-6 space-y-8 min-h-full bg-card/50">
      {/* Header moved to middle panel */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-[240px] flex-1">
          <h1 className="mb-0.5 text-[20px] font-bold break-words leading-tight">{name || t("builds.editor.header.title")}</h1>
          <p className="text-[13px] text-muted-foreground">{t("builds.editor.header.subtitle")}</p>
        </div>
        <div
          className="inline-flex shrink-0 items-center gap-0.5 rounded-lg border bg-muted p-0.5"
          title={t("builds.editor.flow.hintBar")}
        >
          <button
            type="button"
            onClick={() => setViewMode("config")}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition-colors",
              viewMode === "config" ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground",
            )}
          >
            <Settings2 className="h-3 w-3" />
            {t("builds.editor.viewTabs.configure")}
          </button>
          <button
            type="button"
            onClick={() => setViewMode("flow")}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition-colors",
              viewMode === "flow" ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground",
            )}
          >
            <Workflow className="h-3 w-3" />
            {t("builds.editor.viewTabs.flow")}
          </button>
        </div>
        <div className="flex max-w-full flex-1 flex-wrap items-center justify-end gap-2 xl:gap-4">
          {readOnly ? (
            <div className="flex items-center gap-2 rounded-full border border-amber-300 bg-amber-50 px-3 py-1.5 text-sm font-medium text-amber-700 dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-400">
              <Eye className="h-4 w-4" />
              {t("builds.editor.header.readOnly")}
            </div>
          ) : (
            <>
              <Button
                variant="outline"
                className={cn(
                  "flex items-center gap-2 transition-colors",
                  showAIAssistant
                    ? "bg-primary/10 text-primary border-primary/20 hover:bg-primary/20 hover:text-primary"
                    : "text-muted-foreground hover:text-foreground"
                )}
                onClick={() => setShowAIAssistant(!showAIAssistant)}
              >
                <Bot className="h-4 w-4" />
                {t("builds.editor.aiAssistant", { appName: branding.appName })}
              </Button>

              <Button
                onClick={handleCreate}
                disabled={isCreating || loadingAgent || (isEditMode && !isDirty)}
              >
                {isCreating
                  ? isEditMode
                    ? t("builds.editor.header.updating")
                    : t("builds.editor.header.creating")
                  : isEditMode
                    ? t("builds.editor.header.update")
                    : t("builds.editor.header.create")}
              </Button>

              {isEditMode && (
                originalData?.status === "published" ? (
                  <Button
                    variant="outline"
                    onClick={handleUnpublish}
                    disabled={isCreating || loadingAgent}
                  >
                    {t("builds.editor.header.unpublish")}
                  </Button>
                ) : (
                  <Button
                    variant="secondary"
                    onClick={handlePublish}
                    disabled={isCreating || loadingAgent || isDirty}
                  >
                    {t("builds.editor.header.publish")}
                  </Button>
                )
              )}
            </>
          )}
        </div>
      </div>

      {/* Step guide stays interactive in read-only mode: it only scrolls. Only relevant to the Configure form. */}
      {viewMode === "config" && (
        <div className="rounded-xl border border-primary/15 bg-primary/5 px-4 py-4">
          <div className="mb-3 text-sm font-semibold text-primary">
            {t("builds.editor.stepGuide.title")}
          </div>
          <div className="overflow-x-auto pb-1">
            <div className="flex min-w-max items-center gap-3">
              {buildSteps.map((step, index) => (
                <React.Fragment key={step.key}>
                  <button
                    type="button"
                    className={cn(
                      "flex items-center gap-3 rounded-md transition-colors",
                      step.key === "configure" && "cursor-pointer hover:bg-primary/5 px-1 py-1"
                    )}
                    onClick={step.key === "configure" ? scrollToModelSection : undefined}
                  >
                    <div
                      className={cn(
                        "flex h-7 w-7 shrink-0 items-center justify-center rounded-full border text-xs font-semibold",
                        getStepStatusClasses(step.status)
                      )}
                    >
                      {step.status === "complete" ? <Check className="h-4 w-4" /> : index + 1}
                    </div>
                    <span className={cn("text-sm font-medium whitespace-nowrap", getStepLabelClasses(step.status))}>
                      {step.label}
                    </span>
                  </button>
                  {index < buildSteps.length - 1 && (
                    <div className={cn("h-px min-w-10 flex-1", getStepConnectorClasses(step.status))} />
                  )}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
      )}

      {viewMode === "config" && shouldShowCompletedBanner && (
        <div className="rounded-lg border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700 dark:border-green-900/40 dark:bg-green-900/20 dark:text-green-400">
          <div className="flex items-center gap-2">
            <Check className="h-4 w-4 shrink-0" />
            <span>{t("builds.editor.stepGuide.completed")}</span>
          </div>
        </div>
      )}

      <fieldset disabled={readOnly} className={viewMode === "flow" ? "hidden" : "contents"}>
      <div className="space-y-6">
        {/* Logo Upload */}
        <div className="space-y-2">
          <Label>{t("builds.configForm.logo.label")}</Label>
          <div className="flex items-center gap-4">
            <div
              className={`h-16 w-16 rounded-lg border border-dashed border-muted-foreground/50 flex items-center justify-center bg-background overflow-hidden transition-colors ${readOnly ? "cursor-default" : "cursor-pointer hover:bg-muted/50"}`}
              onClick={readOnly ? undefined : () => fileInputRef.current?.click()}
            >
              {logoFile ? (
                <img src={URL.createObjectURL(logoFile)} alt="Logo" className="h-full w-full object-cover" />
              ) : logoUrl ? (
                <img src={`${getApiUrl()}${logoUrl}`} alt="Logo" className="h-full w-full object-cover" />
              ) : (
                <Upload className="h-6 w-6 text-muted-foreground" />
              )}
            </div>
            <input
              type="file"
              accept="image/*"
              className="hidden"
              ref={fileInputRef}
              onChange={handleLogoUpload}
            />
          </div>
        </div>

        {/* Name */}
        <div className="space-y-2">
          <Label htmlFor="name">
            {t("builds.configForm.name.label")} <span className="text-destructive">*</span>
          </Label>
          <Input
            id="name"
            placeholder={t("builds.configForm.name.placeholder")}
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>

        {/* Description */}
        <div className="space-y-2">
          <Label htmlFor="description">{t("builds.configForm.description.label")}</Label>
          <Textarea
            id="description"
            placeholder={t("builds.configForm.description.placeholder")}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
        </div>

        {/* Instructions */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="instructions">
              {t("builds.configForm.instructions.label")} <span className="text-destructive">*</span>
            </Label>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-xs text-muted-foreground hover:text-primary"
              onClick={handleOptimizeInstructions}
              disabled={isOptimizing || !instructions.trim()}
            >
              {isOptimizing ? (
                <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
              ) : (
                <Sparkles className="mr-1.5 h-3.5 w-3.5" />
              )}
              {isOptimizing ? t("builds.configForm.instructions.optimizing") : t("builds.configForm.instructions.optimize")}
            </Button>
          </div>
          <div className="relative" ref={containerRef}>
            <FileMentionDropdown
              show={fileMention.showFilePicker}
              isLoading={fileMention.isLoadingFiles}
              filteredFiles={fileMention.filteredFiles}
              selectedFileIndex={fileMention.selectedFileIndex}
              onInsert={fileMention.insertFile}
              t={t}
              position={fileMention.dropdownPosition}
            />

            <div
              className={cn(
                "relative rounded-md border shadow-sm transition-all duration-300 bg-background",
                isInstructionsFocused ? "border-primary ring-1 ring-primary" : "border-input hover:border-border",
                isOptimizing ? "opacity-50 pointer-events-none" : ""
              )}
            >
              <div
                ref={instructionsRef}
                contentEditable={!isOptimizing && !readOnly}
                className="h-[220px] min-h-[150px] max-h-[520px] w-full rounded-md bg-transparent px-3 py-2 font-mono text-sm outline-none overflow-y-auto resize-y break-words whitespace-pre-wrap text-left"
                onInput={handleInstructionsInput}
                onKeyDown={fileMention.handleKeyDown}
                onPaste={handleInstructionsPaste as any}
                onFocus={() => setIsInstructionsFocused(true)}
                onBlur={() => setIsInstructionsFocused(false)}
                role="textbox"
                aria-multiline="true"
              />
              {!instructions && (
                <div className="absolute top-2 left-3 text-muted-foreground pointer-events-none text-sm font-mono">
                  {t("builds.configForm.instructions.placeholder")}
                </div>
              )}
            </div>
            {instructions.length >= MAX_INSTRUCTIONS_LENGTH && (
              <div className="flex items-center gap-2 mt-2 text-destructive bg-destructive/10 px-3 py-2 rounded-md text-sm">
                <XCircle className="h-4 w-4" />
                <span>{t("builds.configForm.instructions.maxLengthExceeded")}</span>
              </div>
            )}
          </div>
        </div>

        {/* Execution Mode */}
        <div className="space-y-2">
          <Label>{t("builds.configForm.executionMode.label")}</Label>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-2">
            <button
              type="button"
              className={`px-3 py-2 text-sm border rounded-md transition-colors ${executionMode === "flash"
                ? "bg-primary text-primary-foreground border-primary"
                : "bg-background hover:bg-accent"
                }`}
              onClick={() => setExecutionMode("flash")}
            >
              <div className="flex items-center justify-center gap-1 mb-1">
                <Zap className="h-3.5 w-3.5" />
                <div className="font-medium">{t("builds.configForm.executionMode.flash.title")}</div>
              </div>
              <div className="text-xs opacity-80">{t("builds.configForm.executionMode.flash.description")}</div>
            </button>
            <button
              type="button"
              className={`px-3 py-2 text-sm border rounded-md transition-colors ${executionMode === "balanced"
                ? "bg-primary text-primary-foreground border-primary"
                : "bg-background hover:bg-accent"
                }`}
              onClick={() => setExecutionMode("balanced")}
            >
              <div className="flex items-center justify-center gap-1 mb-1">
                <Gauge className="h-3.5 w-3.5" />
                <div className="font-medium">{t("builds.configForm.executionMode.balanced.title")}</div>
              </div>
              <div className="text-xs opacity-80">{t("builds.configForm.executionMode.balanced.description")}</div>
            </button>
            <button
              type="button"
              className={`px-3 py-2 text-sm border rounded-md transition-colors ${executionMode === "think"
                ? "bg-primary text-primary-foreground border-primary"
                : "bg-background hover:bg-accent"
                }`}
              onClick={() => setExecutionMode("think")}
            >
              <div className="flex items-center justify-center gap-1 mb-1">
                <Brain className="h-3.5 w-3.5" />
                <div className="font-medium">{t("builds.configForm.executionMode.think.title")}</div>
              </div>
              <div className="text-xs opacity-80">{t("builds.configForm.executionMode.think.description")}</div>
            </button>
          </div>
        </div>

        {/* Model Selection */}
        <div ref={modelSectionRef} className="space-y-4">
          <div className="flex items-center justify-between">
            <Label>{t("builds.configForm.model.label")}</Label>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-xs text-muted-foreground hover:text-foreground"
              onClick={() => setIsModelConfigOpen(true)}
            >
              <Settings2 className="h-3.5 w-3.5 md:mr-1.5" />
              <span className="hidden md:inline">{t("builds.configForm.model.configure")}</span>
            </Button>
          </div>

          {models.length > 0 ? (
            <div className="space-y-1">
              <div className="flex items-center gap-1.5">
                <Label className="text-xs text-muted-foreground">
                  {t("builds.configForm.model.types.general")}
                </Label>
                <InfoTooltip content={t("builds.configForm.model.tips.general")} />
              </div>
              <Select
                value={modelConfig.general?.toString() || ""}
                disabled={readOnly}
                onValueChange={(value) => setModelConfig(prev => ({
                  ...prev,
                  general: value ? Number(value) : null
                }))}
                options={modelOptions}
                placeholder="--"
              />
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">
              {t("builds.configForm.model.noData")}
            </div>
          )}

          <Dialog open={isModelConfigOpen} onOpenChange={setIsModelConfigOpen}>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>{t("builds.configForm.model.configure")}</DialogTitle>
                <DialogDescription className="flex items-center gap-1.5">
                  {t("builds.configForm.model.configureDescription")}
                  <a
                    href="https://docs.xagent.co/models/overview"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center text-muted-foreground hover:text-primary transition-colors"
                    title="View Documentation"
                  >
                    <BookOpen className="h-3.5 w-3.5" />
                  </a>
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                {/* Small & Fast Model */}
                <div className="space-y-1">
                  <div className="flex items-center gap-1.5">
                    <Label className="text-xs text-muted-foreground">
                      {t("builds.configForm.model.types.smallFast")}
                    </Label>
                    <InfoTooltip content={t("builds.configForm.model.tips.smallFast")} />
                  </div>
                  <Select
                    value={modelConfig.small_fast?.toString() || ""}
                    disabled={readOnly}
                    onValueChange={(value) => setModelConfig(prev => ({
                      ...prev,
                      small_fast: value ? Number(value) : null
                    }))}
                    options={modelOptions}
                    placeholder="--"
                  />
                </div>

                {/* Visual Model */}
                <div className="space-y-1">
                  <div className="flex items-center gap-1.5">
                    <Label className="text-xs text-muted-foreground">
                      {t("builds.configForm.model.types.visual")}
                    </Label>
                    <InfoTooltip content={t("builds.configForm.model.tips.visual")} />
                  </div>
                  <Select
                    value={modelConfig.visual?.toString() || ""}
                    disabled={readOnly}
                    onValueChange={(value) => setModelConfig(prev => ({
                      ...prev,
                      visual: value ? Number(value) : null
                    }))}
                    options={modelOptions}
                    placeholder="--"
                  />
                </div>

                {/* Compact Model */}
                <div className="space-y-1">
                  <div className="flex items-center gap-1.5">
                    <Label className="text-xs text-muted-foreground">
                      {t("builds.configForm.model.types.compact")}
                    </Label>
                    <InfoTooltip content={t("builds.configForm.model.tips.compact")} />
                  </div>
                  <Select
                    value={modelConfig.compact?.toString() || ""}
                    disabled={readOnly}
                    onValueChange={(value) => setModelConfig(prev => ({
                      ...prev,
                      compact: value ? Number(value) : null
                    }))}
                    options={modelOptions}
                    placeholder="--"
                  />
                </div>
              </div>
              <DialogFooter>
                <Button onClick={() => setIsModelConfigOpen(false)}>
                  {t("common.confirm")}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>

        {/* Knowledge Base - Multi Select */}
        <div className={getConfigSectionClasses(shouldHighlightKbSection)}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Label>{t("builds.configForm.knowledgeBase.label")}</Label>
              <InfoTooltip content={t("builds.configForm.model.tips.knowledgeBase")} />
              {kbs.length > 0 && (
                <div className="ml-2 flex items-center gap-1.5 border-l pl-2 border-border">
                  <Switch
                    id="selectAllKbs"
                    checked={selectedKbs.length === kbOptions.length && kbOptions.length > 0}
                    onCheckedChange={(checked: boolean) => {
                      if (checked) {
                        const allValues = kbOptions.map((item) => item.value)
                        setSelectedKbs(allValues)
                        if (!selectedToolCategories.includes("knowledge")) {
                          setSelectedToolCategories(prev => [...prev, "knowledge"])
                        }
                      } else {
                        setSelectedKbs([])
                      }
                    }}
                    className="scale-75"
                  />
                  <Label htmlFor="selectAllKbs" className="text-xs text-muted-foreground cursor-pointer">
                    {t("builds.configForm.knowledgeBase.selectAll")}
                  </Label>
                </div>
              )}
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-xs text-muted-foreground hover:text-foreground"
              onClick={() => setIsKbModalOpen(true)}
            >
              <PlusCircle className="h-4 w-4 md:mr-2" />
              <span className="hidden md:inline">{t("builds.configForm.knowledgeBase.create")}</span>
            </Button>
          </div>

          <MultiSelect
            values={selectedKbs || []}
            onValuesChange={(newValues) => {
              setSelectedKbs(newValues)
              if (newValues.length > 0 && !selectedToolCategories.includes("knowledge")) {
                setSelectedToolCategories(prev => [...prev, "knowledge"])
              }
            }}
            options={kbOptions}
            placeholder={t("builds.configForm.knowledgeBase.placeholder")}
            disabled={readOnly}
          />
        </div>

        {/* Skills - Multi Select */}
        <div className={getConfigSectionClasses(shouldHighlightSkillsSection)}>
          <div className="flex items-center gap-1.5">
            <Label>{t("builds.configForm.skills.label")}</Label>
            <InfoTooltip content={t("builds.configForm.model.tips.skills")} />
            {skills.length > 0 && (
              <div className="ml-2 flex items-center gap-1.5 border-l pl-2 border-border">
                <Switch
                  id="selectAllSkills"
                  checked={selectedSkills.length === skillOptions.length && skillOptions.length > 0}
                  onCheckedChange={(checked: boolean) => {
                    if (checked) {
                      const allValues = skillOptions.map((item: any) => item.value)
                      setSelectedSkills(allValues)
                    } else {
                      setSelectedSkills([])
                    }
                  }}
                  className="scale-75"
                />
                <Label htmlFor="selectAllSkills" className="text-xs text-muted-foreground cursor-pointer">
                  {t("builds.configForm.skills.selectAll")}
                </Label>
              </div>
            )}
          </div>
          {skills.length > 0 ? (
            <MultiSelect
              values={selectedSkills || []}
              onValuesChange={setSelectedSkills}
              options={skillOptions}
              placeholder={t("builds.configForm.skills.placeholder")}
              disabled={readOnly}
            />
          ) : (
            <div className="text-sm text-muted-foreground">
              {t("builds.configForm.skills.noData")}
            </div>
          )}
        </div>

        {/* Tools - Multi Select by Category */}
        <div className={getConfigSectionClasses(shouldHighlightToolsSection)}>
          <div className="flex items-center gap-1.5">
            <Label>{t("builds.configForm.tools.label")}</Label>
            <InfoTooltip content={t("builds.configForm.model.tips.tools")} />
            {toolCategories.length > 0 && (
              <div className="ml-2 flex items-center gap-1.5 border-l pl-2 border-border">
                <Switch
                  id="selectAllTools"
                  checked={selectedToolCategories.length === toolCategoryOptions.length && toolCategoryOptions.length > 0}
                  onCheckedChange={(checked: boolean) => {
                    if (checked) {
                      const allValues = toolCategoryOptions.map((item: any) => item.value)
                      setSelectedToolCategories(allValues)
                    } else {
                      setSelectedToolCategories([])
                    }
                  }}
                  className="scale-75"
                />
                <Label htmlFor="selectAllTools" className="text-xs text-muted-foreground cursor-pointer">
                  {t("builds.configForm.tools.selectAll")}
                </Label>
              </div>
            )}
          </div>
          {toolCategories.length > 0 ? (
            <MultiSelect
              values={selectedToolCategories || []}
              onValuesChange={setSelectedToolCategories}
              options={toolCategoryOptions}
              placeholder={t("builds.configForm.tools.placeholder")}
              disabled={readOnly}
            />
          ) : (
            <div className="text-sm text-muted-foreground">
              {t("builds.configForm.tools.noData")}
            </div>
          )}
          {selectedToolCategories.length > 0 && (
            <div className="text-xs text-muted-foreground">
              {t("builds.configForm.tools.selectedCount", {
                count: selectedToolCategories.length,
                tools: tools.filter(t => selectedToolCategories.includes(t.category)).length
              })}
            </div>
          )}
        </div>

        {failedStagedTriggers.length > 0 && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>{t("triggers.staging.failedTitle")}</AlertTitle>
            <AlertDescription>
              <div className="mt-2 space-y-2">
                {failedStagedTriggers.map((entry) => (
                  <div key={entry.staged.clientId} className="flex flex-wrap items-center gap-2">
                    <span className="min-w-0 flex-1 truncate text-xs">
                      <span className="font-medium">{entry.staged.name}</span>
                      {` — ${entry.error}`}
                    </span>
                    <Button
                      type="button"
                      size="sm"
                      variant="secondary"
                      className="h-7 px-2 text-xs"
                      onClick={() => void retryFailedTrigger(entry.staged.clientId)}
                    >
                      {t("triggers.actions.retry")}
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant={
                        discardConfirmClientId === entry.staged.clientId
                          ? "destructive"
                          : "ghost"
                      }
                      className="h-7 px-2 text-xs"
                      onClick={() => discardFailedTrigger(entry.staged.clientId)}
                    >
                      {discardConfirmClientId === entry.staged.clientId
                        ? t("triggers.actions.confirmDiscard")
                        : t("triggers.actions.discard")}
                    </Button>
                  </div>
                ))}
              </div>
            </AlertDescription>
          </Alert>
        )}

        {/* One-time reveal for webhook secrets generated outside the creation
            success dialog (e.g. a retried staged trigger). Hidden while the
            dialog is open — it shows the same list itself. */}
        {!showSuccessDialog && (
          <WebhookSecretsAlert secrets={createdWebhookSecrets} onDismiss={dismissWebhookSecrets} />
        )}

        {effectiveTriggerSummary.some((trigger) => trigger.enabled) && (
          <div className="space-y-2">
            <div className="flex items-center gap-1.5">
              <Label>{t("triggers.builder.title")}</Label>
              <InfoTooltip content={t("triggers.builder.tooltip")} />
            </div>
            <div className="grid gap-2 sm:grid-cols-2">
              {([
                {
                  type: "webhook" as const,
                  icon: Webhook,
                  title: t("triggers.cards.webhook.title"),
                  iconClass: "bg-fuchsia-50 text-fuchsia-600 dark:bg-fuchsia-950/40 dark:text-fuchsia-300",
                },
                {
                  type: "scheduled" as const,
                  icon: CalendarClock,
                  title: t("triggers.cards.scheduled.title"),
                  iconClass: "bg-amber-50 text-amber-600 dark:bg-amber-950/40 dark:text-amber-300",
                },
                {
                  type: "gmail" as const,
                  icon: Mail,
                  title: t("triggers.cards.gmail.title"),
                  iconClass: "bg-rose-50 text-rose-600 dark:bg-rose-950/40 dark:text-rose-300",
                },
              ])
                .filter((item) => triggerStats[item.type].enabled > 0)
                .map((item) => {
                  const stat = triggerStats[item.type]
                  const openDialog = () => {
                    setTriggerDialogInitialType(item.type)
                    setIsTriggersDialogOpen(true)
                  }
                  return (
                    <div
                      key={item.type}
                      className="flex min-w-0 items-center gap-2 rounded-md border border-primary/40 bg-primary/[0.03] px-2.5 py-2 text-left"
                    >
                      <button
                        type="button"
                        onClick={openDialog}
                        className="flex min-w-0 flex-1 items-center gap-2 text-left"
                      >
                        <div className={cn("flex h-6 w-6 shrink-0 items-center justify-center rounded", item.iconClass)}>
                          <item.icon className="h-4 w-4" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <span className="truncate text-sm font-medium">{item.title}</span>
                            <span className="rounded-full bg-emerald-50 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300">
                              {t("triggers.cards.activeCount", { count: stat.enabled })}
                            </span>
                          </div>
                        </div>
                      </button>
                      <Switch checked onCheckedChange={openDialog} className="scale-75" />
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={openDialog}
                        className="h-7 w-7 text-muted-foreground hover:text-foreground"
                        title={t("triggers.builder.configure")}
                      >
                        <Settings2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  )
                })}
            </div>
          </div>
        )}

        <div className={getConfigSectionClasses(shouldHighlightConnectorSection)}>
          <div className="flex items-center gap-1.5">
            <Label>{t("tools.mcp.dialog.connector")}</Label>
          </div>
          <div className="flex flex-col gap-2">
            {selectedMcpServers.map((serverName, index) => {
              const connectedServer = findMatchingMcpServer(mcpServers, serverName)
              const matchingApp = findMatchingMcpApp(officialApps, serverName)
              const isConnected = Boolean(connectedServer || matchingApp?.is_connected)
              const isSupported = Boolean(matchingApp)

              let statusDesc = ""

              if (connectedServer) {
                statusDesc = connectedServer.description || ""
              } else if (matchingApp?.is_connected) {
                statusDesc = matchingApp.description || ""
              } else if (isSupported) {
                statusDesc = t("tools.mcp.notConnected")
              } else {
                statusDesc = t("tools.mcp.notSupported")
              }

              const server = { name: connectedServer?.name || matchingApp?.name || serverName, description: statusDesc }
              const icon = getAppIcon(server.name)
              return (
                <div key={index} className={cn("flex items-center gap-3 p-2 rounded-md border", !isConnected && "opacity-50 bg-muted/50")}>
                  <div className="bg-slate-100 p-1.5 rounded">
                    {icon ? (
                      <img src={icon} alt={server.name} className={cn("h-5 w-5 object-contain", !isConnected && "grayscale")} />
                    ) : (
                      <span className="text-xl">🔌</span>
                    )}
                  </div>
                  <div>
                    <div className="text-sm font-medium flex items-center gap-2">
                      {server.name}
                      {!isConnected && (
                        <span className="text-[10px] bg-yellow-100 text-yellow-700 px-1.5 py-0.5 rounded font-normal whitespace-nowrap">
                          {t("tools.mcp.mcpUnavailable")}
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground">{server.description}</div>
                  </div>
                  <div className="ml-auto">
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-8 w-8 p-0 text-red-500 hover:text-red-600 hover:bg-red-50"
                      onClick={() => setSelectedMcpServers(prev => prev.filter(name => !mcpNameMatches(name, serverName)))}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Connector + Triggers — shared integration buttons row */}
        <div className="flex flex-wrap gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => setIsConnectMcpOpen(true)}
            className="h-7 border-dashed border-primary/45 bg-primary/5 px-2 text-xs text-primary hover:border-primary hover:bg-primary/10"
          >
            <PlusCircle className="mr-1 h-3.5 w-3.5" />
            {t('tools.mcp.dialog.connector')}
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => {
              setTriggerDialogInitialType(null)
              setIsTriggersDialogOpen(true)
            }}
            className="h-7 border-dashed border-primary/45 bg-primary/5 px-2 text-xs text-primary hover:border-primary hover:bg-primary/10"
          >
            <Zap className="mr-1 h-3.5 w-3.5" />
            {t("triggers.builder.open")}
          </Button>
        </div>

        {/* Suggested Prompts */}
        <div className="space-y-2">
          <Label>{t("builds.configForm.suggestedPrompts.label")}</Label>
          <div className="text-xs text-muted-foreground mb-2">
            {t("builds.configForm.suggestedPrompts.description")}
          </div>
          <div className="space-y-3">
            {(suggestedPrompts || []).map((prompt, index) => (
              <div key={index} className="flex gap-2 items-start">
                <Input
                  value={prompt}
                  onChange={(e) => {
                    const newPrompts = [...suggestedPrompts]
                    newPrompts[index] = e.target.value
                    setSuggestedPrompts(newPrompts)
                  }}
                  placeholder={t("builds.configForm.suggestedPrompts.placeholder", { index: index + 1 })}
                  className="flex-1"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    const newPrompts = suggestedPrompts.filter((_, i) => i !== index)
                    setSuggestedPrompts(newPrompts)
                  }}
                >
                  {t("builds.configForm.suggestedPrompts.delete")}
                </Button>
              </div>
            ))}
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setSuggestedPrompts([...suggestedPrompts, ""])}
            >
              {t("builds.configForm.suggestedPrompts.add")}
            </Button>
          </div>
        </div>

        {/* Ownership + team visibility (SaaS team context only) */}
        {inTeam && (
          <div className="space-y-2">
            <Label>{t("builds.configForm.ownership.label")}</Label>
            <div className="text-xs text-muted-foreground mb-2">
              {t("builds.configForm.ownership.hint")}
            </div>
            <SelectRadix value={ownership} onValueChange={(v) => setOwnership(v as "personal" | "team")}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="personal">{t("builds.configForm.ownership.personal")}</SelectItem>
                <SelectItem value="team">{t("builds.configForm.ownership.team")}</SelectItem>
              </SelectContent>
            </SelectRadix>

            {ownership === "team" && (
              <div className="space-y-2 pt-2">
                <Label>{t("builds.configForm.visibility.label")}</Label>
                <div className="text-xs text-muted-foreground mb-2">
                  {t("builds.configForm.visibility.desc")}
                </div>
                <SelectRadix value={visibility} onValueChange={(v) => setVisibility(v as "team" | "admins")}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="team">{t("builds.configForm.visibility.team")}</SelectItem>
                    {/* Admins-only is an admin-controlled setting; non-admins
                        never see the option (the backend also gates it). */}
                    {canSetAdminsOnly && (
                      <SelectItem value="admins">
                        {t("builds.configForm.visibility.admins")}
                      </SelectItem>
                    )}
                  </SelectContent>
                </SelectRadix>
              </div>
            )}
          </div>
        )}
      </div>
    </fieldset>

      {viewMode === "flow" && (
        <div className="flex flex-col overflow-hidden rounded-xl border bg-background" style={{ height: "min(76vh, 980px)" }}>
          <AgentFlowView
            name={name}
            modelLabel={generalModelLabel}
            executionMode={executionMode as "flash" | "balanced" | "think"}
            instructions={instructions}
            onInstructionsChange={setInstructions}
            readOnly={readOnly}
            maxInstructionsLength={MAX_INSTRUCTIONS_LENGTH}
            kbSelected={selectedKbs}
            kbOptions={kbOptions}
            onKbChange={(values) => {
              setSelectedKbs(values)
              if (values.length > 0 && !selectedToolCategories.includes("knowledge")) {
                setSelectedToolCategories((prev) => [...prev, "knowledge"])
              }
            }}
            skillsSelected={selectedSkills}
            skillOptions={skillOptions}
            onSkillsChange={setSelectedSkills}
            toolsSelected={selectedToolCategories}
            toolOptions={toolCategoryOptions}
            onToolsChange={setSelectedToolCategories}
            connectorNames={connectorDisplayNames}
            onOpenConnectors={() => setIsConnectMcpOpen(true)}
            triggerRows={flowTriggerRows}
            onOpenTriggers={() => {
              setTriggerDialogInitialType(null)
              setIsTriggersDialogOpen(true)
            }}
            promptCount={suggestedPrompts.length}
          />
        </div>
      )}
    </div>
  )

  const RightPanel = (
    <div className="flex flex-col flex-1 min-h-0 h-full bg-background border-l">
      <div className="h-14 border-b flex items-center px-4 gap-2 bg-card/30">
        <MessageSquare className="h-5 w-5 text-muted-foreground" />
        <span className="font-medium">{t("builds.preview.title")}</span>
        <div className={`ml-2 px-2 py-0.5 rounded-full text-xs font-medium flex items-center gap-1 transition-all duration-300 ${configSynced
          ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
          : "bg-muted text-muted-foreground"
          }`}>
          {configSynced ? <Check className="h-3 w-3" /> : <Zap className="h-3 w-3" />}
          <span>{configSynced ? t("builds.preview.synced") : t("builds.preview.live")}</span>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-muted-foreground hover:text-foreground"
            onClick={resetPreviewSession}
            title={t("common.clear") || "Clear"}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>
      <div className="flex-1 min-h-0">
        <TaskConversationPanel
          mode="embedded-preview"
          showTaskActions={true}
          showTokenUsage={false}
          showDagPreview={false}
          showTaskFiles={true}
          autoFocusInput={false}
          onSend={handlePreviewSendMessage}
        />
      </div>
    </div>
  )

  if (notFound) {
    return (
      <div className="flex h-full min-h-[calc(100dvh-4rem)] w-full flex-col items-center justify-center bg-background p-4 text-center">
        <Bot className="w-16 h-16 text-muted-foreground mb-4 opacity-20" />
        <h2 className="text-2xl font-bold mb-2">{t("builds.editor.error.notFound")}</h2>
        <p className="text-muted-foreground max-w-md mb-6">
          {t("builds.editor.error.notFoundDesc")}
        </p>
        <Button onClick={() => router.push("/build/new")}>
          {t("builds.editor.header.create")}
        </Button>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex-1 min-h-0 w-full overflow-y-auto md:overflow-hidden">
        <ResizableThreeColumnLayout
          showLeftPanel={showAIAssistant && !readOnly}
          leftPanel={<AgentBuilderChat
            agentConfig={{
              id: localAgentId ? parseInt(localAgentId) : undefined,
              name, description, instructions, executionMode, suggestedPrompts,
              modelConfig, selectedKbs, selectedSkills, selectedToolCategories
            }}
            onUpdateConfig={(updates) => {
              if (updates.id !== undefined) setLocalAgentId(updates.id.toString());
              if (updates.name !== undefined) setName(updates.name);
              if (updates.description !== undefined) setDescription(updates.description);
              if (updates.instructions !== undefined) setInstructions(updates.instructions);
              if (updates.executionMode !== undefined) setExecutionMode(updates.executionMode);
              if (updates.suggestedPrompts !== undefined) setSuggestedPrompts(updates.suggestedPrompts);
              if (updates.modelConfig !== undefined) setModelConfig(updates.modelConfig);
              if (updates.selectedKbs !== undefined) setSelectedKbs(updates.selectedKbs);
              if (updates.selectedSkills !== undefined) setSelectedSkills(updates.selectedSkills);
              if (updates.selectedToolCategories !== undefined) setSelectedToolCategories(updates.selectedToolCategories);
            }}
            availableOptions={{
              models: (Array.isArray(models) ? models : []).map(m => ({ id: m.id, name: m.model_name || m.model_id })),
              knowledgeBases: (Array.isArray(kbs) ? kbs : []).map(k => ({ name: k.name })),
              skills: (Array.isArray(skills) ? skills : []).map(s => ({ name: s.name })),
              toolCategories: Array.from(new Set((Array.isArray(tools) ? tools : []).map(t => t.category)))
            }}
          />}
          middlePanel={LeftPanel}
          rightPanel={RightPanel}
          initialLeftWidth={20}
          initialMiddleWidth={50}
          initialRightWidth={30}
          minLeftWidth={15}
          minMiddleWidth={45}
          minRightWidth={20}
        />
      </div>
      {/* Success Dialog */}
      <Dialog open={showSuccessDialog} onOpenChange={handleDialogClose}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("builds.editor.success.created")}</DialogTitle>
            <DialogDescription>
              {t("builds.editor.success.createdDesc", { name: createdAgent?.name })}
            </DialogDescription>
          </DialogHeader>
          {/* One-time reveal of webhook secrets generated for staged triggers
              (#928). Closing this dialog usually navigates away and remounts
              the builder, so this is the moment to copy them. */}
          <WebhookSecretsAlert secrets={createdWebhookSecrets} />
          <DialogFooter className="gap-2 sm:justify-end">
            <div className="flex w-full sm:w-auto gap-2 justify-end">
              <Button variant="outline" onClick={handleDialogClose}>
                {t("common.cancel")}
              </Button>
              <Button onClick={handleDialogPublish}>
                {t("builds.editor.header.publish")}
              </Button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Share-connectors-first prompt (promote-team 422) */}
      <Dialog
        open={unsharedConnectors.length > 0 || unsharedKnowledgeBases.length > 0}
        onOpenChange={(open) => { if (!open) handleCancelShareConnectors() }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("builds.configForm.connectorNotShared.title")}</DialogTitle>
            <DialogDescription>
              {t(
                hasUnresolvedConnectors
                  ? "builds.configForm.connectorNotShared.unresolvedDesc"
                  : "builds.configForm.connectorNotShared.desc",
              )}
            </DialogDescription>
          </DialogHeader>
          <ul className="list-disc pl-5 text-sm space-y-1">
            {unsharedConnectors.map((c) => (
              <li key={`${c.type}:${c.id ?? c.name}`}>
                {c.name}
                {c.reason === "unresolved" && ` — ${t("builds.configForm.connectorNotShared.unresolved")}`}
              </li>
            ))}
            {unsharedKnowledgeBases.map((kb) => (
              <li key={`kb:${kb.name}`}>{kb.name}</li>
            ))}
          </ul>
          <DialogFooter className="gap-2 sm:justify-end">
            <div className="flex w-full sm:w-auto gap-2 justify-end">
              <Button variant="outline" onClick={handleCancelShareConnectors} disabled={isSharingConnectors}>
                {t("builds.configForm.connectorNotShared.cancel")}
              </Button>
              {!hasUnresolvedConnectors && (
                <Button onClick={handleShareConnectorsAndContinue} disabled={isSharingConnectors}>
                  {isSharingConnectors && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                  {t("builds.configForm.connectorNotShared.shareAndContinue")}
                </Button>
              )}
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <KnowledgeBaseCreationDialog
        open={isKbModalOpen}
        onOpenChange={setIsKbModalOpen}
        onSuccess={(createdCollections) => {
          refreshKbs()
          if (createdCollections && createdCollections.length > 0) {
            setSelectedKbs(prev => {
              const newKbs = Array.from(new Set([...prev, ...createdCollections]))
              return newKbs
            })
            if (!selectedToolCategories.includes("knowledge")) {
              setSelectedToolCategories(prev => [...prev, "knowledge"])
            }
          }
        }}
      />

      <AgentTriggersDialog
        agentId={localAgentId ? Number(localAgentId) : null}
        staged={localAgentId ? null : { triggers: stagedTriggers, onChange: setStagedTriggers }}
        agentName={name}
        open={isTriggersDialogOpen}
        onOpenChange={(dialogOpen) => {
          setIsTriggersDialogOpen(dialogOpen)
          if (!dialogOpen) {
            setTriggerDialogInitialType(null)
            void refreshTriggerSummary()
          }
        }}
        onChanged={refreshTriggerSummary}
        initialType={triggerDialogInitialType}
        gmailConnection={gmailConnection}
        onConnectGmail={() => {
          setIsTriggersDialogOpen(false)
          setIsConnectMcpOpen(true)
        }}
      />

      {state.filePreview.isOpen && (
        <div className="absolute inset-y-0 right-0 z-50 w-full max-w-[720px] p-4 pointer-events-none">
          <div className="h-full pointer-events-auto">
            <BuildFilePreviewSheet />
          </div>
        </div>
      )}
      {!readOnly && !loadingAgent && (
        <ConnectMcpDialog
          open={isConnectMcpOpen}
          onOpenChange={setIsConnectMcpOpen}
          selectedMcpServers={selectedMcpServers}
          onConnectSelected={(selectedApps) => {
            setSelectedMcpServers(selectedApps)
          }}
          onSuccess={() => {
            refreshMcpApps().catch(console.error)
            apiRequest(`${getApiUrl()}/api/mcp/servers`)
              .then(res => res.json())
              .then(data => setMcpServers(data || []))
              .catch(console.error)
          }}
        />
      )}
    </div>
  )
}
