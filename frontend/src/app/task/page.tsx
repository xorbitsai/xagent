"use client";

import React, { useState, useEffect, useMemo, useRef } from "react";
import { Bot } from "lucide-react";
import { useI18n } from "@/contexts/i18n-context";
import { useApp } from "@/contexts/app-context-chat";
import { ChatStartScreen, AgentCard } from "@/components/chat/ChatStartScreen";
import { FilePreviewDialog } from "@/components/file/file-preview-dialog";
import { getBrandingFromEnv } from "@/lib/branding";
import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";
import { findRunnableAgentById } from "@/lib/agent-ui-access";
import { toAgentId } from "@/lib/template-agent-resolution";
import { categoryLabel } from "@/lib/template-categories";
import type { Template } from "@/types/template";
import { useSearchParams } from "next/navigation";
import { toast } from "sonner";

function TaskHomePageContent() {
  const { t, locale } = useI18n();
  const { sendMessage, state, dispatch, closeFilePreview } = useApp();
  const searchParams = useSearchParams();
  const starter = searchParams.get("starter");
  const promptFromQuery = searchParams.get("prompt");
  const agentFromQuery = searchParams.get("agent");
  const appliedAgentFromQueryRef = useRef<string | null>(null);
  // The exact prompt text auto-filled by the last "My Team" pick, so
  // deselecting only clears the composer when the user hasn't since typed
  // over it - never clobber a real edit.
  const loadedPromptRef = useRef<string | null>(null);

  const [files, setFiles] = useState<File[]>([]);
  const [agents, setAgents] = useState<AgentCard[]>([]);
  const [selectedAgents, setSelectedAgents] = useState<AgentCard[]>([]);
  const [selectedAgentConfig, setSelectedAgentConfig] = useState<{
    model?: string;
    executionMode?: "auto" | "flash" | "balanced" | "think";
  }>();
  const branding = getBrandingFromEnv();

  // Clear state on mount to ensure we are in "new task" mode
  useEffect(() => {
    dispatch({ type: "RESET_STATE" });
  }, [dispatch]);

  // Fetch agents on mount
  useEffect(() => {
    const fetchAgents = async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/agents`);
        if (response.ok) {
          const data = await response.json();
          setAgents(
            Array.isArray(data)
              ? data.filter(
                (agent) =>
                  agent &&
                  typeof agent === "object" &&
                  agent.status === "published"
              )
              : []
          );
        }
      } catch (error) {
        console.error("Failed to fetch agents:", error);
      }
    };
    fetchAgents();
  }, []);

  // Best-effort enrichment only: a hired agent traces back to the template
  // it came from via `template_id`, and this lookup supplies that
  // template's persona photo/category for the "My Team" picker row - a
  // scratch-built agent has no `template_id` and simply renders without
  // them.
  const [templatesById, setTemplatesById] = useState<Record<string, Template>>({});
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/templates/?lang=${locale}`);
        if (!response.ok || cancelled) return;
        const data: unknown = await response.json();
        if (cancelled) return;
        const map: Record<string, Template> = {};
        // A single malformed entry (missing/non-string `id`) must not drop
        // every other otherwise-valid template out of the map - skip just
        // that entry instead of letting the `for` loop throw and lose the
        // whole batch to the catch below.
        for (const template of Array.isArray(data) ? data : []) {
          if (template && typeof template === "object" && typeof template.id === "string") {
            map[template.id] = template;
          }
        }
        setTemplatesById(map);
      } catch {
        // Picker just renders without persona photos/specialty labels.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [locale]);

  const teammates = useMemo(
    () =>
      agents.map((agent) => {
        const template = agent.template_id ? templatesById[agent.template_id] : undefined;
        if (!template) return agent;
        // A hired agent's own `suggested_prompts` (from `agent_config`) is
        // almost never populated by our built-in templates - the marketplace
        // prompt shown for it lives on the template's `sample_prompts`
        // instead, so prefer that for the "My Team" auto-fill.
        const samplePrompt = template.sample_prompts?.[0]?.prompt;
        return {
          ...agent,
          persona_avatar: template.persona?.avatar,
          specialty: categoryLabel(t, template.category),
          suggested_prompts: samplePrompt ? [samplePrompt] : agent.suggested_prompts,
        };
      }),
    [agents, templatesById, t]
  );

  useEffect(() => {
    if (!agentFromQuery || appliedAgentFromQueryRef.current === agentFromQuery || teammates.length === 0) {
      return;
    }

    const selectedAgent = findRunnableAgentById(teammates, agentFromQuery);
    if (!selectedAgent) {
      return;
    }

    setSelectedAgents([selectedAgent]);
    appliedAgentFromQueryRef.current = agentFromQuery;
  }, [agentFromQuery, teammates]);

  useEffect(() => {
    let cancelled = false;

    const fetchSelectedAgentConfig = async () => {
      const selectedAgent = selectedAgents[0];
      const selectedAgentId = Number(selectedAgent?.id);
      if (Number.isNaN(selectedAgentId)) {
        setSelectedAgentConfig(undefined);
        return;
      }
      if (selectedAgent?.readonly === true || selectedAgent?.can_edit === false) {
        setSelectedAgentConfig(undefined);
        return;
      }

      try {
        const response = await apiRequest(`${getApiUrl()}/api/agents/${selectedAgentId}`);
        if (!response.ok) {
          if (!cancelled) {
            setSelectedAgentConfig(undefined);
          }
          return;
        }

        const data = await response.json();
        if (!cancelled) {
          setSelectedAgentConfig({
            model: data?.models?.general,
            executionMode: data?.execution_mode,
          });
        }
      } catch (error) {
        console.error("Failed to fetch selected agent config:", error);
        if (!cancelled) {
          setSelectedAgentConfig(undefined);
        }
      }
    };

    fetchSelectedAgentConfig();

    return () => {
      cancelled = true;
    };
  }, [selectedAgents]);

  // Deep-link preset for "?starter=presentation" (the Welcome modal's
  // Presentation Builder card). Only this one key is ever populated or read,
  // so a plain check is clearer here than a one-entry lookup map.
  const starterPreset = useMemo(() => {
    if (starter !== "presentation") return null;

    return {
      prompt: "Build a N-slide presentation on topic for audience.",
      highlights: ["N", "topic", "audience"],
    };
  }, [starter]);

  const queryInputValue = starterPreset?.prompt || promptFromQuery || "";
  const queryPromptHighlightTerms = useMemo(
    () => starterPreset?.highlights || [],
    [starterPreset]
  );

  const [inputValue, setInputValue] = useState(() => queryInputValue);
  const [promptHighlightTerms, setPromptHighlightTerms] = useState<string[]>(() => queryPromptHighlightTerms);

  useEffect(() => {
    setInputValue(queryInputValue);
    setPromptHighlightTerms(queryPromptHighlightTerms);
  }, [queryInputValue, queryPromptHighlightTerms]);

  const handleSend = async (message: string, filesToSend: File[], config?: any) => {
    if (state.isProcessing) return;

    const agentId = toAgentId(selectedAgents[0]);
    const nextConfig = {
      ...config,
      agentId: agentId ?? undefined,
    };

    try {
      // Use sendMessage from AppContext - it will create task and send files via WebSocket
      await sendMessage(message, nextConfig, filesToSend || files);

      setFiles([]);
      setInputValue("");
      setPromptHighlightTerms([]);
      setSelectedAgents([]);
      loadedPromptRef.current = null;
    } catch (error) {
      console.error("Failed to send message:", error);
      toast.error(error instanceof Error ? error.message : t("builds.list.chat.sendFailed"));
    }
  };

  const handlePromptSelect = (prompt: string, highlights?: string[]) => {
    setInputValue(prompt);
    setPromptHighlightTerms(highlights || []);
  };

  const handleInputChange = (value: string) => {
    setInputValue(value);
  };

  // Clears the composer only if it still holds exactly the prompt a "My
  // Team" pick auto-filled - never clobber whatever the user has since
  // typed, matching the reference behavior this pill row is modeled on.
  const clearAutoFilledPromptIfUnchanged = () => {
    if (loadedPromptRef.current !== null && inputValue === loadedPromptRef.current) {
      setInputValue("");
      setPromptHighlightTerms([]);
    }
    loadedPromptRef.current = null;
  };

  const handleRemoveSelectedAgent = (agentId: number | string) => {
    setSelectedAgents((prev) => prev.filter((agent) => agent.id !== agentId));
    clearAutoFilledPromptIfUnchanged();
  };

  // Picking a teammate assigns them as the task's lead and, when they carry
  // a suggested prompt, fills the composer with it - clicking the already-
  // selected pill again clears both (see clearAutoFilledPromptIfUnchanged).
  // Only one teammate can lead a task, so this always replaces rather than
  // appending to `selectedAgents`.
  const handleAgentClick = (agent: AgentCard) => {
    if (selectedAgents[0]?.id === agent.id) {
      setSelectedAgents([]);
      clearAutoFilledPromptIfUnchanged();
      return;
    }

    // Switching straight to a teammate with no suggested prompt of their
    // own must still clear whatever the previous teammate auto-filled -
    // otherwise their (unedited) prompt is stranded in the composer with
    // nothing tracking it, since `loadedPromptRef` is about to be
    // overwritten below regardless of which branch runs next.
    clearAutoFilledPromptIfUnchanged();

    setSelectedAgents([agent]);
    const prompt = agent.suggested_prompts?.[0];
    if (prompt) {
      setInputValue(prompt);
      setPromptHighlightTerms([]);
      loadedPromptRef.current = prompt;
    } else {
      loadedPromptRef.current = null;
    }
  };

  return (
    <div className="h-full bg-background flex flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto">
        <main className="container max-w-4xl mx-auto px-4 py-8">
          <ChatStartScreen
            title={t("chatPage.page.emptyTitle", { appName: branding.appName })}
            description={t("chatPage.page.emptyDescription")}
            icon={<Bot className="w-10 h-10 text-[hsl(var(--gradient-from))]" />}
            agents={teammates}
            onAgentClick={handleAgentClick}
            selectedAgents={selectedAgents}
            onRemoveSelectedAgent={handleRemoveSelectedAgent}
            onSend={handleSend}
            isSending={state.isProcessing}
            files={files}
            onFilesChange={setFiles}
            inputValue={inputValue}
            onInputChange={handleInputChange}
            onPromptSelect={handlePromptSelect}
            promptHighlightTerms={promptHighlightTerms}
            readOnlyConfig={selectedAgents.length > 0}
            taskConfig={selectedAgents.length > 0 ? selectedAgentConfig : undefined}
            showModeToggle={true}
            autoFocus={true}
            inputMinHeightClass="min-h-[200px]"
          />
        </main>
      </div>

      {/* File Preview Modal */}
      <FilePreviewDialog
        open={state.filePreview.isOpen}
        onOpenChange={(open) => {
          if (!open) closeFilePreview()
        }}
      />
    </div>
  );
}

export default TaskHomePageContent;
