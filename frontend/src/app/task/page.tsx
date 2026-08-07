"use client";

import { useState, useEffect, useMemo, useRef } from "react";
import { Bot, CheckCircle2 } from "lucide-react";
import { useI18n } from "@/contexts/i18n-context";
import { useApp } from "@/contexts/app-context-chat";
import { ChatStartScreen, AgentCard } from "@/components/chat/ChatStartScreen";
import { FilePreviewDialog } from "@/components/file/file-preview-dialog";
import { Button } from "@/components/ui/button";
import { getBrandingFromEnv } from "@/lib/branding";
import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";
import { findRunnableAgentById } from "@/lib/agent-ui-access";
import { resolveAgentForTemplate, toAgentId } from "@/lib/template-agent-resolution";
import { useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import type { Template, SamplePrompt } from "@/types/template";
import { FEATURED_CATEGORY_ID } from "@/lib/template-categories";

function TaskHomePageContent() {
  const { t, locale } = useI18n();
  const { sendMessage, state, dispatch, closeFilePreview } = useApp();
  const router = useRouter();
  const searchParams = useSearchParams();
  const starter = searchParams.get("starter");
  const promptFromQuery = searchParams.get("prompt");
  const agentFromQuery = searchParams.get("agent");
  const appliedAgentFromQueryRef = useRef<string | null>(null);

  const [files, setFiles] = useState<File[]>([]);
  const [agents, setAgents] = useState<AgentCard[]>([]);
  const [selectedAgents, setSelectedAgents] = useState<AgentCard[]>([]);
  const [selectedAgentConfig, setSelectedAgentConfig] = useState<{
    model?: string;
    executionMode?: "flash" | "balanced" | "think";
  }>();
  const [templates, setTemplates] = useState<Template[]>([]);
  const [templatesLoading, setTemplatesLoading] = useState(false);
  const [templatesError, setTemplatesError] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string>(FEATURED_CATEGORY_ID);
  const [selectedTemplate, setSelectedTemplate] = useState<{ id: string; name: string } | null>(null);
  const [selectedPromptKey, setSelectedPromptKey] = useState<string | null>(null);
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

  // Fetch templates on mount (and when locale changes) for the quick-access grid
  const fetchTemplates = async () => {
    setTemplatesLoading(true);
    setTemplatesError(false);
    try {
      const response = await apiRequest(`${getApiUrl()}/api/templates/?lang=${locale}`);
      if (response.ok) {
        const data = await response.json();
        // Quick-access resolves a template straight into a single published
        // agent (POST /api/agents/from-template/resolve) - a workforce
        // template has no single-agent config to resolve, so it's excluded
        // here rather than surfacing as a broken agent.
        const agentTemplates = Array.isArray(data)
          ? data.filter((template) => template?.type !== "workforce")
          : [];
        setTemplates(agentTemplates);
      } else {
        setTemplatesError(true);
      }
    } catch (error) {
      console.error("Failed to fetch templates:", error);
      setTemplatesError(true);
    } finally {
      setTemplatesLoading(false);
    }
  };

  useEffect(() => {
    fetchTemplates();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [locale]);

  useEffect(() => {
    if (!agentFromQuery || appliedAgentFromQueryRef.current === agentFromQuery || agents.length === 0) {
      return;
    }

    const selectedAgent = findRunnableAgentById(agents, agentFromQuery);
    if (!selectedAgent) {
      return;
    }

    setSelectedAgents([selectedAgent]);
    setSelectedTemplate(null);
    setSelectedPromptKey(null);
    appliedAgentFromQueryRef.current = agentFromQuery;
  }, [agentFromQuery, agents]);

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

    let agentId = toAgentId(selectedAgents[0]);

    if (agentId === null && selectedTemplate) {
      let result: { agent: AgentCard; created: boolean };
      try {
        result = await resolveAgentForTemplate(selectedTemplate.id);
      } catch (error) {
        console.error("Failed to create agent from template:", error);
        // Rethrow (translated) rather than swallowing: ChatInput's own
        // catch around onSend shows this as a toast, and - critically -
        // only clears the typed message/chip when onSend actually
        // resolves, so a failed creation no longer silently wipes the
        // user's draft.
        throw new Error(t("chatPage.templateQuickAccess.createAgentError"));
      }

      agentId = toAgentId(result.agent);
      // Keep local `agents` state in sync for other consumers (e.g. the
      // `?agent=` deep link) - resolveAgentForTemplate always asks the
      // server (see its own docstring for why), this does not shortcut
      // that. Only add it if published: `agents` is otherwise exclusively
      // published agents (see the mount-time fetch above), and the resolve
      // flow's reuse branch can now return a still-unpublished draft
      // as-is (PR review finding B3) rather than always republishing it.
      if (result.agent.status === "published") {
        setAgents((prev) =>
          prev.some((agent) => agent.id === result.agent.id) ? prev : [...prev, result.agent]
        );
      }

      if (result.created) {
        const templateName = selectedTemplate.name;
        toast.custom((toastId) => (
          <div className="flex w-full flex-col gap-3 rounded-xl border border-green-600 bg-background p-4 shadow-lg">
            <div className="flex items-start gap-2">
              <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-green-600" />
              <span className="text-sm font-medium text-green-600">
                {t("chatPage.templateQuickAccess.agentCreatedToast", { name: templateName })}
              </span>
            </div>
            <Button
              size="sm"
              className="self-end"
              onClick={() => {
                toast.dismiss(toastId);
                router.push("/build");
              }}
            >
              {t("chatPage.templateQuickAccess.viewInAgents")}
            </Button>
          </div>
        ));
      }
    }

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
      setSelectedTemplate(null);
      setSelectedPromptKey(null);
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

  const handleRemoveSelectedAgent = (agentId: number | string) => {
    setSelectedAgents((prev) => prev.filter((agent) => agent.id !== agentId));
  };

  const handleTemplatePromptSelect = (template: Template, prompt: SamplePrompt, index: number) => {
    setInputValue(prompt.prompt);
    setPromptHighlightTerms(prompt.highlights || []);
    setSelectedTemplate({ id: template.id, name: template.name });
    setSelectedPromptKey(`${template.id}:${index}`);
    setSelectedAgents([]);
  };

  const handleRemoveSelectedTemplate = () => {
    setSelectedTemplate(null);
    setSelectedPromptKey(null);
  };

  return (
    <div className="h-full bg-background flex flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto">
        <main className="container max-w-4xl mx-auto px-4 py-8">
          {/* No `agents`/`onAgentClick` props here: the template quick-access
              grid (passed via `templates` below) replaces the old "Chat with
              Agents" avatar picker on this page, matching the redesigned
              Task page. `agents` state itself is still fetched and used -
              for the `?agent=` deep link above - just not rendered as a
              picker here. Template-reuse resolution goes through the server
              unconditionally (see resolveAgentForTemplate); there is no
              client-side matching against this list anymore. */}
          <ChatStartScreen
            title={t("chatPage.page.emptyTitle", { appName: branding.appName })}
            description={t("chatPage.page.emptyDescription")}
            icon={<Bot className="w-10 h-10 text-[hsl(var(--gradient-from))]" />}
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
            templates={templates}
            templatesLoading={templatesLoading}
            templatesError={templatesError}
            onRetryTemplates={fetchTemplates}
            selectedCategory={selectedCategory}
            onCategoryChange={setSelectedCategory}
            selectedTemplate={selectedTemplate}
            onRemoveSelectedTemplate={handleRemoveSelectedTemplate}
            selectedPromptKey={selectedPromptKey}
            onTemplatePromptSelect={handleTemplatePromptSelect}
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
