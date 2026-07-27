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
import { useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import type { Template, SamplePrompt } from "@/types/template";
import { FEATURED_CATEGORY_ID } from "@/lib/template-categories";

// Deep-link presets for "?starter=" (e.g. the Welcome modal's Presentation Builder
// card). Kept separate from the template quick-access system below.
const STARTER_PROMPTS: Record<string, { prompt: string; highlights: string[] }> = {
  presentation: {
    prompt: "Build a N-slide presentation on topic for audience.",
    highlights: ["N", "topic", "audience"],
  },
};

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
  useEffect(() => {
    const fetchTemplates = async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/templates/?lang=${locale}`);
        if (response.ok) {
          const data = await response.json();
          setTemplates(Array.isArray(data) ? data : []);
        }
      } catch (error) {
        console.error("Failed to fetch templates:", error);
      }
    };
    fetchTemplates();
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

  const starterPreset = useMemo(() => {
    const found = starter ? STARTER_PROMPTS[starter] : null;
    if (!found) return null;

    return {
      prompt: found.prompt,
      highlights: found.highlights,
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

  // The backend has no dedicated "created from template X" link on an Agent
  // record, so we correlate by name: a template-created agent is always
  // named exactly `template.name` on first creation (see
  // create_agent_from_template in agent_management.py). Renaming that agent
  // later breaks this correlation - a known tradeoff of not having a
  // persisted template_id column.
  const findExistingAgentIdForTemplate = (templateName: string, agentList: AgentCard[]) => {
    const match = agentList.find((agent) => agent.name === templateName);
    const matchId = Number(match?.id);
    return Number.isNaN(matchId) ? null : matchId;
  };

  // Reuse an already-created agent for this template if one exists; only
  // create + publish a new one the first time a template is used.
  const resolveAgentForTemplate = async (
    templateId: string,
    templateName: string
  ): Promise<{ agentId: number; created: boolean }> => {
    const knownExistingId = findExistingAgentIdForTemplate(templateName, agents);
    if (knownExistingId !== null) {
      return { agentId: knownExistingId, created: false };
    }

    const createResponse = await apiRequest(`${getApiUrl()}/api/agents/from-template`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ template_id: templateId }),
    });

    if (createResponse.status === 400) {
      // Name collision our local (published-only) list didn't know about -
      // e.g. a draft agent, or one created in another tab/session since we
      // last fetched. Look it up and reuse it rather than minting a
      // duplicate under a disambiguated name.
      const listResponse = await apiRequest(`${getApiUrl()}/api/agents`);
      if (listResponse.ok) {
        const allAgents = await listResponse.json();
        const match = Array.isArray(allAgents)
          ? allAgents.find((agent) => agent && agent.name === templateName)
          : undefined;
        const matchId = Number(match?.id);
        if (match && !Number.isNaN(matchId)) {
          if (match.status !== "published") {
            const publishResponse = await apiRequest(
              `${getApiUrl()}/api/agents/${matchId}/publish`,
              { method: "POST" }
            );
            if (!publishResponse.ok) {
              throw new Error(`Failed to publish agent (${publishResponse.status})`);
            }
          }
          return { agentId: matchId, created: false };
        }
      }
      throw new Error(`Failed to create agent from template (${createResponse.status})`);
    }

    if (!createResponse.ok) {
      throw new Error(`Failed to create agent from template (${createResponse.status})`);
    }
    const createdAgent = await createResponse.json();

    const publishResponse = await apiRequest(
      `${getApiUrl()}/api/agents/${createdAgent.id}/publish`,
      { method: "POST" }
    );
    if (!publishResponse.ok) {
      throw new Error(`Failed to publish agent (${publishResponse.status})`);
    }

    return { agentId: createdAgent.id, created: true };
  };

  const handleSend = async (message: string, filesToSend: File[], config?: any) => {
    if (state.isProcessing) return;

    let agentId = Number(selectedAgents[0]?.id);

    if (Number.isNaN(agentId) && selectedTemplate) {
      try {
        const result = await resolveAgentForTemplate(selectedTemplate.id, selectedTemplate.name);
        agentId = result.agentId;
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
          // Keep local state in sync so sending from this template again in
          // the same session (without a page reload) reuses this agent
          // instead of creating another one.
          setAgents((prev) => [
            ...prev,
            { id: result.agentId, name: selectedTemplate.name, status: "published" },
          ]);
        }
      } catch (error) {
        console.error("Failed to create agent from template:", error);
        toast.error(t("chatPage.templateQuickAccess.createAgentError"));
        return;
      }
    }

    const nextConfig = {
      ...config,
      agentId: Number.isNaN(agentId) ? undefined : agentId,
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

  const handleAgentClick = (agent: AgentCard) => {
    setSelectedAgents((prev) => {
      const currentSelected = prev[0];
      if (currentSelected?.id === agent.id) {
        return [];
      }
      return [agent];
    });
    setSelectedTemplate(null);
    setSelectedPromptKey(null);
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
          <ChatStartScreen
            title={t("chatPage.page.emptyTitle", { appName: branding.appName })}
            description={t("chatPage.page.emptyDescription")}
            icon={<Bot className="w-10 h-10 text-[hsl(var(--gradient-from))]" />}
            agents={agents}
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
            templates={templates}
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
