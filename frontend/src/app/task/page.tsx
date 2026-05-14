"use client";

import { useState, useEffect, useMemo } from "react";
import { Bot, Presentation, Search, Smartphone, Wand2 } from "lucide-react";
import { useI18n } from "@/contexts/i18n-context";
import { useApp } from "@/contexts/app-context-chat";
import { ChatStartScreen, AgentCard } from "@/components/chat/ChatStartScreen";
import { FilePreviewDialog } from "@/components/file/file-preview-dialog";
import { getBrandingFromEnv } from "@/lib/branding";
import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";
import { useSearchParams } from "next/navigation";

function TaskHomePageContent() {
  const { t } = useI18n();
  const { sendMessage, state, dispatch, closeFilePreview } = useApp();
  const searchParams = useSearchParams();
  const starter = searchParams.get("starter");
  const promptFromQuery = searchParams.get("prompt");

  const [files, setFiles] = useState<File[]>([]);
  const [agents, setAgents] = useState<AgentCard[]>([]);
  const [selectedAgents, setSelectedAgents] = useState<AgentCard[]>([]);
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

  const samplePrompts = useMemo(() => ([
    {
      id: "research",
      icon: Search,
      title: t("chatPage.cards.research.title"),
      prompt: "Research topic and deliver a structured report with key findings, data points, and sources.",
      promptHighlights: ["topic"],
    },
    {
      id: "linkedin",
      icon: Smartphone,
      title: t("chatPage.cards.linkedin.title"),
      prompt: "Write a LinkedIn post about topic or achievement. Tone: professional / inspirational.",
      promptHighlights: ["topic or achievement", "professional / inspirational"],
    },
    {
      id: "poster",
      icon: Wand2,
      title: t("chatPage.cards.poster.title"),
      prompt: "Create a promotional poster for event or product. Style: modern / bold / minimal.",
      promptHighlights: ["event or product", "modern / bold / minimal"],
    },
    {
      id: "compare",
      icon: Search,
      title: t("chatPage.cards.compare.title"),
      prompt: "Compare product A vs product B across key criteria. Provide a detailed analysis with a recommendation.",
      promptHighlights: ["product A", "product B", "key criteria"],
    },
    {
      id: "visual",
      icon: Wand2,
      title: t("chatPage.cards.visual.title"),
      prompt: "Create a platform graphic for campaign or brand. Size: square / story / banner. Theme: colour or style.",
      promptHighlights: ["platform", "campaign or brand", "square / story / banner", "colour or style"],
    },
    {
      id: "presentation",
      icon: Presentation,
      title: t("chatPage.cards.presentation.title"),
      prompt: "Build a N-slide presentation on topic for audience.",
      promptHighlights: ["N", "topic", "audience"],
    }
  ]), [t]);

  const starterPreset = useMemo(() => {
    const found = samplePrompts.find((prompt) => prompt.id === starter);
    if (!found) return null;

    return {
      prompt: found.prompt,
      highlights: found.promptHighlights,
    };
  }, [starter, samplePrompts]);

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

    const nextConfig = {
      ...config,
      delegateAgentIds: selectedAgents.map((agent) => Number(agent.id)).filter((id) => !Number.isNaN(id)),
    };

    // Use sendMessage from AppContext - it will create task and send files via WebSocket
    await sendMessage(message, nextConfig, filesToSend || files);

    // Clear files after sending
    setFiles([]);
    setInputValue("");
    setPromptHighlightTerms([]);
    setSelectedAgents([]);
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
  };

  const handleRemoveSelectedAgent = (agentId: number | string) => {
    setSelectedAgents((prev) => prev.filter((agent) => agent.id !== agentId));
  };

  return (
    <div className="h-full bg-background flex flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto">
        <main className="container max-w-4xl mx-auto px-4 py-8">
          <ChatStartScreen
            title={t("chatPage.page.emptyTitle", { appName: branding.appName })}
            description={t("chatPage.page.emptyDescription")}
            icon={<Bot className="w-10 h-10 text-[hsl(var(--gradient-from))]" />}
            prompts={samplePrompts}
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
