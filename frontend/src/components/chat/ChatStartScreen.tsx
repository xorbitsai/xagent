import React from "react";
import { Bot, Sparkles, Smartphone } from "lucide-react";
import { ChatInput } from "@/components/chat/ChatInput";
import { useI18n } from "@/contexts/i18n-context";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { getApiUrl } from "@/lib/utils";

export interface PromptCard {
  icon?: any;
  title?: string;
  description?: string;
  prompt: string;
  promptHighlights?: string[];
  color?: string;
  bg?: string;
}

export interface AgentCard {
  id: number | string;
  name: string;
  description?: string | null;
  logo_url?: string | null;
  status?: string;
  readonly?: boolean;
  can_edit?: boolean;
}

interface ChatStartScreenProps {
  title: string;
  description?: string;
  icon?: React.ReactNode | string; // URL string or ReactNode
  prompts?: (PromptCard | string)[];
  agents?: AgentCard[];
  onAgentClick?: (agent: AgentCard) => void;
  selectedAgents?: AgentCard[];
  onRemoveSelectedAgent?: (agentId: number | string) => void;
  onSend: (message: string, files: File[], config?: any) => void;
  isSending?: boolean;
  inputValue?: string;
  onInputChange?: (value: string) => void;
  onPromptSelect?: (prompt: string, promptHighlights?: string[]) => void;
  promptHighlightTerms?: string[];
  files?: File[];
  onFilesChange?: (files: File[]) => void;
  showModeToggle?: boolean;
  readOnlyConfig?: boolean;
  hideConfig?: boolean;
  compactInput?: boolean;
  deferFileUpload?: boolean;
  taskConfig?: any;
  autoFocus?: boolean;
  inputMinHeightClass?: string;
}

export function ChatStartScreen({
  title,
  description,
  icon,
  prompts,
  agents,
  onAgentClick,
  selectedAgents = [],
  onRemoveSelectedAgent,
  onSend,
  isSending = false,
  inputValue,
  onInputChange,
  onPromptSelect,
  promptHighlightTerms = [],
  files = [],
  onFilesChange,
  showModeToggle = false,
  readOnlyConfig = false,
  hideConfig = false,
  compactInput = false,
  deferFileUpload = false,
  taskConfig,
  autoFocus = false,
  inputMinHeightClass
}: ChatStartScreenProps) {
  const { t } = useI18n();

  const handlePromptClick = (prompt: string, promptHighlights?: string[]) => {
    if (onPromptSelect) {
      onPromptSelect(prompt, promptHighlights);
      return;
    }
    if (onInputChange) {
      onInputChange(prompt);
    }
  };

  return (
    <div className="flex flex-col items-center justify-start min-h-[80vh] pt-10 pb-16 px-6 text-center">
      <h2 className="text-[26px] font-extrabold mb-2 bg-gradient-to-r from-[hsl(234_62%_45%)] to-[hsl(234_62%_60%)] bg-clip-text text-transparent leading-[1.2] tracking-tight">
        {title}
      </h2>
      {description && (
        <p className="text-[13.5px] text-muted-foreground mb-7 max-w-md leading-[1.7]">{description}</p>
      )}

      <div className="w-full max-w-[680px] mx-auto space-y-6">
        <div>
          <ChatInput
            onSend={(msg, config) => onSend(msg, files, config)}
            isLoading={isSending}
            files={files}
            onFilesChange={onFilesChange || (() => { })}
            showModeToggle={showModeToggle}
            inputValue={inputValue}
            onInputChange={onInputChange}
            promptHighlightTerms={promptHighlightTerms}
            readOnlyConfig={readOnlyConfig}
            hideConfig={hideConfig}
            compact={compactInput}
            deferFileUpload={deferFileUpload}
            taskConfig={taskConfig}
            autoFocus={autoFocus}
            minHeightClass={inputMinHeightClass}
            selectedAgents={selectedAgents}
            onRemoveSelectedAgent={onRemoveSelectedAgent}
          />
        </div>

        {prompts && prompts.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-[10.5px] font-semibold text-muted-foreground uppercase tracking-[0.08em] px-1">
              <Sparkles className="w-3.5 h-3.5" />
              <span>{t("chatPage.sections.startingPrompts")}</span>
            </div>
            <div className={`grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4`}>
              {prompts.map((item, index) => {
                const isString = typeof item === 'string';
                const promptText = isString ? item : item.prompt;

                if (isString) {
                  return (
                    <div
                      key={index}
                      onClick={() => handlePromptClick(promptText)}
                      className="group relative p-4 h-28 rounded-xl border border-border bg-card hover:bg-muted/50 cursor-pointer transition-all duration-300 flex flex-col justify-center text-left"
                    >
                      <p className="text-sm text-foreground/90 line-clamp-3">{promptText}</p>
                    </div>
                  );
                }

                // Card style for Task Page
                return (
                  <div
                    key={index}
                    onClick={() => handlePromptClick(promptText, item.promptHighlights)}
                    className="group relative px-4 py-3 min-h-[72px] rounded-xl border border-border bg-card hover:bg-muted/50 cursor-pointer transition-all duration-300 flex flex-row items-center text-left gap-4"
                  >
                    <div className="flex items-center justify-center shrink-0 h-10 w-10 rounded-lg bg-blue-50 dark:bg-blue-900/20 text-blue-500">
                      {item.icon && <item.icon className="w-5 h-5" />}
                    </div>
                    <h3 className="font-medium text-[14px] text-foreground/90 leading-snug">{item.title}</h3>
                  </div>
                );
              })}
            </div>

            {/* Chat with Agents section */}
            {(agents && agents.length > 0) || !agents ? (
              <>
                <div className="flex items-center gap-2 text-[10.5px] font-semibold text-muted-foreground uppercase tracking-[0.08em] mt-6 px-1">
                  <Bot className="w-3.5 h-3.5" />
                  <span>{t("chatPage.sections.chatWithAgents")}</span>
                </div>
                {/* Horizontal scroll container for agents, limited width to show about 3 items initially */}
                <TooltipProvider delayDuration={200}>
                  <div className="flex gap-8 mt-4 overflow-x-auto pb-4 pt-2 px-1 snap-x snap-mandatory scrollbar-hide" style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
                    {agents ? (
                      agents.map((agent) => {
                        const isSelected = selectedAgents.some(
                          (selectedAgent) => selectedAgent.id === agent.id
                        );

                        return (
                          <Tooltip key={agent.id}>
                            <TooltipTrigger asChild>
                              <div
                                className="flex flex-col items-center gap-2 cursor-pointer group flex-shrink-0 snap-start w-[64px]"
                                onClick={() => onAgentClick?.(agent)}
                              >
                                <div className={`w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 shadow-sm border overflow-hidden transition-all ${isSelected ? "border-primary ring-2 ring-primary/20" : "border-blue-200 group-hover:shadow-md"}`}>
                                  {agent.logo_url ? (
                                    <img src={agent.logo_url.startsWith('http') ? agent.logo_url : `${getApiUrl()}${agent.logo_url}`} alt={agent.name} className="w-full h-full object-cover" />
                                  ) : (
                                    <Bot className="w-6 h-6" />
                                  )}
                                </div>
                                <span className={`text-xs font-medium text-center leading-tight max-w-[64px] line-clamp-2 ${isSelected ? "text-primary" : "text-muted-foreground"}`} title={agent.name}>{agent.name}</span>
                              </div>
                            </TooltipTrigger>
                            {agent.description ? (
                              <TooltipContent side="top" className="max-w-[240px] text-left">
                                <div className="space-y-1">
                                  <div className="font-medium">{agent.name}</div>
                                  <p className="text-xs text-muted-foreground">{agent.description}</p>
                                </div>
                              </TooltipContent>
                            ) : null}
                          </Tooltip>
                        );
                      })
                    ) : (
                      // Fallback mocked agents to match design if no agents prop provided
                      <>
                        <div className="flex flex-col items-center gap-2 cursor-pointer group flex-shrink-0 snap-start w-[64px]">
                          <div className="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 shadow-sm border border-blue-200 group-hover:shadow-md transition-all">
                            <Bot className="w-6 h-6" />
                          </div>
                          <span className="text-xs text-muted-foreground font-medium text-center leading-tight max-w-[64px]">{t("chatPage.agents.researcher")}</span>
                        </div>
                        <div className="flex flex-col items-center gap-2 cursor-pointer group flex-shrink-0 snap-start w-[64px]">
                          <div className="w-12 h-12 rounded-full bg-purple-100 flex items-center justify-center text-purple-600 shadow-sm border border-purple-200 group-hover:shadow-md transition-all">
                            <Sparkles className="w-6 h-6" />
                          </div>
                          <span className="text-xs text-muted-foreground font-medium text-center leading-tight max-w-[64px]">{t("chatPage.agents.poster")}</span>
                        </div>
                        <div className="flex flex-col items-center gap-2 cursor-pointer group flex-shrink-0 snap-start w-[64px]">
                          <div className="w-12 h-12 rounded-full bg-blue-50 flex items-center justify-center text-blue-500 shadow-sm border border-blue-200 group-hover:shadow-md transition-all">
                            <Smartphone className="w-6 h-6" />
                          </div>
                          <span className="text-xs text-muted-foreground font-medium text-center leading-tight max-w-[64px]">{t("chatPage.agents.linkedin")}</span>
                        </div>
                      </>
                    )}
                  </div>
                </TooltipProvider>
              </>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}
