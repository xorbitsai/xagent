import React, { useEffect, useRef, useState } from "react";
import { Sparkles } from "lucide-react";
import { ChatInput } from "@/components/chat/ChatInput";
import { useI18n } from "@/contexts/i18n-context";
import { resolveAgentLogoUrl, getApiUrl } from "@/lib/utils";
import { PersonaAvatar } from "@/components/templates/persona-avatar";
import { getBrandingFromEnv } from "@/lib/branding";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

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
  // Built-in template id this agent was instantiated from, or null/undefined
  // for agents built from scratch. Used to key template-reuse matching off a
  // stable id instead of the user-editable display name.
  template_id?: string | null;
  // The agent's own suggested starting prompts (from its `suggested_prompts`
  // config). Read by callers that auto-fill the composer from the first one
  // when this agent is picked as the task's lead - not consumed here.
  suggested_prompts?: string[];
  // A hired agent's persona portrait (resolved by the caller from the
  // template it traces back to via `template_id`) - preferred over
  // `logo_url` when present, since a hired agent's `logo_url` is never
  // actually set to its persona's face today.
  persona_avatar?: string | null;
  // A short label shown next to the agent's name in the picker row (e.g.
  // its template's category) - purely cosmetic, omitted when unknown.
  specialty?: string;
}

// A hired agent's persona photo, falling back to a resolved `logo_url` -
// shared by the hero swap and the picker pill so both always agree on
// which photo an agent shows.
function resolveTeammateAvatar(agent: Pick<AgentCard, "persona_avatar" | "logo_url">): string | null {
  return agent.persona_avatar || resolveAgentLogoUrl(agent.logo_url ?? null, getApiUrl());
}

type ChatStartScreenProps = {
  title: string;
  description?: string;
  icon?: React.ReactNode | string; // URL string or ReactNode
  prompts?: (PromptCard | string)[];
  agents?: AgentCard[];
  // True when the caller's agent-list fetch failed - shown in place of the
  // picker row (which would otherwise look identical to "no published
  // agents exist" with no way to recover).
  agentsError?: boolean;
  onRetryAgents?: () => void;
  onAgentClick?: (agent: AgentCard) => void;
  selectedAgents?: AgentCard[];
  onSend: (message: string, files: File[], config?: any) => void | Promise<void>;
  isSending?: boolean;
  inputValue?: string;
  onInputChange?: (value: string) => void;
  onFocusChange?: (focused: boolean) => void;
  onPromptSelect?: (prompt: string, promptHighlights?: string[]) => void;
  promptHighlightTerms?: string[];
  files?: File[];
  onFilesChange?: (files: File[]) => void;
  showModeToggle?: boolean;
  readOnlyConfig?: boolean;
  hideConfig?: boolean;
  compactInput?: boolean;
  deferFileUpload?: boolean;
  filesDisabled?: boolean;
  voiceInputEnabled?: boolean;
  taskConfig?: any;
  autoFocus?: boolean;
  inputMinHeightClass?: string;
};

export function ChatStartScreen({
  title,
  description,
  icon,
  prompts,
  agents,
  agentsError = false,
  onRetryAgents,
  onAgentClick,
  selectedAgents = [],
  onSend,
  isSending = false,
  inputValue,
  onInputChange,
  onFocusChange,
  onPromptSelect,
  promptHighlightTerms = [],
  files = [],
  onFilesChange,
  showModeToggle = false,
  readOnlyConfig = false,
  hideConfig = false,
  compactInput = false,
  deferFileUpload = false,
  filesDisabled = false,
  voiceInputEnabled = true,
  taskConfig,
  autoFocus = false,
  inputMinHeightClass,
}: ChatStartScreenProps) {
  const { t } = useI18n();
  const branding = getBrandingFromEnv();
  const enabledFiles = filesDisabled ? [] : files;

  const handlePromptClick = (prompt: string, promptHighlights?: string[]) => {
    if (onPromptSelect) {
      onPromptSelect(prompt, promptHighlights);
      return;
    }
    if (onInputChange) {
      onInputChange(prompt);
    }
  };

  // Only meaningful in the "My Team" picker context (`agents` supplied) -
  // other callers (an agent's own chat, widget pages) pass `selectedAgents`
  // for a different purpose or not at all, so this stays undefined for them
  // and the header falls back to the plain title/description below.
  const leadAgent = agents && agents.length > 0 ? selectedAgents[0] : undefined;

  // The right-edge fade only earns its place when there's actually
  // something past the edge to scroll to - otherwise it's a gradient
  // fading into nothing over the last, fully-visible pill.
  const teamStripRef = useRef<HTMLDivElement>(null);
  const [teamStripHasMore, setTeamStripHasMore] = useState(false);
  useEffect(() => {
    const strip = teamStripRef.current;
    if (!strip) {
      setTeamStripHasMore(false);
      return;
    }
    const sync = () => {
      setTeamStripHasMore(strip.scrollWidth - strip.scrollLeft - strip.clientWidth > 8);
    };
    sync();
    strip.addEventListener("scroll", sync);
    // A ResizeObserver on the strip itself catches every way its available
    // width can change (window resize, but also a sidebar toggling or any
    // other local layout shift) - a window `resize` listener alone would
    // miss the ones that aren't the window itself resizing.
    const observer = new ResizeObserver(sync);
    observer.observe(strip);
    return () => {
      strip.removeEventListener("scroll", sync);
      observer.disconnect();
    };
  }, [agents]);

  return (
    <div className="flex flex-col items-center justify-start min-h-[80vh] pt-10 pb-16 px-6 text-center">
      {leadAgent ? (
        <div className="w-full max-w-[680px] mx-auto flex items-center gap-4 mb-6 text-left">
          <PersonaAvatar
            persona={{
              name: leadAgent.name,
              avatar: resolveTeammateAvatar(leadAgent),
            }}
            sizeClassName="h-16 w-16"
            textClassName="text-xl"
            className="rounded-[22px] shrink-0"
          />
          <div className="min-w-0">
            <h2 className="text-2xl font-bold leading-tight break-words">{title}</h2>
            <p className="text-sm text-muted-foreground mt-1 break-words">
              {t("chatPage.sections.leadReady", { name: leadAgent.name })}
            </p>
          </div>
        </div>
      ) : (
        <>
          <h2 className="text-[26px] font-extrabold mb-2 bg-gradient-to-r from-[hsl(234_62%_45%)] to-[hsl(234_62%_60%)] bg-clip-text text-transparent leading-[1.2] tracking-tight">
            {title}
          </h2>
          {description && (
            <p className="text-[13.5px] text-muted-foreground mb-7 max-w-md leading-[1.7]">{description}</p>
          )}
        </>
      )}

      <div className="w-full max-w-[680px] mx-auto space-y-6">
        <div>
          <ChatInput
            onSend={(msg, config) => onSend(msg, enabledFiles, config)}
            isLoading={isSending}
            files={enabledFiles}
            onFilesChange={
              filesDisabled ? undefined : (onFilesChange || (() => { }))
            }
            showModeToggle={showModeToggle}
            inputValue={inputValue}
            onInputChange={onInputChange}
            onFocusChange={onFocusChange}
            promptHighlightTerms={promptHighlightTerms}
            readOnlyConfig={readOnlyConfig}
            hideConfig={hideConfig}
            compact={compactInput}
            deferFileUpload={deferFileUpload}
            filesDisabled={filesDisabled}
            voiceInputEnabled={voiceInputEnabled}
            hideFileUpload={filesDisabled}
            taskConfig={taskConfig}
            autoFocus={autoFocus}
            minHeightClass={inputMinHeightClass}
            selectedAgents={selectedAgents}
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
          </div>
        )}

        {agentsError ? (
          <div className="flex items-center justify-between gap-3 rounded-xl border border-destructive/30 bg-destructive/[0.04] px-4 py-3 text-left">
            <p className="text-xs text-destructive">{t("chatPage.sections.assignToTeammateError")}</p>
            <button
              type="button"
              onClick={onRetryAgents}
              className="shrink-0 text-xs font-semibold text-destructive underline underline-offset-2 hover:no-underline"
            >
              {t("common.retry")}
            </button>
          </div>
        ) : agents && agents.length > 0 && (
          <div className="space-y-2.5 text-left">
            <p className="text-xs text-muted-foreground px-1">
              <b className="font-semibold text-foreground/85">
                {t("chatPage.sections.assignToTeammateLead")}
              </b>
              {t("chatPage.sections.assignToTeammateHint", { appName: branding.appName })}
            </p>
            <div className="relative">
              <div
                ref={teamStripRef}
                data-testid="team-strip"
                role="group"
                aria-label={t("chatPage.sections.assignToTeammate")}
                className="flex items-center gap-2 overflow-x-auto p-1 pr-8 [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden"
              >
                <span className="shrink-0 text-[11px] font-bold uppercase tracking-[0.07em] text-muted-foreground/80 whitespace-nowrap">
                  {t("chatPage.sections.assignToTeammate")}
                </span>
                <TooltipProvider delayDuration={200}>
                  {agents.map((agent) => {
                    const isSelected = selectedAgents.some(
                      (selectedAgent) => selectedAgent.id === agent.id
                    );
                    const avatarUrl = resolveTeammateAvatar(agent);

                    const pill = (
                      <button
                        type="button"
                        aria-pressed={isSelected}
                        onClick={() => onAgentClick?.(agent)}
                        className={`shrink-0 flex items-center gap-[7px] rounded-full border pl-[3px] pr-3 py-[3px] transition-all whitespace-nowrap ${
                          isSelected
                            ? "border-primary/40 bg-primary/[0.06] shadow-[0_4px_14px_hsl(var(--primary)/0.11)]"
                            : "border-border bg-background hover:border-primary/30 hover:bg-primary/[0.04] hover:-translate-y-px"
                        }`}
                      >
                        <PersonaAvatar
                          persona={{ name: agent.name, avatar: avatarUrl }}
                          sizeClassName="h-[27px] w-[27px]"
                          textClassName="text-[11px]"
                          className={isSelected ? "shadow-[0_0_0_2px_hsl(var(--background)),0_0_0_3px_hsl(var(--primary))]" : ""}
                          decorative
                        />
                        <b className={`text-[12.5px] font-semibold ${isSelected ? "text-primary" : "text-foreground"}`}>
                          {agent.name}
                        </b>
                        {agent.specialty && (
                          <span className="text-[11.5px] text-muted-foreground">{agent.specialty}</span>
                        )}
                      </button>
                    );

                    return (
                      <Tooltip key={agent.id}>
                        <TooltipTrigger asChild>{pill}</TooltipTrigger>
                        {agent.description && (
                          <TooltipContent side="top" className="max-w-[240px] text-left">
                            <div className="space-y-1">
                              <div className="font-medium">{agent.name}</div>
                              <p className="text-xs text-muted-foreground">{agent.description}</p>
                            </div>
                          </TooltipContent>
                        )}
                      </Tooltip>
                    );
                  })}
                </TooltipProvider>
              </div>
              {/* Fades the right edge so a horizontally-clipped pill reads as
                  "more to scroll" rather than an abruptly truncated name -
                  only when there's actually more past the edge, or it's a
                  gradient fading into nothing over a fully-visible pill. */}
              {teamStripHasMore && (
                <div className="pointer-events-none absolute inset-y-0 right-0 w-8 bg-gradient-to-l from-background to-transparent" />
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
