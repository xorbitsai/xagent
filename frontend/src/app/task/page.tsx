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

// A blank/whitespace-only entry (nothing enforces non-blank strings at
// every writer of an agent's or template's prompts) must not shadow a
// perfectly usable one that comes after it - shared by every place below
// that needs "the first prompt actually worth auto-filling with".
function firstNonBlankPrompt(prompts?: Array<string | undefined | null>): string | undefined {
  return prompts?.find((prompt): prompt is string => Boolean(prompt && prompt.trim().length > 0));
}

function TaskHomePageContent() {
  const { t, locale } = useI18n();
  const { sendMessage, state, dispatch, closeFilePreview } = useApp();
  const searchParams = useSearchParams();
  const starter = searchParams.get("starter");
  const promptFromQuery = searchParams.get("prompt");
  const agentFromQuery = searchParams.get("agent");
  const appliedAgentFromQueryRef = useRef<string | null>(null);
  // True once the composer holds something we did not put there ourselves
  // - the user typed/edited/cleared it, or it was seeded from a
  // `?prompt=`/`?starter=` deep link. Once true, no auto-fill or
  // auto-clear logic may touch the composer again until the task is sent.
  // This is deliberately a flag, not "does the current text match the
  // text we last filled": that comparison breaks the moment the user
  // clears the box back to empty - the exact state auto-fill considers
  // "untouched" - which would then silently reinsert the very prompt they
  // just deleted.
  const composerDirtyRef = useRef(false);
  // ChatInput clears its own controlled value via `onInputChange("")` right
  // after a successful send (its own post-submit reset), which arrives
  // through this exact same callback as a genuine user edit - without this
  // flag, that call would immediately mark the just-cleared composer dirty
  // again, permanently disabling auto-fill for the rest of the session
  // after the very first send. Set synchronously right before the resolved
  // `onSend` promise hands control back to ChatInput, so there is no
  // window for a real keystroke to be mistaken for the echo.
  const suppressNextInputChangeRef = useRef(false);

  const [files, setFiles] = useState<File[]>([]);
  const [agents, setAgents] = useState<AgentCard[]>([]);
  const [agentsError, setAgentsError] = useState(false);
  const [agentsRetryToken, setAgentsRetryToken] = useState(0);
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

  // Fetch agents on mount (and again on retry) - a failure here must not
  // just silently render zero teammates, since that reads identically to
  // "no published agents exist" and gives the user nothing to act on.
  useEffect(() => {
    let cancelled = false;
    const fetchAgents = async () => {
      setAgentsError(false);
      try {
        const response = await apiRequest(`${getApiUrl()}/api/agents`);
        if (cancelled) return;
        if (!response.ok) {
          setAgentsError(true);
          return;
        }
        const data = await response.json();
        if (cancelled) return;
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
      } catch (error) {
        if (!cancelled) {
          console.error("Failed to fetch agents:", error);
          setAgentsError(true);
        }
      }
    };
    fetchAgents();
    return () => {
      cancelled = true;
    };
  }, [agentsRetryToken]);

  const handleRetryAgents = () => {
    setAgentsRetryToken((token) => token + 1);
  };

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
        // almost never populated by our built-in templates, so fall back to
        // the template's marketplace `sample_prompts` for the "My Team"
        // auto-fill in that case - but never override an agent's own
        // prompts once set, since /build's Suggested Prompts editor lets a
        // user customize them after hiring, and that edit must win here.
        const ownPrompts = agent.suggested_prompts;
        // A blank first entry must not read as "this agent has its own
        // prompts" (masking a perfectly good template sample) any more
        // than it should read as "auto-fill with nothing" at click time -
        // both need the first actually-usable entry, not just index 0.
        const hasUsableOwnPrompt = firstNonBlankPrompt(ownPrompts) !== undefined;
        const samplePrompt = firstNonBlankPrompt(template.sample_prompts?.map((p) => p.prompt));
        return {
          ...agent,
          persona_avatar: template.persona?.avatar,
          specialty: categoryLabel(t, template.category),
          suggested_prompts: hasUsableOwnPrompt ? ownPrompts : (samplePrompt ? [samplePrompt] : ownPrompts),
        };
      }),
    [agents, templatesById, t]
  );

  // `selectedAgents` can be set from a pre-enrichment `teammates` snapshot
  // (the user clicks a hired agent's pill before its template lookup has
  // resolved, so the object captured at click time has no persona photo or
  // sample prompt yet). Re-resolving by id against the latest `teammates`
  // on every render means the hero portrait and the picker's `isSelected`
  // pill stay in sync once enrichment lands, instead of being stuck on the
  // stale pre-enrichment object until the user deselects and reselects.
  const resolvedSelectedAgents = useMemo(
    () =>
      selectedAgents.map(
        (agent) => teammates.find((teammate) => teammate.id === agent.id) ?? agent
      ),
    [selectedAgents, teammates]
  );

  useEffect(() => {
    if (!agentFromQuery || appliedAgentFromQueryRef.current === agentFromQuery || teammates.length === 0) {
      return;
    }

    const selectedAgent = findRunnableAgentById(teammates, agentFromQuery);
    if (!selectedAgent) {
      return;
    }

    // Batched into the same commit as setSelectedAgents below (both are
    // called synchronously in this effect body) - see the longer note on
    // this in handleAgentClick for why this must not be left to the
    // separate config-fetch effect alone.
    setSelectedAgentConfig(undefined);
    setSelectedAgents([selectedAgent]);
    appliedAgentFromQueryRef.current = agentFromQuery;
  }, [agentFromQuery, teammates]);

  useEffect(() => {
    let cancelled = false;
    // Defense-in-depth only: every caller that changes `selectedAgents`
    // (handleAgentClick, the ?agent= deep-link effect) already clears
    // `selectedAgentConfig` itself in the SAME synchronous batch as the
    // selection change, so there is no render where a new lead's id is
    // paired with a previous lead's config. This effect firing later
    // (after render/paint, not synchronously with the state update that
    // triggered it) would otherwise leave exactly that window open.
    setSelectedAgentConfig(undefined);

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
    // A non-empty `?prompt=`/`?starter=` deep link is already an explicit
    // choice (a pasted link, or a Welcome-modal card the user clicked) -
    // treat it the same as the user's own text so a teammate pick can't
    // silently blow it away.
    composerDirtyRef.current = Boolean(queryInputValue);
  }, [queryInputValue, queryPromptHighlightTerms]);

  // Companion fix for the same pre-enrichment race `resolvedSelectedAgents`
  // addresses above: if the agent had no prompt to offer at click time
  // (its template hadn't loaded yet) but one arrives once enrichment
  // lands, fill it in - but only while the composer is still untouched.
  useEffect(() => {
    if (composerDirtyRef.current) return;
    const lead = resolvedSelectedAgents[0];
    const prompt = firstNonBlankPrompt(lead?.suggested_prompts);
    if (prompt && inputValue !== prompt) {
      setInputValue(prompt);
      setPromptHighlightTerms([]);
    }
  }, [resolvedSelectedAgents, inputValue]);

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
      composerDirtyRef.current = false;
      // ChatInput's own post-submit reset calls this same onInputChange
      // callback right after this promise resolves - suppress that echo
      // so it isn't mistaken for a real edit.
      suppressNextInputChangeRef.current = true;
    } catch (error) {
      console.error("Failed to send message:", error);
      // Rethrow rather than swallow: ChatInput's handleSubmit awaits this
      // function and, on rejection, takes its own catch path instead of
      // its success continuation - the one that resets `deliveryAttemptRef`
      // (its retry/idempotency identity for this attempt) and clears the
      // composer. Treating every send as successful here previously meant
      // a failure could still reset that identity, so a retry after a
      // delivery outcome that's actually unknown (e.g. the socket closing
      // after the server durably accepted the task) would get a brand new
      // attempt id instead of reusing the one the server may have already
      // seen - risking the same task being enqueued twice. ChatInput's
      // catch already shows the user-facing error toast, so this function
      // no longer needs its own.
      throw error;
    }
  };

  // Not currently reachable on this page (it never passes `prompts` to
  // ChatStartScreen), but marks the pick as dirty for consistency with
  // every other explicit "put this specific text in the composer" action -
  // if a starting-prompt grid is ever added here, a teammate pick must not
  // silently overwrite a prompt the user just chose.
  const handlePromptSelect = (prompt: string, highlights?: string[]) => {
    composerDirtyRef.current = true;
    setInputValue(prompt);
    setPromptHighlightTerms(highlights || []);
  };

  const handleInputChange = (value: string) => {
    if (suppressNextInputChangeRef.current) {
      // ChatInput's own post-submit reset, not a real edit - only reached
      // after a successful send now that handleSend rethrows on failure
      // (ChatInput takes its own catch path instead, which never calls
      // this). handleSend has already cleared the composer itself, so
      // this echo carries no new information - ignored outright rather
      // than reapplied, since reapplying it would still be harmless here
      // but there's nothing for it to usefully do.
      suppressNextInputChangeRef.current = false;
      return;
    }
    composerDirtyRef.current = true;
    setInputValue(value);
  };

  // Clears the composer only if we're the ones who last wrote to it -
  // never clobber whatever the user has since typed.
  const clearComposerIfOurs = () => {
    if (!composerDirtyRef.current) {
      setInputValue("");
      setPromptHighlightTerms([]);
    }
  };

  // Picking a teammate assigns them as the task's lead and, when the
  // composer is still untouched, fills it with their suggested prompt (or
  // clears it, if they don't have one) - clicking the already-selected
  // pill again clears both. Only one teammate can lead a task, so this
  // always replaces rather than appending to `selectedAgents`. Re-clicking
  // the same pill is the only supported way to deselect - there's no
  // visible "remove" affordance in the composer, since the hero swap
  // above already shows who's leading.
  const handleAgentClick = (agent: AgentCard) => {
    if (selectedAgents[0]?.id === agent.id) {
      setSelectedAgents([]);
      clearComposerIfOurs();
      return;
    }

    // Cleared here, in the same synchronous batch as setSelectedAgents,
    // not left to the separate config-fetch effect: that effect only
    // runs after this render commits and paints, so relying on it alone
    // leaves a real render where `selectedAgents` already names the new
    // lead while `taskConfig` (read straight from `selectedAgentConfig`
    // at submit time) still holds the previous lead's - a submit landing
    // in exactly that window would send the new agent's id with the old
    // agent's model/executionMode.
    setSelectedAgentConfig(undefined);
    setSelectedAgents([agent]);
    if (composerDirtyRef.current) {
      // The user's own text stays exactly as they left it.
      return;
    }

    setInputValue(firstNonBlankPrompt(agent.suggested_prompts) || "");
    setPromptHighlightTerms([]);
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
            agentsError={agentsError}
            onRetryAgents={handleRetryAgents}
            onAgentClick={handleAgentClick}
            selectedAgents={resolvedSelectedAgents}
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
