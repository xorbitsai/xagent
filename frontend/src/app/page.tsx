"use client";

import React from "react";
import { useRouter } from "next/navigation";
import {
  ChevronRight, Layers, Bot, Database,
  Sparkles, Play, Heart, Clock, Send, ListChecks, Loader2, Mic, Square
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import Link from "next/link";
import { useState, useEffect, useRef } from "react";
import { apiRequest, isJsonRecord, parseApiResponse } from "@/lib/api-wrapper";
import { cn, getApiUrl, resolveAgentLogoUrl } from "@/lib/utils";
import { formatDisplayDate } from "@/lib/time-utils";
import { resolveTaskLlmSelection } from "@/lib/models";
import { normalizeTaskPromptTitle, parseTaskCreateCore } from "@/lib/task-create";
import { useI18n } from "@/contexts/i18n-context";
import { useApp } from "@/contexts/app-context-chat";
import { WelcomeModal } from "@/components/welcome-modal";
import { getBrandingFromEnv } from "@/lib/branding";
import { useVoiceInputControls } from "@/components/voice-input-controller";
import * as homePageExtensionModule from "@/lib/home-page-extension";
import { HomePageExtension } from "@/lib/home-page-extension";
import type { HomeGetStartedDestinationOverrides } from "@/lib/page-extension-contracts";
import { toast } from "@/components/ui/sonner";

interface HomeTemplateConnection {
  name: string;
  logo: string | null;
}

interface HomeTemplateCard {
  id: string;
  name: string;
  category: string;
  description: string;
  features: string[];
  connections: HomeTemplateConnection[];
  setup_time: string;
  likes: number;
  used_count: number;
  type: string;
}

interface RecentTask {
  task_id: number;
  title: string;
  created_at?: string | null;
  agent_name?: string;
  agent_logo_url?: string | null;
}

function isSafeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value);
}

function decodeHomeTemplateCard(value: unknown): HomeTemplateCard | null {
  if (
    !isJsonRecord(value) ||
    typeof value.id !== "string" ||
    typeof value.name !== "string" ||
    typeof value.category !== "string" ||
    typeof value.description !== "string" ||
    !Array.isArray(value.features) ||
    !value.features.every((feature) => typeof feature === "string") ||
    !Array.isArray(value.connections) ||
    typeof value.setup_time !== "string" ||
    !isSafeInteger(value.likes) ||
    !isSafeInteger(value.used_count)
  ) return null;

  const connections: HomeTemplateConnection[] = [];
  for (const connection of value.connections) {
    if (
      !isJsonRecord(connection) ||
      typeof connection.name !== "string" ||
      (typeof connection.logo !== "string" && connection.logo !== null)
    ) return null;
    connections.push({ name: connection.name, logo: connection.logo });
  }

  return {
    id: value.id,
    name: value.name,
    category: value.category,
    description: value.description,
    features: [...value.features],
    connections,
    setup_time: value.setup_time,
    likes: value.likes,
    used_count: value.used_count,
    // Optional on the wire (older cached responses, other clients); default
    // to "agent" like the backend does everywhere else.
    type: typeof value.type === "string" ? value.type : "agent",
  };
}

function decodeHomeTemplates(value: unknown): HomeTemplateCard[] | null {
  if (!Array.isArray(value)) return null;
  const templates: HomeTemplateCard[] = [];
  for (const template of value) {
    const decoded = decodeHomeTemplateCard(template);
    if (!decoded) return null;
    // The home page's "Use Template" button only knows how to build a
    // single agent (/build/new?template=). Workforce templates need the
    // dedicated instantiation flow on the Templates page instead, so they
    // are left out of this surface rather than silently producing a
    // broken, empty-instruction agent.
    if (decoded.type === "workforce") continue;
    templates.push(decoded);
  }
  return templates;
}

function decodeRecentTask(value: unknown): RecentTask | null {
  if (
    !isJsonRecord(value) ||
    !isSafeInteger(value.task_id) ||
    value.task_id <= 0 ||
    typeof value.title !== "string" ||
    (value.created_at !== undefined && value.created_at !== null && typeof value.created_at !== "string") ||
    (value.agent_name !== undefined && typeof value.agent_name !== "string") ||
    (value.agent_logo_url !== undefined && value.agent_logo_url !== null && typeof value.agent_logo_url !== "string")
  ) return null;

  return {
    task_id: value.task_id,
    title: value.title,
    created_at: value.created_at,
    agent_name: value.agent_name,
    agent_logo_url: value.agent_logo_url,
  };
}

function decodeRecentTasks(value: unknown): RecentTask[] | null {
  if (!isJsonRecord(value) || !Array.isArray(value.tasks)) return null;
  const tasks: RecentTask[] = [];
  for (const task of value.tasks) {
    const decoded = decodeRecentTask(task);
    if (!decoded) return null;
    tasks.push(decoded);
  }
  return tasks;
}

// `homeGetStartedDestinationOverrides` is an OPTIONAL export of the replaceable
// home-page-extension module: a replacement that only implements the required
// `HomePageExtension` export must still build, so this is read through the
// module namespace with a fallback rather than a static named import.
const homeGetStartedDestinationOverrides: HomeGetStartedDestinationOverrides =
  (homePageExtensionModule as { homeGetStartedDestinationOverrides?: HomeGetStartedDestinationOverrides })
    .homeGetStartedDestinationOverrides ?? {}

const defaultHomeGetStartedDestinations: Record<keyof HomeGetStartedDestinationOverrides, string | null> = {
  video: null,
  docs: "https://docs.xagent.co/api-reference/introduction",
  guides: "https://docs.xagent.co/models/overview",
  whatsNew: "https://docs.xagent.co/release-notes",
}

function resolveHomeGetStartedDestination(
  configured: unknown,
  defaultDestination: string | null,
): string | null {
  if (configured === undefined) return defaultDestination
  if (typeof configured !== "string" || configured.trim().length === 0) return null
  return configured
}

export default function Home() {
  const router = useRouter();
  const { t, locale } = useI18n();
  const { setPendingMessage, setTaskId } = useApp();
  const branding = getBrandingFromEnv();
  const [templates, setTemplates] = useState<HomeTemplateCard[]>([]);
  const [recentTasks, setRecentTasks] = useState<RecentTask[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [showNoModelAlert, setShowNoModelAlert] = useState(false);
  const [visibleGetStartedVideos, setVisibleGetStartedVideos] = useState<Set<number>>(new Set());
  const getStartedSectionRef = useRef<HTMLDivElement | null>(null);
  const homeChatInputRef = useRef<HTMLTextAreaElement | null>(null);
  const mountedRef = useRef(false);
  const activeTaskCreateAttemptRef = useRef<number | null>(null);
  const taskCreateCounterRef = useRef(0);
  const draftRevisionRef = useRef(0);
  const homeVoiceInput = useVoiceInputControls();

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; activeTaskCreateAttemptRef.current = null; };
  }, []);

  useEffect(() => {
    let active = true;
    const isCurrent = () => active;

    const fetchTemplates = async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/templates/?lang=${locale}`);
        if (!isCurrent()) return;
        if (!response.ok) throw new Error(`Template request failed: ${response.status}`);

        const parsed = await parseApiResponse(response);
        if (!isCurrent()) return;
        const decoded = decodeHomeTemplates(parsed.data);
        if (!decoded) throw new Error("Invalid template response");

        setTemplates(decoded.slice(0, 3));
      } catch (error) {
        if (isCurrent()) {
          setTemplates([]);
          console.error("Failed to fetch templates:", error);
        }
      }
    };

    void fetchTemplates();
    return () => { active = false; };
  }, [locale]);

  useEffect(() => {
    let active = true;

    const fetchRecentTasks = async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/chat/tasks?page=1&per_page=5`);
        if (!active) return;
        if (!response.ok) throw new Error(`Recent task request failed: ${response.status}`);

        const parsed = await parseApiResponse(response);
        if (!active) return;
        const decoded = decodeRecentTasks(parsed.data);
        if (!decoded) throw new Error("Invalid recent task response");

        setRecentTasks(decoded);
      } catch (error) {
        if (active) console.error("Failed to fetch recent tasks:", error);
      }
    };

    void fetchRecentTasks();
    return () => { active = false; };
  }, []);

  useEffect(() => {
    const section = getStartedSectionRef.current;
    if (!section || typeof IntersectionObserver === "undefined") {
      setVisibleGetStartedVideos(new Set([0, 1]));
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        setVisibleGetStartedVideos((prev) => {
          const next = new Set(prev);
          let changed = false;

          for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            const index = Number((entry.target as HTMLElement).dataset.videoIndex);
            if (!Number.isNaN(index) && !next.has(index)) {
              next.add(index);
              changed = true;
            }
          }

          return changed ? next : prev;
        });
      },
      {
        rootMargin: "200px 0px",
        threshold: 0.1,
      }
    );

    const targets = section.querySelectorAll<HTMLElement>("[data-get-started-video='true']");
    targets.forEach((target) => observer.observe(target));

    return () => observer.disconnect();
  }, []);

  const handleUseTemplate = async (templateId: string) => {
    try {
      await apiRequest(`${getApiUrl()}/api/templates/${templateId}/use`, { method: "POST" });
    } catch (error) {
      console.error("Failed to record template usage:", error);
    }
    router.push(`/build/new?template=${templateId}`);
  };

  const handleCreateTask = async (content: string) => {
    const submittedPrompt = content.trim();
    if (!submittedPrompt) return;
    if (activeTaskCreateAttemptRef.current !== null) return;

    const attempt = ++taskCreateCounterRef.current;
    activeTaskCreateAttemptRef.current = attempt;
    const revision = draftRevisionRef.current;
    const isCurrent = () => mountedRef.current && activeTaskCreateAttemptRef.current === attempt;

    setIsCreating(true);

    try {
      let task = null;

      try {
        const selection = await resolveTaskLlmSelection();
        if (!isCurrent()) return;

        if (selection.kind === "no_model") {
          setShowNoModelAlert(true);
          return;
        }
        if (selection.kind === "operational_error") throw selection.error;

        const taskResponse = await apiRequest(`${getApiUrl()}/api/chat/task/create`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            title: normalizeTaskPromptTitle(submittedPrompt),
            description: submittedPrompt,
            llm_ids: selection.llmIds,
          }),
        });

        if (!isCurrent()) return;
        const parsed = await parseApiResponse(taskResponse);
        if (!isCurrent()) return;

        task = taskResponse.ok ? parseTaskCreateCore(parsed.data) : null;
        if (!task) throw new Error("Failed to create task");
      } catch (error) {
        if (isCurrent()) {
          console.error("Failed to create task:", error);
          toast.error(t("common.errors.taskFailed"));
        }
        return;
      }

      if (!isCurrent() || !task) return;

      try {
        if (!isCurrent()) return;
        setPendingMessage({
          message: submittedPrompt,
          files: [],
          targetTaskId: task.taskId,
        });

        if (!isCurrent()) return;
        setTaskId(task.taskId);

        if (!isCurrent()) return;
        if (draftRevisionRef.current === revision && homeChatInputRef.current) {
          homeChatInputRef.current.value = "";
          homeChatInputRef.current.style.height = "auto";
          draftRevisionRef.current += 1;
        }
      } catch (error) {
        if (isCurrent()) console.error("Failed to commit task creation:", error);
      }
    } finally {
      if (isCurrent()) {
        activeTaskCreateAttemptRef.current = null;
        setIsCreating(false);
      }
    }
  };

  const handleChatButtonClick = () => {
    const val = homeChatInputRef.current?.value;
    if (val && val.trim()) {
      handleCreateTask(val);
    }
  };

  const homeVoiceInputLabel =
    homeVoiceInput.status === "recording"
      ? t("voiceInput.stop")
      : homeVoiceInput.status === "transcribing"
        ? t("voiceInput.transcribing")
        : t("voiceInput.start");
  const homeVoiceInputDisabled =
    homeVoiceInput.status === "transcribing" ||
    (homeVoiceInput.status === "idle" && isCreating);
  const handleHomeVoiceInputClick = () => {
    if (homeVoiceInput.status === "recording") {
      homeVoiceInput.stopRecording();
      return;
    }
    if (homeVoiceInput.status === "idle") {
      homeVoiceInput.startRecording(homeChatInputRef.current);
    }
  };

  return (
    <div className="h-full flex flex-col overflow-hidden bg-[#FAFAFA] dark:bg-background overflow-y-auto">
      <WelcomeModal />
      {/* Hero Section */}
      <div className="relative shrink-0 flex items-center justify-center overflow-hidden py-14 px-8 sm:px-16 bg-[linear-gradient(160deg,hsl(230_72%_10%)_0%,hsl(234_62%_15%)_35%,hsl(255_60%_17%)_70%,hsl(262_55%_13%)_100%)]">
        {/* grid background */}
        <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(rgba(255,255,255,0.028)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.028)_1px,transparent_1px)] bg-[size:48px_48px]" />
        {/* central orb */}
        <div className="absolute w-[700px] h-[340px] rounded-full bg-[radial-gradient(ellipse,hsl(234_80%_55%/0.18)_0%,transparent_70%)] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none" />

        <div className="z-10 flex flex-col items-center w-full max-w-3xl">
          <img src={branding.whiteLogoPath} alt={branding.appName} className="w-14 h-14 mb-6 object-contain rounded-[16px] shadow-2xl" />
          <h1 className="text-[34px] font-extrabold text-white mb-3 tracking-tight text-center leading-[1.15]">{t("home.hero.title", { appName: branding.appName })}</h1>
          <p className="text-[13.5px] text-gray-400 text-center mb-8 font-medium max-w-xl leading-[1.7]">
            {t("home.hero.subtitle")}
          </p>

          <div className="flex flex-wrap justify-center items-center gap-1.5 sm:gap-2 bg-[hsl(234_30%_25%/0.4)] rounded-full border border-[hsl(234_30%_35%)] p-1.5 mb-10 backdrop-blur-md">
            <Link href="/templates" className="flex items-center gap-2 px-4 py-2 rounded-full hover:bg-[hsl(234_30%_35%)] text-white transition-colors text-[14px] font-semibold">
              <Layers className="w-4 h-4" /> <span className="hidden sm:inline">{t("nav.templates")}</span>
            </Link>
            <div className="w-px h-5 bg-[hsl(234_30%_40%)] mx-0.5 sm:mx-1 hidden sm:block" />
            <Link href="/build" className="flex items-center gap-2 px-4 py-2 rounded-full hover:bg-[hsl(234_30%_35%)] text-white transition-colors text-[14px] font-semibold">
              <Bot className="w-4 h-4" /> <span className="hidden sm:inline">{t("nav.build")}</span>
            </Link>
            <div className="w-px h-5 bg-[hsl(234_30%_40%)] mx-0.5 sm:mx-1 hidden sm:block" />
            <Link href="/task" className="flex items-center gap-2 px-4 py-2 rounded-full bg-[hsl(234_40%_40%)] hover:bg-[hsl(234_40%_45%)] text-white transition-colors text-[14px] font-semibold shadow-sm">
              <Sparkles className="w-4 h-4" /> <span className="hidden sm:inline">{t("nav.task")}</span>
            </Link>
            <div className="w-px h-5 bg-[hsl(234_30%_40%)] mx-0.5 sm:mx-1 hidden sm:block" />
            <Link href="/kb" className="flex items-center gap-2 px-4 py-2 rounded-full hover:bg-[hsl(234_30%_35%)] text-white transition-colors text-[14px] font-semibold">
              <Database className="w-4 h-4" /> <span className="hidden sm:inline">{t("nav.knowledgeBase")}</span>
            </Link>
          </div>

          <div className="w-full max-w-2xl bg-[hsl(234_30%_25%/0.4)] border border-[hsl(234_30%_35%)] rounded-[18px] p-3 flex items-end shadow-[0_12px_40px_rgba(0,0,0,0.25)] backdrop-blur-md focus-within:border-[hsl(234_50%_50%)] focus-within:shadow-[0_0_0_4px_hsl(234_50%_50%/0.2),0_12px_40px_rgba(0,0,0,0.25)] transition-all duration-200">
            <textarea
              ref={homeChatInputRef}
              data-voice-input="false"
              placeholder={t("home.hero.searchPlaceholder")}
              className="border-0 bg-transparent text-white text-[16px] leading-relaxed placeholder:text-[hsl(240_5%_60%)] focus-visible:ring-0 focus-visible:outline-none flex-1 resize-none overflow-hidden min-h-[28px] max-h-[120px] py-1 px-2"
              rows={1}
              onInput={(e) => {
                draftRevisionRef.current += 1;
                const target = e.target as HTMLTextAreaElement;
                target.style.height = "auto";
                target.style.height = Math.min(target.scrollHeight, 120) + "px";
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (e.currentTarget.value.trim() && !isCreating) {
                    handleCreateTask(e.currentTarget.value);
                  }
                }
              }}
            />
            {homeVoiceInput.hasAsrModel && (
              <Button
                type="button"
                size="icon"
                variant="ghost"
                aria-label={homeVoiceInputLabel}
                title={homeVoiceInputLabel}
                className={`shrink-0 w-9 h-9 ml-2 rounded-full transition-colors ${
                  homeVoiceInput.status === "recording"
                    ? "bg-red-500 text-white hover:bg-red-600 hover:text-white"
                    : "text-white/70 hover:bg-[hsl(234_30%_35%)] hover:text-white"
                } ${homeVoiceInput.status === "transcribing" ? "cursor-wait opacity-80" : ""}`}
                disabled={homeVoiceInputDisabled}
                onMouseDown={(event) => event.preventDefault()}
                onClick={handleHomeVoiceInputClick}
              >
                {homeVoiceInput.status === "recording" ? (
                  <Square className="w-3.5 h-3.5 fill-current" />
                ) : homeVoiceInput.status === "transcribing" ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Mic className="w-4 h-4" />
                )}
              </Button>
            )}
            <Button
              size="icon"
              className="bg-[hsl(234_40%_40%)] hover:bg-[hsl(234_40%_45%)] text-white rounded-[12px] shrink-0 w-9 h-9 ml-3 transition-colors shadow-none disabled:opacity-50"
              onClick={handleChatButtonClick}
              disabled={isCreating}
            >
              {isCreating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </Button>
          </div>
        </div>
      </div>
      <AlertDialog open={showNoModelAlert} onOpenChange={setShowNoModelAlert}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("common.notice")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("chatPage.input.noModelAlert")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction onClick={() => router.push("/models")}>
              {t("common.confirm")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Main Content Scrollable */}
      <div className="flex-1">
        <div className="mx-auto py-9 px-8 sm:px-14">

          {/* Get Started Section */}
          <h2 className="text-[16px] font-bold mb-4 text-foreground">{t("home.getStarted.title")}</h2>
          <div ref={getStartedSectionRef} className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 mb-10">
            {[
              { title: t("home.getStarted.video.title"), desc: t("home.getStarted.video.description", { appName: branding.appName }), video: "/videos/Tutorial.mp4", link: resolveHomeGetStartedDestination(homeGetStartedDestinationOverrides.video, defaultHomeGetStartedDestinations.video) },
              { title: t("home.getStarted.docs.title"), desc: t("home.getStarted.docs.description"), video: "/videos/Documentation.mp4", link: resolveHomeGetStartedDestination(homeGetStartedDestinationOverrides.docs, defaultHomeGetStartedDestinations.docs) },
              { title: t("home.getStarted.guides.title"), desc: t("home.getStarted.guides.description"), icon: <ListChecks className="w-8 h-8 text-green-500" />, bg: "bg-green-50 dark:bg-green-950/30", link: resolveHomeGetStartedDestination(homeGetStartedDestinationOverrides.guides, defaultHomeGetStartedDestinations.guides) },
              { title: t("home.getStarted.whatsNew.title"), desc: t("home.getStarted.whatsNew.description"), icon: <Sparkles className="w-8 h-8 text-orange-500" />, bg: "bg-orange-50 dark:bg-orange-950/30", link: resolveHomeGetStartedDestination(homeGetStartedDestinationOverrides.whatsNew, defaultHomeGetStartedDestinations.whatsNew) }
            ].map((card, i) => {
              const shouldLoadVideo = card.video ? visibleGetStartedVideos.has(i) : false;
              const isLinked = typeof card.link === "string";
              const cardContent = (
                <Card className={cn(
                  "py-0 gap-0 overflow-hidden border-border/60 transition-all duration-300 bg-card rounded-xl flex flex-col h-full",
                  isLinked && "hover:shadow-md group cursor-pointer",
                )}>
                  <div
                    className={`h-[180px] relative flex items-center justify-center overflow-hidden ${card.video ? 'bg-muted' : card.bg}`}
                    data-get-started-video={card.video ? "true" : undefined}
                    data-video-index={card.video ? String(i) : undefined}
                  >
                    {card.video ? (
                      shouldLoadVideo ? (
                        <video
                          src={card.video}
                          autoPlay
                          loop
                          muted
                          playsInline
                          preload="metadata"
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,hsl(231_55%_62%/0.35),transparent_55%),linear-gradient(160deg,hsl(229_39%_16%)_0%,hsl(236_42%_20%)_100%)]">
                          <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.04)_1px,transparent_1px)] bg-[size:28px_28px]" />
                          <div className="relative z-10 flex h-full items-center justify-center text-white/85">
                            <Play className="h-10 w-10 fill-current" />
                          </div>
                        </div>
                      )
                    ) : (
                      <div className={cn(
                        "transition-transform duration-300",
                        isLinked && "group-hover:scale-110",
                      )}>
                        {card.icon}
                      </div>
                    )}
                  </div>
                  <CardContent className="p-4 flex-1">
                    <h3 className={cn(
                      "font-semibold text-[13px] mb-1 transition-colors",
                      isLinked && "group-hover:text-primary",
                    )}>{card.title}</h3>
                    <p className="text-[12px] text-muted-foreground leading-relaxed">{card.desc}</p>
                  </CardContent>
                </Card>
              );

              return isLinked ? (
                <a key={i} href={card.link!} target="_blank" rel="noopener noreferrer" className="block rounded-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background">
                  {cardContent}
                </a>
              ) : (
                <div key={i} className="block">
                  {cardContent}
                </div>
              );
            })}
          </div>

          {/* Build agents with templates */}
          {templates.length > 0 && (
            <>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-[16px] font-bold text-foreground">{t("home.templates.title")}</h2>
                <Link href="/templates" className="text-[14px] font-semibold text-primary hover:underline flex items-center group">
                  {t("home.templates.viewAll")} <ChevronRight className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" />
                </Link>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 mb-10">
                {templates.map(template => (
                  <Card key={template.id} className="flex flex-col border-border/60 hover:shadow-md transition-all duration-300 p-5 group bg-card rounded-xl">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-[11px] font-bold text-primary tracking-wider uppercase bg-primary/10 px-2.5 py-1 rounded-md">
                        {template.category}
                      </span>
                      <div className="flex items-center gap-1.5 text-muted-foreground text-xs font-medium">
                        <Clock className="w-3.5 h-3.5" />
                        <span>{template.setup_time || t("home.templates.setupTime", { time: "5 min" })}</span>
                      </div>
                    </div>
                    <h3 className="font-bold text-[15px] text-foreground group-hover:text-primary transition-colors line-clamp-1">
                      {template.name}
                    </h3>
                    <div className="flex-1 space-y-2.5">
                      {(template.features && template.features.length > 0) ? (
                        template.features.slice(0, 3).map((feature: string, idx: number) => (
                          <div key={idx} className="flex items-start gap-2 text-[14px] text-muted-foreground">
                            <ChevronRight className="w-4 h-4 text-primary shrink-0 mt-0.5 opacity-70" />
                            <span className="line-clamp-2 leading-snug">{feature}</span>
                          </div>
                        ))
                      ) : (
                        <div className="flex items-start gap-2 text-[14px] text-muted-foreground">
                          <ChevronRight className="w-4 h-4 text-primary shrink-0 mt-0.5 opacity-70" />
                          <span className="line-clamp-3 leading-snug">{template.description}</span>
                        </div>
                      )}
                    </div>
                    <div className="h-[1px] bg-border/60" />
                    <div className="mt-auto">
                      <div className="flex items-center justify-between text-sm text-muted-foreground mb-5">
                        <div className="flex items-center">
                          {template.connections && template.connections.length > 0 ? (
                            <div className="flex gap-1.5">
                              {template.connections.slice(0, 4).map((conn, idx: number) => (
                                <div key={idx} className="w-8 h-8 rounded-lg bg-background border border-border flex items-center justify-center overflow-hidden shadow-sm">
                                  {conn.logo ? <img src={conn.logo} alt={conn.name} className="w-5 h-5 object-contain" /> : <span className="text-[10px] font-bold text-primary/70">{(conn.name || "").substring(0, 2).toUpperCase()}</span>}
                                </div>
                              ))}
                            </div>
                          ) : <div className="h-8" />}
                        </div>
                        <div className="flex items-center gap-4">
                          <div className="flex items-center gap-1.5">
                            <Play className="w-3.5 h-3.5 fill-current text-primary/60" />
                            <span className="font-semibold text-foreground/80">{template.used_count || 0}</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <Heart className="w-3.5 h-3.5 fill-current text-rose-400/70" />
                            <span className="font-semibold text-foreground/80">{template.likes || 0}</span>
                          </div>
                        </div>
                      </div>
                      <button
                        onClick={() => handleUseTemplate(template.id)}
                        className="w-full py-2.5 text-primary text-[13px] font-bold uppercase tracking-wide rounded-xl border border-primary/20 hover:bg-primary hover:text-primary-foreground transition-all duration-300"
                      >
                        {t("home.templates.useTemplate")}
                      </button>
                    </div>
                  </Card>
                ))}
              </div>
            </>
          )}

          {/* Recent Tasks */}
          {recentTasks.length > 0 && (
            <>
              <h2 className="text-[16px] font-bold mb-4 text-foreground">{t("home.recent.title")}</h2>
              <div className="space-y-3">
                {recentTasks.map((task) => {
                  const resolvedLogoUrl = resolveAgentLogoUrl(task.agent_logo_url, getApiUrl());
                  const displayDate = formatDisplayDate(task.created_at, locale, {
                    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
                  });
                  return (
                  <Link key={task.task_id} href={`/task/${task.task_id}`} className="flex items-center justify-between p-4 rounded-2xl border border-border/60 bg-card hover:border-primary/30 hover:shadow-md transition-all duration-300 group">
                    <div className="flex items-center gap-5">
                      <div className="w-12 h-12 rounded-xl bg-primary/5 flex items-center justify-center shrink-0 border border-primary/10">
                        {resolvedLogoUrl ? (
                          <img src={resolvedLogoUrl} alt="Agent" className="w-7 h-7 rounded object-cover" />
                        ) : (
                          <Bot className="w-6 h-6 text-primary/80" />
                        )}
                      </div>
                      <div>
                        <h4 className="font-semibold text-[16px] group-hover:text-primary transition-colors">{task.title || t("home.recent.untitledTask")}</h4>
                        <p className="text-[13px] text-muted-foreground mt-0.5 font-medium">
                          {task.agent_name || t("home.recent.defaultAgent")}{displayDate ? ` • ${displayDate}` : ""}
                        </p>
                      </div>
                    </div>
                    <div className="w-8 h-8 rounded-full bg-accent/50 flex items-center justify-center group-hover:bg-primary group-hover:text-primary-foreground transition-all duration-300 mr-2">
                      <ChevronRight className="w-4 h-4" />
                    </div>
                  </Link>
                  );
                })}
              </div>
            </>
          )}

        </div>
      </div>
      <div data-slot="home-page-extension" className="shrink-0">
        <HomePageExtension />
      </div>
    </div>
  );
}
