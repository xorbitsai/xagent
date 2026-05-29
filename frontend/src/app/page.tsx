"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
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
import { Button } from "@/components/ui/button";
import {
  ArrowUpRight,
  Bot,
  ChevronRight,
  File as FileIcon,
  FileImage,
  FileText,
  Loader2,
  Paperclip,
  Presentation,
  Plus,
  Search,
  Send,
  Sparkles,
  Table2,
  X,
} from "lucide-react";
import { apiRequest } from "@/lib/api-wrapper";
import { HomeTemplateCard } from "@/components/templates/home-template-card";
import { useApp } from "@/contexts/app-context-chat";
import { useAuth } from "@/contexts/auth-context";
import { useI18n } from "@/contexts/i18n-context";
import { getAgentChatHref } from "@/lib/agent-ui-access";
import { getApiUrl } from "@/lib/utils";
import type { Template } from "@/types/template";
import { WelcomeModal } from "@/components/welcome-modal";

interface RecentTask {
  task_id: number | string;
  title?: string | null;
  agent_name?: string | null;
  agent_logo_url?: string | null;
  status?: "completed" | "running" | "failed" | "pending" | "paused" | "waiting_for_user" | string;
  created_at: string;
}

function getGreetingTranslationKey(hour: number) {
  if (hour < 12) return "home.revamp.greetingMorning";
  if (hour < 18) return "home.revamp.greetingAfternoon";
  return "home.revamp.greetingEvening";
}

interface LlmModel {
  model_id: string;
  is_default?: boolean;
}

interface DefaultModelRecord {
  config_type?: "general" | "small_fast" | "visual" | "compact";
  model?: {
    model_id?: string;
  } | null;
}

interface HomeAgent {
  id: number;
  name: string;
  description?: string | null;
  logo_url?: string | null;
  status?: string;
  updated_at?: string;
}

const parseDateMs = (value: string) => {
  const timestamp = new Date(value).getTime();
  return Number.isNaN(timestamp) ? null : timestamp;
};

const formatDateTime = (value: string, locale: string) => {
  const timestamp = parseDateMs(value);
  if (timestamp === null) return "";

  return new Date(timestamp).toLocaleDateString(locale === "zh" ? "zh-CN" : "en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const formatElapsed = (createdAtMs: number, now: number) => {
  const elapsedSeconds = Math.max(0, Math.floor((now - createdAtMs) / 1000));
  const minutes = Math.floor(elapsedSeconds / 60);
  const seconds = elapsedSeconds % 60;
  return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
};

function SectionHeader({
  title,
  subtitle,
  actionLabel,
  href,
}: {
  title: string;
  subtitle: string;
  actionLabel: string;
  href: string;
}) {
  return (
    <div className="flex items-start justify-between gap-4">
      <div>
        <h2 className="text-[22px] font-bold tracking-tight text-foreground">{title}</h2>
        <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p>
      </div>
      <Link
        href={href}
        className="inline-flex items-center gap-1 rounded-full border border-border/70 bg-background px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:border-primary/20 hover:text-primary"
      >
        {actionLabel}
        <ChevronRight className="h-4 w-4" />
      </Link>
    </div>
  );
}

export default function Home() {
  const router = useRouter();
  const { t, locale } = useI18n();
  const { setPendingMessage, setTaskId } = useApp();
  const { user } = useAuth();
  const [templates, setTemplates] = useState<Template[]>([]);
  const [recentTasks, setRecentTasks] = useState<RecentTask[]>([]);
  const [agents, setAgents] = useState<HomeAgent[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [showNoModelAlert, setShowNoModelAlert] = useState(false);
  const [currentHour, setCurrentHour] = useState<number | null>(null);
  const [homeInputValue, setHomeMessageValue] = useState("");
  const [homeFiles, setHomeFiles] = useState<File[]>([]);
  const [now, setNow] = useState(() => Date.now());
  const homeChatInputRef = useRef<HTMLTextAreaElement | null>(null);
  const homeFileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [templatesRes, tasksRes, agentsRes] = await Promise.all([
          apiRequest(`${getApiUrl()}/api/templates/?lang=${locale}`),
          apiRequest(`${getApiUrl()}/api/chat/tasks?page=1&per_page=20`),
          apiRequest(`${getApiUrl()}/api/agents`),
        ]);

        if (templatesRes.ok) {
          const data = await templatesRes.json();
          setTemplates(Array.isArray(data) ? data : []);
        }

        if (tasksRes.ok) {
          const data = await tasksRes.json();
          setRecentTasks((data.tasks || (Array.isArray(data) ? data : [])) as RecentTask[]);
        }

        if (agentsRes.ok) {
          const data = await agentsRes.json();
          setAgents(Array.isArray(data) ? data : []);
        }
      } catch (error) {
        console.error("Failed to fetch data", error);
      }
    };
    fetchData();
  }, [locale]);

  useEffect(() => {
    setCurrentHour(new Date().getHours());
  }, []);

  const handleUseTemplate = async (templateId: string) => {
    try {
      await apiRequest(`${getApiUrl()}/api/templates/${templateId}/use`, { method: "POST" });
    } catch (error) {
      console.error("Failed to record template usage:", error);
    }
    router.push(`/build/new?template=${templateId}`);
  };

  const resolveTaskLlmIds = async (): Promise<[string, string | null, string | null, string | null] | null> => {
    const apiUrl = getApiUrl();
    const [modelsResponse, defaultResponse] = await Promise.all([
      apiRequest(`${apiUrl}/api/models/?category=llm`, { headers: {} }),
      apiRequest(`${apiUrl}/api/models/user-default`, { headers: {} }),
    ]);

    let allModels: LlmModel[] = [];
    if (modelsResponse.ok) {
      const modelsData = await modelsResponse.json();
      if (Array.isArray(modelsData)) {
        allModels = modelsData as LlmModel[];
      }
    }

    const defaultModels: Record<string, string | undefined> = {};
    if (defaultResponse.ok) {
      const defaultsData = await defaultResponse.json();
      if (Array.isArray(defaultsData)) {
        defaultsData.forEach((defaultConfig: DefaultModelRecord) => {
          if (defaultConfig?.config_type && defaultConfig.model?.model_id) {
            defaultModels[defaultConfig.config_type] = defaultConfig.model.model_id;
          }
        });
      }
    }

    const generalModelId =
      defaultModels.general ||
      allModels.find((model) => model.is_default)?.model_id ||
      allModels[0]?.model_id;

    if (!generalModelId) {
      return null;
    }

    return [
      generalModelId,
      defaultModels.small_fast ?? null,
      defaultModels.visual ?? null,
      defaultModels.compact ?? null,
    ];
  };

  const handleCreateTask = async (content: string, filesToSend: File[] = homeFiles) => {
    if (isCreating) return;
    setIsCreating(true);
    try {
      const llmIds = await resolveTaskLlmIds();
      if (!llmIds) {
        setShowNoModelAlert(true);
        return;
      }

      const normalizedContent = content.trim() || t("home.revamp.fileOnlyPrompt");

      const requestBody = {
        title: normalizedContent,
        description: normalizedContent,
        llm_ids: llmIds,
      };

      const taskResponse = await apiRequest(`${getApiUrl()}/api/chat/task/create`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (taskResponse.ok) {
        const taskData = await taskResponse.json();
        const taskId = taskData.id || taskData.task_id;

        if (taskId) {
          const parsedTaskId = typeof taskId === "string" ? parseInt(taskId, 10) : taskId;

          setPendingMessage({
            message: normalizedContent,
            files: filesToSend,
            targetTaskId: parsedTaskId,
          });

          setHomeMessageValue("");
          setHomeFiles([]);
          setTaskId(parsedTaskId);
          router.push(`/task/${parsedTaskId}`);
        }
      } else {
        console.error("Failed to create task");
      }
    } catch (err) {
      console.error("Failed to send message:", err);
    } finally {
      setIsCreating(false);
    }
  };

  const handleChatButtonClick = () => {
    if (homeInputValue.trim() || homeFiles.length > 0) {
      handleCreateTask(homeInputValue, homeFiles);
    }
  };

  const setHomeInputValue = (value: string) => {
    const input = homeChatInputRef.current;
    if (!input) return;

    setHomeMessageValue(value);
    input.value = value;
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
    input.focus();
    input.setSelectionRange(value.length, value.length);
  };

  const handleHomeFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.target.files || []);
    if (selectedFiles.length > 0) {
      setHomeFiles((prev) => [...prev, ...selectedFiles]);
    }

    if (homeFileInputRef.current) {
      homeFileInputRef.current.value = "";
    }
  };

  const removeHomeFile = (indexToRemove: number) => {
    setHomeFiles((prev) => prev.filter((_, index) => index !== indexToRemove));
  };

  const capabilityPills = useMemo(
    () => [
      {
        id: "slides",
        label: t("home.revamp.capabilities.slides.label"),
        icon: Presentation,
        toneClassName: "bg-amber-400/15 text-amber-300",
        prompt: t("home.revamp.capabilities.slides.prompt"),
      },
      {
        id: "sheets",
        label: t("home.revamp.capabilities.sheets.label"),
        icon: Table2,
        toneClassName: "bg-emerald-400/15 text-emerald-300",
        prompt: t("home.revamp.capabilities.sheets.prompt"),
      },
      {
        id: "docs",
        label: t("home.revamp.capabilities.docs.label"),
        icon: FileText,
        toneClassName: "bg-blue-400/15 text-blue-300",
        prompt: t("home.revamp.capabilities.docs.prompt"),
      },
      {
        id: "pdf",
        label: t("home.revamp.capabilities.pdf.label"),
        icon: FileText,
        toneClassName: "bg-rose-400/15 text-rose-300",
        prompt: t("home.revamp.capabilities.pdf.prompt"),
      },
      {
        id: "image",
        label: t("home.revamp.capabilities.image.label"),
        icon: FileImage,
        toneClassName: "bg-fuchsia-400/15 text-fuchsia-300",
        prompt: t("home.revamp.capabilities.image.prompt"),
      },
      {
        id: "research",
        label: t("home.revamp.capabilities.research.label"),
        icon: Search,
        toneClassName: "bg-indigo-400/15 text-indigo-300",
        prompt: t("home.revamp.capabilities.research.prompt"),
      },
    ],
    [t]
  );

  const runningTasks = useMemo(
    () =>
      recentTasks
        .filter((task) => task.status === "running")
        .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()),
    [recentTasks]
  );
  const primaryTask = runningTasks[0];
  const primaryTaskCreatedAtMs = primaryTask ? parseDateMs(primaryTask.created_at) : null;

  useEffect(() => {
    if (!primaryTask) return;

    setNow(Date.now());
    const intervalId = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(intervalId);
  }, [primaryTask]);

  const elapsedLabel = primaryTaskCreatedAtMs === null ? null : formatElapsed(primaryTaskCreatedAtMs, now);
  const progressPercent = primaryTask
    && primaryTaskCreatedAtMs !== null
    ? Math.min(100, Math.max(8, Math.floor((now - primaryTaskCreatedAtMs) / 18000)))
    : 0;
  const displayAgents = useMemo(() => {
    const published = agents.filter((agent) => agent.status === "published");
    const source = published.length > 0 ? published : agents;
    return source.slice(0, 9);
  }, [agents]);
  const greetingLabel = t(
    currentHour === null ? "home.revamp.greeting" : getGreetingTranslationKey(currentHour)
  );
  const greetingText = user?.username ? `${greetingLabel}, ${user.username}` : greetingLabel;
  const canSubmitHomeTask = !isCreating && (homeInputValue.trim().length > 0 || homeFiles.length > 0);

  return (
    <div className="h-full overflow-y-auto bg-[#F5F7FB] dark:bg-background">
      <WelcomeModal />
      <div className="mx-auto flex w-full flex-col gap-8 p-6">
        {primaryTask ? (
          <Link
            href={`/task/${primaryTask.task_id}`}
            className="flex flex-wrap items-center gap-3 rounded-xl border border-[rgba(99,102,241,0.2)] bg-[linear-gradient(90deg,rgba(99,102,241,0.08),rgba(168,85,247,0.05))] px-4 py-3 text-[13px] shadow-sm transition-all hover:border-[rgba(99,102,241,0.35)]"
          >
            <span className="h-2 w-2 animate-blink rounded-full bg-primary shadow-[0_0_0_4px_rgba(59,90,246,0.18)]" />
            <span className="font-semibold text-foreground">{t("home.revamp.running", { count: runningTasks.length })}</span>
            <span className="text-muted-foreground">·</span>
            <span className="min-w-0 flex-1 truncate text-foreground/85">
              {primaryTask.title || t("home.recent.untitledTask")}
            </span>
            <div className="h-1 w-24 overflow-hidden rounded-full bg-slate-200/90">
              <div
                className="h-full rounded-full bg-[linear-gradient(90deg,#6366F1,#A855F7,#EC4899)]"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
            <span className="font-mono text-[11.5px] text-muted-foreground">{elapsedLabel}</span>
            <div className="inline-flex items-center gap-1 rounded-md border border-border/80 bg-background px-2.5 py-1 font-medium text-foreground">
              {t("home.revamp.liveView")}
              <ChevronRight className="h-3.5 w-3.5" />
            </div>
          </Link>
        ) : null}

        <section className="relative overflow-hidden rounded-[24px] bg-[linear-gradient(135deg,#1D0F4D_0%,#26115E_36%,#2B1769_64%,#22104E_100%)] px-8 py-8 text-white shadow-[0_18px_60px_rgba(28,18,89,0.22)]">
          <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.065)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.065)_1px,transparent_1px)] bg-[size:24px_24px] opacity-40" />
          <div className="absolute inset-y-0 right-0 w-[42%] bg-[radial-gradient(circle_at_center,rgba(120,119,255,0.16),transparent_60%)]" />

          <div className="relative z-10 flex flex-col gap-6">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-white/65">
              <Sparkles className="h-4 w-4 text-fuchsia-300" />
              <span>{greetingText}</span>
            </div>

            <div className="max-w-3xl">
              <h1 className="text-[42px] font-bold tracking-[-0.02em] text-white">
                {t("home.revamp.goalTitle")}
              </h1>
            </div>

            <div className="max-w-[780px] rounded-[16px] border border-white/10 bg-white/5 p-2.5 backdrop-blur-xl">
              <input
                ref={homeFileInputRef}
                type="file"
                multiple
                onChange={handleHomeFileSelect}
                className="hidden"
                accept=".pdf,.doc,.docx,.txt,.md,.csv,.json,.xlsx,.xls,.ppt,.pptx,.png,.jpg,.jpeg,.gif,.webp"
              />
              <div className="flex flex-col gap-3 rounded-[14px] bg-white px-4 py-3 shadow-[0_8px_28px_rgba(0,0,0,0.18)] sm:flex-row sm:items-center">
                <textarea
                  ref={homeChatInputRef}
                  value={homeInputValue}
                  placeholder={t("home.revamp.askPlaceholder")}
                  className="min-h-[26px] flex-1 resize-none border-0 bg-transparent py-1 text-[15px] leading-relaxed text-slate-900 placeholder:text-slate-400 focus-visible:outline-none focus-visible:ring-0"
                  rows={1}
                  onInput={(event) => {
                    const target = event.target as HTMLTextAreaElement;
                    setHomeMessageValue(target.value);
                    target.style.height = "auto";
                    target.style.height = `${Math.min(target.scrollHeight, 120)}px`;
                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      if ((event.currentTarget.value.trim() || homeFiles.length > 0) && !isCreating) {
                        handleCreateTask(event.currentTarget.value, homeFiles);
                      }
                    }
                  }}
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-9 w-9 shrink-0 rounded-[10px] text-slate-500 hover:bg-slate-100 hover:text-slate-700"
                  onClick={() => homeFileInputRef.current?.click()}
                >
                  <Paperclip className="h-4 w-4" />
                </Button>
                <Button
                  className="h-9 rounded-[10px] px-4 text-sm font-semibold shadow-none"
                  onClick={handleChatButtonClick}
                  disabled={!canSubmitHomeTask}
                >
                  {isCreating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                  {t("home.revamp.start")}
                </Button>
              </div>

              {homeFiles.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2 px-1">
                  {homeFiles.map((file, index) => (
                    <div
                      key={`${file.name}-${file.lastModified}-${index}`}
                      className="inline-flex h-8 items-center gap-2 rounded-md border border-white/15 bg-white/10 px-3 text-sm text-white/90 backdrop-blur-sm"
                    >
                      <FileIcon className="h-3.5 w-3.5" />
                      <span className="max-w-[220px] truncate font-medium">{file.name}</span>
                      <button
                        type="button"
                        onClick={() => removeHomeFile(index)}
                        className="rounded-sm p-0.5 text-white/65 transition-colors hover:bg-white/10 hover:text-white"
                        title={t("common.remove")}
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <div className="mt-3 flex flex-wrap gap-2 px-1">
                {capabilityPills.map((action) => (
                  <button
                    key={action.id}
                    type="button"
                    onClick={() => setHomeInputValue(action.prompt)}
                    className="inline-flex items-center gap-2 rounded-full border border-transparent bg-transparent px-[5px] py-[5px] text-[12px] font-medium text-white/80 transition-all hover:border-white/20 hover:bg-white/10 hover:text-white"
                  >
                    <span className={`grid h-[22px] w-[22px] place-items-center rounded-full ${action.toneClassName}`}>
                      <action.icon className="h-3 w-3" />
                    </span>
                    <span>{action.label}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="flex max-w-[780px] flex-col items-start justify-between gap-3 rounded-[12px] border border-white/10 bg-white/5 px-4 py-3 backdrop-blur-xl sm:flex-row sm:items-center">
              <div className="min-w-0">
                <p className="text-sm font-semibold text-white">{t("home.revamp.followupTitle")}</p>
                <p className="mt-1 text-sm text-white/65">{t("home.revamp.followupDescription")}</p>
              </div>
              <Button
                asChild
                variant="secondary"
                className="rounded-xl border border-white/15 bg-white px-4 text-slate-900 hover:bg-white/90"
              >
                <Link href="/build?create=true">
                  {t("home.revamp.buildAgent")}
                  <ArrowUpRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>
          </div>
        </section>

        <section className="space-y-5">
          <SectionHeader
            title={t("home.agents.title")}
            subtitle={t("home.agents.subtitle")}
            actionLabel={t("home.agents.manageAll")}
            href="/build"
          />

          <div className="py-2">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-5">
              <Link
                href="/build?create=true"
                className="group flex min-h-[76px] w-full cursor-pointer items-center gap-3 rounded-2xl border border-dashed border-primary/30 bg-white px-4 py-3 shadow-sm transition-all hover:border-primary/50 hover:bg-primary/5"
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[linear-gradient(135deg,#9B5BFF,#6C63FF)] text-white shadow-sm">
                  <Plus className="h-4 w-4" />
                </div>
                <div className="min-w-0">
                  <h3 className="truncate text-[14px] font-semibold text-foreground">{t("home.agents.newAgent")}</h3>
                  <p className="mt-0.5 text-[12px] text-muted-foreground">{t("home.agents.buildTime")}</p>
                </div>
              </Link>

              {displayAgents.map((agent) => (
                <Link
                  key={agent.id}
                  href={agent.status === "published" ? getAgentChatHref(agent) : `/build/${agent.id}`}
                  className="group flex min-h-[76px] w-full items-center gap-3 rounded-2xl border border-border/70 bg-white px-4 py-3 shadow-sm transition-all hover:-translate-y-0.5 hover:border-primary/20 hover:shadow-md"
                >
                  <div className="flex h-10 w-10 items-center justify-center overflow-hidden rounded-xl border border-border/60 bg-primary/5">
                    {agent.logo_url ? (
                      <img
                        src={agent.logo_url.startsWith("http") ? agent.logo_url : `${getApiUrl()}${agent.logo_url}`}
                        alt={agent.name}
                        className="h-full w-full object-cover"
                      />
                    ) : (
                      <Bot className="h-4.5 w-4.5 text-primary" />
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <h3 className="truncate text-[14px] font-semibold text-foreground group-hover:text-primary">
                      {agent.name}
                    </h3>
                    <p className="mt-0.5 truncate text-[12px] text-muted-foreground">
                      {agent.updated_at ? formatDateTime(agent.updated_at, locale) : (agent.status || t("nav.build"))}
                    </p>
                  </div>
                  <ArrowUpRight className="h-4 w-4 shrink-0 text-muted-foreground transition-colors group-hover:text-primary" />
                </Link>
              ))}
            </div>
          </div>
        </section>

        <section className="space-y-5">
          <SectionHeader
            title={t("home.templates.title")}
            subtitle={t("home.templates.subtitle")}
            actionLabel={t("home.templates.browseLibrary")}
            href="/templates"
          />

          <div className="overflow-x-auto py-2">
            <div
              className="grid min-w-full grid-flow-col gap-4"
              style={{
                gridAutoColumns: "max(248px, calc((100% - 4rem) / 5))",
              }}
            >
              {templates.map((template) => (
                <HomeTemplateCard
                  key={template.id}
                  template={template}
                  categoryLabel={template.category}
                  runsLabel={t("templates.runs")}
                  onUse={handleUseTemplate}
                />
              ))}
            </div>
          </div>
        </section>
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
    </div>
  );
}
