"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowRight, Check, Plus } from "lucide-react";
import { toast } from "sonner";
import { useAuth } from "@/contexts/auth-context";
import { useI18n } from "@/contexts/i18n-context";
import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";
import { cn } from "@/lib/utils";
import { updateUserPreferences } from "@/lib/user-preferences";
import { hireAgentFromTemplate } from "@/lib/hire-agent";
import { PersonaAvatar } from "@/components/templates/persona-avatar";
import {
  ONBOARDING_DEFAULT_VOICE,
  ONBOARDING_GOALS,
  ONBOARDING_VOICES,
  ONBOARDING_WORK,
  joinWithAnd,
  recommendedTemplates,
  reorderGoalsByWork,
  type OnboardingVoiceId,
  type OnboardingWorkId,
} from "@/lib/onboarding-data";
import type { Template } from "@/types/template";
import type { TranslationKey } from "@/i18n/translations";

// Sidebar groups - "My team" deliberately spans both the team-pick and
// voice-pick steps (matching the reference UI exactly), not one group per
// step; only "Launch" is its own single-step group.
const GROUP_KEYS: TranslationKey[] = [
  "onboarding.rail.aboutYou",
  "onboarding.rail.goals",
  "onboarding.rail.myTeam",
  "onboarding.rail.launch",
];

type StepId = "welcome" | "business" | "goals" | "team" | "voice" | "done";
const STEP_GROUP: Record<StepId, number> = {
  welcome: 0,
  business: 0,
  goals: 1,
  team: 2,
  voice: 2,
  done: 3,
};
const STEP_ORDER: StepId[] = ["welcome", "business", "goals", "team", "voice", "done"];

function Chip({
  selected,
  onClick,
  children,
  icon,
}: {
  selected: boolean;
  onClick: () => void;
  children: React.ReactNode;
  icon?: React.ReactNode;
}) {
  return (
    <button
      type="button"
      aria-pressed={selected}
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-4 py-2 text-[13.5px] font-medium transition-colors",
        selected
          ? "border-primary bg-primary/5 text-primary"
          : "border-border bg-card text-foreground/80 hover:bg-muted/60"
      )}
    >
      {icon}
      {children}
    </button>
  );
}

export default function OnboardingPage() {
  const router = useRouter();
  const { user } = useAuth();
  const { t, locale } = useI18n();

  const [stepIndex, setStepIndex] = useState(0);
  const [furthest, setFurthest] = useState(0);
  const step = STEP_ORDER[stepIndex];

  const [work, setWork] = useState<OnboardingWorkId | "">("");
  const [industry, setIndustry] = useState("");
  const [goals, setGoals] = useState<string[]>([]);
  const [agentTemplateId, setAgentTemplateId] = useState("");
  const [voice, setVoice] = useState<OnboardingVoiceId>(ONBOARDING_DEFAULT_VOICE);

  const [templates, setTemplates] = useState<Template[]>([]);
  const [templatesLoading, setTemplatesLoading] = useState(true);
  const [launching, setLaunching] = useState(false);
  const launchingRef = useRef(false);
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/templates/?lang=${locale}`);
        if (cancelled || !response.ok) return;
        const data = await response.json();
        if (!cancelled && Array.isArray(data)) setTemplates(data);
      } catch {
        // The team step falls back to a loading state indefinitely rather
        // than a broken card list - see templatesLoading below.
      } finally {
        if (!cancelled) setTemplatesLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [locale]);

  const templateById = useMemo(() => {
    const map = new Map<string, Template>();
    for (const template of templates) map.set(template.id, template);
    return map;
  }, [templates]);

  const recommended = useMemo(() => recommendedTemplates(goals), [goals]);

  // The static goal->template catalog can point at a template id that never
  // loaded (fetch failure) or has no persona - filtering here keeps the team
  // step from ever landing on a recommendation with nothing to render, which
  // would otherwise leave "Continue" enabled with an empty grid and a blank
  // (persona-less) Done step with no way to finish other than Back/Skip.
  const validRecommended = useMemo(() => {
    if (templatesLoading) return recommended;
    return recommended.filter((r) => templateById.get(r.templateId)?.persona);
  }, [recommended, templatesLoading, templateById]);

  // Keep the selected agent valid as the recommendation list changes -
  // default to the first recommendation whenever the current pick falls
  // out of the list (including the very first render).
  useEffect(() => {
    if (!validRecommended.some((r) => r.templateId === agentTemplateId)) {
      setAgentTemplateId(validRecommended[0]?.templateId ?? "");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [validRecommended]);

  const orderedGoals = useMemo(() => reorderGoalsByWork(work), [work]);

  const firstName = user?.username?.trim() || t("onboarding.welcome.fallbackName");

  const isBusinessValid = work !== "" && (work !== "other" || industry.trim() !== "");
  const isGoalsValid = goals.length > 0;
  const isTeamValid = agentTemplateId !== "";

  const goTo = (index: number) => {
    const clamped = Math.max(0, Math.min(STEP_ORDER.length - 1, index));
    setStepIndex(clamped);
    setFurthest((prev) => Math.max(prev, clamped));
  };
  const next = () => goTo(stepIndex + 1);
  const back = () => goTo(stepIndex - 1);

  const toggleGoal = (id: string) => {
    setGoals((prev) => (prev.includes(id) ? prev.filter((g) => g !== id) : [...prev, id]));
  };

  const persistAndLeave = async (destination: string, includeAll: boolean) => {
    // Awaited, not fire-and-forget: AuthGuard's onboarding-redirect check runs
    // a GET as soon as `destination` mounts, and that GET winning the race
    // against this PATCH would read the old onboarded:false and bounce the
    // user straight back into onboarding right after they left it.
    await updateUserPreferences({
      onboarded: true,
      ...(work ? { department: work } : {}),
      ...(work === "other" && industry.trim() ? { industry: industry.trim() } : {}),
      ...(includeAll && goals.length ? { goals } : {}),
      ...(includeAll ? { voice } : {}),
    });
    router.push(destination);
  };

  const handleLaunch = async () => {
    const selected = templateById.get(agentTemplateId);
    if (!selected || !selected.persona || launchingRef.current) return;

    launchingRef.current = true;
    setLaunching(true);
    try {
      const saved = await updateUserPreferences({
        onboarded: true,
        ...(work ? { department: work } : {}),
        ...(work === "other" && industry.trim() ? { industry: industry.trim() } : {}),
        goals,
        voice,
      });
      // Don't proceed to hire on a failed save: onboarded would never be
      // persisted, yet the user would land on the agent's task page believing
      // setup is done - only to be bounced back into onboarding next session.
      if (!saved.ok) {
        if (isMountedRef.current) {
          toast.error(t("onboarding.done.saveFailed"));
          launchingRef.current = false;
          setLaunching(false);
        }
        return;
      }

      if (selected.hired && selected.hired_agent_id) {
        if (isMountedRef.current) router.push(`/agent/${selected.hired_agent_id}`);
        return;
      }

      const result = await hireAgentFromTemplate({
        templateId: selected.id,
        persona: selected.persona,
        strings: {
          beforeWeStart: t("templates.marketplace.beforeWeStart"),
          closingNote: t("templates.marketplace.hireClosingNote"),
          connectAppsLabel: t("chatPage.clarification.connectApps.title"),
        },
        connections: selected.connections,
      });
      if (!isMountedRef.current) return;
      router.push(`/task/${result.taskId}`);
    } catch {
      if (!isMountedRef.current) return;
      toast.error(
        t("templates.marketplace.hireFailed", { name: selected.persona?.name || selected.name })
      );
      launchingRef.current = false;
      setLaunching(false);
    }
  };

  return (
    <div className="relative flex min-h-screen flex-col overflow-hidden bg-background">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(circle_at_20%_20%,rgba(37,54,224,0.12),transparent_45%),radial-gradient(circle_at_80%_30%,rgba(192,57,159,0.10),transparent_45%),radial-gradient(circle_at_50%_90%,rgba(34,160,91,0.08),transparent_50%)]"
      />

      <header className="flex items-center justify-between px-6 py-5">
        <button
          type="button"
          onClick={() => persistAndLeave("/task", false)}
          className="text-[13px] font-medium text-muted-foreground hover:text-foreground"
        >
          {t("onboarding.skip")}
        </button>
        <span className="text-[15px] font-semibold text-foreground">Xagent</span>
        <div className="flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center rounded-full bg-primary text-[12px] font-semibold text-primary-foreground">
            {firstName.slice(0, 1).toUpperCase()}
          </span>
          <span className="text-[13px] font-medium text-foreground">{firstName}</span>
        </div>
      </header>

      <div className="flex flex-1">
        <nav aria-label={t("onboarding.rail.label")} className="w-48 flex-shrink-0 px-6 py-2">
          <ul className="flex flex-col gap-4">
            {GROUP_KEYS.map((key, groupIndex) => {
              const isNow = STEP_GROUP[step] === groupIndex;
              const isDone = STEP_GROUP[step] > groupIndex;
              const canGo = groupIndex < STEP_GROUP[step] && groupIndex <= STEP_GROUP[STEP_ORDER[furthest]];
              return (
                <li key={key}>
                  <button
                    type="button"
                    disabled={!canGo}
                    onClick={() => {
                      const targetStep = STEP_ORDER.findIndex((s) => STEP_GROUP[s] === groupIndex);
                      if (targetStep >= 0) goTo(targetStep);
                    }}
                    className={cn(
                      "flex items-center gap-2 text-[13.5px]",
                      isNow ? "font-semibold text-foreground" : "text-muted-foreground",
                      canGo && "cursor-pointer hover:text-foreground",
                      !canGo && "cursor-default"
                    )}
                  >
                    <span
                      className={cn(
                        "flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full border",
                        isDone
                          ? "border-primary bg-primary text-primary-foreground"
                          : isNow
                            ? "border-primary text-primary"
                            : "border-border text-transparent"
                      )}
                    >
                      {isDone ? <Check className="h-3 w-3" /> : null}
                    </span>
                    {t(key)}
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>

        <main className="flex flex-1 items-center justify-center px-6 py-10">
          <div className="w-full max-w-[560px] text-center">
            {step === "welcome" && (
              <>
                <h1 className="text-[38px] font-extrabold leading-[1.15] text-foreground">
                  {t("onboarding.welcome.titlePrefix")}
                  <br />
                  <em className="text-primary not-italic">{firstName}</em>.
                </h1>
                <p className="mt-5 text-[15px] leading-relaxed text-muted-foreground">
                  {t("onboarding.welcome.subtitle")}
                </p>
                <button
                  type="button"
                  onClick={next}
                  className="mt-8 inline-flex items-center gap-2 rounded-full bg-primary px-6 py-3 text-[14.5px] font-semibold text-primary-foreground hover:bg-primary/90"
                >
                  {t("onboarding.welcome.cta")}
                  <ArrowRight className="h-4 w-4" />
                </button>
              </>
            )}

            {step === "business" && (
              <>
                <h1 className="text-[32px] font-extrabold leading-[1.2] text-foreground">
                  {t("onboarding.business.titleLine1")}
                  <br />
                  {t("onboarding.business.titleLine2")}
                </h1>
                <p className="mt-4 text-[14.5px] text-muted-foreground">
                  {t("onboarding.business.subtitle")}
                </p>
                <div className="mt-8 flex flex-wrap justify-center gap-2.5">
                  {ONBOARDING_WORK.map((option) => (
                    <Chip
                      key={option.id}
                      selected={work === option.id}
                      onClick={() => {
                        setWork(option.id);
                        if (option.id !== "other") setIndustry("");
                      }}
                    >
                      {t(option.labelKey)}
                    </Chip>
                  ))}
                </div>
                {work === "other" && (
                  <div className="mx-auto mt-6 max-w-xs text-left">
                    <label htmlFor="onboarding-industry" className="text-[12.5px] font-medium text-muted-foreground">
                      {t("onboarding.business.industryLabel")}
                    </label>
                    <input
                      id="onboarding-industry"
                      autoFocus
                      value={industry}
                      onChange={(event) => setIndustry(event.target.value)}
                      placeholder={t("onboarding.business.industryPlaceholder")}
                      className="mt-1.5 w-full rounded-[10px] border border-border bg-card px-3.5 py-2.5 text-[14px] outline-none focus:border-primary"
                    />
                  </div>
                )}
                <div className="mt-8 flex items-center justify-center gap-4">
                  <button type="button" onClick={back} className="text-[13.5px] font-medium text-muted-foreground hover:text-foreground">
                    {t("common.back")}
                  </button>
                  <button
                    type="button"
                    disabled={!isBusinessValid}
                    onClick={next}
                    className="rounded-full bg-primary px-6 py-2.5 text-[14px] font-semibold text-primary-foreground disabled:opacity-40"
                  >
                    {t("onboarding.continue")}
                  </button>
                </div>
              </>
            )}

            {step === "goals" && (
              <>
                <h1 className="text-[32px] font-extrabold leading-[1.2] text-foreground">
                  {t("onboarding.goals.titleLine1")}
                  <br />
                  {t("onboarding.goals.titleLine2")}
                </h1>
                <p className="mt-4 text-[14.5px] text-muted-foreground">
                  {t("onboarding.goals.subtitle")}
                </p>
                <div className="mt-8 flex flex-wrap justify-center gap-2.5">
                  {orderedGoals.map((goal) => {
                    const selected = goals.includes(goal.id);
                    return (
                      <Chip
                        key={goal.id}
                        selected={selected}
                        onClick={() => toggleGoal(goal.id)}
                        icon={
                          selected ? (
                            <Check className="h-3.5 w-3.5" />
                          ) : (
                            <Plus className="h-3.5 w-3.5 opacity-60" />
                          )
                        }
                      >
                        {t(goal.labelKey)}
                      </Chip>
                    );
                  })}
                </div>
                <div className="mt-8 flex items-center justify-center gap-4">
                  <button type="button" onClick={back} className="text-[13.5px] font-medium text-muted-foreground hover:text-foreground">
                    {t("common.back")}
                  </button>
                  <button
                    type="button"
                    disabled={!isGoalsValid}
                    onClick={next}
                    className="rounded-full bg-primary px-6 py-2.5 text-[14px] font-semibold text-primary-foreground disabled:opacity-40"
                  >
                    {t("onboarding.continue")}
                  </button>
                </div>
                <button
                  type="button"
                  onClick={() => persistAndLeave("/templates", false)}
                  className="mt-6 text-[12.5px] font-medium text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
                >
                  {t("onboarding.goals.skip")}
                </button>
              </>
            )}

            {step === "team" && (
              <>
                <h1 className="text-[32px] font-extrabold leading-[1.2] text-foreground">
                  {t("onboarding.team.title")}
                </h1>
                <p className="mt-4 text-[14.5px] text-muted-foreground">
                  {goals.length === 0
                    ? t("onboarding.team.subtitleNoGoals")
                    : `${t("onboarding.team.subtitleBase")}${
                        goals.length - validRecommended.length > 0
                          ? ` ${t(
                              goals.length - validRecommended.length === 1
                                ? "onboarding.team.subtitleExtraOne"
                                : "onboarding.team.subtitleExtraMany",
                              { count: goals.length - validRecommended.length }
                            )}`
                          : "."
                      }`}
                </p>
                {templatesLoading ? (
                  <div className="mt-10 flex justify-center">
                    <div className="h-8 w-8 animate-spin rounded-full border-2 border-muted-foreground/30 border-t-primary" />
                  </div>
                ) : (
                  <div
                    className={cn(
                      "mt-8 grid gap-4",
                      validRecommended.length === 1 ? "grid-cols-1" : "grid-cols-1 sm:grid-cols-3"
                    )}
                  >
                    {validRecommended.map((rec) => {
                      const template = templateById.get(rec.templateId);
                      if (!template || !template.persona) return null;
                      const goal = ONBOARDING_GOALS.find((g) => g.id === rec.goalId);
                      const selected = agentTemplateId === template.id;
                      return (
                        <button
                          key={template.id}
                          type="button"
                          aria-pressed={selected}
                          onClick={() => setAgentTemplateId(template.id)}
                          className={cn(
                            "relative flex flex-col items-center rounded-[16px] border p-5 text-center transition-colors",
                            selected ? "border-primary bg-primary/5" : "border-border bg-card hover:bg-muted/40"
                          )}
                        >
                          {selected && (
                            <span className="absolute right-3 top-3 flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground">
                              <Check className="h-3 w-3" />
                            </span>
                          )}
                          <PersonaAvatar persona={template.persona} sizeClassName="h-16 w-16" textClassName="text-lg" />
                          <div className="mt-3 text-[15px] font-bold uppercase tracking-wide text-foreground">
                            {template.persona.name}
                          </div>
                          <div className="mt-1 text-[12px] text-muted-foreground">{template.persona.role}</div>
                          <span className="mt-2 inline-flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1 text-[11px] text-muted-foreground">
                            <span className="h-1.5 w-1.5 rounded-full bg-current" />
                            {template.category}
                          </span>
                          <p className="mt-3 text-[13px] leading-relaxed text-foreground/80">
                            {template.description}
                          </p>
                          {goal && (
                            <span className="mt-3 text-[11px] text-muted-foreground/80">{t(goal.labelKey)}</span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                )}
                <div className="mt-8 flex items-center justify-center gap-4">
                  <button type="button" onClick={back} className="text-[13.5px] font-medium text-muted-foreground hover:text-foreground">
                    {t("common.back")}
                  </button>
                  <button
                    type="button"
                    disabled={!isTeamValid}
                    onClick={next}
                    className="rounded-full bg-primary px-6 py-2.5 text-[14px] font-semibold text-primary-foreground disabled:opacity-40"
                  >
                    {t("onboarding.continue")}
                  </button>
                </div>
              </>
            )}

            {step === "voice" && (
              <>
                <h1 className="text-[32px] font-extrabold leading-[1.2] text-foreground">
                  {t("onboarding.voice.titleLine1")}
                  <br />
                  {t("onboarding.voice.titleLine2")}
                </h1>
                <p className="mt-4 text-[14.5px] text-muted-foreground">
                  {t("onboarding.voice.subtitle")}
                </p>
                <div className="mt-8 flex flex-wrap justify-center gap-2.5">
                  {ONBOARDING_VOICES.map((option) => (
                    <Chip key={option.id} selected={voice === option.id} onClick={() => setVoice(option.id)}>
                      {t(option.nameKey)}
                    </Chip>
                  ))}
                </div>
                {(() => {
                  const active = ONBOARDING_VOICES.find((v) => v.id === voice) ?? ONBOARDING_VOICES[0];
                  return (
                    <div className="mx-auto mt-6 max-w-md rounded-[14px] bg-muted/50 p-5 text-left">
                      <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                        {t("onboarding.voice.soundsLike", { name: t(active.nameKey) })}
                      </div>
                      <p className="mt-2 text-[14px] italic leading-relaxed text-foreground/90">{t(active.sayKey)}</p>
                    </div>
                  );
                })()}
                <div className="mt-8 flex items-center justify-center gap-4">
                  <button type="button" onClick={back} className="text-[13.5px] font-medium text-muted-foreground hover:text-foreground">
                    {t("common.back")}
                  </button>
                  <button
                    type="button"
                    onClick={next}
                    className="rounded-full bg-primary px-6 py-2.5 text-[14px] font-semibold text-primary-foreground"
                  >
                    {t("onboarding.continue")}
                  </button>
                </div>
              </>
            )}

            {step === "done" &&
              (() => {
                const selected = templateById.get(agentTemplateId);
                if (!selected || !selected.persona) return null;
                const persona = selected.persona;
                const workOption = ONBOARDING_WORK.find((w) => w.id === work);
                const workLabel = work === "other" ? industry.trim() : workOption ? t(workOption.labelKey).toLowerCase() : "";
                const jobCount = goals.length || 3;
                const voiceOption = ONBOARDING_VOICES.find((v) => v.id === voice) ?? ONBOARDING_VOICES[0];
                const appNames = (selected.connections || []).map((c) => c.name).filter(Boolean);

                return (
                  <>
                    <h1 className="text-[32px] font-extrabold leading-[1.2] text-foreground">
                      {t("onboarding.done.title")}
                    </h1>
                    <p className="mt-4 text-[14.5px] text-muted-foreground">
                      {t("onboarding.done.subtitle", { name: persona.name })}
                    </p>

                    <div className="mx-auto mt-8 max-w-md rounded-[16px] border border-border bg-card p-5 text-left">
                      <div className="flex items-center gap-3 border-b border-border pb-4">
                        <PersonaAvatar persona={persona} sizeClassName="h-12 w-12" textClassName="text-base" />
                        <div>
                          <div className="text-[15px] font-bold text-foreground">{persona.name}</div>
                          <div className="text-[12px] text-muted-foreground">{persona.role}</div>
                        </div>
                      </div>
                      <div className="mt-4 flex flex-col gap-2.5 text-[13.5px]">
                        <div className="flex items-start gap-2">
                          <Check className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-green-600" />
                          <span>
                            {t("onboarding.done.workingInPrefix")} <b>{workLabel}</b>
                          </span>
                        </div>
                        <div className="flex items-start gap-2">
                          <Check className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-green-600" />
                          <span>
                            {t("onboarding.done.briefedPrefix")} <b>{jobCount}</b>{" "}
                            {t(jobCount === 1 ? "onboarding.done.briefedSuffixOne" : "onboarding.done.briefedSuffixOther")}
                          </span>
                        </div>
                        <div className="flex items-start gap-2">
                          <Check className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-green-600" />
                          <span>
                            {t("onboarding.done.writingInPrefix")} <b>{t(voiceOption.nameKey).toLowerCase()}</b>{" "}
                            {t("onboarding.done.writingInSuffix")}
                          </span>
                        </div>
                        <div className="flex items-start gap-2">
                          <Check
                            className={cn(
                              "mt-0.5 h-3.5 w-3.5 flex-shrink-0",
                              appNames.length ? "text-muted-foreground/50" : "text-green-600"
                            )}
                          />
                          {appNames.length ? (
                            <span>
                              {t("onboarding.done.willConnectPrefix")} <b>{joinWithAnd(appNames)}</b>{" "}
                              {t("onboarding.done.willConnectSuffix")}
                            </span>
                          ) : (
                            <span>{t("onboarding.done.noAccounts")}</span>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="mt-8 flex items-center justify-center gap-4">
                      <button type="button" onClick={back} className="text-[13.5px] font-medium text-muted-foreground hover:text-foreground">
                        {t("common.back")}
                      </button>
                      <button
                        type="button"
                        disabled={launching}
                        onClick={handleLaunch}
                        className="inline-flex items-center gap-2 rounded-full bg-primary px-6 py-2.5 text-[14px] font-semibold text-primary-foreground disabled:opacity-60"
                      >
                        {launching ? t("templates.marketplace.hiring") : t("onboarding.done.start", { name: persona.name })}
                        {!launching && <ArrowRight className="h-4 w-4" />}
                      </button>
                    </div>
                    <button
                      type="button"
                      onClick={() => persistAndLeave("/templates", true)}
                      className="mt-6 text-[12.5px] font-medium text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
                    >
                      {t("onboarding.done.skip")}
                    </button>
                  </>
                );
              })()}
          </div>
        </main>
      </div>
    </div>
  );
}
