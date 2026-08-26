"use client";

import "./onboarding.css";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowRight, Check, Plus } from "lucide-react";
import { toast } from "sonner";
import { useAuth } from "@/contexts/auth-context";
import { useI18n } from "@/contexts/i18n-context";
import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";
import { cn } from "@/lib/utils";
import { getBrandingFromEnv } from "@/lib/branding";
import { updateUserPreferences } from "@/lib/user-preferences";
import { hireAgentFromTemplate } from "@/lib/hire-agent";
import { PersonaAvatar } from "@/components/templates/persona-avatar";
import {
  ONBOARDING_DEFAULT_VOICE,
  ONBOARDING_FALLBACK_TEMPLATE_IDS,
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

const branding = getBrandingFromEnv();

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

// persistAndLeave gives up and navigates anyway after this many consecutive
// failures saving to the SAME destination, rather than trapping the user on
// this full-screen page forever. 2 (not 1): a single transient blip
// shouldn't skip the "retry once in place" step entirely; not higher, since
// this is the only page in the app with no other navigation affordance if
// the backend really is down.
const MAX_SAVE_FAILURES_BEFORE_ESCAPE = 2;

// The category ring/dot colors below are the reference UI's fixed literal
// values (onboarding.html's [data-cat] rules), not the app's theme tokens -
// this whole page is a one-off bespoke screen matched pixel-for-pixel against
// that reference, same reasoning as the .ob-* styles further down.
const CATEGORY_STYLE: Record<string, { ring: string; dot: string }> = {
  Marketing: { ring: "rgba(192,57,159,.18)", dot: "#C0399F" },
  Support: { ring: "rgba(37,54,224,.18)", dot: "#2536E0" },
  Operations: { ring: "rgba(200,138,20,.20)", dot: "#C88A14" },
  Sales: { ring: "rgba(34,160,91,.20)", dot: "#22A05B" },
};
const DEFAULT_CATEGORY_STYLE = { ring: "rgba(37,54,224,.10)", dot: "#8A8A94" };

function Orb({ small }: { small?: boolean }) {
  return <div aria-hidden="true" className={cn("ob-orb", small && "sm")} />;
}

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
    <button type="button" aria-pressed={selected} onClick={onClick} className="ob-chip">
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
  const step = STEP_ORDER[stepIndex];
  // Which step within each rail group was last visited, so jumping back to
  // an already-passed group (e.g. "About you" from a later step) returns to
  // where the user actually was in it, not always that group's first step -
  // "About you" spans welcome+business, "My team" spans team+voice.
  const lastVisitedInGroupRef = useRef<Record<number, number>>({});

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
  // Consecutive persistAndLeave save failures, keyed by destination - see the
  // escape hatch there. Per-destination (not one global count): "/task" and
  // "/templates" represent genuinely different user intents, and a failure
  // on one shouldn't spend down the other's own first-attempt retry-in-place
  // chance. header Skip -> "/task"; goals-step and done-step skip both ->
  // "/templates", which is intentional (both mean the same "leave to browse
  // templates" exit, just from different steps).
  const saveFailureCountByDestRef = useRef<Record<string, number>>({});

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    // A locale change mid-flow re-runs this effect (it's a dep below) - reset
    // to loading rather than leaving the previous locale's templates/team
    // step displayed with no indication a refetch is even happening.
    setTemplatesLoading(true);
    (async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/templates/?lang=${locale}`);
        if (cancelled || !response.ok) return;
        const data = await response.json();
        if (!cancelled && Array.isArray(data)) {
          setTemplates(data);
          setTemplatesLoading(false);
        }
      } catch {
        // Deliberately leave templatesLoading true on failure - the team
        // step falls back to a loading state indefinitely rather than a
        // broken, un-populated card list with no way to tell it apart from
        // "there really are no recommendations" (see templatesLoading below).
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
  // While templates are still loading, templateById is empty so this is
  // naturally [] too - no separate loading branch needed, and isTeamValid
  // (below) doesn't need its own !templatesLoading check as a result.
  const validRecommended = useMemo(() => {
    const filtered = recommended.filter((r) => templateById.get(r.templateId)?.persona);
    if (filtered.length > 0) return filtered;
    // Every recommendation for the selected goals failed to load or has no
    // persona (or nothing has loaded yet) - fall back to the same 3 defaults
    // recommendedTemplates() uses when no goals were picked at all, so a
    // template-catalog gap can't leave this step with zero cards to show.
    return ONBOARDING_FALLBACK_TEMPLATE_IDS.map((templateId) => ({ templateId, goalId: null })).filter(
      (r) => templateById.get(r.templateId)?.persona
    );
  }, [recommended, templateById]);

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
    lastVisitedInGroupRef.current[STEP_GROUP[STEP_ORDER[clamped]]] = clamped;
  };
  const next = () => goTo(stepIndex + 1);
  const back = () => goTo(stepIndex - 1);

  const toggleGoal = (id: string) => {
    setGoals((prev) => (prev.includes(id) ? prev.filter((g) => g !== id) : [...prev, id]));
  };

  // Matches the reference UI's finish() exactly: every exit path (header
  // "Skip setup", the goals step's "Not sure yet", the done step's "Take me
  // to the catalogue") persists whatever's been picked so far, unconditionally
  // - there is no partial-vs-full distinction there, and there shouldn't be
  // one here either. Bailing out via the goals step after already selecting
  // a couple of goals must not silently discard them.
  const persistAndLeave = async (destination: string) => {
    // Reuses the same launching/launchingRef guard as handleLaunch below -
    // a double-click on any exit button before the PATCH resolves would
    // otherwise fire two concurrent saves.
    if (launchingRef.current) return;
    launchingRef.current = true;
    setLaunching(true);
    // Awaited, not fire-and-forget: AuthGuard's onboarding-redirect check runs
    // a GET as soon as `destination` mounts, and that GET winning the race
    // against this PATCH would read the old onboarded:false and bounce the
    // user straight back into onboarding right after they left it.
    const saved = await updateUserPreferences({
      onboarded: true,
      ...(work ? { department: work } : {}),
      ...(work === "other" && industry.trim() ? { industry: industry.trim() } : {}),
      ...(goals.length ? { goals } : {}),
      voice,
    });
    if (!saved.ok) {
      const failureCount = (saveFailureCountByDestRef.current[destination] ?? 0) + 1;
      saveFailureCountByDestRef.current[destination] = failureCount;
      if (isMountedRef.current) {
        toast.error(t("onboarding.done.saveFailed"));
        launchingRef.current = false;
        setLaunching(false);
      }
      // Don't navigate on the FIRST failure to this destination: onboarded
      // would stay false server-side, and AuthGuard would just bounce the
      // user right back here - retrying in place is the better first
      // response. But this is the only full-screen page in the app with no
      // other nav affordance, so a save that keeps failing (backend down, a
      // persistent 422) must not trap the user here forever with zero way
      // out - let them through after a couple of tries even though the save
      // didn't land; they'll just be asked again next session instead of
      // being stuck this one.
      if (failureCount < MAX_SAVE_FAILURES_BEFORE_ESCAPE) return;
    } else {
      saveFailureCountByDestRef.current[destination] = 0;
    }
    if (isMountedRef.current) router.push(destination);
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
    <div className="ob-shell">
      <div aria-hidden="true" className="ob-aurora">
        <b className="ob-a1" />
        <b className="ob-a2" />
        <b className="ob-a3" />
      </div>

      <header className="ob-top">
        <button type="button" disabled={launching} onClick={() => persistAndLeave("/task")} className="ob-exit">
          {t("onboarding.skip")}
        </button>
        {/* eslint-disable-next-line @next/next/no-img-element -- fixed 26px brand mark, not a candidate for next/image */}
        <img className="ob-logo" src={branding.logoPath} alt={branding.appName} />
        <div className="ob-acct">
          <span>{firstName.slice(0, 1).toUpperCase()}</span>
          <b>{firstName}</b>
        </div>
      </header>

      <nav aria-label={t("onboarding.rail.label")} className="ob-rail">
        {GROUP_KEYS.map((key, groupIndex) => {
          const isNow = STEP_GROUP[step] === groupIndex;
          const isDone = STEP_GROUP[step] > groupIndex;
          // Navigation is strictly linear (next() only ever advances by one
          // step), so an earlier group being reachable at all already means
          // every step in it was visited - no separate "how far has the user
          // gotten" tracking needed here.
          const canGo = groupIndex < STEP_GROUP[step];
          return (
            <button
              key={key}
              type="button"
              disabled={!canGo || launching}
              onClick={() => {
                const targetStep =
                  lastVisitedInGroupRef.current[groupIndex] ??
                  STEP_ORDER.findIndex((s) => STEP_GROUP[s] === groupIndex);
                if (targetStep >= 0) goTo(targetStep);
              }}
              className={cn("ob-rl", isNow && "is-now", isDone && "is-done", canGo && "can-go")}
            >
              <i>
                {isDone ? <Check className="h-2.5 w-2.5" /> : isNow ? <b className="ob-rl-dot" /> : null}
              </i>
              <span>{t(key)}</span>
            </button>
          );
        })}
      </nav>

      <main className="ob-body">
        <div className="ob-step">
          {step === "welcome" && (
            <>
              <Orb />
              <h1>
                {t("onboarding.welcome.titlePrefix")}
                <br />
                <em>{firstName}</em>.
              </h1>
              <p className="ob-sub">{t("onboarding.welcome.subtitle")}</p>
              <div className="ob-cta">
                <button type="button" disabled={launching} onClick={next} className="ob-btn-next">
                  {t("onboarding.welcome.cta")}
                  <ArrowRight className="h-4 w-4" />
                </button>
              </div>
            </>
          )}

          {step === "business" && (
            <>
              <Orb small />
              <h1>
                {t("onboarding.business.titleLine1")}
                <br />
                {t("onboarding.business.titleLine2")}
              </h1>
              <p className="ob-sub">{t("onboarding.business.subtitle")}</p>
              <div className="ob-chips" style={{ marginTop: 34 }}>
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
                <div className="ob-form">
                  <label htmlFor="onboarding-industry" className="ob-fl">
                    {t("onboarding.business.industryLabel")}
                  </label>
                  <input
                    id="onboarding-industry"
                    autoFocus
                    value={industry}
                    onChange={(event) => setIndustry(event.target.value)}
                    placeholder={t("onboarding.business.industryPlaceholder")}
                    // Matches PREFERENCES_TEXT_FIELD_MAX_LENGTH in
                    // src/xagent/web/api/auth.py - a defensive client-side
                    // cap, not a replacement for the backend's own validation.
                    maxLength={200}
                    className="ob-input"
                  />
                </div>
              )}
              <div className="ob-cta">
                <button type="button" disabled={launching} onClick={back} className="ob-btn-back">
                  {t("common.back")}
                </button>
                <button type="button" disabled={!isBusinessValid || launching} onClick={next} className="ob-btn-next">
                  {t("onboarding.continue")}
                </button>
              </div>
            </>
          )}

          {step === "goals" && (
            <>
              <Orb small />
              <h1>
                {t("onboarding.goals.titleLine1")}
                <br />
                {t("onboarding.goals.titleLine2")}
              </h1>
              <p className="ob-sub">{t("onboarding.goals.subtitle")}</p>
              <div className="ob-chips" style={{ marginTop: 34 }}>
                {orderedGoals.map((goal) => {
                  const selected = goals.includes(goal.id);
                  return (
                    <Chip
                      key={goal.id}
                      selected={selected}
                      onClick={() => toggleGoal(goal.id)}
                      icon={
                        selected ? (
                          <Check className="ob-ic h-3.5 w-3.5" />
                        ) : (
                          <Plus className="ob-ic h-3.5 w-3.5" />
                        )
                      }
                    >
                      {t(goal.labelKey)}
                    </Chip>
                  );
                })}
              </div>
              <div className="ob-cta">
                <button type="button" disabled={launching} onClick={back} className="ob-btn-back">
                  {t("common.back")}
                </button>
                <button type="button" disabled={!isGoalsValid || launching} onClick={next} className="ob-btn-next">
                  {t("onboarding.continue")}
                </button>
              </div>
              <button type="button" disabled={launching} onClick={() => persistAndLeave("/templates")} className="ob-skip">
                {t("onboarding.goals.skip")}
              </button>
            </>
          )}

          {step === "team" && (
            <>
              <Orb small />
              <h1>{t("onboarding.team.title")}</h1>
              <p className="ob-sub">
                {goals.length === 0
                  ? t("onboarding.team.subtitleNoGoals")
                  : // Gated on !templatesLoading like the grid below it: while
                    // still loading, validRecommended is always [] (nothing has
                    // resolved yet), which would otherwise claim every goal has
                    // an "other match waiting" before we actually know that.
                    `${t("onboarding.team.subtitleBase")}${
                      !templatesLoading && goals.length - validRecommended.length > 0
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
                <div style={{ marginTop: 40, display: "flex", justifyContent: "center" }}>
                  <div className="h-8 w-8 animate-spin rounded-full border-2 border-[#E7E7EC] border-t-[#2536E0]" />
                </div>
              ) : (
                <div
                  className={cn(
                    "ob-team",
                    validRecommended.length === 1 ? "one" : validRecommended.length === 2 ? "two" : ""
                  )}
                >
                  {validRecommended.map((rec) => {
                    const template = templateById.get(rec.templateId);
                    if (!template || !template.persona) return null;
                    const goal = ONBOARDING_GOALS.find((g) => g.id === rec.goalId);
                    const selected = agentTemplateId === template.id;
                    const { ring, dot } = CATEGORY_STYLE[template.category] ?? DEFAULT_CATEGORY_STYLE;
                    return (
                      <button
                        key={template.id}
                        type="button"
                        aria-pressed={selected}
                        onClick={() => setAgentTemplateId(template.id)}
                        className="ob-tm"
                      >
                        {selected && (
                          <span className="ob-tm-tick">
                            <Check className="h-3 w-3" />
                          </span>
                        )}
                        <PersonaAvatar
                          persona={template.persona}
                          sizeClassName="h-[84px] w-[84px]"
                          textClassName="text-[33px]"
                          className="ob-tm-av"
                          style={{ boxShadow: `0 0 0 1px #F0F0F4, 0 0 0 5px #fff, 0 0 0 6px ${ring}` }}
                        />
                        <div className="ob-tm-name">{template.persona.name}</div>
                        <div className="ob-tm-role">{template.persona.role}</div>
                        <span className="ob-tm-cat">
                          <i style={{ background: dot }} />
                          {template.category}
                        </span>
                        <p className="ob-tm-desc">{template.description}</p>
                        {goal && (
                          <span className="ob-tm-why">
                            <i />
                            {t(goal.labelKey)}
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              )}
              <div className="ob-cta">
                <button type="button" disabled={launching} onClick={back} className="ob-btn-back">
                  {t("common.back")}
                </button>
                <button type="button" disabled={!isTeamValid || launching} onClick={next} className="ob-btn-next">
                  {t("onboarding.continue")}
                </button>
              </div>
            </>
          )}

          {step === "voice" && (
            <>
              <Orb small />
              <h1>
                {t("onboarding.voice.titleLine1")}
                <br />
                {t("onboarding.voice.titleLine2")}
              </h1>
              <p className="ob-sub">{t("onboarding.voice.subtitle")}</p>
              <div className="ob-chips" style={{ marginTop: 34 }}>
                {ONBOARDING_VOICES.map((option) => (
                  <Chip key={option.id} selected={voice === option.id} onClick={() => setVoice(option.id)}>
                    {t(option.nameKey)}
                  </Chip>
                ))}
              </div>
              {(() => {
                const active = ONBOARDING_VOICES.find((v) => v.id === voice) ?? ONBOARDING_VOICES[0];
                return (
                  <div className="ob-say">
                    <b>{t("onboarding.voice.soundsLike", { name: t(active.nameKey) })}</b>
                    <i>{t(active.sayKey)}</i>
                  </div>
                );
              })()}
              <div className="ob-cta">
                <button type="button" disabled={launching} onClick={back} className="ob-btn-back">
                  {t("common.back")}
                </button>
                <button type="button" disabled={launching} onClick={next} className="ob-btn-next">
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
              const { ring } = CATEGORY_STYLE[selected.category] ?? DEFAULT_CATEGORY_STYLE;

              return (
                <>
                  <Orb />
                  <h1>{t("onboarding.done.title")}</h1>
                  <p className="ob-sub">{t("onboarding.done.subtitle", { name: persona.name })}</p>

                  <div className="ob-sum">
                    <div className="ob-sum-hd">
                      <PersonaAvatar
                        persona={persona}
                        sizeClassName="h-[62px] w-[62px]"
                        textClassName="text-2xl"
                        className="ob-tm-av"
                        style={{ boxShadow: `0 0 0 1px #F0F0F4, 0 0 0 4px #fff, 0 0 0 5px ${ring}` }}
                      />
                      <div>
                        <div className="ob-sum-nm">{persona.name}</div>
                        <div className="ob-sum-rl">{persona.role}</div>
                      </div>
                    </div>
                    <div className="ob-sum-list">
                      <div className="ob-sum-li">
                        <Check className="mt-0.5 h-[18px] w-[18px]" style={{ color: "#22A05B" }} />
                        <span>
                          {t("onboarding.done.workingInPrefix")} <b>{workLabel}</b>
                        </span>
                      </div>
                      <div className="ob-sum-li">
                        <Check className="mt-0.5 h-[18px] w-[18px]" style={{ color: "#22A05B" }} />
                        <span>
                          {t("onboarding.done.briefedPrefix")} <b>{jobCount}</b>{" "}
                          {t(jobCount === 1 ? "onboarding.done.briefedSuffixOne" : "onboarding.done.briefedSuffixOther")}
                        </span>
                      </div>
                      <div className="ob-sum-li">
                        <Check className="mt-0.5 h-[18px] w-[18px]" style={{ color: "#22A05B" }} />
                        <span>
                          {t("onboarding.done.writingInPrefix")} <b>{t(voiceOption.nameKey).toLowerCase()}</b>{" "}
                          {t("onboarding.done.writingInSuffix")}
                        </span>
                      </div>
                      <div className="ob-sum-li">
                        <Check
                          className="mt-0.5 h-[18px] w-[18px]"
                          style={{ color: appNames.length ? "#8A8A94" : "#22A05B" }}
                        />
                        {appNames.length ? (
                          <span>
                            {t("onboarding.done.willConnectPrefix")}{" "}
                            <b>{joinWithAnd(appNames, t("onboarding.done.willConnectAnd"))}</b>{" "}
                            {t("onboarding.done.willConnectSuffix")}
                          </span>
                        ) : (
                          <span>{t("onboarding.done.noAccounts")}</span>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="ob-cta">
                    <button type="button" disabled={launching} onClick={back} className="ob-btn-back">
                      {t("common.back")}
                    </button>
                    <button type="button" disabled={launching} onClick={handleLaunch} className="ob-btn-next">
                      {launching ? t("templates.marketplace.hiring") : t("onboarding.done.start", { name: persona.name })}
                      {!launching && <ArrowRight className="h-4 w-4" />}
                    </button>
                  </div>
                  <button type="button" disabled={launching} onClick={() => persistAndLeave("/templates")} className="ob-skip">
                    {t("onboarding.done.skip")}
                  </button>
                </>
              );
            })()}
        </div>
      </main>
    </div>
  );
}
