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
import { getBrandingFromEnv } from "@/lib/branding";
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

// The category ring/dot colors below are the reference UI's fixed literal
// values (onboarding.html's [data-cat] rules), not the app's theme tokens -
// this whole page is a one-off bespoke screen matched pixel-for-pixel against
// that reference, same reasoning as the .ob-* styles further down.
const CATEGORY_RING: Record<string, string> = {
  Marketing: "rgba(192,57,159,.18)",
  Support: "rgba(37,54,224,.18)",
  Operations: "rgba(200,138,20,.20)",
  Sales: "rgba(34,160,91,.20)",
};
const CATEGORY_DOT: Record<string, string> = {
  Marketing: "#C0399F",
  Support: "#2536E0",
  Operations: "#C88A14",
  Sales: "#22A05B",
};

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
    <div className="ob-shell">
      <div aria-hidden="true" className="ob-aurora">
        <b className="ob-a1" />
        <b className="ob-a2" />
        <b className="ob-a3" />
      </div>

      <header className="ob-top">
        <button type="button" onClick={() => persistAndLeave("/task", false)} className="ob-exit">
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
          const canGo = groupIndex < STEP_GROUP[step] && groupIndex <= STEP_GROUP[STEP_ORDER[furthest]];
          return (
            <button
              key={key}
              type="button"
              disabled={!canGo}
              onClick={() => {
                const targetStep = STEP_ORDER.findIndex((s) => STEP_GROUP[s] === groupIndex);
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
                <button type="button" onClick={next} className="ob-btn-next">
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
                    className="ob-input"
                  />
                </div>
              )}
              <div className="ob-cta">
                <button type="button" onClick={back} className="ob-btn-back">
                  {t("common.back")}
                </button>
                <button type="button" disabled={!isBusinessValid} onClick={next} className="ob-btn-next">
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
                <button type="button" onClick={back} className="ob-btn-back">
                  {t("common.back")}
                </button>
                <button type="button" disabled={!isGoalsValid} onClick={next} className="ob-btn-next">
                  {t("onboarding.continue")}
                </button>
              </div>
              <button type="button" onClick={() => persistAndLeave("/templates", false)} className="ob-skip">
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
                    const ring = CATEGORY_RING[template.category] ?? "rgba(37,54,224,.10)";
                    const dot = CATEGORY_DOT[template.category] ?? "#8A8A94";
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
                <button type="button" onClick={back} className="ob-btn-back">
                  {t("common.back")}
                </button>
                <button type="button" disabled={!isTeamValid} onClick={next} className="ob-btn-next">
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
                <button type="button" onClick={back} className="ob-btn-back">
                  {t("common.back")}
                </button>
                <button type="button" onClick={next} className="ob-btn-next">
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
              const ring = CATEGORY_RING[selected.category] ?? "rgba(37,54,224,.10)";

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
                            {t("onboarding.done.willConnectPrefix")} <b>{joinWithAnd(appNames)}</b>{" "}
                            {t("onboarding.done.willConnectSuffix")}
                          </span>
                        ) : (
                          <span>{t("onboarding.done.noAccounts")}</span>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="ob-cta">
                    <button type="button" onClick={back} className="ob-btn-back">
                      {t("common.back")}
                    </button>
                    <button type="button" disabled={launching} onClick={handleLaunch} className="ob-btn-next">
                      {launching ? t("templates.marketplace.hiring") : t("onboarding.done.start", { name: persona.name })}
                      {!launching && <ArrowRight className="h-4 w-4" />}
                    </button>
                  </div>
                  <button type="button" onClick={() => persistAndLeave("/templates", true)} className="ob-skip">
                    {t("onboarding.done.skip")}
                  </button>
                </>
              );
            })()}
        </div>
      </main>

      <style jsx>{`
        .ob-aurora {
          position: fixed;
          inset: 0;
          overflow: hidden;
          pointer-events: none;
          z-index: 0;
        }
        .ob-aurora b {
          position: absolute;
          display: block;
          border-radius: 50%;
          filter: blur(80px);
        }
        .ob-aurora::after {
          content: "";
          position: absolute;
          inset: 0;
          pointer-events: none;
          opacity: 0.16;
          mix-blend-mode: soft-light;
          background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/></filter><rect width='160' height='160' filter='url(%23n)'/></svg>");
        }
        .ob-a1 {
          width: min(1280px, 128vw);
          height: 740px;
          left: 50%;
          top: 44%;
          transform: translate(-50%, -50%);
          background: radial-gradient(
            closest-side,
            rgba(150, 198, 255, 0.58),
            rgba(150, 198, 255, 0.4) 38%,
            rgba(150, 198, 255, 0.16) 64%,
            rgba(150, 198, 255, 0.04) 84%,
            rgba(150, 198, 255, 0) 100%
          );
          animation: ob-drift1 26s ease-in-out infinite;
        }
        .ob-a2 {
          width: min(1080px, 116vw);
          height: 680px;
          left: 46%;
          top: 92%;
          transform: translate(-50%, -50%);
          background: radial-gradient(
            closest-side,
            rgba(247, 190, 226, 0.5),
            rgba(247, 190, 226, 0.33) 38%,
            rgba(247, 190, 226, 0.13) 64%,
            rgba(247, 190, 226, 0.03) 84%,
            rgba(247, 190, 226, 0) 100%
          );
          animation: ob-drift2 31s ease-in-out infinite;
        }
        .ob-a3 {
          width: 680px;
          height: 560px;
          left: 63%;
          top: 50%;
          transform: translate(-50%, -50%);
          background: radial-gradient(
            closest-side,
            rgba(186, 176, 255, 0.38),
            rgba(186, 176, 255, 0.24) 40%,
            rgba(186, 176, 255, 0.09) 66%,
            rgba(186, 176, 255, 0) 100%
          );
          animation: ob-drift3 23s ease-in-out infinite;
        }
        @keyframes ob-drift1 {
          50% {
            transform: translate(-52%, -46%) scale(1.06);
          }
        }
        @keyframes ob-drift2 {
          50% {
            transform: translate(-44%, -54%) scale(1.08);
          }
        }
        @keyframes ob-drift3 {
          50% {
            transform: translate(-58%, -50%) scale(0.94);
          }
        }

        .ob-shell {
          position: relative;
          z-index: 1;
          min-height: 100vh;
          display: flex;
          flex-direction: column;
        }

        .ob-top {
          position: relative;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 20px 26px;
          flex: 0 0 auto;
        }
        .ob-logo {
          height: 26px;
          width: auto;
          display: block;
        }
        .ob-acct {
          position: absolute;
          right: 26px;
          top: 50%;
          transform: translateY(-50%);
          display: flex;
          align-items: center;
          gap: 9px;
          padding: 5px 14px 5px 5px;
          border: 1px solid #e7e7ec;
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.72);
          backdrop-filter: blur(8px);
          font-size: 13px;
          font-weight: 550;
          letter-spacing: -0.01em;
          color: #1a1a1f;
        }
        .ob-acct span {
          width: 26px;
          height: 26px;
          border-radius: 50%;
          display: grid;
          place-items: center;
          background: #2536e0;
          color: #fff;
          font-size: 11.5px;
          font-weight: 650;
        }
        .ob-acct b {
          font-weight: 550;
        }
        .ob-exit {
          position: absolute;
          left: 26px;
          top: 50%;
          transform: translateY(-50%);
          font-size: 12.5px;
          color: #8a8a94;
          padding: 7px 11px;
          border-radius: 8px;
          border: 0;
          background: none;
          cursor: pointer;
          transition: color 0.14s, background 0.14s;
        }
        .ob-exit:hover {
          color: #3d3d46;
          background: #f1f1f4;
        }

        .ob-rail {
          position: fixed;
          left: 26px;
          top: 74px;
          z-index: 2;
          display: flex;
          flex-direction: column;
          gap: 14px;
        }
        .ob-rl {
          display: flex;
          align-items: center;
          gap: 12px;
          font-size: 14px;
          letter-spacing: -0.012em;
          color: #8a8a94;
          padding: 2px 4px 2px 0;
          border-radius: 10px;
          text-align: left;
          transition: color 0.16s;
          cursor: default;
          border: 0;
          background: none;
        }
        .ob-rl i {
          flex: 0 0 auto;
          width: 19px;
          height: 19px;
          border-radius: 50%;
          border: 1.6px solid #e7e7ec;
          display: grid;
          place-items: center;
          color: #fff;
          transition: border-color 0.18s, background 0.18s;
        }
        .ob-rl-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #2536e0;
          display: block;
        }
        .ob-rl.is-now {
          color: #1a1a1f;
          font-weight: 600;
        }
        .ob-rl.is-now i {
          border-color: #2536e0;
        }
        .ob-rl.is-done {
          color: #3d3d46;
        }
        .ob-rl.is-done i {
          background: #2536e0;
          border-color: #2536e0;
        }
        .ob-rl.can-go {
          cursor: pointer;
        }
        .ob-rl.can-go:hover {
          color: #1a1a1f;
        }

        .ob-body {
          flex: 1 1 auto;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 8px 26px 48px;
          min-height: 0;
        }
        .ob-step {
          width: 100%;
          max-width: 980px;
          margin: 0 auto;
          text-align: center;
          animation: ob-rise 0.38s cubic-bezier(0.2, 0.7, 0.3, 1) both;
        }
        @keyframes ob-rise {
          from {
            opacity: 0;
            transform: translateY(12px);
          }
          to {
            opacity: 1;
            transform: none;
          }
        }

        /* Orb is its own component, so styled-jsx's scope class (added only
           to elements written literally inside OnboardingPage's own JSX)
           never reaches the div it renders - :global() on the orb part of
           each selector opts out of that scoping, same trick as .ob-tm-av
           below for PersonaAvatar. .ob-step itself stays scoped as the
           anchor. */
        .ob-step :global(.ob-orb) {
          position: relative;
          width: 58px;
          height: 58px;
          margin: 0 auto 26px;
        }
        .ob-step :global(.ob-orb)::before,
        .ob-step :global(.ob-orb)::after {
          content: "";
          position: absolute;
          inset: 0;
        }
        .ob-step :global(.ob-orb)::before {
          border-radius: 52% 48% 44% 56% / 50% 54% 46% 50%;
          background: conic-gradient(from 200deg, #8cc6ff, #b7a6ff, #f3aadb, #8fd8ff, #8cc6ff);
          filter: blur(6px);
          animation: ob-orbm 8s ease-in-out infinite;
        }
        .ob-step :global(.ob-orb)::after {
          inset: 9px;
          border-radius: 50%;
          filter: blur(4px);
          background: radial-gradient(
            circle at 36% 30%,
            rgba(255, 255, 255, 0.92),
            rgba(255, 255, 255, 0.2) 32%,
            rgba(255, 255, 255, 0) 58%
          );
          animation: ob-orbm 8s ease-in-out infinite reverse;
        }
        @keyframes ob-orbm {
          50% {
            border-radius: 44% 56% 54% 46% / 46% 44% 56% 54%;
            transform: rotate(140deg) scale(1.08);
          }
        }
        .ob-step :global(.ob-orb.sm) {
          width: 44px;
          height: 44px;
          margin-bottom: 20px;
        }

        .ob-step :global(h1) {
          margin: 0;
          font-size: clamp(32px, 5.2vw, 58px);
          line-height: 1.05;
          font-weight: 760;
          letter-spacing: -0.036em;
          color: #16162e;
        }
        .ob-step :global(h1 em) {
          font-style: normal;
          color: #2536e0;
        }
        .ob-sub {
          margin: 18px auto 0;
          max-width: 520px;
          font-size: 15.5px;
          line-height: 1.55;
          color: #6b6b76;
        }

        .ob-cta {
          margin-top: 34px;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
        }
        .ob-btn-next {
          display: inline-flex;
          align-items: center;
          gap: 9px;
          height: 50px;
          padding: 0 30px;
          border-radius: 999px;
          border: 0;
          background: #2536e0;
          color: #fff;
          font-size: 15px;
          font-weight: 600;
          letter-spacing: -0.012em;
          box-shadow: 0 10px 26px rgba(37, 54, 224, 0.26);
          cursor: pointer;
          transition: background 0.14s, transform 0.12s, box-shadow 0.14s, opacity 0.14s;
        }
        .ob-btn-next:hover:not(:disabled) {
          background: #1e2fdb;
          box-shadow: 0 12px 30px rgba(37, 54, 224, 0.32);
        }
        .ob-btn-next:active:not(:disabled) {
          transform: translateY(1px);
        }
        .ob-btn-next:disabled {
          background: #b7b7c4;
          box-shadow: none;
          cursor: not-allowed;
        }
        .ob-btn-back {
          height: 50px;
          padding: 0 20px;
          border-radius: 999px;
          border: 0;
          background: none;
          font-size: 14.5px;
          font-weight: 500;
          color: #6b6b76;
          cursor: pointer;
          transition: color 0.14s, background 0.14s;
        }
        .ob-btn-back:hover {
          color: #1a1a1f;
          background: #f1f1f4;
        }
        .ob-skip {
          margin-top: 18px;
          font-size: 12.5px;
          color: #8a8a94;
          padding: 7px 11px;
          border-radius: 8px;
          border: 0;
          background: none;
          cursor: pointer;
          transition: color 0.14s, background 0.14s;
        }
        .ob-skip:hover {
          color: #3d3d46;
          background: #f1f1f4;
        }

        .ob-form {
          margin: 24px auto 0;
          max-width: 420px;
          text-align: left;
        }
        .ob-fl {
          display: block;
          font-size: 13.5px;
          font-weight: 600;
          letter-spacing: -0.01em;
          margin-bottom: 10px;
          color: #1a1a1f;
        }
        .ob-input {
          width: 100%;
          height: 56px;
          padding: 0 18px;
          border: 1px solid #e7e7ec;
          border-radius: 14px;
          background: #fff;
          font-size: 15px;
          letter-spacing: -0.01em;
          transition: border-color 0.14s, box-shadow 0.14s;
        }
        .ob-input::placeholder {
          color: #8a8a94;
        }
        .ob-input:focus {
          outline: none;
          border-color: #2536e0;
          box-shadow: 0 0 0 4px rgba(37, 54, 224, 0.1);
        }

        .ob-chips {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          justify-content: center;
          max-width: 760px;
          margin: 0 auto;
        }
        /* Chip is its own component - same :global() reasoning as .ob-orb
           above, anchored on .ob-chips (written inline in OnboardingPage's
           own JSX) instead. */
        .ob-chips :global(.ob-chip) {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          height: 42px;
          padding: 0 18px;
          border-radius: 999px;
          border: 1px solid #e7e7ec;
          background: rgba(255, 255, 255, 0.62);
          font-size: 14px;
          letter-spacing: -0.012em;
          color: #3d3d46;
          cursor: pointer;
          transition: border-color 0.15s, background 0.15s, color 0.15s, transform 0.12s, box-shadow 0.15s;
        }
        .ob-chips :global(.ob-chip):hover {
          border-color: #d3d3dc;
          background: #fff;
          box-shadow: 0 6px 16px rgba(20, 22, 40, 0.06);
        }
        .ob-chips :global(.ob-chip):active {
          transform: translateY(1px);
        }
        .ob-chips :global(.ob-chip)[aria-pressed="true"] {
          border-color: #2536e0;
          background: #eef1fe;
          color: #2536e0;
          font-weight: 560;
        }
        .ob-chips :global(.ob-chip) :global(.ob-ic) {
          opacity: 0.55;
        }
        .ob-chips :global(.ob-chip)[aria-pressed="true"] :global(.ob-ic) {
          opacity: 1;
        }

        .ob-team {
          margin: 36px auto 0;
          display: grid;
          gap: 16px;
          grid-template-columns: repeat(3, minmax(0, 1fr));
        }
        .ob-team.two {
          grid-template-columns: repeat(2, minmax(0, 1fr));
          max-width: 680px;
        }
        .ob-team.one {
          grid-template-columns: minmax(0, 340px);
          justify-content: center;
        }
        .ob-tm {
          position: relative;
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 26px 20px 22px;
          border: 1px solid #e7e7ec;
          border-radius: 22px;
          background: #fff;
          text-align: center;
          cursor: pointer;
          transition: border-color 0.16s, box-shadow 0.18s, transform 0.18s;
        }
        .ob-tm:hover {
          border-color: #dcdce4;
          box-shadow: 0 14px 34px rgba(20, 22, 40, 0.09);
          transform: translateY(-3px);
        }
        .ob-tm[aria-pressed="true"] {
          border-color: #2536e0;
          box-shadow: 0 0 0 1px #2536e0, 0 16px 38px rgba(37, 54, 224, 0.14);
        }
        .ob-tm-tick {
          position: absolute;
          top: 14px;
          right: 14px;
          width: 22px;
          height: 22px;
          border-radius: 50%;
          background: #2536e0;
          color: #fff;
          display: grid;
          place-items: center;
        }
        .ob-tm :global(.ob-tm-av) {
          background: linear-gradient(150deg, #eef0fe, #e4e7fc 55%, #dce0fa);
        }
        .ob-tm-name {
          margin-top: 15px;
          font-size: 17px;
          font-weight: 650;
          letter-spacing: -0.015em;
          text-transform: uppercase;
          color: #1a1a1f;
        }
        .ob-tm-role {
          margin-top: 5px;
          font-size: 12px;
          font-weight: 500;
          color: #6b6b76;
        }
        .ob-tm-cat {
          margin-top: 11px;
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 3px 10px;
          border-radius: 999px;
          background: #fafafa;
          border: 1px solid #f0f0f4;
          font-size: 11px;
          color: #6b6b76;
        }
        .ob-tm-cat i {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          display: block;
        }
        .ob-tm-desc {
          margin-top: 13px;
          font-size: 13px;
          line-height: 1.55;
          color: #3d3d46;
          min-height: 60px;
        }
        .ob-tm-why {
          margin-top: 12px;
          font-size: 11.5px;
          color: #8a8a94;
          display: inline-flex;
          align-items: center;
          gap: 6px;
        }
        .ob-tm-why i {
          width: 5px;
          height: 5px;
          border-radius: 50%;
          background: #2536e0;
          opacity: 0.5;
          display: block;
        }

        .ob-say {
          margin: 24px auto 0;
          max-width: 560px;
          padding: 16px 20px;
          border-radius: 18px;
          background: rgba(255, 255, 255, 0.78);
          border: 1px solid #f0f0f4;
          box-shadow: 0 8px 24px rgba(20, 22, 40, 0.05);
          text-align: left;
        }
        .ob-say b {
          display: block;
          font-size: 10.5px;
          font-weight: 650;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: #8a8a94;
          margin-bottom: 7px;
        }
        .ob-say i {
          font-style: normal;
          font-size: 14.5px;
          line-height: 1.6;
          color: #3d3d46;
        }

        .ob-sum {
          margin: 34px auto 0;
          max-width: 520px;
          border: 1px solid #e7e7ec;
          border-radius: 22px;
          background: #fff;
          padding: 26px 24px;
          text-align: left;
          box-shadow: 0 18px 44px rgba(20, 22, 40, 0.08);
        }
        .ob-sum-hd {
          display: flex;
          align-items: center;
          gap: 15px;
        }
        .ob-sum-nm {
          font-size: 17px;
          font-weight: 660;
          letter-spacing: -0.018em;
          color: #1a1a1f;
        }
        .ob-sum-rl {
          margin-top: 3px;
          font-size: 12.5px;
          color: #6b6b76;
        }
        .ob-sum-list {
          margin: 20px 0 0;
          padding: 18px 0 0;
          border-top: 1px solid #f0f0f4;
          display: grid;
          gap: 12px;
        }
        .ob-sum-li {
          display: grid;
          grid-template-columns: 18px 1fr;
          gap: 11px;
          align-items: start;
          font-size: 13.5px;
          color: #3d3d46;
        }
        .ob-sum-li :global(b) {
          font-weight: 600;
          color: #1a1a1f;
        }

        @media (max-width: 1180px) {
          .ob-rail {
            position: static;
            flex-direction: row;
            justify-content: center;
            gap: 8px;
            padding: 0 26px 4px;
            margin-bottom: 8px;
          }
          .ob-rl span {
            display: none;
          }
          .ob-rl.is-now span {
            display: inline;
          }
          .ob-rl i {
            width: 16px;
            height: 16px;
          }
        }
        @media (max-width: 880px) {
          .ob-team {
            grid-template-columns: 1fr;
            max-width: 400px;
          }
          .ob-team.two,
          .ob-team.one {
            grid-template-columns: 1fr;
            max-width: 400px;
          }
          .ob-tm-desc {
            min-height: 0;
          }
          .ob-body {
            padding-bottom: 36px;
          }
        }
        @media (prefers-reduced-motion: reduce) {
          .ob-aurora b,
          .ob-orb::before,
          .ob-orb::after,
          .ob-step {
            animation-duration: 0.01ms !important;
          }
        }
      `}</style>
    </div>
  );
}
