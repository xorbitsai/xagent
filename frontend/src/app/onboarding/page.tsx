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
import { markOnboardingSaveEscaped, updateUserPreferences, type UserPreferences } from "@/lib/user-preferences";
import { hireAgentFromTemplate } from "@/lib/hire-agent";
import { PersonaAvatar } from "@/components/templates/persona-avatar";
import { categoryLabel } from "@/lib/template-categories";
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
import type { Template, TemplateDetail } from "@/types/template";
import type { TranslationKey } from "@/i18n/translations";

const branding = getBrandingFromEnv();

type StepId = "welcome" | "business" | "goals" | "team" | "voice" | "done";

// Single source of truth for the rail's grouping - "My team" deliberately
// spans both the team-pick and voice-pick steps (matching the reference UI
// exactly), not one group per step; only "Launch" is its own single-step
// group. STEP_ORDER/STEP_GROUP/GROUP_KEYS all used to be separately authored
// literals that had to be hand-kept in sync; deriving them here instead
// makes that desync structurally impossible.
const STEP_GROUPS: { key: TranslationKey; steps: StepId[] }[] = [
  { key: "onboarding.rail.aboutYou", steps: ["welcome", "business"] },
  { key: "onboarding.rail.goals", steps: ["goals"] },
  { key: "onboarding.rail.myTeam", steps: ["team", "voice"] },
  { key: "onboarding.rail.launch", steps: ["done"] },
];
const GROUP_KEYS: TranslationKey[] = STEP_GROUPS.map((g) => g.key);
const STEP_ORDER: StepId[] = STEP_GROUPS.flatMap((g) => g.steps);
const STEP_GROUP: Record<StepId, number> = STEP_GROUPS.reduce((acc, group, groupIndex) => {
  for (const step of group.steps) acc[step] = groupIndex;
  return acc;
}, {} as Record<StepId, number>);

// Shared by persistAndLeave's exits AND handleLaunch's "Start with X": both
// give up and proceed anyway after this many consecutive failures saving to
// the same bucket, rather than trapping the user on this full-screen page
// forever. 2 (not 1): a single transient blip shouldn't skip the "retry once
// in place" step entirely; not higher, since this is the only page in the
// app with no other navigation affordance if the backend really is down.
const MAX_SAVE_FAILURES_BEFORE_ESCAPE = 2;
// handleLaunch's fixed, non-URL bucket in saveFailureCountByDestRef - see
// the comment on that ref for why it can't use a real destination key.
const LAUNCH_FAILURE_KEY = "__launch__";

// The category ring/dot colors below are the reference UI's fixed literal
// values (onboarding.html's [data-cat] rules), deliberately left un-themed:
// each color is this category's fixed identity (a small solid dot, a low-alpha
// ring tint), not text-on-background contrast that shifts per theme - unlike
// the .ob-* styles further down, which now mostly use theme tokens.
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
  disabled,
}: {
  selected: boolean;
  onClick: () => void;
  children: React.ReactNode;
  icon?: React.ReactNode;
  disabled?: boolean;
}) {
  return (
    <button type="button" aria-pressed={selected} onClick={onClick} disabled={disabled} className="ob-chip">
      {icon}
      {children}
    </button>
  );
}

export default function OnboardingPage() {
  const router = useRouter();
  const { user } = useAuth();
  const { t, locale } = useI18n();
  // The identity this wizard was opened for - a PR review finding caught
  // that this page's own state (work/industry/goals/voice, the selected
  // template, etc.) has no identity binding of its own, unlike the escape
  // flag and AuthGuard's own onboarding-redirect check, which both account
  // for AuthProvider replacing `user` (via its `storage` event listener,
  // on a same-origin cross-tab login as a DIFFERENT user) with no remount
  // of this page. Without this, a half-filled wizard's answers - or an
  // in-flight save/hire that started under the original identity - could
  // be sent, or completed, under a swapped-in identity's session. This
  // deliberately does NOT reset or rehydrate the wizard for the new
  // identity (a larger redesign); it only guards the handful of mutating
  // calls below so identity A's data is never written or completed as
  // identity B.
  const wizardUserIdRef = useRef(user?.id);
  // Read live at every guard checkpoint below, NOT the `user` variable
  // itself - a self-review agent caught that `user` comes from a plain
  // `const` destructure, so it's frozen to whatever AuthProvider returned
  // at the specific render that created the CURRENTLY RUNNING closure
  // (handleLaunch/trySavePreferences/persistAndLeave are all recreated
  // fresh every render, but an already-invoked call keeps running with
  // the closure it started with). AuthProvider replaces `user` with a
  // brand new object via React state (setProjection in auth-context.tsx),
  // not an in-place mutation, so an identity swap that happens WHILE one
  // of those functions is already awaiting a save/hire call is invisible
  // to a `user?.id` read inside it - exactly the scenario this guard
  // exists for. A ref updated every render via the effect below is always
  // read live regardless of which render's closure is doing the reading.
  const currentUserIdRef = useRef(user?.id);
  useEffect(() => {
    currentUserIdRef.current = user?.id;
  }, [user?.id]);

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
  // Consecutive save failures, keyed by "destination" - see the escape
  // hatches in persistAndLeave/handleLaunch below. Per-destination (not one
  // global count): "/task" and "/templates" represent genuinely different
  // user intents, and a failure on one shouldn't spend down the other's own
  // first-attempt retry-in-place chance. header Skip -> "/task"; goals-step
  // and done-step skip both -> "/templates", which is intentional (both mean
  // the same "leave to browse templates" exit, just from different steps).
  // LAUNCH_FAILURE_KEY is the one non-URL entry: "Start with X" doesn't know
  // its real destination (an existing agent vs. a freshly hired one) until
  // after the save succeeds, so it gets its own fixed bucket instead.
  const saveFailureCountByDestRef = useRef<Record<string, number>>({});
  // Whether ANY failure in the current streak for this key was non-retryable
  // (a permanent 4xx rejection of this exact payload) - a PR review finding
  // caught that handleLaunch's escape only checked the LATEST failure's
  // retryable bit, so a permanent rejection followed by a later transient
  // one (e.g. 422 then a 503) still escalated to an irreversible hire, even
  // though the payload itself was already known to be rejected outright.
  // Reset alongside the count, on the same success/streak-reset events.
  const saveFailureHasPermanentByDestRef = useRef<Record<string, boolean>>({});

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
  // Goal-matched recommendations with a real persona, filtered BEFORE
  // capping at 3 (not after) - a 4th-ranked match with a real persona
  // should fill a slot vacated by a top-3 match that turns out to have
  // none, not lose out to the cap being applied on the unfiltered list
  // first. Kept separate from validRecommended below (rather than just
  // reading its length) - a PR review finding caught that validRecommended
  // can be entirely fallback filler cards (goalId: null) when nothing
  // actually matched, and the team step's "N other matches waiting"
  // subtitle must count real goal matches specifically, not whatever ends
  // up rendered.
  const matchedRecommended = useMemo(
    () => recommended.filter((r) => templateById.get(r.templateId)?.persona),
    [recommended, templateById]
  );

  const validRecommended = useMemo(() => {
    if (matchedRecommended.length > 0) return matchedRecommended.slice(0, 3);
    // Every recommendation for the selected goals failed to load or has no
    // persona (or nothing has loaded yet) - fall back to the same 3 defaults
    // recommendedTemplates() uses when no goals were picked at all, so a
    // template-catalog gap can't leave this step with zero cards to show.
    return ONBOARDING_FALLBACK_TEMPLATE_IDS.map((templateId) => ({ templateId, goalId: null }))
      .filter((r) => templateById.get(r.templateId)?.persona)
      .slice(0, 3);
  }, [matchedRecommended, templateById]);

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

  // "professional" is a real, legitimate voice choice, not an empty
  // placeholder - so unlike goals (empty array = clearly untouched), the
  // default value alone can't distinguish "user picked this" from "user
  // never got here." Only include it once the voice step has actually been
  // reached, so e.g. the header Skip (reachable from any step at any time)
  // can't persist a policy the user never saw.
  //
  // A latch, not a live `stepIndex >= indexOf("voice")` comparison: the
  // latter briefly looked right but silently DROPPED an already-made choice
  // the moment the user pressed Back past the voice step - reaching voice,
  // picking one, going back to team, then hitting the header Skip lost the
  // pick entirely, the same class of bug this round fixed for goals/voice
  // in the other direction (persisting a choice never made).
  const hasReachedVoiceRef = useRef(false);

  // Step transitions swap the entire JSX subtree (only one `step === "..."`
  // branch is ever mounted), so the browser drops focus to <body> on every
  // step change - a keyboard/screen-reader user gets no indication a new
  // step even loaded and has to Tab from the very top of the page each time.
  // Attached to whichever step's <h1> is currently mounted (only one is at
  // once), moved there on every step change below.
  const stepHeadingRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    stepHeadingRef.current?.focus();
  }, [step]);

  const goTo = (index: number) => {
    const clamped = Math.max(0, Math.min(STEP_ORDER.length - 1, index));
    setStepIndex(clamped);
    lastVisitedInGroupRef.current[STEP_GROUP[STEP_ORDER[clamped]]] = clamped;
    if (STEP_ORDER[clamped] === "voice") hasReachedVoiceRef.current = true;
  };
  const next = () => goTo(stepIndex + 1);
  const back = () => goTo(stepIndex - 1);

  const toggleGoal = (id: string) => {
    setGoals((prev) => (prev.includes(id) ? prev.filter((g) => g !== id) : [...prev, id]));
  };

  // Shared by persistAndLeave and handleLaunch - these used to independently
  // hand-build the same PATCH payload and had already drifted (goals was
  // conditional in one, unconditional in the other) before this existed.
  // Explicit `null` (not omission) once a real, non-"other" department is
  // chosen: a PR review finding caught that switching away from "Other"
  // after typing an industry only cleared the LOCAL `industry` state -
  // since the save is a merge PATCH, omitting the key left a stale
  // industry value stored server-side, silently paired with the new
  // department (e.g. {department: "other", industry: "Legal"} becoming
  // {department: "sales", industry: "Legal"}). `null` is this endpoint's
  // actual "clear this field" signal (see UpdatePreferencesRequest).
  // `undefined` (an untouched field, work never set) still omits the key.
  const industryPayloadValue = (): string | null | undefined => {
    if (work === "other") return industry.trim() || undefined;
    return work ? null : undefined;
  };

  // `includeOnboarded: false` is handleLaunch's own case - see the comment
  // where it calls this. Every other caller wants the default (true).
  const buildPreferencesPayload = ({ includeOnboarded = true }: { includeOnboarded?: boolean } = {}): UserPreferences => {
    const industryValue = industryPayloadValue();
    return {
      ...(includeOnboarded ? { onboarded: true } : {}),
      ...(work ? { department: work } : {}),
      ...(industryValue !== undefined ? { industry: industryValue } : {}),
      ...(goals.length ? { goals } : {}),
      ...(hasReachedVoiceRef.current ? { voice } : {}),
    };
  };

  // The three-way result of one preferences-save attempt, shared by both
  // persistAndLeave's exits and handleLaunch's "Start with X" - both used to
  // hand-roll this same failure-count/toast/reset/escape policy
  // independently, which had already drifted once (this round's PR review
  // found handleLaunch's copy missing the retryable check persistAndLeave's
  // never needed) and kept costing a fresh review cycle every time a new
  // edge case surfaced in one copy but not the other.
  type SavePreferencesOutcome = "saved" | "retry_in_place" | "escaped";

  // `requireRetryableToEscape` is the one real behavioral difference between
  // the two callers, so it stays a parameter rather than being hidden inside
  // this function: persistAndLeave's escape just leaves without saving (not
  // irreversible), so it must escape after MAX_SAVE_FAILURES_BEFORE_ESCAPE
  // regardless of failure type - the only way out on the one page in the app
  // with no other nav affordance. handleLaunch's escape proceeds to an
  // IRREVERSIBLE hireAgentFromTemplate call, so it must only escape for a
  // RETRYABLE failure - a non-retryable one (a 4xx) will keep rejecting the
  // identical payload no matter how many times it's retried, so escalating
  // anyway would hire an agent while preferences were never actually saved.
  const trySavePreferences = async (
    failureKey: string,
    { requireRetryableToEscape, includeOnboarded = true }: { requireRetryableToEscape: boolean; includeOnboarded?: boolean }
  ): Promise<SavePreferencesOutcome> => {
    // See wizardUserIdRef's and currentUserIdRef's comments - never send
    // this identity's answers under a swapped-in session. Not counted as a
    // failure of THIS payload (it was never actually sent), so it doesn't
    // feed the escape counter.
    if (currentUserIdRef.current !== wizardUserIdRef.current) {
      if (isMountedRef.current) toast.error(t("onboarding.done.saveFailed"));
      return "retry_in_place";
    }
    const saved = await updateUserPreferences(buildPreferencesPayload({ includeOnboarded }));
    if (saved.ok) {
      saveFailureCountByDestRef.current[failureKey] = 0;
      saveFailureHasPermanentByDestRef.current[failureKey] = false;
      return "saved";
    }
    const failureCount = (saveFailureCountByDestRef.current[failureKey] ?? 0) + 1;
    saveFailureCountByDestRef.current[failureKey] = failureCount;
    if (!saved.retryable) saveFailureHasPermanentByDestRef.current[failureKey] = true;
    if (isMountedRef.current) toast.error(t("onboarding.done.saveFailed"));
    // Don't escape on the FIRST failure: onboarded would stay false
    // server-side, and AuthGuard would just bounce the user right back -
    // retrying in place is the better first response. But a save that keeps
    // failing must not trap the caller forever - let it through after a
    // couple of tries even though the save didn't land; they'll just be
    // asked again next session instead of being stuck this one.
    //
    // requireRetryableToEscape callers must look at whether a non-retryable
    // failure was EVER seen in this streak, not just this latest attempt -
    // a PR review finding caught that checking only the latest attempt let a
    // permanent 4xx rejection escalate anyway if the very next retry
    // happened to fail transiently instead (e.g. 422 then a network blip).
    // The identical, already-rejected payload doesn't become escapable just
    // because a later attempt failed a different way.
    const canEscape =
      failureCount >= MAX_SAVE_FAILURES_BEFORE_ESCAPE &&
      (!requireRetryableToEscape || !saveFailureHasPermanentByDestRef.current[failureKey]);
    return canEscape ? "escaped" : "retry_in_place";
  };

  // handleLaunch's own last-mile save - see the comment where it's called
  // for why onboarded:true has to wait until this point. Only safe to call
  // once an agent genuinely exists (already hired, freshly re-confirmed
  // hired, or just hired). Awaited so a failure here can still fall back to
  // the escape flag (this save's own retries/escalation don't apply - the
  // agent is real either way, so navigation must not be blocked on this one
  // best-effort field ever landing).
  //
  // Sends the FULL payload (buildPreferencesPayload's default
  // includeOnboarded:true), not just `{onboarded: true}` alone - a PR
  // review finding caught that the earlier main save in handleLaunch below
  // is deliberately allowed to escape (proceed anyway) after repeated
  // RETRYABLE failures, meaning department/industry/goals/voice can still
  // be genuinely unsaved by the time this runs. Sending only the marker
  // here would let this call succeed on its own and durably complete the
  // account - onboarded:true, but with the rest of the answers never
  // actually persisted, and no way back in to finish since onboarded:true
  // skips onboarding entirely. Resending everything together here means a
  // successful completion always carries the real answers with it.
  const markOnboardedAndNavigate = async (destination: string) => {
    // See wizardUserIdRef's and currentUserIdRef's comments - never
    // complete onboarding, or write this identity's collected answers,
    // under a swapped-in session. No escape marker here: that flag is for
    // a save that genuinely failed, not for an attempt this identity never
    // actually got to make.
    if (currentUserIdRef.current !== wizardUserIdRef.current) {
      if (isMountedRef.current) {
        launchingRef.current = false;
        setLaunching(false);
      }
      return;
    }
    const onboardedSave = await updateUserPreferences(buildPreferencesPayload());
    if (isMountedRef.current) {
      // wizardUserIdRef.current, not user?.id/currentUserIdRef - this flag
      // is inherently about the ORIGINAL wizard identity's save outcome
      // (the guard above already confirmed identity hadn't swapped when
      // this PATCH was sent), not whoever happens to be live by the time
      // this line runs after the await.
      if (!onboardedSave.ok) markOnboardingSaveEscaped(wizardUserIdRef.current);
      router.replace(destination);
    }
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
    const outcome = await trySavePreferences(destination, { requireRetryableToEscape: false });
    if (outcome === "retry_in_place") {
      if (isMountedRef.current) {
        launchingRef.current = false;
        setLaunching(false);
      }
      return;
    }
    // replace, not push: leaving a /onboarding entry in history means a
    // single Back press would return the user to a stale, reset-to-step-0
    // wizard even though they've just finished with it (or escaped it).
    if (isMountedRef.current) {
      // Reset on "escaped" too (not just retry_in_place above), matching
      // the pre-refactor behavior of resetting on every failure regardless
      // of outcome - NOT on "saved", which correctly leaves the button in
      // its launching state through a successful exit's navigation, same
      // as before. The extraction that introduced trySavePreferences
      // originally dropped this reset on the "escaped" outcome specifically,
      // on the reasoning that a navigation is about to unmount the page
      // anyway - but that leaves the exit button/spinner visibly stuck in
      // its launching state for however long router.replace takes to
      // actually resolve and unmount, a real (if narrow) regression caught
      // by a dedicated self-review pass on the refactor itself.
      if (outcome === "escaped") {
        launchingRef.current = false;
        setLaunching(false);
      }
      // Escaping despite the failure only works if AuthGuard's own
      // onboarding check on `destination` doesn't immediately see
      // onboarded:false and bounce the user right back (self-review found
      // this happens for real: sometimes as a bounce loop, sometimes it
      // silently means the escape never actually reaches its destination).
      // This flag tells that check to stand down once. wizardUserIdRef.current,
      // not user?.id - see markOnboardedAndNavigate's identical note.
      if (outcome === "escaped") markOnboardingSaveEscaped(wizardUserIdRef.current);
      router.replace(destination);
    }
  };

  const handleLaunch = async () => {
    const selected = templateById.get(agentTemplateId);
    if (!selected || !selected.persona || launchingRef.current) return;

    launchingRef.current = true;
    setLaunching(true);
    try {
      // onboarded is deliberately EXCLUDED from this save (includeOnboarded:
      // false) - a PR review finding caught that saving it durably true
      // here, before the freshness re-check and hireAgentFromTemplate below
      // (both of which can still fail or the component can still unmount),
      // let a later failure leave the backend saying "onboarded" with no
      // agent/task ever actually delivered - a guard check afterward would
      // then skip onboarding entirely with no way back in to finish it.
      // It's saved separately, only once an agent is actually in hand, via
      // markOnboardedAndNavigate above.
      const outcome = await trySavePreferences(LAUNCH_FAILURE_KEY, {
        requireRetryableToEscape: true,
        includeOnboarded: false,
      });
      if (outcome === "retry_in_place") {
        if (isMountedRef.current) {
          launchingRef.current = false;
          setLaunching(false);
        }
        return;
      }

      if (selected.hired && selected.hired_agent_id) {
        await markOnboardedAndNavigate(`/agent/${selected.hired_agent_id}`);
        return;
      }

      // `selected.hired` was fetched once on mount and never revalidated, so
      // hiring the same template from another tab/session mid-wizard would
      // still see hired: false here. Mirrors the identical re-check in
      // templates/[id]/page-client.tsx: hireAgentFromTemplate's resolve step
      // is idempotent (reuses the agent), but task/create always mints a new
      // task, so without this a race would seed a second opening message
      // onto an agent the user already has a real conversation with. Best
      // effort only: if the recheck itself fails, fall through to hiring
      // normally rather than blocking the action on it.
      // Captured rather than navigating from inside this try - the block
      // below is only for the best-effort recheck fetch itself; if
      // markOnboardedAndNavigate ran here and somehow threw, this catch
      // would wrongly treat it as "recheck failed, fall through to hiring"
      // instead of the save/navigation failure it actually is.
      let freshlyHiredAgentId: number | undefined;
      try {
        const freshCheck = await apiRequest(
          `${getApiUrl()}/api/templates/${encodeURIComponent(selected.id)}?lang=${locale}`
        );
        if (freshCheck.ok) {
          const freshTemplate = (await freshCheck.json()) as TemplateDetail;
          if (freshTemplate.hired && freshTemplate.hired_agent_id) {
            freshlyHiredAgentId = freshTemplate.hired_agent_id;
          }
        }
      } catch {
        // Fall through - see the best-effort note above.
      }
      if (freshlyHiredAgentId !== undefined) {
        await markOnboardedAndNavigate(`/agent/${freshlyHiredAgentId}`);
        return;
      }
      if (!isMountedRef.current) return;
      // See wizardUserIdRef's and currentUserIdRef's comments - never hire
      // an agent under a swapped-in identity's session on this identity's
      // behalf.
      if (currentUserIdRef.current !== wizardUserIdRef.current) {
        launchingRef.current = false;
        setLaunching(false);
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
      // Not `if (!isMountedRef.current) return;` here (unlike the guard
      // before hireAgentFromTemplate above, which is fine to skip attempting
      // entirely): self-review found that pre-checking mount state here
      // skipped markOnboardedAndNavigate's own save attempt too, not just
      // its navigation - the same "agent created but onboarded never even
      // attempted" bug this whole splitting was meant to fix, just
      // reintroduced by an unmount-mid-flight (e.g. a Back press) racing
      // hireAgentFromTemplate. markOnboardedAndNavigate's own isMountedRef
      // check already correctly gates only the navigation half.
      await markOnboardedAndNavigate(`/task/${result.taskId}`);
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
          {/* Array.from, not .slice(0, 1): a PR review finding caught that
              .slice indexes by UTF-16 code unit, so a name starting with a
              non-BMP character (e.g. an emoji) would split its surrogate
              pair and render as a broken glyph. Array.from iterates by
              Unicode code point instead. */}
          <span>{(Array.from(firstName)[0] ?? "").toUpperCase()}</span>
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
              aria-current={isNow ? "step" : undefined}
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
              <h1 ref={stepHeadingRef} tabIndex={-1}>
                {t("onboarding.welcome.titlePrefix", { appName: branding.appName })}
                <br />
                <em>{firstName}</em>
                {t("onboarding.welcome.titleSuffix")}
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
              <h1 ref={stepHeadingRef} tabIndex={-1}>
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
                    disabled={launching}
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
                    // Disabled while a save is in flight (not just the exit
                    // buttons): buildPreferencesPayload() snapshots this
                    // value before the PATCH is sent, so an edit made during
                    // that window would silently never be saved, then get
                    // discarded entirely once the page navigates away.
                    disabled={launching}
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
              <h1 ref={stepHeadingRef} tabIndex={-1}>
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
                      disabled={launching}
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
              <h1 ref={stepHeadingRef} tabIndex={-1}>{t("onboarding.team.title")}</h1>
              <p className="ob-sub">
                {/* No goals.length === 0 branch - a self-review finding
                    confirmed it's unreachable: Continue into this step
                    requires isGoalsValid (goals.length > 0), and the goals
                    step's only other action ("Not sure yet") exits the
                    wizard entirely via persistAndLeave rather than
                    continuing here. */}
                {/* Gated on !templatesLoading like the grid below it: while
                    still loading, matchedRecommended is always [] (nothing
                    has resolved yet), which would otherwise claim every
                    goal has an "other match waiting" before we actually
                    know that.

                    Counts against matchedRecommended, NOT validRecommended
                    - a PR review finding caught that validRecommended can
                    be entirely fallback filler cards (goalId: null) when
                    nothing actually matched any selected goal, which would
                    make this math claim real matches were found (a small
                    "extra" count) when the true number of goal matches was
                    zero. */}
                {`${t("onboarding.team.subtitleBase")}${
                  !templatesLoading && goals.length - matchedRecommended.length > 0
                    ? t(
                        goals.length - matchedRecommended.length === 1
                          ? "onboarding.team.subtitleExtraOne"
                          : "onboarding.team.subtitleExtraMany",
                        { count: goals.length - matchedRecommended.length }
                      )
                    : t("onboarding.team.subtitleEnd")
                }`}
              </p>
              {templatesLoading ? (
                <div
                  role="status"
                  style={{ marginTop: 40, display: "flex", justifyContent: "center" }}
                >
                  <div
                    aria-hidden="true"
                    data-testid="onboarding-team-loading"
                    // motion-reduce:animate-none - a PR review finding caught that this
                    // spinner isn't covered by onboarding.css's prefers-reduced-motion
                    // rule (which only targets the aurora/orb/step selectors), so it kept
                    // spinning under reduced motion. Tailwind's built-in variant handles
                    // it without adding a dedicated class to that CSS rule.
                    className="h-8 w-8 animate-spin motion-reduce:animate-none rounded-full border-2 border-[hsl(var(--border))] border-t-[hsl(var(--primary))]"
                  />
                  <span className="sr-only">{t("common.loading")}</span>
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
                        disabled={launching}
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
                          decorative
                          style={{
                            boxShadow: `0 0 0 1px hsl(var(--border)), 0 0 0 5px hsl(var(--card)), 0 0 0 6px ${ring}`,
                          }}
                        />
                        <div className="ob-tm-name">{template.persona.name}</div>
                        <div className="ob-tm-role">{template.persona.role}</div>
                        <span className="ob-tm-cat">
                          <i style={{ background: dot }} />
                          {categoryLabel(t, template.category)}
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
              <h1 ref={stepHeadingRef} tabIndex={-1}>
                {t("onboarding.voice.titleLine1")}
                <br />
                {t("onboarding.voice.titleLine2")}
              </h1>
              <p className="ob-sub">{t("onboarding.voice.subtitle")}</p>
              <div className="ob-chips" style={{ marginTop: 34 }}>
                {ONBOARDING_VOICES.map((option) => (
                  <Chip key={option.id} selected={voice === option.id} disabled={launching} onClick={() => setVoice(option.id)}>
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
                  <h1 ref={stepHeadingRef} tabIndex={-1}>{t("onboarding.done.title")}</h1>
                  <p className="ob-sub">{t("onboarding.done.subtitle", { name: persona.name })}</p>

                  <div className="ob-sum">
                    <div className="ob-sum-hd">
                      <PersonaAvatar
                        persona={persona}
                        sizeClassName="h-[62px] w-[62px]"
                        textClassName="text-2xl"
                        className="ob-tm-av"
                        decorative
                        style={{
                          boxShadow: `0 0 0 1px hsl(var(--border)), 0 0 0 4px hsl(var(--card)), 0 0 0 5px ${ring}`,
                        }}
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
                          {t("onboarding.done.workingInPrefix")}<b>{workLabel}</b>
                        </span>
                      </div>
                      <div className="ob-sum-li">
                        <Check className="mt-0.5 h-[18px] w-[18px]" style={{ color: "#22A05B" }} />
                        <span>
                          {t("onboarding.done.briefedPrefix")}<b>{jobCount}</b>
                          {t(jobCount === 1 ? "onboarding.done.briefedSuffixOne" : "onboarding.done.briefedSuffixOther")}
                        </span>
                      </div>
                      <div className="ob-sum-li">
                        <Check className="mt-0.5 h-[18px] w-[18px]" style={{ color: "#22A05B" }} />
                        <span>
                          {t("onboarding.done.writingInPrefix")}<b>{t(voiceOption.nameKey).toLowerCase()}</b>
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
                            {t("onboarding.done.willConnectPrefix")}
                            <b>
                              {joinWithAnd(
                                appNames,
                                t("onboarding.done.willConnectAnd"),
                                t("onboarding.done.willConnectSeparator")
                              )}
                            </b>
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
