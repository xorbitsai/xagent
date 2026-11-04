"use client";

import "./onboarding.css";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowRight, Check, ChevronDown, Plus } from "lucide-react";
import { toast } from "sonner";
import { useAuth } from "@/contexts/auth-context";
import { useI18n } from "@/contexts/i18n-context";
import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";
import { cn } from "@/lib/utils";
import { getBrandingFromEnv } from "@/lib/branding";
import { markOnboardingSaveEscaped, updateUserPreferences, type UserPreferences } from "@/lib/user-preferences";
import { readAuthSessionSnapshot } from "@/lib/auth-cache";
import { hireAgentFromTemplate } from "@/lib/hire-agent";
import { PersonaAvatar } from "@/components/templates/persona-avatar";
import { categoryLabel } from "@/lib/template-categories";
import { getAgentTaskHref } from "@/lib/agent-ui-access";
import {
  ONBOARDING_DEFAULT_VOICE,
  ONBOARDING_FALLBACK_TEMPLATE_IDS,
  ONBOARDING_GOALS,
  ONBOARDING_VOICES,
  inferWorkFromGoals,
  joinWithAnd,
  recommendedTemplates,
  type OnboardingVoiceId,
} from "@/lib/onboarding-data";
import type { Template, TemplateDetail } from "@/types/template";
import type { TranslationKey } from "@/i18n/translations";

const branding = getBrandingFromEnv();

type StepId = "goals" | "team";
const STEP_ORDER: StepId[] = ["goals", "team"];
const GROUP_KEYS: TranslationKey[] = ["onboarding.rail.need", "onboarding.rail.team"];

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
  const wizardUserIdRef = useRef(readAuthSessionSnapshot().userId);
  // Reads storage directly, live, at every guard checkpoint below - NOT the
  // `user` variable, and NOT a ref kept fresh by a useEffect. Two separate
  // self-review findings ruled out both of those:
  //
  // - `user` comes from a plain `const` destructure, so it's frozen to
  //   whatever AuthProvider returned at the specific render that created
  //   the CURRENTLY RUNNING closure (handleLaunch/trySavePreferences/
  //   persistAndLeave are all recreated fresh every render, but an
  //   already-invoked call keeps running with the closure it started
  //   with). AuthProvider replaces `user` with a brand new object via
  //   React state (setProjection in auth-context.tsx), not an in-place
  //   mutation, so an identity swap that happens WHILE one of those
  //   functions is already awaiting a save/hire call is invisible to a
  //   `user?.id` read inside it.
  // - A ref kept fresh via `useEffect(() => { ref.current = user?.id },
  //   [user?.id])` fixes that, but only while this component stays
  //   mounted - the effect simply stops running once it unmounts, so the
  //   ref freezes at whatever it last held. handleLaunch's async chain
  //   keeps executing after an unmount (only navigation/state-setting are
  //   gated on isMountedRef, not the save/hire calls themselves - see
  //   their own comments for why), so an identity swap that happens
  //   AFTER an unmount mid-flight was invisible to that ref too - an
  //   external review round caught this exact gap.
  //
  // readAuthSessionSnapshot() (the same function apiRequest itself calls
  // to decide which credentials go out on the wire - see api-wrapper.ts)
  // reads live storage synchronously with no dependency on React
  // rendering or mount state at all, so it can't go stale either way.
  const currentSessionUserId = () => readAuthSessionSnapshot().userId;

  const [stepIndex, setStepIndex] = useState(0);
  const step = STEP_ORDER[stepIndex];
  const [goals, setGoals] = useState<string[]>([]);
  const [agentTemplateId, setAgentTemplateId] = useState("");
  const [voice, setVoice] = useState<OnboardingVoiceId>(ONBOARDING_DEFAULT_VOICE);

  const [voiceOpen, setVoiceOpen] = useState(false);
  const [templatesError, setTemplatesError] = useState(false);
  const [templatesAttempt, setTemplatesAttempt] = useState(0);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [templatesLoading, setTemplatesLoading] = useState(true);
  const [launching, setLaunching] = useState(false);
  const launchingRef = useRef(false);
  const isMountedRef = useRef(true);
  // Consecutive save failures, keyed by "destination" - see the escape
  // hatches in persistAndLeave/handleLaunch below. All Skip buttons share
  // the "/templates" bucket because they express the same catalogue intent.
  // Hiring has a separate bucket, so a failed Skip never spends its retry.
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
  // The exact payload (serialized) that was last attempted and failed for
  // this key - a PR review finding caught that the count/permanent-flag
  // above are keyed only by destination, not by payload content, so a
  // failure streak survives the user going back and editing their answers:
  // failing once, editing goals/voice/etc., then failing or succeeding
  // again would wrongly inherit (or wrongly escalate/block on) a PREVIOUS,
  // different payload's failure history. Compared against the current
  // payload on every failure below; a mismatch starts a fresh streak for
  // the new payload instead of continuing the old one.
  const saveFailurePayloadByDestRef = useRef<Record<string, string>>({});

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
    setTemplatesError(false);
    (async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/templates/?lang=${locale}`);
        if (cancelled) return;
        if (!response.ok) throw new Error("Failed to load teammates");
        const data = await response.json();
        if (!Array.isArray(data)) throw new Error("Invalid teammate catalog");
        if (!cancelled) setTemplates(data);
      } catch {
        if (!cancelled) setTemplatesError(true);
      } finally {
        if (!cancelled) setTemplatesLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [locale, templatesAttempt]);

  const templateById = useMemo(() => {
    const map = new Map<string, Template>();
    for (const template of templates) map.set(template.id, template);
    return map;
  }, [templates]);

  const recommended = useMemo(() => recommendedTemplates(goals), [goals]);

  // The static goal->template catalog can point at a template id that never
  // loaded or has no persona. Filter before rendering and selecting cards;
  // isTeamValid separately blocks launching during loading/error states.
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

  // How many selected goals have no card actually shown right now, for the
  // team step's "N other matches waiting" subtitle - a PR review finding
  // caught that counting against matchedRecommended (uncapped) instead of
  // validRecommended (the actually-rendered, capped-at-3 list) had swapped
  // one bug for another: it fixed the fallback-inflation case below, but
  // then undercounted the OPPOSITE case - 4+ goals that all genuinely
  // matched a template report zero "extra" even though the 4th+ real match
  // is hidden by the display cap, because matchedRecommended.length still
  // equals goals.length there.
  //
  // Only goals.length - validRecommended.length is correct when there's at
  // least one real match (matchedRecommended.length > 0): validRecommended
  // is exactly matchedRecommended sliced to 3 in that branch, so this
  // counts both a fully-unmatched goal AND a real match hidden by the cap.
  // But when NOTHING matched (matchedRecommended.length === 0),
  // validRecommended is 3 fallback filler cards unrelated to any goal, and
  // goals.length - validRecommended.length would wrongly subtract them -
  // every selected goal is still unmatched in that case, full stop.
  const omittedGoalCount =
    matchedRecommended.length > 0 ? goals.length - validRecommended.length : goals.length;

  // Keep the selected agent valid as the recommendation list changes -
  // default to the first recommendation whenever the current pick falls
  // out of the list (including the very first render).
  useEffect(() => {
    if (!validRecommended.some((r) => r.templateId === agentTemplateId)) {
      setAgentTemplateId(validRecommended[0]?.templateId ?? "");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [validRecommended]);

  const work = useMemo(() => inferWorkFromGoals(goals), [goals]);
  const selectedTemplate = templateById.get(agentTemplateId);
  const selectedVoice = ONBOARDING_VOICES.find((option) => option.id === voice)!;
  const appNames = [...new Set((selectedTemplate?.connections ?? []).map((connection) => connection.name.trim()).filter(Boolean))];

  const firstName = user?.username?.trim() || t("onboarding.welcome.fallbackName");

  const isGoalsValid = goals.length > 0;
  const isTeamValid = !templatesLoading && !templatesError && !!selectedTemplate?.persona;

  // Persist the visible default once the voice row has been reached, even
  // when the user goes back to the first step before leaving.
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
    if (STEP_ORDER[clamped] === "team") hasReachedVoiceRef.current = true;
  };
  const next = () => goTo(stepIndex + 1);
  const back = () => goTo(stepIndex - 1);

  // Mirrors hasReachedVoiceRef's latch, for the same reason: a PR review
  // finding caught that goals.length === 0 omits `goals` entirely from the
  // payload below, and the merge PATCH treats an omitted key as "leave
  // whatever's already stored" - not "clear it." That's correct for a
  // user who never touched goals at all (nothing to clear), but wrong for
  // one who selected goals, saved (goals now nonempty server-side), came
  // back via the rail, and deselected every chip before exiting - their
  // explicit clear would silently do nothing, leaving a stale nonempty
  // list stored under an account now marked onboarded. Latched (not a
  // live length check) so it stays true even after clearing back to zero.
  const hasToggledGoalsRef = useRef(false);
  const toggleGoal = (id: string) => {
    hasToggledGoalsRef.current = true;
    setGoals((prev) => (prev.includes(id) ? prev.filter((g) => g !== id) : [...prev, id]));
  };

  // `includeOnboarded: false` is handleLaunch's own case - see the comment
  // where it calls this. Every other caller wants the default (true).
  const buildPreferencesPayload = ({ includeOnboarded = true }: { includeOnboarded?: boolean } = {}): UserPreferences => {
    return {
      ...(includeOnboarded ? { onboarded: true } : {}),
      ...(hasToggledGoalsRef.current ? { department: work || null, industry: null } : {}),
      ...(goals.length ? { goals } : hasToggledGoalsRef.current ? { goals: [] } : {}),
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
    // See wizardUserIdRef's and currentSessionUserId's comments - never send
    // this identity's answers under a swapped-in session. Not counted as a
    // failure of THIS payload (it was never actually sent), so it doesn't
    // feed the escape counter.
    if (currentSessionUserId() !== wizardUserIdRef.current) {
      if (isMountedRef.current) toast.error(t("onboarding.done.saveFailed"));
      return "retry_in_place";
    }
    const payload = buildPreferencesPayload({ includeOnboarded });
    const saved = await updateUserPreferences(payload);
    if (saved.ok) {
      saveFailureCountByDestRef.current[failureKey] = 0;
      saveFailureHasPermanentByDestRef.current[failureKey] = false;
      delete saveFailurePayloadByDestRef.current[failureKey];
      return "saved";
    }
    // A different payload than the one that failed last time for this key
    // starts a fresh streak - see saveFailurePayloadByDestRef's comment.
    const payloadKey = JSON.stringify(payload);
    if (saveFailurePayloadByDestRef.current[failureKey] !== payloadKey) {
      saveFailureCountByDestRef.current[failureKey] = 0;
      saveFailureHasPermanentByDestRef.current[failureKey] = false;
      saveFailurePayloadByDestRef.current[failureKey] = payloadKey;
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
    // See wizardUserIdRef's and currentSessionUserId's comments - never
    // complete onboarding, or write this identity's collected answers,
    // under a swapped-in session. No escape marker here: that flag is for
    // a save that genuinely failed, not for an attempt this identity never
    // actually got to make.
    if (currentSessionUserId() !== wizardUserIdRef.current) {
      if (isMountedRef.current) {
        launchingRef.current = false;
        setLaunching(false);
      }
      return;
    }
    const onboardedSave = await updateUserPreferences(buildPreferencesPayload());
    // See wizardUserIdRef's and currentSessionUserId's comments - re-checked
    // AFTER the await, not just before it. A PR review finding caught that
    // this continuation used to navigate unconditionally once the PATCH
    // above resolved, even if identity had swapped DURING that PATCH's own
    // round trip. The save itself was sent under the original identity
    // (confirmed by the guard above), so it's left in place, but
    // navigating now would route the NEW identity to a destination that
    // was only ever meant for the original one. Nothing to mark as
    // escaped here either - the original identity isn't actually leaving
    // via this tab anymore, so there's no exit for the flag to cover.
    if (currentSessionUserId() !== wizardUserIdRef.current) {
      if (isMountedRef.current) {
        launchingRef.current = false;
        setLaunching(false);
      }
      return;
    }
    if (isMountedRef.current) {
      // wizardUserIdRef.current, not user?.id/currentSessionUserId() - this flag
      // is inherently about the ORIGINAL wizard identity's save outcome
      // (both guards above already confirmed identity hadn't swapped when
      // this PATCH was sent or when it resolved), not whoever happens to
      // be live by the time this exact line runs.
      if (!onboardedSave.ok) markOnboardingSaveEscaped(wizardUserIdRef.current);
      router.replace(destination);
    }
  };

  // Every Skip path saves the choices made so far before opening the
  // catalogue. Skipping after selecting goals must not discard them.
  const persistAndLeave = async (destination: string) => {
    // Deliberately does NOT re-check currentSessionUserId/wizardUserIdRef after
    // the trySavePreferences await below, unlike markOnboardedAndNavigate's
    // post-await check - the risk that guards against (routing a
    // swapped-in identity to a destination that only makes sense for the
    // original one, e.g. /task?agent={id} or /task/{id}) doesn't apply here:
    // every caller of persistAndLeave passes a fixed, identity-agnostic
    // route (`/templates`, never a specific resource id), so a
    // swapped-in identity landing there just lands on their own generic
    // page - nothing dead or unauthorized to reach. The entry guard inside
    // trySavePreferences (which persistAndLeave still goes through) is
    // what actually matters here: it stops this identity's answers from
    // being sent under someone else's session.
    //
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
        await markOnboardedAndNavigate(getAgentTaskHref(selected.hired_agent_id));
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
        await markOnboardedAndNavigate(getAgentTaskHref(freshlyHiredAgentId));
        return;
      }
      if (!isMountedRef.current) return;
      // See wizardUserIdRef's and currentSessionUserId's comments - never hire
      // an agent under a swapped-in identity's session on this identity's
      // behalf. No dedicated post-unmount test for this specific checkpoint
      // (unlike markOnboardedAndNavigate's entry check) - the isMountedRef
      // check directly above it already returns before this is ever
      // reached once genuinely unmounted, so this identity check's own
      // unmount-robustness has no separately observable effect to test
      // here; abortIfIdentityChanged below (reached only if this passes)
      // is what actually needs to survive an unmount mid-hire, since
      // hireAgentFromTemplate's own calls are NOT gated by isMountedRef.
      if (currentSessionUserId() !== wizardUserIdRef.current) {
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
        welcome: {
          description: selected.description,
          openingMessage: t("onboarding.firstChat.greeting", { name: selected.persona.name, role: selected.persona.role }),
          prompt: t("onboarding.firstChat.prompt"),
        },
        // See wizardUserIdRef's and currentSessionUserId's comments - the guard
        // right before this call only proves identity hadn't swapped BEFORE
        // hireAgentFromTemplate was invoked. That function itself makes 2
        // independent network calls (resolve, then task/create), so without
        // this, a swap landing in the gap between them would still let
        // task/create run under a different identity than the one that just
        // resolved the agent.
        abortIfIdentityChanged: () => currentSessionUserId() !== wizardUserIdRef.current,
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
        <b className="ob-a1" /><b className="ob-a2" /><b className="ob-a3" />
      </div>
      <header className="ob-top">
        <button type="button" disabled={launching} aria-busy={launching}
          onClick={() => persistAndLeave("/templates")} className="ob-exit">
          {launching ? t("onboarding.saving") : t("onboarding.skip")}
        </button>
        {/* eslint-disable-next-line @next/next/no-img-element -- small brand mark */}
        <img className="ob-logo" src={branding.logoPath} alt={branding.appName} />
        <div className="ob-acct">
          <span>{(Array.from(firstName)[0] ?? "").toUpperCase()}</span><b>{firstName}</b>
        </div>
      </header>
      <nav aria-label={t("onboarding.rail.label")} className="ob-rail">
        {GROUP_KEYS.map((key, index) => (
          <button key={key} type="button" aria-current={index === stepIndex ? "step" : undefined}
            disabled={launching || index >= stepIndex} onClick={() => goTo(index)}
            className={cn("ob-rl", index === stepIndex && "is-now", index < stepIndex && "is-done can-go")}>
            <i>{index < stepIndex && <Check className="h-3 w-3" />}</i><span>{t(key)}</span>
          </button>
        ))}
      </nav>
      <main className="ob-body">
        <div key={step} className="ob-step is-in">
          {step === "goals" ? (
            <>
              <Orb />
              <h1 ref={stepHeadingRef} tabIndex={-1}>
                {t("onboarding.need.greeting", { name: firstName.split(/\s+/)[0] })}<br />
                {t("onboarding.need.title")}
              </h1>
              <p className="ob-sub">{t("onboarding.need.subtitle")}</p>
              <div className="ob-chips ob-goals">
                {ONBOARDING_GOALS.map((goal) => {
                  const selected = goals.includes(goal.id);
                  return (
                    <Chip key={goal.id} selected={selected} disabled={launching} onClick={() => toggleGoal(goal.id)}
                      icon={selected ? <Check className="ob-ic h-3.5 w-3.5" /> : <Plus className="ob-ic h-3.5 w-3.5" />}>
                      {t(goal.labelKey)}
                    </Chip>
                  );
                })}
              </div>
              <div className="ob-cta">
                <button type="button" disabled={!isGoalsValid || launching} onClick={next} className="ob-btn-next">
                  {t("onboarding.continue")}
                </button>
              </div>
              <button type="button" disabled={launching} aria-busy={launching}
                onClick={() => persistAndLeave("/templates")} className="ob-skip">
                {launching ? t("onboarding.saving") : t("onboarding.need.skip")}
              </button>
            </>
          ) : (
            <>
              <Orb small />
              <h1 ref={stepHeadingRef} tabIndex={-1}>{t("onboarding.team.title")}</h1>
              <p className="ob-sub">
                {t("onboarding.team.subtitleBase")}
                {!templatesLoading && !templatesError && omittedGoalCount > 0
                  ? t(omittedGoalCount === 1 ? "onboarding.team.subtitleExtraOne" : "onboarding.team.subtitleExtraMany", { count: omittedGoalCount })
                  : t("onboarding.team.subtitleEnd")}
              </p>
              {templatesLoading ? (
                <div role="status" className="ob-catalog-status">
                  <div aria-hidden="true" data-testid="onboarding-team-loading"
                    className="h-8 w-8 animate-spin motion-reduce:animate-none rounded-full border-2 border-border border-t-primary" />
                  <span className="sr-only">{t("common.loading")}</span>
                </div>
              ) : templatesError ? (
                <div role="alert" className="ob-catalog-status">
                  <p>{t("onboarding.team.loadFailed")}</p>
                  <button type="button" disabled={launching} className="ob-btn-back"
                    onClick={() => setTemplatesAttempt((value) => value + 1)}>{t("common.retry")}</button>
                </div>
              ) : validRecommended.length === 0 ? (
                <p className="ob-catalog-status">{t("onboarding.team.empty")}</p>
              ) : (
                <div className={cn("ob-team", validRecommended.length === 1 ? "one" : validRecommended.length === 2 ? "two" : "")}>
                  {validRecommended.map((rec) => {
                    const template = templateById.get(rec.templateId);
                    if (!template?.persona) return null;
                    const goal = ONBOARDING_GOALS.find((g) => g.id === rec.goalId);
                    const selected = agentTemplateId === template.id;
                    const { ring, dot } = CATEGORY_STYLE[template.category] ?? DEFAULT_CATEGORY_STYLE;
                    return (
                      <button key={template.id} type="button" aria-pressed={selected} disabled={launching}
                        onClick={() => setAgentTemplateId(template.id)} className="ob-tm">
                        {selected && <span className="ob-tm-tick"><Check className="h-3 w-3" /></span>}
                        <PersonaAvatar persona={template.persona} sizeClassName="h-[84px] w-[84px]" textClassName="text-[33px]"
                          className="ob-tm-av" decorative
                          style={{ boxShadow: `0 0 0 1px hsl(var(--border)), 0 0 0 5px hsl(var(--card)), 0 0 0 6px ${ring}` }} />
                        <span className="ob-tm-name">{template.persona.name}</span>
                        <span className="ob-tm-role">{template.persona.role}</span>
                        <span className="ob-tm-cat"><i style={{ background: dot }} />{categoryLabel(t, template.category)}</span>
                        <span className="ob-tm-desc">{template.description}</span>
                        {goal && <span className="ob-tm-why"><i />{t(goal.labelKey)}</span>}
                      </button>
                    );
                  })}
                </div>
              )}
              <div className="ob-more">
                <button type="button" className="ob-more-toggle" disabled={launching} aria-expanded={voiceOpen}
                  aria-controls="onboarding-voice" onClick={() => setVoiceOpen((open) => !open)}>
                  {t("onboarding.team.voice")}: <b>{t(selectedVoice.nameKey)}</b>
                  <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", voiceOpen && "rotate-180")} />
                </button>
                {voiceOpen && (
                  <div id="onboarding-voice" className="ob-more-body">
                    <div className="ob-chips">
                      {ONBOARDING_VOICES.map((option) => (
                        <Chip key={option.id} selected={voice === option.id} disabled={launching} onClick={() => setVoice(option.id)}>
                          {t(option.nameKey)}
                        </Chip>
                      ))}
                    </div>
                    <p className="ob-say"><b>{t("onboarding.team.voicePreview", { voice: t(selectedVoice.nameKey) })}</b><i>{t(selectedVoice.sayKey)}</i></p>
                  </div>
                )}
              </div>
              {isTeamValid && selectedTemplate?.persona && (
                <p className="ob-next">
                  {appNames.length
                    ? t("onboarding.team.connectNext", { name: selectedTemplate.persona.name, apps: joinWithAnd(appNames, t("onboarding.done.willConnectAnd"), t("onboarding.done.willConnectSeparator")) })
                    : t("onboarding.team.noAccounts", { name: selectedTemplate.persona.name })}
                  <span>{t("onboarding.team.reviewNote")}</span>
                </p>
              )}
              <div className="ob-cta">
                <button type="button" disabled={launching} onClick={back} className="ob-btn-back">{t("common.back")}</button>
                <button type="button" disabled={!isTeamValid || launching} aria-busy={launching}
                  onClick={handleLaunch} className="ob-btn-next">
                  {launching ? t("templates.marketplace.hiring")
                    : t("onboarding.done.start", { name: selectedTemplate?.persona?.name || t("onboarding.team.teammate") })}
                  {!launching && <ArrowRight className="h-4 w-4" />}
                </button>
              </div>
              <button type="button" disabled={launching} aria-busy={launching}
                onClick={() => persistAndLeave("/templates")} className="ob-skip">
                {launching ? t("onboarding.saving") : t("onboarding.done.skip")}
              </button>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
