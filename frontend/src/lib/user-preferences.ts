"use client";

import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";
import type { OnboardingVoiceId } from "@/lib/onboarding-data";

/** Mirrors UpdatePreferencesRequest (src/xagent/web/api/auth.py) - onboarded/
 * department/industry/voice/goals, all optional so a caller can PATCH just
 * the fields it has. `voice` must be one of VALID_USER_VOICES or the
 * backend 422s. */
export interface UserPreferences {
  onboarded?: boolean;
  department?: string;
  // null is this endpoint's merge-PATCH "clear this field" signal (see
  // buildPreferencesPayload in page.tsx) - not the same as omitting the key.
  industry?: string | null;
  voice?: OnboardingVoiceId;
  goals?: string[];
}

/** Fetches the current user's stored preferences via GET /api/auth/me -
 * deliberately not threaded through the cached auth-context session
 * (AuthCacheUser carries only id/username/email/is_admin), since only the
 * onboarding-redirect check and the onboarding page itself need this, not
 * every authenticated page render.
 *
 * Returns `null` (not `{}`) when the fetch fails or the response isn't ok -
 * that's "unknown," not "confirmed not onboarded," and callers must not
 * treat the two the same. Conflating them used to mean a transient error
 * (deploy, brief backend blip) redirected an already-onboarded, active user
 * into the onboarding wizard - the exact "forced redirect loop" this was
 * meant to prevent, just triggered by an error instead of a stale read. */
export async function fetchUserPreferences(): Promise<UserPreferences | null> {
  try {
    const response = await apiRequest(`${getApiUrl()}/api/auth/me`);
    if (!response.ok) return null;
    const body = await response.json();
    const preferences = body?.user?.preferences;
    return preferences && typeof preferences === "object" ? preferences : {};
  } catch {
    return null;
  }
}

/** PATCH /api/auth/me/preferences - merges the given fields into the
 * stored preferences server-side (a partial update, not a replace), so
 * callers only need to send what they actually collected.
 *
 * `retryable` distinguishes a transient failure (network error, 5xx) from a
 * permanent one (4xx - the backend rejected this exact payload, e.g. a
 * malformed/oversized field): a caller that gives up and proceeds anyway
 * after repeated failures (see MAX_SAVE_FAILURES_BEFORE_ESCAPE in page.tsx)
 * must not treat the two the same when what follows is irreversible -
 * retrying an identical rejected payload will only ever 4xx again. */
export async function updateUserPreferences(
  updates: UserPreferences
): Promise<{ ok: boolean; retryable: boolean }> {
  try {
    const response = await apiRequest(`${getApiUrl()}/api/auth/me/preferences`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates),
    });
    return { ok: response.ok, retryable: response.ok || response.status >= 500 };
  } catch {
    return { ok: false, retryable: true };
  }
}

const SAVE_ESCAPE_KEY = "xagent-onboarding-save-escape";
// Must be consumed by the very next onboarding check after being set - see
// the caller in auth-guard.tsx for why that's now guaranteed structurally,
// not just by convention. This TTL is defense-in-depth on top of that, not
// the primary mechanism: a stale/leftover value (a caller that changes in
// the future and stops consuming it promptly, a corrupted value) expires
// instead of silently suppressing some unrelated future redirect forever.
const SAVE_ESCAPE_TTL_MS = 30_000;

/** Called by the onboarding page right before it gives up and navigates away
 * despite a failed preferences save (see MAX_SAVE_FAILURES_BEFORE_ESCAPE in
 * page.tsx). Without this, AuthGuard's own onboarding-redirect check on the
 * destination route would see the still-`onboarded:false` preferences and
 * immediately bounce the user right back into onboarding - defeating the
 * whole point of the escape hatch (in the worst case, an infinite bounce
 * loop between the destination and /onboarding, each one resetting the
 * failure counter on remount). sessionStorage (not a ref/module variable)
 * because the check that needs to see this runs in AuthGuard, a different
 * component the onboarding page has no direct handle on.
 *
 * Bound to `userId`: a PR review finding caught that a bare timestamp is
 * identity-agnostic, so a cross-tab identity swap (the same vulnerability
 * class the checkedOnboardingUserIdRef fix in auth-guard.tsx closed for the
 * normal check) could let a DIFFERENT user who logs in in this tab within
 * the TTL window consume user A's leftover escape and skip their own
 * mandatory onboarding check entirely. No-ops without a userId (rather than
 * falling back to an unscoped flag) - no worse than the pre-existing
 * behavior of not marking anything at all. */
export function markOnboardingSaveEscaped(userId: string | undefined): void {
  if (!userId) return;
  try {
    window.sessionStorage.setItem(SAVE_ESCAPE_KEY, JSON.stringify({ userId, setAt: Date.now() }));
  } catch {
    // Storage unavailable (private mode, disabled) - AuthGuard just won't
    // see the flag and will redirect as it did before this existed; no
    // worse than the pre-existing behavior.
  }
}

/** Reads and clears the flag set by markOnboardingSaveEscaped() - removed
 * on a resolved read (one-shot; a stale, corrupt, or mismatched value must
 * not linger either), and only honored within SAVE_ESCAPE_TTL_MS of being
 * set AND for the SAME user id that set it - see markOnboardingSaveEscaped's
 * comment on why an identity-agnostic flag is unsafe. `userId` is the
 * CURRENTLY authenticated identity at the moment of the check (may be null
 * while auth is still resolving).
 *
 * Deliberately NOT removed when `userId` is null: self-review of this same
 * identity-binding fix found that unconditionally consuming (removing) on
 * every call - including calls made before auth has resolved on a fresh
 * page load, which AuthGuard's effect does unconditionally - deleted a
 * live flag before its identity could ever actually be checked. The very
 * next call, once auth resolves with the real user id, would then find
 * nothing and silently fail to honor an escape that was never actually
 * matched against anyone - the exact bounce-loop this mechanism exists to
 * prevent. Leaving an unresolvable flag in place costs nothing: it's still
 * fresh (TTL not yet reached) and gets a real, resolved check - matched or
 * not - the next time this runs with a known identity. */
export function consumeOnboardingSaveEscapeFlag(userId: string | null): boolean {
  try {
    const raw = window.sessionStorage.getItem(SAVE_ESCAPE_KEY);
    if (!raw) return false;
    if (userId === null) return false;
    window.sessionStorage.removeItem(SAVE_ESCAPE_KEY);
    const parsed = JSON.parse(raw) as { userId?: unknown; setAt?: unknown };
    if (typeof parsed?.userId !== "string" || typeof parsed?.setAt !== "number") return false;
    if (parsed.userId !== userId) return false;
    return Number.isFinite(parsed.setAt) && Date.now() - parsed.setAt < SAVE_ESCAPE_TTL_MS;
  } catch {
    return false;
  }
}
