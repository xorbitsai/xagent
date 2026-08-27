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
  industry?: string;
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
 * callers only need to send what they actually collected. */
export async function updateUserPreferences(
  updates: UserPreferences
): Promise<{ ok: boolean }> {
  try {
    const response = await apiRequest(`${getApiUrl()}/api/auth/me/preferences`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates),
    });
    return { ok: response.ok };
  } catch {
    return { ok: false };
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
 * component the onboarding page has no direct handle on. */
export function markOnboardingSaveEscaped(): void {
  try {
    window.sessionStorage.setItem(SAVE_ESCAPE_KEY, String(Date.now()));
  } catch {
    // Storage unavailable (private mode, disabled) - AuthGuard just won't
    // see the flag and will redirect as it did before this existed; no
    // worse than the pre-existing behavior.
  }
}

/** Reads and clears the flag set by markOnboardingSaveEscaped() - always
 * removed on read (one-shot; a stale or corrupt value must not linger
 * either), and only honored within SAVE_ESCAPE_TTL_MS of being set. */
export function consumeOnboardingSaveEscapeFlag(): boolean {
  try {
    const raw = window.sessionStorage.getItem(SAVE_ESCAPE_KEY);
    window.sessionStorage.removeItem(SAVE_ESCAPE_KEY);
    if (!raw) return false;
    const setAt = Number(raw);
    return Number.isFinite(setAt) && Date.now() - setAt < SAVE_ESCAPE_TTL_MS;
  } catch {
    return false;
  }
}
