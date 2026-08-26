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
