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
 * every authenticated page render. Returns `{}` (never onboarded) on any
 * failure - a transient error here must not force a fresh login into a
 * redirect loop. */
export async function fetchUserPreferences(): Promise<UserPreferences> {
  try {
    const response = await apiRequest(`${getApiUrl()}/api/auth/me`);
    if (!response.ok) return {};
    const body = await response.json();
    const preferences = body?.user?.preferences;
    return preferences && typeof preferences === "object" ? preferences : {};
  } catch {
    return {};
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
