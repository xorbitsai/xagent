"use client"

import { useEffect, useMemo, useState } from "react"
import { useAuth } from "@/contexts/auth-context"
import { fetchUserPreferences, type VoiceConversationPreferences } from "@/lib/user-preferences"

const defaults: VoiceConversationPreferences = {}

export function useVoicePreferences(enabled: boolean) {
  const auth = useAuth()
  const signedIn = Boolean(auth.token)
  const owner = useMemo(() => ({}), [enabled, auth.user?.id, auth.session?.sessionId, signedIn])
  const [state, setState] = useState<{ owner: object | null; preferences: VoiceConversationPreferences; error: boolean }>({ owner: null, preferences: defaults, error: false })
  useEffect(() => {
    if (!enabled || !signedIn) return
    let active = true
    void fetchUserPreferences().then(result => {
      if (active) setState({ owner, preferences: result?.voice_conversation ?? defaults, error: result === null })
    })
    return () => { active = false }
  }, [enabled, signedIn, owner])
  const current = state.owner === owner
  return {
    preferences: current ? state.preferences : defaults,
    loading: enabled && signedIn && !current,
    error: current && state.error,
  }
}
