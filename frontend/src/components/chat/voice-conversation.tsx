"use client"

import { usePublishVoiceCall } from "@/contexts/voice-call-store"
import { useVoiceCall, type VoiceCallOptions } from "@/hooks/use-voice-call"

/**
 * Runs a voice call and hands what it shows to the composer.
 *
 * It draws nothing itself: the call is the microphone bar and orb in the
 * message composer (see VoiceCallBar), not a dialog. It stays mounted where the
 * call's task and login are known and publishes the call's state for the
 * composer to render.
 */
export function VoiceConversation(props: VoiceCallOptions) {
  usePublishVoiceCall(useVoiceCall(props))
  return null
}
