"use client"

import { usePublishVoiceCall } from "@/contexts/voice-call-store"
import { useWidgetVoiceCall, type WidgetVoiceCallOptions } from "@/hooks/use-widget-voice-call"

/**
 * Runs an embedded chat's voice call and hands what it shows to the composer,
 * like VoiceConversation does for the signed-in call. It draws nothing.
 */
export function WidgetVoiceConversation(props: WidgetVoiceCallOptions) {
  usePublishVoiceCall(useWidgetVoiceCall(props))
  return null
}
