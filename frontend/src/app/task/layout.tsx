"use client"

import React, { Suspense } from "react"
import { TaskVoiceProvider } from "@/contexts/task-voice-context"
import { VoiceCallStoreProvider } from "@/contexts/voice-call-store"

export default function TaskLayout({ children }: { children: React.ReactNode }) {
  return <Suspense><VoiceCallStoreProvider><TaskVoiceProvider>{children}</TaskVoiceProvider></VoiceCallStoreProvider></Suspense>
}
