"use client"

import React, { Suspense } from "react"
import { TaskVoiceProvider } from "@/contexts/task-voice-context"

export default function TaskLayout({ children }: { children: React.ReactNode }) {
  return <Suspense><TaskVoiceProvider>{children}</TaskVoiceProvider></Suspense>
}
