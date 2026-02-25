import { notFound } from "next/navigation"
import React from "react"

export default function DebugLayout({
  children,
}: {
  children: React.ReactNode
}) {
  if (process.env.ENABLE_DEBUG_PAGE !== "true") {
    notFound()
  }
  return <>{children}</>
}
