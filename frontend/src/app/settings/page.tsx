"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"

const COMPUTER_SECTION_HASHES = new Set(["#browser-relay", "#desktop-relay"])

export default function SettingsPage() {
  const router = useRouter()

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const tab = params.get("tab") === "computer" || COMPUTER_SECTION_HASHES.has(window.location.hash)
      ? "computer"
      : params.get("tab") === "voice" ? "voice" : "general"
    router.replace(`/task?settings=${tab}${window.location.hash}`)
  }, [router])

  return null
}
