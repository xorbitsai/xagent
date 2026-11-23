import React from "react"
import type { Viewport } from "next"
import PageClient from "./page-client"

// The page insets the iframe and its close button with env(safe-area-inset-*),
// which iOS reports as 0 unless the viewport covers the whole screen.
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
}

// Host page of a one-time chat link. The static `link` segment takes precedence
// over the dynamic `[token]` segment next to it. The link code arrives in the
// URL fragment, which never reaches the server, so this page is a static shell.
export default function Page() {
  return <PageClient />
}
