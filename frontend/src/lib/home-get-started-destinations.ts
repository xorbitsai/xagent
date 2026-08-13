import type { HomeGetStartedDestinationOverrides } from "@/lib/page-extension-contracts"

// Canonical OSS default destinations for the home page "Get started" cards.
// Kept out of `page.tsx` (a Next.js App Router page file, which may only
// export the reserved page members) so it can be imported by both the page
// and its tests as a single source of truth.
export const defaultHomeGetStartedDestinations: Record<keyof HomeGetStartedDestinationOverrides, string> = {
  docs: "https://help.xagent.co/overview.html",
  guides: "https://help.xagent.co/user-guide/overview.html",
  whatsNew: "https://docs.xagent.co/release-notes",
}
