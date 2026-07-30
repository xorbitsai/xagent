import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { Sidebar } from "./sidebar"

const authState = vi.hoisted(() => ({ logout: vi.fn<() => Promise<boolean>>(), user: { id: "1", username: "alice" } }))
const toast = vi.hoisted(() => ({ error: vi.fn() }))

vi.mock("next/navigation", () => ({ usePathname: () => "/task", useRouter: () => ({ push: vi.fn() }) }))
vi.mock("next/image", () => ({ default: (props: React.ImgHTMLAttributes<HTMLImageElement>) => <img {...props} /> }))
vi.mock("next/link", () => ({ default: ({ children, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => <a {...props}>{children}</a> }))
vi.mock("@/contexts/auth-context", () => ({ useAuth: () => authState }))
vi.mock("@/contexts/app-context-chat", () => ({ useApp: () => ({ state: { lastTaskUpdate: 0 } }) }))
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))
vi.mock("@/components/ui/sonner", () => ({ toast }))
vi.mock("@/lib/branding", () => ({ getBrandingFromEnv: () => ({ appName: "Xagent" }) }))
vi.mock("@/lib/extra-nav", () => ({ default: [] }))
vi.mock("@/lib/sidebar-navigation", () => ({
  getNavigationGroupsForUser: () => [], getUserMenuItemsForUser: () => [],
}))

describe("Sidebar logout", () => {
  beforeEach(() => {
    authState.logout.mockReset()
    toast.error.mockReset()
    authState.logout.mockResolvedValue(false)
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 500 })))
  })
  afterEach(() => vi.unstubAllGlobals())

  it("keeps the menu open and reports a localized failure when logout cannot clear auth", async () => {
    render(<Sidebar />)
    fireEvent.click(screen.getByRole("button", { name: /alice/i }))
    fireEvent.click(screen.getByRole("button", { name: "sidebar.user.logoutTitle" }))
    await waitFor(() => expect(authState.logout).toHaveBeenCalledOnce())
    expect(toast.error).toHaveBeenCalledWith("sidebar.user.logoutFailed")
    expect(screen.getByRole("button", { name: "sidebar.user.logoutTitle" })).toBeInTheDocument()
  })
})
