import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { LayoutContent } from "./layout-content"

const route = vi.hoisted(() => ({ pathname: "/" as string | null }))

vi.mock("next/navigation", () => ({
  usePathname: () => route.pathname,
}))

vi.mock("@/components/layout/sidebar", () => ({
  Sidebar: () => <div data-testid="sidebar" />,
}))

vi.mock("@/components/layout/mobile-header", () => ({
  MobileHeader: () => <div data-testid="mobile-header" />,
}))

vi.mock("@/contexts/app-context-chat", () => ({
  AppProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

// Pins a PR review test-coverage gap: the chromeless branch this PR added
// (skip the sidebar shell for /onboarding, same as the existing auth-public-
// path branch) had no test at all.
describe("LayoutContent", () => {
  afterEach(cleanup)

  it("renders the sidebar shell for a normal authenticated route", () => {
    route.pathname = "/task"
    render(<LayoutContent><div data-testid="children" /></LayoutContent>)

    expect(screen.getByTestId("sidebar")).toBeInTheDocument()
    expect(screen.getByTestId("mobile-header")).toBeInTheDocument()
    expect(screen.getByTestId("children")).toBeInTheDocument()
  })

  it("skips the sidebar shell for an auth-public page", () => {
    route.pathname = "/login"
    render(<LayoutContent><div data-testid="children" /></LayoutContent>)

    expect(screen.queryByTestId("sidebar")).not.toBeInTheDocument()
    expect(screen.queryByTestId("mobile-header")).not.toBeInTheDocument()
    expect(screen.getByTestId("children")).toBeInTheDocument()
  })

  it("skips the sidebar shell for /onboarding without treating it as a public (unauthenticated) page", () => {
    route.pathname = "/onboarding"
    render(<LayoutContent><div data-testid="children" /></LayoutContent>)

    expect(screen.queryByTestId("sidebar")).not.toBeInTheDocument()
    expect(screen.queryByTestId("mobile-header")).not.toBeInTheDocument()
    expect(screen.getByTestId("children")).toBeInTheDocument()
  })

  it("does not treat an unrelated path as chromeless", () => {
    route.pathname = "/settings"
    render(<LayoutContent><div data-testid="children" /></LayoutContent>)

    expect(screen.getByTestId("sidebar")).toBeInTheDocument()
  })
})
