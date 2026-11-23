import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

let pathname: string | null = null
vi.mock("next/navigation", () => ({ usePathname: () => pathname }))

const mounted = vi.hoisted(() => ({ names: [] as string[] }))

function marker(name: string) {
  return function Marker({ children }: { children?: React.ReactNode }) {
    mounted.names.push(name)
    return <div data-testid={name}>{children}</div>
  }
}

vi.mock("@/components/auth/auth-guard", () => ({ AuthGuard: marker("AuthGuard") }))
vi.mock("@/components/layout/layout-content", () => ({ LayoutContent: marker("LayoutContent") }))
vi.mock("@/components/task-error-controller", () => ({ TaskErrorController: marker("TaskErrorController") }))
vi.mock("@/components/voice-input-controller", () => ({ VoiceInputController: marker("VoiceInputController") }))
vi.mock("@/contexts/auth-context", () => ({
  AnonymousAuthProvider: marker("AnonymousAuthProvider"),
  AuthProvider: marker("AuthProvider"),
}))
vi.mock("@/contexts/connector-runtime-dialog-context", () => ({
  ConnectorRuntimeDialogProvider: marker("ConnectorRuntimeDialogProvider"),
}))
vi.mock("@/contexts/mcp-apps-context", () => ({ McpAppsProvider: marker("McpAppsProvider") }))

import { ApplicationShell } from "./application-shell"

// The link host page embeds the chat page in a same-origin iframe, so nothing
// outside the chat page may run in front of it. The shell for external routes
// must therefore stay free of the application layout, which is where analytics
// and the support chat are mounted.
describe("ApplicationShell on the chat link routes", () => {
  beforeEach(() => {
    mounted.names.length = 0
  })

  afterEach(cleanup)

  it.each(["/widget/chat/link", "/widget/chat/link/ended"])("renders %s inside the anonymous provider only", (route) => {
    pathname = route
    render(<ApplicationShell><span>link page</span></ApplicationShell>)
    expect(screen.getByText("link page")).toBeInTheDocument()
    expect(mounted.names).toEqual(["AnonymousAuthProvider"])
    expect(screen.queryByTestId("LayoutContent")).toBeNull()
  })

  it("still mounts the application layout for an application route", () => {
    pathname = "/agents"
    render(<ApplicationShell><span>app page</span></ApplicationShell>)
    expect(screen.getByTestId("LayoutContent")).toBeInTheDocument()
  })
})
