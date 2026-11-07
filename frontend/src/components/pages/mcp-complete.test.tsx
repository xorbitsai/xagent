import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import McpCompletePage from "@/app/auth/mcp-complete/page"

afterEach(() => { cleanup(); vi.restoreAllMocks() })
describe("public MCP completion", () => {
  it("closes without authentication or opener messaging", () => {
    window.history.replaceState({}, "", "/auth/mcp-complete?mcp_oauth_success=1")
    const close = vi.spyOn(window, "close").mockImplementation(() => {})
    render(<McpCompletePage />)
    expect(close).toHaveBeenCalledOnce()
    expect(screen.getByText("Authorization completed")).toBeTruthy()
  })
  it("never renders provider errors or treats cancellation as success", () => {
    window.history.replaceState({}, "", "/auth/mcp-complete?mcp_oauth_error=authorization_failed&mcp_oauth_error_message=secret")
    vi.spyOn(window, "close").mockImplementation(() => {})
    render(<McpCompletePage />)
    expect(screen.getByText("Authorization not completed")).toBeTruthy()
    expect(screen.queryByText("secret")).toBeNull()
  })
})
