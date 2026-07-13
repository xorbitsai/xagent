/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const toastSuccessMock = vi.hoisted(() => vi.fn())
const copyToClipboardMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
  }
})

vi.mock("@/lib/browser-location", () => ({
  getBrowserLocationOrigin: () => "http://app.local",
}))

vi.mock("@/lib/clipboard", () => ({
  copyToClipboard: copyToClipboardMock,
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: toastErrorMock,
    success: toastSuccessMock,
  },
}))

import { DeployAgentDialog, type Agent } from "./deploy-agent-dialog"

function jsonResponse(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  })
}

const baseAgent: Agent = {
  id: 42,
  name: "Widget Agent",
  description: "test",
  logo_url: null,
  status: "published",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  widget_enabled: true,
  allowed_domains: ["example.com"],
}

function renderDialog(props?: Partial<React.ComponentProps<typeof DeployAgentDialog>>) {
  const onUpdate = vi.fn()
  render(
    <DeployAgentDialog
      deployAgent={baseAgent}
      onClose={vi.fn()}
      onUpdate={onUpdate}
      {...props}
    />,
  )
  return { onUpdate }
}

async function openEmbedView() {
  fireEvent.click(await screen.findByText("deploy_agent.options.embed.title"))
}

async function expandAdvancedOptions() {
  fireEvent.click(await screen.findByText("deploy_agent.embed_snippet.advanced_toggle"))
}

describe("DeployAgentDialog embed view", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    toastSuccessMock.mockReset()
    copyToClipboardMock.mockReset()
    copyToClipboardMock.mockResolvedValue(true)
    apiRequestMock.mockImplementation((url: string, options?: { body?: string }) => {
      if (url.endsWith("/widget-end-user-secret/rotate")) {
        return Promise.resolve(
          jsonResponse({ agent_id: 42, widget_end_user_secret: "secret-rotated" }),
        )
      }
      if (url.endsWith("/widget-end-user-secret")) {
        return Promise.resolve(
          jsonResponse({ agent_id: 42, widget_end_user_secret: "secret-test-value" }),
        )
      }
      if (url.endsWith("/widget-key/rotate")) {
        return Promise.resolve(
          jsonResponse({ agent_id: 42, widget_enabled: true, widget_key: "wk-rotated-key" }),
        )
      }
      if (url.endsWith("/widget-key")) {
        return Promise.resolve(
          jsonResponse({ agent_id: 42, widget_enabled: true, widget_key: "wk-test-key" }),
        )
      }
      const updates = options?.body ? JSON.parse(options.body) : {}
      return Promise.resolve(
        jsonResponse({ ...baseAgent, ...updates }),
      )
    })
  })

  afterEach(() => {
    cleanup()
  })

  it("fetches and displays the widget key and embed snippet using it", async () => {
    renderDialog()
    await openEmbedView()

    await waitFor(() => {
      expect(screen.getByText("wk-test-key")).toBeInTheDocument()
    })
    expect(
      screen.getByText((content) => content.includes('data-widget-key="wk-test-key"')),
    ).toBeInTheDocument()
  })

  it("renders advanced options collapsed by default and expands on click", async () => {
    renderDialog()
    await openEmbedView()

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/agents/42/widget-key")
    })
    const details = (await screen.findByText("appWidget.dialog.rotateWidgetKey")).closest("details")
    expect(details).not.toHaveAttribute("open")

    await expandAdvancedOptions()

    expect(details).toHaveAttribute("open")
  })

  it("copies the widget key via copyToClipboard", async () => {
    renderDialog()
    await openEmbedView()
    await expandAdvancedOptions()

    await waitFor(() => {
      expect(screen.getByText("wk-test-key")).toBeInTheDocument()
    })
    fireEvent.click(screen.getByTitle("appWidget.dialog.copyWidgetKey"))

    await waitFor(() => {
      expect(copyToClipboardMock).toHaveBeenCalledWith("wk-test-key")
    })
    expect(toastSuccessMock).toHaveBeenCalled()
  })

  it("rotates the widget key after confirmation", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true)
    try {
      renderDialog()
      await openEmbedView()
      await expandAdvancedOptions()

      await waitFor(() => {
        expect(screen.getByText("wk-test-key")).toBeInTheDocument()
      })
      fireEvent.click(screen.getByText("appWidget.dialog.rotateWidgetKey"))

      await waitFor(() => {
        expect(apiRequestMock).toHaveBeenCalledWith(
          "http://api.local/api/agents/42/widget-key/rotate",
          { method: "POST" },
        )
      })
      await waitFor(() => {
        expect(screen.getByText("wk-rotated-key")).toBeInTheDocument()
      })
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it("copies the end-user secret via copyToClipboard", async () => {
    renderDialog()
    await openEmbedView()
    await expandAdvancedOptions()

    await waitFor(() => {
      expect(screen.getByText("secret-test-value")).toBeInTheDocument()
    })
    fireEvent.click(screen.getByTitle("appWidget.dialog.copyEndUserSecret"))

    await waitFor(() => {
      expect(copyToClipboardMock).toHaveBeenCalledWith("secret-test-value")
    })
    expect(toastSuccessMock).toHaveBeenCalled()
  })

  it("rotates the end-user secret after confirmation", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true)
    try {
      renderDialog()
      await openEmbedView()
      await expandAdvancedOptions()

      await waitFor(() => {
        expect(screen.getByText("secret-test-value")).toBeInTheDocument()
      })
      fireEvent.click(screen.getByText("appWidget.dialog.rotateEndUserSecret"))

      await waitFor(() => {
        expect(apiRequestMock).toHaveBeenCalledWith(
          "http://api.local/api/agents/42/widget-end-user-secret/rotate",
          { method: "POST" },
        )
      })
      await waitFor(() => {
        expect(screen.getByText("secret-rotated")).toBeInTheDocument()
      })
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it("does not rotate the end-user secret when the operator declines", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false)
    try {
      renderDialog()
      await openEmbedView()
      await expandAdvancedOptions()

      await waitFor(() => {
        expect(screen.getByText("secret-test-value")).toBeInTheDocument()
      })
      fireEvent.click(screen.getByText("appWidget.dialog.rotateEndUserSecret"))

      expect(apiRequestMock).not.toHaveBeenCalledWith(
        "http://api.local/api/agents/42/widget-end-user-secret/rotate",
        { method: "POST" },
      )
      expect(screen.getByText("secret-test-value")).toBeInTheDocument()
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it("adds an allowed domain through the agent update endpoint", async () => {
    renderDialog()
    await openEmbedView()

    const input = await screen.findByPlaceholderText("deploy_agent.access_control.domain_placeholder")
    fireEvent.change(input, { target: { value: "new-domain.com" } })
    fireEvent.click(screen.getByText("deploy_agent.access_control.add_btn"))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/agents/42",
        expect.objectContaining({ method: "PUT" }),
      )
    })
    const updateCall = apiRequestMock.mock.calls.find(
      ([url, options]) => url === "http://api.local/api/agents/42" && options?.method === "PUT",
    )
    expect(JSON.parse(updateCall?.[1]?.body as string)).toMatchObject({
      allowed_domains: ["example.com", "new-domain.com"],
    })
  })
})
