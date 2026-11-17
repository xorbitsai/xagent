/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const copyToClipboardMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const translateMock = vi.hoisted(() => vi.fn((key: string) => key))
let deploymentConfigFails = false

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) =>
    values.filter(Boolean).join(" "),
  getApiUrl: () => "https://configured-api.example.test",
}))

vi.mock("@/contexts/i18n-context", () => ({
  // The production context memoizes `t`. Preserve that contract so effects
  // which correctly depend on it do not rerun solely because of the test mock.
  useI18n: () => ({ t: translateMock }),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: toastErrorMock,
    success: vi.fn(),
  },
}))

vi.mock("@/lib/clipboard", () => ({
  copyToClipboard: copyToClipboardMock,
}))

vi.mock("@/lib/browser-location", () => ({
  getBrowserLocationOrigin: () => "https://cloud.example.test",
}))

vi.mock("@/lib/deploy-agent-extension", () => ({
  DeployAgentEmbedExtension: ({
    agentId,
    widgetOrigin,
    children,
  }: {
    agentId: number
    widgetOrigin: string
    children: React.ReactNode
  }) => (
    <div
      data-testid="embed-extension"
      data-agent-id={agentId}
      data-widget-origin={widgetOrigin}
    >
      {children}
    </div>
  ),
}))

vi.mock("@/lib/agent-channel-extension", () => ({
  AgentChannelOption: ({ agent, onOpen }: { agent: { name: string }; onOpen: () => void }) => (
    <button type="button" onClick={onOpen}>{`channel-option ${agent.name}`}</button>
  ),
  AgentChannelView: ({
    agent,
    onBack,
    onClose,
  }: {
    agent: { name: string }
    onBack: () => void
    onClose: () => void
  }) => (
    <div data-testid="channel-view">
      {agent.name}
      <button type="button" onClick={onBack}>channel-back</button>
      <button type="button" onClick={onClose}>channel-close</button>
    </div>
  ),
}))

import { __resetDeploymentConfigCache } from "@/lib/deployment-config"
import { DeployAgentDialog, type Agent } from "./deploy-agent-dialog"

const AGENT: Agent = {
  id: 7,
  name: "Regional agent",
  description: "",
  logo_url: null,
  template_id: null,
  status: "published",
  created_at: "2026-07-24T00:00:00Z",
  updated_at: "2026-07-24T00:00:00Z",
  widget_enabled: true,
  allowed_domains: ["*"],
  share_enabled: true,
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
}

describe("DeployAgentDialog regional targets", () => {
  beforeEach(() => {
    __resetDeploymentConfigCache()
    deploymentConfigFails = false
    apiRequestMock.mockReset()
    apiRequestMock.mockImplementation((url: string) => {
      if (url.endsWith("/api/deployment-config")) {
        if (deploymentConfigFails) {
          return Promise.reject(new Error("deployment config unavailable"))
        }
        return Promise.resolve(
          jsonResponse({
            deployment_origin: "https://sg-origin.cloud.example.test",
            app_origin: "https://cloud.example.test",
            region: "sg",
          }),
        )
      }
      if (url.endsWith("/api/agents/7/widget-key")) {
        return Promise.resolve(
          jsonResponse({
            agent_id: 7,
            widget_enabled: true,
            widget_key: "wk-regional",
          }),
        )
      }
      if (url.endsWith("/api/agents/7/share-link")) {
        return Promise.resolve(
          jsonResponse({
            agent_id: 7,
            share_enabled: true,
            share_token: "regional-share",
            share_updated_at: "2026-07-24T00:00:00Z",
          }),
        )
      }
      throw new Error(`Unexpected API request: ${url}`)
    })
    copyToClipboardMock.mockReset()
    copyToClipboardMock.mockResolvedValue(true)
    toastErrorMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it("keeps API copy disabled until config retry succeeds", async () => {
    deploymentConfigFails = true

    render(
      <DeployAgentDialog
        deployAgent={AGENT}
        onClose={vi.fn()}
        onUpdate={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByText("deploy_agent.options.rest_api.title"))

    expect(await screen.findByText("deployment_config.messages.load_failed")).toBeInTheDocument()
    const copyButton = screen.getByTitle("deploy_agent.api_panel.copy_btn")
    expect(copyButton).toBeDisabled()
    expect(
      screen.queryByText((content) =>
        content.includes("https://cloud.example.test/v1/chat/tasks"),
      ),
    ).not.toBeInTheDocument()
    expect(toastErrorMock).toHaveBeenCalledWith(
      "deployment_config.messages.load_failed",
    )

    expect(screen.getByText("deployment_config.messages.load_failed")).toBeInTheDocument()
    deploymentConfigFails = false
    fireEvent.click(screen.getByRole("button", {
      name: "deployment_config.actions.retry",
    }))

    expect(
      await screen.findByText((content) =>
        content.includes(
          "https://sg-origin.cloud.example.test/v1/chat/tasks",
        ),
      ),
    ).toBeInTheDocument()
    expect(
      screen.queryByText("deployment_config.messages.load_failed"),
    ).not.toBeInTheDocument()
    expect(copyButton).toBeEnabled()
  })

  it("keeps the share panel available when deployment configuration fails", async () => {
    deploymentConfigFails = true

    render(
      <DeployAgentDialog
        deployAgent={AGENT}
        onClose={vi.fn()}
        onUpdate={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByText("deploy_agent.options.shareable_link.title"))

    expect(
      await screen.findByText("deployment_config.messages.load_failed"),
    ).toBeInTheDocument()
    expect(await screen.findByRole("textbox")).toHaveValue("")
    expect(screen.getByRole("button", { name: "common.copy" })).toBeDisabled()
    expect(
      screen.queryByText("deploy_agent.messages.share_failed"),
    ).not.toBeInTheDocument()
  })

  it("uses the deployment origin for API and SDK snippets", async () => {
    render(
      <DeployAgentDialog
        deployAgent={AGENT}
        onClose={vi.fn()}
        onUpdate={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByText("deploy_agent.options.rest_api.title"))

    expect(
      await screen.findByText((content) =>
        content.includes(
          "https://sg-origin.cloud.example.test/v1/chat/tasks",
        ),
      ),
    ).toBeInTheDocument()
  })

  it("uses the deployment origin for widget snippets", async () => {
    render(
      <DeployAgentDialog
        deployAgent={AGENT}
        onClose={vi.fn()}
        onUpdate={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByText("deploy_agent.options.embed.title"))
    const copyButton = await screen.findByTitle(
      "deploy_agent.embed_snippet.copy_btn",
    )
    await waitFor(() => {
      expect(copyButton).not.toBeDisabled()
    })
    fireEvent.click(copyButton)

    await waitFor(() => {
      expect(copyToClipboardMock).toHaveBeenCalledWith(
        expect.stringContaining(
          'src="https://sg-origin.cloud.example.test/widget.js"',
        ),
      )
    })
  })

  it("hands the embed snippet to the deploy extension slot", async () => {
    render(
      <DeployAgentDialog
        deployAgent={AGENT}
        onClose={vi.fn()}
        onUpdate={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByText("deploy_agent.options.embed.title"))
    const slot = await screen.findByTestId("embed-extension")
    await waitFor(() => {
      expect(slot).toHaveAttribute(
        "data-widget-origin",
        "https://sg-origin.cloud.example.test",
      )
    })
    expect(slot).toHaveAttribute("data-agent-id", "7")
    expect(slot).toContainElement(
      screen.getByTitle("deploy_agent.embed_snippet.copy_btn"),
    )
  })

  it("keeps widget copy disabled when deployment configuration fails", async () => {
    deploymentConfigFails = true

    render(
      <DeployAgentDialog
        deployAgent={AGENT}
        onClose={vi.fn()}
        onUpdate={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByText("deploy_agent.options.embed.title"))

    expect(
      await screen.findByText("deployment_config.messages.load_failed"),
    ).toBeInTheDocument()
    const copyButton = await screen.findByTitle(
      "deploy_agent.embed_snippet.copy_btn",
    )
    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "https://configured-api.example.test/api/agents/7/widget-key",
      )
    })
    expect(screen.getByText("…")).toBeInTheDocument()
    expect(copyButton).toBeDisabled()
    expect(copyToClipboardMock).not.toHaveBeenCalled()
  })

  it("bootstraps the region for public share links", async () => {
    render(
      <DeployAgentDialog
        deployAgent={AGENT}
        onClose={vi.fn()}
        onUpdate={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByText("deploy_agent.options.shareable_link.title"))

    expect(
      await screen.findByDisplayValue(
        "https://cloud.example.test/change-region?region=sg&next=%2Fshare%2Fregional-share",
      ),
    ).toBeInTheDocument()
  })

  it("copies share links through the checked clipboard helper", async () => {
    render(
      <DeployAgentDialog
        deployAgent={AGENT}
        onClose={vi.fn()}
        onUpdate={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByText("deploy_agent.options.shareable_link.title"))
    await screen.findByDisplayValue(
      "https://cloud.example.test/change-region?region=sg&next=%2Fshare%2Fregional-share",
    )
    fireEvent.click(screen.getByText("common.copy"))

    await waitFor(() => {
      expect(copyToClipboardMock).toHaveBeenCalledWith(
        "https://cloud.example.test/change-region?region=sg&next=%2Fshare%2Fregional-share",
      )
    })
  })

  describe("channel extension", () => {
    const renderDialog = (props: Partial<React.ComponentProps<typeof DeployAgentDialog>> = {}) =>
      render(
        <DeployAgentDialog
          deployAgent={AGENT}
          onClose={vi.fn()}
          onUpdate={vi.fn()}
          {...props}
        />,
      )

    it("opens on the options grid by default, with the channel option first", () => {
      renderDialog()
      expect(screen.queryByTestId("channel-view")).not.toBeInTheDocument()
      const option = screen.getByText("channel-option Regional agent")
      expect(
        option.compareDocumentPosition(screen.getByText("deploy_agent.options.embed.title"))
          & Node.DOCUMENT_POSITION_FOLLOWING,
      ).toBeTruthy()
    })

    it("opens directly on the channel view with initialView=channel and no default content", () => {
      renderDialog({ initialView: "channel" })
      expect(screen.getByTestId("channel-view")).toHaveTextContent("Regional agent")
      expect(screen.queryByText("deploy_agent.options.embed.title")).not.toBeInTheDocument()
    })

    it("enters the channel view from the option and returns to the options on back", () => {
      renderDialog()
      fireEvent.click(screen.getByText("channel-option Regional agent"))
      expect(screen.getByTestId("channel-view")).toBeInTheDocument()
      expect(screen.queryByText("deploy_agent.options.embed.title")).not.toBeInTheDocument()
      fireEvent.click(screen.getByText("channel-back"))
      expect(screen.queryByTestId("channel-view")).not.toBeInTheDocument()
      expect(screen.getByText("deploy_agent.options.embed.title")).toBeInTheDocument()
    })

    it("closes through the view's onClose", () => {
      const onClose = vi.fn()
      renderDialog({ initialView: "channel", onClose })
      fireEvent.click(screen.getByText("channel-close"))
      expect(onClose).toHaveBeenCalledTimes(1)
    })
  })
})
