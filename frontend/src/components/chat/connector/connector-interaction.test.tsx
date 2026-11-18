/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { resolveTranslation } from "@/i18n/translations"
import { cosmosConnector } from "./fixtures"
import { parseConnectorInfo, parseResultCard } from "./types"

const appMock = vi.hoisted(() => ({ state: { currentTask: null as unknown } }))
vi.mock("@/contexts/app-context-chat", () => ({ useApp: () => appMock }))
vi.mock("@/lib/connector-connect", () => ({
  saveConnector: vi.fn(), skipConnector: vi.fn(), resumeConnector: vi.fn(), getConnectorState: vi.fn().mockResolvedValue({ state: "pending", tools: [] }),
}))
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) =>
      resolveTranslation("en", key as Parameters<typeof resolveTranslation>[1], vars),
  }),
}))

import { ConnectorInteraction } from "./ConnectorInteraction"


describe("ConnectorInteraction probe replay", () => {
  const waiting = { status: "waiting_for_user", waitingInteractions: [{ type: "connect_connector", interaction_id: cosmosConnector.interaction_id }] }
  beforeEach(() => { window.sessionStorage.clear(); appMock.state.currentTask = waiting })
  afterEach(() => { cleanup(); vi.restoreAllMocks() })

  it("animates the first time a pending card is seen, even under StrictMode, and not after a reload", () => {
    const view = render(<React.StrictMode><ConnectorInteraction connector={cosmosConnector} active /></React.StrictMode>)
    expect(screen.getByText("Checking for a native connector…")).toBeInTheDocument()
    view.unmount()
    // A reload: the same interaction again in the same tab.
    render(<React.StrictMode><ConnectorInteraction connector={cosmosConnector} active /></React.StrictMode>)
    expect(screen.getByText("Found a way · MCP server")).toBeInTheDocument()
  })

  it("still renders, animating, when sessionStorage throws", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("denied") })
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("denied") })
    render(<ConnectorInteraction connector={cosmosConnector} active />)
    expect(screen.getByText("Checking for a native connector…")).toBeInTheDocument()
  })

  it("does not animate a card the task is no longer waiting on", () => {
    render(<ConnectorInteraction connector={cosmosConnector} active={false} />)
    expect(screen.getByText("Found a way · MCP server")).toBeInTheDocument()
  })
})

describe("wire parsers", () => {
  it("reads the id only from the interaction item and requires one", () => {
    expect(parseConnectorInfo(cosmosConnector, "from-interaction")?.interaction_id).toBe("from-interaction")
    expect(parseConnectorInfo({ ...cosmosConnector, interaction_id: "in-payload" }, undefined)).toBeNull()
    expect(parseConnectorInfo(cosmosConnector, "")).toBeNull()
  })
  it("rejects a payload without usable fields or a name", () => {
    expect(parseConnectorInfo({ ...cosmosConnector, fields: [] }, "i")).toBeNull()
    expect(parseConnectorInfo({ ...cosmosConnector, fields: [{ label: "no key" }] }, "i")).toBeNull()
    expect(parseConnectorInfo({ ...cosmosConnector, name: "" }, "i")).toBeNull()
    expect(parseConnectorInfo("nope", "i")).toBeNull()
  })
  it("parses a result card and drops malformed parts", () => {
    expect(parseResultCard({ title: "T", source: { name: "S" }, items: ["a", 1], footnote: "" })).toMatchObject({
      title: "T", items: ["a"], footnote: null, source: { name: "S" },
    })
    expect(parseResultCard({ items: [] })).toBeNull()
  })
})
