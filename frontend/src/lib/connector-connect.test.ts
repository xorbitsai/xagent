import { afterEach, describe, expect, it, vi } from "vitest"
import { getConnectorState, resumeConnector, saveConnector, skipConnector } from "./connector-connect"

describe("the OSS connector-connect default", () => {
  afterEach(() => vi.unstubAllGlobals())

  it("answers every call as unavailable without making any request", async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal("fetch", fetchMock)
    const results = await Promise.all([
      saveConnector("i-1", "cosmos", { KEY: "secret-value" }),
      skipConnector("i-1"),
      getConnectorState("i-1"),
      resumeConnector("i-1"),
    ])
    for (const result of results) {
      expect(result).toMatchObject({ state: "unavailable", errorCode: null, tools: [], taskResumed: null })
    }
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
