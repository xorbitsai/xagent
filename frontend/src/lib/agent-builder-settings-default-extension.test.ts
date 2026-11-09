import { describe, expect, it } from "vitest"

import { useAgentBuilderSettingsExtension } from "./agent-builder-settings-extension"

describe("Agent Builder shipped default settings", () => {
  it("has no panel or submitted settings", () => {
    expect(useAgentBuilderSettingsExtension({ agentId: 7, enabled: true, saving: false })).toEqual({})
  })
})
