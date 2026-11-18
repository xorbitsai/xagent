import { renderHook } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { usePersonaAccentColor } from "./avatar-style-extension"

describe("avatar style shipped default", () => {
  it("reports no color for an uploaded image, a neutral initial, an agent or no persona", () => {
    const colorOf = (persona: Parameters<typeof usePersonaAccentColor>[0]) =>
      renderHook(() => usePersonaAccentColor(persona)).result.current

    expect(colorOf({ name: "Maya", avatar: "/avatars/maya.png" })).toBeNull()
    expect(colorOf({ name: "Maya" })).toBeNull()
    expect(colorOf({ name: "Maya", agentId: 7 })).toBeNull()
    expect(colorOf(undefined)).toBeNull()
    expect(colorOf(null)).toBeNull()
  })
})
