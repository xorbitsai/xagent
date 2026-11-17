import { renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const personaAccentColorMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/avatar-style-extension", () => ({
  usePersonaAccentColor: personaAccentColorMock,
}))

import { readableAccent, useAgentAccentAttributes } from "./agent-accent"

function contrastWithWhite(color: readonly [number, number, number]) {
  const [red, green, blue] = color.map((channel) => {
    const value = channel / 255
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4
  })
  return 1.05 / (0.2126 * red + 0.7152 * green + 0.0722 * blue + 0.05)
}

describe("readableAccent", () => {
  it("leaves a color alone when white text already reads on it", () => {
    expect(readableAccent([17, 78, 184])).toEqual([17, 78, 184])
  })

  it.each([
    ["mint", [29, 188, 143]],
    ["amber", [225, 151, 35]],
    ["sky", [48, 151, 222]],
    ["white", [255, 255, 255]],
  ] as const)("darkens %s until white text reads on it, keeping its hue", (_name, color) => {
    const shade = readableAccent(color)

    expect(contrastWithWhite(color)).toBeLessThan(4.5)
    expect(contrastWithWhite(shade)).toBeGreaterThanOrEqual(4.5)
    // Every channel is scaled by the same factor, so their order survives.
    const order = (rgb: readonly number[]) => [0, 1, 2].sort((a, b) => rgb[a] - rgb[b])
    expect(order(shade)).toEqual(order(color))
  })

  it("darkens no further than readability needs", () => {
    const shade = readableAccent([29, 188, 143])
    const slightlyLighter = shade.map((channel) => Math.round(channel * 1.05)) as [number, number, number]

    expect(contrastWithWhite(slightlyLighter)).toBeLessThan(4.5)
  })
})

describe("useAgentAccentAttributes", () => {
  beforeEach(() => {
    personaAccentColorMock.mockReset()
  })

  it("is empty when the agent's avatar has no color of its own", () => {
    personaAccentColorMock.mockReturnValue(null)

    const { result } = renderHook(() => useAgentAccentAttributes({ name: "Maya", avatar: "/avatars/maya.png" }))

    expect(result.current).toEqual({})
  })

  it("is empty when there is no agent", () => {
    personaAccentColorMock.mockReturnValue(null)

    expect(renderHook(() => useAgentAccentAttributes(undefined)).result.current).toEqual({})
    expect(renderHook(() => useAgentAccentAttributes(null)).result.current).toEqual({})
  })

  it("marks the wrapper and carries the readable shade of the avatar's color", () => {
    personaAccentColorMock.mockReturnValue([17, 78, 184])
    const persona = { name: "Maya", avatar: null }

    const { result } = renderHook(() => useAgentAccentAttributes(persona))

    expect(result.current).toEqual({
      "data-agent-accent": "",
      style: { "--agent-accent": "rgb(17 78 184)" },
    })
    expect(personaAccentColorMock).toHaveBeenCalledWith(persona)
  })

  it("follows a color that only becomes known later, and keeps one object while it is unchanged", () => {
    personaAccentColorMock.mockReturnValue(null)
    const { result, rerender } = renderHook(() => useAgentAccentAttributes({ name: "Maya" }))
    expect(result.current).toEqual({})

    personaAccentColorMock.mockReturnValue([17, 78, 184])
    rerender()
    const tinted = result.current
    expect(tinted["data-agent-accent"]).toBe("")

    // A new color array with the same values is no change.
    personaAccentColorMock.mockReturnValue([17, 78, 184])
    rerender()
    expect(result.current).toBe(tinted)
  })
})
