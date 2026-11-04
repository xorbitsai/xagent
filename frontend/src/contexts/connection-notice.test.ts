import { describe, expect, it } from "vitest"
import { deriveConnectionNotice } from "./connection-notice"

describe("deriveConnectionNotice", () => {
  it.each([
    ["reconnecting after an open", false, true, "reconnecting", false, "reconnecting"],
    ["lost after an open", false, true, "lost", false, "lost"],
    ["lost without an open for this task", false, false, "lost", false, null],
    ["routine socket swap", false, true, "connecting", false, null],
    ["Session transport disconnected after an open", true, true, "lost", false, "reconnecting"],
    ["Session transport connected after an open", true, true, "connected", true, null],
    ["Session transport that never opened", true, false, "lost", false, null],
  ] as const)("%s", (_name, isSessionTransport, hadOpen, phase, isConnected, expected) => {
    expect(
      deriveConnectionNotice({ isSessionTransport, hadOpen, phase, isConnected }),
    ).toBe(expected)
  })
})
