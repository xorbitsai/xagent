import { describe, expect, it } from "vitest"
import { websocketAuthProtocols } from "./websocket-auth"

describe("private WebSocket header credentials", () => {
  it("carries the exact JWT as an offered protocol, not as a URL", () => {
    expect(websocketAuthProtocols("abc.def.ghi")).toEqual([
      "xagent-auth-v1", "xagent-auth-token.abc.def.ghi",
    ])
  })

  it.each(["", "a b", "a,b", "a\nb", "a".repeat(8193)])("rejects invalid protocol values without exposing them", token => {
    expect(() => websocketAuthProtocols(token)).toThrow("Invalid WebSocket credential")
  })
})
