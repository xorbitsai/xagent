import { describe, expect, it } from "vitest"
import { FAILED_RESULT, UNAVAILABLE_RESULT, parseConnectorConnectResult } from "./connector-connect-response"

describe("parseConnectorConnectResult", () => {
  it("maps a full response, keeping only known error codes and well-formed tools", () => {
    expect(parseConnectorConnectResult({
      state: "failed", error_code: "auth_failed", error_field: "KEY", secret_masked: "•••• 1234",
      scope: "local", agent_name: "Cesar", task_resumed: false,
      tools: [{ name: "q", label: "Query" }, { name: "r" }, { label: "no name" }, "junk"],
    })).toEqual({
      state: "failed", errorCode: "auth_failed", errorField: "KEY", secretMasked: "•••• 1234",
      scope: "local", agentName: "Cesar", tools: [{ name: "q", label: "Query" }, { name: "r", label: "r" }],
      taskResumed: false,
    })
  })

  it("drops an unknown error code and a non-boolean task_resumed", () => {
    const result = parseConnectorConnectResult({ state: "failed", error_code: "weird", task_resumed: "yes" })
    expect(result?.errorCode).toBeNull()
    expect(result?.taskResumed).toBeNull()
  })

  it.each([null, "x", [], {}, { state: "nonsense" }, { state: 3 }])("rejects %j", (body) => {
    expect(parseConnectorConnectResult(body)).toBeNull()
  })

  it("accepts every contract state", () => {
    for (const state of ["pending", "connected", "failed", "skipped", "unavailable"]) {
      expect(parseConnectorConnectResult({ state })?.state).toBe(state)
    }
  })

  it("exposes the failed and unavailable results", () => {
    expect(FAILED_RESULT.state).toBe("failed")
    expect(UNAVAILABLE_RESULT.state).toBe("unavailable")
  })
})
