import { describe, expect, it } from "vitest"
import { MessageDeliveryError } from "@/hooks/use-websocket"
import {
  createClarificationSendFailure,
  readSendDisposition,
  readSendErrorCode,
  readSendReason,
} from "./clarification-delivery"

describe("clarification-delivery readers", () => {
  it("reads every contract field off a real MessageDeliveryError", () => {
    const error = new MessageDeliveryError(
      "A previous guidance message is still being applied.",
      "outcome_unknown",
      { retryWithNewId: true, userFacing: true, errorCode: "guidance_in_progress" },
    )

    expect(readSendDisposition(error)).toBe("outcome_unknown")
    expect(readSendReason(error)).toBe(
      "A previous guidance message is still being applied.",
    )
    expect(readSendErrorCode(error)).toBe("guidance_in_progress")
  })

  it("the factory's product satisfies the same readers", () => {
    const failure = createClarificationSendFailure("Failed to send interaction", "not_sent")

    expect(readSendDisposition(failure)).toBe("not_sent")
    expect(readSendReason(failure)).toBe("")
    expect(readSendErrorCode(failure)).toBeNull()
  })
})
