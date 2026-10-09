import { describe, expect, it } from "vitest"

import { buildModelRequestPayload } from "./model-provider-settings"

describe("buildModelRequestPayload", () => {
  it("preserves a manually entered Bedrock profile ARN and provider settings", () => {
    const payload = buildModelRequestPayload({
      model_provider: "bedrock",
      model_name: "arn:aws:bedrock:us-west-2:123456789012:inference-profile/example",
      bedrock_region: " us-west-2 ",
      bedrock_auth_mode: "credentials_chain" as const,
      base_url: "https://bedrock-runtime.us-west-2.amazonaws.com",
      api_key: "",
    })

    expect(payload).toEqual({
      model_provider: "bedrock",
      model_name: "arn:aws:bedrock:us-west-2:123456789012:inference-profile/example",
      bedrock_region: "us-west-2",
      bedrock_auth_mode: "credentials_chain",
      base_url: "https://bedrock-runtime.us-west-2.amazonaws.com",
      api_key: "",
    })
  })
})
