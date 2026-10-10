import { describe, expect, it } from "vitest"

import {
  buildModelRequestPayload,
  providerAllowsEmptyApiKey,
  shouldFetchProviderModelCatalog,
} from "./model-provider-settings"

describe("providerAllowsEmptyApiKey", () => {
  it("allows the AWS credential chain without an API key", () => {
    expect(providerAllowsEmptyApiKey("bedrock", "credentials_chain")).toBe(true)
    expect(providerAllowsEmptyApiKey("bedrock", "api_key")).toBe(false)
  })

  it("preserves Xinference's optional API key behavior", () => {
    expect(providerAllowsEmptyApiKey("xinference")).toBe(true)
    expect(providerAllowsEmptyApiKey("openai")).toBe(false)
  })
})

describe("shouldFetchProviderModelCatalog", () => {
  it("skips providers that advertise no catalog support", () => {
    expect(shouldFetchProviderModelCatalog(false)).toBe(false)
    expect(shouldFetchProviderModelCatalog(true)).toBe(true)
    expect(shouldFetchProviderModelCatalog(undefined)).toBe(true)
  })
})

describe("buildModelRequestPayload", () => {
  it.each([
    "anthropic.claude-sonnet-4-5-v1:0",
    "us.anthropic.claude-sonnet-4-5-v1:0",
    "arn:aws:bedrock:us-west-2:123456789012:inference-profile/example",
  ])("preserves a manually entered Bedrock model or profile identifier: %s", (modelName) => {
    const payload = buildModelRequestPayload({
      model_provider: "bedrock",
      model_name: modelName,
      bedrock_region: " us-west-2 ",
      bedrock_auth_mode: "credentials_chain" as const,
      base_url: "https://bedrock-runtime.us-west-2.amazonaws.com",
      api_key: "",
    })

    expect(payload).toEqual({
      model_provider: "bedrock",
      model_name: modelName,
      bedrock_region: "us-west-2",
      bedrock_auth_mode: "credentials_chain",
      base_url: "https://bedrock-runtime.us-west-2.amazonaws.com",
      api_key: "",
    })
  })

  it("defaults Bedrock authentication to an API key", () => {
    expect(buildModelRequestPayload({
      model_provider: "bedrock",
      model_name: "anthropic.claude-sonnet-4-5-v1:0",
      bedrock_region: "us-east-1",
      api_key: "test-token",
    })).toEqual({
      model_provider: "bedrock",
      model_name: "anthropic.claude-sonnet-4-5-v1:0",
      bedrock_region: "us-east-1",
      bedrock_auth_mode: "api_key",
      api_key: "test-token",
    })
  })

  it("removes Bedrock-only settings from other provider payloads", () => {
    expect(buildModelRequestPayload({
      model_provider: "openai",
      model_name: "gpt-5",
      bedrock_region: undefined,
      bedrock_auth_mode: "api_key" as const,
      api_key: "test-token",
    })).toEqual({
      model_provider: "openai",
      model_name: "gpt-5",
      api_key: "test-token",
    })
  })
})
