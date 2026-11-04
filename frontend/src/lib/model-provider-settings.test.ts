import { describe, expect, it } from "vitest"

import {
  bedrockAuthModesForUser,
  bedrockDefaults,
  buildModelRequestPayload,
  hasInvalidModelProviderSettings,
  isBedrockCredentialChainBlocked,
  modelToFormData,
  providerAllowsEmptyApiKey,
  shouldFetchProviderModelCatalog,
} from "./model-provider-settings"

describe("Bedrock credential-chain permissions", () => {
  it("offers the credential chain only to administrators creating a model", () => {
    expect(bedrockAuthModesForUser(false)).toEqual(["api_key"])
    expect(bedrockAuthModesForUser(true)).toEqual(["api_key", "credentials_chain"])
  })

  it("keeps a historical credential-chain value visible so it can be changed", () => {
    expect(bedrockAuthModesForUser(false, "credentials_chain")).toEqual([
      "api_key",
      "credentials_chain",
    ])
  })

  it("blocks only a non-admin Bedrock credential-chain submission", () => {
    expect(isBedrockCredentialChainBlocked("bedrock", "credentials_chain", false)).toBe(true)
    expect(isBedrockCredentialChainBlocked("bedrock", "credentials_chain", true)).toBe(false)
    expect(isBedrockCredentialChainBlocked("bedrock", "api_key", false)).toBe(false)
    expect(isBedrockCredentialChainBlocked("openai", undefined, false)).toBe(false)
  })
})

describe("Bedrock form state", () => {
  it("starts new Bedrock configurations with an empty region", () => {
    expect(bedrockDefaults("bedrock")).toStrictEqual({
      bedrock_region: "",
      bedrock_auth_mode: "api_key",
    })
  })

  it("preserves saved Bedrock settings and clears them for other providers", () => {
    const previous = {
      bedrock_region: "eu-west-1",
      bedrock_auth_mode: "credentials_chain" as const,
    }

    expect(bedrockDefaults("bedrock", previous)).toStrictEqual(previous)
    expect(bedrockDefaults("openai", previous)).toStrictEqual({
      bedrock_region: undefined,
      bedrock_auth_mode: undefined,
    })
  })

  it("maps a saved model into the form without duplicating reload behavior", () => {
    const arn = "arn:aws:bedrock:us-west-2:123456789012:inference-profile/example"

    expect(modelToFormData({
      model_id: "bedrock-example",
      category: "llm",
      model_provider: "bedrock",
      model_name: arn,
      base_url: "https://bedrock-runtime.us-west-2.amazonaws.com",
      bedrock_region: "us-west-2",
      bedrock_auth_mode: "credentials_chain",
      context_window: 200000,
      abilities: ["chat", "vision"],
      is_shared: true,
    }, ["general", "visual"])).toStrictEqual({
      model_id: "bedrock-example",
      category: "llm",
      model_provider: "bedrock",
      model_name: arn,
      api_key: "",
      base_url: "https://bedrock-runtime.us-west-2.amazonaws.com",
      bedrock_region: "us-west-2",
      bedrock_auth_mode: "credentials_chain",
      temperature: undefined,
      context_window: 200000,
      dimension: undefined,
      abilities: ["chat", "vision"],
      default_config_types: ["general", "visual"],
      share_with_users: true,
    })
  })
})

describe("hasInvalidModelProviderSettings", () => {
  const settings = {
    model_provider: "bedrock",
    api_key: "",
    bedrock_region: "us-west-2",
    bedrock_auth_mode: "api_key" as const,
  }

  it("requires a region and a key for new Bedrock API-key configurations", () => {
    expect(hasInvalidModelProviderSettings(settings, { isEditing: false })).toBe(true)
    expect(hasInvalidModelProviderSettings({ ...settings, api_key: "token" }, { isEditing: false })).toBe(false)
    expect(hasInvalidModelProviderSettings({
      ...settings,
      api_key: "token",
      bedrock_region: " ",
    }, { isEditing: false })).toBe(true)
  })

  it("allows an unchanged saved key but requires a new key after leaving the credential chain", () => {
    expect(hasInvalidModelProviderSettings(settings, {
      isEditing: true,
      initialBedrockAuthMode: "api_key",
    })).toBe(false)
    expect(hasInvalidModelProviderSettings(settings, {
      isEditing: true,
      initialBedrockAuthMode: "credentials_chain",
    })).toBe(true)
  })

  it("allows credential-chain and Xinference submissions without an API key", () => {
    expect(hasInvalidModelProviderSettings({
      ...settings,
      bedrock_auth_mode: "credentials_chain",
    }, { isEditing: false })).toBe(false)
    expect(hasInvalidModelProviderSettings({
      model_provider: "xinference",
      api_key: "",
    }, { isEditing: false })).toBe(false)
  })
})

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
  it("skips Bedrock without requiring provider metadata", () => {
    expect(shouldFetchProviderModelCatalog("bedrock")).toBe(false)
    expect(shouldFetchProviderModelCatalog("claude")).toBe(true)
    expect(shouldFetchProviderModelCatalog("openai")).toBe(true)
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
      bedrock_region: "us-west-2",
      bedrock_auth_mode: "api_key" as const,
      api_key: "test-token",
    })).toStrictEqual({
      model_provider: "openai",
      model_name: "gpt-5",
      api_key: "test-token",
    })
  })
})
