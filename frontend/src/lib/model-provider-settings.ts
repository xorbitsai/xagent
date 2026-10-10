export type BedrockAuthMode = "api_key" | "credentials_chain"

type ModelProviderSettings = {
  model_provider: string
  bedrock_region?: string
  bedrock_auth_mode?: BedrockAuthMode
}

export function providerAllowsEmptyApiKey(
  providerId: string,
  bedrockAuthMode?: BedrockAuthMode,
): boolean {
  return providerId === "xinference"
    || (providerId === "bedrock" && bedrockAuthMode === "credentials_chain")
}

export function shouldFetchProviderModelCatalog(
  supportsModelListing?: boolean,
): boolean {
  return supportsModelListing !== false
}

export function buildModelRequestPayload<T extends ModelProviderSettings>(data: T): T {
  if (data.model_provider !== "bedrock") {
    const payload = { ...data }
    delete payload.bedrock_region
    delete payload.bedrock_auth_mode
    return payload
  }

  return {
    ...data,
    bedrock_region: data.bedrock_region?.trim(),
    bedrock_auth_mode: data.bedrock_auth_mode || "api_key",
  }
}
