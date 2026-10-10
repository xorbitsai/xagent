export type BedrockAuthMode = "api_key" | "credentials_chain"

type ModelProviderSettings = {
  model_provider: string
  bedrock_region?: string
  bedrock_auth_mode?: BedrockAuthMode
}

export function buildModelRequestPayload<T extends ModelProviderSettings>(data: T): T {
  if (data.model_provider !== "bedrock") return { ...data }

  return {
    ...data,
    bedrock_region: data.bedrock_region?.trim(),
    bedrock_auth_mode: data.bedrock_auth_mode || "api_key",
  }
}
