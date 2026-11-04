export type BedrockAuthMode = "api_key" | "credentials_chain"

type ModelProviderSettings = {
  model_provider: string
  bedrock_region?: string
  bedrock_auth_mode?: BedrockAuthMode
}

export type ModelFormData = ModelProviderSettings & {
  model_id: string
  category: string
  model_name: string
  api_key: string
  base_url?: string
  temperature?: number
  context_window?: number
  dimension?: number
  abilities?: string[]
  default_config_types?: string[]
  share_with_users?: boolean
}

type ModelFormSource = Omit<
  ModelFormData,
  "api_key" | "default_config_types" | "share_with_users"
> & {
  is_shared?: boolean
}

export function bedrockDefaults(
  providerId: string,
  previous?: Pick<ModelProviderSettings, "bedrock_region" | "bedrock_auth_mode">,
): Pick<ModelProviderSettings, "bedrock_region" | "bedrock_auth_mode"> {
  if (providerId !== "bedrock") {
    return {
      bedrock_region: undefined,
      bedrock_auth_mode: undefined,
    }
  }

  return {
    bedrock_region: previous?.bedrock_region || "",
    bedrock_auth_mode: previous?.bedrock_auth_mode || "api_key",
  }
}

export function modelToFormData(
  model: ModelFormSource,
  defaultConfigTypes: string[],
): ModelFormData {
  return {
    model_id: model.model_id,
    category: model.category,
    model_provider: model.model_provider,
    model_name: model.model_name,
    api_key: "",
    base_url: model.base_url || "",
    ...bedrockDefaults(model.model_provider, model),
    temperature: model.temperature,
    context_window: model.context_window,
    dimension: model.dimension,
    abilities: model.abilities || [],
    default_config_types: defaultConfigTypes,
    share_with_users: model.is_shared,
  }
}

export function providerAllowsEmptyApiKey(
  providerId: string,
  bedrockAuthMode?: BedrockAuthMode,
): boolean {
  return providerId === "xinference"
    || (providerId === "bedrock" && bedrockAuthMode === "credentials_chain")
}

export function bedrockAuthModesForUser(
  isAdmin: boolean,
  currentMode?: BedrockAuthMode,
): BedrockAuthMode[] {
  if (isAdmin || currentMode === "credentials_chain") {
    return ["api_key", "credentials_chain"]
  }
  return ["api_key"]
}

export function isBedrockCredentialChainBlocked(
  providerId: string,
  authMode: BedrockAuthMode | undefined,
  isAdmin: boolean,
): boolean {
  return providerId === "bedrock"
    && authMode === "credentials_chain"
    && !isAdmin
}

export function hasInvalidModelProviderSettings(
  data: Pick<
    ModelFormData,
    "model_provider" | "api_key" | "bedrock_region" | "bedrock_auth_mode"
  >,
  options: {
    isEditing: boolean
    initialBedrockAuthMode?: BedrockAuthMode
  },
): boolean {
  if (data.model_provider === "bedrock" && !data.bedrock_region?.trim()) {
    return true
  }

  if (providerAllowsEmptyApiKey(data.model_provider, data.bedrock_auth_mode)) {
    return false
  }

  if (data.api_key.trim()) {
    return false
  }

  if (!options.isEditing) {
    return true
  }

  return data.model_provider === "bedrock"
    && options.initialBedrockAuthMode === "credentials_chain"
    && data.bedrock_auth_mode === "api_key"
}

export function shouldFetchProviderModelCatalog(
  providerId: string,
): boolean {
  return providerId !== "bedrock"
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
