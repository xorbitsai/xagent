/**
 * Model management API service
 */

import { getApiUrl } from './utils';
import { apiRequest } from './api-wrapper';

export type DefaultModelType =
  | 'general'
  | 'small_fast'
  | 'visual'
  | 'compact'
  | 'embedding'
  | 'image'
  | 'image_edit'
  | 'asr'
  | 'tts'
  | 'speech';

export interface Model {
  id: number;
  name: string;
  model_id: string;
  model_name?: string;
  provider: string;
  model_provider: string;
  category?: 'llm' | 'embedding' | 'image' | 'speech';
  api_key?: string;
  base_url?: string;
  max_tokens?: number;
  temperature?: number;
  dimension?: number;
  abilities?: string[];
  is_shared: boolean;
  created_by?: number;
  created_at: string;
  updated_at: string;
}

export interface UserDefaultModel {
  id: number;
  user_id: number;
  config_type: DefaultModelType;
  model_id: number;
  created_at: string;
  updated_at: string;
}

export interface ModelConfig {
  id: number;
  model: Model;
}

export interface DefaultModelConfig {
  general?: ModelConfig;
  small_fast?: ModelConfig;
  visual?: ModelConfig;
  compact?: ModelConfig;
  embedding?: ModelConfig;
  image?: ModelConfig;
  image_edit?: ModelConfig;
  asr?: ModelConfig;
  tts?: ModelConfig;
  speech?: ModelConfig;
}

/**
 * Get all models for current user
 */
export async function getUserModels(_token: string): Promise<Model[]> {
  const apiUrl = getApiUrl()
  const response = await apiRequest(`${apiUrl}/api/models/`);

  if (!response.ok) {
    throw new Error('Failed to fetch models');
  }

  return response.json();
}

/**
 * Get user's default model configuration
 */
export async function getUserDefaultModels(_token: string): Promise<DefaultModelConfig> {
  const apiUrl = getApiUrl()
  const response = await apiRequest(`${apiUrl}/api/models/user-default`);

  if (!response.ok) {
    throw new Error('Failed to fetch default models');
  }

  return response.json();
}

/**
 * Set user's default model for a specific type
 */
export async function setUserDefaultModel(
  _token: string,
  configType: DefaultModelType,
  modelId: number
): Promise<void> {
  const apiUrl = getApiUrl()
  const response = await apiRequest(`${apiUrl}/api/models/user-default`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      config_type: configType,
      model_id: modelId,
    }),
  });

  if (!response.ok) {
    throw new Error('Failed to set default model');
  }
}

/**
 * Remove user's default model for a specific type
 */
export async function removeUserDefaultModel(
  _token: string,
  configType: DefaultModelType
): Promise<void> {
  const apiUrl = getApiUrl()
  const response = await apiRequest(`${apiUrl}/api/models/user-default/${configType}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error('Failed to remove default model');
  }
}

/**
 * Get system default models (fallback)
 */
export async function getSystemDefaultModels(_token: string): Promise<DefaultModelConfig> {
  const apiUrl = getApiUrl()
  const [general, smallFast, visual, compact, embedding] = await Promise.all([
    apiRequest(`${apiUrl}/api/models/default/general`)
      .then(res => res.json().catch(() => null)),
    apiRequest(`${apiUrl}/api/models/default/small-fast`)
      .then(res => res.json().catch(() => null)),
    apiRequest(`${apiUrl}/api/models/default/visual`)
      .then(res => res.json().catch(() => null)),
    apiRequest(`${apiUrl}/api/models/default/compact`)
      .then(res => res.json().catch(() => null)),
    apiRequest(`${apiUrl}/api/models/default/embedding`)
      .then(res => res.json().catch(() => null)),
  ]);

  return {
    general,
    small_fast: smallFast,
    visual,
    compact,
    embedding,
  };
}

export interface Provider {
  id: string;
  name: string;
  description: string;
  requires_base_url?: boolean;
  icon?: string;
  default_base_url?: string;
}

export interface ProviderModel {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  model_type?: string;
  model_ability?: string[];
  abilities?: string[];  // Added for xagent compatibility
  category?: string;
  model_provider?: string;
  description?: string;
}

/**
 * Get list of supported model providers
 */
export async function getSupportedProviders(): Promise<Provider[]> {
  const apiUrl = getApiUrl()
  const response = await apiRequest(`${apiUrl}/api/models/providers/supported`);

  if (!response.ok) {
    throw new Error('Failed to fetch supported providers');
  }

  const data = await response.json();
  if (Array.isArray(data)) {
    return data;
  }
  if (data && Array.isArray(data.providers)) {
    return data.providers;
  }
  return [];
}

/**
 * Fetch models from a specific provider
 */
export async function getProviderModels(
  provider: string,
  config?: { api_key?: string; base_url?: string; category?: string }
): Promise<ProviderModel[]> {
  const apiUrl = getApiUrl()

  const response = await apiRequest(`${apiUrl}/api/models/providers/${provider}/models`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      api_key: config?.api_key ?? '',
      base_url: config?.base_url,
      category: config?.category,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to fetch provider models');
  }

  const data = await response.json();
  if (data && Array.isArray(data.models)) {
    return data.models;
  }
  return Array.isArray(data) ? data : [];
}

/**
 * Result of looking up abilities for a (provider, model_name) pair against
 * the curated ability catalog on the backend.
 *
 * `source` semantics:
 *   - "exact":              matched a provider-specific rule
 *   - "wildcard_provider":  matched a cross-provider rule (e.g. DeepSeek
 *                           served via an OpenAI-compatible endpoint)
 *   - "none":               no rule matched; abilities array will be empty
 */
export interface AbilitySuggestion {
  abilities: string[]
  matched_pattern: string | null
  source: "exact" | "wildcard_provider" | "none"
}

/**
 * Ask the backend which abilities to pre-select for a given model.
 * Returns `{ abilities: [], source: "none" }` for unknown models so callers
 * can simply test `source !== "none"` to decide whether to auto-fill.
 *
 * Network failures are swallowed and surfaced as `source: "none"` — the
 * Add-Model wizard should keep working even if the catalog endpoint is down.
 */
export async function getAbilitySuggestion(
  provider: string,
  modelName: string,
): Promise<AbilitySuggestion> {
  if (!provider || !modelName) {
    return { abilities: [], matched_pattern: null, source: "none" }
  }
  const apiUrl = getApiUrl()
  const qs = new URLSearchParams({ provider, model_name: modelName }).toString()
  try {
    const response = await apiRequest(`${apiUrl}/api/models/abilities/suggest?${qs}`, {
      method: "GET",
    })
    if (!response.ok) {
      return { abilities: [], matched_pattern: null, source: "none" }
    }
    const data = (await response.json()) as Partial<AbilitySuggestion>
    // Whitelist-validate `source` instead of casting: an unexpected value
    // from the backend would otherwise leak through the cast into React
    // state and break downstream union-narrowing.
    const validSources: AbilitySuggestion["source"][] = [
      "exact",
      "wildcard_provider",
      "none",
    ]
    const source: AbilitySuggestion["source"] = validSources.includes(
      data.source as AbilitySuggestion["source"],
    )
      ? (data.source as AbilitySuggestion["source"])
      : "none"
    return {
      abilities: Array.isArray(data.abilities) ? data.abilities : [],
      matched_pattern: data.matched_pattern ?? null,
      source,
    }
  } catch (error) {
    console.error("Failed to get ability suggestion:", error)
    return { abilities: [], matched_pattern: null, source: "none" }
  }
}
