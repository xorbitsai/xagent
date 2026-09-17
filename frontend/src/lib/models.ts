/**
 * Model management API service
 */

import { getApiUrl } from './utils';
import { apiRequest, isJsonRecord, parseApiResponse } from './api-wrapper';

const currentBackendDefaultModelTypes = [
  'general',
  'small_fast',
  'visual',
  'compact',
  'embedding',
  'image',
  'image_edit',
  'video',
  'asr',
  'tts',
  'speech',
  'sound_effect',
  'music',
  'rerank',
] as const;

export type DefaultModelType = (typeof currentBackendDefaultModelTypes)[number];

export interface Model {
  id: number;
  name: string;
  model_id: string;
  model_name?: string;
  provider: string;
  model_provider: string;
  category?: 'llm' | 'embedding' | 'image' | 'video' | 'speech' | 'sound_effect' | 'music' | 'rerank';
  api_key?: string;
  base_url?: string;
  max_tokens?: number;
  temperature?: number;
  context_window?: number;
  dimension?: number;
  abilities?: string[];
  is_shared: boolean;
  created_by?: number;
  created_at: string;
  updated_at: string;
}

// Platform (built-in) models are seeded by xagent cloud with a `platform/`
// model-id prefix. Central home for that convention so it can't drift across
// call sites. Inert for OSS, where no model carries the prefix.
export const isBuiltinModel = (model: { model_id?: string | null }): boolean =>
  Boolean(model.model_id?.startsWith("platform/"));

// Host (including port) extracted from a base URL, for disambiguating
// same-named models across endpoints. Uses `.host` so non-standard ports
// (xinference :9997, Ollama :11434) survive. Returns "" for empty or
// non-absolute URLs so callers fall back to a provider-only label instead
// of rendering a garbled raw string.
export const hostnameFromUrl = (url?: string | null): string => {
  if (!url) return "";
  try {
    return new URL(url).host;
  } catch {
    return "";
  }
};

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
  video?: ModelConfig;
  asr?: ModelConfig;
  tts?: ModelConfig;
  speech?: ModelConfig;
  sound_effect?: ModelConfig;
  music?: ModelConfig;
}

export interface ModelWithAccess {
  id: number;
  model_id: string;
  category: string;
  model_provider: string;
  model_name: string;
  base_url: string | null;
  temperature: number | null;
  context_window: number | null;
  dimension: number | null;
  abilities: string[] | null;
  description: string | null;
  created_at: string | null;
  updated_at: string | null;
  is_active: boolean;
  is_owner: boolean;
  can_edit: boolean;
  can_delete: boolean;
  is_shared: boolean;
}

export interface UserDefaultModelEntry {
  id: number;
  user_id: number;
  model_id: number;
  config_type: DefaultModelType;
  created_at: string | null;
  updated_at: string | null;
  model: ModelWithAccess;
}

export type UserDefaultModelMap = Partial<Record<DefaultModelType, UserDefaultModelEntry>>;

const defaultTypes: ReadonlySet<string> = new Set(currentBackendDefaultModelTypes);

const positiveSafeInt = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value > 0;

const nullableString = (value: unknown): value is string | null =>
  value === null || typeof value === 'string';

const nullableFinite = (value: unknown): value is number | null =>
  value === null || (typeof value === 'number' && Number.isFinite(value));

const nullableInt = (value: unknown): value is number | null =>
  value === null || (typeof value === 'number' && Number.isInteger(value));

function isKnownDefaultModelType(value: string): value is DefaultModelType {
  return defaultTypes.has(value);
}

export function parseModelList(value: unknown): ModelWithAccess[] | null {
  if (!Array.isArray(value)) return null;

  const result: ModelWithAccess[] = [];
  for (const item of value) {
    if (!isJsonRecord(item)) return null;

    const {
      id,
      model_id,
      category,
      model_provider,
      model_name,
      base_url,
      temperature,
      context_window,
      dimension,
      abilities,
      description,
      created_at,
      updated_at,
      is_active,
      is_owner,
      can_edit,
      can_delete,
      is_shared,
    } = item;
    const validAbilities = abilities === null
      || (Array.isArray(abilities) && abilities.every((ability) => typeof ability === 'string'));

    if (
      !positiveSafeInt(id)
      || typeof model_id !== 'string'
      || typeof category !== 'string'
      || typeof model_provider !== 'string'
      || typeof model_name !== 'string'
      || !nullableString(base_url)
      || !nullableFinite(temperature)
      || !nullableInt(context_window)
      || !nullableInt(dimension)
      || !validAbilities
      || !nullableString(description)
      || !nullableString(created_at)
      || !nullableString(updated_at)
      || typeof is_active !== 'boolean'
      || typeof is_owner !== 'boolean'
      || typeof can_edit !== 'boolean'
      || typeof can_delete !== 'boolean'
      || typeof is_shared !== 'boolean'
    ) {
      return null;
    }

    result.push({
      id,
      model_id,
      category,
      model_provider,
      model_name,
      base_url,
      temperature,
      context_window,
      dimension,
      abilities: abilities === null ? null : [...abilities],
      description,
      created_at,
      updated_at,
      is_active,
      is_owner,
      can_edit,
      can_delete,
      is_shared,
    });
  }

  return result;
}

export function parseUserDefaultModels(value: unknown): UserDefaultModelMap | null {
  if (!Array.isArray(value)) return null;

  const result: UserDefaultModelMap = {};
  for (const item of value) {
    if (
      !isJsonRecord(item)
      || !positiveSafeInt(item.id)
      || !positiveSafeInt(item.user_id)
      || !positiveSafeInt(item.model_id)
      || typeof item.config_type !== 'string'
      || !nullableString(item.created_at)
      || !nullableString(item.updated_at)
    ) {
      return null;
    }

    const models = parseModelList([item.model]);
    if (!models) return null;

    if (isKnownDefaultModelType(item.config_type)) {
      if (result[item.config_type]) return null;

      result[item.config_type] = {
        id: item.id,
        user_id: item.user_id,
        model_id: item.model_id,
        config_type: item.config_type,
        created_at: item.created_at,
        updated_at: item.updated_at,
        model: models[0],
      };
    }
  }

  return result;
}

/**
 * Get all models for current user
 */
export async function getUserModels(options: { category?: string } = {}): Promise<ModelWithAccess[]> {
  const apiUrl = getApiUrl();
  const query = options.category ? `?category=${encodeURIComponent(options.category)}` : '';
  const response = await apiRequest(`${apiUrl}/api/models/${query}`);

  if (!response.ok) {
    throw new Error('Failed to fetch models');
  }
  const parsed = await parseApiResponse(response);
  const models = parseModelList(parsed.data);
  if (!models || (options.category && models.some((model) => model.category !== options.category))) {
    throw new Error('Invalid models response');
  }

  return models;
}

/**
 * Get user's default model configuration
 */
export async function getUserDefaultModels(): Promise<UserDefaultModelMap> {
  const apiUrl = getApiUrl();
  const response = await apiRequest(`${apiUrl}/api/models/user-default`);

  if (!response.ok) {
    throw new Error('Failed to fetch default models');
  }

  const parsed = await parseApiResponse(response);
  const defaults = parseUserDefaultModels(parsed.data);
  if (!defaults) throw new Error('Invalid default models response');

  return defaults;
}

export type TaskLlmSelection =
  | { kind: 'success'; llmIds: [string, string | null, string | null, string | null] }
  | { kind: 'no_model' }
  | { kind: 'operational_error'; error: Error }

export async function resolveTaskLlmSelection(): Promise<TaskLlmSelection> {
  const [models, defaults] = await Promise.allSettled([
    getUserModels({ category: 'llm' }),
    getUserDefaultModels(),
  ]);
  const defaultMap = defaults.status === 'fulfilled' ? defaults.value : null;
  const fallback = models.status === 'fulfilled'
    ? models.value.find((model) => model.model_id)?.model_id
    : undefined;
  const general = defaultMap?.general?.model.model_id || fallback;

  if (general) {
    return {
      kind: 'success',
      llmIds: [
        general,
        defaultMap?.small_fast?.model.model_id || null,
        defaultMap?.visual?.model.model_id || null,
        defaultMap?.compact?.model.model_id || null,
      ],
    };
  }

  if (models.status === 'rejected') {
    return {
      kind: 'operational_error',
      error: models.reason instanceof Error ? models.reason : new Error('Failed to fetch models'),
    };
  }

  if (defaults.status === 'rejected') {
    return {
      kind: 'operational_error',
      error: defaults.reason instanceof Error ? defaults.reason : new Error('Failed to fetch default models'),
    };
  }

  return { kind: 'no_model' };
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
  const [general, smallFast, visual, compact, embedding, video, soundEffect, music] = await Promise.all([
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
    apiRequest(`${apiUrl}/api/models/default/video`)
      .then(res => res.json().catch(() => null)),
    apiRequest(`${apiUrl}/api/models/default/sound_effect`)
      .then(res => res.json().catch(() => null)),
    apiRequest(`${apiUrl}/api/models/default/music`)
      .then(res => res.json().catch(() => null)),
  ]);

  return {
    general,
    small_fast: smallFast,
    visual,
    compact,
    embedding,
    video,
    sound_effect: soundEffect,
    music,
  };
}

export interface Provider {
  id: string;
  name: string;
  description: string;
  category?: string[];
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
  base_url?: string;
  default_base_url?: string;
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
