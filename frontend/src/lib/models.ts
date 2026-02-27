/**
 * Model management API service
 */

import { getApiUrl } from './utils';
import { apiRequest } from './api-wrapper';

export interface Model {
  id: number;
  name: string;
  model_id: string;
  provider: string;
  model_provider: 'llm' | 'embedding' | 'image';
  api_key?: string;
  base_url?: string;
  max_tokens?: number;
  temperature?: number;
  is_shared: boolean;
  created_by?: number;
  created_at: string;
  updated_at: string;
}

export interface UserDefaultModel {
  id: number;
  user_id: number;
  config_type: 'general' | 'small_fast' | 'visual' | 'compact' | 'embedding';
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
}

/**
 * Get all models for current user
 */
export async function getUserModels(token: string): Promise<Model[]> {
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
export async function getUserDefaultModels(token: string): Promise<DefaultModelConfig> {
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
  token: string,
  configType: 'general' | 'small_fast' | 'visual' | 'compact' | 'embedding',
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
  token: string,
  configType: 'general' | 'small_fast' | 'visual' | 'compact' | 'embedding'
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
export async function getSystemDefaultModels(token: string): Promise<DefaultModelConfig> {
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
  config?: { api_key?: string; base_url?: string }
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
