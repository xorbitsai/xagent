import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"

export interface PersonalApiKeyOwner {
  id: number
  username: string
  email: string | null
}

export interface PersonalApiKeyListItem {
  id: number
  key_prefix: string
  masked_key: string
  status: "active" | "expired" | "revoked"
  revoked_at: string | null
  expires_at: string | null
  created_at: string
  owner: PersonalApiKeyOwner
}

export interface PersonalApiKeyListResponse {
  items: PersonalApiKeyListItem[]
  can_manage_others: boolean
}

export interface PersonalApiKeyCreated {
  id: number
  full_key: string
  key_prefix: string
  created_at: string
  expires_at: string | null
}

export interface PersonalApiKeyRevoked {
  revoked: boolean
  revoked_at: string | null
}

const MANAGEMENT_URL = `${getApiUrl()}/api/personal-api-keys`
const CREATE_URL = `${getApiUrl()}/api/me/personal-keys`

export async function listPersonalApiKeys(): Promise<PersonalApiKeyListResponse> {
  const response = await apiRequest(MANAGEMENT_URL, { method: "GET" })
  if (!response.ok) throw new Error(`Failed to load personal API keys (${response.status})`)
  return (await response.json()) as PersonalApiKeyListResponse
}

export async function createPersonalApiKey(): Promise<PersonalApiKeyCreated> {
  const response = await apiRequest(CREATE_URL, { method: "POST" })
  if (!response.ok) throw new Error(`Failed to create personal API key (${response.status})`)
  return (await response.json()) as PersonalApiKeyCreated
}

export async function revokePersonalApiKey(keyId: number): Promise<PersonalApiKeyRevoked> {
  const response = await apiRequest(`${MANAGEMENT_URL}/${keyId}`, { method: "DELETE" })
  if (!response.ok) throw new Error(`Failed to revoke personal API key (${response.status})`)
  return (await response.json()) as PersonalApiKeyRevoked
}
