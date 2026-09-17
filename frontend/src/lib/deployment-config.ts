import { apiRequest } from "@/lib/api-wrapper"
import { resolveApiSnippetBaseUrl } from "@/lib/api-snippet-target"
import { getApiUrl } from "@/lib/utils"

export const DEPLOYMENT_CONFIG_LOAD_FAILED_FALLBACK =
  "Failed to load deployment configuration. Retry before copying deployment details."

export interface DeploymentConfig {
  /**
   * Origin external API and widget clients should call. Hosting layers may
   * advertise a region-pinned origin that differs from the owner's browser.
   */
  deployment_origin: string | null
  /** Canonical application base URL used for browser share links. */
  app_origin: string | null
  /**
   * Optional routing region. When present, share links first visit the
   * canonical region bootstrap so a new recipient does not need prior state.
   */
  region: string | null
}

let deploymentConfigRequest: Promise<DeploymentConfig> | null = null

async function loadDeploymentConfig(): Promise<DeploymentConfig> {
  const response = await apiRequest(`${getApiUrl()}/api/deployment-config`)
  if (!response.ok) {
    throw new Error("Failed to load deployment configuration")
  }
  return parseDeploymentConfig(await response.json())
}

function parseDeploymentConfig(value: unknown): DeploymentConfig {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Invalid deployment configuration")
  }

  const candidate = value as Record<string, unknown>
  if (
    !isNullableOrigin(candidate.deployment_origin)
    || !isNullableAppBaseUrl(candidate.app_origin)
    || !isNullableString(candidate.region)
  ) {
    throw new Error("Invalid deployment configuration")
  }

  return {
    deployment_origin: candidate.deployment_origin,
    app_origin: candidate.app_origin,
    region: candidate.region,
  }
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === "string"
}

function isNullableOrigin(value: unknown): value is string | null {
  return isNullableHttpUrl(value, false)
}

function isNullableAppBaseUrl(value: unknown): value is string | null {
  return isNullableHttpUrl(value, true)
}

function isNullableHttpUrl(
  value: unknown,
  allowPath: boolean,
): value is string | null {
  if (value === null) return true
  if (typeof value !== "string") return false

  try {
    const url = new URL(value)
    return (
      (url.protocol === "http:" || url.protocol === "https:")
      && !url.username
      && !url.password
      && (allowPath || url.pathname === "/")
      && !url.search
      && !url.hash
    )
  } catch {
    return false
  }
}

/**
 * Load deployment targets once per browser session.
 *
 * The value is immutable for a deployed frontend. Reusing the same promise
 * deduplicates consumers that mount together; a failed request is deliberately
 * evicted so a later dialog open can recover from a transient network error.
 */
export function fetchDeploymentConfig(): Promise<DeploymentConfig> {
  if (!deploymentConfigRequest) {
    deploymentConfigRequest = loadDeploymentConfig().catch((error) => {
      deploymentConfigRequest = null
      throw error
    })
  }
  return deploymentConfigRequest
}

/**
 * Clear the session memo so tests can isolate request outcomes.
 *
 * Production code must not call this function: deployment configuration is
 * immutable for the lifetime of a loaded frontend, and failed requests already
 * evict themselves. Tests need an explicit reset because Vitest reuses this
 * module instance across cases.
 */
export function __resetDeploymentConfigCache(): void {
  deploymentConfigRequest = null
}

/**
 * Resolve an external-client origin only after deployment configuration has
 * loaded. A configured object with no override represents standalone XAgent,
 * where the browser origin remains the correct target.
 */
export function resolveDeploymentOrigin(
  config: DeploymentConfig | null,
  browserOrigin: string,
): string {
  if (!config) return ""
  return resolveApiSnippetBaseUrl(
    config.deployment_origin || browserOrigin,
    browserOrigin,
  )
}

/**
 * Build a public share URL without exposing the full application on a hidden
 * regional origin.
 *
 * Standalone deployments use their ordinary application URL. Multi-region
 * hosts advertise a region and canonical application base URL, causing the
 * recipient to establish routing state through `/change-region` before the
 * browser opens `/share/<token>`.
 */
export function buildDeploymentShareUrl(
  shareToken: string,
  config: DeploymentConfig | null,
  browserOrigin: string,
): string {
  if (!shareToken || !config) return ""

  const appOrigin = resolveApiSnippetBaseUrl(
    config.app_origin || browserOrigin,
    browserOrigin,
  )
  if (!appOrigin) return ""

  const shareUrl = appendDeploymentPath(
    appOrigin,
    `share/${encodeURIComponent(shareToken)}`,
  )
  if (!config.region) {
    return shareUrl.toString()
  }

  const bootstrap = appendDeploymentPath(appOrigin, "change-region")
  bootstrap.searchParams.set("region", config.region)
  bootstrap.searchParams.set("next", shareUrl.pathname)
  return bootstrap.toString()
}

/** Append an application route without discarding a configured path prefix. */
function appendDeploymentPath(baseUrl: string, path: string): URL {
  const url = new URL(baseUrl)
  const basePath = url.pathname.replace(/\/+$/, "")
  url.pathname = `${basePath}/${path}`
  return url
}
