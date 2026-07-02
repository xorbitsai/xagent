/**
 * Shared utilities for MCP Server and Custom API configurations
 */

import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"

export type McpConnectionStatus = "not_connected" | "pending" | "connected" | "expired" | "error"

export interface McpConnectionResponse {
    status: McpConnectionStatus
}

export interface McpConnectResponse {
    authorization_url: string
}

/**
 * Start the OAuth connect flow for a remote MCP server (auth type `oauth_mcp`).
 * Returns the authorization URL to open (typically in a popup).
 */
export async function connectMcpServer(serverId: number | string): Promise<McpConnectResponse> {
    const response = await apiRequest(`${getApiUrl()}/api/mcp/${serverId}/connect`, {
        method: "POST",
    })
    if (!response.ok) {
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || "Failed to start MCP OAuth connection")
    }
    return response.json()
}

/**
 * Get the current per-user OAuth connection status for a remote MCP server.
 */
export async function getMcpConnection(serverId: number | string): Promise<McpConnectionResponse> {
    const response = await apiRequest(`${getApiUrl()}/api/mcp/${serverId}/connection`)
    if (!response.ok) {
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || "Failed to fetch MCP connection status")
    }
    return response.json()
}

/**
 * Disconnect the current user's OAuth connection for a remote MCP server.
 */
export async function disconnectMcpServer(serverId: number | string): Promise<void> {
    const response = await apiRequest(`${getApiUrl()}/api/mcp/${serverId}/connection`, {
        method: "DELETE",
    })
    if (!response.ok) {
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || "Failed to disconnect MCP server")
    }
    await response.json().catch(() => undefined)
}

export function isValidMcpName(name: string): boolean {
    const nameRegex = /^[a-zA-Z0-9_-]+$/;
    return nameRegex.test(name.trim());
}

export function buildCustomApiPayload(
    mcpFormData: Record<string, any>,
    customApiEnv: { key: string; value: string }[]
): { isValid: boolean; payload?: any; errorKey?: string } {
    const validEnv = customApiEnv.filter(env => env.key.trim() && env.value.trim());

    // Allow empty env for Custom APIs that don't need authentication
    const envObj: Record<string, string> | null = validEnv.length > 0 ? {} : null;

    if (envObj) {
        validEnv.forEach(env => {
            envObj[env.key.trim()] = env.value.trim();
        });
    }

    // Custom API payload structure expects env at top level, no config/transport
    const payload = { ...mcpFormData };
    payload.env = envObj || {}; // Send empty object to clear env
    delete payload.config;
    delete payload.transport;

    return { isValid: true, payload };
}
