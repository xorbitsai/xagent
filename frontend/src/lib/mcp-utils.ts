/**
 * Shared utilities for MCP Server and Custom API configurations
 */

export function isValidMcpName(name: string): boolean {
    const nameRegex = /^[a-zA-Z0-9_-]+$/;
    return nameRegex.test(name.trim());
}

export function buildCustomApiPayload(
    mcpFormData: Record<string, any>,
    customApiEnv: { key: string; value: string }[]
): { isValid: boolean; payload?: any; errorKey?: string } {
    const validEnv = customApiEnv.filter(env => env.key.trim() && env.value.trim());

    if (validEnv.length === 0) {
        return { isValid: false, errorKey: 'tools.mcp.alerts.atLeastOneSecret' };
    }

    const envObj: Record<string, string> = {};
    validEnv.forEach(env => {
        envObj[env.key.trim()] = env.value.trim();
    });

    // Custom API payload structure expects env at top level, no config/transport
    const payload = { ...mcpFormData };
    payload.env = envObj;
    delete payload.config;
    delete payload.transport;

    return { isValid: true, payload };
}
