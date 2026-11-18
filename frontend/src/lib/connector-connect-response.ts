import type {
  ConnectorConnectResult,
  ConnectorConnectState,
  ConnectorErrorCode,
} from "@/lib/page-extension-contracts"
import { isJsonRecord } from "@/lib/api-wrapper"

const ERROR_CODES: readonly string[] = ["auth_failed", "unreachable", "timeout", "no_tools", "invalid_field"]
const STATES: readonly string[] = ["pending", "connected", "failed", "skipped", "unavailable"]

const str = (value: unknown): string | null =>
  typeof value === "string" && value ? value : null

/** A failed result with nothing else to say (an error the host did not describe). */
export const FAILED_RESULT: ConnectorConnectResult = {
  state: "failed", errorCode: null, errorField: null, secretMasked: null, scope: null,
  agentName: null, tools: [], taskResumed: null,
}

/** The result of an extension that has no host service behind it. */
export const UNAVAILABLE_RESULT: ConnectorConnectResult = { ...FAILED_RESULT, state: "unavailable" }

/**
 * Parses a response body of the connect endpoints; null when it is not one.
 * Keeps only known states and error codes and well-formed tools.
 */
export function parseConnectorConnectResult(data: unknown): ConnectorConnectResult | null {
  if (!isJsonRecord(data) || typeof data.state !== "string" || !STATES.includes(data.state)) return null
  const tools = Array.isArray(data.tools)
    ? data.tools.flatMap((tool) =>
        isJsonRecord(tool) && typeof tool.name === "string"
          ? [{ name: tool.name, label: str(tool.label) ?? tool.name }]
          : [],
      )
    : []
  return {
    state: data.state as ConnectorConnectState,
    errorCode: ERROR_CODES.includes(String(data.error_code)) ? (data.error_code as ConnectorErrorCode) : null,
    errorField: str(data.error_field),
    secretMasked: str(data.secret_masked),
    scope: str(data.scope),
    agentName: str(data.agent_name),
    tools,
    taskResumed: typeof data.task_resumed === "boolean" ? data.task_resumed : null,
  }
}
