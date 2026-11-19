import type { ConnectorConnectExtension } from "@/lib/page-extension-contracts"
import { UNAVAILABLE_RESULT } from "@/lib/connector-connect-response"

// Extension point for connecting an app from chat (see the contract in
// page-extension-contracts.ts). The OSS build has no host service, so every
// call answers "unavailable" without any request, and the card says so.
export const saveConnector: ConnectorConnectExtension["saveConnector"] = async () => UNAVAILABLE_RESULT

export const skipConnector: ConnectorConnectExtension["skipConnector"] = async () => UNAVAILABLE_RESULT

export const getConnectorState: ConnectorConnectExtension["getConnectorState"] = async () => UNAVAILABLE_RESULT

export const resumeConnector: ConnectorConnectExtension["resumeConnector"] = async () => UNAVAILABLE_RESULT
