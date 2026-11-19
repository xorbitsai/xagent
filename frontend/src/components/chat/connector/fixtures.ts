import type { ConnectorInfo, ResultCardData } from "./types"

/** Sample data for tests and visual checks. */
export const cosmosConnector: ConnectorInfo = {
  interaction_id: "11111111-2222-3333-4444-555555555555",
  app_id: "cosmos",
  name: "Azure Cosmos DB",
  logo: { mark: "◎", bg: "#0078D4", fg: "#fff", icon_url: null },
  route: "mcp",
  what: "databases, containers and documents",
  key_label: "your account endpoint, a read-only key and the database name",
  how: [
    "In the Azure portal open your Cosmos DB account → Settings → Keys.",
    "Copy the URI, and the read-only primary key. Use read-only unless you want me to write.",
    "Note the database you want me to query.",
  ],
  fields: [
    { key: "COSMOS_ENDPOINT", label: "Account endpoint", placeholder: "https://your-account.documents.azure.com:443/", secret: false },
    { key: "COSMOS_KEY", label: "Read-only key", placeholder: "", secret: true },
    { key: "COSMOS_DATABASE", label: "Database", placeholder: "customers", secret: false },
  ],
  probe: { checks: ["native", "mcp"], found: "mcp" },
  agent_name: "Cesar",
}

export const cosmosResultCard: ResultCardData = {
  title: "Query · top customers by revenue",
  source: { name: "Azure Cosmos DB", logo: cosmosConnector.logo },
  items: ["Contoso Ltd · $182,400", "Northwind Traders · $149,950", "Fabrikam Inc · $121,300"],
  footnote: "3 of 1,284 customers · read-only query",
  created_at: new Date().toISOString(),
}
