import { beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", () => ({
  getApiUrl: () => "http://api.local",
}))

import { connectMcpServer, disconnectMcpServer, getMcpConnection } from "./mcp-utils"

function jsonResponse(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  })
}

describe("MCP OAuth connect API client", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  it("starts the OAuth connect flow with a POST request", async () => {
    apiRequestMock.mockResolvedValue(
      jsonResponse({ authorization_url: "https://auth.example.com/authorize?state=abc" }),
    )

    const result = await connectMcpServer(42)

    expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/mcp/42/connect", {
      method: "POST",
    })
    expect(result.authorization_url).toBe("https://auth.example.com/authorize?state=abc")
  })

  it("fetches the current connection status with a GET request", async () => {
    apiRequestMock.mockResolvedValue(jsonResponse({ status: "connected" }))

    const result = await getMcpConnection(42)

    expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/mcp/42/connection")
    expect(result.status).toBe("connected")
  })

  it("disconnects with a DELETE request", async () => {
    apiRequestMock.mockResolvedValue(jsonResponse({ status: "not_connected" }))

    await disconnectMcpServer(42)

    expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/mcp/42/connection", {
      method: "DELETE",
    })
  })

  it("surfaces backend error details when the connect request fails", async () => {
    apiRequestMock.mockResolvedValue(
      jsonResponse({ detail: "MCP server not found" }, { status: 404 }),
    )

    await expect(connectMcpServer(99)).rejects.toThrow("MCP server not found")
  })
})
