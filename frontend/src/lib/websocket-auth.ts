/** Browser WebSocket APIs cannot set Authorization. Carry login credentials
 * in the upgrade header, never the URL or a negotiated response protocol. */
export function websocketAuthProtocols(token: string): string[] {
  if (!/^[A-Za-z0-9._~-]{1,8192}$/.test(token)) {
    throw new Error("Invalid WebSocket credential")
  }
  return ["xagent-auth-v1", `xagent-auth-token.${token}`]
}
