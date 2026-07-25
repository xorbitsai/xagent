import assert from "node:assert/strict"
import test from "node:test"

import {
  MAX_RECONNECT_DELAY_MS,
  normalizeRelayUrl,
  parsePairingSetup,
  parseServerMessage,
  reconnectDelayMs,
} from "../.test-build/protocol.js"

test("pairing setup accepts the Xagent Settings payload", () => {
  assert.deepEqual(
    parsePairingSetup(
      JSON.stringify({
        websocket_url: "wss://xagent.example/ws/browser-relay",
        pairing_token: "pair-once",
      }),
    ),
    {
      relayUrl: "wss://xagent.example/ws/browser-relay",
      pairingToken: "pair-once",
    },
  )
})

test("pairing setup rejects credentials and query parameters in relay URLs", () => {
  assert.throws(
    () =>
      parsePairingSetup(
        JSON.stringify({
          relayUrl: "wss://user:secret@xagent.example/ws?token=leak",
          pairingToken: "pair-once",
        }),
      ),
    /must not include credentials/,
  )
  assert.throws(() => normalizeRelayUrl("https://xagent.example"), /ws:\/\//)
})

test("server messages enforce the relay protocol version", () => {
  assert.equal(
    parseServerMessage(
      JSON.stringify({ type: "pong", protocol_version: 1 }),
    ).type,
    "pong",
  )
  assert.throws(
    () =>
      parseServerMessage(
        JSON.stringify({ type: "pong", protocol_version: 2 }),
      ),
    /version mismatch/,
  )
})

test("reconnect delay uses bounded exponential backoff with jitter", () => {
  assert.equal(reconnectDelayMs(1, 0), 800)
  assert.equal(reconnectDelayMs(1, 1), 1_200)
  assert.equal(reconnectDelayMs(4, 0.5), 8_000)
  assert.equal(reconnectDelayMs(100, 1), MAX_RECONNECT_DELAY_MS)
})
