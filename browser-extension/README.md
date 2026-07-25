# Xagent Browser Relay

This Manifest V3 extension lets Xagent control one Chrome tab that the user
explicitly approves. It never attaches to a tab automatically.

## Development install

```bash
npm install
npm run build
```

Open `chrome://extensions`, enable Developer mode, choose **Load unpacked**,
and select `browser-extension/dist`.

In Xagent Settings, create a one-time browser pairing token. Paste both the
WebSocket URL and token into the extension, connect, then open the tab you want
to use and select **Approve current tab**.

The extension requests Chrome's `debugger` permission because it transports the
same screenshot and input operations used by the provider-neutral Xagent
`computer` tool. Chrome visibly indicates while a tab is being debugged.

Pairing tokens are single-use and expire after ten minutes. The exchanged relay
session stays in `chrome.storage.local` for reconnects. It expires after seven
days or immediately when the relay is revoked from Settings. Without Redis, an
Xagent server restart also invalidates the process-local session. Successfully
pairing a new extension rotates the previous relay session for that user.

Relay coordination uses Redis when `XAGENT_REDIS_URL` is configured, allowing
the WebSocket endpoint and agent execution to run in different processes or
replicas. Without Redis, coordination falls back to process-local memory for
single-process development. Browser commands, screenshots, and DOM observations
travel over ephemeral Redis Pub/Sub channels and are not stored as Redis keys or
streams. Page URLs and titles are also excluded from Redis connection metadata.
