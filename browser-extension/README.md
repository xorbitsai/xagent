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
session stays in `chrome.storage.local` for reconnects, but it is invalidated by
an Xagent server restart or by revoking the relay from Settings.

The current relay registry is process-local. The WebSocket endpoint and the
agent execution must therefore run in the same Xagent process; distributed
workers will require a shared relay backend in a later phase.
