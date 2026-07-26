# Xagent Browser Relay

This Manifest V3 extension lets Xagent control one Chrome tab that the user
explicitly approves. It never attaches to a tab automatically.

## Install a packaged build

```bash
npm install
npm run package
```

The command creates:

- `artifacts/xagent-browser-relay-<version>.zip`
- `artifacts/xagent-browser-relay-<version>.zip.sha256`

Verify the checksum, unzip the archive, open `chrome://extensions`, enable
Developer mode, choose **Load unpacked**, and select the unzipped directory.
CI publishes the same ZIP and checksum as the `xagent-browser-relay` artifact.

For development, run `npm run build` and load `browser-extension/dist`.

## Connect Xagent

1. Open **Settings → Computer Use → User Browser Relay** in Xagent.
2. Choose **Create pairing token**, then **Copy pairing setup**.
3. Open the extension, paste the JSON setup, and choose **Pair and connect**.
4. Open the browser tab Xagent should use and choose
   **Approve current tab** from the extension.

The extension badge displays `ON` only while the relay is connected and a tab
is approved. `...` means it is connecting; `!` means an approved tab is
temporarily offline. Transient failures use bounded exponential backoff and a
Chrome alarm so reconnects survive Manifest V3 service-worker suspension.

## Audio and video capture

The `computer` tool can request a confirmed `capture_media` action from the
approved tab. Audio is returned as WebM/Opus; video is returned as WebM with
available tab audio. Captures last from 1 to 30 seconds. Relay sends bounded,
ordered chunks; Xagent verifies their final size and SHA-256 checksum, merges
them atomically, and registers one task output file. The complete capture is
capped at 32 MiB. Chrome or the site may block capture of protected media.

## Login and manual takeover

Xagent can use the approved tab's existing login session. It does not read the
browser's cookie or password stores. When the page requires a password,
one-time code, passkey, CAPTCHA, or payment field, the computer policy pauses
the agent:

1. Enter the sensitive value directly in the approved tab.
2. Complete the sensitive step yourself.
3. Return to Xagent and confirm that it may continue.

Xagent captures a fresh screenshot before continuing. An approved action is
not executed if the page, target, browser session, or worker changed while it
was waiting.

## Development install

```bash
npm install
npm run check
```

Open `chrome://extensions`, enable Developer mode, choose **Load unpacked**,
and select `browser-extension/dist`.

The extension requests Chrome's `debugger` permission because it transports the
same screenshot and input operations used by the provider-neutral Xagent
`computer` tool. Chrome visibly indicates while a tab is being debugged.

Pairing tokens are single-use and expire after ten minutes. The exchanged relay
session stays in `chrome.storage.local` for reconnects. It expires after seven
days or immediately when the relay is revoked from Settings. Without Redis, an
Xagent server restart also invalidates the process-local session. Successfully
pairing a new extension rotates the previous relay session for that user.
Forgetting the relay or receiving an invalid/revoked session detaches the
approved tab and removes the stored session token.

Relay coordination uses Redis when `XAGENT_REDIS_URL` is configured, allowing
the WebSocket endpoint and agent execution to run in different processes or
replicas. Without Redis, coordination falls back to process-local memory for
single-process development. Browser commands, screenshots, and DOM observations
travel over ephemeral Redis Pub/Sub channels and are not stored as Redis keys or
streams. Page URLs and titles are also excluded from Redis connection metadata.

Run the headed login-takeover acceptance test locally with:

```bash
uv run playwright install chromium
XAGENT_RUN_BROWSER_EXTENSION_E2E=1 \
  uv run pytest -q tests/e2e/test_browser_extension_login_takeover.py
```

Set `XAGENT_BROWSER_EXTENSION_CHROMIUM_PATH` to an existing Chromium executable
when the Playwright-managed browser is installed in a non-default location.
