# Xagent Desktop Relay

Desktop Relay is the macOS companion for Xagent's `desktop_relay` computer
runtime. It connects outbound to Xagent, asks the user to authorize exactly one
window, captures only that window, and sends input only while the authorization
is active.

## Build and run

macOS 14 or later and Swift 6 are required.

```bash
cd desktop-relay
swift build -c release
.build/release/xagent-desktop-relay --self-test
.build/release/xagent-desktop-relay --setup-file /path/to/pairing.json
```

Create the one-time pairing JSON from **Settings > Computer Use** in Xagent.
Keep it outside the source tree and pass it to the first launch with
`--setup-file`. After pairing succeeds, Desktop Relay stores only the server
WebSocket address in
`$XAGENT_STORAGE_ROOT/desktop-relay/config.json` (default:
`~/.xagent/desktop-relay/config.json`) and stores the session credential in
macOS Keychain. Later launches need no setup argument:

```bash
.build/release/xagent-desktop-relay
```

A pairing file placed in the managed desktop-relay configuration directory is
removed after successful pairing. The desktop and browser relays use separate
credentials and can run together.

On first launch, macOS prompts for:

- **Screen Recording**, to capture the authorized window.
- **Accessibility**, to inspect controls and deliver mouse or keyboard input.

After granting a permission, restart the relay. The command prints all
shareable windows and requires an explicit selection. For managed launches, a
currently shareable window can instead be selected with `--window-id ID`.

## User controls

- `Command-Option-P`: pause or resume agent input.
- `Command-Option-Escape`: emergency stop, clear the window authorization, and
  exit the relay.

Closing or replacing the selected window invalidates authorization. Secure text
fields are never exposed to the agent and must be completed by the user.

## Xagent configuration

Choose **My computer** when creating the task. That task-bound target takes
precedence over the deployment default, so ordinary UI use does not require
`XAGENT_BROWSER_RUNTIME_KIND`. The runtime keeps the existing `computer` tool
schema and ReAct execution pipeline. Desktop navigation is intentionally
unavailable; applications own their own navigation.
