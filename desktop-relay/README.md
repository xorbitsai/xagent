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
.build/release/xagent-desktop-relay --setup-file pairing.json
```

Create `pairing.json` from **Settings > Desktop Computer Relay** in Xagent. The
one-time token is exchanged for a session credential stored in macOS Keychain.
The desktop and browser relays use separate credentials and can run together.

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

Set the runtime before starting the Xagent backend:

```bash
XAGENT_BROWSER_RUNTIME_KIND=desktop_relay
```

The runtime keeps the existing `computer` tool schema and ReAct execution
pipeline. Desktop navigation is intentionally unavailable; applications own
their own navigation.
