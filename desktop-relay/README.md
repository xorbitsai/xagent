# Xagent Desktop Relay

Desktop Relay is the macOS companion for Xagent's `desktop_relay` computer
runtime. It connects outbound to Xagent and asks the user to authorize either
one window or one entire display. A window grant remains confined to that
window; a display grant can switch between applications on that display.

## Build

macOS 14 or later and Swift 6 are required.

```bash
cd desktop-relay
swift build -c release
.build/release/xagent-desktop-relay --self-test
```

## First-time pairing

1. Open **Settings > Computer Use > Desktop Computer Relay** in Xagent.
2. Select **Create desktop pairing**, then copy the generated setup JSON.
3. From the `desktop-relay` source directory, run:

```bash
relay_storage_root="${XAGENT_STORAGE_ROOT:-$HOME/.xagent}"
relay_pairing_file="$relay_storage_root/desktop-relay/pairing.json"

mkdir -p "$relay_storage_root/desktop-relay"
umask 077
pbpaste > "$relay_pairing_file"
chmod 600 "$relay_pairing_file"

.build/release/xagent-desktop-relay --setup-file "$relay_pairing_file"
```

The command asks whether to authorize an entire display or one window. A
successful first connection prints:

```text
Desktop Relay configuration saved to .../desktop-relay/config.json
Desktop Relay is running.
```

Desktop Relay then:

- removes the managed one-time `pairing.json`;
- stores only the WebSocket server address in
  `$XAGENT_STORAGE_ROOT/desktop-relay/config.json` (default:
  `~/.xagent/desktop-relay/config.json`);
- stores the reusable session credential in macOS Keychain.

The pairing setup expires after 10 minutes. Generate a new one if it expires.
Avoid `--setup '<json>'` for normal use because inline credentials may be
captured by shell history or process inspection.

## Later launches and changing targets

After the first successful pairing, start the relay without a setup file:

```bash
cd desktop-relay
.build/release/xagent-desktop-relay
```

Select a display or window and leave the process running. To switch the
authorized target, stop the relay with `Control-C`, run the same command again,
and select another target. Re-pairing is not required.

Create a new pairing only after revoking access, losing or expiring the
Keychain session, changing the Xagent server, or moving to another Mac. Repeat
the first-time pairing steps in those cases. The desktop and browser relays use
separate credentials and can run together.

On first launch, macOS prompts for:

- **Screen Recording**, to capture the authorized display or window.
- **Accessibility**, to inspect controls and deliver mouse or keyboard input.

After granting a permission, restart the relay. The command requires an
explicit target selection. For managed launches, a currently shareable target
can instead be selected with `--window-id ID` or `--display-id ID`.

In Xagent, the relay is ready when **Settings > Computer Use** reports that
Desktop Relay is connected, the expected target is attached, and both macOS
permissions are enabled.

## User controls

- `Command-Option-P`: pause or resume agent input.
- `Command-Option-Escape`: emergency stop, clear the target authorization, and
  exit the relay.

Closing or replacing a selected window invalidates a window authorization.
Secure text fields are never exposed to the agent and must be completed by the
user.

## Audio and video capture

The `computer` tool can request a `capture_media` action for either `audio` or
`video`. Each capture:

- requires an explicit user confirmation;
- records only the authorized window or display;
- lasts from 1 to 30 seconds;
- is streamed in bounded, ordered chunks and atomically merged by Xagent;
- is capped at 32 MiB in total;
- becomes a registered task output (`.m4a` for audio or `.mp4` for video).

The final size and SHA-256 checksum are verified before the file is exposed.
Video capture includes available target audio. macOS and protected-content
policies may omit DRM-protected audio or video.

## Xagent configuration

Choose **My computer** when creating the task. That task-bound target takes
precedence over the deployment default, so ordinary UI use does not require
`XAGENT_BROWSER_RUNTIME_KIND`. The runtime keeps the existing `computer` tool
schema and ReAct execution pipeline. Desktop navigation is intentionally
unavailable; applications own their own navigation.
