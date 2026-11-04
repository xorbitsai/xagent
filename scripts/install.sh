#!/bin/sh
# Xagent installer — https://get.xagent.co
#
#   curl -fsSL https://get.xagent.co | sh
#
# Installs the `xagent-ai` package (backend + bundled web UI) as an isolated uv
# tool, so nothing touches your system Python and PEP 668 never bites. On
# success the `xagent` command is available; start it and open the browser.
#
# Options (environment variables):
#   XAGENT_VERSION   pin a specific version, e.g. XAGENT_VERSION=0.6.0
#   XAGENT_SKIP_BROWSER_INSTALL=1
#                    skip the Playwright Chromium browser download
#   XAGENT_SKIP_ROUTER_INSTALL=1
#                    skip the OpenRouter "auto" model routing runtime
#
# Prefer not to pipe curl into sh? The equivalent manual install is:
#   uv tool install 'xagent-ai[browser,router]'
#   "$(uv tool dir)/xagent-ai/bin/python" -m playwright install chromium
#   # Or, in a virtualenv: pip install 'xagent-ai[browser,router]'
#   # followed by: python -m playwright install chromium
set -eu

# The user's PATH before this script mutates it (below, when bootstrapping uv).
# Used at the end to warn correctly about whether the parent shell will find the
# installed command.
ORIG_PATH="$PATH"

APP="xagent-ai"
CMD="xagent"

info() { printf '\033[1;34m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$1" >&2; }
err() {
  printf '\033[1;31merror:\033[0m %s\n' "$1" >&2
  exit 1
}
is_truthy() {
  case "${1:-}" in
    1 | [Tt][Rr][Uu][Ee] | [Yy][Ee][Ss] | [Oo][Nn]) return 0 ;;
    *) return 1 ;;
  esac
}

# uv supports Linux and macOS. Windows users should use pip in a venv.
os="$(uname -s)"
case "$os" in
  Linux | Darwin) ;;
  *) err "Unsupported OS '$os'. On Windows, install with: pip install $APP (in a virtualenv)." ;;
esac

# Ensure uv is available (isolates the install; avoids system-Python/PEP 668).
if ! command -v uv >/dev/null 2>&1; then
  info "Installing uv (Python tool manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installs into ~/.local/bin (or ~/.cargo/bin on older installers); make it
  # visible to the rest of this script without requiring a new shell.
  for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    [ -d "$d" ] && PATH="$d:$PATH"
  done
  export PATH
fi
command -v uv >/dev/null 2>&1 || err "uv not found on PATH after install; open a new shell and re-run."

extras="browser,router"
if is_truthy "${XAGENT_SKIP_ROUTER_INSTALL:-}"; then
  extras="browser"
  warn "Skipping OpenRouter auto-routing runtime (XAGENT_SKIP_ROUTER_INSTALL is set)."
fi

spec="${APP}[$extras]"
if [ -n "${XAGENT_VERSION:-}" ]; then
  # Strip a leading 'v' (e.g. v0.6.0 -> 0.6.0) so a git-tag-style value works.
  version="${XAGENT_VERSION#v}"
  [ -n "$version" ] || err "XAGENT_VERSION='$XAGENT_VERSION' is not a valid version."
  spec="${spec}==$version"
fi

info "Installing $spec ..."
uv tool install --upgrade "$spec"

if is_truthy "${XAGENT_SKIP_BROWSER_INSTALL:-}"; then
  warn "Skipping Playwright Chromium installation (XAGENT_SKIP_BROWSER_INSTALL is set)."
else
  tool_python="$(uv tool dir)/$APP/bin/python"
  [ -x "$tool_python" ] || err "Xagent tool Python not found at '$tool_python'."
  info "Installing Playwright Chromium browser..."
  # xagent is already installed and usable at this point; the browser binary is
  # an optional enhancement for browser-enabled tasks. Don't let a transient
  # download failure abort the whole install (and swallow the "Next steps"
  # message) — warn and continue so the user can retry manually.
  if ! "$tool_python" -m playwright install chromium; then
    warn "Playwright Chromium download failed. Xagent is installed, but browser-enabled tasks need it."
    warn "Retry with: \"$tool_python\" -m playwright install chromium"
    warn "Or skip it on re-run with XAGENT_SKIP_BROWSER_INSTALL=1."
  fi
fi

printf '\n'
info "Installed. Next steps:"
printf '\n'
printf '  Start Xagent:   %s\n' "$CMD"
printf '  Open:           http://127.0.0.1:8000\n'
printf '  Configure an LLM key (e.g. OPENAI_API_KEY) via a .env file or env var.\n'
printf '\n'

if ! PATH="$ORIG_PATH" command -v "$CMD" >/dev/null 2>&1; then
  warn "'$CMD' is not on your PATH in this shell yet."
  if PATH="$ORIG_PATH" command -v uv >/dev/null 2>&1; then
    warn "Run 'uv tool update-shell' and open a new terminal, then run '$CMD'."
  else
    # uv was just installed by this script and isn't on the parent shell's PATH.
    warn "Open a new terminal, or run: export PATH=\"\$HOME/.local/bin:\$PATH\""
  fi
fi
