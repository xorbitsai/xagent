// Cloudflare Worker backing https://get.xagent.co
//
// Serves scripts/install.sh from the xagent repo so users can run:
//
//   curl -fsSL https://get.xagent.co | sh
//
// The script is pinned to the latest GitHub *release tag*, so the public
// one-liner only ever serves a shipped, immutable version. It fails closed:
// if the release can't be resolved or the tag doesn't contain the script, it
// returns 502 rather than falling back to a floating ref like `main` — a public
// `curl | sh` endpoint must never serve unreleased code. (This means the
// endpoint only works once a release that includes scripts/install.sh exists.)
// Deploy with `wrangler deploy`.

const REPO = "xorbitsai/xagent";
const SCRIPT_PATH = "scripts/install.sh";
// Cache the resolved script at the edge to avoid hitting GitHub on every hit.
const CACHE_TTL_SECONDS = 300;

async function latestReleaseTag() {
  // Never throw: on any API failure return null so the caller fails closed.
  try {
    const res = await fetch(`https://api.github.com/repos/${REPO}/releases/latest`, {
      headers: { "User-Agent": "get.xagent.co", Accept: "application/vnd.github+json" },
      cf: { cacheTtl: CACHE_TTL_SECONDS, cacheEverything: true },
      signal: AbortSignal.timeout(5000), // don't hang the client on a stuck upstream
    });
    if (!res.ok) return null;
    const data = await res.json();
    return typeof data.tag_name === "string" && data.tag_name ? data.tag_name : null;
  } catch {
    return null;
  }
}

async function fetchScript(ref) {
  const url = `https://raw.githubusercontent.com/${REPO}/${ref}/${SCRIPT_PATH}`;
  return fetch(url, {
    headers: { "User-Agent": "get.xagent.co" },
    cf: { cacheTtl: CACHE_TTL_SECONDS, cacheEverything: true },
    signal: AbortSignal.timeout(10000), // a timeout here throws -> handler returns 502
  });
}

const UNAVAILABLE = () =>
  new Response("# Xagent installer temporarily unavailable\n", {
    status: 502,
    headers: { "content-type": "text/plain; charset=utf-8" },
  });

export default {
  async fetch(request) {
    // Only the root path serves the installer; ignore /favicon.ico etc.
    if (new URL(request.url).pathname !== "/") {
      return new Response("Not Found\n", {
        status: 404,
        headers: { "content-type": "text/plain; charset=utf-8" },
      });
    }

    try {
      // Fail closed: only ever serve a resolved, immutable release tag.
      const ref = await latestReleaseTag();
      if (!ref) return UNAVAILABLE();

      const res = await fetchScript(ref);
      if (!res.ok) return UNAVAILABLE();

      const body = await res.text();
      return new Response(body, {
        status: 200,
        headers: {
          // text/plain so `curl | sh` gets the raw script, never rendered HTML.
          "content-type": "text/plain; charset=utf-8",
          "cache-control": `public, max-age=${CACHE_TTL_SECONDS}`,
          "x-xagent-install-ref": ref,
        },
      });
    } catch {
      // Any unexpected error → clean 502 text, never a Cloudflare HTML 500
      // (which would break a piped `curl | sh`).
      return UNAVAILABLE();
    }
  },
};
