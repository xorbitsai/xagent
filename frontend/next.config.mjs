import path from "node:path";
import { fileURLToPath } from "node:url";
import { browserSecurityHeaders } from "./browser-security.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Output mode (build only):
//   'export'     -> static assets in out/, served by FastAPI in the single-process
//                   (pip / uvx) deployment. This is the default for builds.
//   'standalone' -> self-contained Node server, used by the multi-container Docker
//                   deployment (set NEXT_OUTPUT=standalone in Dockerfile.frontend).
//
// Never set an output mode in `next dev`: 'export' forces dynamicParams=false, so
// navigating to a real dynamic route (/task/<id>, /agent/<id>, ...) whose id is
// not in generateStaticParams throws. In production the FastAPI SPA fallback maps
// those paths to the __shell__ page; the dev server has no such layer.
const isDev = process.env.NODE_ENV === "development";
const outputMode = isDev
  ? undefined
  : process.env.NEXT_OUTPUT === "standalone"
    ? "standalone"
    : "export";

/** @type {import('next').NextConfig} */
const nextConfig = {
  ...(outputMode ? { output: outputMode } : {}),
  // Exported pages receive the equivalent policy from FastAPI. Next headers
  // only apply when the independent Node server actually serves the pages.
  ...(outputMode !== "export" ? {
    async headers() {
      return [
        {
          source: "/:path((?!widget(?:/|$)).*)",
          headers: browserSecurityHeaders({ development: isDev }),
        },
        {
          source: "/widget/:path*",
          headers: browserSecurityHeaders({ embeddable: true, development: isDev }),
        },
      ];
    },
  } : {}),
  images: { unoptimized: true },
  outputFileTracingRoot: __dirname,
  experimental: {
    optimizeCss: false,
  },
  // Strip debug logging from production bundles. The embedded widget runs on
  // third-party sites, and the chat context logs whole WebSocket frames (user
  // messages, agent output, tool arguments and results) into their console.
  // Next hands this straight to SWC with no development guard of its own
  // (build/swc/options.js), so `next dev` has to be excluded explicitly for
  // local debugging to keep its log/debug/info/trace output.
  compiler: {
    removeConsole: isDev ? false : { exclude: ["error", "warn"] },
  },
  // 解决开发模式错误
  reactStrictMode: true,
  devIndicators: {
    position: 'bottom-right',
  },
  typescript: {
    ignoreBuildErrors: false,
  },
  eslint: {
    ignoreDuringBuilds: false,
  },
};

export default nextConfig;
