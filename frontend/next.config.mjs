import path from "node:path";
import { fileURLToPath } from "node:url";

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
  images: { unoptimized: true },
  outputFileTracingRoot: __dirname,
  experimental: {
    optimizeCss: false,
  },
  // 确保CSS正确处理
  compiler: {
    removeConsole: false,
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
