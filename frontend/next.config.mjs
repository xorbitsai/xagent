import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Output mode:
//   'export'     -> static assets in out/, served by FastAPI in the single-process
//                   (pip / uvx) deployment. This is the default.
//   'standalone' -> self-contained Node server, used by the multi-container Docker
//                   deployment (set NEXT_OUTPUT=standalone in Dockerfile.frontend).
const outputMode = process.env.NEXT_OUTPUT === 'standalone' ? 'standalone' : 'export';

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: outputMode,
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
