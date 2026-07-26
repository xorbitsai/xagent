import { defineConfig, mergeConfig } from "vitest/config"
import baseConfig from "./vitest.config"

const widgetConfig = mergeConfig(baseConfig, defineConfig({
  test: {
    coverage: {
      provider: "v8",
      include: [
        "public/widget.js",
        "src/components/widget/public-agent-chat-page.tsx",
      ],
      reporter: ["text", "json-summary"],
      reportsDirectory: "coverage/widget",
      thresholds: {
        // Keep a practical per-file floor while failing if either shared
        // widget surface loses most of its regression coverage.
        perFile: true,
        statements: 80,
        branches: 55,
        functions: 45,
        lines: 80,
      },
    },
  },
}))

export default defineConfig({
  ...widgetConfig,
  test: {
    ...widgetConfig.test,
    // Vite concatenates arrays during merge, so replace the base glob after
    // merging to keep this command strictly targeted.
    include: [
      "src/components/widget/widget-bootstrap.test.ts",
      "src/components/widget/public-agent-chat-page.test.tsx",
    ],
  },
})
