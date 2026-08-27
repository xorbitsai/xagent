import { defineConfig, mergeConfig } from "vitest/config"
import baseConfig from "./vitest.config"

const widgetConfig = mergeConfig(baseConfig, defineConfig({
  test: {
    coverage: {
      provider: "v8",
      include: [
        "public/widget.js",
        "src/app/widget/chat/[[]token[]]/page-client.tsx",
        "src/components/chat/ChatInput.tsx",
        "src/components/chat/ChatMessage.tsx",
        "src/components/chat/clarification-form.tsx",
        "src/components/chat/TraceEventRenderer.tsx",
        "src/components/file/file-preview-content.tsx",
        "src/components/file/file-viewer.tsx",
        "src/components/file/inline-file-preview.tsx",
        "src/components/file/pptx-preview-renderer.tsx",
        "src/components/task/task-conversation-panel.tsx",
        "src/components/ui/markdown-renderer.tsx",
        "src/components/widget/public-agent-chat-page.tsx",
        "src/components/widget/session-agent-chat-page.tsx",
        "src/components/widget/use-widget-session.ts",
        "src/components/widget/widget-chrome-controls.tsx",
        "src/contexts/app-context-chat.tsx",
        "src/contexts/auth-context.tsx",
        "src/contexts/file-access-context.tsx",
        "src/hooks/use-file-mention.ts",
        "src/hooks/use-websocket.ts",
        "src/lib/api-wrapper.ts",
        "src/lib/auth-cache.ts",
        "src/lib/files-disabled-presentation.ts",
        "src/contexts/presentation-capabilities.tsx",
        "src/app/settings/page.tsx",
        "src/components/layout/sidebar.tsx",
        "src/components/pages/login.tsx",
        "src/components/pages/oidc-callback.tsx",
      ],
      reporter: ["text", "json-summary"],
      reportsDirectory: "coverage/widget",
      thresholds: {
        // Every included owner has an explicit floor below. Vitest applies
        // numeric top-level thresholds to every file when perFile is enabled,
        // so keeping a global floor here would override the intentional floors
        // for large shared owners.
        perFile: true,

        // The original widget surfaces retain their existing contract floors.
        "public/widget.js": {
          statements: 95, branches: 90, functions: 95, lines: 95,
        },
        "src/components/widget/public-agent-chat-page.tsx": {
          statements: 80, branches: 55, functions: 45, lines: 80,
        },

        // Dedicated Session and presentation owners have focused regression
        // suites, so retain stronger per-file floors for those contracts.
        "src/app/widget/chat/[[]token[]]/page-client.tsx": {
          statements: 90, branches: 75, functions: 90, lines: 90,
        },
        "src/components/widget/session-agent-chat-page.tsx": {
          statements: 90, branches: 85, functions: 75, lines: 90,
        },
        "src/components/widget/use-widget-session.ts": {
          statements: 95, branches: 80, functions: 90, lines: 95,
        },
        "src/components/widget/widget-chrome-controls.tsx": {
          statements: 95, branches: 90, functions: 95, lines: 95,
        },
        "src/lib/files-disabled-presentation.ts": {
          statements: 85, branches: 80, functions: 90, lines: 85,
        },
        "src/lib/auth-cache.ts": { statements: 90, branches: 80, functions: 90, lines: 90 },
        "src/contexts/presentation-capabilities.tsx": {
          statements: 100, branches: 100, functions: 100, lines: 100,
        },
        "src/contexts/file-access-context.tsx": {
          statements: 85, branches: 75, functions: 85, lines: 85,
        },

        // Large shared owners have broad behavior outside this lane. Keep an
        // explicit nonzero regression floor instead of weakening widget floors
        // globally or excluding the owners from required CI.
        "src/contexts/app-context-chat.tsx": {
          statements: 40, branches: 60, functions: 60, lines: 40,
        },
        "src/hooks/use-websocket.ts": { statements: 80, branches: 75, functions: 65, lines: 80 },
        "src/lib/api-wrapper.ts": { statements: 75, branches: 65, functions: 60, lines: 75 },
        "src/contexts/auth-context.tsx": { statements: 70, branches: 65, functions: 90, lines: 70 },
        "src/app/settings/page.tsx": { statements: 75, branches: 50, functions: 50, lines: 75 },
        "src/components/layout/sidebar.tsx": { statements: 35, branches: 40, functions: 10, lines: 35 },
        "src/components/pages/login.tsx": { statements: 85, branches: 55, functions: 60, lines: 85 },
        "src/components/pages/oidc-callback.tsx": { statements: 75, branches: 45, functions: 95, lines: 75 },
        "src/components/chat/ChatInput.tsx": { statements: 60, branches: 60, functions: 40, lines: 60 },
        "src/components/chat/ChatMessage.tsx": { statements: 50, branches: 50, functions: 40, lines: 50 },
        "src/components/chat/clarification-form.tsx": {
          statements: 76, branches: 68, functions: 62, lines: 76,
        },
        "src/components/chat/TraceEventRenderer.tsx": {
          statements: 80, branches: 75, functions: 75, lines: 80,
        },
        "src/components/file/file-preview-content.tsx": {
          statements: 70, branches: 50, functions: 55, lines: 70,
        },
        "src/components/file/file-viewer.tsx": {
          statements: 70, branches: 55, functions: 80, lines: 70,
        },
        "src/components/file/inline-file-preview.tsx": {
          statements: 70, branches: 55, functions: 60, lines: 70,
        },
        "src/components/file/pptx-preview-renderer.tsx": {
          statements: 45, branches: 35, functions: 30, lines: 45,
        },
        "src/components/task/task-conversation-panel.tsx": {
          statements: 80, branches: 70, functions: 60, lines: 80,
        },
        "src/components/ui/markdown-renderer.tsx": { statements: 65, branches: 65, functions: 75, lines: 65 },
        "src/hooks/use-file-mention.ts": { statements: 60, branches: 60, functions: 60, lines: 60 },
      },
    },
  },
}))

export default defineConfig({
  ...widgetConfig,
  test: {
    ...widgetConfig.test,
    // Vite concatenates arrays during merge, so replace the base glob after
    // merging to keep this required lane targeted to the production paths
    // exercised by the widget, Session, and browser-auth contracts.
    include: [
      "src/app/layout.test.tsx",
      "src/app/widget/chat/[token]/page-client.test.tsx",
      "src/app/settings/page.test.tsx",
      "src/components/chat/ChatInput.test.tsx",
      "src/components/chat/chat-input-public-file-access.test.tsx",
      "src/components/chat/ChatMessage.test.tsx",
      "src/components/chat/TraceEventRenderer.test.tsx",
      "src/components/chat/clarification-form.test.tsx",
      "src/components/file/file-preview-content.test.tsx",
      "src/components/file/file-viewer.test.tsx",
      "src/components/file/inline-file-preview.test.tsx",
      "src/components/file/pptx-preview-renderer.test.tsx",
      "src/components/layout/sidebar.test.tsx",
      "src/components/pages/login.test.tsx",
      "src/components/pages/oidc-callback.test.tsx",
      "src/components/task/task-conversation-panel.test.tsx",
      "src/components/ui/__tests__/markdown-renderer.test.tsx",
      "src/components/widget/widget-bootstrap.test.ts",
      "src/components/widget/widget-session.test.ts",
      "src/components/widget/widget-chrome.test.ts",
      "src/components/widget/widget-chrome-controls.test.tsx",
      "src/components/widget/public-agent-chat-page.test.tsx",
      "src/components/widget/session-agent-chat-page.test.tsx",
      "src/components/widget/session-agent-chat-page.integration.test.tsx",
      "src/components/widget/use-widget-session.test.tsx",
      "src/contexts/app-context-chat.test.tsx",
      "src/contexts/auth-context.test.tsx",
      "src/contexts/file-access-context.test.tsx",
      "src/hooks/use-file-mention.test.tsx",
      "src/hooks/use-websocket.test.ts",
      "src/lib/api-wrapper.test.ts",
      "src/lib/auth-cache.test.ts",
      "src/lib/files-disabled-presentation.test.ts",
    ],
  },
})
