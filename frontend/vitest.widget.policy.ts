import type { InlineConfig } from "vitest"

export type CoverageMetric = "statements" | "branches" | "functions" | "lines"

type WidgetCoverageThresholds = Record<CoverageMetric, number>

export interface WidgetCoverageOwner {
  sourcePath: string
  coveragePattern?: string
  thresholds: WidgetCoverageThresholds
}

export const widgetTestFiles = [
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
]

export const widgetCoverageOwners: WidgetCoverageOwner[] = [
  {
    sourcePath: "public/widget.js",
    thresholds: { statements: 95, branches: 90, functions: 95, lines: 95 },
  },
  {
    sourcePath: "src/app/widget/chat/[token]/page-client.tsx",
    coveragePattern: "src/app/widget/chat/[[]token[]]/page-client.tsx",
    thresholds: { statements: 90, branches: 75, functions: 90, lines: 90 },
  },
  {
    sourcePath: "src/components/chat/ChatInput.tsx",
    thresholds: { statements: 60, branches: 60, functions: 40, lines: 60 },
  },
  {
    sourcePath: "src/components/chat/ChatMessage.tsx",
    thresholds: { statements: 50, branches: 50, functions: 40, lines: 50 },
  },
  {
    sourcePath: "src/components/chat/TraceEventRenderer.tsx",
    thresholds: { statements: 80, branches: 75, functions: 75, lines: 80 },
  },
  {
    sourcePath: "src/components/file/file-preview-content.tsx",
    thresholds: { statements: 70, branches: 50, functions: 55, lines: 70 },
  },
  {
    sourcePath: "src/components/file/file-viewer.tsx",
    thresholds: { statements: 70, branches: 55, functions: 80, lines: 70 },
  },
  {
    sourcePath: "src/components/file/inline-file-preview.tsx",
    thresholds: { statements: 70, branches: 55, functions: 60, lines: 70 },
  },
  {
    sourcePath: "src/components/file/pptx-preview-renderer.tsx",
    thresholds: { statements: 45, branches: 35, functions: 30, lines: 45 },
  },
  {
    sourcePath: "src/components/task/task-conversation-panel.tsx",
    thresholds: { statements: 80, branches: 70, functions: 60, lines: 80 },
  },
  {
    sourcePath: "src/components/ui/markdown-renderer.tsx",
    thresholds: { statements: 65, branches: 65, functions: 75, lines: 65 },
  },
  {
    sourcePath: "src/components/widget/public-agent-chat-page.tsx",
    thresholds: { statements: 80, branches: 55, functions: 45, lines: 80 },
  },
  {
    sourcePath: "src/components/widget/session-agent-chat-page.tsx",
    thresholds: { statements: 90, branches: 85, functions: 75, lines: 90 },
  },
  {
    sourcePath: "src/components/widget/use-widget-session.ts",
    thresholds: { statements: 95, branches: 80, functions: 90, lines: 95 },
  },
  {
    sourcePath: "src/contexts/app-context-chat.tsx",
    thresholds: { statements: 40, branches: 60, functions: 60, lines: 40 },
  },
  {
    sourcePath: "src/contexts/auth-context.tsx",
    thresholds: { statements: 70, branches: 65, functions: 90, lines: 70 },
  },
  {
    sourcePath: "src/contexts/file-access-context.tsx",
    thresholds: { statements: 85, branches: 75, functions: 85, lines: 85 },
  },
  {
    sourcePath: "src/hooks/use-file-mention.ts",
    thresholds: { statements: 60, branches: 60, functions: 60, lines: 60 },
  },
  {
    sourcePath: "src/hooks/use-websocket.ts",
    thresholds: { statements: 80, branches: 75, functions: 65, lines: 80 },
  },
  {
    sourcePath: "src/lib/api-wrapper.ts",
    thresholds: { statements: 75, branches: 65, functions: 60, lines: 75 },
  },
  {
    sourcePath: "src/lib/auth-cache.ts",
    thresholds: { statements: 90, branches: 80, functions: 90, lines: 90 },
  },
  {
    sourcePath: "src/lib/files-disabled-presentation.ts",
    thresholds: { statements: 85, branches: 80, functions: 90, lines: 85 },
  },
  {
    sourcePath: "src/contexts/presentation-capabilities.tsx",
    thresholds: { statements: 100, branches: 100, functions: 100, lines: 100 },
  },
  {
    sourcePath: "src/app/settings/page.tsx",
    thresholds: { statements: 75, branches: 50, functions: 50, lines: 75 },
  },
  {
    sourcePath: "src/components/layout/sidebar.tsx",
    thresholds: { statements: 35, branches: 40, functions: 10, lines: 35 },
  },
  {
    sourcePath: "src/components/pages/login.tsx",
    thresholds: { statements: 85, branches: 55, functions: 60, lines: 85 },
  },
  {
    sourcePath: "src/components/pages/oidc-callback.tsx",
    thresholds: { statements: 75, branches: 45, functions: 95, lines: 75 },
  },
]

export const widgetCoverageExtensions = [".js", ".ts", ".tsx"]

export function buildWidgetTestOptions(baseTest: InlineConfig | undefined): InlineConfig {
  const coverageOwners = widgetCoverageOwners.map(
    (owner) => owner.coveragePattern ?? owner.sourcePath,
  )

  return {
    ...baseTest,
    include: [...widgetTestFiles],
    coverage: {
      provider: "v8",
      all: true,
      include: coverageOwners,
      exclude: [],
      extension: [...widgetCoverageExtensions],
      reporter: ["text", "json-summary"],
      reportsDirectory: "coverage/widget",
      thresholds: {
        perFile: true,
        ...Object.fromEntries(
          widgetCoverageOwners.map((owner) => [
            owner.coveragePattern ?? owner.sourcePath,
            owner.thresholds,
          ]),
        ),
      },
    },
  }
}
