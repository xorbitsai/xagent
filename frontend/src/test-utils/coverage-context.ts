import type { TestProject } from "vitest/node"

declare module "vitest" {
  export interface ProvidedContext {
    widgetCoverageEnabled: boolean
  }
}

export default function provideWidgetCoverage(project: TestProject) {
  project.provide("widgetCoverageEnabled", project.config.coverage.enabled === true)
}
