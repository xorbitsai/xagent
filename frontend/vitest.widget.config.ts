import { defineConfig } from "vitest/config"
import baseConfig from "./vitest.config"
import { buildWidgetTestOptions } from "./vitest.widget.policy"

export default defineConfig({
  ...baseConfig,
  test: buildWidgetTestOptions(baseConfig.test),
})
