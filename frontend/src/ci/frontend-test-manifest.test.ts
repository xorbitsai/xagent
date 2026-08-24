/**
 * Guards accidental drift across .github/workflows/ci.yml, package.json launchers,
 * and vitest.config.ts discovery. test:ci-manifest and test:run are independent
 * launchers; legitimate changes update the owner files and these invariants together.
 */
import { spawnSync } from "node:child_process"
import { existsSync, readdirSync, readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import ts from "typescript"
import { isMap, isScalar, isSeq, parseDocument } from "yaml"
import type { Node } from "yaml"
import { describe, expect, it, vi } from "vitest"
import vitestConfig from "../../vitest.config"
import widgetVitestConfig from "../../vitest.widget.config"
import {
  buildWidgetTestOptions,
  widgetCoverageExtensions,
  widgetCoverageOwners,
  widgetTestFiles,
} from "../../vitest.widget.policy"
import type { WidgetCoverageOwner } from "../../vitest.widget.policy"

// jsdom can load Vitest config in a second realm where vitest/config cannot initialize.
vi.mock("vitest/config", () => ({
  defineConfig: <T>(config: T) => config,
}))

const manifestCommand = "vitest run --config vitest.config.ts src/ci/frontend-test-manifest.test.ts"
const fullSuiteCommand = "vitest run"
const ciSummaryCondition =
  "always() && (github.event_name != 'pull_request' || github.event.pull_request.draft == false)"
const frontendSummaryCheckCommand =
  'check_job "frontend-build" "${{ needs[\'frontend-build\'].result }}"'
const ciSummaryFailurePropagationCommands = [
  "set -e",
  "failed=0",
  "check_job() {",
  'local name="$1"',
  'local result="$2"',
  'if [ "$result" != "success" ]; then',
  'echo "::error::$name finished with result: $result"',
  "failed=1",
  "fi",
  "}",
  'check_job "prepare-deepdoc-cache" "${{ needs[\'prepare-deepdoc-cache\'].result }}"',
  'check_job "pre-commit" "${{ needs[\'pre-commit\'].result }}"',
  'check_job "pytest-fast" "${{ needs[\'pytest-fast\'].result }}"',
  'check_job "pytest-fast-deepdoc" "${{ needs[\'pytest-fast-deepdoc\'].result }}"',
  'check_job "pytest-slow" "${{ needs[\'pytest-slow\'].result }}"',
  'check_job "e2e" "${{ needs.e2e.result }}"',
  frontendSummaryCheckCommand,
  'exit "$failed"',
] as const
const requiredFrontendSteps = [
  { command: "npm run test:widget:coverage", requiresExplicitBash: false },
  { command: "npm run test:ci-manifest", requiresExplicitBash: true },
  { command: "npm run test:run", requiresExplicitBash: true },
] as const
const retiredFrontendLaunchers = new Set([
  "npm run test:pages",
  "npm run test:kb-components",
  "npm run test:app-pages",
  "npm run test:home-build-contracts",
])
const moduleDir = path.dirname(fileURLToPath(import.meta.url))
const frontendRoot = path.resolve(moduleDir, "../..")
const frontendRootEntryNames = readdirSync(frontendRoot)
const recognizedWorkspaceProjectFilenames = ["vitest.workspace", "vitest.projects"].flatMap(
  (basename) =>
    [".ts", ".mts", ".cts", ".js", ".mjs", ".cjs", ".json"].map(
      (extension) => `${basename}${extension}`,
    ),
)
const packageJsonPath = path.resolve(moduleDir, "../../package.json")
const workflowPath = path.resolve(moduleDir, "../../../.github/workflows/ci.yml")
const widgetConfigPath = path.resolve(moduleDir, "../../vitest.widget.config.ts")
const realWorkflowSource = readFileSync(workflowPath, "utf8")

function replaceExactlyOnce(source: string, search: string, replacement: string, owner: string) {
  if (search.length === 0) {
    throw new Error(`${owner} search marker must not be empty`)
  }

  const count = countOccurrences(source, search)
  if (count !== 1) {
    throw new Error(`${owner} must appear exactly once; found ${count}`)
  }

  const mutated = source.replace(search, () => replacement)
  if (mutated === source) {
    throw new Error(`${owner} mutation must change the source`)
  }
  return mutated
}

function countOccurrences(source: string, search: string) {
  let count = 0
  let offset = 0
  while ((offset = source.indexOf(search, offset)) !== -1) {
    count += 1
    offset += search.length
  }
  return count
}

function parseWorkflowDocument(source: string, owner?: string) {
  const document = parseDocument(source, {
    version: "1.2",
    uniqueKeys: true,
    merge: false,
    keepSourceTokens: true,
  })
  if (document.errors.length > 0) {
    if (owner !== undefined) {
      throw new Error(`${owner} requires valid YAML`)
    }
    throw document.errors[0]
  }
  if (document.warnings.length > 0) {
    const warning = document.warnings[0]!
    if (owner !== undefined) {
      throw new Error(`${owner} requires warning-free YAML [${warning.code ?? "UNKNOWN"}]`)
    }
    throw new Error(`workflow YAML warning [${warning.code ?? "UNKNOWN"}]`)
  }
  return document
}

function replaceWorkflowJob(source: string, jobName: string, replacement: string, owner: string) {
  const document = parseWorkflowDocument(source, owner)

  const workflow = document.contents
  const jobs = isMap(workflow) ? workflow.get("jobs", true) : undefined
  if (!isMap(jobs)) {
    throw new Error(`${owner} requires a jobs mapping`)
  }

  const matches = jobs.items.filter(
    (pair) => isScalar(pair.key) && pair.key.value === jobName,
  )
  if (matches.length !== 1) {
    throw new Error(`${owner} must appear exactly once; found ${matches.length}`)
  }

  const pair = matches[0]!
  if (!isScalar(pair.key) || pair.key.range == null) {
    throw new Error(`${owner} key must have a source range`)
  }
  const jobStart = source.lastIndexOf("\n", pair.key.range[0] - 1) + 1
  const nextPair = jobs.items[jobs.items.indexOf(pair) + 1]
  const nextKey = nextPair?.key
  const nextPairStart = nextPair?.srcToken?.start[0]?.offset
  let jobEnd = source.length
  if (nextPairStart !== undefined) {
    jobEnd = nextPairStart
  } else if (nextKey !== undefined && isScalar(nextKey) && nextKey.range != null) {
    jobEnd = source.lastIndexOf("\n", nextKey.range[0] - 1) + 1
  }
  const mutated = `${source.slice(0, jobStart)}${replacement}${source.slice(jobEnd)}`
  if (mutated === source) {
    throw new Error(`${owner} mutation must change the source`)
  }
  return mutated
}

function removeWorkflowStepByCommand(
  source: string,
  jobName: string,
  command: string,
  owner: string,
) {
  const document = parseWorkflowDocument(source, owner)
  const workflow = document.contents
  const jobs = isMap(workflow) ? workflow.get("jobs", true) : undefined
  if (!isMap(jobs)) {
    throw new Error(`${owner} requires a jobs mapping`)
  }

  const job = jobs.get(jobName, true)
  if (!isMap(job)) {
    throw new Error(`${owner} requires jobs.${jobName} to be a mapping`)
  }

  const steps = job.get("steps", true)
  if (!isSeq(steps)) {
    throw new Error(`${owner} requires jobs.${jobName}.steps to be a sequence`)
  }

  const matches = steps.items.filter((step): step is Node => {
    const run = isMap(step) ? step.get("run", true) : undefined
    return isScalar(run) && run.value === command
  })
  if (matches.length !== 1) {
    throw new Error(`${owner} must match exactly one ${jobName} step; found ${matches.length}`)
  }

  const step = matches[0]!
  if (step.range == null) {
    throw new Error(`${owner} matched step must have a source range`)
  }
  const stepStart = source.lastIndexOf("\n", step.range[0] - 1) + 1
  const lineEnd = source.indexOf("\n", step.range[2])
  const stepEnd = lineEnd === -1 ? source.length : lineEnd + 1
  const mutated = `${source.slice(0, stepStart)}${source.slice(stepEnd)}`
  if (mutated === source) {
    throw new Error(`${owner} mutation must change the source`)
  }
  return mutated
}

function requireRecord(value: unknown, owner: string): asserts value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${owner} must be an object`)
  }
}

function assertBashCompatibleDefaultShell(value: Record<string, unknown>, owner: string) {
  const defaults = value.defaults
  if (defaults === undefined) {
    return
  }

  requireRecord(defaults, `${owner}.defaults`)
  const run = defaults.run
  if (run === undefined) {
    return
  }

  requireRecord(run, `${owner}.defaults.run`)
  if (run.shell !== undefined && run.shell !== "bash") {
    throw new Error(`${owner} defaults.run.shell must be bash when set`)
  }
}

function assertWorkflowTriggerContract(workflow: Record<string, unknown>) {
  requireRecord(workflow.on, "workflow triggers")
  const triggers = workflow.on
  for (const trigger of ["workflow_dispatch", "pull_request", "push", "merge_group"]) {
    if (!Object.hasOwn(triggers, trigger)) {
      throw new Error(`workflow triggers must include ${trigger}`)
    }
  }

  requireRecord(triggers.pull_request, "workflow pull_request")
  const pullRequest = triggers.pull_request
  if (
    !Array.isArray(pullRequest.branches) ||
    pullRequest.branches.filter((branch) => branch === "main").length !== 1
  ) {
    throw new Error("workflow pull_request.branches must contain main exactly once")
  }
  if (pullRequest.paths !== undefined || pullRequest["paths-ignore"] !== undefined) {
    throw new Error("workflow pull_request must not set paths or paths-ignore")
  }
  const requiredPullRequestTypes = ["opened", "synchronize", "reopened", "ready_for_review"]
  const pullRequestTypes = pullRequest.types
  if (
    !Array.isArray(pullRequestTypes) ||
    pullRequestTypes.length !== requiredPullRequestTypes.length ||
    requiredPullRequestTypes.some(
      (type) => pullRequestTypes.filter((candidate) => candidate === type).length !== 1,
    )
  ) {
    throw new Error(
      "workflow pull_request.types must contain opened, synchronize, reopened, ready_for_review exactly once",
    )
  }
}

function getNonEmptyScriptCommands(run: string) {
  return run
    .split(/\r?\n/)
    .map((line, index) => ({ index, value: line.trim() }))
    .filter(({ value }) => value !== "" && !value.startsWith("#"))
}

function executeCiSummaryScript(source: string) {
  const workflow = parseWorkflowDocument(source).toJS({ maxAliasCount: 100 })
  requireRecord(workflow, "workflow root")
  requireRecord(workflow.jobs, "jobs")
  requireRecord(workflow.jobs["ci-summary"], "jobs.ci-summary")
  const ciSummary = workflow.jobs["ci-summary"]
  if (!Array.isArray(ciSummary.steps)) {
    throw new Error("jobs.ci-summary.steps must be an array")
  }
  const checkStep = ciSummary.steps.find((step) => {
    requireRecord(step, "jobs.ci-summary.steps entry")
    return step.name === "Check required jobs"
  })
  requireRecord(checkStep, "Check required jobs")
  if (typeof checkStep.run !== "string") {
    throw new Error("Check required jobs run must be a string")
  }

  const expandedScript = checkStep.run.replace(
    /\$\{\{ needs(?:\[['"][^'"]+['"]\]|\.[A-Za-z0-9_-]+)\.result \}\}/g,
    "failure",
  )
  return spawnSync("bash", ["-c", expandedScript], { encoding: "utf8" })
}

function assertCiSummaryFailurePropagation(run: string) {
  const commands = getNonEmptyScriptCommands(run).map(({ value }) => value)
  if (
    commands.length !== ciSummaryFailurePropagationCommands.length ||
    commands.some((command, index) => command !== ciSummaryFailurePropagationCommands[index])
  ) {
    throw new Error("Check required jobs must use the supported failure-propagation command sequence")
  }
}

function assertCiSummaryContract(jobs: Record<string, unknown>) {
  const ciSummary = jobs["ci-summary"]
  requireRecord(ciSummary, "jobs.ci-summary")

  if (!Array.isArray(ciSummary.needs)) {
    throw new Error("jobs.ci-summary.needs must be an array")
  }
  if (ciSummary.needs.filter((job) => job === "frontend-build").length !== 1) {
    throw new Error("jobs.ci-summary.needs must contain frontend-build exactly once")
  }
  if (ciSummary["continue-on-error"] !== undefined) {
    throw new Error("jobs.ci-summary must not set continue-on-error")
  }
  if (ciSummary.if !== ciSummaryCondition) {
    throw new Error("jobs.ci-summary has an unexpected if policy")
  }
  if (!Array.isArray(ciSummary.steps)) {
    throw new Error("jobs.ci-summary.steps must be an array")
  }

  const matchingSteps = ciSummary.steps.filter((step) => {
    requireRecord(step, "jobs.ci-summary.steps entry")
    return step.name === "Check required jobs"
  })
  if (matchingSteps.length !== 1) {
    throw new Error("Check required jobs must appear in exactly one ci-summary step")
  }

  const checkStep = matchingSteps[0]!
  if (checkStep.shell !== "bash") {
    throw new Error("Check required jobs must use bash")
  }
  if (checkStep.if !== undefined) {
    throw new Error("Check required jobs must not set if")
  }
  if (checkStep["continue-on-error"] !== undefined) {
    throw new Error("Check required jobs must not set continue-on-error")
  }
  const run = checkStep.run
  if (typeof run !== "string") {
    throw new Error("Check required jobs run must be a string")
  }

  const frontendCheckLines = run
    .split(/\r?\n/)
    .filter((line) => line === frontendSummaryCheckCommand)
  if (frontendCheckLines.length !== 1) {
    throw new Error("Check required jobs must check frontend-build exactly once")
  }
  assertCiSummaryFailurePropagation(run)
}

function assertSemanticWorkflowManifest(source: string) {
  const document = parseWorkflowDocument(source)
  const workflow = document.toJS({ maxAliasCount: 100 })
  requireRecord(workflow, "workflow root")
  assertWorkflowTriggerContract(workflow)
  requireRecord(workflow.jobs, "jobs")
  assertCiSummaryContract(workflow.jobs)
  const frontendBuild = workflow.jobs["frontend-build"]
  requireRecord(frontendBuild, "jobs.frontend-build")
  if (!Array.isArray(frontendBuild.steps)) {
    throw new Error("jobs.frontend-build.steps must be an array")
  }
  if (frontendBuild["runs-on"] !== "ubuntu-latest") {
    throw new Error("jobs.frontend-build.runs-on must be ubuntu-latest")
  }

  assertBashCompatibleDefaultShell(workflow, "workflow root")
  assertBashCompatibleDefaultShell(frontendBuild, "jobs.frontend-build")
  if (frontendBuild["continue-on-error"] !== undefined) {
    throw new Error("jobs.frontend-build must not set continue-on-error")
  }

  for (const step of frontendBuild.steps) {
    requireRecord(step, "jobs.frontend-build.steps entry")
    if (step["continue-on-error"] !== undefined) {
      throw new Error("frontend-build steps must not set continue-on-error")
    }
  }

  for (const requiredStep of requiredFrontendSteps) {
    const matchingSteps = frontendBuild.steps.filter((step) => {
      requireRecord(step, "jobs.frontend-build.steps entry")
      return step.run === requiredStep.command
    })

    if (matchingSteps.length !== 1) {
      throw new Error(`${requiredStep.command} must appear in exactly one frontend step`)
    }

    const step = matchingSteps[0]!
    if (step["working-directory"] !== "./frontend") {
      throw new Error(`${requiredStep.command} must use ./frontend`)
    }
    if (step.if !== undefined) {
      throw new Error(`${requiredStep.command} must not set if`)
    }
    if (step["continue-on-error"] !== undefined) {
      throw new Error(`${requiredStep.command} must not set continue-on-error`)
    }
    const hasAllowedShell = requiredStep.requiresExplicitBash
      ? step.shell === "bash"
      : step.shell === undefined || step.shell === "bash"
    if (!hasAllowedShell) {
      throw new Error(`${requiredStep.command} has an unexpected shell policy`)
    }
  }

  const retiredDirectLauncher = frontendBuild.steps
    .map((step) => {
      requireRecord(step, "jobs.frontend-build.steps entry")
      return typeof step.run === "string" ? step.run.trim() : undefined
    })
    .find((run) => run !== undefined && retiredFrontendLaunchers.has(run))

  if (retiredDirectLauncher !== undefined) {
    throw new Error(
      `jobs.frontend-build must not directly run retired targeted launcher ${retiredDirectLauncher}`,
    )
  }
}

function assertRegularSuiteDiscovery(
  config: unknown,
  scripts: Record<string, string>,
  rootEntryNames: readonly string[],
) {
  requireRecord(config, "regular Vitest config")
  if (scripts["test:run"] !== fullSuiteCommand) {
    throw new Error("regular launcher must keep test:run as vitest run")
  }

  const testConfig = config.test
  requireRecord(testConfig, "regular Vitest config.test")
  const expectedBaseInclude = ["src/**/*.test.ts", "src/**/*.test.tsx"]
  const actualBaseInclude = Array.isArray(testConfig.include) ? [...testConfig.include].sort() : []
  if (JSON.stringify(actualBaseInclude) !== JSON.stringify(expectedBaseInclude)) {
    throw new Error("regular base discovery must preserve automatic discovery")
  }
  if (testConfig.exclude !== undefined) {
    throw new Error("regular base discovery must preserve automatic discovery")
  }
  if (Boolean(testConfig.passWithNoTests)) {
    throw new Error("regular base discovery must preserve automatic discovery")
  }

  if (
    config.workspace !== undefined ||
    testConfig.workspace !== undefined ||
    rootEntryNames.some((entryName) => recognizedWorkspaceProjectFilenames.includes(entryName))
  ) {
    throw new Error("regular workspace/project graph must be disabled")
  }

  const selectionValues = [
    config.root,
    testConfig.root,
    testConfig.dir,
    config.testNamePattern,
    testConfig.testNamePattern,
    config.related,
    testConfig.related,
    config.changed,
    testConfig.changed,
    config.shard,
    testConfig.shard,
    config.project,
    testConfig.project,
    config.filters,
    testConfig.filters,
    config.cliExclude,
    testConfig.cliExclude,
  ]
  if (
    selectionValues.some((value) => value !== undefined) ||
    Boolean(config.standalone) ||
    Boolean(testConfig.standalone) ||
    Boolean(config.allowOnly) ||
    Boolean(testConfig.allowOnly)
  ) {
    throw new Error("regular execution must be selection-neutral")
  }
}

const coverageMetrics = ["statements", "branches", "functions", "lines"] as const
const expectedWidgetCoverageExtensions = [".js", ".ts", ".tsx"]
const widgetCoverageRawKeys = [
  "provider",
  "all",
  "include",
  "exclude",
  "extension",
  "reporter",
  "reportsDirectory",
  "thresholds",
].sort()

function assertFrontendRootRelativePath(value: string, owner: string) {
  if (
    value.length === 0 ||
    path.isAbsolute(value) ||
    path.win32.isAbsolute(value) ||
    value.includes("\\") ||
    value.split("/").includes("..")
  ) {
    throw new Error(`${owner} must be a frontend-root-relative POSIX path`)
  }

  const resolvedPath = path.resolve(frontendRoot, value)
  if (!resolvedPath.startsWith(`${frontendRoot}${path.sep}`)) {
    throw new Error(`${owner} must resolve under the frontend root`)
  }
  return resolvedPath
}

function escapeCoverageBrackets(sourcePath: string) {
  return sourcePath.replace(/[\[\]]/g, (character) => (character === "[" ? "[[]" : "[]]"))
}

function assertWidgetTestFilePolicy(testFiles: readonly string[]) {
  const seen = new Set<string>()
  for (const testFile of testFiles) {
    const resolvedPath = assertFrontendRootRelativePath(testFile, "Widget test path")
    if (seen.has(testFile)) {
      throw new Error("Widget test paths must be unique")
    }
    seen.add(testFile)
    if (!testFile.startsWith("src/")) {
      throw new Error("Widget test paths must remain under src/")
    }
    if (!testFile.endsWith(".test.ts") && !testFile.endsWith(".test.tsx")) {
      throw new Error("Widget test paths must name Vitest test files")
    }
    if (!existsSync(resolvedPath)) {
      throw new Error("Widget test paths must exist")
    }
  }
}

function assertWidgetCoverageOwnerPolicy(owners: readonly WidgetCoverageOwner[]) {
  const sourcePaths = new Set<string>()
  const coveragePatterns = new Set<string>()
  for (const owner of owners) {
    const resolvedPath = assertFrontendRootRelativePath(owner.sourcePath, "Widget coverage source path")
    if (sourcePaths.has(owner.sourcePath)) {
      throw new Error("Widget coverage source paths must be unique")
    }
    sourcePaths.add(owner.sourcePath)
    if (!existsSync(resolvedPath)) {
      throw new Error("Widget coverage source paths must exist")
    }
    if (/[*?{}()!]/.test(owner.sourcePath)) {
      throw new Error("Widget coverage source paths must not contain glob metacharacters")
    }

    const coveragePattern = owner.coveragePattern ?? owner.sourcePath
    if (coveragePatterns.has(coveragePattern)) {
      throw new Error("Widget coverage patterns must be unique")
    }
    coveragePatterns.add(coveragePattern)
    const hasBrackets = /[\[\]]/.test(owner.sourcePath)
    if (hasBrackets && owner.coveragePattern === undefined) {
      throw new Error("Widget coverage patterns must escape bracketed source paths")
    }
    if (!hasBrackets && owner.coveragePattern !== undefined) {
      throw new Error("Widget coverage patterns must be absent for unbracketed source paths")
    }
    if (owner.coveragePattern !== undefined && owner.coveragePattern !== escapeCoverageBrackets(owner.sourcePath)) {
      throw new Error("Widget coverage patterns must use canonical bracket escaping")
    }
    if (!expectedWidgetCoverageExtensions.includes(path.extname(owner.sourcePath))) {
      throw new Error("Widget coverage source paths must use an owned extension")
    }

    const thresholdKeys = Object.keys(owner.thresholds).sort()
    if (JSON.stringify(thresholdKeys) !== JSON.stringify([...coverageMetrics].sort())) {
      throw new Error("Widget coverage thresholds must contain exactly four metrics")
    }
    for (const metric of coverageMetrics) {
      const threshold = owner.thresholds[metric]
      if (
        typeof threshold !== "number" ||
        !Number.isFinite(threshold) ||
        threshold <= 0 ||
        threshold > 100
      ) {
        throw new Error("Widget coverage thresholds must be finite positive percentages")
      }
    }
  }
}

function assertWidgetCoveragePolicy(
  coverage: unknown,
  owners: readonly WidgetCoverageOwner[] = widgetCoverageOwners,
) {
  requireRecord(coverage, "Widget coverage")
  if (JSON.stringify(Object.keys(coverage).sort()) !== JSON.stringify(widgetCoverageRawKeys)) {
    throw new Error("Widget coverage raw keys must match the policy")
  }
  const effectiveOwners = owners.map((owner) => owner.coveragePattern ?? owner.sourcePath)
  if (coverage.provider !== "v8") {
    throw new Error("Widget coverage provider must be v8")
  }
  if (coverage.all !== true) {
    throw new Error("Widget coverage all must be true")
  }
  if (JSON.stringify(coverage.include) !== JSON.stringify(effectiveOwners)) {
    throw new Error("Widget coverage include must match the owner policy")
  }
  if (JSON.stringify(coverage.exclude) !== JSON.stringify([])) {
    throw new Error("Widget coverage exclude must be empty")
  }
  if (JSON.stringify(coverage.extension) !== JSON.stringify(expectedWidgetCoverageExtensions)) {
    throw new Error("Widget coverage extension must match the policy")
  }
  if (JSON.stringify(coverage.reporter) !== JSON.stringify(["text", "json-summary"])) {
    throw new Error("Widget coverage reporter must match the policy")
  }
  if (coverage.reportsDirectory !== "coverage/widget") {
    throw new Error("Widget coverage reports directory must match the policy")
  }

  requireRecord(coverage.thresholds, "Widget coverage thresholds")
  const thresholds = coverage.thresholds
  const thresholdKeys = ["perFile", ...effectiveOwners].sort()
  if (JSON.stringify(Object.keys(thresholds).sort()) !== JSON.stringify(thresholdKeys)) {
    throw new Error("Widget coverage threshold keys must match the owner policy")
  }
  if (thresholds.perFile !== true) {
    throw new Error("Widget coverage perFile must be true")
  }
  for (const owner of owners) {
    const coveragePattern = owner.coveragePattern ?? owner.sourcePath
    if (JSON.stringify(thresholds[coveragePattern]) !== JSON.stringify(owner.thresholds)) {
      throw new Error("Widget coverage thresholds must match the owner policy")
    }
  }
}

function buildValidWidgetCoverageFixture(owners: readonly WidgetCoverageOwner[] = widgetCoverageOwners) {
  return {
    provider: "v8",
    all: true,
    include: owners.map((owner) => owner.coveragePattern ?? owner.sourcePath),
    exclude: [],
    extension: [...expectedWidgetCoverageExtensions],
    reporter: ["text", "json-summary"],
    reportsDirectory: "coverage/widget",
    thresholds: {
      perFile: true,
      ...Object.fromEntries(owners.map((owner) => [
        owner.coveragePattern ?? owner.sourcePath,
        owner.thresholds,
      ])),
    },
  }
}

function assertWidgetConfigSourceConsumesPolicy(source: string) {
  const ownerError = "Widget config test must be exactly buildWidgetTestOptions(baseConfig.test)"
  const sourceFile = ts.createSourceFile(
    widgetConfigPath,
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TS,
  )
  const defaultExports = sourceFile.statements.filter(
    (statement): statement is ts.ExportAssignment =>
      ts.isExportAssignment(statement) && !statement.isExportEquals,
  )
  if (defaultExports.length !== 1) {
    throw new Error(ownerError)
  }

  const defineConfigCall = defaultExports[0]!.expression
  if (
    !ts.isCallExpression(defineConfigCall) ||
    !ts.isIdentifier(defineConfigCall.expression) ||
    defineConfigCall.expression.text !== "defineConfig" ||
    defineConfigCall.arguments.length !== 1
  ) {
    throw new Error(ownerError)
  }
  const configObject = defineConfigCall.arguments[0]!
  if (!ts.isObjectLiteralExpression(configObject)) {
    throw new Error(ownerError)
  }

  const testProperties = configObject.properties.filter((property) => {
    if (!("name" in property) || property.name === undefined) {
      return false
    }
    if (ts.isIdentifier(property.name) || ts.isStringLiteral(property.name)) {
      return property.name.text === "test"
    }
    return (
      ts.isComputedPropertyName(property.name) &&
      ts.isStringLiteral(property.name.expression) &&
      property.name.expression.text === "test"
    )
  })
  if (testProperties.length !== 1 || !ts.isPropertyAssignment(testProperties[0]!)) {
    throw new Error(ownerError)
  }
  const testProperty = testProperties[0]
  const laterProperties = configObject.properties.slice(
    configObject.properties.indexOf(testProperty) + 1,
  )
  if (laterProperties.length !== 0) {
    throw new Error(ownerError)
  }

  const builderCall = testProperty.initializer
  if (
    !ts.isCallExpression(builderCall) ||
    !ts.isIdentifier(builderCall.expression) ||
    builderCall.expression.text !== "buildWidgetTestOptions" ||
    builderCall.arguments.length !== 1
  ) {
    throw new Error(ownerError)
  }
  const baseTest = builderCall.arguments[0]!
  if (
    !ts.isPropertyAccessExpression(baseTest) ||
    baseTest.questionDotToken !== undefined ||
    !ts.isIdentifier(baseTest.expression) ||
    baseTest.expression.text !== "baseConfig" ||
    baseTest.name.text !== "test"
  ) {
    throw new Error(ownerError)
  }
}

function assertWidgetConfigConsumesPolicy(config: unknown) {
  requireRecord(config, "Widget config")
  requireRecord(config.test, "Widget config.test")
  const expected = buildWidgetTestOptions(vitestConfig.test)
  if (JSON.stringify(config.test.include) !== JSON.stringify(expected.include)) {
    throw new Error("Widget config must consume policy test files")
  }
  if (JSON.stringify(config.test.coverage) !== JSON.stringify(expected.coverage)) {
    throw new Error("Widget config must consume policy coverage")
  }
  assertWidgetCoveragePolicy(config.test.coverage)
}

describe("frontend CI test manifest", () => {
  it("keeps the current Widget coverage contract available to the real config", () => {
    const expectedWidgetTestFiles = [
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
    const expectedWidgetCoverageThresholds = {
      "public/widget.js": { statements: 95, branches: 90, functions: 95, lines: 95 },
      "src/app/widget/chat/[[]token[]]/page-client.tsx": { statements: 90, branches: 75, functions: 90, lines: 90 },
      "src/components/chat/ChatInput.tsx": { statements: 60, branches: 60, functions: 40, lines: 60 },
      "src/components/chat/ChatMessage.tsx": { statements: 50, branches: 50, functions: 40, lines: 50 },
      "src/components/chat/TraceEventRenderer.tsx": { statements: 80, branches: 75, functions: 75, lines: 80 },
      "src/components/file/file-preview-content.tsx": { statements: 70, branches: 50, functions: 55, lines: 70 },
      "src/components/file/file-viewer.tsx": { statements: 70, branches: 55, functions: 80, lines: 70 },
      "src/components/file/inline-file-preview.tsx": { statements: 70, branches: 55, functions: 60, lines: 70 },
      "src/components/file/pptx-preview-renderer.tsx": { statements: 45, branches: 35, functions: 30, lines: 45 },
      "src/components/task/task-conversation-panel.tsx": { statements: 80, branches: 70, functions: 60, lines: 80 },
      "src/components/ui/markdown-renderer.tsx": { statements: 65, branches: 65, functions: 75, lines: 65 },
      "src/components/widget/public-agent-chat-page.tsx": { statements: 80, branches: 55, functions: 45, lines: 80 },
      "src/components/widget/session-agent-chat-page.tsx": { statements: 90, branches: 85, functions: 75, lines: 90 },
      "src/components/widget/use-widget-session.ts": { statements: 95, branches: 80, functions: 90, lines: 95 },
      "src/contexts/app-context-chat.tsx": { statements: 40, branches: 60, functions: 60, lines: 40 },
      "src/contexts/auth-context.tsx": { statements: 70, branches: 65, functions: 90, lines: 70 },
      "src/contexts/file-access-context.tsx": { statements: 85, branches: 75, functions: 85, lines: 85 },
      "src/hooks/use-file-mention.ts": { statements: 60, branches: 60, functions: 60, lines: 60 },
      "src/hooks/use-websocket.ts": { statements: 80, branches: 75, functions: 65, lines: 80 },
      "src/lib/api-wrapper.ts": { statements: 75, branches: 65, functions: 60, lines: 75 },
      "src/lib/auth-cache.ts": { statements: 90, branches: 80, functions: 90, lines: 90 },
      "src/lib/files-disabled-presentation.ts": { statements: 85, branches: 80, functions: 90, lines: 85 },
      "src/contexts/presentation-capabilities.tsx": { statements: 100, branches: 100, functions: 100, lines: 100 },
      "src/app/settings/page.tsx": { statements: 75, branches: 50, functions: 50, lines: 75 },
      "src/components/layout/sidebar.tsx": { statements: 35, branches: 40, functions: 10, lines: 35 },
      "src/components/pages/login.tsx": { statements: 85, branches: 55, functions: 60, lines: 85 },
      "src/components/pages/oidc-callback.tsx": { statements: 75, branches: 45, functions: 95, lines: 75 },
    }
    const widgetTest = widgetVitestConfig.test as Record<string, unknown>
    const coverage = widgetTest.coverage as Record<string, unknown>
    const thresholds = coverage.thresholds as Record<string, unknown>

    expect(widgetTest.include).toEqual(expectedWidgetTestFiles)
    expect(widgetCoverageExtensions).toEqual(expectedWidgetCoverageExtensions)
    expect(coverage.provider).toBe("v8")
    expect(coverage.reporter).toEqual(["text", "json-summary"])
    expect(coverage.reportsDirectory).toBe("coverage/widget")
    expect(thresholds.perFile).toBe(true)
    expect(Object.keys(thresholds).filter((key) => key !== "perFile").sort()).toEqual(
      [...(coverage.include as string[])].sort(),
    )
    expect(Object.fromEntries(widgetCoverageOwners.map((owner) => [
      owner.coveragePattern ?? owner.sourcePath,
      owner.thresholds,
    ]))).toEqual(expectedWidgetCoverageThresholds)
  })

  it("requires the real Widget config to fail close on every coverage owner", () => {
    const widgetTest = widgetVitestConfig.test as Record<string, unknown>
    const coverage = widgetTest.coverage as Record<string, unknown>
    const widgetConfigSource = readFileSync(widgetConfigPath, "utf8")

    expect.soft(coverage.all).toBe(true)
    expect.soft(coverage.exclude).toEqual([])
    expect.soft(coverage.extension).toEqual([".js", ".ts", ".tsx"])
    expect(() => assertWidgetConfigSourceConsumesPolicy(widgetConfigSource)).not.toThrow()
  })

  it("keeps Widget policy paths, patterns, and thresholds independently valid", () => {
    assertWidgetTestFilePolicy(widgetTestFiles)
    assertWidgetCoverageOwnerPolicy(widgetCoverageOwners)
    assertWidgetCoveragePolicy(buildWidgetTestOptions(vitestConfig.test).coverage)
  })

  it("makes the real Widget config consume the complete policy builder", () => {
    assertWidgetConfigConsumesPolicy(widgetVitestConfig)
  })

  it.each([
    ["duplicate test paths", () => assertWidgetTestFilePolicy([...widgetTestFiles, widgetTestFiles[0]!]), "Widget test paths must be unique"],
    ["a missing test path", () => assertWidgetTestFilePolicy([""]), "Widget test path must be a frontend-root-relative POSIX path"],
    ["a nonexistent test path", () => assertWidgetTestFilePolicy(["src/ci/missing.test.ts"]), "Widget test paths must exist"],
    ["a parent-relative test path", () => assertWidgetTestFilePolicy(["../outside.test.ts"]), "Widget test path must be a frontend-root-relative POSIX path"],
    ["an absolute test path", () => assertWidgetTestFilePolicy(["/tmp/outside.test.ts"]), "Widget test path must be a frontend-root-relative POSIX path"],
    ["a backslash test path", () => assertWidgetTestFilePolicy(["src\\ci\\frontend-test-manifest.test.ts"]), "Widget test path must be a frontend-root-relative POSIX path"],
  ])("rejects %s at the Widget test-path owner", (_, mutate, expectedError) => {
    expect(mutate).toThrow(expectedError)
  })

  it.each([
    [
      "duplicate source paths",
      () => assertWidgetCoverageOwnerPolicy([...widgetCoverageOwners, { ...widgetCoverageOwners[0]! }]),
      "Widget coverage source paths must be unique",
    ],
    [
      "duplicate effective coverage patterns",
      () => assertWidgetCoverageOwnerPolicy([
        ...widgetCoverageOwners,
        {
          sourcePath: "package.json",
          coveragePattern: "public/widget.js",
          thresholds: { statements: 1, branches: 1, functions: 1, lines: 1 },
        },
      ]),
      "Widget coverage patterns must be unique",
    ],
    [
      "a missing source path",
      () => assertWidgetCoverageOwnerPolicy([
        ...widgetCoverageOwners,
        {
          sourcePath: "",
          thresholds: { statements: 1, branches: 1, functions: 1, lines: 1 },
        },
      ]),
      "Widget coverage source path must be a frontend-root-relative POSIX path",
    ],
    [
      "a nonexistent source path",
      () => assertWidgetCoverageOwnerPolicy([
        ...widgetCoverageOwners,
        {
          sourcePath: "src/ci/missing-source.ts",
          thresholds: { statements: 1, branches: 1, functions: 1, lines: 1 },
        },
      ]),
      "Widget coverage source paths must exist",
    ],
    [
      "an outside-root source path",
      () => assertWidgetCoverageOwnerPolicy([
        ...widgetCoverageOwners,
        {
          sourcePath: "../outside.ts",
          thresholds: { statements: 1, branches: 1, functions: 1, lines: 1 },
        },
      ]),
      "Widget coverage source path must be a frontend-root-relative POSIX path",
    ],
    [
      "a mismatched bracket escape",
      () => assertWidgetCoverageOwnerPolicy(widgetCoverageOwners.map((owner, index) => index === 1
        ? { ...owner, coveragePattern: "src/app/widget/chat/[token]/page-client.tsx" }
        : owner)),
      "Widget coverage patterns must use canonical bracket escaping",
    ],
    [
      "an unsupported source extension",
      () => assertWidgetCoverageOwnerPolicy([
        ...widgetCoverageOwners,
        {
          sourcePath: "package.json",
          thresholds: { statements: 1, branches: 1, functions: 1, lines: 1 },
        },
      ]),
      "Widget coverage source paths must use an owned extension",
    ],
    [
      "a missing coverage metric",
      () => assertWidgetCoverageOwnerPolicy(widgetCoverageOwners.map((owner, index) => index === 0
        ? {
            ...owner,
            thresholds: { statements: 1, branches: 1, functions: 1 } as unknown as WidgetCoverageOwner["thresholds"],
          }
        : owner)),
      "Widget coverage thresholds must contain exactly four metrics",
    ],
    [
      "a zero coverage metric",
      () => assertWidgetCoverageOwnerPolicy(widgetCoverageOwners.map((owner, index) => index === 0
        ? { ...owner, thresholds: { ...owner.thresholds, lines: 0 } }
        : owner)),
      "Widget coverage thresholds must be finite positive percentages",
    ],
    [
      "a NaN coverage metric",
      () => assertWidgetCoverageOwnerPolicy(widgetCoverageOwners.map((owner, index) => index === 0
        ? { ...owner, thresholds: { ...owner.thresholds, statements: Number.NaN } }
        : owner)),
      "Widget coverage thresholds must be finite positive percentages",
    ],
    [
      "an infinite coverage metric",
      () => assertWidgetCoverageOwnerPolicy(widgetCoverageOwners.map((owner, index) => index === 0
        ? { ...owner, thresholds: { ...owner.thresholds, statements: Number.POSITIVE_INFINITY } }
        : owner)),
      "Widget coverage thresholds must be finite positive percentages",
    ],
  ])("rejects %s at the Widget coverage-owner owner", (_, mutate, expectedError) => {
    expect(mutate).toThrow(expectedError)
  })

  it.each([
    ["all: false", (coverage: Record<string, unknown>) => ({ ...coverage, all: false }), "Widget coverage all must be true"],
    ["an owner exclusion", (coverage: Record<string, unknown>) => ({ ...coverage, exclude: ["public/widget.js"] }), "Widget coverage exclude must be empty"],
    ["a lost extension", (coverage: Record<string, unknown>) => ({ ...coverage, extension: [".js", ".ts"] }), "Widget coverage extension must match the policy"],
    ["a changed provider", (coverage: Record<string, unknown>) => ({ ...coverage, provider: "istanbul" }), "Widget coverage provider must be v8"],
    ["a changed reporter", (coverage: Record<string, unknown>) => ({ ...coverage, reporter: ["text"] }), "Widget coverage reporter must match the policy"],
    ["a changed report directory", (coverage: Record<string, unknown>) => ({ ...coverage, reportsDirectory: "coverage/other" }), "Widget coverage reports directory must match the policy"],
    [
      "a disabled per-file threshold",
      (coverage: Record<string, unknown>) => ({
        ...coverage,
        thresholds: { ...(coverage.thresholds as Record<string, unknown>), perFile: false },
      }),
      "Widget coverage perFile must be true",
    ],
    [
      "an orphan threshold",
      (coverage: Record<string, unknown>) => ({
        ...coverage,
        thresholds: {
          ...(coverage.thresholds as Record<string, unknown>),
          orphan: { statements: 1, branches: 1, functions: 1, lines: 1 },
        },
      }),
      "Widget coverage threshold keys must match the owner policy",
    ],
    [
      "a missing owner threshold",
      (coverage: Record<string, unknown>) => {
        const thresholds = { ...(coverage.thresholds as Record<string, unknown>) }
        delete thresholds["public/widget.js"]
        return { ...coverage, thresholds }
      },
      "Widget coverage threshold keys must match the owner policy",
    ],
    [
      "a global floor",
      (coverage: Record<string, unknown>) => ({
        ...coverage,
        thresholds: { ...(coverage.thresholds as Record<string, unknown>), 100: 1 },
      }),
      "Widget coverage threshold keys must match the owner policy",
    ],
    ["an unowned coverage key", (coverage: Record<string, unknown>) => ({ ...coverage, autoUpdate: true }), "Widget coverage raw keys must match the policy"],
  ])("rejects %s at the Widget coverage-config owner", (_, mutate, expectedError) => {
    const coverage = buildValidWidgetCoverageFixture()
    expect(() => assertWidgetCoveragePolicy(mutate(coverage))).toThrow(expectedError)
  })

  it("rejects a Widget config detached from the policy builder", () => {
    expect(() => assertWidgetConfigConsumesPolicy({
      ...widgetVitestConfig,
      test: {
        ...widgetVitestConfig.test,
        include: [],
      },
    })).toThrow("Widget config must consume policy test files")
    expect(() => assertWidgetConfigConsumesPolicy({
      ...widgetVitestConfig,
      test: {
        ...widgetVitestConfig.test,
        coverage: {},
      },
    })).toThrow("Widget config must consume policy coverage")
  })

  it("rejects a redundant Widget include override at the config-source owner", () => {
    const widgetConfigSource = readFileSync(widgetConfigPath, "utf8")
    const source = replaceExactlyOnce(
      widgetConfigSource,
      "  test: buildWidgetTestOptions(baseConfig.test),\n",
      "  test: {\n    ...buildWidgetTestOptions(baseConfig.test),\n    include: Array.from(widgetTestFiles),\n  },\n",
      "Widget config redundant include override",
    )

    expect(() => assertWidgetConfigSourceConsumesPolicy(source)).toThrow(
      "Widget config test must be exactly buildWidgetTestOptions(baseConfig.test)",
    )
  })

  it("rejects a later dynamic Widget test override at the config-source owner", () => {
    const widgetConfigSource = readFileSync(widgetConfigPath, "utf8")
    const withComputedKey = replaceExactlyOnce(
      widgetConfigSource,
      'import { buildWidgetTestOptions } from "./vitest.widget.policy"\n',
      'import { buildWidgetTestOptions } from "./vitest.widget.policy"\n\nconst widgetTestKey = "test"\n',
      "Widget config computed test key",
    )
    const source = replaceExactlyOnce(
      withComputedKey,
      "  test: buildWidgetTestOptions(baseConfig.test),\n",
      "  test: buildWidgetTestOptions(baseConfig.test),\n  [widgetTestKey]: {\n    ...buildWidgetTestOptions(baseConfig.test),\n    include: Array.from(widgetTestFiles),\n  },\n",
      "Widget config dynamic test override",
    )

    expect(() => assertWidgetConfigSourceConsumesPolicy(source)).toThrow(
      "Widget config test must be exactly buildWidgetTestOptions(baseConfig.test)",
    )
  })

  it("requires the full regular suite in frontend-build", () => {
    expect(() => assertSemanticWorkflowManifest(realWorkflowSource)).not.toThrow()
  })

  it.each([
    [
      "npm run test:pages",
      "      - name: Retired page test lane\n        working-directory: ./frontend\n        run: |\n\n          npm run test:pages\n\n",
    ],
    [
      "npm run test:kb-components",
      "      - name: Retired KB component test lane\n        working-directory: ./frontend\n        run: npm run test:kb-components\n\n",
    ],
    [
      "npm run test:app-pages",
      "      - name: Retired App Router test lane\n        working-directory: ./frontend\n        run: npm run test:app-pages\n\n",
    ],
    [
      "npm run test:home-build-contracts",
      "      - name: Retired home build contract lane\n        working-directory: ./frontend\n        run: npm run test:home-build-contracts\n\n",
    ],
  ])("rejects retired direct launcher %s", (launcher, retiredStep) => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "      - name: Build frontend (static export)\n",
      `${retiredStep}      - name: Build frontend (static export)\n`,
      `${launcher} direct launcher insertion`,
    )

    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      `jobs.frontend-build must not directly run retired targeted launcher ${launcher}`,
    )
  })

  it.each([
    ["LF", realWorkflowSource],
    ["CRLF", realWorkflowSource.replace(/\r?\n/g, "\r\n")],
  ])("accepts the real workflow with %s line endings", (_, source) => {
    expect(() => assertSemanticWorkflowManifest(source)).not.toThrow()
  })

  it("removes the full regular suite semantically when its presentation drifts", () => {
    const workflowSource = realWorkflowSource
    const source = replaceExactlyOnce(
      workflowSource,
      "      - name: Run full frontend test suite\n        working-directory: ./frontend\n        shell: bash\n        run: npm run test:run\n",
      "      - run: npm run test:run\n        env:\n          FULL_SUITE_MODE: manifest\n        working-directory: ./frontend\n        shell: bash\n        name: Run full frontend test suite\n",
      "full regular suite presentation drift",
    )
    const followingJobs = source.slice(source.indexOf("\n  ci-summary:\n"))
    const withoutFullSuiteStep = removeWorkflowStepByCommand(
      source,
      "frontend-build",
      "npm run test:run",
      "full regular suite removal",
    )

    expect(withoutFullSuiteStep).toContain(followingJobs)
    expect(() => assertSemanticWorkflowManifest(withoutFullSuiteStep)).toThrow(
      "npm run test:run must appear in exactly one frontend step",
    )
  })

  it("rejects pull request trigger path masking", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "  pull_request:\n    branches: [main]\n",
      "  pull_request:\n    branches: [main]\n    paths-ignore: [frontend/**]\n",
      "pull_request paths-ignore insertion",
    )

    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "workflow pull_request must not set paths or paths-ignore",
    )
  })

  it("rejects a closed-only pull request trigger type", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "    types: [opened, synchronize, reopened, ready_for_review]\n",
      "    types: [closed]\n",
      "closed-only pull_request type",
    )

    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "workflow pull_request.types must contain opened, synchronize, reopened, ready_for_review exactly once",
    )
  })

  it("rejects a pull request trigger without synchronize", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "    types: [opened, synchronize, reopened, ready_for_review]\n",
      "    types: [opened, reopened, ready_for_review]\n",
      "missing synchronize pull_request type",
    )

    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "workflow pull_request.types must contain opened, synchronize, reopened, ready_for_review exactly once",
    )
  })

  it("rejects a pull request trigger without ready_for_review", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "    types: [opened, synchronize, reopened, ready_for_review]\n",
      "    types: [opened, synchronize, reopened]\n",
      "missing ready_for_review pull_request type",
    )

    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "workflow pull_request.types must contain opened, synchronize, reopened, ready_for_review exactly once",
    )
  })

  it.each([
    ["pull_request", "  pull_request:\n", "  pull-request:\n"],
    ["merge_group", "  merge_group:\n", "  merge-group:\n"],
  ])("requires the %s workflow trigger", (_, search, replacement) => {
    const source = replaceExactlyOnce(realWorkflowSource, search, replacement, `${_} trigger`)

    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      `workflow triggers must include ${_}`,
    )
  })

  it("requires the frontend build runner contract", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "  frontend-build:\n    runs-on: ubuntu-latest\n",
      "  frontend-build:\n    runs-on: windows-latest\n",
      "frontend-build runner",
    )

    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "jobs.frontend-build.runs-on must be ubuntu-latest",
    )
  })

  it("rejects a no-op summary function", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      '          check_job() {\n            local name="$1"\n            local result="$2"\n            if [ "$result" != "success" ]; then\n              echo "::error::$name finished with result: $result"\n              failed=1\n            fi\n          }\n',
      "          check_job() {\n            :\n          }\n",
      "check_job function body",
    )

    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "Check required jobs must use the supported failure-propagation command sequence",
    )
  })

  it("rejects an inverted summary result condition", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      'if [ "$result" != "success" ]; then',
      'if [ "$result" = "success" ]; then',
      "check_job result condition",
    )

    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "Check required jobs must use the supported failure-propagation command sequence",
    )
  })

  it("accepts summary comments and blank lines", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "          failed=0\n",
      "          # initialize the aggregate result\n\n          failed=0\n",
      "summary comment insertion",
    )

    expect(() => assertSemanticWorkflowManifest(source)).not.toThrow()
  })

  it("runs the accepted summary script to a failed result", () => {
    const result = executeCiSummaryScript(realWorkflowSource)

    expect(result.error).toBeUndefined()
    expect(result.status).toBe(1)
  })

  it("rejects an early return in check_job that leaves failed clear", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      '          check_job() {\n            local name="$1"\n',
      '          check_job() {\n            return 0\n            local name="$1"\n',
      "check_job early return",
    )

    expect(executeCiSummaryScript(source).status).toBe(0)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "Check required jobs must use the supported failure-propagation command sequence",
    )
  })

  it("rejects a later function-form check_job redefinition", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      '          }\n\n          check_job "prepare-deepdoc-cache"',
      '          }\n\n          function check_job { :; }\n\n          check_job "prepare-deepdoc-cache"',
      "check_job redefinition",
    )

    expect(executeCiSummaryScript(source).status).toBe(0)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "Check required jobs must use the supported failure-propagation command sequence",
    )
  })

  it("rejects an early successful summary exit", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "          failed=0\n",
      "          failed=0\n          exit 0\n",
      "early summary exit",
    )

    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "Check required jobs must use the supported failure-propagation command sequence",
    )
  })

  it("requires the failed exit to be terminal", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      '          exit "$failed"\n',
      '          exit "$failed"\n          echo "summary complete"\n',
      "summary terminal exit",
    )

    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "Check required jobs must use the supported failure-propagation command sequence",
    )
  })

  it.each([
    [
      "an anchored sibling",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "\n  ci-summary:\n",
          "\n  manifest-anchor: &manifest_base\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo manifest anchor\n\n  manifest-alias: *manifest_base\n\n  ci-summary:\n",
          "ci-summary insertion point",
        ),
    ],
    [
      "an aliased sibling",
      (source: string) => {
        const anchored = replaceExactlyOnce(
          source,
          "  prepare-deepdoc-cache:\n",
          "  prepare-deepdoc-cache: &manifest_base\n",
          "prepare-deepdoc-cache anchor",
        )
        return replaceExactlyOnce(
          anchored,
          "\n  ci-summary:\n",
          "\n  manifest-alias: *manifest_base\n\n  ci-summary:\n",
          "ci-summary insertion point",
        )
      },
    ],
  ])("accepts %s", (_, transform) => {
    const source = transform(realWorkflowSource)

    expect(source).toContain("&manifest_base")
    expect(source).toContain("*manifest_base")
    expect(() => assertSemanticWorkflowManifest(source)).not.toThrow()
  })

  it("accepts a folded required command with the same semantic value", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "        run: npm run test:run\n",
      "        run: >-\n          npm run\n          test:run\n",
      "full regular suite command",
    )
    expect(() => assertSemanticWorkflowManifest(source)).not.toThrow()
  })

  it("accepts an explicit bash shell on the Widget coverage step", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "        run: npm run test:widget:coverage\n",
      "        shell: bash\n        run: npm run test:widget:coverage\n",
      "Widget coverage shell insertion",
    )
    expect(() => assertSemanticWorkflowManifest(source)).not.toThrow()
  })

  it.each([
    [
      "workflow",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "jobs:\n",
          "defaults:\n  run:\n    shell: bash\n\njobs:\n",
          "workflow jobs owner",
        ),
    ],
    [
      "frontend-build job",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "  frontend-build:\n",
          "  frontend-build:\n    defaults:\n      run:\n        shell: bash\n",
          "frontend-build owner",
        ),
    ],
  ])("accepts an explicit bash default at %s scope", (_, transform) => {
    const source = transform(realWorkflowSource)
    expect(() => assertSemanticWorkflowManifest(source)).not.toThrow()
  })

  it.each([
    [
      "manifest launcher with workflow defaults",
      (source: string) =>
        replaceExactlyOnce(
          replaceExactlyOnce(
            source,
            "jobs:\n",
            "defaults:\n  run:\n    shell: bash\n\njobs:\n",
            "workflow jobs owner",
          ),
          "        shell: bash\n        run: npm run test:ci-manifest\n",
          "        run: npm run test:ci-manifest\n",
          "manifest launcher shell",
        ),
      "npm run test:ci-manifest has an unexpected shell policy",
    ],
    [
      "full suite launcher with job defaults",
      (source: string) =>
        replaceExactlyOnce(
          replaceExactlyOnce(
            source,
            "  frontend-build:\n",
            "  frontend-build:\n    defaults:\n      run:\n        shell: bash\n",
            "frontend-build owner",
          ),
          "        shell: bash\n        run: npm run test:run\n",
          "        run: npm run test:run\n",
          "full suite launcher shell",
        ),
      "npm run test:run has an unexpected shell policy",
    ],
  ])("requires explicit bash for the %s", (_, transform, expectedError) => {
    const source = transform(realWorkflowSource)

    expect(() => assertSemanticWorkflowManifest(source)).toThrow(expectedError)
  })

  it("accepts a colon-shaped line in a block scalar", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "        run: npm run build\n",
      "        run: |\n          echo 'label: value'\n          npm run build\n",
      "frontend build command",
    )

    expect(source).toContain("          echo 'label: value'\n")
    expect(() => assertSemanticWorkflowManifest(source)).not.toThrow()
  })

  it("rejects an unknown YAML directive warning before conversion", () => {
    const workflowSource = realWorkflowSource
    const source = `%BAD_DIRECTIVE\n---\n${workflowSource}`

    expect(source).not.toBe(workflowSource)
    expect(source).toMatch(/^%BAD_DIRECTIVE\n---\n/)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "workflow YAML warning [BAD_DIRECTIVE]",
    )
  })

  it("fails a mutation when its owner marker is missing or duplicated", () => {
    expect(() => replaceExactlyOnce("alpha", "missing", "replacement", "fixture marker")).toThrow(
      "fixture marker must appear exactly once; found 0",
    )
    expect(() =>
      replaceExactlyOnce("alpha alpha", "alpha", "replacement", "fixture marker"),
    ).toThrow("fixture marker must appear exactly once; found 2")
  })

  it("rejects an anchored job expanded through more than 100 aliases", () => {
    const aliases = Array.from(
      { length: 101 },
      (_, index) => `  manifest-alias-${index + 1}: *manifest_base`,
    ).join("\n")
    const workflowSource = realWorkflowSource
    const source = replaceExactlyOnce(
      workflowSource,
      "\n  ci-summary:\n",
      `\n  manifest-anchor: &manifest_base\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo manifest anchor\n${aliases}\n\n  ci-summary:\n`,
      "ci-summary insertion point",
    )

    expect(source).not.toBe(workflowSource)
    expect(source).toContain("  manifest-anchor: &manifest_base\n")
    expect(source.match(/manifest-alias-\d+: \*manifest_base/g)).toHaveLength(101)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow()
  })

  it.each([
    ["an empty source", ""],
    ["malformed YAML", "jobs:\n  frontend-build: ["],
    ["missing jobs", "name: Manifest fixture\n"],
    ["a non-mapping jobs owner", "jobs: []\n"],
    ["an unresolved alias", "jobs:\n  frontend-build: *missing\n"],
  ])("rejects %s", (_, source) => {
    expect(() => assertSemanticWorkflowManifest(source)).toThrow()
  })

  it.each([
    [
      "missing frontend-build",
      (source: string) => replaceWorkflowJob(source, "frontend-build", "", "frontend-build job"),
      "jobs.frontend-build must be an object",
    ],
    [
      "a non-mapping frontend-build owner",
      (source: string) =>
        replaceWorkflowJob(
          source,
          "frontend-build",
          "  frontend-build: []\n",
          "frontend-build job",
        ),
      "jobs.frontend-build must be an object",
    ],
    [
      "a non-sequence frontend-build.steps owner",
      (source: string) =>
        replaceWorkflowJob(
          source,
          "frontend-build",
          "  frontend-build:\n    steps: {}\n",
          "frontend-build job",
        ),
      "jobs.frontend-build.steps must be an array",
    ],
  ])("rejects %s at its intended owner", (_, transform, expectedError) => {
    const source = transform(realWorkflowSource)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(expectedError)
  })

  it("rejects semantically complete duplicate jobs and frontend-build mappings", () => {
    const workflowSource = realWorkflowSource
    const duplicateJobs = `${replaceExactlyOnce(
      workflowSource,
      "\njobs:\n",
      "\njobs: &manifest_jobs\n",
      "jobs owner",
    ).trimEnd()}\n'jobs': *manifest_jobs\n`
    const anchoredFrontendBuild = replaceExactlyOnce(
      workflowSource,
      "  frontend-build:\n",
      "  frontend-build: &manifest_frontend_build\n",
      "frontend-build owner",
    )
    const duplicateFrontendBuild = replaceExactlyOnce(
      anchoredFrontendBuild,
      "\n  ci-summary:\n",
      "\n  'frontend-build': *manifest_frontend_build\n\n  ci-summary:\n",
      "ci-summary insertion point",
    )

    expect(duplicateJobs).not.toBe(workflowSource)
    expect(duplicateJobs).toContain("jobs: &manifest_jobs\n")
    expect(duplicateJobs).toContain("'jobs': *manifest_jobs\n")
    expect(duplicateFrontendBuild).not.toBe(workflowSource)
    expect(duplicateFrontendBuild).toContain("frontend-build: &manifest_frontend_build\n")
    expect(duplicateFrontendBuild).toContain("'frontend-build': *manifest_frontend_build\n")
    expect(() => assertSemanticWorkflowManifest(duplicateJobs)).toThrow()
    expect(() => assertSemanticWorkflowManifest(duplicateFrontendBuild)).toThrow()
  })

  it("rejects a required command moved to a sibling job", () => {
    const withoutFullSuiteStep = removeWorkflowStepByCommand(
      realWorkflowSource,
      "frontend-build",
      "npm run test:run",
      "full regular suite removal",
    )
    const source = replaceExactlyOnce(
      withoutFullSuiteStep,
      "\n  ci-summary:\n",
      "\n  manifest-sibling:\n    runs-on: ubuntu-latest\n    steps:\n      - working-directory: ./frontend\n        shell: bash\n        run: npm run test:run\n\n  ci-summary:\n",
      "ci-summary insertion point",
    )

    expect(source).toContain("  manifest-sibling:\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "npm run test:run must appear in exactly one frontend step",
    )
  })

  it("keeps the moved-to-sibling fixture live when full-suite presentation drifts", () => {
    const workflowSource = realWorkflowSource
    const driftedSource = replaceExactlyOnce(
      workflowSource,
      "      - name: Run full frontend test suite\n        working-directory: ./frontend\n        shell: bash\n        run: npm run test:run\n",
      "      - run: npm run test:run\n        env:\n          FULL_SUITE_MODE: manifest\n        working-directory: ./frontend\n        shell: bash\n        name: Run full frontend test suite\n",
      "full regular suite presentation drift",
    )
    const withoutFullSuiteStep = removeWorkflowStepByCommand(
      driftedSource,
      "frontend-build",
      "npm run test:run",
      "full regular suite removal",
    )

    expect(driftedSource).not.toBe(workflowSource)
    expect(() => assertSemanticWorkflowManifest(withoutFullSuiteStep)).toThrow(
      "npm run test:run must appear in exactly one frontend step",
    )
  })

  it("rejects a same-job heredoc decoy after the full regular suite is removed", () => {
    const withoutFullSuiteStep = replaceExactlyOnce(
      realWorkflowSource,
      "\n      - name: Run full frontend test suite\n        working-directory: ./frontend\n        shell: bash\n        run: npm run test:run\n",
      "",
      "full regular suite removal",
    )
    const source = replaceExactlyOnce(
      withoutFullSuiteStep,
      "\n      - name: Build frontend (static export)\n",
      "\n      - name: Preserve full-suite text as a shell heredoc\n        run: |\n          run: npm run test:run\n\n      - name: Build frontend (static export)\n",
      "frontend build insertion point",
    )

    expect(source).not.toContain("      - name: Run full frontend test suite\n")
    expect(source).toContain("          run: npm run test:run\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "npm run test:run must appear in exactly one frontend step",
    )
  })

  it("rejects missing and duplicate required semantic steps", () => {
    const source = realWorkflowSource
    const missing = replaceExactlyOnce(
      source,
      "        run: npm run test:run\n",
      "",
      "full regular suite command removal",
    )
    const duplicate = replaceExactlyOnce(
      source,
      "\n      - name: Run full frontend test suite\n",
      "\n      - name: Duplicate full frontend test suite\n        working-directory: ./frontend\n        shell: bash\n        run: npm run test:run\n\n      - name: Run full frontend test suite\n",
      "full regular suite duplicate insertion point",
    )

    expect(missing).not.toContain("        run: npm run test:run\n")
    expect(duplicate.match(/run: npm run test:run/g)).toHaveLength(2)
    expect(() => assertSemanticWorkflowManifest(missing)).toThrow(
      "npm run test:run must appear in exactly one frontend step",
    )
    expect(() => assertSemanticWorkflowManifest(duplicate)).toThrow(
      "npm run test:run must appear in exactly one frontend step",
    )
  })

  it("rejects a required step with the wrong working directory", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "      - name: Run full frontend test suite\n        working-directory: ./frontend\n",
      "      - name: Run full frontend test suite\n        working-directory: .\n",
      "full regular suite working directory",
    )

    expect(source).toContain("      - name: Run full frontend test suite\n        working-directory: .\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "npm run test:run must use ./frontend",
    )
  })

  it("rejects a required step-level condition", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "      - name: Run full frontend test suite\n",
      "      - name: Run full frontend test suite\n        if: github.event_name == 'schedule'\n",
      "full regular suite condition",
    )

    expect(source).toContain("        if: github.event_name == 'schedule'\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "npm run test:run must not set if",
    )
  })

  it("rejects custom shells on the Widget coverage step", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "      - name: Run widget regression tests with coverage\n        working-directory: ./frontend\n",
      "      - name: Run widget regression tests with coverage\n        working-directory: ./frontend\n        shell: echo {0}\n",
      "Widget coverage shell",
    )

    expect(source).toContain("        shell: echo {0}\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "npm run test:widget:coverage has an unexpected shell policy",
    )
  })

  it.each([
    [
      "test:ci-manifest",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "        shell: bash\n        run: npm run test:ci-manifest\n",
          "        shell: echo {0}\n        run: npm run test:ci-manifest\n",
          "manifest launcher shell",
        ),
    ],
    [
      "test:run",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "      - name: Run full frontend test suite\n        working-directory: ./frontend\n        shell: bash\n",
          "      - name: Run full frontend test suite\n        working-directory: ./frontend\n        shell: echo {0}\n",
          "full regular suite launcher shell",
        ),
    ],
  ])("rejects a non-bash %s launcher", (_, transform) => {
    const source = transform(realWorkflowSource)

    expect(source).toContain("        shell: echo {0}\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      `npm run ${_} has an unexpected shell policy`,
    )
  })

  it.each([
    [
      "a step continue-on-error",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "        run: npm run test:run\n",
          "        continue-on-error: true\n        run: npm run test:run\n",
          "full regular suite continue-on-error insertion",
        ),
      "frontend-build steps must not set continue-on-error",
    ],
    [
      "a job continue-on-error",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "  frontend-build:\n",
          "  frontend-build:\n    continue-on-error: true\n",
          "frontend-build continue-on-error insertion",
        ),
      "jobs.frontend-build must not set continue-on-error",
    ],
  ])("rejects %s", (_, transform, expectedError) => {
    const source = transform(realWorkflowSource)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(expectedError)
  })

  it.each([
    [
      "workflow root",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "jobs:\n",
          "defaults:\n  run:\n    shell: echo {0}\n\njobs:\n",
          "workflow jobs owner",
        ),
    ],
    [
      "jobs.frontend-build",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "  frontend-build:\n",
          "  frontend-build:\n    defaults:\n      run:\n        shell: echo {0}\n",
          "frontend-build owner",
        ),
    ],
  ])("rejects a custom default shell at %s scope", (owner, transform) => {
    const source = transform(realWorkflowSource)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      `${owner} defaults.run.shell must be bash when set`,
    )
  })

  it.each([
    [
      "a missing ci-summary",
      (source: string) => replaceWorkflowJob(source, "ci-summary", "", "ci-summary job"),
      "jobs.ci-summary must be an object",
    ],
    [
      "a non-mapping ci-summary",
      (source: string) =>
        replaceWorkflowJob(source, "ci-summary", "  ci-summary: []\n", "ci-summary job"),
      "jobs.ci-summary must be an object",
    ],
    [
      "non-array ci-summary needs",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "    needs:\n      - prepare-deepdoc-cache\n",
          "    needs: prepare-deepdoc-cache\n",
          "ci-summary needs owner",
        ),
      "jobs.ci-summary.needs must be an array",
    ],
  ])("rejects %s", (_, transform, expectedError) => {
    const source = transform(realWorkflowSource)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(expectedError)
  })

  it.each([
    [
      "plain",
      "\n\n  after-summary:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo after summary\n",
      (source: string) => source,
    ],
    [
      "single-quoted",
      "\n\n  'after-summary':\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo after summary\n",
      (source: string) => source,
    ],
    [
      "double-quoted",
      '\n\n  "after-summary":\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo after summary\n',
      (source: string) => source,
    ],
    [
      "anchored",
      "\n\n  # anchored following job\n  after-summary: &after_summary\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo after summary\n",
      (source: string) => source,
    ],
    [
      "aliased",
      "\n\n  # aliased following job\n  after-summary: *after_summary\n",
      (source: string) =>
        replaceExactlyOnce(
          source,
          "  prepare-deepdoc-cache:\n",
          "  prepare-deepdoc-cache: &after_summary\n",
          "prepare-deepdoc-cache anchor",
        ),
    ],
  ])("preserves a complete %s following sibling when mutating ci-summary", (_, block, prepare) => {
    const workflowSource = `${prepare(realWorkflowSource).trimEnd()}${block}`
    const source = replaceWorkflowJob(workflowSource, "ci-summary", "", "ci-summary job")

    expect(source.slice(-block.length)).toBe(block)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "jobs.ci-summary must be an object",
    )
  })

  it("rejects removing frontend-build from ci-summary.needs", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "      - frontend-build\n",
      "",
      "ci-summary frontend-build need",
    )

    expect(source).not.toContain("      - frontend-build\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "jobs.ci-summary.needs must contain frontend-build exactly once",
    )
  })

  it("rejects duplicate frontend-build entries in ci-summary.needs", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "      - frontend-build\n",
      "      - frontend-build\n      - frontend-build\n",
      "ci-summary frontend-build need",
    )

    expect(source.match(/^      - frontend-build$/gm)).toHaveLength(2)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "jobs.ci-summary.needs must contain frontend-build exactly once",
    )
  })

  it("rejects ci-summary job-level continue-on-error", () => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "  ci-summary:\n",
      "  ci-summary:\n    continue-on-error: true\n",
      "ci-summary owner",
    )

    expect(source).toContain("  ci-summary:\n    continue-on-error: true\n")
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "jobs.ci-summary must not set continue-on-error",
    )
  })

  it.each([
    [
      "missing always()",
      "if: github.event_name != 'pull_request' || github.event.pull_request.draft == false",
    ],
    ["a different condition", "if: always()"],
  ])("rejects ci-summary with %s", (_, replacement) => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "if: always() && (github.event_name != 'pull_request' || github.event.pull_request.draft == false)",
      replacement,
      "ci-summary if policy",
    )

    expect(source).toContain(`    ${replacement}\n`)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "jobs.ci-summary has an unexpected if policy",
    )
  })

  it("rejects missing and duplicate Check required jobs owner steps", () => {
    const workflowSource = realWorkflowSource
    const missing = replaceExactlyOnce(
      workflowSource,
      "      - name: Check required jobs\n",
      "      - name: Renamed required jobs\n",
      "ci-summary check step name",
    )
    const duplicate = replaceExactlyOnce(
      workflowSource,
      "      - name: Check required jobs\n",
      `      - name: Check required jobs\n        shell: bash\n        run: |\n          ${frontendSummaryCheckCommand}\n\n      - name: Check required jobs\n`,
      "ci-summary check step duplication",
    )

    expect(missing).not.toContain("      - name: Check required jobs\n")
    expect(duplicate.match(/name: Check required jobs/g)).toHaveLength(2)
    expect(() => assertSemanticWorkflowManifest(missing)).toThrow(
      "Check required jobs must appear in exactly one ci-summary step",
    )
    expect(() => assertSemanticWorkflowManifest(duplicate)).toThrow(
      "Check required jobs must appear in exactly one ci-summary step",
    )
  })

  it.each([
    ["a non-bash shell", "      - name: Check required jobs\n        shell: sh\n"],
    [
      "a step-level condition",
      "      - name: Check required jobs\n        shell: bash\n        if: success()\n",
    ],
    [
      "step-level continue-on-error",
      "      - name: Check required jobs\n        shell: bash\n        continue-on-error: true\n",
    ],
  ])("rejects the summary check step with %s", (_, replacement) => {
    const source = replaceExactlyOnce(
      realWorkflowSource,
      "      - name: Check required jobs\n        shell: bash\n        run: |\n",
      `${replacement}        run: |\n`,
      "ci-summary check step contract",
    )
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      _ === "a non-bash shell"
        ? "Check required jobs must use bash"
        : _ === "a step-level condition"
          ? "Check required jobs must not set if"
          : "Check required jobs must not set continue-on-error",
    )
  })

  it("rejects a missing, duplicate, or non-full-line frontend summary check", () => {
    const workflowSource = realWorkflowSource
    const checkLine = 'check_job "frontend-build" "${{ needs[\'frontend-build\'].result }}"'
    const indentedCheckLine = `          ${checkLine}\n`
    const missing = replaceExactlyOnce(
      workflowSource,
      indentedCheckLine,
      "",
      "frontend summary check line removal",
    )
    const duplicate = replaceExactlyOnce(
      workflowSource,
      indentedCheckLine,
      indentedCheckLine.repeat(2),
      "frontend summary check line duplication",
    )
    const embedded = replaceExactlyOnce(
      workflowSource,
      indentedCheckLine,
      `          echo '${checkLine}'\n`,
      "frontend summary check line embedding",
    )

    expect(missing).not.toContain(indentedCheckLine)
    expect(duplicate.match(/check_job "frontend-build"/g)).toHaveLength(2)
    expect(embedded).toContain(`          echo '${checkLine}'\n`)
    expect(() => assertSemanticWorkflowManifest(missing)).toThrow(
      "Check required jobs must check frontend-build exactly once",
    )
    expect(() => assertSemanticWorkflowManifest(duplicate)).toThrow(
      "Check required jobs must check frontend-build exactly once",
    )
    expect(() => assertSemanticWorkflowManifest(embedded)).toThrow(
      "Check required jobs must check frontend-build exactly once",
    )
  })

  it("does not accept a frontend summary check decoy outside the owned step", () => {
    const workflowSource = realWorkflowSource
    const checkLine = 'check_job "frontend-build" "${{ needs[\'frontend-build\'].result }}"'
    const withoutOwnedCheck = replaceExactlyOnce(
      workflowSource,
      `          ${checkLine}\n`,
      "",
      "owned frontend summary check line",
    )
    const source = replaceExactlyOnce(
      withoutOwnedCheck,
      '          exit "$failed"\n',
      `          exit "$failed"\n\n      - name: Preserve frontend summary text outside owner\n        shell: bash\n        run: |\n          ${checkLine}\n`,
      "ci-summary decoy insertion point",
    )

    expect(source).toContain("      - name: Preserve frontend summary text outside owner\n")
    expect(source.match(/check_job "frontend-build"/g)).toHaveLength(1)
    expect(() => assertSemanticWorkflowManifest(source)).toThrow(
      "Check required jobs must check frontend-build exactly once",
    )
  })

  const buildRegularSuiteFixture = () => {
    const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8")) as {
      scripts: Record<string, string>
    }
    return {
      config: { ...vitestConfig, test: { ...vitestConfig.test } } as Record<string, unknown>,
      scripts: { ...packageJson.scripts },
      rootEntryNames: [...frontendRootEntryNames],
    }
  }

  it.each([
    ["a missing test:run launcher", (scripts: Record<string, string>) => delete scripts["test:run"]],
    ["a wrong test:run launcher", (scripts: Record<string, string>) => (scripts["test:run"] = "vitest")],
  ])("rejects %s at the regular launcher owner", (_, mutate) => {
    const fixture = buildRegularSuiteFixture()
    mutate(fixture.scripts)

    expect(() => assertRegularSuiteDiscovery(fixture.config, fixture.scripts, fixture.rootEntryNames)).toThrow(
      "regular launcher must keep test:run as vitest run",
    )
  })

  it.each([
    ["a wrong include", (test: Record<string, unknown>) => (test.include = ["src/**/*.test.ts"])],
    ["any exclude", (test: Record<string, unknown>) => (test.exclude = [])],
    ["truthy passWithNoTests", (test: Record<string, unknown>) => (test.passWithNoTests = true)],
  ])("rejects %s at the regular base discovery owner", (_, mutate) => {
    const fixture = buildRegularSuiteFixture()
    mutate(fixture.config.test as Record<string, unknown>)

    expect(() => assertRegularSuiteDiscovery(fixture.config, fixture.scripts, fixture.rootEntryNames)).toThrow(
      "regular base discovery must preserve automatic discovery",
    )
  })

  it.each([
    ["a top-level root", "top-level", "root", "src"],
    ["a test-level root", "test", "root", "src"],
    ["a test-level dir", "test", "dir", "src"],
    ["a top-level test name pattern", "top-level", "testNamePattern", "frontend CI test manifest"],
    ["a test-level test name pattern", "test", "testNamePattern", "frontend CI test manifest"],
    ["a top-level related selector", "top-level", "related", ["src/components/pages"]],
    ["a test-level related selector", "test", "related", ["src/components/pages"]],
    ["a top-level changed selector", "top-level", "changed", true],
    ["a test-level changed selector", "test", "changed", true],
    ["a top-level partial shard", "top-level", "shard", { index: 1, count: 2 }],
    ["a test-level partial shard", "test", "shard", { index: 1, count: 2 }],
    ["a top-level project selector", "top-level", "project", "focused"],
    ["a test-level project selector", "test", "project", "focused"],
    ["a top-level filters selector", "top-level", "filters", ["src"]],
    ["a test-level filters selector", "test", "filters", ["src"]],
    ["a top-level cliExclude selector", "top-level", "cliExclude", ["src/components"]],
    ["a test-level cliExclude selector", "test", "cliExclude", ["src/components"]],
    ["a truthy top-level standalone", "top-level", "standalone", true],
    ["a truthy test-level standalone", "test", "standalone", true],
    ["a truthy top-level allowOnly", "top-level", "allowOnly", true],
    ["a truthy test-level allowOnly", "test", "allowOnly", true],
  ])("rejects %s at the regular selection owner", (_, location, key, value) => {
    const fixture = buildRegularSuiteFixture()
    const owner = location === "test" ? fixture.config.test : fixture.config
    ;(owner as Record<string, unknown>)[key as string] = value

    expect(() => assertRegularSuiteDiscovery(fixture.config, fixture.scripts, fixture.rootEntryNames)).toThrow(
      "regular execution must be selection-neutral",
    )
  })

  it.each([
    ["top-level workspace", "top-level"],
    ["test workspace", "test"],
  ])("rejects %s at the regular workspace/project graph owner", (_, location) => {
    const fixture = buildRegularSuiteFixture()
    const owner = location === "test" ? fixture.config.test : fixture.config
    ;(owner as Record<string, unknown>).workspace = "vitest.workspace.ts"

    expect(() => assertRegularSuiteDiscovery(fixture.config, fixture.scripts, fixture.rootEntryNames)).toThrow(
      "regular workspace/project graph must be disabled",
    )
  })

  it.each(recognizedWorkspaceProjectFilenames)(
    "rejects the recognized workspace/project filename %s at the graph owner",
    (filename) => {
      const fixture = buildRegularSuiteFixture()
      fixture.rootEntryNames.push(filename)

      expect(() =>
        assertRegularSuiteDiscovery(fixture.config, fixture.scripts, fixture.rootEntryNames),
      ).toThrow("regular workspace/project graph must be disabled")
    },
  )

  it.each([
    ["top-level standalone false", "top-level", "standalone", false],
    ["top-level standalone zero", "top-level", "standalone", 0],
    ["top-level standalone empty string", "top-level", "standalone", ""],
    ["test standalone false", "test", "standalone", false],
    ["test standalone zero", "test", "standalone", 0],
    ["test standalone empty string", "test", "standalone", ""],
    ["top-level allowOnly false", "top-level", "allowOnly", false],
    ["top-level allowOnly zero", "top-level", "allowOnly", 0],
    ["top-level allowOnly empty string", "top-level", "allowOnly", ""],
    ["test allowOnly false", "test", "allowOnly", false],
    ["test allowOnly zero", "test", "allowOnly", 0],
    ["test allowOnly empty string", "test", "allowOnly", ""],
  ])("accepts falsey %s", (_, location, key, value) => {
    const fixture = buildRegularSuiteFixture()
    const owner = location === "test" ? fixture.config.test : fixture.config
    ;(owner as Record<string, unknown>)[key as string] = value

    expect(() =>
      assertRegularSuiteDiscovery(fixture.config, fixture.scripts, fixture.rootEntryNames),
    ).not.toThrow()
  })

  it("keeps package launchers and regular Vitest discovery contracts source-locked", () => {
    const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8")) as {
      scripts: Record<string, string>
    }

    expect(packageJson.scripts["test:ci-manifest"]).toBe(manifestCommand)
    assertRegularSuiteDiscovery(vitestConfig, packageJson.scripts, frontendRootEntryNames)
  })
})
