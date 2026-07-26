import { execFileSync } from "node:child_process"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const extensionRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const repositoryRoot = resolve(extensionRoot, "..")

export function resolveExtensionVersion({
  xagentVersion = process.env.XAGENT_VERSION,
  gitDescribe,
} = {}) {
  if (xagentVersion?.trim()) {
    return fromXagentVersion(xagentVersion.trim())
  }
  const description =
    gitDescribe ??
    execFileSync(
      "git",
      [
        "describe",
        "--tags",
        "--match",
        "v[0-9]*",
        "--long",
        "--dirty",
      ],
      {
        cwd: repositoryRoot,
        encoding: "utf8",
      },
    ).trim()
  return fromGitDescribe(description)
}

export function fromGitDescribe(description) {
  const match =
    /^v?(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?-(\d+)-g([0-9a-f]+)(-dirty)?$/i.exec(
      description.trim(),
    )
  if (!match) {
    throw new Error(
      `Cannot derive the Browser Relay version from git describe output: ${description}`,
    )
  }
  const [, major, minor, patch, postRaw, distanceRaw, revision, dirtyRaw] =
    match
  const post = Number(postRaw ?? 0)
  const distance = Number(distanceRaw)
  const dirty = Boolean(dirtyRaw)
  const release = [major, minor, patch].map(Number)

  if (distance === 0 && !dirty) {
    return buildVersion({
      chromeParts: post > 0 ? [...release, post] : release,
      displayVersion: `${release.join(".")}${post > 0 ? `.post${post}` : ""}`,
      release: true,
    })
  }

  const build = post + Math.max(distance, 1)
  const nextRelease = [release[0], release[1], release[2] + 1].join(".")
  return buildVersion({
    chromeParts: [...release, build],
    displayVersion: `${nextRelease}.dev${distance}+g${revision.toLowerCase()}${
      dirty ? ".dirty" : ""
    }`,
    release: false,
  })
}

export function fromXagentVersion(rawVersion) {
  const version = rawVersion.replace(/^v/, "")
  const match =
    /^(\d+)\.(\d+)\.(\d+)(?:\.post(\d+)|\.dev(\d+))?(?:\+([0-9A-Za-z.-]+))?$/.exec(
      version,
    )
  if (!match) {
    throw new Error(
      `XAGENT_VERSION must be a numeric release or PEP 440 development version, got: ${rawVersion}`,
    )
  }
  const [, majorRaw, minorRaw, patchRaw, postRaw, devRaw] = match
  const major = Number(majorRaw)
  const minor = Number(minorRaw)
  const patch = Number(patchRaw)
  if (devRaw !== undefined) {
    if (patch === 0) {
      throw new Error(
        "Development XAGENT_VERSION values with a zero patch require git metadata.",
      )
    }
    return buildVersion({
      chromeParts: [major, minor, patch - 1, Math.max(1, Number(devRaw))],
      displayVersion: version,
      release: false,
    })
  }
  return buildVersion({
    chromeParts:
      postRaw === undefined
        ? [major, minor, patch]
        : [major, minor, patch, Number(postRaw)],
    displayVersion: version,
    release: true,
  })
}

function buildVersion({ chromeParts, displayVersion, release }) {
  if (
    chromeParts.length < 1 ||
    chromeParts.length > 4 ||
    chromeParts.some(
      (part) => !Number.isInteger(part) || part < 0 || part > 65_535,
    ) ||
    chromeParts.every((part) => part === 0)
  ) {
    throw new Error(
      `Xagent version cannot be represented as a Chrome extension version: ${chromeParts.join(
        ".",
      )}`,
    )
  }
  const chromeVersion = chromeParts.join(".")
  return {
    chromeVersion,
    displayVersion,
    artifactVersion: displayVersion.replaceAll("+", "-"),
    release,
  }
}
