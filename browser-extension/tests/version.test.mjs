import assert from "node:assert/strict"
import test from "node:test"

import {
  fromGitDescribe,
  fromXagentVersion,
} from "../scripts/version.mjs"

test("release tags produce the same Browser Relay release version", () => {
  assert.deepEqual(fromGitDescribe("v0.6.4-0-gabc1234"), {
    chromeVersion: "0.6.4",
    displayVersion: "0.6.4",
    artifactVersion: "0.6.4",
    release: true,
  })
  assert.deepEqual(fromGitDescribe("v0.6.0.post1-0-gabc1234"), {
    chromeVersion: "0.6.0.1",
    displayVersion: "0.6.0.post1",
    artifactVersion: "0.6.0.post1",
    release: true,
  })
})

test("development builds add a descriptive suffix and monotonic Chrome build", () => {
  assert.deepEqual(fromGitDescribe("v0.6.3-29-g675f9f21"), {
    chromeVersion: "0.6.3.29",
    displayVersion: "0.6.4.dev29+g675f9f21",
    artifactVersion: "0.6.4.dev29-g675f9f21",
    release: false,
  })
  assert.deepEqual(fromGitDescribe("v0.6.3-0-g675f9f21-dirty"), {
    chromeVersion: "0.6.3.1",
    displayVersion: "0.6.4.dev0+g675f9f21.dirty",
    artifactVersion: "0.6.4.dev0-g675f9f21.dirty",
    release: false,
  })
})

test("explicit Xagent release versions override git-derived metadata", () => {
  assert.deepEqual(fromXagentVersion("v0.7.0"), {
    chromeVersion: "0.7.0",
    displayVersion: "0.7.0",
    artifactVersion: "0.7.0",
    release: true,
  })
  assert.deepEqual(fromXagentVersion("0.6.4.dev29+g675f9f21"), {
    chromeVersion: "0.6.3.29",
    displayVersion: "0.6.4.dev29+g675f9f21",
    artifactVersion: "0.6.4.dev29-g675f9f21",
    release: false,
  })
})
