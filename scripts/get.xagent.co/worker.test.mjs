// Functional tests for the get.xagent.co Worker. Uses Node's built-in test
// runner (no dependencies) and stubs global fetch. Run: node --test
import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import worker from "./worker.js";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

const get = (path = "/") => worker.fetch(new Request(`https://get.xagent.co${path}`));

test("non-root paths 404", async () => {
  const res = await get("/favicon.ico");
  assert.equal(res.status, 404);
});

test("fails closed (502) when the latest release can't be resolved", async () => {
  globalThis.fetch = async () => new Response("boom", { status: 500 });
  const res = await get("/");
  assert.equal(res.status, 502);
});

test("fails closed (502) when the release tag lacks the script", async () => {
  globalThis.fetch = async (url) =>
    String(url).includes("/releases/latest")
      ? new Response(JSON.stringify({ tag_name: "v9.9.9" }), { status: 200 })
      : new Response("not found", { status: 404 });
  const res = await get("/");
  assert.equal(res.status, 502);
});

test("serves the script from the resolved release tag", async () => {
  globalThis.fetch = async (url) =>
    String(url).includes("/releases/latest")
      ? new Response(JSON.stringify({ tag_name: "v1.2.3" }), { status: 200 })
      : new Response("#!/bin/sh\necho hi\n", { status: 200 });
  const res = await get("/");
  assert.equal(res.status, 200);
  assert.equal(res.headers.get("content-type"), "text/plain; charset=utf-8");
  assert.equal(res.headers.get("x-xagent-install-ref"), "v1.2.3");
  assert.match(await res.text(), /echo hi/);
});

test("fails closed (502) when the release lookup rejects (e.g. timeout abort)", async () => {
  // Exercises latestReleaseTag()'s try/catch (an AbortSignal.timeout abort or
  // malformed JSON lands here) -> null -> 502.
  globalThis.fetch = async () => {
    throw new Error("boom");
  };
  const res = await get("/");
  assert.equal(res.status, 502);
});

test("fails closed (502) when the script fetch rejects (e.g. timeout abort)", async () => {
  // Tag resolves, then the raw fetch rejects -> the handler's outer try/catch.
  globalThis.fetch = async (url) => {
    if (String(url).includes("/releases/latest")) {
      return new Response(JSON.stringify({ tag_name: "v1.2.3" }), { status: 200 });
    }
    throw new Error("boom");
  };
  const res = await get("/");
  assert.equal(res.status, 502);
});
