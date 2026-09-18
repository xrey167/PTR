import assert from "node:assert/strict";
import test from "node:test";
import { PtrClient, PtrHttpError } from "../src/index.js";

function response(status, value) {
  return {
    ok: status >= 200 && status < 300,
    status,
    async json() { return value; },
    async text() { return typeof value === "string" ? value : JSON.stringify(value); },
  };
}

test("request uses versioned JSON endpoint", async () => {
  const calls = [];
  const client = new PtrClient({
    baseUrl: "http://localhost:8080/",
    fetch: async (url, init) => {
      calls.push({ url, init });
      return response(200, { id: "r1", revision: 7, text: "ok" });
    },
  });
  const result = await client.request({ id: "r1", text: "hello" });
  assert.deepEqual(result, { id: "r1", revision: 7, text: "ok" });
  assert.equal(calls[0].url, "http://localhost:8080/v1/requests");
  assert.equal(calls[0].init.method, "POST");
});

test("non-success response becomes PtrHttpError", async () => {
  const client = new PtrClient({
    baseUrl: "http://localhost",
    fetch: async () => response(409, "stale revision"),
  });
  await assert.rejects(
    client.request({ id: "r1", text: "hello" }),
    (error) => error instanceof PtrHttpError && error.status === 409,
  );
});
