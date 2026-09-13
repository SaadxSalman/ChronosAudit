import { test } from "node:test";
import assert from "node:assert/strict";

import { clampInt, formatIsoDate, parseIsoDate } from "../utils/format.js";
import { config } from "../config.js";

test("clampInt clamps and falls back", () => {
  assert.equal(clampInt("50", 6, 1, 20), 20);
  assert.equal(clampInt("abc", 6, 1, 20), 6);
  assert.equal(clampInt("3", 6, 1, 20), 3);
  assert.equal(clampInt(undefined, 6, 1, 20), 6);
});

test("parseIsoDate accepts YYYY-MM-DD only", () => {
  const d = parseIsoDate("2024-07-01");
  assert.ok(d);
  assert.equal(d?.toISOString().slice(0, 10), "2024-07-01");
  assert.equal(parseIsoDate("July 2024"), undefined);
  assert.equal(parseIsoDate(undefined), undefined);
});

test("formatIsoDate round-trips", () => {
  assert.equal(formatIsoDate(new Date(Date.UTC(2024, 0, 5))), "2024-01-05");
});

test("config loads sane defaults", () => {
  assert.ok(config.port > 0);
  assert.ok(config.workerBaseUrl.startsWith("http"));
  assert.ok(config.maxUploadBytes > 0);
});