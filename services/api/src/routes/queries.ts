/** Temporal Q&A routes — thin proxies to the worker's retrieval engine. */
import { Router } from "express";
import { asyncHandler } from "../types.js";
import { workerClient } from "../services/workerClient.js";

export const queriesRouter = Router();

// ---------------------------------------------------------------- single
queriesRouter.post(
  "/",
  asyncHandler(async (req, res) => {
    const question = String(req.body?.question ?? "").trim();
    if (!question) {
      res.status(422).json({ error: "question is required" });
      return;
    }
    const window = req.body?.window ? String(req.body.window) : undefined;
    const topK = clampInt(req.body?.top_k, 6, 1, 20);
    const result = await workerClient.query(question, window, topK);
    res.json(result);
  })
);

// ---------------------------------------------------------------- compare
queriesRouter.post(
  "/compare",
  asyncHandler(async (req, res) => {
    const question = String(req.body?.question ?? "").trim();
    const windowA = String(req.body?.window_a ?? "").trim();
    const windowB = String(req.body?.window_b ?? "").trim();
    if (!question || !windowA || !windowB) {
      res.status(422).json({ error: "question, window_a and window_b are required" });
      return;
    }
    const topK = clampInt(req.body?.top_k, 6, 1, 20);
    const result = await workerClient.compare(question, windowA, windowB, topK);
    res.json(result);
  })
);

// ---------------------------------------------------------------- graph diff
queriesRouter.post(
  "/diff",
  asyncHandler(async (req, res) => {
    const windowA = String(req.body?.window_a ?? "").trim();
    const windowB = String(req.body?.window_b ?? "").trim();
    if (!windowA || !windowB) {
      res.status(422).json({ error: "window_a and window_b are required" });
      return;
    }
    res.json(await workerClient.graphDiff(windowA, windowB));
  })
);

function clampInt(value: unknown, fallback: number, min: number, max: number): number {
  const parsed = parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}