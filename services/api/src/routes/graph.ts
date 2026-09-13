/** Knowledge-graph explorer routes — proxy to the worker's NetworkX graph. */
import { Router } from "express";
import { asyncHandler } from "../types.js";
import { workerClient } from "../services/workerClient.js";

export const graphRouter = Router();

graphRouter.get(
  "/stats",
  asyncHandler(async (_req, res) => {
    res.json(await workerClient.graphStats());
  })
);

graphRouter.get(
  "/snapshot",
  asyncHandler(async (req, res) => {
    const maxEdges = clampInt(req.query.max_edges, 500, 1, 5000);
    res.json(
      await workerClient.graphSnapshot({
        validAt: cleanUndef(req.query.valid_at),
        start: cleanUndef(req.query.start),
        end: cleanUndef(req.query.end),
        maxEdges,
      })
    );
  })
);

graphRouter.get(
  "/neighbors",
  asyncHandler(async (req, res) => {
    const entity = String(req.query.entity ?? "");
    if (!entity) {
      res.status(422).json({ error: "entity query parameter is required" });
      return;
    }
    res.json(
      await workerClient.graphNeighbors(
        entity,
        cleanUndef(req.query.valid_at),
        clampInt(req.query.hop, 2, 1, 3)
      )
    );
  })
);

graphRouter.get(
  "/evolution",
  asyncHandler(async (req, res) => {
    const entity = String(req.query.entity ?? "");
    if (!entity) {
      res.status(422).json({ error: "entity query parameter is required" });
      return;
    }
    res.json(
      await workerClient.graphEvolution(entity, cleanUndef(req.query.predicate as string | undefined))
    );
  })
);

graphRouter.get(
  "/diff",
  asyncHandler(async (req, res) => {
    const windowA = String(req.query.window_a ?? "");
    const windowB = String(req.query.window_b ?? "");
    if (!windowA || !windowB) {
      res.status(422).json({ error: "window_a and window_b query parameters are required" });
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

function cleanUndef(value: unknown): string | undefined {
  const s = typeof value === "string" ? value : "";
  return s.length > 0 ? s : undefined;
}