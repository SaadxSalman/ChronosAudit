/** Health & system routes — no upstream dependency. */
import { Router } from "express";
import { config } from "../config.js";
import { asyncHandler } from "../types.js";
import { workerClient } from "../services/workerClient.js";

export const healthRouter = Router();

healthRouter.get(
  "/health",
  asyncHandler(async (req, res) => {
    let worker: Record<string, unknown> | null = null;
    let workerOk = false;
    try {
      worker = await workerClient.health();
      workerOk = Boolean(worker.ok);
    } catch {
      worker = null;
    }
    res.json({
      ok: workerOk,
      service: "chronos-audit-api",
      version: "1.0.0",
      worker_upstream: workerOk,
      worker: worker,
      timestamp: new Date().toISOString(),
    });
  })
);

healthRouter.get(
  "/system/info",
  asyncHandler(async (_req, res) => {
    let upstream: Record<string, unknown> | null = null;
    try {
      upstream = await workerClient.systemInfo();
    } catch {
      upstream = null;
    }
    res.json({
      api_port: config.port,
      worker_base_url: config.workerBaseUrl,
      upstream: upstream,
    });
  })
);