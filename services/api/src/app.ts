/** Express application wiring: middleware, static web, routers, error handler. */
import { promises as fsp } from "node:fs";
import path from "node:path";
import express, { ErrorRequestHandler, NextFunction, Request, Response } from "express";
import cors from "cors";
import { config } from "./config.js";
import { ApiError } from "./types.js";
import { healthRouter } from "./routes/health.js";
import { documentsRouter, handleMulterError } from "./routes/documents.js";
import { queriesRouter } from "./routes/queries.js";
import { graphRouter } from "./routes/graph.js";
import { initDocStore } from "./services/docStore.js";

export async function createApp() {
  await initDocStore();
  const app = express();

  app.use(cors());
  app.use(express.json({ limit: "5mb" }));

  // Banner
  app.get("/", (_req, res) => {
    res
      .status(200)
      .type("text/html")
      .send(
        "<h1>ChronosAudit API</h1><p>See <a href='/ui'>the web dashboard</a> or the " +
          "<a href='/api/health'>health endpoint</a>.</p>"
      );
  });

  // REST facade
  app.use("/api", healthRouter);
  app.use("/api/documents", documentsRouter);
  app.use("/api/query", queriesRouter);
  app.use("/api/graph", graphRouter);
  app.get("/api/system/health", async (_req, res) => {
    res.json({ ok: true, service: "chronos-audit-api", upstream: config.workerBaseUrl });
  });

  // Static web dashboard (built once, served as-is)
  try {
    await fsp.stat(path.join(config.webDir, "index.html"));
    app.use("/ui", express.static(config.webDir));
    app.use("/ui/*", express.static(config.webDir));
    app.get("/ui/sample-query", (_req, res) => res.redirect("/ui"));
  } catch {
    app.get("/ui", (_req, res) =>
      res.status(200).type("text/plain").send("Web dashboard not found. Set WEB_DIR or run from the repo root.")
    );
  }

  // 404
  app.use((_req, res) => {
    res.status(404).json({ error: "Not found" });
  });

  // Central error handler
  app.use(errorHandler);
  return app;
}

const errorHandler: ErrorRequestHandler = (err: unknown, _req: Request, res: Response, _next: NextFunction) => {
  if (res.headersSent) {
    return;
  }
  const multerErr = handleMulterError(err);
  if (multerErr) {
    res.status(multerErr.status).json({ error: multerErr.message });
    return;
  }
  if (err instanceof ApiError) {
    res.status(err.status).json({ error: err.message });
    return;
  }
  const message = err instanceof Error ? err.message : String(err);
  console.error("[api] unhandled error:", message);
  res.status(502).json({ error: "Upstream error", detail: message.slice(0, 300) });
};