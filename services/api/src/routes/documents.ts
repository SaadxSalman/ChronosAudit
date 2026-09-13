/** Document upload + ingestion routes. */
import { promises as fsp } from "node:fs";
import { randomUUID } from "node:crypto";
import path from "node:path";
import { Router } from "express";
import multer from "multer";
import { config } from "../config.js";
import { ApiError, asyncHandler, DocumentRecord } from "../types.js";
import {
  getDocument,
  listDocuments,
  removeDocument,
  requireDocument,
  syncFromWorker,
  upsertRecord,
} from "../services/docStore.js";
import { workerClient } from "../services/workerClient.js";

const upload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, config.uploadDir),
    filename: (_req, file, cb) => {
      const id = randomUUID().replace(/-/g, "").slice(0, 12);
      cb(null, `${id}_${file.originalname.replace(/[^\w.\-]/g, "_")}`);
    },
  }),
  limits: { fileSize: config.maxUploadBytes, files: 1 },
  fileFilter: (_req, file, cb) => {
    if (file.mimetype === "application/pdf" || file.originalname.toLowerCase().endsWith(".pdf")) {
      cb(null, true);
    } else {
      cb(new ApiError(415, "Only PDF files are accepted"));
    }
  },
});

export const documentsRouter = Router();

// ---------------------------------------------------------------- upload
documentsRouter.post(
  "/upload",
  upload.single("file"),
  asyncHandler(async (req, res) => {
    if (!req.file) throw new ApiError(400, "Missing multipart field 'file'");
    const id = `doc-${randomUUID().replace(/-/g, "").slice(0, 12)}`;
    const absolutePath = path.resolve(req.file.path);
    const stat = await fsp.stat(absolutePath);
    const record: DocumentRecord = {
      id,
      filename: req.file.originalname,
      path: absolutePath,
      pages: 0,
      chunk_count: 0,
      entity_count: 0,
      relation_count: 0,
      size_bytes: stat.size,
      status: "uploaded",
      progress: 0,
      stage_detail: "uploaded; waiting for ingestion",
      error: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    await upsertRecord(record);

    const wait = String(req.body?.wait ?? "false").toLowerCase() === "true";
    const timeoutSeconds = Number(req.body?.timeout_seconds ?? 120);

    // Kick off ingestion on the worker (sync mode runs it in-process).
    const ingest = await workerClient.ingest(id, record.filename, absolutePath, false);

    if (!wait) {
      res.status(202).json({ id, filename: record.filename, status: "queued", ingest });
      return;
    }

    let finalRecord = await waitForIndexedWithTimeout(id, timeoutSeconds);
    finalRecord = await syncFromWorker(id);
    res.status(201).json(finalRecord);
  })
);

async function waitForIndexedWithTimeout(id: string, timeoutSeconds: number): Promise<DocumentRecord> {
  const { waitForIndexed } = await import("../services/docStore.js");
  return waitForIndexed(id, timeoutSeconds);
}

// ---------------------------------------------------------------- CRUD
documentsRouter.get(
  "/",
  asyncHandler(async (_req, res) => {
    res.json(listDocuments());
  })
);

documentsRouter.get(
  "/:id",
  asyncHandler(async (req, res) => {
    const local = getDocument(req.params.id);
    try {
      const fresh = await syncFromWorker(req.params.id);
      res.json(fresh);
    } catch {
      if (local) return void res.json(local);
      throw new ApiError(404, `No such document: ${req.params.id}`);
    }
  })
);

documentsRouter.get(
  "/:id/status",
  asyncHandler(async (req, res) => {
    const rec = getDocument(req.params.id);
    try {
      const fresh = await syncFromWorker(req.params.id);
      res.json(fresh);
    } catch {
      if (rec) return void res.json(rec);
      throw new ApiError(404, `No such document: ${req.params.id}`);
    }
  })
);

documentsRouter.post(
  "/:id/reindex",
  asyncHandler(async (req, res) => {
    const rec = requireDocument(req.params.id);
    const ingest = await workerClient.ingest(rec.id, rec.filename, rec.path, false);
    res.json({ id: rec.id, queued: true, ingest });
  })
);

documentsRouter.delete(
  "/:id",
  asyncHandler(async (req, res) => {
    const rec = getDocument(req.params.id);
    try {
      await workerClient.deleteDocument(req.params.id);
    } catch {
      /* the worker may not know the doc; clean local state anyway */
    }
    await removeDocument(req.params.id);
    res.json({ deleted: req.params.id, filename: rec?.filename });
  })
);

// Inject the shared error converter for multer errors
export function handleMulterError(err: unknown): ApiError | null {
  if (err instanceof multer.MulterError) {
    if (err.code === "LIMIT_FILE_SIZE") return new ApiError(413, "File exceeds the maximum upload size");
    return new ApiError(400, `Upload error: ${err.code}`);
  }
  if (err instanceof ApiError) return err;
  return null;
}