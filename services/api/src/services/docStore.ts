/**
 * Persistent document registry on the Express side.
 *
 * Each uploaded PDF is written to ``data/uploads`` and its record (mirroring the
 * worker's DocumentRecord) is persisted to ``data/meta/documents.json`` so the
 * dashboard survives API restarts even before the worker finishes indexing.
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import { config } from "../config.js";
import { ApiError, DocumentRecord } from "../types.js";

const META_FILE = "documents.json";
let records: DocumentRecord[] = [];

export async function initDocStore(): Promise<void> {
  await fs.mkdir(config.dataDir, { recursive: true });
  await fs.mkdir(config.uploadDir, { recursive: true });
  await fs.mkdir(config.metaDir, { recursive: true });
  const metaPath = path.join(config.metaDir, META_FILE);
  try {
    const raw = await fs.readFile(metaPath, "utf-8");
    records = JSON.parse(raw) as DocumentRecord[];
  } catch {
    records = [];
  }
}

async function persist(): Promise<void> {
  await fs.writeFile(
    path.join(config.metaDir, META_FILE),
    JSON.stringify(records, null, 2),
    "utf-8"
  );
}

export function listDocuments(): DocumentRecord[] {
  return [...records].sort((a, b) => (b.created_at ?? "").localeCompare(a.created_at ?? ""));
}

export function getDocument(id: string): DocumentRecord | undefined {
  return records.find((r) => r.id === id);
}

/** Full reload of one record from the worker (source of truth for status). */
export async function syncFromWorker(id: string): Promise<DocumentRecord> {
  const { workerClient } = await import("./workerClient.js");
  const fresh = await workerClient.documentStatus(id);
  const existing = getDocument(id);
  const merged: DocumentRecord = { ...existing, ...(fresh as DocumentRecord), id };
  const idx = records.findIndex((r) => r.id === id);
  if (idx >= 0) records[idx] = merged;
  else records.push(merged);
  await persist();
  return merged;
}

export async function upsertRecord(record: DocumentRecord): Promise<DocumentRecord> {
  const idx = records.findIndex((r) => r.id === record.id);
  if (idx >= 0) records[idx] = { ...records[idx], ...record };
  else records.push(record);
  await persist();
  return idx >= 0 ? records[idx] : record;
}

export async function removeDocument(id: string): Promise<void> {
  const rec = getDocument(id);
  if (rec) {
    try {
      await fs.unlink(rec.path);
    } catch {
      /* file may already be gone */
    }
  }
  records = records.filter((r) => r.id !== id);
  await persist();
}

export async function waitForIndexed(
  id: string,
  timeoutSeconds = 120,
  onProgress?: (record: DocumentRecord) => void
): Promise<DocumentRecord> {
  const deadline = Date.now() + timeoutSeconds * 1000;
  let last = getDocument(id) ?? (await syncing(id));
  while (Date.now() < deadline) {
    if (last.status === "indexed" || last.status === "failed") return last;
    await new Promise((r) => setTimeout(r, 700));
    last = await syncing(id);
    onProgress?.(last);
  }
  return last;
}

export async function syncing(id: string): Promise<DocumentRecord> {
  try {
    return await syncFromWorker(id);
  } catch {
    return getDocument(id) ?? (await upsertRecord({
      id,
      filename: "unknown",
      path: "",
      pages: 0,
      chunk_count: 0,
      entity_count: 0,
      relation_count: 0,
      size_bytes: 0,
      status: "failed",
      progress: 0,
      stage_detail: "worker unreachable",
      error: "worker unreachable",
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }));
  }
}

export function requireDocument(id: string): DocumentRecord {
  const rec = getDocument(id);
  if (!rec) throw new ApiError(404, `No such document: ${id}`);
  return rec;
}