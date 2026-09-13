/**
 * Minimal HTTP client for the Python worker bridge (FastAPI :8100).
 * Uses Node's built-in fetch with bounded timeouts + a single retry.
 */
import { config } from "../config.js";
import { ApiError } from "../types.js";

const RETRIABLE = [502, 503, 504];
const TIMEOUT_MS = 12_000;

async function rawRequest<T>(method: "GET" | "POST" | "DELETE", path: string, body?: unknown): Promise<T> {
  const url = `${config.workerBaseUrl}${path}`;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (config.workerApiToken) headers["Authorization"] = `Bearer ${config.workerApiToken}`;

  const attempt = async (): Promise<T> => {
    const res = await Promise.race([fetchWith(url, method, headers, body), timeout(url)]);
    const text = await Promise.race([res.text(), timeout(url)]);
    let json: unknown;
    try {
      json = text ? JSON.parse(text) : {};
    } catch {
      json = { raw: text.slice(0, 300) };
    }
    if (res.ok) return json as T;
    if (RETRIABLE.includes(res.status)) throw new ApiError(res.status, `Worker ${path}: ${res.status}`);
    throw new ApiError(res.status, `Worker error ${res.status} on ${path}`);
  };

  try {
    return await attempt();
  } catch (err) {
    // Retry once for transient upstream failures (network blips, busy loop)
    if (err instanceof ApiError && RETRIABLE.includes(err.status)) return attempt();
    throw err;
  }
}

const fetchWith = (url: string, method: string, headers: Record<string, string>, body?: unknown) =>
  fetch(url, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });

const timeout = (url: string): Promise<never> =>
  new Promise((_, reject) =>
    setTimeout(() => reject(new ApiError(504, `Worker request timed out: ${url}`)), TIMEOUT_MS)
  );

export interface IngestResult {
  doc_id: string;
  queued: boolean;
  mode: "sync" | "celery";
  status?: string;
  error?: string | null;
}

export const workerClient = {
  health: () => rawRequest<Record<string, unknown>>("GET", "/health"),

  systemInfo: () => rawRequest<Record<string, unknown>>("GET", "/system/info"),

  ingest: (docId: string, filename: string, absolutePath: string, forceSync = false) =>
    rawRequest<IngestResult>("POST", "/documents/ingest", {
      doc_id: docId,
      filename,
      path: absolutePath,
      async_: !forceSync,
    }),

  documentStatus: (docId: string) => rawRequest<Record<string, unknown>>("GET", `/documents/${docId}/status`),

  listDocuments: () => rawRequest<Record<string, unknown>[]>("GET", "/documents"),

  deleteDocument: (docId: string) => rawRequest<{ deleted: string }>("DELETE", `/documents/${docId}`),

  query: (question: string, window: string | undefined, topK = 6) =>
    rawRequest<Record<string, unknown>>("POST", "/query", {
      question,
      window: window || null,
      top_k: topK,
    }).then(serialize),

  compare: (question: string, windowA: string, windowB: string, topK = 6) =>
    rawRequest<Record<string, unknown>>("POST", "/query/compare", {
      question,
      window_a: windowA,
      window_b: windowB,
      top_k: topK,
    }).then(serialize),

  graphStats: () => rawRequest<Record<string, unknown>>("GET", "/graph/stats"),

  graphSnapshot: (params: { validAt?: string; start?: string; end?: string; maxEdges?: number }) => {
    const qp = new URLSearchParams();
    if (params.validAt) qp.set("valid_at", params.validAt);
    if (params.start) qp.set("start", params.start);
    if (params.end) qp.set("end", params.end);
    if (params.maxEdges) qp.set("max_edges", String(params.maxEdges));
    const qs = qp.toString();
    return rawRequest<Record<string, unknown>>("GET", `/graph/snapshot${qs ? `?${qs}` : ""}`);
  },

  graphNeighbors: (entity: string, validAt?: string, hop = 2) => {
    const qp = new URLSearchParams({ entity, hop: String(hop) });
    if (validAt) qp.set("valid_at", validAt);
    return rawRequest<Record<string, unknown>>("GET", `/graph/neighbors?${qp.toString()}`);
  },

  graphEvolution: (entity: string, predicate?: string) => {
    const qp = new URLSearchParams({ entity });
    if (predicate) qp.set("predicate", predicate);
    return rawRequest<Record<string, unknown>>("GET", `/graph/evolution?${qp.toString()}`);
  },

  graphDiff: (windowA: string, windowB: string) =>
    rawRequest<Record<string, unknown>>("POST", "/graph/diff", { window_a: windowA, window_b: windowB }),
};

/** Dates arrive as ISO strings; keep them JSON-safe. */
function serialize<T>(payload: T): T {
  return JSON.parse(JSON.stringify(payload)) as T;
}