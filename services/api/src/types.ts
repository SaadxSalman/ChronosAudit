/** Shared JSON contracts between the Express API and its consumers. */

import type { Request, Response } from "express";

/** Mirror of the worker's DocumentRecord (see services/worker/chronos/models.py). */
export interface DocumentRecord {
  id: string;
  filename: string;
  path: string;
  pages: number;
  chunk_count: number;
  entity_count: number;
  relation_count: number;
  size_bytes: number;
  status:
    | "uploaded"
    | "queued"
    | "parsing"
    | "chunking"
    | "extracting"
    | "embedding"
    | "indexing"
    | "indexed"
    | "failed";
  progress: number;
  stage_detail: string;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

/** Proxied /graph/stats payload. */
export interface GraphStats {
  node_count: number;
  edge_count: number;
  document_count: number;
  entity_types: Record<string, number>;
  predicates: Record<string, number>;
  temporally_validated_edges: number;
  open_ended_edges: number;
  temporal_range_start?: string | null;
  temporal_range_end?: string | null;
  chunk_vector_count: number;
  vector_dim: number;
}

/** Proxied /query payload (AnswerResponse from the worker). */
export interface AnswerResponse {
  question: string;
  answer: string;
  citations: Citation[];
  facts: Fact[];
  context?: RetrievalContext | undefined;
  engine: string;
  generated_at: string;
}

export interface Citation {
  key: string;
  kind?: string;
  doc_id?: string;
  doc_name?: string;
  page?: number;
  chunk_id?: string;
  validity?: string;
  score?: number;
  snippet?: string;
  fact?: string;
  evidence?: string;
}

export interface Fact {
  subject: string;
  predicate: string;
  object: string;
  validity?: string;
  start?: string;
  end?: string;
  doc_id: string;
  chunk_id: string;
  evidence: string;
}

export interface RetrievalContext {
  question: string;
  window_start?: string | null;
  window_end?: string | null;
  window_label: string;
  hits: VectorHit[];
  graph_edges: GraphEdge[];
  seed_entities: string[];
  trace: string[];
}

export interface VectorHit {
  chunk_id: string;
  doc_id: string;
  doc_name: string;
  text: string;
  page: number;
  start?: string | null;
  end?: string | null;
  start_label?: string | null;
  end_label?: string | null;
  score: number;
  entities: string[];
}

export interface GraphEdge {
  subject: string;
  predicate: string;
  object: string;
  start?: string | null;
  end?: string | null;
  start_label?: string | null;
  end_label?: string | null;
  doc_id: string;
  chunk_id: string;
  evidence: string;
  weight: number;
}

export interface GraphSnapshot {
  valid_at?: string | null;
  start?: string | null;
  end?: string | null;
  nodes: NodeInfo[];
  edges: GraphEdge[];
  edge_count: number;
  node_count: number;
}

export interface NodeInfo {
  id: string;
  name: string;
  type: string;
  doc_ids: string[];
  mention_count: number;
  degree: number;
}

export interface CompareResponse {
  question: string;
  window_a_label: string;
  window_b_label: string;
  comparison: Comparison;
  context_a?: RetrievalContext | undefined;
  context_b?: RetrievalContext | undefined;
  narrative: string;
  engine: string;
  generated_at: string;
}

export interface Comparison {
  window_start?: string | null;
  window_end?: string | null;
  window_label: string;
  added: GraphEdge[];
  removed: GraphEdge[];
  changed: { before: GraphEdge; after: GraphEdge }[];
  unchanged: GraphEdge[];
  facts: string[];
}

export interface SystemInfo {
  version: string;
  worker_mode: string;
  model_provider: string;
  extraction_model: string;
  embedding_provider: string;
  embedding_dim: number;
  chunk_size_tokens: number;
  chunk_overlap_tokens: number;
  lance_db_path: string;
  graph_store_path: string;
}

export interface WorkerHealth {
  ok: boolean;
  service: string;
  mode: string;
  model_provider: string;
  embedding_provider: string;
  embedding_dim: number;
  lancedb_chunks: number;
  graph_nodes: number;
  graph_edges: number;
  documents: number;
  broker: string;
}

/** Async handler wrapper so controllers stay clean of try/catch boilerplate. */
export const asyncHandler =
  <T>(fn: (req: Request, res: Response) => Promise<T>) =>
  (req: Request, res: Response) => {
    fn(req, res).catch((err) => {
      const status = err instanceof ApiError ? err.status : 502;
      const message = err instanceof ApiError ? err.message : "Upstream worker error";
      const detail = err instanceof ApiError ? undefined : String(err.message ?? err);
      if (!res.headersSent) {
        res.status(status).json({ error: message, detail });
      }
    });
  };

/** Typed API error. */
export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}