/**
 * Central configuration for the ChronosAudit Express API.
 * Values are resolved from the environment (or a local `.env` file).
 */
import path from "node:path";
import dotenv from "dotenv";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.resolve(__dirname, "..", ".env") });

export interface Config {
  port: number;
  host: string;
  workerBaseUrl: string;
  workerApiToken: string;
  dataDir: string;
  uploadDir: string;
  metaDir: string;
  webDir: string;
  maxUploadBytes: number;
}

function envInt(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const parsed = parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function absPrefix(...parts: string[]): string {
  return path.resolve(path.join(__dirname, "..", ...parts));
}

export const config: Config = {
  port: envInt("PORT", 4000),
  host: process.env.HOST ?? "0.0.0.0",
  workerBaseUrl: (process.env.WORKER_BASE_URL ?? "http://localhost:8100").replace(/\/+$/, ""),
  workerApiToken: process.env.WORKER_API_TOKEN ?? "",
  dataDir: path.resolve(process.env.DATA_DIR ?? absPrefix("data")),
  uploadDir: path.resolve(
    process.env.UPLOAD_DIR ?? path.join(path.resolve(process.env.DATA_DIR ?? absPrefix("data")), "uploads")
  ),
  metaDir: path.resolve(
    process.env.META_DIR ?? path.join(path.resolve(process.env.DATA_DIR ?? absPrefix("data")), "meta")
  ),
  webDir: path.resolve(process.env.WEB_DIR ?? absPrefix("..", "..", "web")),
  maxUploadBytes: envInt("MAX_UPLOAD_BYTES", 50 * 1024 * 1024),
};