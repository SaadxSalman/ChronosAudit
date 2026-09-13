/** Small formatting / validation helpers shared by API modules. */

/** Clamp an integer from arbitrary input. */
export function clampInt(value: unknown, fallback: number, min: number, max: number): number {
  const parsed = parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

/** Parse "YYYY-MM-DD" → Date or undefined (UTC-midnight safe). */
export function parseIsoDate(text: string | undefined | null): Date | undefined {
  if (!text) return undefined;
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(text);
  if (!m) return undefined;
  return new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])));
}

/** Format a date as YYYY-MM-DD. */
export function formatIsoDate(date: Date): string {
  return date.toISOString().slice(0, 10);
}

/** Human-friendly window label from two dates. */
export function windowLabel(start?: string | Date | null, end?: string | Date | null): string {
  const s = start ? (start instanceof Date ? formatIsoDate(start) : start) : "…";
  const e = end ? (end instanceof Date ? formatIsoDate(end) : end) : "…";
  return s === e ? `${s} (as of)` : `${s} → ${e}`;
}

/** Deterministic short hash used to key graph nodes in the UI. */
export function shortHash(text: string, length = 8): string {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = (h * 16777619) & 0xffffffff;
  }
  return `n${(h >>> 0).toString(16).padStart(8, "0").slice(0, length)}`;
}