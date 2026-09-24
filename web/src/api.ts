import type {
  BatchScoreResponse,
  CallListItem,
  HealthResponse,
  LeadMetadata,
  ModelCardResponse,
  ScoreResult,
  SimilarLeadsResponse,
} from "./types";

const BASE_URL =
  (import.meta as any).env?.VITE_LEADIQ_API_URL ||
  (typeof window !== "undefined" &&
  window.location.hostname === "localhost"
    ? "/api"
    : "/api");

// Upper bound for any single API call. Scoring 500 enquiries takes ~10s;
// anything beyond this means the service is stuck, and the UI must show
// an error instead of loading forever.
const REQUEST_TIMEOUT_MS = 120_000;

// Dedupe identical in-flight requests. React StrictMode mounts every page
// twice in dev, and an impatient Refresh click fires another full batch
// while one is running. The single uvicorn worker serves those one by one,
// so without this the call list pays 2x latency. Concurrent identical
// calls share one promise; sequential calls are unaffected.
const inflight = new Map<string, Promise<unknown>>();

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const method = init?.method ?? "GET";
  const key = `${method} ${path} ${typeof init?.body === "string" ? init.body : ""}`;
  const existing = inflight.get(key);
  if (existing) return existing as Promise<T>;
  const p = doRequest<T>(path, init).finally(() => {
    if (inflight.get(key) === p) inflight.delete(key);
  });
  inflight.set(key, p);
  return p;
}

async function doRequest<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      credentials: "omit",
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      ...init,
    });
  } catch (err: any) {
    if (err?.name === "TimeoutError" || err?.name === "AbortError") {
      throw new Error(
        "The scoring service is taking too long to respond. Please retry."
      );
    }
    throw new Error(
      "Couldn't reach the scoring service. Make sure the API is running, then retry."
    );
  }
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export async function scoreLead(leadId: string): Promise<ScoreResult> {
  return request<ScoreResult>("/v1/score", {
    method: "POST",
    body: JSON.stringify({ lead_id: leadId }),
  });
}

export async function scoreBatch(
  leadIds: string[]
): Promise<BatchScoreResponse> {
  return request<BatchScoreResponse>("/v1/score/batch", {
    method: "POST",
    body: JSON.stringify({ lead_ids: leadIds }),
  });
}

export async function getModelCard(): Promise<ModelCardResponse> {
  return request<ModelCardResponse>("/v1/model", { method: "GET" });
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/healthz", { method: "GET" });
}

export async function getSimilarLeads(
  leadId: string,
  k = 10
): Promise<SimilarLeadsResponse> {
  const url = `/v1/leads/${encodeURIComponent(leadId)}/similar?k=${k}`;
  return request<SimilarLeadsResponse>(url, { method: "GET" });
}

export interface LoadLeadsResult {
  leads: LeadMetadata[];
  count: number;
  total: number;
}

/**
 * Load enquiry metadata from the backend (`GET /v1/leads`).
 * Supports pagination via limit and offset.
 */
export async function loadLeadMetadatas(
  limit: number = 100,
  offset: number = 0
): Promise<LoadLeadsResult> {
  const data = await request<{ leads: LeadMetadata[]; count?: number; total?: number }>(
    `/v1/leads?limit=${limit}&offset=${offset}`,
    { method: "GET" }
  );
  if (!data.leads || (data.leads.length === 0 && offset === 0)) {
    throw new Error("The enquiry list is empty.");
  }
  return {
    leads: data.leads || [],
    count: data.count ?? data.leads?.length ?? 0,
    total: data.total ?? data.leads?.length ?? 0,
  };
}

/**
 * Load one enquiry's metadata for the detail page.
 */
export async function loadLeadMetadata(leadId: string): Promise<LeadMetadata> {
  return request<LeadMetadata>(
    `/v1/leads/${encodeURIComponent(leadId)}`,
    { method: "GET" }
  );
}

/**
 * Score enquiries through the Module 5 batch endpoint.
 *
 * Never falls back to fabricated scores: a counselor call list built on
 * fake priorities would send counselors after the wrong enquiries, so an
 * API failure surfaces as an error the UI shows with a retry button.
 *
 * Scores stream in chunks of 100: `onPartial` fires after each chunk so
 * the list paints its first rows in well under a second.
 * Caches scored leads in `scoreCache` when provided to make page-switching instant.
 */
export async function scoreMetadatas(
  metadatas: LeadMetadata[],
  onPartial?: (items: CallListItem[]) => void,
  scoreCache?: Map<string, ScoreResult>
): Promise<CallListItem[]> {
  const scoresById = new Map<string, ScoreResult>();

  // Check cache for existing scores
  const uncached: LeadMetadata[] = [];
  for (const m of metadatas) {
    if (scoreCache && scoreCache.has(m.lead_id)) {
      scoresById.set(m.lead_id, scoreCache.get(m.lead_id)!);
    } else {
      uncached.push(m);
    }
  }

  // If some or all are cached, show them right away
  if (scoresById.size > 0 && onPartial) {
    onPartial(finalizeItems(metadatas, scoresById));
  }

  if (uncached.length > 0) {
    const CHUNK = 100;
    for (let i = 0; i < uncached.length; i += CHUNK) {
      const slice = uncached.slice(i, i + CHUNK);
      const response = await scoreBatch(slice.map((m) => m.lead_id));
      for (const s of response.results) {
        scoresById.set(s.lead_id, s);
        if (scoreCache) {
          scoreCache.set(s.lead_id, s);
        }
      }
      if (onPartial) onPartial(finalizeItems(metadatas, scoresById));
    }
  }

  const items = finalizeItems(metadatas, scoresById);
  if (items.length === 0) {
    throw new Error("The scoring service returned no scores.");
  }
  return items;
}

function finalizeItems(
  metadatas: LeadMetadata[],
  scoresById: Map<string, ScoreResult>
): CallListItem[] {
  const now = Date.now();
  return metadatas
    .map((m) => {
      const score = scoresById.get(m.lead_id);
      if (!score) return null;
      const createdMs = parseFlexibleDate(m.created_at).getTime();
      const ageHours = Math.max(
        0,
        Math.min(24 * 30 * 12, (now - createdMs) / (1000 * 60 * 60))
      );
      const waitingMinutes = ageHours * 60;
      const urgent =
        score.band === "P1" && waitingMinutes > 15;
      return {
        metadata: m,
        score,
        urgent,
        age_hours: ageHours,
      } satisfies CallListItem;
    })
    .filter((x): x is CallListItem => x !== null)
    .sort(
      (a, b) =>
        b.score.probability - a.score.probability ||
        (b.urgent ? 1 : 0) - (a.urgent ? 1 : 0)
    );
}

export function parseFlexibleDate(s: string | null | undefined): Date {
  if (!s) return new Date(0);
  const str = String(s).trim();
  if (!str) return new Date(0);

  // ISO with T
  if (str.includes("T")) {
    const d = new Date(str);
    if (!isNaN(d.getTime())) return d;
  }

  // Naive IST: 2025-09-01 00:58:10 — parse parts directly so we never hit
  // browser-dependent quirks with "+05:30" concatenation.
  const match = str.match(
    /^(\d{4})-(\d{1,2})-(\d{1,2})[ T](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?/
  );
  if (match) {
    const [, y, mo, d, h, mi, se] = match;
    const ms = Date.UTC(
      Number(y),
      Number(mo) - 1,
      Number(d),
      Number(h) - 5, // convert IST (+05:30) → UTC
      Number(mi) - 30,
      se ? Number(se) : 0
    );
    const parsed = new Date(ms);
    if (!isNaN(parsed.getTime())) return parsed;
  }

  // Any locale-ish fallback
  const d = new Date(str);
  if (!isNaN(d.getTime())) return d;
  return new Date(Date.now());
}
