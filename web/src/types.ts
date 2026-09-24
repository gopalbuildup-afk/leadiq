export type Band = "P1" | "P2" | "P3";

export interface Reason {
  en: string;
  hi: string;
}

export interface ScoreResult {
  lead_id: string;
  asof: string;
  probability: number;
  band: Band;
  reasons: Reason[];
  model_version: string;
  duplicate_cluster_size: number;
}

export interface BatchScoreResponse {
  results: ScoreResult[];
  count: number;
}

export interface SimilarLead {
  lead_id: string;
  similarity: number;
  outcome: 0 | 1;
}

export interface SimilarLeadsResponse {
  lead_id: string;
  model_version: string;
  k: number;
  similar: SimilarLead[];
  admissions: number;
  total: number;
}

export interface ModelCardMetrics {
  pr_auc?: number;
  ece_10bin?: number;
  brier_score?: number;
  roc_auc?: number;
  [key: string]: number | undefined;
}

export interface ModelCardResponse {
  version: string;
  training_window: string;
  data_sha256: string;
  metrics: ModelCardMetrics;
  feature_list: string[];
  status: "champion" | "challenger" | "retired";
  created_at?: string;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  database_configured: boolean;
  model_loaded: boolean;
  champion_version?: string;
}

export interface LeadMetadata {
  lead_id: string;
  full_name: string | null;
  source: string;
  branch: string;
  course_interest: string;
  age: number | null;
  home_locality: string | null;
  phone: string | null;
  email: string | null;
  created_at: string;
  assigned_counselor_id: string | null;
}

export interface LeadListResponse {
  leads: LeadMetadata[];
  count: number;
  total: number;
}

export interface CallListItem {
  metadata: LeadMetadata;
  score: ScoreResult;
  /**
   * True if P1 and waiting longer than 15 minutes without a call.
   * Computed on the client using created_at / call history.
   */
  urgent: boolean;
  /**
   * Hours since enquiry arrived.
   */
  age_hours: number;
  /**
   * Optional cluster link id.
   */
  cluster_id?: string;
}

export type Language = "en" | "hi";

export interface Filters {
  branch: Set<string>;
  source: Set<string>;
  band: Set<Band>;
  search: string;
}
