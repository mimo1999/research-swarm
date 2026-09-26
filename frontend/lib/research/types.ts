export type NodeUpdate = { node: string; update: Record<string, unknown>; ts: number };

export type Finding = {
  id: string;
  claim: string;
  confidence: number;
  sub_question: string;
  evidence?: unknown[];
};

export type Critique = {
  finding_id: string | null;
  verdict: "supported" | "weak" | "refuted";
  reasoning: string;
};

export type ReportSection = { heading: string; body_md: string; citations: number[] };

export type SourceType = "web" | "arxiv" | "pubmed" | "pdf" | "retriever";

export type Source = {
  url: string;
  title: string;
  snippet: string;
  source_type: SourceType;
  credibility_score: number;
};

// Embedding-based scoring. `overall` is a Python @property, not a model
// field, so it never reaches the wire -- compute it client-side (see
// lib/research/reportMetrics.ts) rather than expecting it here.
export type QualityScore = {
  faithfulness: number;
  relevance: number | null;
  completeness: number | null;
};

export type JudgeVerdict = "approve" | "revise" | "reject";

// Independent LLM read of the finished report. Same "no `overall` on the
// wire" caveat as QualityScore.
export type LLMJudgeResult = {
  coherence: number;
  relevance: number;
  completeness: number;
  citation_quality: number;
  verdict: JudgeVerdict;
  reasoning: string;
};

export type FinalReport = {
  title: string;
  exec_summary: string;
  sections: ReportSection[];
  references: Source[];
  methodology: string;
  limitations: string;
  quality_score: QualityScore | null;
  llm_judge: LLMJudgeResult | null;
};

export type RunState =
  | { status: "idle" }
  | { status: "running"; sessionId: string; trace: NodeUpdate[] }
  | {
      status: "interrupted";
      sessionId: string;
      trace: NodeUpdate[];
      findings: Finding[];
      critiques: Critique[];
    }
  | { status: "done"; sessionId: string; trace: NodeUpdate[]; report: FinalReport }
  | { status: "failed"; sessionId: string; trace: NodeUpdate[] };

export type RunError = { message: string } | null;

export type ConfigOptions = {
  providers: string[];
  models: { anthropic: string[]; openai: string[]; ollama_cloud: string[] };
  depths: string[];
  defaults: {
    provider: string;
    model: string;
    max_sources: number;
    ollama_url: string;
    ollama_model: string;
    ollama_cloud_model: string;
    ollama_deployment: string;
  };
};

export type OllamaStatus = {
  reachable: boolean;
  model_pulled: boolean;
  logged_in: boolean | null;
  message: string;
};

export type SessionSummary = {
  thread_id: string;
  created_at: string | null;
  updated_at: string | null;
  step_count: number;
  has_report: boolean;
};
