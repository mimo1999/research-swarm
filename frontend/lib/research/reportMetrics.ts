import type { QualityScore, LLMJudgeResult } from "./types";

// Mirrors ReportQualityScore.overall / LLMJudgeResult.overall in
// research_swarm/schemas/report.py + judge.py. Both are plain Python
// @property (not @computed_field), so they never get serialized into the
// API payload -- computed here from the same raw fields instead.

export function qualityOverall(qs: QualityScore): number {
  const scores = [qs.faithfulness, qs.relevance, qs.completeness].filter(
    (v): v is number => v !== null
  );
  if (scores.length === 0) return 0;
  return scores.reduce((a, b) => a + b, 0) / scores.length;
}

export function judgeOverall(j: LLMJudgeResult): number {
  return (j.coherence + j.relevance + j.completeness + j.citation_quality) / 4;
}

export function fmtPct(v: number): string {
  return `${Math.round(v * 100)}%`;
}
