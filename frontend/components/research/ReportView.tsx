"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { FileText, BookOpen } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from "@/components/ui/accordion";
import { VerdictBadge } from "@/components/research/verdict-badge";
import { CitationMark, MarkdownLink } from "@/components/research/citation-mark";
import { linkifyCitations, inlineCitationNumbers } from "@/lib/research/citations";
import { qualityOverall, judgeOverall, fmtPct } from "@/lib/research/reportMetrics";
import { cn } from "@/lib/utils";
import type { FinalReport, Source } from "@/lib/research/types";

const PROSE = "max-w-[65ch] text-sm leading-relaxed";

export function ReportView({ report }: { report: FinalReport }) {
  const [activeRef, setActiveRef] = useState<number | null>(null);
  const focusTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => {
    if (focusTimeout.current) clearTimeout(focusTimeout.current);
  }, []);

  function onFocusReference(n: number) {
    setActiveRef(n);
    document.getElementById(`ref-${n}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    if (focusTimeout.current) clearTimeout(focusTimeout.current);
    focusTimeout.current = setTimeout(() => setActiveRef(null), 2000);
  }

  const markdownComponents: Components = useMemo(
    () => ({
      h1: ({ children }) => <h4 className="mt-4 mb-1.5 text-sm font-semibold first:mt-0">{children}</h4>,
      h2: ({ children }) => <h4 className="mt-4 mb-1.5 text-sm font-semibold first:mt-0">{children}</h4>,
      h3: ({ children }) => <h4 className="mt-4 mb-1.5 text-sm font-semibold first:mt-0">{children}</h4>,
      p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
      ul: ({ children }) => <ul className="mb-3 list-disc space-y-1 pl-5">{children}</ul>,
      ol: ({ children }) => <ol className="mb-3 list-decimal space-y-1 pl-5">{children}</ol>,
      li: ({ children }) => <li>{children}</li>,
      strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
      blockquote: ({ children }) => (
        <blockquote className="border-l-2 border-border pl-3 text-muted-foreground italic">{children}</blockquote>
      ),
      code: ({ children }) => <code className="rounded bg-muted px-1 py-0.5 text-xs">{children}</code>,
      table: ({ children }) => (
        <div className="mb-3 overflow-x-auto">
          <table className="w-full text-sm">{children}</table>
        </div>
      ),
      th: ({ children }) => <th className="border-b py-1 pr-3 text-left font-medium">{children}</th>,
      td: ({ children }) => <td className="border-b py-1 pr-3 align-top">{children}</td>,
      a: ({ href, children }) => (
        <MarkdownLink href={href} references={report.references} onFocusReference={onFocusReference}>
          {children}
        </MarkdownLink>
      ),
    }),
    [report.references]
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-xl">
          <FileText className="size-4.5 text-muted-foreground" />
          {report.title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-8">
        {(report.quality_score || report.llm_judge) && (
          <QualitySummary qualityScore={report.quality_score} judge={report.llm_judge} />
        )}

        <section>
          <h3 className="mb-2 text-base font-semibold">Executive summary</h3>
          <div className={PROSE}>
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {linkifyCitations(report.exec_summary)}
            </ReactMarkdown>
          </div>
        </section>

        {report.sections.map((s, i) => {
          const inline = inlineCitationNumbers(s.body_md);
          const orphaned = s.citations.filter((c) => !inline.has(c));
          return (
            <section key={i}>
              <h3 className="mb-2 text-base font-semibold">
                {i + 1}. {s.heading}
              </h3>
              <div className={PROSE}>
                <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                  {linkifyCitations(s.body_md)}
                </ReactMarkdown>
              </div>
              {orphaned.length > 0 && (
                <div className="mt-1 flex flex-wrap items-center gap-1 text-xs text-muted-foreground">
                  Also drawn from:
                  {orphaned.map((n) => (
                    <CitationMark key={n} n={n} references={report.references} onFocusReference={onFocusReference} />
                  ))}
                </div>
              )}
            </section>
          );
        })}

        {report.references.length > 0 && (
          <section>
            <h3 className="mb-3 flex items-center gap-2 text-base font-semibold">
              <BookOpen className="size-4 text-muted-foreground" />
              References ({report.references.length})
            </h3>
            <ol className="space-y-3">
              {report.references.map((r, i) => (
                <ReferenceRow key={i} n={i + 1} source={r} active={activeRef === i + 1} />
              ))}
            </ol>
          </section>
        )}

        {(report.methodology || report.limitations) && (
          <Accordion type="single" collapsible>
            <AccordionItem value="notes">
              <AccordionTrigger className="text-sm">Methodology &amp; limitations</AccordionTrigger>
              <AccordionContent className="space-y-4">
                {report.methodology && (
                  <div>
                    <h4 className="mb-1 font-medium">Methodology</h4>
                    <p className={cn(PROSE, "whitespace-pre-wrap")}>{report.methodology}</p>
                  </div>
                )}
                {report.limitations && (
                  <div>
                    <h4 className="mb-1 font-medium">Limitations</h4>
                    <p className={cn(PROSE, "whitespace-pre-wrap")}>{report.limitations}</p>
                  </div>
                )}
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        )}
      </CardContent>
    </Card>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-base font-semibold tabular-nums">{value}</p>
    </div>
  );
}

function QualitySummary({
  qualityScore,
  judge,
}: {
  qualityScore: FinalReport["quality_score"];
  judge: FinalReport["llm_judge"];
}) {
  return (
    <section className="space-y-4 rounded-lg border p-4">
      {qualityScore && (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="Faithfulness" value={fmtPct(qualityScore.faithfulness)} />
          <Stat
            label="Relevance"
            value={qualityScore.relevance !== null ? fmtPct(qualityScore.relevance) : "Not computed"}
          />
          <Stat
            label="Completeness"
            value={qualityScore.completeness !== null ? fmtPct(qualityScore.completeness) : "Not computed"}
          />
          <Stat label="Overall" value={fmtPct(qualityOverall(qualityScore))} />
        </div>
      )}

      {judge && (
        <div className={cn("space-y-3", qualityScore && "border-t pt-4")}>
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">LLM judge</span>
            <VerdictBadge verdict={judge.verdict} />
            <span className="text-sm text-muted-foreground">{judgeOverall(judge).toFixed(1)}/5</span>
          </div>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Coherence" value={`${judge.coherence}/5`} />
            <Stat label="Relevance" value={`${judge.relevance}/5`} />
            <Stat label="Completeness" value={`${judge.completeness}/5`} />
            <Stat label="Citation quality" value={`${judge.citation_quality}/5`} />
          </div>
          <Accordion type="single" collapsible>
            <AccordionItem value="judge-reasoning">
              <AccordionTrigger className="text-sm">Judge reasoning</AccordionTrigger>
              <AccordionContent className="text-sm text-muted-foreground">{judge.reasoning}</AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>
      )}
    </section>
  );
}

function ReferenceRow({ n, source, active }: { n: number; source: Source; active: boolean }) {
  return (
    <li
      id={`ref-${n}`}
      className={cn(
        "scroll-mt-24 rounded-lg border border-transparent p-2 text-sm transition-colors",
        active && "border-primary/40 bg-primary/5"
      )}
    >
      <div className="flex items-start gap-2">
        <span className="mt-0.5 shrink-0 text-xs font-medium text-muted-foreground">[{n}]</span>
        <div className="min-w-0 flex-1">
          <a
            href={source.url}
            target="_blank"
            rel="noreferrer"
            className="font-medium break-words underline underline-offset-2 hover:text-foreground"
          >
            {source.title || source.url}
          </a>
          {source.snippet && <p className="mt-0.5 text-muted-foreground line-clamp-2">{source.snippet}</p>}
          <p className="mt-1 text-xs text-muted-foreground">
            <span className="capitalize">{source.source_type}</span>
            {" · "}
            {Math.round(source.credibility_score * 100)}% credibility
          </p>
        </div>
      </div>
    </li>
  );
}
