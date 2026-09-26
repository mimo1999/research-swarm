"use client";

import { Route, ClipboardList, Loader2 } from "lucide-react";
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from "@/components/ui/accordion";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { NodeMarker } from "@/components/research/node-meta";
import { VerdictBadge } from "@/components/research/verdict-badge";
import { nodeConfig, connectorLabel } from "@/lib/research/nodeConfig";
import { buildTraceSegments, type TraceSegment } from "@/lib/research/traceSegments";
import type { NodeUpdate } from "@/lib/research/types";

export function TraceView({ trace, live }: { trace: NodeUpdate[]; live: boolean }) {
  const segments = buildTraceSegments(trace);

  return (
    <div className="space-y-3">
      <div>
        <h3 className="flex items-center gap-2 text-lg font-semibold">
          <Route className="size-4.5 text-muted-foreground" />
          Agent trace
        </h3>
        <p className="text-sm text-muted-foreground">The path each agent takes through the research graph.</p>
      </div>

      <div className="relative">
        {(segments.length > 0 || live) && (
          <div className="absolute left-3.5 top-1 bottom-1 w-px bg-border" aria-hidden />
        )}
        <div className="space-y-4">
          {segments.map((segment, i) => (
            <SegmentRow key={`${segment.node}-${i}`} segment={segment} />
          ))}

          {live && (
            <div className="flex items-center gap-4">
              <div className="relative z-10 flex size-7 shrink-0 items-center justify-center">
                <span className="flex size-5 items-center justify-center rounded-full bg-background text-muted-foreground">
                  <Loader2 className="size-3 animate-spin" />
                </span>
              </div>
              <p className="text-xs text-muted-foreground">Waiting for the next agent…</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SegmentRow({ segment }: { segment: TraceSegment }) {
  if (segment.kind === "connector") {
    const lastUpdate = segment.entries[segment.entries.length - 1].update;
    return (
      <div className="flex items-center gap-4">
        <NodeMarker node={segment.node} kind="connector" />
        <p className="text-xs text-muted-foreground">{connectorLabel(segment.node, lastUpdate)}</p>
      </div>
    );
  }

  const { label } = nodeConfig(segment.node);
  return (
    <div className="flex gap-4">
      <NodeMarker node={segment.node} kind="stop" />
      <Card className="flex-1">
        <CardContent className="pt-4">
          <p className="mb-2 font-medium">{label}</p>
          <StopBody node={segment.node} entries={segment.entries} />
        </CardContent>
      </Card>
    </div>
  );
}

function StopBody({ node, entries }: { node: string; entries: NodeUpdate[] }) {
  switch (node) {
    case "supervisor":
      return <PlanBody u={entries[0].update as any} />;
    case "worker_node":
    case "document_worker_node":
    case "researcher":
      return <FindingsBody entries={entries} />;
    case "critic":
      return <CritiqueBody entries={entries} />;
    case "fact_checker":
      return <FactCheckBody entries={entries} />;
    case "writer":
      return <WriterBody u={entries[entries.length - 1].update as any} />;
    default:
      return <RawBody entries={entries} />;
  }
}

function PlanBody({ u }: { u: any }) {
  const subQs: string[] = u.plan?.sub_questions ?? [];
  if (subQs.length === 0) {
    return <p className="text-sm text-muted-foreground">Planning the research approach…</p>;
  }
  return (
    <Accordion type="single" collapsible>
      <AccordionItem value="plan">
        <AccordionTrigger className="gap-1.5 text-sm">
          <ClipboardList className="size-3.5 text-muted-foreground" />
          Research plan ({subQs.length} sub-questions)
        </AccordionTrigger>
        <AccordionContent>
          <ol className="list-decimal list-inside space-y-1">
            {subQs.map((q, i) => (
              <li key={i}>{q}</li>
            ))}
          </ol>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}

function FindingsBody({ entries }: { entries: NodeUpdate[] }) {
  const findings = entries.flatMap((e) => ((e.update as any).findings as any[]) ?? []);
  return (
    <div className="space-y-2 text-sm">
      <p className="text-muted-foreground">
        <span className="font-medium text-foreground">{findings.length}</span> finding(s) produced
        {entries.length > 1 && <> across {entries.length} worker(s)</>}
      </p>
      {findings.length > 0 && (
        <Accordion type="single" collapsible>
          <AccordionItem value="findings">
            <AccordionTrigger className="text-sm">View findings</AccordionTrigger>
            <AccordionContent className="space-y-3">
              {findings.map((f: any) => (
                <div key={f.id} className="border-b pb-2 last:border-0">
                  <p className="font-medium">{f.sub_question}</p>
                  <p>{f.claim}</p>
                  <p className="text-xs text-muted-foreground">
                    confidence: {Number(f.confidence).toFixed(2)} · {f.evidence?.length ?? 0} source(s)
                  </p>
                </div>
              ))}
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}
    </div>
  );
}

function CritiqueBody({ entries }: { entries: NodeUpdate[] }) {
  const critiques = entries.flatMap((e) => ((e.update as any).critiques as any[]) ?? []);
  return (
    <div className="space-y-2 text-sm">
      <p className="text-muted-foreground">
        <span className="font-medium text-foreground">{critiques.length}</span> critique(s) produced
      </p>
      {critiques.length > 0 && (
        <Accordion type="single" collapsible>
          <AccordionItem value="critiques">
            <AccordionTrigger className="text-sm">View critiques</AccordionTrigger>
            <AccordionContent className="space-y-3">
              {critiques.map((c: any, i: number) => (
                <div key={i} className="border-b pb-2 last:border-0 space-y-1">
                  <div className="flex items-center gap-2">
                    <VerdictBadge verdict={c.verdict} />
                    {c.finding_id && <code className="text-xs text-muted-foreground">{c.finding_id.slice(0, 8)}</code>}
                  </div>
                  <p>{c.reasoning}</p>
                </div>
              ))}
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}
    </div>
  );
}

function FactCheckBody({ entries }: { entries: NodeUpdate[] }) {
  const updated = entries.flatMap((e) => ((e.update as any).findings as any[]) ?? []);
  return (
    <div className="space-y-2 text-sm">
      <p className="text-muted-foreground">
        <span className="font-medium text-foreground">{updated.length}</span> finding(s) fact-checked
      </p>
      {updated.length > 0 && (
        <Accordion type="single" collapsible>
          <AccordionItem value="scores">
            <AccordionTrigger className="text-sm">Updated confidence scores</AccordionTrigger>
            <AccordionContent className="space-y-3">
              {updated.map((f: any) => (
                <div key={f.id}>
                  <p className="font-medium">{f.sub_question}</p>
                  <p className="text-xs text-muted-foreground mb-1">
                    {Number(f.confidence).toFixed(2)} confidence
                  </p>
                  <Progress value={f.confidence * 100} />
                </div>
              ))}
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}
    </div>
  );
}

function WriterBody({ u }: { u: any }) {
  return u.final_report ? (
    <p className="text-sm text-success">
      Report complete: <span className="font-medium">{u.final_report.title}</span>
    </p>
  ) : (
    <p className="text-sm text-muted-foreground">Writing report…</p>
  );
}

function RawBody({ entries }: { entries: NodeUpdate[] }) {
  return (
    <Accordion type="single" collapsible>
      <AccordionItem value="raw">
        <AccordionTrigger className="text-sm">Raw update</AccordionTrigger>
        <AccordionContent className="space-y-2">
          {entries.map((e, i) => (
            <pre key={i} className="text-xs bg-muted rounded p-2 overflow-x-auto">
              {JSON.stringify(e.update, null, 2)}
            </pre>
          ))}
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
