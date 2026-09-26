"use client";

import { Waypoints, RotateCcw } from "lucide-react";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { useResearchRun } from "@/lib/research/useResearchRun";
import { QueryForm } from "@/components/research/QueryForm";
import { TraceView } from "@/components/research/TraceView";
import { HitlPanel } from "@/components/research/HitlPanel";
import { ReportView } from "@/components/research/ReportView";

export default function ResearchPage() {
  const { state, error, submit, resume, reset, clearError } = useResearchRun();

  return (
    <div className="mx-auto max-w-3xl px-4 py-10 sm:py-14 space-y-8">
      <div className="flex items-center gap-3">
        <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground">
          <Waypoints className="size-5" />
        </span>
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Research Swarm</h1>
          <p className="text-sm text-muted-foreground">
            Autonomous multi-agent research, built on LangGraph and LlamaIndex.
          </p>
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error.message}</AlertDescription>
          <Button size="sm" variant="ghost" className="mt-2" onClick={clearError}>
            Dismiss
          </Button>
        </Alert>
      )}

      {state.status === "idle" && <QueryForm onSubmit={submit} />}

      {(state.status === "running" ||
        state.status === "interrupted" ||
        state.status === "done" ||
        state.status === "failed") && <TraceView trace={state.trace} live={state.status === "running"} />}

      {state.status === "interrupted" && (
        <HitlPanel findings={state.findings} critiques={state.critiques} onResume={resume} />
      )}

      {state.status === "done" && (
        <>
          <ReportView report={state.report} />
          <Button variant="outline" onClick={reset}>
            <RotateCcw /> Start new research
          </Button>
        </>
      )}

      {state.status === "failed" && (
        <>
          <Alert variant="destructive">
            <AlertTitle>Research run failed</AlertTitle>
            <AlertDescription>
              The graph stopped before producing a report. See the error above for details.
            </AlertDescription>
          </Alert>
          <Button variant="outline" onClick={reset}>
            <RotateCcw /> Start new research
          </Button>
        </>
      )}
    </div>
  );
}
