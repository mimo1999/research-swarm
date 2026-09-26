"use client";

import { useMemo, useState } from "react";
import { UserCheck, Check, Pencil, Trash2 } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { VerdictBadge } from "@/components/research/verdict-badge";
import type { Critique, Finding } from "@/lib/research/types";

export function HitlPanel({
  findings,
  critiques,
  onResume,
}: {
  findings: Finding[];
  critiques: Critique[];
  onResume: (action: "approve" | "edit" | "discard", feedback?: string) => void;
}) {
  const [feedback, setFeedback] = useState("");

  const verdictByFinding = useMemo(() => {
    const map = new Map<string, string>();
    for (const c of critiques) if (c.finding_id) map.set(c.finding_id, c.verdict);
    return map;
  }, [critiques]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <UserCheck className="size-4 text-muted-foreground" />
          Human review required
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Alert>
          <AlertDescription>
            The graph has paused before writing. Review the findings below and choose how to proceed.
          </AlertDescription>
        </Alert>

        <div>
          <p className="font-medium mb-2">Findings ({findings.length})</p>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {findings.map((f) => {
              const verdict = verdictByFinding.get(f.id) ?? "pending";
              return (
                <div key={f.id} className="border-b pb-2 last:border-0 text-sm space-y-1">
                  <div className="flex items-center gap-2">
                    <VerdictBadge verdict={verdict} />
                    <span className="font-medium">{f.sub_question}</span>
                  </div>
                  <p>{f.claim}</p>
                  <p className="text-xs text-muted-foreground">
                    confidence: {Number(f.confidence).toFixed(2)}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        <Textarea
          placeholder="e.g. 'Focus more on economic impact. Exclude the speculative claims.'"
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
        />

        <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
          <Button onClick={() => onResume("approve", feedback || "Approved.")}>
            <Check /> Approve &amp; write
          </Button>
          <Button
            variant="outline"
            onClick={() =>
              onResume("edit", feedback || "Please re-research weak findings more thoroughly.")
            }
          >
            <Pencil /> Edit &amp; retry
          </Button>
          <Button variant="destructive" onClick={() => onResume("discard")}>
            <Trash2 /> Discard
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
