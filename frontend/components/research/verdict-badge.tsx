import { CheckCircle2, AlertTriangle, XCircle, CircleDashed, type LucideIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const VERDICT_META: Record<string, { label: string; icon: LucideIcon; className: string }> = {
  supported: {
    label: "Supported",
    icon: CheckCircle2,
    className: "bg-success/10 text-success",
  },
  weak: {
    label: "Weak",
    icon: AlertTriangle,
    className: "bg-warning/10 text-warning",
  },
  refuted: {
    label: "Refuted",
    icon: XCircle,
    className: "bg-destructive/10 text-destructive",
  },
  pending: {
    label: "Pending",
    icon: CircleDashed,
    className: "bg-muted text-muted-foreground",
  },
  // LLM-judge verdicts on a finished report -- same three-way semantics
  // (good / needs work / bad) as the finding verdicts above, so they share
  // the same color language instead of inventing a second palette.
  approve: {
    label: "Approved",
    icon: CheckCircle2,
    className: "bg-success/10 text-success",
  },
  revise: {
    label: "Needs revision",
    icon: AlertTriangle,
    className: "bg-warning/10 text-warning",
  },
  reject: {
    label: "Rejected",
    icon: XCircle,
    className: "bg-destructive/10 text-destructive",
  },
};

export function VerdictBadge({ verdict, className }: { verdict: string; className?: string }) {
  const meta = VERDICT_META[verdict] ?? VERDICT_META.pending;
  const Icon = meta.icon;
  return (
    <Badge variant="outline" className={cn("border-transparent", meta.className, className)}>
      <Icon />
      {meta.label}
    </Badge>
  );
}
