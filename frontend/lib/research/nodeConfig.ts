import {
  Compass,
  Files,
  FileSearch,
  Split,
  Search,
  GitMerge,
  ScanSearch,
  ShieldCheck,
  PenLine,
  Cog,
  type LucideIcon,
} from "lucide-react";

// "stop" nodes produce real research output and get a full marker + card.
// "connector" nodes are deterministic bookkeeping/routing steps between
// stops -- they render as a single line on the timeline, not a card. This
// mirrors the graph's actual shape (see research_swarm/graph/nodes.py):
// plan -> dispatch -> [worker fan-out] -> collect -> (loop or) critic ->
// fact-check -> write.
export type SegmentKind = "stop" | "connector";

type NodeConfig = { icon: LucideIcon; label: string; kind: SegmentKind };

export const NODE_CONFIG: Record<string, NodeConfig> = {
  supervisor: { icon: Compass, label: "Planning", kind: "stop" },
  document_pass_node: { icon: Files, label: "Documents", kind: "connector" },
  document_worker_node: { icon: FileSearch, label: "Document extraction", kind: "stop" },
  dispatch_node: { icon: Split, label: "Dispatch", kind: "connector" },
  worker_node: { icon: Search, label: "Research", kind: "stop" },
  researcher: { icon: Search, label: "Research", kind: "stop" },
  collect_node: { icon: GitMerge, label: "Collect", kind: "connector" },
  critic: { icon: ScanSearch, label: "Critic review", kind: "stop" },
  fact_checker: { icon: ShieldCheck, label: "Fact-check", kind: "stop" },
  writer: { icon: PenLine, label: "Report", kind: "stop" },
};

const DEFAULT_NODE_CONFIG: NodeConfig = { icon: Cog, label: "Agent", kind: "stop" };

export function nodeConfig(node: string): NodeConfig {
  return NODE_CONFIG[node] ?? DEFAULT_NODE_CONFIG;
}

// The supervisor only carries a plan on its first (real) call -- later
// no-op calls just re-confirm routing, so they read as a connector too.
export function segmentKind(node: string, update: Record<string, unknown>): SegmentKind {
  if (node === "supervisor") return update?.plan ? "stop" : "connector";
  return nodeConfig(node).kind;
}

export function connectorLabel(node: string, update: Record<string, any>): string {
  switch (node) {
    case "dispatch_node": {
      const n = Array.isArray(update.pre_dispatch_finding_ids) ? update.pre_dispatch_finding_ids.length : 0;
      return n > 0 ? `Dispatching next round · ${n} finding(s) so far` : "Dispatching research workers";
    }
    case "collect_node": {
      const round = update.research_rounds ?? "?";
      const next = update.next_agent === "critic" ? "moving to review" : "another research pass";
      return `Round ${round} complete → ${next}`;
    }
    case "document_pass_node":
      return "Preparing document extraction";
    case "supervisor":
      return `Routing to ${update.next_agent ?? "next agent"}`;
    default:
      return nodeConfig(node).label;
  }
}
