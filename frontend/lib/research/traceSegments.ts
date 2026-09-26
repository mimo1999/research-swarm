import type { NodeUpdate } from "./types";
import { segmentKind, type SegmentKind } from "./nodeConfig";

export type TraceSegment = {
  node: string;
  kind: SegmentKind;
  entries: NodeUpdate[];
};

// Parallel worker fan-out (one Send per sub-question) streams as several
// consecutive same-node entries. Merge those into a single timeline stop so
// a research round reads as one step, not N near-identical cards.
export function buildTraceSegments(trace: NodeUpdate[]): TraceSegment[] {
  const segments: TraceSegment[] = [];
  for (const entry of trace) {
    const kind = segmentKind(entry.node, entry.update);
    const last = segments[segments.length - 1];
    if (kind === "stop" && last && last.kind === "stop" && last.node === entry.node) {
      last.entries.push(entry);
    } else {
      segments.push({ node: entry.node, kind, entries: [entry] });
    }
  }
  return segments;
}
