import { nodeConfig, type SegmentKind } from "@/lib/research/nodeConfig";

// Shares one size-7 column across both marker types so the connecting line
// (positioned at its horizontal center) threads through every row, whether
// it's a solid "stop" badge or a bare "connector" glyph.
export function NodeMarker({ node, kind }: { node: string; kind: SegmentKind }) {
  const { icon: Icon } = nodeConfig(node);
  return (
    <div className="relative z-10 flex size-7 shrink-0 items-center justify-center">
      {kind === "stop" ? (
        <span className="flex size-7 items-center justify-center rounded-full bg-muted text-foreground">
          <Icon className="size-3.5" />
        </span>
      ) : (
        <span className="flex size-5 items-center justify-center rounded-full bg-background text-muted-foreground">
          <Icon className="size-3" />
        </span>
      )}
    </div>
  );
}
