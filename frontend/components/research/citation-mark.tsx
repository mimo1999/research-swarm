import { HoverCard, HoverCardTrigger, HoverCardContent } from "@/components/ui/hover-card";
import type { Source } from "@/lib/research/types";

// The [N] markers the writer LLM puts inline get rewritten (see
// lib/research/citations.ts) into markdown links pointing at "#cite-N", so
// react-markdown hands them to us here instead of rendering plain <a> tags.
export function CitationMark({
  n,
  references,
  onFocusReference,
}: {
  n: number;
  references: Source[];
  onFocusReference: (n: number) => void;
}) {
  const ref = references[n - 1];

  if (!ref) {
    // The judge's citation_quality dimension exists precisely because this
    // happens sometimes -- a citation number with no matching reference.
    // Surface it rather than silently rendering a dead link.
    return (
      <sup className="mx-0.5 text-[10px] font-medium text-destructive" title="Citation has no matching reference">
        [{n}]
      </sup>
    );
  }

  return (
    <HoverCard openDelay={150} closeDelay={50}>
      <HoverCardTrigger asChild>
        <button
          type="button"
          onClick={() => onFocusReference(n)}
          className="mx-0.5 inline-flex h-4 min-w-4 items-center justify-center rounded bg-muted px-1 align-super text-[10px] font-medium text-muted-foreground no-underline hover:bg-primary hover:text-primary-foreground"
        >
          {n}
        </button>
      </HoverCardTrigger>
      <HoverCardContent>
        <p className="font-medium leading-snug">{ref.title || ref.url}</p>
        {ref.snippet && <p className="mt-1 text-muted-foreground line-clamp-3">{ref.snippet}</p>}
        <p className="mt-2 truncate text-xs text-muted-foreground">{ref.url}</p>
      </HoverCardContent>
    </HoverCard>
  );
}

export function MarkdownLink({
  href,
  children,
  references,
  onFocusReference,
}: {
  href?: string;
  children?: React.ReactNode;
  references: Source[];
  onFocusReference: (n: number) => void;
}) {
  const citeMatch = href?.match(/^#cite-(\d+)$/);
  if (citeMatch) {
    return <CitationMark n={Number(citeMatch[1])} references={references} onFocusReference={onFocusReference} />;
  }
  return (
    <a href={href} target="_blank" rel="noreferrer" className="underline underline-offset-2 hover:text-foreground">
      {children}
    </a>
  );
}
