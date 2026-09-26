// The writer LLM formats in-text citations as literal "[N]" markers inside
// body_md. Turn those into markdown links pointing at a synthetic
// "#cite-N" fragment so react-markdown emits real <a> elements we can
// intercept and render as interactive footnote marks (see
// components/research/citation-mark.tsx). The negative lookahead skips
// anything already written as a real markdown link, e.g. "[3](https://...)".
export function linkifyCitations(md: string): string {
  return md.replace(/\[(\d+)\](?!\()/g, (_, n: string) => `[${n}](#cite-${n})`);
}

// Citation numbers the LLM actually placed inline, vs. the section's
// structured `citations` list -- the two can drift (a source the writer
// drew on but didn't explicitly mark inline). Used to surface any
// structured citation that has no inline marker instead of silently
// dropping it.
export function inlineCitationNumbers(md: string): Set<number> {
  const nums = new Set<number>();
  for (const m of md.matchAll(/\[(\d+)\]/g)) nums.add(Number(m[1]));
  return nums;
}
