# Domain glossary

Terms the research pipeline's code and design discussions use. Architecture vocabulary (module,
interface, seam, adapter, depth, leverage, locality) follows the codebase-design skill.

**Question frame**: what the question is really asking, extracted once before planning: the key
constraint (what separates it from its general subject), its phrasings, confusable topics, strict
terms and their proof criterion. Enforced in code at every stage.

**Plan**: the sub-questions for a run (count set by the depth profile), each with a keyword search
query and a literature domain.

**Depth profile**: everything that scales with the chosen depth (sub-questions, candidates,
papers kept, gap-fill workers and rounds, full-text reads, packet budget), in
`settings.depth_profiles`. Not exposed in the UI.

**Source**: one document the run read: a paper abstract (plus full-text excerpts after a deep
read), a fetched web page, or an uploaded document.

**Sentence ID**: the stable address of one sentence of one source in a run, written `S<source>.<sentence>`
(e.g. `S3.4`). Sources are segmented once; every citation in the packet path is a sentence ID, and
code resolves it to the exact source text. Replaces verbatim-quote grounding.

**Evidence packet**: the budgeted set of source sentences, each with its sentence ID, grouped by
sub-question, that the synthesis call reads. Built by code first (segment, dedupe, score per
sub-question, fit the budget); the local small model screens passages only when candidates
exceed the budget. It is the only view of the sources the large model gets, and the only text a
citation may point at.

**Packet budget**: the most source-sentence tokens an evidence packet may hold: 2k / 4k / 8k for
shallow / standard / deep (a depth-profile key). Sources that fit go in whole; above it, code
scoring and then **overflow screening** decide what stays: sentences are grouped into passages of
3-5 sentences, the local small model marks each passage relevant / not for its sub-question in
batched calls, and the best-scored relevant passages are kept until the budget is full.

**Planning** (packet path): the question frame and plan run only when there is something to search.
When the sources are supplied and nothing is searched (e.g. a claim-check over given documents),
the whole question is the single sub-question and no planning call is made.

**Review pause** (packet path): the human-in-the-loop pause shows the evidence packet itself,
sentences with their IDs and sources grouped by sub-question, with gaps flagged. "Edit &
Re-research" adds keywords to gap-fill searches and the packet is rebuilt.

**Coverage** (packet path): a sub-question is covered when enough of its screened sentences match
the question frame's scope, decided in code. A thin sub-question triggers **gap fill**: search,
fetch and segment straight into the packet, with no LLM extraction call.

**Synthesis**: the single large-model call that reads the evidence packet and returns the direct
answer, the claim verdict (for claim-check questions), the stance, and the report sentences grouped
by section, each citing sentence IDs. The large model is the judge; code is the auditor. A
SUPPORT / CONTRADICT verdict must cite at least one packet sentence, or code turns it into the
insufficient-evidence label. Its output reaches code render through an adapter: each packet
sentence becomes one fact whose evidence is exactly that sentence.

**Code render**: the deterministic step after synthesis that resolves sentence IDs, drops any
sentence citing an unknown ID or stating numbers / scope / strict terms its cited sentences do not
contain, attaches numbered citations and shapes the report for the audience.

**Fact chain** (legacy): the current evidence path, extract facts with quotes, locate the quotes,
verify each fact, label claim relations, then outline / write sections / review. Kept beside the
packet path until the packet path wins on the paired benchmark, then removed. **Winning** means
large-model tokens below 2k per task (the fact chain uses ~7.2k) with verdict accuracy within
noise of the fact chain on the same tasks.

**Pipeline mode**: the setting that selects the packet path or the fact chain for a run, so both
can be compared on the same tasks. Each mode is its own graph, chosen when the graph is built:
the packet path wires packet and synthesis nodes in place of the paper worker, verifier and
writer; search, scouting, deep read, gap-fill search and the review pause are shared.
