# Comparison: Manual PubMed/bioRxiv Research vs. Research Swarm

**Topic:** GLP-1 receptor agonists as a neuroprotective, disease-modifying therapy in Parkinson's disease
**Manual method:** PubMed MCP (`search_articles` + `get_article_metadata`) + bioRxiv MCP (`search_preprints`), synthesized by hand into `glp1_parkinsons_report.md` / `glp1_parkinsons_findings.json`
**Swarm method:** `research_swarm` graph, `gemma4:31b-cloud` (all tiers), depth=standard, live Tavily web search + arXiv (no RAG pre-ingestion) — output in `swarm_output.json`

## Headline result

Mixed, in an informative way: **the swarm's live web search found something the manual PubMed search missed** (a consequential February 2025 Lancet phase 3 trial), but **the swarm's report lost or degraded roughly half of what its own workers actually found**, due to two synthesis failures and one bad-source hallucination that its own critic correctly caught. Neither method alone would have produced the best possible report; the manual approach had better precision, the swarm had better recall on recency.

## What the swarm got right that the manual report missed

The manual PubMed report (written first, before the swarm ran) covered two positive phase 2 trials (exenatide 2017, lixisenatide/LIXIPARK 2024) and stopped there — a reasonable snapshot, but incomplete. The swarm's `industry` worker surfaced a **phase 3 exenatide trial (Vijiaratnam et al., *Lancet*, Feb 2025, n=194, 96 weeks)** that found **no significant benefit over placebo** (adjusted coefficient 0.92, 95% CI −1.56 to 3.39, p=0.47) — directly complicating the phase 2 signal. I verified this against PubMed ([PMID 39919773](https://pubmed.ncbi.nlm.nih.gov/39919773), [DOI](https://doi.org/10.1016/S0140-6736(24)02808-3)) — it's real, not a hallucination, and it's arguably the single most important recent fact for this topic. This is now recorded as `F6` in `glp1_parkinsons_findings.json`, explicitly flagged as missed by my initial manual pass. **This is the clearest win for the swarm**: a single relevance-sorted PubMed query, done once, is not guaranteed to surface the newest publication, whereas live web search picked up recent news coverage of it directly.

## Where the swarm's report degraded relative to what it actually retrieved

Two of the swarm's four workers (`skeptic`, `benchmark`) hit a structured-output parsing failure at the synthesis step — `gemma4:31b-cloud` echoed the JSON *schema* back (`{"properties": {"claim": ...}, "required": [...], "type": "object"}`) instead of filling it in as a flat object. The raw completion text, visible in the run log, actually contained a **correct, detailed, well-cited claim** (e.g., "Lixisenatide... -0.04 points... compared to a 3.04-point decline in placebo" — matching `F3` almost exactly). But because the wrapper failed to parse, the code's designed fallback replaced it with a generic `[Research incomplete for: ...]` placeholder at confidence 0.15 — the correctly-synthesized numbers never reached the final report at all, despite the underlying evidence (10 sources) having been retrieved successfully. The critic even flagged this directly: *"the cited sources actually provide substantial information... yet [the finding] is marked as 'Research incomplete'"* — the pipeline's own quality-control step correctly diagnosed the bug but had no mechanism to recover the lost synthesis.

Separately, one worker's finding (mechanism/academic) was **correctly refuted by the critic**: its three cited sources turned out to be about ChatGPT-based side-effect detection, PD subtype clustering via ML, and LRRK2/voice biomarkers — none relevant to GLP-1 mechanism. This is the critic mechanism working exactly as intended (catching bad grounding before it reaches the writer), but it also shows Tavily web search returning topically-adjacent-but-wrong sources for a specific biomedical mechanism query, something PubMed's structured indexing doesn't have as a failure mode.

Net effect: of 4 sub-questions dispatched, only 1 (mechanism... wait, that one was refuted) — practically, **only the "industry pipeline" finding survived critique with usable content**; the other three were either refuted or degraded to a low-confidence placeholder. The final report's actual substance is thinner than its polished prose and 30-reference list suggest.

## Precision and specificity

The manual report carries **exact effect sizes with confidence intervals and p-values** for every trial claim, because each claim is traced to one paper's abstract, read directly (e.g., "-3.5 points, 95% CI −6.7 to −0.3, p=0.0318"). The swarm's report is almost entirely qualitative ("positive signal," "modest biomarker improvements," "failed to meet primary clinical endpoints") — the one place a worker *did* produce numbers, synthesis failed and the numbers were dropped (see above). This is a direct, measurable precision gap, not a subjective one.

## Honesty about limitations

Both reports are well-calibrated about their own gaps. The swarm's `limitations` section explicitly states it "relies heavily on high-level summaries... rather than granular quantitative data" because of the incomplete findings — an accurate self-assessment given what actually happened. This matches the "appropriate epistemic restraint" pattern noted in this project's own benchmark history (`gemma4` under-claims rather than hallucinating confidently).

## Scope

The manual report stayed tightly on Parkinson's disease as scoped. The swarm's supervisor broadened the plan to "neurodegeneration" generally, and roughly a third of its final report (and references) is actually about the semaglutide/EVOKE **Alzheimer's** trials, not Parkinson's — a real scope drift from the requested topic, though arguably useful comparative context.

## Bug found in the course of this test

Running the swarm at `standard` depth (rather than the `shallow` depth used in every benchmark run so far this session) triggered a previously-undetected **infinite loop**: `route_from_dispatch`'s no-target bounce sent `collect_node` a `Send()` payload missing `research_rounds`/`pre_dispatch_finding_ids`/`findings`/`critiques`, so those fields silently reset to their defaults on every bounce, `should_stop`'s hard round cap never fired, and the graph recursed until LangGraph's recursion limit killed it. This would have broken every `standard`/`deep`-depth request in production (the Streamlit app's default is `shallow`, which happens to be immune since it hard-caps at round 1). Fixed in `research_swarm/graph/nodes.py` (`route_from_dispatch` / `_collect_bounce_payload`), with a new end-to-end regression test (`TestStandardDepthTerminates`) that exercises the real graph wiring and would have caught this.

## Bottom line

| Dimension | Manual (PubMed) | Swarm |
|---|---|---|
| Recency / recall | Missed the most important recent trial | Found it |
| Precision (numbers, CIs) | High | Low (lost by a bug, not absent by design) |
| Grounding accuracy | 100% (hand-verified) | 3/4 sub-questions compromised (1 refuted, 2 degraded) |
| Self-awareness of gaps | N/A (complete) | Good — flagged its own limitation accurately |
| Scope discipline | On-topic | Drifted into Alzheimer's |
| Effort / time | ~20 tool calls, human-paced | 95 seconds, fully automated |

The swarm's architecture (plan → parallel research → critique → fact-check → write) is sound and its critic caught a real error — but a single JSON-parsing quirk in a mid-tier open model silently deleted good research on two of four sub-questions before the critic ever saw it, and no mechanism currently recovers a raw-but-well-formed completion when strict schema parsing fails. That's the most actionable finding here: the fallback-to-generic-placeholder-on-parse-failure is a bigger quality risk than anything else observed in this comparison.
