# Independent quality / latency review of the research swarm

> **Historical document.** This log describes an earlier version of the system. It reviewed the pre-v2 pipeline (fetch-and-embed pass, ReAct workers, critic and fact-checker), which the findings here led to replacing. For the current architecture see [README.md](../../README.md), [TECHNICAL_HANDOFF.md](../../TECHNICAL_HANDOFF.md) and [CLAUDE.md](../../CLAUDE.md).

Setup: local Ollama daemon proxying `nemotron-3-nano:30b-cloud` for all tiers (product defaults),
`ollama_reasoning=True`, Windows CPU embeddings. Harness: `run_review.py`; traces in
`data/quality_review/*/traces/*.jsonl`; aggregation: `analyze_traces.py`.

Runs completed: shallow q4-q8 (5), standard q2, q3 (2). Standard q1 died twice on Ollama
429 "timed out waiting for a concurrent request slot" (no report). Standard q4-q8 not run
(time). Small sample; numbers are indicative, not statistics.

## 1. Where the time goes

Shallow (1 sub-question, 1 tool turn) - wall seconds per stage:

| q | total | supervisor | **fetch pass** | worker | critic | fact-check | writer |
|---|---|---|---|---|---|---|---|
| q4 | 331 | 7 | **250** | 35 | 7 | 5 | 28 |
| q5 | 224 | 6 | **185** | 24 | 6 | 1 | 3 |
| q6 | 773 | 10 | **274** | 31 | 4 | 4 | **450** (runaway) |
| q7 | 160 | 10 | **101** | 28 | 5 | 4 | 11 |
| q8 | 259 | 14 | **177** | 30 | 4 | 4 | 30 |

Standard (4 sub-questions):

| q | total | fetch pass | worker phase | of which LLM "summarizer" | critic+FC+writer |
|---|---|---|---|---|---|
| q2 | 2742 | 674 | 2011 | 1551 (5 calls, avg 310 s) | 42 (critic/FC skipped, see 3) |
| q3 | 1889 | 792 | 983 | 658 (15 calls) | 91 |
| q1 (failed) | 1986 | 586 | 1332+ | 479+ (8 calls) | - |

Ranking by time: (1) fetch pass, (2) worker-phase snippet summarizer, (3) worker tool-loop LLM
turns + rework rounds, (4) writer only when it runs away. Supervisor, critic, fact-checker,
collect are each < 45 s and are not bottlenecks.

## 2. Where quality is lost

1. Source fetching / retrieval (biggest). Fetch pass embeds 300-2000 chunks per run, but
   workers call `retrieve_from_rag` once or twice in total. In direct-Ollama-Cloud config it
   returns 401 every time (RAG's LlamaIndex Ollama LLM has no bearer token), so the whole pass
   is dead weight there. arXiv is searched/embedded for biomedical questions. Evidence is
   often off-topic: PD safety question answered with diabetes/bariatric-surgery numbers; q6
   cites 5 arXiv papers, claim has no measured figure; q7 references include a local file path
   and a Substack post.
2. Worker synthesis: claims stuffed with numbers, sometimes unsupported. q1: finding with 0
   evidence sources self-scored 0.90. Worker self-confidence (0.8-0.96 everywhere) is
   uninformative.
3. Critic / fact-checker judge on 150-200 character snippets and at most 3-5 sources, so they
   cannot verify specific numbers -> noisy weak/refuted verdicts. q5: sound claim marked
   "refuted" -> writer emitted "Insufficient evidence" (8 words, judge: reject). Fact-checker
   cut confidence by -0.65 / -0.71 on q4/q7 for the same reason.
4. Rework loop is blind: `suggested_followup` is written by the critic and never read by any
   agent. A rework worker re-runs the identical prompt. q3: 3 rounds, 39 research LLM calls.
   Also standard depth always burns an empty "first round - no comparison basis" bounce.
5. Writer: with 1 finding (shallow) reports are 8-220 words. q6 writer hit the 131,072-token
   output cap with empty content (reasoning loop, 446 s), then fell back to a 33-word report.
6. Silent review skipping: `max_tokens_per_session=200000` is exhausted by the summarizer's
   output (q2: 228k used) so critic and fact-checker were skipped entirely; q3 lost
   fact-check. Report ships unreviewed with no user-visible warning.
7. Infra: effective provider concurrency is ~1 request at a time (N parallel calls -> latency
   scales ~N x). No retry/backoff on 429; a 300 s queue wait kills the run and discards all
   research.

Root cause of both wait and quality: reasoning is on for every call (`ollama_reasoning=True`).
Summarizer emitted 46k-162k output tokens for 13k-56k input tokens (up to 510k chars of hidden
reasoning per run) just to shorten snippets.

## 3. Strategies (est. saving, quality effect)

| # | change | saves | quality |
|---|---|---|---|
| 1 | Turn reasoning off (`think=false`) for summarizer, critic, fact-checker, supervisor; keep it only for writer | 60-80 % of LLM time on those stages; ~1500 s on q2 | neutral/+ (structured extraction gains nothing from thinking) |
| 2 | Replace LLM snippet summarizer with extractive trim (top sentences by embedding to the sub-question) | 480-1550 s standard, 10-20 s shallow | + (no paraphrase drift) |
| 3 | Fetch pass: default OFF, or lazy: abstracts only, no arXiv PDFs, domain-route (skip arXiv for biomedical, PubMed for CS), cap chunks/sub-question, run in parallel with supervisor | 100-270 s shallow, 400-800 s standard | + (less off-topic evidence) |
| 4 | Set `num_predict` per call (e.g. writer 4k) and a wall timeout | prevents 446 s runaways | + |
| 5 | Give critic the full snippets/more sources; feed `suggested_followup` into the rework worker; only rework `refuted`; drop empty first-round bounce; max 1 rework | 1 round (~700 s) per standard run | ++ |
| 6 | Don't drop findings/produce empty report on a single "refuted"; require >= 1 evidence source else cap confidence at 0.3 | - | ++ |
| 7 | Retry with backoff on 429; global LLM semaphore (2); raise/rebase token cap or surface "review skipped" in the report | avoids lost runs | + |
| 8 | Fix RAG LLM auth for direct Ollama Cloud (or drop the unused synthesis LLM in `no_text` use) | wasted call per retrieval | + |

Projected: shallow ~260 s -> ~60-90 s; standard 30-45 min -> ~6-10 min.
