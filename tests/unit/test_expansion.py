"""Question frame (agents/expansion.py) and the scope enforcement it drives downstream:
planner enforcement, scout pools + scoring, the coverage gate, verifier relevance, and the writer's
gap reporting. Includes a replay of the KV-cache incident (a cross-model question answered with
compression papers). LLMs and search tools are mocked; nothing touches the network."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_swarm.agents import expansion
from research_swarm.agents.expansion import normalize_frame, scope_hit
from research_swarm.agents.writer_render import (
    ANALYSIS_HEADING,
    DraftSection,
    DraftSentence,
    WriterDraft,
    WriterDraftWithAnalysis,
    render_report,
)
from research_swarm.config import settings
from research_swarm.eval.claims import split_claims
from research_swarm.schemas import Finding, ResearchPlan, Source
from research_swarm.schemas.frame import QuestionFrame
from research_swarm.schemas.worker import SubQuestionAssignment
from tests.unit.test_graph import _make_state

KV_TOPIC = "Can we losslessly migrate KV cache from one LLM to another"
KV_FRAME = QuestionFrame(
    interpretation="Whether one LLM's KV cache can be used by a different LLM without loss.",
    key_constraint="across different LLMs",
    constraint_terms=["cross-model", "across models", "KV cache sharing across models"],
    confusable_topics=["KV cache compression", "multi-GPU KV cache migration"],
    define_terms=["lossless"],
    search_queries=["cross-model KV cache transfer", "KV cache sharing across LLMs"],
    topic=KV_TOPIC,
)


def _llm_returning(value):
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=value)
    return llm


def _fact(i, claim, sq="sq", relevance="unknown", url=None, grounding="quote"):
    return Finding(
        id=f"f{i}", claim=claim, sub_question=sq, relevance=relevance, grounding=grounding,
        evidence=[Source(url=url or f"http://x/{i}", title=f"T{i}", snippet=claim)],
    )


# --- scope matching ------------------------------------------------------------------------

def test_scope_hit_is_tolerant_of_wording_but_rejects_the_general_subject():
    assert scope_hit("KV cache sharing across fine-tuned model variants", KV_FRAME)
    assert scope_hit("Cross-Model KV Cache Transfer in LLM Families", KV_FRAME)
    assert not scope_hit("KV cache quantization down to 1 bit per channel", KV_FRAME)


def test_scope_hit_ignores_the_questions_subject_words():
    # "KV cache sharing across models" shares "KV cache" (+ "model") with any compression paper;
    # only the distinctive words (sharing, across, models) may count.
    assert not scope_hit("CQ preserves model quality with KV cache quantized to 1 bit", KV_FRAME)


def test_a_phrasing_made_only_of_subject_words_never_matches():
    # Seen live: gemma offered "KV Cache Migration" as a phrasing of "from one LLM to another".
    from research_swarm.agents.expansion import distinctive_phrases

    frame = KV_FRAME.model_copy(update={
        "key_constraint": "from one LLM to another",
        "constraint_terms": ["KV Cache Migration", "Cross-Model KV Cache Transfer"],
    })
    assert not scope_hit("Mell migrates KV cache between GPUs", frame)
    assert scope_hit("Cross-model KV cache transfer between LLM families", frame)
    assert "KV Cache Migration" not in distinctive_phrases(frame)


def test_scope_hit_is_always_true_without_a_constraint():
    assert scope_hit("anything at all", None)
    assert scope_hit("anything at all", QuestionFrame())


# --- expansion ---------------------------------------------------------------------------

def test_normalize_frame_dedupes_caps_and_fills_fallbacks():
    frame = normalize_frame(QuestionFrame(
        key_constraint="  across   different LLMs ", constraint_terms=[],
        confusable_topics=["a", "A", "", *[f"t{i}" for i in range(9)]],
    ), KV_TOPIC)
    assert frame.key_constraint == "across different LLMs"
    assert frame.constraint_terms == ["across different LLMs"]            # terms fallback
    assert frame.search_queries and "across different LLMs" in frame.search_queries[0]
    assert frame.confusable_topics[0] == "a" and len(frame.confusable_topics) == 5


@pytest.mark.parametrize("placeholder", ["none", "None.", "N/A", "", "no constraint", "-"])
def test_normalize_frame_treats_placeholder_constraints_as_empty(placeholder):
    # Seen live: gemma returned key_constraint="none" for "What is the boiling point of water?"
    frame = normalize_frame(QuestionFrame(key_constraint=placeholder,
                                          search_queries=["boiling point water"]), "q")
    assert not frame.has_constraint and frame.search_queries == []


def test_normalize_frame_drops_confusable_topics_that_are_the_question_itself():
    # kv-report-8ae16676: "cross-model KV cache sharing" was listed as NOT the question.
    frame = normalize_frame(QuestionFrame(
        key_constraint="across different LLMs",
        constraint_terms=["cross-model", "KV cache sharing across models"],
        confusable_topics=["KV-cache compression", "same-model cache migration",
                           "cross-model KV cache sharing"],
    ), KV_TOPIC)
    assert frame.confusable_topics == ["KV-cache compression", "same-model cache migration"]


def test_proof_criterion_and_compare_items_are_normalized_and_reach_the_planner():
    frame = normalize_frame(QuestionFrame(
        key_constraint="across different LLMs", define_terms=["lossless"],
        proof_criterion="Identical logits vs native prefill; accuracy is not enough.",
        compare_items=["direct reuse", "transformation", "none", "direct reuse"],
    ), KV_TOPIC)
    assert frame.compare_items == ["direct reuse", "transformation"]
    block = expansion.frame_prompt_block(frame)
    assert "What would establish them: Identical logits" in block
    assert "Items to compare (cover each): direct reuse; transformation" in block
    # without a strict qualifier there is nothing to establish
    assert normalize_frame(QuestionFrame(proof_criterion="x"), KV_TOPIC).proof_criterion == ""


async def test_the_writer_is_told_what_would_establish_the_strict_requirement():
    from research_swarm.agents.writer import run_attributed_writer

    frame = KV_FRAME.model_copy(update={
        "define_terms": ["lossless"], "proof_criterion": "Identical logits vs native prefill."})
    plan = ResearchPlan(sub_questions=["sq"], strategy="s", frame=frame)
    llm = _llm_returning(_draft(sections=[DraftSection(heading="A", sentences=[
        DraftSentence(text="Matched pairs retain 73% accuracy.", facts=[1])])]))
    state = _make_state(plan=plan, findings=[_fact(1, "Matched pairs retain 73% accuracy.")])
    await run_attributed_writer(state, llm)
    system = llm.with_structured_output.return_value.ainvoke.call_args[0][0][0].content
    assert "Strict requirement in the question: lossless" in system
    assert "Identical logits vs native prefill." in system
    assert "never treat them as synonyms" in system


def test_normalize_frame_without_a_constraint_has_no_search_queries():
    frame = normalize_frame(QuestionFrame(search_queries=["x y z"]), "boiling point of water")
    assert not frame.has_constraint and frame.search_queries == []


async def test_expand_question_parses_and_attaches_probe_hits():
    hits = [{"url": "u1", "title": "DroidSpeak: KV cache sharing across model variants"}]
    llm = _llm_returning(KV_FRAME)
    frame = await expansion.expand_question(KV_TOPIC, hits, llm, "s")
    assert frame.key_constraint == "across different LLMs" and frame.probe_hits == hits
    prompt = llm.with_structured_output.return_value.ainvoke.call_args[0][0][1].content
    assert "DroidSpeak" in prompt                                  # the probe titles are shown


async def test_expand_question_failure_gives_an_empty_frame():
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=ValueError("bad"))
    frame = await expansion.expand_question(KV_TOPIC, [], llm, "s")
    assert frame == QuestionFrame()


async def test_probe_makes_no_llm_call_and_survives_tool_failure():
    tool = MagicMock()
    tool.invoke.side_effect = RuntimeError("down")
    with patch("research_swarm.agents.papers.tool_registry", return_value={"arxiv": tool}):
        assert await expansion.probe(KV_TOPIC, "s") == []


# --- planner enforcement ---------------------------------------------------------------------

def _kv_plan() -> ResearchPlan:
    sqs = [
        "What are the theoretical bottlenecks for lossless KV cache migration between different LLMs?",
        "How do quantization and compression impact the fidelity of the migrated KV cache?",
        "What are the empirical differences when migrating KV caches across architectures?",
    ]
    queries = ["KV cache migration theory bottlenecks", "KV cache quantization fidelity migration",
               "KV cache migration empirical comparison"]
    return ResearchPlan(sub_questions=sqs, strategy="s", assignments=[
        SubQuestionAssignment(sub_question=sq, search_query=q) for sq, q in zip(sqs, queries)
    ])


def test_enforce_plan_appends_the_constraint_to_drifted_questions_and_every_query():
    from research_swarm.agents.supervisor import _enforce_plan

    plan = _enforce_plan(_kv_plan(), KV_FRAME, 4)
    assert plan.frame == KV_FRAME
    assert plan.sub_questions[0] == _kv_plan().sub_questions[0]      # already in scope
    assert plan.sub_questions[1].endswith("(across different LLMs)")  # drifted: fixed
    for sq in plan.sub_questions:
        a = plan.assignment_for(sq)
        assert a is not None and scope_hit(a.search_query, KV_FRAME)   # the incident's bug


def test_enforce_plan_truncates_to_the_depth_budget():
    from research_swarm.agents.supervisor import _enforce_plan

    plan = _enforce_plan(_kv_plan(), QuestionFrame(), 1)
    assert len(plan.sub_questions) == 1 and len(plan.assignments) == 1


def test_enforce_plan_with_an_empty_frame_leaves_the_plan_unchanged():
    from research_swarm.agents.supervisor import _enforce_plan

    original = _kv_plan()
    plan = _enforce_plan(original, QuestionFrame(), 4)
    assert plan.sub_questions == original.sub_questions
    assert plan.assignments == original.assignments


async def test_run_supervisor_puts_the_frame_in_the_planner_prompt_and_on_the_plan():
    from research_swarm.agents.supervisor import SupervisorDecision, run_supervisor
    from research_swarm.schemas.query import ResearchDepth, ResearchQuery

    decision = SupervisorDecision(reasoning="r", next_agent="dispatch", plan=_kv_plan())
    llm = _llm_returning(decision)
    state = _make_state(plan=None, query=ResearchQuery(topic=KV_TOPIC, depth=ResearchDepth.shallow))
    with patch.object(settings, "query_expansion_enabled", True), \
         patch.object(settings, "sub_questions_by_depth", {"shallow": 1}), \
         patch.object(expansion, "probe", AsyncMock(return_value=[])), \
         patch("research_swarm.agents.supervisor.probe", AsyncMock(return_value=[])), \
         patch("research_swarm.agents.supervisor.expand_question",
               AsyncMock(return_value=KV_FRAME)):
        out = await run_supervisor(state, llm)
    prompt = llm.with_structured_output.return_value.ainvoke.call_args[0][0][1].content
    assert "Key constraint" in prompt and "across different LLMs" in prompt
    assert "NOT this question" in prompt
    assert out.plan.frame == KV_FRAME
    assert len(out.plan.sub_questions) == 1               # the depth's count, enforced in code


def test_supervisor_decision_accepts_nemotrons_flattened_plan():
    # kv-report-fd84dcea: a good plan failed to parse (no next_agent; assignments and strategy
    # beside `plan` instead of inside it) and the run fell back to a one-question plan.
    from research_swarm.agents.supervisor import SupervisorDecision

    raw = {
        "reasoning": "r", "strategy": "Decompose into four.", "complexity_score": 0.9,
        "assignments": [{"sub_question": "A?", "search_query": "a query",
                         "domain": "cs_ml_physics_math"}],
        "plan": {"sub_questions": ["A?"], "strategy": "Plan strategy.", "complexity_score": 0.9},
    }
    decision = SupervisorDecision.model_validate(raw)
    assert decision.next_agent == "dispatch"
    assert decision.plan.strategy == "Plan strategy."              # the plan's own value wins
    assert decision.plan.assignment_for("A?").search_query == "a query"


def test_supervisor_decision_without_reasoning_still_parses():
    # kv-audit-1d76136e: a complete plan without `reasoning` fell back to a one-question plan.
    from research_swarm.agents.supervisor import SupervisorDecision

    decision = SupervisorDecision.model_validate(
        {"plan": {"sub_questions": ["A?"], "strategy": "s"}, "next_agent": "dispatch"})
    assert decision.plan.sub_questions == ["A?"] and decision.reasoning == ""


def test_supervisor_decision_builds_the_plan_from_top_level_fields():
    from research_swarm.agents.supervisor import SupervisorDecision

    decision = SupervisorDecision.model_validate(
        {"reasoning": "r", "sub_questions": ["A?", "B?"], "strategy": "s"})
    assert decision.plan.sub_questions == ["A?", "B?"]


# --- scout ---------------------------------------------------------------------------------

async def test_score_pool_prompt_carries_scope_and_confusable_topics():
    from research_swarm.agents.papers import PaperScores, score_pool

    llm = _llm_returning(PaperScores(scores=[]))
    await score_pool(KV_TOPIC, "sq", [{"title": "t", "snippet": "s"}], llm, frame=KV_FRAME)
    system, user = llm.with_structured_output.return_value.ainvoke.call_args[0][0]
    assert "Specific scope of the question: across different LLMs" in user.content
    assert "NOT the question: KV cache compression" in user.content
    assert "scores at most 4" in system.content


async def test_scout_adds_frame_queries_and_probe_hits_to_every_pool():
    from research_swarm.graph.nodes import paper_scout_node

    def paper(n, kind):
        return {"url": f"https://{kind}/{n}", "title": f"{kind} {n}",
                "snippet": "x" * 200, "source_type": "arxiv"}

    tool = MagicMock()
    tool.invoke.side_effect = lambda args: (
        [paper(1, "frame")] if "cross-model" in args["query"] else [paper(1, args["query"][:4])]
    )
    frame = KV_FRAME.model_copy(update={"probe_hits": [paper(9, "probe")]})
    tasks = [{"sub_question": f"sq{i}", "search_query": f"q{i} kv", "domain": "cs_ml_physics_math"}
             for i in range(2)]
    pools: dict[str, list[str]] = {}

    async def fake_score(topic, sq, pool, llm, frame=None):
        pools[sq] = [p["url"] for p in pool]
        assert frame is not None and frame.has_constraint
        return {}

    state = _make_state(scout_tasks=tasks, frame=frame)
    with patch("research_swarm.graph.nodes._get_tiered_state_llm", return_value=MagicMock()), \
         patch("research_swarm.agents.papers.tool_registry", return_value={"arxiv": tool}), \
         patch("research_swarm.agents.papers.score_pool", fake_score):
        await paper_scout_node(state)
    for sq in ("sq0", "sq1"):
        assert "https://frame/1" in pools[sq] and "https://probe/9" in pools[sq]


def test_prefilter_keeps_on_topic_in_scope_primary_candidates_for_the_llm_scorer():
    from research_swarm.agents.papers import prefilter_candidates

    def cand(i, title, url="https://arxiv.org/abs/2600.0{:04d}"):
        return {"url": url.format(i), "title": title, "snippet": ""}

    pool = [cand(i, f"Unrelated paper {i} on protein folding") for i in range(30)]
    pool += [cand(100, "Cross-model KV cache transfer between LLMs"),
             cand(101, "KV cache transfer across models", url="https://medium.com/p/{}"),
             cand(102, "KV cache quantization for LLM serving")]
    kept = prefilter_candidates(pool, "Can a KV cache be transferred across models?",
                                "cross-model KV cache transfer", KV_FRAME, keep=3)
    titles = [c["title"] for c in kept]
    assert titles[0] == "Cross-model KV cache transfer between LLMs"       # scope + primary
    assert "KV cache quantization for LLM serving" in titles
    assert not any("protein" in t for t in titles)
    assert prefilter_candidates(pool[:2], "q", "q", None, keep=3) == pool[:2]   # small pool


# --- coverage gate --------------------------------------------------------------------------

def _gate(findings, frame=KV_FRAME):
    from research_swarm.graph.nodes import _research_targets

    plan = ResearchPlan(sub_questions=["sq"], strategy="s", frame=frame)
    with patch.object(settings, "min_grounded_facts", 2):
        return _research_targets(_make_state(plan=plan, findings=findings, research_rounds=0))


def test_background_facts_do_not_count_as_coverage():
    facts = [_fact(i, "KV cache sharing across models works", relevance="background")
             for i in range(2)]
    assert _gate(facts) == ["sq"]


def test_direct_facts_that_miss_the_scope_lexically_do_not_count():
    facts = [_fact(i, "KV cache quantization to 1 bit keeps quality", relevance="direct")
             for i in range(2)]
    assert _gate(facts) == ["sq"]


def test_direct_in_scope_facts_count():
    facts = [_fact(i, f"KV cache sharing across models, variant {i}", relevance="direct")
             for i in range(2)]
    assert _gate(facts) == []


def test_without_a_frame_the_gate_behaves_as_before():
    facts = [_fact(i, "KV cache quantization", relevance="unknown") for i in range(2)]
    assert _gate(facts, frame=None) == []


# --- verifier -------------------------------------------------------------------------------

async def test_verifier_copies_relevance_without_changing_critique_verdicts():
    from research_swarm.agents.verifier import FactVerdict, VerifyBatch, run_verifier

    facts = [_fact(1, "a", relevance="direct"), _fact(2, "b", relevance="direct")]
    llm = _llm_returning(VerifyBatch(verdicts=[
        FactVerdict(fact=1, verdict="supported", relevance="background"),
        FactVerdict(fact=2, verdict="supported"),                        # no label: keep
    ]))
    plan = ResearchPlan(sub_questions=["sq"], strategy="s", frame=KV_FRAME)
    updated, critiques, _ = await run_verifier(_make_state(plan=plan, findings=facts), llm)
    by_id = {f.id: f.relevance for f in updated}
    assert by_id == {"f1": "background", "f2": "direct"}
    assert {c.verdict.value for c in critiques} == {"supported"}
    system = llm.with_structured_output.return_value.ainvoke.call_args[0][0][0].content
    assert "across different LLMs" in system


async def test_verifier_cannot_downgrade_a_fact_that_states_the_scope():
    # Seen live: gemma4:e2b labelled "cross-model KV cache transfer allows the receiver to reuse
    # the source's KV cache" as background for a cross-model question.
    from research_swarm.agents.verifier import FactVerdict, VerifyBatch, run_verifier

    facts = [_fact(1, "Cross-model KV cache transfer lets a receiver reuse the source's cache."),
             _fact(2, "KVComp compresses the KV cache by 83%.")]
    llm = _llm_returning(VerifyBatch(verdicts=[
        FactVerdict(fact=1, verdict="supported", relevance="background"),
        FactVerdict(fact=2, verdict="supported", relevance="background"),
    ]))
    plan = ResearchPlan(sub_questions=["sq"], strategy="s", frame=KV_FRAME)
    updated, _, _ = await run_verifier(_make_state(plan=plan, findings=facts), llm)
    assert {f.id: f.relevance for f in updated} == {"f1": "direct", "f2": "background"}


def test_interleave_keeps_one_copy_of_an_arxiv_paper_across_mirrors():
    from research_swarm.agents.papers import interleave

    ranked = {
        "web": [{"url": "https://www.alphaxiv.org/abs/2608.03893", "title": "A"},
                {"url": "https://www.emergentmind.com/papers/2608.03893", "title": "B"},
                {"url": "https://example.com/blog/2608.03893", "title": "C"}],   # not a mirror
        "arxiv": [{"url": "https://arxiv.org/html/2608.03893v1", "title": "D"},
                  {"url": "http://arxiv.org/abs/2501.06709v1", "title": "E"}],
    }
    urls = [p["url"] for p in interleave(ranked, cap=10)]
    assert urls == ["https://www.alphaxiv.org/abs/2608.03893",      # round-robin order
                    "http://arxiv.org/abs/2501.06709v1", "https://example.com/blog/2608.03893"]


# --- writer render --------------------------------------------------------------------------

def _draft(**kw):
    base = dict(title="T", direct_answer="Compression enables lossy migration.",
                answer_facts=[1], stance="partial", summary=[], sections=[])
    base.update(kw)
    return WriterDraft(**base)


def test_no_direct_fact_prefixes_the_answer_and_lists_the_gap():
    facts = [_fact(1, "KVComp reduces KV cache size by 83%.", relevance="background")]
    draft = _draft(sections=[DraftSection(heading="A", sentences=[
        DraftSentence(text="KVComp reduces KV cache size by 83%.", facts=[1])])])
    report, stats = render_report(draft, facts, KV_TOPIC, frame=KV_FRAME, sub_questions=["sq"])
    assert report.exec_summary.startswith(
        "**Answer:** No retrieved source directly addresses across different LLMs")
    assert "[1]" not in report.exec_summary.split("\n")[0]       # background fact not the answer
    assert stats["no_direct_answer"] and stats["gap_sub_questions"] == 1
    assert report.limitations.startswith("Not answered by the retrieved evidence: sq.")


def test_sentence_claiming_the_scope_on_off_scope_evidence_is_dropped():
    facts = [_fact(1, "Mell's scheduling heuristics lack theoretical guarantees.",
                   relevance="direct")]
    draft = _draft(sections=[DraftSection(heading="A", sentences=[
        DraftSentence(text="Bottlenecks for lossless migration stem from scheduling heuristics.",
                      facts=[1]),
        DraftSentence(text="Mell's scheduling heuristics lack theoretical guarantees.",
                      facts=[1]),
    ])])
    report, stats = render_report(draft, facts, KV_TOPIC, frame=KV_FRAME)
    body = report.sections[0].body_md
    assert "lossless" not in body and "lack theoretical guarantees" in body
    assert stats["dropped_scope_overclaim"] == 1


def test_a_sentence_denying_the_strict_bar_is_not_an_overclaim():
    # kv-audit-1d76136e: "does not prove ... lossless" sentences were dropped as overclaims.
    facts = [_fact(1, "The ridge mapper retains 73-98% of standalone accuracy.",
                   relevance="direct")]
    frame = KV_FRAME.model_copy(update={"define_terms": ["lossless"]})
    draft = _draft(direct_answer="No.", sections=[DraftSection(heading="A", sentences=[
        DraftSentence(text="The paper does not prove the transfer is lossless.", facts=[1]),
        DraftSentence(text="The ridge mapper makes the transfer lossless.", facts=[1]),
    ])])
    report, stats = render_report(draft, facts, KV_TOPIC, frame=frame)
    body = report.sections[0].body_md
    assert "does not prove the transfer is lossless" in body
    assert "makes the transfer lossless" not in body and stats["dropped_scope_overclaim"] == 1


def test_scope_sentence_backed_by_its_own_fact_is_kept():
    facts = [_fact(1, "DroidSpeak shares KV cache across models with the same architecture.",
                   relevance="direct")]
    draft = _draft(direct_answer="Partly.", sections=[DraftSection(heading="A", sentences=[
        DraftSentence(text="KV cache can be shared across models of the same architecture.",
                      facts=[1])])])
    report, stats = render_report(draft, facts, KV_TOPIC, frame=KV_FRAME, sub_questions=["sq"])
    assert report.sections and stats["dropped_scope_overclaim"] == 0
    assert not stats["no_direct_answer"] and stats["gap_sub_questions"] == 0


def test_yes_to_a_strict_bar_needs_a_fact_that_states_it():
    # Seen live: "Yes, research has developed frameworks to transfer the KV cache between models
    # in the same family" for a question asking about *lossless* migration.
    facts = [_fact(1, "A closed-form mapping transfers the KV cache between models in the same "
                      "family.", relevance="direct")]
    draft = _draft(direct_answer="Yes, frameworks transfer the KV cache between models.",
                   answer_facts=[1], sections=[DraftSection(heading="A", sentences=[
                       DraftSentence(text="A closed-form mapping transfers the cache.",
                                     facts=[1])])])
    report, stats = render_report(draft, facts, KV_TOPIC, frame=KV_FRAME)
    assert report.exec_summary.startswith(
        "**Answer:** Not established: no retrieved source shows the 'lossless' requirement")
    assert "Yes" not in report.exec_summary.split("\n")[0]
    assert "is met. Frameworks transfer" in report.exec_summary       # re-capitalized
    assert stats["strict_bar_unmet"] == ["lossless"]

    lossless = [_fact(1, "The mapping is lossless: outputs match the native prefill exactly.",
                      relevance="direct")]
    report, stats = render_report(draft, lossless, KV_TOPIC, frame=KV_FRAME)
    assert report.exec_summary.startswith("**Answer:** Yes") and not stats["strict_bar_unmet"]


_EXACT_FRAME = KV_FRAME.model_copy(update={"define_terms": ["exact equivalence"]})


def test_a_no_answer_that_still_asserts_the_strict_bar_is_flagged():
    # kv-report-8ae16676: "No, ... except in the narrow case where the models share identical KV
    # head counts ..., enabling exact equivalence ..." -- the evidence showed 73-98% retention.
    facts = [_fact(1, "Matched-KV pairs retain 73-98% of standalone accuracy.",
                   relevance="direct")]
    answer = ("No, a KV cache cannot be transferred to a different LLM without loss, except in "
              "the narrow case where the models share identical KV head counts, enabling exact "
              "equivalence under specific linear structural conditions.")
    draft = _draft(direct_answer=answer, answer_facts=[1], sections=[DraftSection(
        heading="A", sentences=[DraftSentence(text="Matched pairs retain 73-98% accuracy.",
                                              facts=[1])])])
    report, stats = render_report(draft, facts, KV_TOPIC, frame=_EXACT_FRAME)
    answer = report.exec_summary.split("\n")[0]
    assert answer.startswith(
        "**Answer:** Not established: no retrieved source shows the 'exact equivalence'")
    # the asserting sentence is removed, not kept after a warning (it contradicted the warning);
    # with nothing else left, the answer states what its cited fact shows
    assert "enabling exact" not in answer
    assert "Matched-KV pairs retain 73-98% of standalone accuracy [1]." in answer
    assert stats["strict_bar_unmet"] == ["exact equivalence"]
    assert stats["answer_sentences_removed"] == 1


def test_only_the_asserting_answer_sentence_is_removed():
    facts = [_fact(1, "Matched-KV pairs retain 73-98% of standalone accuracy.",
                   relevance="direct")]
    answer = ("No: transfer is approximate. Under matched dimensions it achieves exact "
              "equivalence.")
    draft = _draft(direct_answer=answer, answer_facts=[1], sections=[DraftSection(
        heading="A", sentences=[DraftSentence(text="Matched pairs retain 73-98% accuracy.",
                                              facts=[1])])])
    report, _ = render_report(draft, facts, KV_TOPIC, frame=_EXACT_FRAME)
    answer_line = report.exec_summary.split("\n")[0]
    assert "No: transfer is approximate" in answer_line and "achieves exact" not in answer_line


@pytest.mark.parametrize("answer", [
    "No, exact equivalence is not achievable across different LLMs.",
    "No: transfer is approximate, and exact equivalence cannot be guaranteed.",
    "No. Transfer retains 73-98% of accuracy rather than exact equivalence.",
])
def test_negated_mentions_of_the_strict_bar_are_not_flagged(answer):
    facts = [_fact(1, "Matched-KV pairs retain 73-98% of standalone accuracy.",
                   relevance="direct")]
    draft = _draft(direct_answer=answer, answer_facts=[1], sections=[DraftSection(
        heading="A", sentences=[DraftSentence(text="Matched pairs retain 73-98% accuracy.",
                                              facts=[1])])])
    report, stats = render_report(draft, facts, KV_TOPIC, frame=_EXACT_FRAME)
    assert not stats["strict_bar_unmet"] and report.exec_summary.startswith("**Answer:** No")


def test_an_affirmed_strict_bar_backed_by_a_cited_fact_is_kept():
    facts = [_fact(1, "RoPE inversion gives exact equivalence of the rotated keys.",
                   relevance="direct")]
    draft = _draft(direct_answer="Partly, with exact equivalence for rotated keys.",
                   answer_facts=[1], sections=[DraftSection(heading="A", sentences=[
                       DraftSentence(text="RoPE inversion is exact.", facts=[1])])])
    _report, stats = render_report(draft, facts, KV_TOPIC, frame=_EXACT_FRAME)
    assert not stats["strict_bar_unmet"]


def test_analysis_section_is_filtered_flag_gated_and_not_scored():
    facts = [_fact(1, "Berlin is the capital of Germany.")]
    draft = WriterDraftWithAnalysis(
        title="T", direct_answer="Yes.", answer_facts=[1], stance="answered",
        summary=[DraftSentence(text="Berlin is the capital of Germany.", facts=[1])],
        analysis=["A cache encodes model-specific projections, so it cannot transfer as is.",
                  "It holds 42 layers.", "See [1] for details."],
    )
    off, _ = render_report(draft, facts, "t")
    assert all(s.heading != ANALYSIS_HEADING for s in off.sections)
    on, stats = render_report(draft, facts, "t", analysis_enabled=True)
    analysis = [s for s in on.sections if s.heading == ANALYSIS_HEADING]
    assert len(analysis) == 1 and stats["analysis_sentences"] == 1   # digits / [n] dropped
    assert not any(c.section == ANALYSIS_HEADING for c in split_claims(on.model_dump()))


# --- regression replay of the KV-cache incident ----------------------------------------------

async def test_kv_incident_replay_gap_fill_dispatches_and_report_admits_the_gap():
    """Replays trace dad9a8de: 3 generic queries, 14 true-but-off-scope compression facts, 0
    gap-fill workers, and a report presenting compression as the answer. With the frame: the
    queries carry the constraint, the facts don't count as coverage, and the report opens by
    saying no source addresses the constraint."""
    from research_swarm.agents.supervisor import _enforce_plan
    from research_swarm.agents.writer import run_attributed_writer
    from research_swarm.graph.nodes import route_from_dispatch
    from research_swarm.schemas import Critique, CritiqueVerdict

    plan = _enforce_plan(_kv_plan(), KV_FRAME, 3)
    assert all(scope_hit(a.search_query, KV_FRAME) for a in plan.assignments)

    claims = ["CQ preserves model quality with KV cache quantized down to 1 bit.",
              "KVComp reduces KV cache size by up to 83% with under 3% accuracy loss.",
              "Mell balances GPU load by swapping KV caches between GPUs."]
    facts = [_fact(i, claims[i % 3], sq=plan.sub_questions[i % 3], relevance="direct")
             for i in range(14)]
    sends = route_from_dispatch(_make_state(plan=plan, findings=facts, research_rounds=0))
    assert {s.node for s in sends} == {"worker_node"} and len(sends) == 3
    assert all(s.arg["scope"] == "across different LLMs" for s in sends)

    background = [f.model_copy(update={"relevance": "background", "confidence": 0.9})
                  for f in facts[:3]]
    draft = WriterDraft(
        title="KV Cache Migration Feasibility",
        direct_answer="Compression methods can achieve high fidelity lossy migration.",
        answer_facts=[1], stance="partial", sections=[DraftSection(heading="Discussion", sentences=[
            DraftSentence(text="Lossless migration between different LLMs is bottlenecked by "
                               "GPU scheduling.", facts=[3]),
            DraftSentence(text=claims[1], facts=[2]),
        ])],
    )
    state = _make_state(
        plan=plan, findings=background, query=_make_state()["query"].model_copy(
            update={"topic": KV_TOPIC, "audience": "academic"}),
        critiques=[Critique(finding_id=f.id, verdict=CritiqueVerdict.supported, reasoning="r")
                   for f in background],
    )
    report = await run_attributed_writer(state, _llm_returning(draft))
    assert report.exec_summary.startswith(
        "**Answer:** No retrieved source directly addresses across different LLMs")
    assert "bottlenecked by GPU scheduling" not in " ".join(s.body_md for s in report.sections)
    assert report.limitations.count("Not answered by the retrieved evidence") == 3


@pytest.fixture(autouse=True)
def _analysis_off(monkeypatch):
    monkeypatch.setattr(settings, "writer_reasoning_section", False)
