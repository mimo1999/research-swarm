"""Direct unit tests for every agent function.

These tests call run_supervisor / run_researcher / run_critic /
run_fact_checker / run_writer and get_agent_llm directly — without going
through the graph nodes — so logic regressions inside the agent functions
are caught independently of the graph wiring.

All LLM calls are replaced with AsyncMock / MagicMock.  No API keys needed.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_swarm.schemas import (
    Critique,
    CritiqueVerdict,
    FinalReport,
    Finding,
    ReportSection,
    ResearchPlan,
    ResearchQuery,
    Source,
)
from research_swarm.schemas.state import AgentState
from research_swarm.schemas.worker import WorkerRole

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_plan(*questions: str) -> ResearchPlan:
    return ResearchPlan(
        sub_questions=list(questions) or ["What is AI safety?"],
        strategy="Search web and arXiv",
        required_tools=["web_search"],
    )


def _make_finding(sub_q: str = "test q", confidence: float = 0.7) -> Finding:
    return Finding(claim=f"Claim for {sub_q}", confidence=confidence, sub_question=sub_q)


def _make_source(url: str = "https://example.com") -> Source:
    return Source(url=url, title="Test Source", snippet="Some evidence text.")


def _make_state(**overrides) -> AgentState:
    base: AgentState = {
        "messages":        [],
        "query":           ResearchQuery(topic="AI safety", audience="technical"),
        "plan":            None,
        "findings":        [],
        "critiques":       [],
        "draft_report":    None,
        "final_report":    None,
        "human_feedback":  None,
        "iteration_count": 0,
        "next_agent":      None,
        "session_id":      "test-sess",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# supervisor.py — _build_system_prompt()
# ---------------------------------------------------------------------------

class TestSupervisorSystemPrompt:
    """"AT MOST N" alone gave the model no pressure to use the sub-question
    budget -- observed producing a single sub-question at standard depth
    (max=5) for a topic explicitly comparing two named techniques, silently
    dropping one side of the comparison. The prompt now fixes an exact count
    (no min/max range for the model to reason about) plus explicit
    comparative-topic guidance; these tests pin both."""

    def test_standard_depth_states_an_exact_count(self):
        from research_swarm.agents.supervisor import _build_system_prompt

        prompt = _build_system_prompt("standard")
        assert "EXACTLY 4" in prompt
        assert "BETWEEN" not in prompt

    def test_deep_depth_states_an_exact_count(self):
        from research_swarm.agents.supervisor import _build_system_prompt

        prompt = _build_system_prompt("deep")
        assert "EXACTLY 6" in prompt

    def test_shallow_depth_states_exactly_one(self):
        """shallow is intentionally fixed at 1 sub-question -- that's the
        whole point of the fast tier, not a bug to harden away."""
        from research_swarm.agents.supervisor import _build_system_prompt

        prompt = _build_system_prompt("shallow")
        assert "EXACTLY 1" in prompt

    def test_instructs_covering_both_sides_of_a_comparison(self):
        from research_swarm.agents.supervisor import _build_system_prompt

        prompt = _build_system_prompt("standard")
        assert "MUST cover each thing individually AND their direct comparison" in prompt
        assert "never collapse a comparison topic into sub-questions about only one side" in prompt

    def test_shallow_comparison_guidance_addresses_single_question_case(self):
        """Shallow only gets one sub-question total, so the comparison
        guidance for it must be phrased for a single question, not a set."""
        from research_swarm.agents.supervisor import _build_system_prompt

        prompt = _build_system_prompt("shallow")
        assert "phrase that one question to address the comparison directly" in prompt


# ---------------------------------------------------------------------------
# get_agent_llm()
# ---------------------------------------------------------------------------

class TestGetAgentLlm:
    """get_agent_llm() must return the right class for each provider."""

    def test_anthropic_returns_chat_anthropic(self):
        from langchain_anthropic import ChatAnthropic

        from research_swarm.agents.base import get_agent_llm
        with patch("research_swarm.agents.base.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock(spec=ChatAnthropic)
            get_agent_llm(provider="anthropic", model="claude-haiku-3-5")
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-3-5"

    def test_openai_returns_chat_openai(self):
        from langchain_openai import ChatOpenAI

        from research_swarm.agents.base import get_agent_llm
        with patch("research_swarm.agents.base.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock(spec=ChatOpenAI)
            get_agent_llm(provider="openai", model="gpt-4o-mini")
        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["model"] == "gpt-4o-mini"

    def test_ollama_returns_chat_ollama(self):
        from research_swarm.agents.base import get_agent_llm
        mock_ollama_cls = MagicMock()
        with patch.dict("sys.modules", {"langchain_ollama": MagicMock(ChatOllama=mock_ollama_cls)}):
            get_agent_llm(provider="ollama", model="llama3.2")
        mock_ollama_cls.assert_called_once()
        assert mock_ollama_cls.call_args.kwargs["model"] == "llama3.2"

    def test_unknown_provider_raises_value_error(self):
        from research_swarm.agents.base import get_agent_llm
        with pytest.raises(ValueError, match="Unsupported provider"):
            get_agent_llm(provider="bedrock", model="any")

    def test_defaults_to_settings_provider_and_model(self, monkeypatch):
        from research_swarm.agents.base import get_agent_llm
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "default_model_provider", "anthropic")
        monkeypatch.setattr(settings, "default_model_name", "claude-haiku-3-5")
        with patch("research_swarm.agents.base.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_agent_llm()
        assert mock_cls.call_args.kwargs["model"] == "claude-haiku-3-5"

    def test_temperature_is_forwarded(self):
        from research_swarm.agents.base import get_agent_llm
        with patch("research_swarm.agents.base.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_agent_llm(provider="anthropic", temperature=0.9)
        assert mock_cls.call_args.kwargs["temperature"] == 0.9


class TestGetTieredLlm:
    """get_tiered_llm('standard', ...) must auto-pick each provider's lowest-grade model."""

    def test_standard_tier_anthropic_uses_haiku(self):
        from research_swarm.agents.base import get_tiered_llm
        with patch("research_swarm.agents.base.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_tiered_llm(tier="standard", provider_override="anthropic")
        assert mock_cls.call_args.kwargs["model"] == "claude-haiku-4-5-20251001"

    def test_standard_tier_openai_uses_gpt5_nano(self):
        from research_swarm.agents.base import get_tiered_llm
        with patch("research_swarm.agents.base.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_tiered_llm(tier="standard", provider_override="openai")
        assert mock_cls.call_args.kwargs["model"] == "gpt-5-nano"

    def test_standard_tier_ollama_local_uses_small_local_model(self, monkeypatch):
        from research_swarm.agents.base import get_tiered_llm
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "ollama_deployment", "local")
        mock_ollama_cls = MagicMock()
        with patch.dict("sys.modules", {"langchain_ollama": MagicMock(ChatOllama=mock_ollama_cls)}):
            get_tiered_llm(tier="standard", provider_override="ollama")
        assert mock_ollama_cls.call_args.kwargs["model"] == settings.tier_standard_model_local

    def test_standard_tier_ollama_cloud_uses_small_cloud_model(self, monkeypatch):
        from research_swarm.agents.base import get_tiered_llm
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "ollama_deployment", "cloud")
        mock_ollama_cls = MagicMock()
        with patch.dict("sys.modules", {"langchain_ollama": MagicMock(ChatOllama=mock_ollama_cls)}):
            get_tiered_llm(tier="standard", provider_override="ollama")
        assert mock_ollama_cls.call_args.kwargs["model"] == settings.tier_standard_model_cloud

    def test_provider_override_beats_static_tier_provider(self, monkeypatch):
        """A session's chosen provider must win over the static tier_standard_provider."""
        from research_swarm.agents.base import get_tiered_llm
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "tier_standard_provider", "ollama")
        with patch("research_swarm.agents.base.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_tiered_llm(tier="standard", provider_override="anthropic")
        mock_cls.assert_called_once()
        assert mock_cls.call_args.kwargs["model"] == "claude-haiku-4-5-20251001"

    def test_no_override_falls_back_to_static_tier_provider(self, monkeypatch):
        from research_swarm.agents.base import get_tiered_llm
        from research_swarm.config import settings
        monkeypatch.setattr(settings, "tier_fast_provider", "anthropic")
        monkeypatch.setattr(settings, "tier_fast_model", "claude-haiku-3-5")
        with patch("research_swarm.agents.base.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_tiered_llm(tier="fast")
        assert mock_cls.call_args.kwargs["model"] == "claude-haiku-3-5"


# ---------------------------------------------------------------------------
# run_researcher() — top-level function
# ---------------------------------------------------------------------------

class TestRunResearcher:
    """Direct tests for run_researcher() logic."""

    def _mock_llm(self, synthesis_claim: str = "Test claim", synthesis_conf: float = 0.75):
        """Return a mock LLM that stops the tool loop immediately and returns a synthesis."""
        from research_swarm.agents.researcher import FindingSynthesis

        ai_stop = MagicMock()
        ai_stop.tool_calls = []  # no tool calls → loop ends immediately

        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.ainvoke = AsyncMock(return_value=ai_stop)

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=FindingSynthesis(claim=synthesis_claim, confidence=synthesis_conf)
        )
        return mock_llm

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_plan(self):
        from research_swarm.agents.researcher import run_researcher
        state = _make_state(plan=None)
        result = await run_researcher(state, MagicMock(), [])
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_all_sub_questions_supported(self):
        """Sub-questions with 'supported' critiques must be skipped."""
        from research_swarm.agents.researcher import run_researcher

        finding = _make_finding("What is AI safety?")
        critique = Critique(
            finding_id=finding.id,
            verdict=CritiqueVerdict.supported,
            reasoning="Well supported.",
        )
        state = _make_state(
            plan=_make_plan("What is AI safety?"),
            findings=[finding],
            critiques=[critique],
        )
        result = await run_researcher(state, MagicMock(), [])
        assert result == []

    @pytest.mark.asyncio
    async def test_produces_finding_per_sub_question(self):
        from research_swarm.agents.researcher import run_researcher

        state = _make_state(plan=_make_plan("Q1?", "Q2?"))
        result = await run_researcher(state, self._mock_llm(), [])

        assert len(result) == 2
        sub_questions = {f.sub_question for f in result}
        assert sub_questions == {"Q1?", "Q2?"}

    @pytest.mark.asyncio
    async def test_finding_has_correct_claim_and_confidence(self):
        from research_swarm.agents.researcher import run_researcher

        state = _make_state(plan=_make_plan("What is X?"))
        llm = self._mock_llm(synthesis_claim="X is important.", synthesis_conf=0.82)
        (finding,) = await run_researcher(state, llm, [])

        assert finding.claim == "X is important."
        assert finding.confidence == pytest.approx(0.82)

    @pytest.mark.asyncio
    async def test_re_research_preserves_finding_id(self):
        """Re-researching a weak finding must reuse the original ID so the
        merge-by-id reducer overwrites the old finding in AgentState."""
        from research_swarm.agents.researcher import run_researcher

        original = _make_finding("What is X?", confidence=0.3)
        critique = Critique(
            finding_id=original.id, verdict=CritiqueVerdict.weak, reasoning="Needs more."
        )
        state = _make_state(
            plan=_make_plan("What is X?"),
            findings=[original],
            critiques=[critique],
        )
        (new_finding,) = await run_researcher(state, self._mock_llm(), [])
        assert new_finding.id == original.id, "re-research must reuse the original finding ID"

    @pytest.mark.asyncio
    async def test_synthesis_failure_produces_placeholder_finding(self):
        """If the synthesis LLM raises, the researcher falls back to a low-confidence placeholder."""
        from research_swarm.agents.researcher import run_researcher

        ai_stop = MagicMock()
        ai_stop.tool_calls = []

        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.ainvoke = AsyncMock(return_value=ai_stop)

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            side_effect=Exception("LLM timeout")
        )

        state = _make_state(plan=_make_plan("Risky question?"))
        (finding,) = await run_researcher(state, mock_llm, [])

        assert finding.confidence == pytest.approx(0.2)
        assert "Research incomplete" in finding.claim

    @pytest.mark.asyncio
    async def test_synthesis_schema_echo_failure_recovers_real_claim(self):
        """When the LLM echoes the JSON schema instead of a flat instance (the
        gemma4:31b-cloud failure mode found in the manual-vs-swarm benchmark),
        the real claim/confidence must be recovered instead of discarded behind
        the generic placeholder."""
        from research_swarm.agents.researcher import run_researcher

        ai_stop = MagicMock()
        ai_stop.tool_calls = []

        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.ainvoke = AsyncMock(return_value=ai_stop)

        parse_exc = Exception(
            'Failed to parse FindingSynthesis from completion '
            '{"properties": {"claim": "Exenatide phase 3 found no significant benefit.", '
            '"confidence": 0.95}, "required": ["claim"], "type": "object"}. '
            'Got: 1 validation error for FindingSynthesis\nclaim\n  Field required'
        )
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=parse_exc)

        state = _make_state(plan=_make_plan("Did the phase 3 trial succeed?"))
        (finding,) = await run_researcher(state, mock_llm, [])

        assert finding.claim == "Exenatide phase 3 found no significant benefit."
        assert finding.confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# _synthesis_prompt() -- shared by run_researcher and run_worker
# ---------------------------------------------------------------------------

class TestSynthesisPrompt:
    def test_instructs_preserving_exact_numbers(self):
        """Without this, workers compress quantitative results (effect sizes,
        p-values, CIs) into a qualitative gist, and the numbers never reach
        the report even though the evidence contained them."""
        from research_swarm.agents.researcher import _synthesis_prompt

        prompt = _synthesis_prompt("Did the trial show a significant effect?")
        assert "p-value" in prompt or "p-values" in prompt
        assert "qualitative gist" in prompt
        assert "Did the trial show a significant effect?" in prompt

    def test_instructs_checking_comparability_before_comparing_numbers(self):
        """A worker gathering evidence across multiple sources/tool calls can
        end up with numbers from mismatched conditions (e.g. one technique's
        result on a 7B model vs another's on an 82M model) and synthesize them
        into one claim as if directly comparable. The prompt must tell it to
        check comparability -- same scale/benchmark/setup -- before presenting
        two numbers side by side, and to flag it in the claim when they don't
        match rather than implying equivalence."""
        from research_swarm.agents.researcher import _synthesis_prompt

        prompt = _synthesis_prompt("How do X and Y compare?")
        assert "comparable conditions" in prompt
        assert "not directly comparable" in prompt


# ---------------------------------------------------------------------------
# run_worker() -- top-level function (the active production research path)
# ---------------------------------------------------------------------------

class TestRunWorkerDirect:
    def _mock_llm(self, ainvoke_side_effect=None, ainvoke_return_value=None):
        ai_stop = MagicMock()
        ai_stop.tool_calls = []

        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.ainvoke = AsyncMock(return_value=ai_stop)

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        if ainvoke_side_effect is not None:
            mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
                side_effect=ainvoke_side_effect
            )
        else:
            mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
                return_value=ainvoke_return_value
            )
        return mock_llm

    @pytest.mark.asyncio
    async def test_synthesis_failure_produces_placeholder_finding(self):
        from research_swarm.agents.workers import run_worker

        mock_llm = self._mock_llm(ainvoke_side_effect=Exception("LLM timeout"))
        state = _make_state(plan=_make_plan("Risky question?"))
        finding = await run_worker("Risky question?", WorkerRole.general, state, mock_llm, [])

        assert finding.confidence == pytest.approx(0.2)
        assert "Research incomplete" in finding.claim

    @pytest.mark.asyncio
    async def test_synthesis_schema_echo_failure_recovers_real_claim(self):
        """Same recovery behaviour as run_researcher, verified on the actual
        production worker path (worker_node -> run_worker)."""
        from research_swarm.agents.workers import run_worker

        parse_exc = Exception(
            'Failed to parse FindingSynthesis from completion '
            '{"properties": {"claim": "Lixisenatide slowed motor decline in phase 2.", '
            '"confidence": 0.9}, "required": ["claim"], "type": "object"}. '
            'Got: 1 validation error for FindingSynthesis\nclaim\n  Field required'
        )
        mock_llm = self._mock_llm(ainvoke_side_effect=parse_exc)
        state = _make_state(plan=_make_plan("Did lixisenatide show benefit?"))
        finding = await run_worker(
            "Did lixisenatide show benefit?", WorkerRole.skeptic, state, mock_llm, []
        )

        assert finding.claim == "Lixisenatide slowed motor decline in phase 2."
        assert finding.confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# ReAct tool loop (_run_tool_loop)
# ---------------------------------------------------------------------------

class TestRunToolLoop:
    """Tests for the tool-calling loop inside researcher."""

    def _mock_tool(self, name: str = "web_search", result=None):
        tool = MagicMock()
        tool.name = name
        tool.invoke.return_value = result or [
            {"url": "https://example.com", "title": "T", "snippet": "S",
             "source_type": "web", "credibility_score": 0.7}
        ]
        return tool

    @pytest.mark.asyncio
    async def test_stops_when_no_tool_calls(self):
        from langchain_core.messages import SystemMessage

        from research_swarm.agents.researcher import _run_tool_loop

        ai_done = MagicMock()
        ai_done.tool_calls = []
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=ai_done)

        msgs = await _run_tool_loop(
            "test q", "sess", mock_llm, {}, SystemMessage(content="sys")
        )
        # system + human + ai response = 3 messages
        assert len(msgs) == 3
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_invokes_tool_and_appends_tool_message(self):
        from langchain_core.messages import SystemMessage, ToolMessage

        from research_swarm.agents.researcher import _run_tool_loop

        tool = self._mock_tool("web_search")

        # First LLM call: wants to call web_search
        ai_with_call = MagicMock()
        ai_with_call.tool_calls = [
            {"name": "web_search", "args": {"query": "AI"}, "id": "call_1"}
        ]
        # Second LLM call: done
        ai_done = MagicMock()
        ai_done.tool_calls = []

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[ai_with_call, ai_done])

        msgs = await _run_tool_loop(
            "test q", "sess", mock_llm, {"web_search": tool},
            SystemMessage(content="sys")
        )

        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "call_1"
        tool.invoke.assert_called_once_with({"query": "AI"})

    @pytest.mark.asyncio
    async def test_unknown_tool_name_injects_error_message(self):
        from langchain_core.messages import SystemMessage, ToolMessage

        from research_swarm.agents.researcher import _run_tool_loop

        ai_with_call = MagicMock()
        ai_with_call.tool_calls = [
            {"name": "nonexistent_tool", "args": {}, "id": "call_x"}
        ]
        ai_done = MagicMock()
        ai_done.tool_calls = []

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[ai_with_call, ai_done])

        msgs = await _run_tool_loop(
            "q", "sess", mock_llm, {},  # empty tool map
            SystemMessage(content="sys")
        )

        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "Unknown tool" in tool_msgs[0].content


# ---------------------------------------------------------------------------
# _extract_sources_from_messages()
# ---------------------------------------------------------------------------

class TestExtractSources:
    @pytest.mark.asyncio
    async def test_extracts_sources_from_tool_message(self):
        from langchain_core.messages import ToolMessage

        from research_swarm.agents.researcher import _extract_sources_from_messages

        payload = [{"url": "https://ex.com", "title": "Ex",
                    "snippet": "content", "source_type": "web", "credibility_score": 0.8}]
        msg = ToolMessage(content=json.dumps(payload), tool_call_id="c1")
        result = await _extract_sources_from_messages([msg])

        assert len(result) == 1
        assert result[0].url == "https://ex.com"

    @pytest.mark.asyncio
    async def test_ignores_non_tool_messages(self):
        from langchain_core.messages import HumanMessage

        from research_swarm.agents.researcher import _extract_sources_from_messages

        result = await _extract_sources_from_messages([HumanMessage(content="hello")])
        assert result == []

    @pytest.mark.asyncio
    async def test_handles_malformed_json_gracefully(self):
        from langchain_core.messages import ToolMessage

        from research_swarm.agents.researcher import _extract_sources_from_messages

        bad = ToolMessage(content="not valid json {{", tool_call_id="c1")
        result = await _extract_sources_from_messages([bad])
        assert result == []

    @pytest.mark.asyncio
    async def test_caps_at_ten_sources(self):
        from langchain_core.messages import ToolMessage

        from research_swarm.agents.researcher import _extract_sources_from_messages

        payload = [{"url": f"https://ex{i}.com", "snippet": "s",
                    "source_type": "web", "credibility_score": 0.5}
                   for i in range(15)]
        msg = ToolMessage(content=json.dumps(payload), tool_call_id="c1")
        result = await _extract_sources_from_messages([msg])
        assert len(result) == 10


# ---------------------------------------------------------------------------
# _rank_sources_by_relevance() -- relevance ranking instead of flat truncation
# ---------------------------------------------------------------------------

class _FakeEmbedModel:
    """Crude but deterministic: embeds by presence of domain keywords, so a
    relevant source scores higher on cosine similarity to a GLP-1/Parkinson's
    sub-question without needing a real model. Implements both the single
    and batch embedding methods -- _rank_sources_by_relevance batches all
    source texts in one get_text_embedding_batch call rather than embedding
    them one at a time."""

    _KEYWORDS = ["glp-1", "parkinson", "dopaminergic", "neuroprotective", "exenatide"]

    def _embed_one(self, text: str) -> list[float]:
        return [float(kw in text.lower()) for kw in self._KEYWORDS] + [1.0]

    def get_text_embedding(self, text):
        return self._embed_one(text)

    def get_text_embedding_batch(self, texts):
        return [self._embed_one(t) for t in texts]


class TestRankSourcesByRelevance:
    @pytest.mark.asyncio
    async def test_returns_unchanged_when_under_cap(self):
        from research_swarm.agents.researcher import (
            MAX_EVIDENCE_SOURCES,
            _rank_sources_by_relevance,
        )

        sources = [_make_source(f"https://ex{i}.com") for i in range(MAX_EVIDENCE_SOURCES)]
        result = await _rank_sources_by_relevance("Any question?", sources)
        assert result == sources

    @pytest.mark.asyncio
    async def test_ranks_by_embedding_similarity_to_sub_question(self):
        from research_swarm.agents.researcher import _rank_sources_by_relevance
        from research_swarm.schemas import Source

        relevant = Source(
            url="https://relevant.com", title="GLP-1 receptor agonists in Parkinson's disease",
            snippet="Exenatide showed neuroprotective effects in dopaminergic neurons.",
        )
        irrelevant = Source(
            url="https://irrelevant.com", title="Ionic liquids for battery electrolytes",
            snippet="Room-temperature ionic liquids improve battery cycle life.",
        )
        sources = [irrelevant, relevant]  # relevant one deliberately listed second

        with patch(
            "research_swarm.rag.indexes.get_embed_model", return_value=_FakeEmbedModel(),
        ):
            result = await _rank_sources_by_relevance(
                "What neuroprotective effects do GLP-1 agonists have in Parkinson's disease?",
                sources, top_k=1,
            )

        assert len(result) == 1
        assert result[0].url == "https://relevant.com"

    @pytest.mark.asyncio
    async def test_embeds_sources_in_one_batched_call(self):
        """Regression test for the efficiency fix: sources must be embedded
        via a single get_text_embedding_batch call, not one get_text_embedding
        call per source."""
        from research_swarm.agents.researcher import _rank_sources_by_relevance

        sources = [_make_source(f"https://ex{i}.com") for i in range(10)]
        fake = _FakeEmbedModel()
        with patch.object(fake, "get_text_embedding_batch", wraps=fake.get_text_embedding_batch) as batch_spy, \
             patch.object(fake, "get_text_embedding", wraps=fake.get_text_embedding) as single_spy, \
             patch("research_swarm.rag.indexes.get_embed_model", return_value=fake):
            await _rank_sources_by_relevance("Any question?", sources, top_k=3)

        batch_spy.assert_called_once()
        # get_text_embedding is still used once, for the sub-question itself.
        single_spy.assert_called_once()

    @pytest.mark.asyncio
    async def test_embedding_failure_falls_back_to_first_n(self):
        from research_swarm.agents.researcher import _rank_sources_by_relevance

        sources = [_make_source(f"https://ex{i}.com") for i in range(10)]
        with patch(
            "research_swarm.rag.indexes.get_embed_model",
            side_effect=Exception("model unavailable"),
        ):
            result = await _rank_sources_by_relevance("Any question?", sources, top_k=3)

        assert [s.url for s in result] == ["https://ex0.com", "https://ex1.com", "https://ex2.com"]


class TestExtractSourcesWithSubQuestion:
    @pytest.mark.asyncio
    async def test_ranks_and_caps_when_sub_question_given(self):
        from langchain_core.messages import ToolMessage

        from research_swarm.agents.researcher import (
            MAX_EVIDENCE_SOURCES,
            _extract_sources_from_messages,
        )

        payload = [{"url": f"https://ex{i}.com", "snippet": "s",
                    "source_type": "web", "credibility_score": 0.5}
                   for i in range(15)]
        msg = ToolMessage(content=json.dumps(payload), tool_call_id="c1")

        with patch(
            "research_swarm.rag.indexes.get_embed_model",
            side_effect=Exception("model unavailable"),
        ):
            result = await _extract_sources_from_messages([msg], sub_question="Any question?")

        # Falls back to first-N-in-order on embedding failure, capped at
        # MAX_EVIDENCE_SOURCES (same value as the legacy cap -- see the
        # constant's docstring for why the cap itself didn't change, only
        # how sources are selected).
        assert len(result) == MAX_EVIDENCE_SOURCES

    @pytest.mark.asyncio
    async def test_no_sub_question_keeps_legacy_cap_of_ten(self):
        from langchain_core.messages import ToolMessage

        from research_swarm.agents.researcher import _extract_sources_from_messages

        payload = [{"url": f"https://ex{i}.com", "snippet": "s",
                    "source_type": "web", "credibility_score": 0.5}
                   for i in range(15)]
        msg = ToolMessage(content=json.dumps(payload), tool_call_id="c1")
        result = await _extract_sources_from_messages([msg])
        assert len(result) == 10


# ---------------------------------------------------------------------------
# run_critic() — direct
# ---------------------------------------------------------------------------

class TestRunCriticDirect:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_findings(self):
        from research_swarm.agents.critic import run_critic
        state = _make_state(findings=[])
        result = await run_critic(state, MagicMock())
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_findings_already_critiqued(self):
        from research_swarm.agents.critic import run_critic

        finding = _make_finding()
        critique = Critique(
            finding_id=finding.id, verdict=CritiqueVerdict.supported, reasoning="ok"
        )
        state = _make_state(findings=[finding], critiques=[critique])
        result = await run_critic(state, MagicMock())
        assert result == []

    @pytest.mark.asyncio
    async def test_forces_finding_id_on_critique(self):
        """The critic must stamp the real finding_id onto the LLM output,
        in case the LLM hallucinated a different ID."""
        from research_swarm.agents.critic import CritiqueBatch, run_critic

        finding = _make_finding("Q?")
        hallucinated = Critique(
            finding_id="hallucinated-id-000",
            verdict=CritiqueVerdict.supported,
            reasoning="ok",
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=CritiqueBatch(critiques=[hallucinated])
        )

        state = _make_state(findings=[finding])
        (result,) = await run_critic(state, mock_llm)

        assert result.finding_id == finding.id, \
            "critic must overwrite the hallucinated finding_id with the real one"

    @pytest.mark.asyncio
    async def test_null_finding_id_is_fixed(self):
        """LLM returning finding_id=null must not be dropped; critic stamps the real ID."""
        from research_swarm.agents.critic import CritiqueBatch, run_critic

        finding = _make_finding("Q?")
        null_id_critique = Critique(
            finding_id=None,
            verdict=CritiqueVerdict.weak,
            reasoning="needs more evidence",
            suggested_followup="Find primary sources",
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=CritiqueBatch(critiques=[null_id_critique])
        )

        state = _make_state(findings=[finding])
        (result,) = await run_critic(state, mock_llm)

        assert result.finding_id == finding.id, \
            "null finding_id from LLM must be replaced with the real finding id"
        assert result.verdict == CritiqueVerdict.weak

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_supported_verdict(self):
        """If the structured output call raises, critic falls back to 'supported'.

        Using 'supported' (rather than 'weak') as the safe default prevents an
        LLM outage from triggering expensive re-research loops.
        """
        from research_swarm.agents.critic import run_critic

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            side_effect=Exception("structured output failed")
        )
        state = _make_state(findings=[_make_finding()])
        (result,) = await run_critic(state, mock_llm)

        assert result.verdict == CritiqueVerdict.supported
        assert result.reasoning != ""

    @pytest.mark.asyncio
    async def test_recovers_critique_from_malformed_json_completion(self):
        """A JSON-escape parse failure (e.g. LaTeX in reasoning) must be
        recovered via recover_from_parse_failure instead of always
        discarding real model output for the generic 'supported' fallback."""
        from langchain_core.exceptions import OutputParserException

        from research_swarm.agents.critic import run_critic

        finding = _make_finding("Q?")
        raw = (
            f'{{"critiques":[{{"finding_id":"{finding.id}","verdict":"weak",'
            r'"reasoning":"Missing \theta calibration.","suggested_followup":""}]}'
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            side_effect=OutputParserException(f"Invalid json output: {raw}", llm_output=raw)
        )
        state = _make_state(findings=[finding])
        (result,) = await run_critic(state, mock_llm)

        assert result.verdict == CritiqueVerdict.weak
        assert result.finding_id == finding.id


# ---------------------------------------------------------------------------
# run_fact_checker() — direct
# ---------------------------------------------------------------------------

class TestRunFactCheckerDirect:
    @pytest.mark.asyncio
    async def test_llm_failure_preserves_original_confidence(self):
        """If the LLM raises during fact-checking, the original confidence is kept."""
        from research_swarm.agents.fact_checker import run_fact_checker

        finding = _make_finding(confidence=0.65)
        finding = finding.model_copy(update={"evidence": [_make_source()]})

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            side_effect=Exception("LLM down")
        )
        state = _make_state(findings=[finding])
        (updated,) = await run_fact_checker(state, mock_llm)

        assert updated.confidence == pytest.approx(0.65), \
            "confidence must be preserved when fact-checker LLM fails"

    @pytest.mark.asyncio
    async def test_finding_with_no_evidence_penalised_to_01(self):
        from research_swarm.agents.fact_checker import run_fact_checker

        finding = _make_finding(confidence=0.9)  # high confidence but no evidence
        state = _make_state(findings=[finding])
        (updated,) = await run_fact_checker(state, MagicMock())

        assert updated.confidence == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_refuted_findings_are_skipped(self):
        from research_swarm.agents.fact_checker import run_fact_checker

        finding = _make_finding()
        critique = Critique(
            finding_id=finding.id, verdict=CritiqueVerdict.refuted, reasoning="wrong"
        )
        state = _make_state(findings=[finding], critiques=[critique])
        result = await run_fact_checker(state, MagicMock())
        assert result == []

    @pytest.mark.asyncio
    async def test_recovers_result_from_malformed_json_completion(self):
        """A JSON-escape parse failure must be recovered instead of always
        falling back to preserving the original (possibly wrong) confidence."""
        from langchain_core.exceptions import OutputParserException

        from research_swarm.agents.fact_checker import run_fact_checker

        finding = _make_finding(confidence=0.5)
        finding = finding.model_copy(update={"evidence": [_make_source()]})
        raw = r'{"results":[{"confidence_score":0.9,"notes":"Backed by \frac{1}{2} of sources."}]}'
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            side_effect=OutputParserException(f"Invalid json output: {raw}", llm_output=raw)
        )
        state = _make_state(findings=[finding])
        (updated,) = await run_fact_checker(state, mock_llm)

        assert updated.confidence == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_updates_confidence_from_llm_score(self):
        from research_swarm.agents.fact_checker import (
            FactCheckBatch,
            FactCheckResult,
            run_fact_checker,
        )

        finding = _make_finding(confidence=0.5)
        finding = finding.model_copy(update={"evidence": [_make_source()]})

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=FactCheckBatch(
                results=[FactCheckResult(confidence_score=0.93, notes="well supported")]
            )
        )
        state = _make_state(findings=[finding])
        (updated,) = await run_fact_checker(state, mock_llm)

        assert updated.confidence == pytest.approx(0.93)


# ---------------------------------------------------------------------------
# run_writer() — direct
# ---------------------------------------------------------------------------

class TestRunWriterDirect:
    def _make_valid_state(self, findings=None, critiques=None):
        """State with one supported finding (high confidence) ready for the writer."""
        f = _make_finding("Main question?", confidence=0.8)
        f = f.model_copy(update={"evidence": [_make_source()]})
        return _make_state(
            plan=_make_plan("Main question?"),
            findings=findings if findings is not None else [f],
            critiques=critiques or [],
        )

    @pytest.mark.asyncio
    async def test_no_valid_findings_returns_insufficient_report(self):
        """Writer with zero valid findings must return a minimal 'insufficient' report."""
        from research_swarm.agents.writer import run_writer

        # Confidence must be below the writer's filter floor (currently 0.1)
        low_conf = _make_finding(confidence=0.05)
        state = _make_state(findings=[low_conf])
        report = await run_writer(state, MagicMock())

        assert "Insufficient" in report.exec_summary

    @pytest.mark.asyncio
    async def test_refuted_findings_excluded_from_report(self):
        """Refuted findings must never appear in the report."""
        from research_swarm.agents.writer import run_writer

        refuted = _make_finding("Refuted claim", confidence=0.8)
        critique = Critique(
            finding_id=refuted.id, verdict=CritiqueVerdict.refuted, reasoning="wrong"
        )
        # Only refuted finding → no valid findings → insufficient report
        state = _make_state(findings=[refuted], critiques=[critique])
        report = await run_writer(state, MagicMock())

        assert "Insufficient" in report.exec_summary

    @pytest.mark.asyncio
    async def test_llm_failure_produces_fallback_report(self):
        """If the structured-output LLM raises, writer must return a fallback report
        that includes sections built from valid findings — not raise an exception."""
        from research_swarm.agents.writer import run_writer

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            side_effect=Exception("LLM down")
        )
        state = self._make_valid_state()
        report = await run_writer(state, mock_llm)

        assert isinstance(report, FinalReport)
        assert len(report.sections) >= 1, "fallback report must include sections from findings"
        assert "fallback" in report.limitations.lower()

    @pytest.mark.asyncio
    async def test_references_populated_from_evidence(self):
        """Report references must include every unique source URL in the findings."""
        from research_swarm.agents.writer import run_writer

        src = _make_source("https://paper.com/1")
        finding = _make_finding(confidence=0.8)
        finding = finding.model_copy(update={"evidence": [src]})
        expected_report = FinalReport(
            title="T", exec_summary="S",
            references=[],   # writer should backfill this
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=expected_report
        )
        state = _make_state(findings=[finding])
        report = await run_writer(state, mock_llm)

        urls = [r.url for r in report.references]
        assert "https://paper.com/1" in urls

    @pytest.mark.asyncio
    async def test_human_feedback_is_threaded_into_prompt(self):
        """Human feedback must appear in the prompt sent to the LLM."""
        from research_swarm.agents.writer import run_writer

        captured_messages = []

        async def capture(msgs):
            captured_messages.extend(msgs)
            return FinalReport(title="T", exec_summary="S")

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=capture)

        state = self._make_valid_state()
        state["human_feedback"] = "Focus on economic impact."
        await run_writer(state, mock_llm)

        all_text = " ".join(str(m.content) for m in captured_messages)
        assert "Focus on economic impact." in all_text

    @pytest.mark.asyncio
    async def test_prompt_instructs_preserving_exact_numbers(self):
        """The writer must be told to preserve exact figures rather than
        paraphrasing a quantitative result into a qualitative gist -- this is
        the fix for reports losing effect sizes/p-values/CIs during synthesis."""
        from research_swarm.agents.writer import run_writer

        captured_messages = []

        async def capture(msgs):
            captured_messages.extend(msgs)
            return FinalReport(title="T", exec_summary="S")

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=capture)

        state = self._make_valid_state()
        await run_writer(state, mock_llm)

        system_text = str(captured_messages[0].content)
        assert "p-values" in system_text or "p-value" in system_text
        assert "qualitative gist" in system_text


# ---------------------------------------------------------------------------
# _faithfulness_rewrite() -- targeted rewrite of under-grounded sections
# ---------------------------------------------------------------------------

class TestFaithfulnessRewrite:
    def _report(self, weak_body="unsupported claim", strong_body="supported claim"):
        return FinalReport(
            title="T", exec_summary="S",
            sections=[
                ReportSection(heading="Weak Section", body_md=weak_body, citations=[1]),
                ReportSection(heading="Strong Section", body_md=strong_body, citations=[1]),
            ],
        )

    @pytest.mark.asyncio
    async def test_returns_original_unchanged_when_all_sections_pass(self):
        from research_swarm.agents.writer import _faithfulness_rewrite

        report = self._report()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock()

        with patch(
            "research_swarm.eval.faithfulness.score_sections",
            return_value=[
                {"heading": "Weak Section", "score": 0.9, "citations": [1]},
                {"heading": "Strong Section", "score": 0.9, "citations": [1]},
            ],
        ):
            result, score = await _faithfulness_rewrite(
                report, [], structured_llm, MagicMock(), MagicMock()
            )

        assert result is report
        assert score == pytest.approx(0.9)
        structured_llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_rewrite_shows_previous_report_and_names_only_weak_sections(self):
        """The rewrite call must not be blind: it needs to see its own previous
        output and know exactly which section(s) failed, or it has no way to
        target the fix rather than blindly regenerating the whole report."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        from research_swarm.agents.writer import _faithfulness_rewrite

        report = self._report()
        rewritten_report = FinalReport(title="T2", exec_summary="S2")

        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=rewritten_report)

        scores_by_call = [
            [
                {"heading": "Weak Section", "score": 0.1, "citations": [1]},
                {"heading": "Strong Section", "score": 0.9, "citations": [1]},
            ],
            [
                {"heading": "Weak Section", "score": 0.8, "citations": [1]},
                {"heading": "Strong Section", "score": 0.9, "citations": [1]},
            ],
        ]
        system_msg = SystemMessage(content="sys")
        user_msg = HumanMessage(content="user")

        with patch(
            "research_swarm.eval.faithfulness.score_sections", side_effect=scores_by_call
        ):
            result, score = await _faithfulness_rewrite(
                report, [], structured_llm, system_msg, user_msg
            )

        assert result.title == "T2"
        assert score == pytest.approx(0.85)
        call_messages = structured_llm.ainvoke.call_args.args[0]
        assert call_messages[0] is system_msg
        assert call_messages[1] is user_msg
        assert isinstance(call_messages[2], AIMessage)
        assert "Weak Section" in call_messages[2].content  # previous report shown back
        assert "Weak Section" in call_messages[3].content
        assert "Strong Section" not in call_messages[3].content  # only the weak one named

    @pytest.mark.asyncio
    async def test_retries_until_section_passes_within_attempt_cap(self):
        """A section still weak after one rewrite gets another attempt (up to
        MAX_FAITHFULNESS_REWRITES) instead of giving up after a single try."""
        from research_swarm.agents.writer import (
            MAX_FAITHFULNESS_REWRITES,
            _faithfulness_rewrite,
        )

        report = self._report()
        rewrite_1 = FinalReport(title="T2", exec_summary="S2")
        rewrite_2 = FinalReport(title="T3", exec_summary="S3")

        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(side_effect=[rewrite_1, rewrite_2])

        scores_by_call = [
            [{"heading": "Weak Section", "score": 0.1, "citations": [1]}],  # initial
            [{"heading": "Weak Section", "score": 0.2, "citations": [1]}],  # after rewrite 1 -- still weak
            [{"heading": "Weak Section", "score": 0.9, "citations": [1]}],  # after rewrite 2 -- passes
        ]

        with patch(
            "research_swarm.eval.faithfulness.score_sections", side_effect=scores_by_call
        ):
            result, score = await _faithfulness_rewrite(
                report, [], structured_llm, MagicMock(), MagicMock()
            )

        assert MAX_FAITHFULNESS_REWRITES >= 2
        assert result.title == "T3"
        assert score == pytest.approx(0.9)
        assert structured_llm.ainvoke.await_count == 2
        # Second attempt's prior-report turn must reflect rewrite 1's output, not the original.
        second_call_messages = structured_llm.ainvoke.await_args_list[1].args[0]
        assert "T2" in second_call_messages[-2].content

    @pytest.mark.asyncio
    async def test_scoring_failure_skips_rewrite(self):
        from research_swarm.agents.writer import _faithfulness_rewrite

        report = self._report()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock()

        with patch(
            "research_swarm.eval.faithfulness.score_sections",
            side_effect=Exception("embedding unavailable"),
        ):
            result, score = await _faithfulness_rewrite(
                report, [], structured_llm, MagicMock(), MagicMock()
            )

        assert result is report
        assert score is None
        structured_llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_rewrite_failure_keeps_original_report(self):
        from research_swarm.agents.writer import _faithfulness_rewrite

        report = self._report()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(side_effect=Exception("LLM down"))

        with patch(
            "research_swarm.eval.faithfulness.score_sections",
            return_value=[
                {"heading": "Weak Section", "score": 0.1, "citations": [1]},
                {"heading": "Strong Section", "score": 0.9, "citations": [1]},
            ],
        ):
            result, score = await _faithfulness_rewrite(
                report, [], structured_llm, MagicMock(), MagicMock()
            )

        assert result is report
        assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _collect_references() and _format_findings() helpers in writer
# ---------------------------------------------------------------------------

class TestWriterHelpers:
    def test_collect_references_deduplicates_by_url(self):
        from research_swarm.agents.writer import _collect_references

        src = _make_source("https://dup.com")
        f1 = _make_finding().model_copy(update={"evidence": [src]})
        f2 = _make_finding().model_copy(update={"evidence": [src]})  # same URL

        refs = _collect_references([f1, f2])
        assert len(refs) == 1
        assert refs[0].url == "https://dup.com"

    def test_collect_references_preserves_insertion_order(self):
        from research_swarm.agents.writer import _collect_references

        s1 = _make_source("https://first.com")
        s2 = _make_source("https://second.com")
        finding = _make_finding().model_copy(update={"evidence": [s1, s2]})

        refs = _collect_references([finding])
        assert [r.url for r in refs] == ["https://first.com", "https://second.com"]

    def test_format_findings_lists_citations_with_titles(self):
        """Citations must be paired with source titles, not bare index numbers --
        the writer prompt relies on titles to tell sources apart when deciding
        which citation actually supports a given sentence."""
        from research_swarm.agents.writer import _format_findings

        s1 = Source(url="https://a.com", title="Paper A: Benchmarks")
        s2 = Source(url="https://b.com", title="Paper B: Ablation Study")
        finding = _make_finding().model_copy(update={"evidence": [s1, s2]})

        text = _format_findings([finding], [s1, s2])

        assert "[1] Paper A: Benchmarks" in text
        assert "[2] Paper B: Ablation Study" in text

    def test_format_findings_source_without_matching_reference_omitted(self):
        """A finding's evidence source that isn't in the references list
        (already deduplicated elsewhere) simply doesn't get a citation."""
        from research_swarm.agents.writer import _format_findings

        s1 = Source(url="https://a.com", title="Paper A")
        finding = _make_finding().model_copy(update={"evidence": [s1]})

        text = _format_findings([finding], [])  # empty references list

        assert "(no sources)" in text
