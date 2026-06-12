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
    ResearchPlan,
    ResearchQuery,
    Source,
)
from research_swarm.schemas.state import AgentState

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
    def test_extracts_sources_from_tool_message(self):
        from langchain_core.messages import ToolMessage

        from research_swarm.agents.researcher import _extract_sources_from_messages

        payload = [{"url": "https://ex.com", "title": "Ex",
                    "snippet": "content", "source_type": "web", "credibility_score": 0.8}]
        msg = ToolMessage(content=json.dumps(payload), tool_call_id="c1")
        result = _extract_sources_from_messages([msg])

        assert len(result) == 1
        assert result[0].url == "https://ex.com"

    def test_ignores_non_tool_messages(self):
        from langchain_core.messages import HumanMessage

        from research_swarm.agents.researcher import _extract_sources_from_messages

        result = _extract_sources_from_messages([HumanMessage(content="hello")])
        assert result == []

    def test_handles_malformed_json_gracefully(self):
        from langchain_core.messages import ToolMessage

        from research_swarm.agents.researcher import _extract_sources_from_messages

        bad = ToolMessage(content="not valid json {{", tool_call_id="c1")
        result = _extract_sources_from_messages([bad])
        assert result == []

    def test_caps_at_ten_sources(self):
        from langchain_core.messages import ToolMessage

        from research_swarm.agents.researcher import _extract_sources_from_messages

        payload = [{"url": f"https://ex{i}.com", "snippet": "s",
                    "source_type": "web", "credibility_score": 0.5}
                   for i in range(15)]
        msg = ToolMessage(content=json.dumps(payload), tool_call_id="c1")
        result = _extract_sources_from_messages([msg])
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
        from research_swarm.agents.critic import run_critic

        finding = _make_finding("Q?")
        hallucinated = Critique(
            finding_id="hallucinated-id-000",
            verdict=CritiqueVerdict.supported,
            reasoning="ok",
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=hallucinated)

        state = _make_state(findings=[finding])
        (result,) = await run_critic(state, mock_llm)

        assert result.finding_id == finding.id, \
            "critic must overwrite the hallucinated finding_id with the real one"

    @pytest.mark.asyncio
    async def test_null_finding_id_is_fixed(self):
        """LLM returning finding_id=null must not be dropped; critic stamps the real ID."""
        from research_swarm.agents.critic import run_critic

        finding = _make_finding("Q?")
        null_id_critique = Critique(
            finding_id=None,
            verdict=CritiqueVerdict.weak,
            reasoning="needs more evidence",
            suggested_followup="Find primary sources",
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=null_id_critique)

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
    async def test_updates_confidence_from_llm_score(self):
        from research_swarm.agents.fact_checker import FactCheckResult, run_fact_checker

        finding = _make_finding(confidence=0.5)
        finding = finding.model_copy(update={"evidence": [_make_source()]})

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=FactCheckResult(confidence_score=0.93, notes="well supported")
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
