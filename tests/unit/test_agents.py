"""Direct unit tests for every agent function.

These tests call get_agent_llm / get_tiered_llm and the supervisor prompt directly, without
going through the graph nodes.

All LLM calls are replaced with AsyncMock / MagicMock.  No API keys needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from research_swarm.schemas import (
    Finding,
    ResearchPlan,
    ResearchQuery,
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


    def test_instructs_covering_both_sides_of_a_comparison(self):
        from research_swarm.agents.supervisor import _build_system_prompt

        prompt = _build_system_prompt("standard")
        assert "MUST cover each thing individually AND their direct comparison" in prompt
        assert "never collapse a comparison topic into sub-questions about only one side" in prompt


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


    def test_unknown_provider_raises_value_error(self):
        from research_swarm.agents.base import get_agent_llm
        with pytest.raises(ValueError, match="Unsupported provider"):
            get_agent_llm(provider="bedrock", model="any")


class TestGetTieredLlm:
    """get_tiered_llm('standard', ...) must auto-pick each provider's lowest-grade model."""


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
