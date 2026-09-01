"""Unit tests for the per-session, per-pool LLM call budget guard.

The "research" pool (supervisor, document workers, dispatch/worker loop) and
the "review" pool (critic/fact-checker/writer/judge) must be independent
counters -- a worker-loop overrun exhausting "research" must never affect
"review"'s remaining allowance, otherwise critic/fact-checker/writer get
starved out and a session with good findings ends up with an empty report.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from research_swarm.runtime.budget import (
    BudgetExceeded,
    clear_budget,
    get_budget,
    session_total_tokens,
)


@pytest.fixture(autouse=True)
def _clean_budget_registry():
    yield
    # Best-effort cleanup for any session_id a test might have created.
    for sid in ("sess-pools", "sess-default", "sess-clear", "sess-limits", "sess-tokens"):
        clear_budget(sid)


class TestBudgetPools:
    def test_default_pool_is_research(self):
        budget = get_budget("sess-default", limit=5)
        assert budget.pool == "research"

    def test_research_and_review_are_independent_counters(self):
        research = get_budget("sess-pools", limit=2, pool="research")
        review = get_budget("sess-pools", limit=2, pool="review")

        research.callback.on_chat_model_start({}, [])
        research.callback.on_chat_model_start({}, [])
        # Research pool is now at its limit -- .check() must raise.
        with pytest.raises(BudgetExceeded):
            research.check()

        # Review pool never had a call recorded -- must still pass.
        review.check()
        assert review.used == 0

    def test_get_budget_returns_same_guard_for_same_pool(self):
        first = get_budget("sess-pools", limit=5, pool="research")
        second = get_budget("sess-pools", pool="research")
        assert first is second

    def test_get_budget_returns_different_guards_for_different_pools(self):
        research = get_budget("sess-pools", limit=5, pool="research")
        review = get_budget("sess-pools", limit=5, pool="review")
        assert research is not review

    def test_clear_budget_removes_all_pools_for_session(self):
        research = get_budget("sess-clear", limit=1, pool="research")
        research.callback.on_chat_model_start({}, [])
        get_budget("sess-clear", limit=1, pool="review")

        clear_budget("sess-clear")

        # Fresh guards after clearing -- counters reset, not just one pool.
        fresh_research = get_budget("sess-clear", limit=1, pool="research")
        assert fresh_research.used == 0

    def test_budget_exceeded_carries_pool_name(self):
        research = get_budget("sess-pools", limit=1, pool="research")
        research.callback.on_chat_model_start({}, [])
        with pytest.raises(BudgetExceeded) as exc_info:
            research.check()
        assert exc_info.value.pool == "research"

    def test_pool_limit_defaults_from_matching_setting(self):
        with patch("research_swarm.config.settings") as mock_settings:
            mock_settings.max_llm_calls = 40
            mock_settings.max_review_llm_calls = 10

            research = get_budget("sess-limits", pool="research")
            review = get_budget("sess-limits", pool="review")

        assert research.limit == 40
        assert review.limit == 10


class TestTokenBudget:
    """Session-wide token cap: unlike the call-count limits, it spans BOTH
    pools, because a shared/rate-limited key (e.g. Ollama Cloud's account-wide
    allowance) doesn't care which pool the tokens came from."""

    def test_session_total_tokens_sums_across_pools(self):
        research = get_budget("sess-tokens", limit=100, pool="research")
        review = get_budget("sess-tokens", limit=100, pool="review")
        research._add_tokens(input_tokens=100, output_tokens=50)
        review._add_tokens(input_tokens=10, output_tokens=5)

        assert session_total_tokens("sess-tokens") == 165

    def test_check_raises_on_token_kind_when_session_total_exceeds_limit(self, monkeypatch):
        from research_swarm.config import settings

        monkeypatch.setattr(settings, "max_tokens_per_session", 100)
        budget = get_budget("sess-tokens", limit=1000, pool="research")
        budget._add_tokens(input_tokens=80, output_tokens=30)

        with pytest.raises(BudgetExceeded) as exc_info:
            budget.check()
        assert exc_info.value.kind == "tokens"
        assert exc_info.value.used == 110
        assert exc_info.value.limit == 100

    def test_token_budget_is_session_wide_not_per_pool(self, monkeypatch):
        """Tokens spent under 'research' must trip 'review's check() too --
        the token cap isn't a second set of per-pool counters."""
        from research_swarm.config import settings

        monkeypatch.setattr(settings, "max_tokens_per_session", 100)
        research = get_budget("sess-tokens", limit=1000, pool="research")
        review = get_budget("sess-tokens", limit=1000, pool="review")
        research._add_tokens(input_tokens=90, output_tokens=20)

        with pytest.raises(BudgetExceeded) as exc_info:
            review.check()
        assert exc_info.value.kind == "tokens"

    def test_call_limit_still_reports_kind_calls_when_tokens_are_fine(self, monkeypatch):
        """Regression guard: exceeding the call count (tokens well under
        budget) must still raise with kind='calls', not get relabeled."""
        from research_swarm.config import settings

        monkeypatch.setattr(settings, "max_tokens_per_session", 1_000_000)
        budget = get_budget("sess-tokens", limit=1, pool="research")
        budget.callback.on_chat_model_start({}, [])

        with pytest.raises(BudgetExceeded) as exc_info:
            budget.check()
        assert exc_info.value.kind == "calls"
        assert exc_info.value.pool == "research"

    def test_under_both_limits_passes(self, monkeypatch):
        from research_swarm.config import settings

        monkeypatch.setattr(settings, "max_tokens_per_session", 1_000_000)
        budget = get_budget("sess-tokens", limit=1000, pool="research")
        budget._add_tokens(input_tokens=10, output_tokens=10)
        budget.check()  # must not raise
