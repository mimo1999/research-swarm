"""Unit tests for the per-session, per-pool LLM call budget guard.

The "research" pool (supervisor, document workers, dispatch/worker loop) and
the "review" pool (critic/fact-checker/writer/judge) must be independent
counters -- a worker-loop overrun exhausting "research" must never affect
"review"'s remaining allowance, otherwise critic/fact-checker/writer get
starved out and a session with good findings ends up with an empty report.
"""
from __future__ import annotations

import pytest

from research_swarm.runtime.budget import (
    BudgetExceeded,
    clear_budget,
    get_budget,
)


@pytest.fixture(autouse=True)
def _clean_budget_registry():
    yield
    # Best-effort cleanup for any session_id a test might have created.
    for sid in ("sess-pools", "sess-default", "sess-clear", "sess-limits", "sess-tokens"):
        clear_budget(sid)


class TestBudgetPools:

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


class TestTokenBudget:
    """Session-wide token cap: unlike the call-count limits, it spans BOTH
    pools, because a shared/rate-limited key (e.g. Ollama Cloud's account-wide
    allowance) doesn't care which pool the tokens came from."""


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
