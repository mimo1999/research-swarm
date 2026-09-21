"""research_swarm.eval.claims: sentence splitting, judging plumbing, and metric arithmetic."""
from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from research_swarm.eval.claims import (
    Claim,
    ClaimVerdict,
    ClaimVerdicts,
    claim_metrics,
    judge_claims,
    split_claims,
)


def _report(summary="", sections=()):
    return {
        "exec_summary": summary,
        "sections": [{"heading": h, "body_md": b, "citations": []} for h, b in sections],
    }


class TestSplitClaims:
    def test_markers_are_removed_and_captured(self):
        claims = split_claims(_report("Paris is the capital of France [1]. It hosts the Louvre [2, 3]."))
        assert [c.text for c in claims] == [
            "Paris is the capital of France.", "It hosts the Louvre.",
        ]
        assert [c.citations for c in claims] == [[1], [2, 3]]
        assert [c.idx for c in claims] == [1, 2]


def _verdict(n, *, cw=True, cited="yes", corpus="yes", contradicted=False) -> ClaimVerdict:
    return ClaimVerdict(claim=n, checkworthy=cw, cited_support=cited, corpus_support=corpus,
                        contradicted=contradicted)


def _fake_llm(policy):
    """A judge that answers every listed claim via policy(claim_number, cites_line)."""
    llm = MagicMock()
    prompts: list[str] = []

    async def ainvoke(messages):
        user = messages[1].content
        prompts.append(user)
        out = []
        for m in re.finditer(r"C(\d+): .*\n    cites: (.*)", user):
            v = policy(int(m.group(1)), m.group(2))
            if v is not None:
                out.append(v)
        return ClaimVerdicts(verdicts=out)

    llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=ainvoke)
    llm.prompts = prompts
    return llm


DOCS = [
    {"url": "u1", "title": "Doc One", "text": "alpha beta"},
    {"url": "u2", "title": "Doc Two", "text": "gamma delta"},
]


class TestJudgeClaims:

    @pytest.mark.asyncio
    async def test_dangling_citations_are_counted_without_the_judge(self):
        claims = [Claim(1, "Alpha is beta.", [1, 9]), Claim(2, "Gamma is delta.", [3])]
        llm = _fake_llm(lambda n, cites: _verdict(n))
        _, dangling = await judge_claims(claims, ["u1", "not-a-supplied-doc"], DOCS, llm)
        # [9] is beyond the reference list; [3] too; ref 2 points at a URL that is not supplied
        assert dangling == {1: (1, 2), 2: (1, 1)}

    @pytest.mark.asyncio
    async def test_uncited_and_all_dangling_claims_are_forced_to_no_cited_support(self):
        claims = [Claim(1, "Uncited fact here.", []), Claim(2, "Dangling cite here.", [7]),
                  Claim(3, "Properly cited fact.", [1])]
        llm = _fake_llm(lambda n, cites: _verdict(n, cited="yes"))      # judge is over-generous
        verdicts, _ = await judge_claims(claims, ["u1"], DOCS, llm)
        assert verdicts[1].cited_support == "no"
        assert verdicts[2].cited_support == "no"
        assert verdicts[3].cited_support == "yes"
        assert verdicts[1].corpus_support == "yes"                      # only the cite part is forced


class TestClaimMetrics:
    def _claims(self):
        return [
            Claim(1, "supported and cited", [1]),
            Claim(2, "supported but uncited", []),
            Claim(3, "cited but unsupported", [1]),
            Claim(4, "not checkworthy", [1]),
            Claim(5, "contradicted claim", [2]),
            Claim(6, "judge failed", [1]),
        ]

    def _verdicts(self):
        return {
            1: _verdict(1, cited="yes", corpus="yes"),
            2: _verdict(2, cited="no", corpus="yes"),
            3: _verdict(3, cited="no", corpus="no"),
            4: _verdict(4, cw=False, cited="no", corpus="no"),
            5: _verdict(5, cited="no", corpus="no", contradicted=True),
            6: None,
        }

    def test_arithmetic(self):
        m = claim_metrics(self._claims(), self._verdicts(), {1: (0, 1), 3: (1, 1), 5: (0, 1)})

        assert m["n_claims"] == 6 and m["n_judged"] == 5 and m["n_checkworthy"] == 4
        assert m["judge_error_rate"] == pytest.approx(1 / 6, abs=1e-3)
        assert m["faithfulness"] == 0.5                       # claims 1, 2 of checkworthy 1,2,3,5
        assert m["unsupported_rate"] == 0.5                   # claims 3, 5
        assert m["citation_recall"] == 0.25                   # only claim 1 is entailed by its cites
        assert m["citation_correctness"] == pytest.approx(1 / 3, abs=1e-3)   # cited cw: 1, 3, 5
        assert m["citation_completeness"] == 0.5              # supported cw: 1 (cited), 2 (not)
        assert m["n_contradicted"] == 1
        assert m["dangling_citation_rate"] == pytest.approx(1 / 3, abs=1e-3)
        assert m["n_citations"] == 3
