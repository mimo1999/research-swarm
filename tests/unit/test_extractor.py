"""agents/extractor.py: packing, numbered facts, grounding, caps, drop accounting."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from research_swarm.agents import extractor as ex
from research_swarm.config import settings

TEXT_A = "Paris is the capital of France. It has 2,100,000 residents in the city proper."
TEXT_B = "Berlin is the capital of Germany. Its metropolitan region is home to many people."


def _src(i, text, url=None):
    return {"url": url or f"http://x/{i}", "title": f"Doc {i}", "text": text,
            "source_type": "pdf", "credibility_score": 0.8}


def _llm(*facts):
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=ex.Extraction(facts=list(facts)))
    return llm


def _fact(source=1, sq=1, claim="Paris is the capital of France.", quote=""):
    return ex.ExtractedFact(source=source, sub_question=sq, claim=claim, quote=quote)


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    monkeypatch.setattr(settings, "extract_max_facts_per_pair", 3)


# --- pack_sources ------------------------------------------------------------

def test_small_docs_pack_into_one_batch_in_order():
    docs = [_src(i, "x" * 600) for i in range(10)]
    batches = ex.pack_sources(docs, 12_000)
    assert len(batches) == 1 and [d["title"] for d in batches[0]] == [f"Doc {i}" for i in range(10)]


def test_oversized_doc_is_split_into_labelled_parts():
    text = ". ".join(f"Sentence {i} has some filler words in it" for i in range(1500))
    assert len(text) > 30_000
    batches = ex.pack_sources([_src(1, text)], 12_000)
    flat = [d for b in batches for d in b]
    assert len(flat) >= 3 and all(len(d["text"]) <= 12_000 for d in flat)
    assert flat[0]["title"].endswith(f"(part 1/{len(flat)})")
    assert {d["url"] for d in flat} == {"http://x/1"}


# --- extract_facts -----------------------------------------------------------

async def test_facts_are_grounded_mapped_and_deterministic():
    f = _fact(quote="Paris is the capital of France.")
    llm = _llm(f)
    a = await ex.extract_facts("Q?", ["What is the capital of France?"], [_src(1, TEXT_A)], llm)
    b = await ex.extract_facts("Q?", ["What is the capital of France?"], [_src(1, TEXT_A)], llm)
    assert len(a) == 1
    assert a[0].sub_question == "What is the capital of France?"       # canonical string
    assert a[0].grounding == "quote" and a[0].confidence == 0.6
    assert "capital of France" in a[0].evidence[0].snippet
    assert a[0].id == b[0].id                                          # merge-by-id friendly


async def test_out_of_range_indexes_are_dropped_and_traced(monkeypatch):
    events = []
    monkeypatch.setattr(ex, "trace_event", lambda *a, **k: events.append((a[1], k)))
    llm = _llm(_fact(source=9), _fact(sq=7), _fact(claim="  "), _fact())
    out = await ex.extract_facts("Q?", ["sq one"], [_src(1, TEXT_A)], llm)
    assert len(out) == 1
    dropped = next(k for name, k in events if name == "extractor.dropped")
    assert dropped["bad_source"] == 1 and dropped["bad_sub_question"] == 1
    assert dropped["empty_claim"] == 1 and dropped["n"] == 3


async def test_per_pair_cap():
    facts = [_fact(claim=f"Paris fact number {i} about the capital of France") for i in range(5)]
    out = await ex.extract_facts("Q?", ["sq"], [_src(1, TEXT_A)], _llm(*facts))
    assert len(out) == 3
    # a different source is a different pair
    facts = facts[:3] + [_fact(source=2, claim="Berlin is the capital of Germany")]
    out = await ex.extract_facts("Q?", ["sq"], [_src(1, TEXT_A), _src(2, TEXT_B)], _llm(*facts))
    assert len(out) == 4


async def test_transient_failure_returns_empty(monkeypatch):
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=TimeoutError("boom"))
    monkeypatch.setattr(ex, "ainvoke_with_retry", AsyncMock(side_effect=TimeoutError("boom")))
    assert await ex.extract_facts("Q?", ["sq"], [_src(1, TEXT_A)], llm) == []


def test_quote_is_required_in_the_schema_but_tolerated_when_missing():
    """Constrained decoding must be made to emit a quote (gemma4:e2b skipped the optional field
    on every fact), but a reply without one still parses and falls back to passage grounding."""
    from research_swarm.agents.extractor import ExtractedFact, Extraction

    fact_schema = Extraction.model_json_schema()["$defs"]["ExtractedFact"]
    assert "quote" in fact_schema["required"]
    fact = ExtractedFact.model_validate({"source": 1, "sub_question": 1, "claim": "c"})
    assert fact.quote == ""
