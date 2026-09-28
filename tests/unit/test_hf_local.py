"""In-process transformers provider (agents/hf_local.py): batching, structured output, wiring.

No model is loaded: the chat template and the GPU runner are replaced with fakes, so these run
offline and without torch (except the logits-processor test, skipped when torch is missing).
"""
from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from research_swarm.agents import hf_local
from research_swarm.agents.base import get_agent_llm, get_tiered_llm, without_thinking
from research_swarm.config import settings
from research_swarm.runtime.limits import set_llm_context


class Verdict(BaseModel):
    label: str
    score: int


class _Templater:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return f"think={enable_thinking}|" + "|".join(
            f"{m['role']}:{m['content']}" for m in messages)


@pytest.fixture
def fake_backend(monkeypatch):
    """A 'loaded' model whose runner echoes JSON and records every batch it receives."""
    batches: list[list[dict]] = []

    def runner(requests):
        batches.append(requests)
        out = []
        for r in requests:
            user = r["prompt"].rsplit("user:", 1)[-1]
            text = user if user.startswith("RAW:") else json.dumps(
                {"label": user, "score": len(requests)})
            out.append({"text": text.removeprefix("RAW:"), "input_tokens": 10,
                        "output_tokens": 5})
        return out

    monkeypatch.setattr(hf_local, "_state",
                        {"model_id": "fake/model", "templater": _Templater()})
    monkeypatch.setattr(hf_local, "_runner", runner)
    monkeypatch.setattr(hf_local, "_batchers", {})
    monkeypatch.setattr(settings, "hf_batch_window_s", 0.05)
    monkeypatch.setattr(settings, "hf_max_batch", 4)
    return batches


def _llm(**kw):
    return hf_local.ChatHFLocal(model="fake/model", **kw)


async def test_concurrent_structured_calls_share_one_batch(fake_backend):
    set_llm_context("huggingface", "s1")
    structured = _llm().with_structured_output(Verdict)
    results = await asyncio.gather(*(
        structured.ainvoke([SystemMessage(content="sys"), HumanMessage(content=f"q{i}")])
        for i in range(3)
    ))
    assert [r.label for r in results] == ["q0", "q1", "q2"]
    assert len(fake_backend) == 1 and len(fake_backend[0]) == 3
    # every row carries its schema for the JSON constraint
    assert all(r["schema"]["properties"].keys() == {"label", "score"} for r in fake_backend[0])


async def test_batches_are_capped_and_split_per_session(fake_backend):
    structured = _llm().with_structured_output(Verdict)

    async def call(session, i):
        set_llm_context("huggingface", session)
        return await structured.ainvoke([HumanMessage(content=f"{session}-{i}")])

    await asyncio.gather(*(call("a", i) for i in range(5)), *(call("b", i) for i in range(2)))
    sizes = sorted(len(b) for b in fake_backend)
    assert sizes == [1, 2, 4]  # session a: 4 + 1 (cap 4); session b: its own batch of 2
    for batch in fake_backend:
        assert len({r["prompt"].rsplit("user:", 1)[-1].split("-")[0] for r in batch}) == 1


async def test_unparseable_reply_raises_with_raw_output(fake_backend):
    set_llm_context("huggingface", "s2")
    structured = _llm().with_structured_output(Verdict)
    with pytest.raises(OutputParserException) as info:
        await structured.ainvoke([HumanMessage(content='RAW:{"label": "x"')])
    assert info.value.llm_output == '{"label": "x"'


async def test_usage_and_thinking_off(fake_backend):
    set_llm_context("huggingface", "s3")
    llm = without_thinking(_llm(reasoning=True), max_tokens=321)
    assert llm.reasoning is False and llm.num_predict == 321
    msg = await llm.ainvoke([HumanMessage(content="hi")])
    assert msg.usage_metadata["total_tokens"] == 15
    request = fake_backend[0][0]
    assert request["max_new_tokens"] == 321 and request["prompt"].startswith("think=False")
    assert request["schema"] is None


def test_factory_routes_huggingface_provider(monkeypatch):
    monkeypatch.setattr(settings, "hf_model_id", "google/gemma-4-E2B-it")
    monkeypatch.setattr(settings, "tier_fast_provider", "huggingface")
    llm = get_tiered_llm("fast")
    assert isinstance(llm, hf_local.ChatHFLocal) and llm.model == "google/gemma-4-E2B-it"
    assert isinstance(get_tiered_llm("standard", provider_override="huggingface"),
                      hf_local.ChatHFLocal)
    assert isinstance(get_agent_llm(provider="huggingface", model="m"), hf_local.ChatHFLocal)


def test_gpu_duration_estimate_is_bounded(monkeypatch):
    monkeypatch.setattr(settings, "hf_gpu_duration_max_s", 120)
    assert hf_local.estimate_gpu_seconds([{"max_new_tokens": 250}]) == 25
    assert hf_local.estimate_gpu_seconds([{"max_new_tokens": 250},
                                          {"max_new_tokens": 8192}]) == 120


def test_constraint_processor_masks_only_constrained_live_rows():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    fns = [lambda row, ids: [1, 3], None, lambda row, ids: [2]]
    proc = hf_local._constraint_processor(fns, prompt_len=2, eos_ids={9})
    input_ids = torch.tensor([[5, 5, 4], [5, 5, 4], [5, 5, 9]])  # row 2 already emitted EOS
    scores = proc(input_ids, torch.zeros(3, 10))
    assert torch.isfinite(scores[0]).nonzero().flatten().tolist() == [1, 3]
    assert torch.isfinite(scores[1]).all()
    assert torch.isfinite(scores[2]).all()
