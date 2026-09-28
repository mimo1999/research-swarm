"""Evidence packet (agents/packet.py): segmentation, sentence IDs, dedupe, budget, screening."""
from __future__ import annotations

from research_swarm.agents import packet as pk


def test_split_sentences_keeps_abbreviations_and_decimals_together():
    text = ("Smith et al. found a 0.92 odds ratio (e.g. in adults). The effect was 3.5 times "
            "larger in Fig. 2 than expected! Was it real? Yes.")
    assert pk.split_sentences(text) == [
        "Smith et al. found a 0.92 odds ratio (e.g. in adults).",
        "The effect was 3.5 times larger in Fig. 2 than expected!",
        "Was it real?",
        "Yes.",
    ]


def _src(url, text, title="T"):
    return {"url": url, "title": title, "text": text}


async def test_sources_that_fit_go_in_whole_with_stable_ids_and_no_duplicates():
    sources = [
        _src("u1", "Aspirin lowers fever in adults. It was tested in 200 patients."),
        _src("u2", "Ibuprofen is an NSAID. Aspirin lowers fever in adults."),   # dup of S1.1
    ]
    packet = await pk.build_packet("Does aspirin lower fever?", [], sources, budget_tokens=1000)
    assert [s.id for s in packet.sentences] == ["S1.1", "S1.2", "S2.1"]
    assert packet.stats["fit"] == "whole" and packet.stats["duplicates"] == 1
    assert packet.sub_questions == ["Does aspirin lower fever?"]
    rendered = packet.render()
    assert "[S1] T\nS1.1 Aspirin lowers fever in adults." in rendered
    assert pk.EvidencePacket.from_dict(packet.to_dict()).by_id()["S2.1"].text == \
        "Ibuprofen is an NSAID."


def _filler(n, topic="gardening"):
    return " ".join(f"Sentence {i} is about {topic} and nothing else here." for i in range(n))


async def test_over_budget_keeps_best_passages_in_source_order():
    sources = [
        _src("u1", _filler(8)),
        _src("u2", "Aspirin reduced fever by 1.5 degrees in adults. " + _filler(3)),
    ]
    packet = await pk.build_packet("Does aspirin reduce fever in adults?", [], sources,
                                   budget_tokens=60)
    assert packet.stats["fit"] == "scored"
    assert packet.tokens() <= 60
    kept = [s.id for s in packet.sentences]
    assert "S2.1" in kept                                      # the only on-topic passage
    order = [(s.source, s.index) for s in packet.sentences]
    assert order == sorted(order)                              # source order kept


async def test_screener_runs_only_on_overflow_and_drops_rejected_passages():
    calls = []

    async def screener(question, sub_questions, passages):
        calls.append(len(passages))
        return ["aspirin" in p.text.lower() for p in passages]     # passage-level verdicts

    small = [_src("u1", "Aspirin lowers fever in adults.")]
    await pk.build_packet("Does aspirin lower fever?", [], small, 1000, screener=screener)
    assert calls == []                                         # fits: no screening call

    # Lexically, the "fever in adults" filler outranks nothing; the screener must still reject
    # every passage that never mentions aspirin, even ones sharing the question's words.
    big = [_src("u1", _filler(8, topic="fever in adults")),
           _src("u2", "Aspirin lowered fever in one trial. " + _filler(3))]
    packet = await pk.build_packet("Does aspirin lower fever in adults?", [], big, 60,
                                   screener=screener)
    assert calls and packet.stats["fit"] == "screened"
    assert packet.stats["rejected_passages"] >= 2
    assert packet.sentences[0].id == "S2.1"                    # the relevant passage leads
    assert packet.tokens() <= 60


async def test_sentences_are_assigned_to_their_best_sub_question():
    sources = [_src("u1", "Aspirin lowers fever quickly. Aspirin irritates the stomach lining.")]
    packet = await pk.build_packet("Aspirin effects", ["Does aspirin lower fever?",
                                                       "Does aspirin harm the stomach?"],
                                   sources, 1000)
    by_id = packet.by_id()
    assert by_id["S1.1"].sub_question == "Does aspirin lower fever?"
    assert by_id["S1.2"].sub_question == "Does aspirin harm the stomach?"
