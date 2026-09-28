"""agents/grounding.py (quote location, evidence windows) and eval/numbers.py."""
from __future__ import annotations

from research_swarm.agents import grounding as g
from research_swarm.eval.numbers import ungrounded_numbers

TEXT = (
    "Background sentence one is here. The trial enrolled 1,200 adults with type 2 diabetes "
    "— mean age 54 — and randomised them to metformin or placebo. After 12 months, "
    "HbA1c fell by 1.2% in the treatment arm. Adverse events were rare.\n\n"
    "A separate paragraph discusses funding and unrelated administrative matters at length."
)


# --- locate_quote ----------------------------------------------------------

def test_exact_match_ignores_case_curly_quotes_dashes_and_spacing():
    quote = 'the trial enrolled  1,200 adults with TYPE 2 diabetes - mean age 54 -'
    m = g.locate_quote(quote, TEXT)
    assert m is not None and m.method == "exact" and m.score == 1.0
    assert TEXT[m.start:m.end].startswith("The trial enrolled 1,200")
    assert TEXT[m.start:m.end].endswith("mean age 54 —")


def test_fuzzy_match_with_one_changed_word():
    text = ("Participants received the drug daily for twelve weeks and were followed up "
            "monthly by trained nurses at home. Nothing else happened.")
    quote = ("Participants received the drug weekly for twelve weeks and were followed up "
             "monthly by trained nurses at home.")
    m = g.locate_quote(quote, text)
    assert m is not None and m.method == "fuzzy" and m.score >= 0.85
    assert text[m.start:m.end].startswith("Participants received the drug daily")


def test_half_changed_quote_is_not_located():
    quote = "Volunteers were given tablets every night for several years and phoned weekly"
    assert g.locate_quote(quote, TEXT) is None


# --- evidence_window --------------------------------------------------------


# --- best_passage -----------------------------------------------------------

def test_best_passage_prefers_the_relevant_sentences():
    span = g.best_passage("HbA1c fell 1.2% in the metformin treatment arm", TEXT)
    assert span is not None
    assert "HbA1c fell by 1.2%" in TEXT[span[0]:span[1]]


# --- ground -----------------------------------------------------------------

def test_ground_quote_passage_none_branches():
    snip, how = g.ground("adults enrolled", "The trial enrolled 1,200 adults with type 2 diabetes",
                         TEXT)
    assert how == "quote" and "1,200 adults" in snip

    snip, how = g.ground("HbA1c fell 1.2% in the treatment arm", "totally invented words here now",
                         TEXT)
    assert how == "passage" and "HbA1c" in snip

    snip, how = g.ground("volcanic basalt lava", "", TEXT)
    assert how == "none" and snip == TEXT[:400]


def test_ground_span_returns_the_located_source_text():
    # the model's quote differs in case and spacing; the span is the text as it is in the source
    _snip, how, span = g.ground_span("adults enrolled",
                                     "the trial  enrolled 1,200 ADULTS", TEXT)
    assert how == "quote" and span == "The trial enrolled 1,200 adults"
    _snip, how, span = g.ground_span("HbA1c fell 1.2% in the treatment arm",
                                     "totally invented words here now", TEXT)
    assert how == "passage" and span in TEXT and "HbA1c" in span
    assert g.ground_span("volcanic basalt lava", "", TEXT)[1:] == ("none", "")


# --- numbers ----------------------------------------------------------------

def test_ungrounded_numbers():
    evidence = "... 12.5 % ... 1200 ..."
    assert ungrounded_numbers("rose 12.5% to 1,200", evidence) == set()
    assert ungrounded_numbers("rose 13%", evidence) == {"13"}
