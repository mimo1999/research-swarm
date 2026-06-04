"""Human-checked golden fixtures for regression testing.

Each fixture is a dict with:
    topic             - research topic string
    expected_claims   - list of substrings that MUST appear in the final report
                        (case-insensitive, at least one section or exec_summary)
    forbidden_claims  - list of substrings that must NOT appear (hallucination guards)
    min_sections      - minimum number of ReportSection objects expected
    min_confidence    - minimum acceptable mean finding confidence after fact-check

These were verified by hand against known-good runs and authoritative sources.
Add new fixtures as regressions are discovered; never delete existing ones.
"""
from __future__ import annotations

GOLDEN_FIXTURES: list[dict] = [
    {
        "topic": "GAMs for temporal data",
        "expected_claims": [
            "generalized additive",   # core concept must be named
            "time series",            # temporal context
            "smooth",                 # core mechanism (smooth functions / splines)
            "autocorrelation",        # key challenge addressed by GAMM
        ],
        "forbidden_claims": [
            "blockchain",             # completely unrelated technology
            "large language model",   # off-topic for a stats paper topic
        ],
        "min_sections": 1,
        "min_confidence": 0.05,       # low bar — shallow runs get modest confidence
    },
    {
        "topic": "transformer attention mechanisms",
        "expected_claims": [
            "attention",              # core concept
            "query",                  # Q/K/V terminology
            "key",
        ],
        "forbidden_claims": [
            "electrical transformer", # wrong domain
            "power grid",
        ],
        "min_sections": 1,
        "min_confidence": 0.05,
    },
    {
        "topic": "CRISPR gene editing applications",
        "expected_claims": [
            "crispr",                 # topic must be addressed
            "gene",                   # biological context
            "cas9",                   # most common Cas protein
        ],
        "forbidden_claims": [
            "cryptocurrency",
            "stock market",
        ],
        "min_sections": 1,
        "min_confidence": 0.05,
    },
]
