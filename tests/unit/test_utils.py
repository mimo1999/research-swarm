"""Unit tests for agents/_utils.py -- shared helpers, especially the
structured-output parse-failure recovery path.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class _Synthesis(BaseModel):
    claim: str = Field(...)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class TestRecoverFromParseFailure:
    def test_recovers_schema_echo_shape(self):
        """Exact reproduction of the gemma4:31b-cloud failure mode observed in
        production: the model wraps the real values under "properties" instead
        of returning a flat object."""
        from research_swarm.agents._utils import recover_from_parse_failure

        exc = Exception(
            'Failed to parse _Synthesis from completion '
            '{"properties": {"claim": "GLP-1 agonists show mixed results.", '
            '"confidence": 0.95}, "required": ["claim"], "type": "object"}. '
            'Got: 1 validation error for _Synthesis\n'
            'claim\n  Field required [type=missing, ...]'
        )
        result = recover_from_parse_failure(exc, _Synthesis)
        assert result is not None
        assert result.claim == "GLP-1 agonists show mixed results."
        assert result.confidence == 0.95


class TestRecoverFromBadEscapes:
    """gemma4:31b-cloud's second observed failure mode: raw LaTeX inside a
    claim (\\in, \\mathbb, \\times, \\text{...}) breaks json.loads with
    "Invalid \\escape" because the backslashes were never escaped for JSON.
    See _repair_json_backslashes / _recover_from_bad_escapes."""

    def test_recovers_real_observed_latex_failure(self):
        """Exact reproduction of the failure from a live LoRA/QLoRA run:
        Worker[academic] synthesis failed with 'Invalid json output' on a
        claim containing \\(W\\in\\mathbb{R}^{d\\times k}\\)."""
        from langchain_core.exceptions import OutputParserException

        from research_swarm.agents._utils import recover_from_parse_failure

        raw = (
            r'{"claim":"LoRA freezes the pretrained weight matrix '
            r'\(W\in\mathbb{R}^{d\times k}\) and injects a trainable low-rank '
            r'update \(\Delta W=A\,B^{\top}\).","confidence":0.9}'
        )
        exc = OutputParserException(f"Invalid json output: {raw}", llm_output=raw)

        result = recover_from_parse_failure(exc, _Synthesis)
        assert result is not None
        assert result.confidence == 0.9
        assert r"\(W\in\mathbb{R}^{d\times k}\)" in result.claim


def test_repair_json_brackets():
    import json

    from research_swarm.agents._utils import repair_json_brackets

    # the live failure: a list closed by "}" before its "]" (sections never closed)
    broken = '{"a": [{"b": [1]}, {"c": [2]}\n}'
    assert json.loads(repair_json_brackets(broken)) == {"a": [{"b": [1]}, {"c": [2]}]}
    # brackets and escaped quotes inside strings are untouched
    tricky = '{"t": "x ] } [ \\" y", "l": [1, 2'
    assert json.loads(repair_json_brackets(tricky)) == {"t": 'x ] } [ " y', "l": [1, 2]}
    # a stray closer is dropped; valid JSON is unchanged
    assert json.loads(repair_json_brackets('{"a": 1}]')) == {"a": 1}
    valid = '{"a": [1, {"b": "c"}]}'
    assert repair_json_brackets(valid) == valid
