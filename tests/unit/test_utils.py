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

    def test_recovers_direct_shape(self):
        from research_swarm.agents._utils import recover_from_parse_failure

        exc = Exception(
            'Failed to parse _Synthesis from completion '
            '{"claim": "Direct claim.", "confidence": 0.7, "extra_junk": "ignored"}. '
            'Got: some other validation error'
        )
        result = recover_from_parse_failure(exc, _Synthesis)
        assert result is not None
        assert result.claim == "Direct claim."
        assert result.confidence == 0.7

    def test_returns_none_when_no_completion_marker(self):
        from research_swarm.agents._utils import recover_from_parse_failure

        result = recover_from_parse_failure(Exception("connection reset"), _Synthesis)
        assert result is None

    def test_returns_none_for_unparseable_json(self):
        from research_swarm.agents._utils import recover_from_parse_failure

        exc = Exception("Failed to parse _Synthesis from completion not json at all. Got: error")
        result = recover_from_parse_failure(exc, _Synthesis)
        assert result is None

    def test_returns_none_when_claim_field_absent_everywhere(self):
        from research_swarm.agents._utils import recover_from_parse_failure

        exc = Exception(
            'Failed to parse _Synthesis from completion '
            '{"properties": {"notes": "no claim field here"}, "type": "object"}. Got: error'
        )
        result = recover_from_parse_failure(exc, _Synthesis)
        assert result is None

    def test_returns_none_when_recovered_value_still_fails_validation(self):
        """confidence out of the [0, 1] bound must not produce an invalid instance."""
        from research_swarm.agents._utils import recover_from_parse_failure

        exc = Exception(
            'Failed to parse _Synthesis from completion '
            '{"properties": {"claim": "ok", "confidence": 5.0}, "type": "object"}. Got: error'
        )
        result = recover_from_parse_failure(exc, _Synthesis)
        assert result is None

    def test_ignores_unknown_fields_when_recovering(self):
        from research_swarm.agents._utils import recover_from_parse_failure

        exc = Exception(
            'Failed to parse _Synthesis from completion '
            '{"properties": {"claim": "ok", "confidence": 0.4, "bogus_field": true}, '
            '"type": "object"}. Got: error'
        )
        result = recover_from_parse_failure(exc, _Synthesis)
        assert result is not None
        assert result.claim == "ok"


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

    def test_does_not_corrupt_latex_commands_starting_with_escape_letters(self):
        """\\text, \\frac, \\theta, \\times all start with a letter (t, f, u) that
        is ALSO a valid JSON escape starter (\\t, \\f, \\u) -- a naive "leave
        valid-looking escapes alone" repair would silently turn \\text into a
        tab character followed by "ext" instead of fixing the parse failure."""
        from langchain_core.exceptions import OutputParserException

        from research_swarm.agents._utils import recover_from_parse_failure

        raw = r'{"claim":"Uses \text{NF4} and \frac{1}{2} with \theta and \times.","confidence":0.8}'
        exc = OutputParserException(f"Invalid json output: {raw}", llm_output=raw)

        result = recover_from_parse_failure(exc, _Synthesis)
        assert result is not None
        assert result.claim == r"Uses \text{NF4} and \frac{1}{2} with \theta and \times."

    def test_strips_markdown_code_fence_before_repairing(self):
        from langchain_core.exceptions import OutputParserException

        from research_swarm.agents._utils import recover_from_parse_failure

        raw = '```json\n{"claim":"Fenced claim with \\in a symbol.","confidence":0.6}\n```'
        exc = OutputParserException(f"Invalid json output: {raw}", llm_output=raw)

        result = recover_from_parse_failure(exc, _Synthesis)
        assert result is not None
        assert result.confidence == 0.6

    def test_trims_leading_and_trailing_prose_around_the_json_object(self):
        from langchain_core.exceptions import OutputParserException

        from research_swarm.agents._utils import recover_from_parse_failure

        raw = 'Here is the result: {"claim":"Trimmed claim.","confidence":0.5} Hope that helps!'
        exc = OutputParserException(f"Invalid json output: {raw}", llm_output=raw)

        result = recover_from_parse_failure(exc, _Synthesis)
        assert result is not None
        assert result.claim == "Trimmed claim."

    def test_returns_none_when_llm_output_missing(self):
        """A plain exception with no llm_output attribute (e.g. the
        schema-echo failure mode) must fall through cleanly, not raise."""
        from research_swarm.agents._utils import _recover_from_bad_escapes

        assert _recover_from_bad_escapes(Exception("connection reset"), _Synthesis) is None

    def test_returns_none_when_still_unparseable_after_repair(self):
        from langchain_core.exceptions import OutputParserException

        from research_swarm.agents._utils import recover_from_parse_failure

        raw = "{not json at all, missing braces and quotes"
        exc = OutputParserException(f"Invalid json output: {raw}", llm_output=raw)

        result = recover_from_parse_failure(exc, _Synthesis)
        assert result is None

    def test_falls_through_to_schema_echo_recovery_when_no_llm_output(self):
        """Existing schema-echo tests use plain Exception(str) with no
        llm_output attribute -- confirms the new bad-escapes path (tried
        first) doesn't break that older recovery path."""
        from research_swarm.agents._utils import recover_from_parse_failure

        exc = Exception(
            'Failed to parse _Synthesis from completion '
            '{"properties": {"claim": "Still works.", "confidence": 0.42}, '
            '"type": "object"}. Got: error'
        )
        result = recover_from_parse_failure(exc, _Synthesis)
        assert result is not None
        assert result.claim == "Still works."
        assert result.confidence == 0.42
