"""Split a user's question into what to research and how to answer it.

A request like "Using only the supplied abstracts, classify the claim as exactly SUPPORT,
CONTRADICT, or NOT_ENOUGH_INFO, then explain the verdict: Cold exposure reduces BAT recruitment"
mixes two things: the *content* to research (the claim) and an *answer format* (pick a label).
Given the whole string, the planner turned the format into research sub-questions ("How do the
abstracts classify the claim?"), and the extractor then produced meta-facts about classification.

Here, deterministically:
  * ``content`` is what the planner / extractors / verifier see,
  * ``instruction`` is the format part, shown only to the writer,
  * ``labels`` are the allowed answers when the question enumerates them (``A, B, or C``), and
    ``label_roles`` maps them to support / contradict / insufficient when they read that way,
    which enables the claim-verdict step (``agents/verdict.py``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# "SUPPORT, CONTRADICT, or NOT_ENOUGH_INFO" / "YES or NO": two or more ALL-CAPS tokens joined by
# commas and a final "or"/"and".
_LABEL_TOKEN = r"[A-Z][A-Z0-9_]{1,}"
_LABEL_LIST_RE = re.compile(
    rf"\b({_LABEL_TOKEN}(?:\s*,\s*{_LABEL_TOKEN})*\s*,?\s*(?:or|and)\s+{_LABEL_TOKEN})\b"
)
# An answer-format instruction OPENS the topic ("Using only ..., classify ...: <claim>",
# "Answer YES or NO: <question>"). Words like "label" or "answer" elsewhere in a title
# ("Machine learning label noise: a survey") must not make its head an instruction.
_INSTRUCTION_OPENER_RE = re.compile(
    r"^\s*(using only|based only|classify|answer|explain|respond|reply|decide|determine whether|"
    r"rate|select|choose|label)\b", re.IGNORECASE,
)
MAX_INSTRUCTION_CHARS = 250

_SUPPORT = re.compile(r"^(SUPPORTS?|SUPPORTED|TRUE|YES|AGREES?|CORRECT|ENTAIL\w*)$")
_CONTRADICT = re.compile(r"^(CONTRADICTS?|CONTRADICTED|REFUTES?|REFUTED|FALSE|NO|DISAGREES?|"
                         r"INCORRECT)$")
_INSUFFICIENT = re.compile(r"(NOT_ENOUGH|NEI|INSUFFICIENT|UNKNOWN|UNVERIFIABLE|NOT_ENOUGH_INFO|"
                           r"CANNOT|UNDETERMINED|NEUTRAL)")


@dataclass(frozen=True)
class QuestionSpec:
    content: str
    instruction: str = ""
    labels: tuple[str, ...] = ()
    # label -> "support" | "contradict" | "insufficient", only when every label maps to one
    label_roles: dict[str, str] = field(default_factory=dict)

    @property
    def is_claim_check(self) -> bool:
        return bool(self.label_roles)

    def label_for(self, role: str) -> str | None:
        return next((lab for lab, r in self.label_roles.items() if r == role), None)


def _role(label: str) -> str | None:
    if _SUPPORT.match(label):
        return "support"
    if _CONTRADICT.match(label):
        return "contradict"
    if _INSUFFICIENT.search(label):
        return "insufficient"
    return None


def parse_labels(text: str) -> tuple[str, ...]:
    match = _LABEL_LIST_RE.search(text or "")
    if not match:
        return ()
    labels = tuple(dict.fromkeys(re.findall(_LABEL_TOKEN, match.group(1))))
    return labels if len(labels) >= 2 else ()


def parse_question(topic: str) -> QuestionSpec:
    """``topic`` -> content / instruction / labels (see the module docstring)."""
    topic = (topic or "").strip()
    content, instruction = topic, ""
    # "<instruction>: <content>" -- split at the first colon when the part before it reads like
    # an instruction and is short; otherwise the whole topic is the content.
    head, sep, tail = topic.partition(":")
    if (sep and tail.strip() and len(head) <= MAX_INSTRUCTION_CHARS
            and _INSTRUCTION_OPENER_RE.search(head)):
        content, instruction = tail.strip(), head.strip()
    # Labels come only from an explicit instruction: an ALL-CAPS "CPU or GPU" in an ordinary
    # topic is subject matter, not a set of allowed answers.
    labels = parse_labels(instruction)
    roles = {lab: _role(lab) for lab in labels}
    complete = labels and all(roles.values()) and set(roles.values()) >= {"support", "contradict"}
    return QuestionSpec(
        content=content, instruction=instruction, labels=labels,
        label_roles={k: v for k, v in roles.items() if v} if complete else {},
    )


def research_topic(query: Any) -> str:
    """What the planner / extractors / verifier should research: the question minus any
    answer-format instruction. Empty string for no query."""
    if query is None:
        return ""
    return parse_question(getattr(query, "topic", "") or "").content


def is_meta_sub_question(sub_question: str, spec: QuestionSpec) -> bool:
    """A planned sub-question about the answer FORMAT rather than the subject matter. Only
    questions that carry a format instruction have any; an ordinary topic keeps its plan whole
    (a sub-question about "lesion classification" is subject matter)."""
    if not spec.instruction:
        return False
    for lab in spec.labels:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(lab)}(?![A-Za-z0-9_])", sub_question):
            return True
    return bool(re.search(r"\bclassif(y|ied|ication)\b", sub_question, re.IGNORECASE))
