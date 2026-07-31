"""Oracle arm: perfect retrieval, to isolate what the reader alone can do.

**This is a diagnostic, not a competitor.** It reads the `has_answer` labels from the
dataset, so it cannot be deployed and must never be presented alongside the other arms as
though it were one. It exists to answer a single question the other four cannot:

    When an arm gets the answer wrong, is that because retrieval missed the evidence, or
    because the reader could not use it?

Full context already hints at the answer -- it has evidence recall 1.000 and 16% accuracy --
but it pays for that with 105,708 tokens of distractors, so its failures are confounded with
long-context degradation. The oracle hands the reader the two or three tagged turns and
nothing else: a few hundred tokens, no distractors, the evidence guaranteed present and
maximally salient.

Whatever accuracy this reaches is the reader's ceiling on this benchmark. Every other arm is
bounded by it, and no retrieval improvement can cross it.
"""

from __future__ import annotations

from harness.tokens import shared

from .base import Selection, assert_fits


class OracleArm:
    name = "oracle"

    def __init__(self) -> None:
        self._evidence: tuple[str, ...] = ()

    def prepare(self, instance) -> None:
        # Label access. This is the line that makes the arm a diagnostic rather than a
        # system: no deployable retriever knows which turns were tagged.
        self._evidence = tuple(instance.evidence_turns())

    def select(self, question: str) -> Selection:
        text = "\n\n".join(self._evidence)
        selection = Selection(
            chunks=self._evidence,
            text=text,
            tokens=shared().count(text),
            meta={
                "arm": self.name,
                "evidence_turns": len(self._evidence),
                "skipped_chunks": 0,
                "uses_labels": True,
            },
        )
        assert_fits(selection)
        return selection
