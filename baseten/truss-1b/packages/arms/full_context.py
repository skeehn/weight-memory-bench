"""Arm A: put the entire haystack in the window.

The ceiling arm. It has access to everything, so whatever accuracy it reaches is roughly
the best the reader can do on this corpus given perfect retrieval. Every other arm is
trying to approach it while paying less.

It is also the most expensive thing in the benchmark by two orders of magnitude: ~106K
tokens per question, prefilled fresh every time. That cost is the entire reason the other
three arms exist, and it is what the headline ratio is measured against.

Worth stating plainly, because it is the honest caveat on this whole corpus: arm A being
*possible* means LongMemEval fits in a modern context window, so on this data a bigger
window substitutes for memory. That is why BEAM exists in this repo.
"""

from __future__ import annotations

from harness.tokens import shared

from .base import Selection, assert_fits


class FullContextArm:
    name = "full_context"

    def __init__(self) -> None:
        self._chunks: tuple[str, ...] = ()

    def prepare(self, instance) -> None:
        # Session-granularity chunks, in chronological order. Order matters: packing is
        # greedy, so if the budget ever binds it is the *latest* sessions that get dropped,
        # which would quietly turn this into a recency arm. assert_fits exists to make sure
        # that never happens silently.
        self._chunks = tuple(instance.session_chunks())

    def select(self, question: str) -> Selection:
        # Assembled directly rather than through `pack`. This arm has no budget to enforce,
        # and pack would tokenize every chunk to check one, then tokenize the joined result
        # again -- twice over a ~106K-token document, on every question.
        text = "\n\n".join(self._chunks)
        selection = Selection(
            chunks=tuple(self._chunks),
            text=text,
            tokens=shared().count(text),
            meta={
                "arm": self.name,
                "available_chunks": len(self._chunks),
                "skipped_chunks": 0,
            },
        )
        assert_fits(selection)
        return selection
