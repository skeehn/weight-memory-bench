"""The shared arm interface.

Every arm answers the same question against the same haystack with the same reader. The
only thing that differs is **which text ends up in the context window**. So that is the only
thing an arm implements: `prepare` builds whatever memory it needs from the haystack, and
`select` decides what to put in front of the reader.

Splitting context selection from reader inference is deliberate and load-bearing for the
budget. Selection is pure text manipulation, so all of arms A, B, and C can be built,
tested, and measured for token cost on a laptop for free. The GPU is only needed to turn a
selected context into an answer, which is the last step and the only one that costs money.

Packing is greedy and budget-bounded, and the budget is counted with the real tokenizer at
pack time rather than estimated. An arm that overruns the window does not get silently
truncated by the reader; it fails the pack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from harness.tokens import READER_CONTEXT_TOKENS, shared

# Room held back from the window for the prompt scaffolding, the question itself, and the
# generated answer. Arm A packs ~106K into a 131K window, so this is not tight, but it is
# the difference between "fits" and "fits with the question attached".
RESERVED_TOKENS = 1_024

# Default budget for retrieval arms. Not a technical limit -- it is the operating point
# those arms are supposed to occupy, and the number the token-cost comparison turns on.
# Overridable per arm so the budget itself can be swept.
DEFAULT_RETRIEVAL_BUDGET = 4_096


class ContextTooLarge(ValueError):
    """Raised when a packed context cannot fit the reader's window.

    Raised rather than truncated. A silently truncated context produces an answer that
    looks like a measurement of the arm but is actually a measurement of the truncation.
    """


@dataclass
class Selection:
    """What an arm decided to show the reader, and what it cost."""

    chunks: tuple[str, ...]
    text: str
    tokens: int
    meta: dict = field(default_factory=dict)

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)


class Arm(Protocol):
    name: str

    def prepare(self, instance) -> None:
        """Build whatever memory this arm uses, from one instance's haystack."""

    def select(self, question: str) -> Selection:
        """Choose the context for one question."""


def pack(
    chunks: Sequence[str],
    budget: int,
    *,
    separator: str = "\n\n",
    meta: dict | None = None,
) -> Selection:
    """Greedily pack chunks in the order given, until the budget is reached.

    Order is the caller's ranking and is never re-sorted here -- an arm's ranking is its
    whole contribution, and quietly reordering it would measure something else.

    A chunk that does not fit is skipped rather than ending the pack, because a single
    outsized chunk should not starve every lower-ranked one. The count of skipped chunks is
    reported in `meta` so that behaviour is visible rather than assumed.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")

    tk = shared()
    sep_tokens = tk.count(separator)

    kept: list[str] = []
    total = 0
    skipped = 0
    for chunk in chunks:
        cost = tk.count(chunk) + (sep_tokens if kept else 0)
        if total + cost > budget:
            skipped += 1
            continue
        kept.append(chunk)
        total += cost

    text = separator.join(kept)
    # Recount the assembled string rather than trusting the running sum. Token merges
    # across a join boundary make the sum an upper bound, not the answer, and the number
    # that gets published must be the one the reader actually pays.
    actual = tk.count(text)

    info = {"skipped_chunks": skipped, "budget": budget, "packed_sum_estimate": total}
    if meta:
        info.update(meta)
    return Selection(chunks=tuple(kept), text=text, tokens=actual, meta=info)


def assert_fits(selection: Selection, reserved: int = RESERVED_TOKENS) -> None:
    """Guard that a selection leaves room for the question and the answer."""
    limit = READER_CONTEXT_TOKENS - reserved
    if selection.tokens > limit:
        raise ContextTooLarge(
            f"context is {selection.tokens} tokens, over the {limit} available "
            f"({READER_CONTEXT_TOKENS} window minus {reserved} reserved)"
        )


def empty_selection(meta: dict | None = None) -> Selection:
    """A context of literally nothing. What arm D is supposed to produce."""
    return Selection(chunks=(), text="", tokens=0, meta=meta or {})
