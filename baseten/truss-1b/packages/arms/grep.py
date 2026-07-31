"""Arm B: keyword search over the transcript.

The complexity floor. If weight memory, or dense retrieval for that matter, cannot beat
literal string matching, the complexity is not earned. It is deliberately dumb: no
embeddings, no model, no index beyond a term set per turn.

It is not a strawman. Keyword matching is genuinely strong on questions that quote their
own answer's vocabulary, which is a large share of any memory benchmark, and it costs
nothing to run.

Ranking is by count of *distinct* query terms matched, not total matches, so a turn that
repeats one word many times does not outrank a turn that covers the whole question. Ties
break toward recency, which is the correct prior for a conversation log.
"""

from __future__ import annotations

import re

from .base import DEFAULT_RETRIEVAL_BUDGET, Selection, pack

# Deliberately small. A large stopword list starts encoding assumptions about the corpus,
# and this arm's value is being the least clever thing in the benchmark.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have he her
    him his how i if in is it its me my of on or our she should so than that the their
    them then there these they this to was we were what when where which who whom why
    will with would you your
    """.split()
)

TOKEN_RE = re.compile(r"[a-z0-9']+")


def terms(text: str) -> set[str]:
    """Content words, lowercased. Single characters are dropped as noise."""
    return {
        t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS
    }


class GrepArm:
    name = "grep"

    def __init__(self, budget: int = DEFAULT_RETRIEVAL_BUDGET) -> None:
        self.budget = budget
        self._turns: list[tuple[str, set[str]]] = []

    def prepare(self, instance) -> None:
        # Turn granularity, because a session-level match would drag in ~10 turns of
        # unrelated text and spend the budget on noise.
        self._turns = [(text, terms(text)) for _sid, text in instance.turn_chunks()]

    def select(self, question: str) -> Selection:
        query = terms(question)
        if not query:
            return pack([], self.budget, meta={"arm": self.name, "query_terms": 0})

        scored = []
        for position, (text, turn_terms) in enumerate(self._turns):
            overlap = len(query & turn_terms)
            if overlap:
                # Negative position so that, at equal overlap, later turns sort first.
                scored.append((overlap, position, text))
        scored.sort(key=lambda s: (-s[0], -s[1]))

        return pack(
            [text for _, _, text in scored],
            self.budget,
            meta={
                "arm": self.name,
                "query_terms": len(query),
                "matching_turns": len(scored),
                "best_overlap": scored[0][0] if scored else 0,
            },
        )
