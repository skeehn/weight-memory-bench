"""LongMemEval-S loader: the token-cost axis.

500 instances, each a question over a haystack of ~50 chat sessions (~490 turns). The whole
haystack is roughly 115K tokens, which means it *fits inside a modern context window*. That
is a caveat and an asset at the same time:

- Caveat: on this corpus a bigger window substitutes for memory, so it measures
  context-management efficiency as much as memory. Never report it as the retention primary.
- Asset: because full-context is actually achievable here, it is a legitimate ceiling, and
  the comparison "115K tokens per query versus ~0" is legible rather than hypothetical.

**30 of the 500 questions are abstention probes** (`question_id` ending in `_abs`): the
answer is not in the haystack and the correct behaviour is to decline. They are what make
`answered_rate` a real measurement rather than a formality, and they are the only reason the
fabrication gate can fire. They are never filtered out.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_PATH = Path(
    os.environ.get(
        "WMB_LONGMEMEVAL_PATH",
        str(Path.home() / "LongMemEval" / "data" / "longmemeval_s_cleaned.json"),
    )
)

# Deterministic split by question_id hash, so the assignment never depends on file order
# and never shifts when the corpus is re-downloaded. dev is for building and tuning; test
# is reported once.
DEV_PERCENTILE = 20


@dataclass(frozen=True)
class Turn:
    role: str
    content: str
    has_answer: bool = False

    def text(self) -> str:
        return f"{self.role}: {self.content}"


@dataclass(frozen=True)
class Session:
    session_id: str
    date: str
    turns: tuple[Turn, ...]

    def text(self) -> str:
        header = f"[session {self.session_id} on {self.date}]"
        return "\n".join([header, *(t.text() for t in self.turns)])


@dataclass(frozen=True)
class Instance:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    sessions: tuple[Session, ...]
    answer_session_ids: tuple[str, ...]

    @property
    def is_abstention(self) -> bool:
        """No answer exists in the haystack; declining is correct.

        Answering these at all is a fabrication, which is why they are kept in the
        denominator rather than filtered out of the run.
        """
        return self.question_id.endswith("_abs")

    def transcript_text(self) -> str:
        """The entire haystack as one string. What arm A pays for."""
        return "\n\n".join(s.text() for s in self.sessions)

    def session_chunks(self) -> list[str]:
        """Session-granularity chunks. The retrieval unit for arms B and C."""
        return [s.text() for s in self.sessions]

    def turn_chunks(self) -> list[tuple[str, str]]:
        """(session_id, turn text) pairs, for finer-grained retrieval and for arm D's
        online update stream."""
        return [(s.session_id, t.text()) for s in self.sessions for t in s.turns]

    def evidence_turns(self) -> list[str]:
        """Turns tagged `has_answer` in the source data.

        Used only for diagnostics -- an oracle ceiling and a retrieval-recall check that
        separates 'never retrieved it' from 'retrieved it and still got it wrong'. Never
        visible to an arm at answer time.
        """
        return [t.text() for s in self.sessions for t in s.turns if t.has_answer]


def _parse(raw: dict) -> Instance:
    dates = raw.get("haystack_dates") or []
    ids = raw.get("haystack_session_ids") or []
    sessions = []
    for i, turns in enumerate(raw.get("haystack_sessions") or []):
        sessions.append(
            Session(
                session_id=ids[i] if i < len(ids) else f"session_{i}",
                date=dates[i] if i < len(dates) else "",
                turns=tuple(
                    Turn(
                        role=t.get("role", ""),
                        content=t.get("content") or "",
                        has_answer=bool(t.get("has_answer")),
                    )
                    for t in turns
                ),
            )
        )
    return Instance(
        question_id=raw["question_id"],
        question_type=raw.get("question_type", ""),
        question=raw.get("question", ""),
        answer=str(raw.get("answer", "")),
        question_date=raw.get("question_date", ""),
        sessions=tuple(sessions),
        answer_session_ids=tuple(raw.get("answer_session_ids") or []),
    )


@lru_cache(maxsize=2)
def load(path: Path | str = DEFAULT_PATH) -> tuple[Instance, ...]:
    """Load and parse the corpus. Cached; the file is ~278MB and parsing is not cheap."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"LongMemEval not found at {path}. Set WMB_LONGMEMEVAL_PATH to override."
        )
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return tuple(_parse(r) for r in raw)


def _bucket_fine(question_id: str) -> int:
    """Stable pseudo-random rank key for a question. Depends only on the id."""
    return int(hashlib.sha256(question_id.encode()).hexdigest()[:16], 16)


def split(instances, name: str):
    """Deterministic dev/test split, keyed on question_id rather than file order.

    `dev` is for building and tuning. `test` is reported once. `all` is for corpus-level
    statistics only and must never be used to report accuracy.

    **Stratified on abstention.** Only 30 of the 500 questions are abstention probes, so an
    unstratified hash split hands dev whatever it happens to land on -- measured, that was 3
    of 102, which is too few to notice an arm that fabricates. Bucketing each group
    separately gives both splits their proportional share. It is the same deterministic
    hash, applied within strata.
    """
    if name == "all":
        return list(instances)
    if name not in ("dev", "test"):
        raise ValueError(f"unknown split {name!r}; expected dev, test, or all")

    want_dev = name == "dev"
    out = []
    for stratum in (False, True):  # non-abstention, then abstention
        group = [i for i in instances if i.is_abstention is stratum]
        # Rank within the stratum by hash so the cut point is proportional to stratum size
        # rather than to how the hashes happened to fall.
        ranked = sorted(group, key=lambda i: (_bucket_fine(i.question_id), i.question_id))
        cut = round(len(ranked) * DEV_PERCENTILE / 100)
        out.extend(ranked[:cut] if want_dev else ranked[cut:])
    return sorted(out, key=lambda i: i.question_id)


def corpus_hash(instances) -> str:
    """Provenance hash over question ids and answers.

    Keyed on content, not the file path or mtime, so a re-download of the same data keeps
    old ledger rows comparable and a changed corpus makes them visibly incomparable.
    """
    h = hashlib.sha256()
    for inst in sorted(instances, key=lambda i: i.question_id):
        h.update(inst.question_id.encode())
        h.update(b"\x00")
        h.update(inst.answer.encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def stats(instances) -> dict:
    n = len(instances)
    by_type: dict[str, int] = {}
    for i in instances:
        by_type[i.question_type] = by_type.get(i.question_type, 0) + 1
    return {
        "n": n,
        "abstention": sum(1 for i in instances if i.is_abstention),
        "sessions_per_instance_median": _median([len(i.sessions) for i in instances]),
        "turns_per_instance_median": _median(
            [sum(len(s.turns) for s in i.sessions) for i in instances]
        ),
        "by_question_type": dict(sorted(by_type.items())),
    }


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2
