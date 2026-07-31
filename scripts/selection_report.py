"""Measure what each arm selects, before any GPU is involved.

Two numbers per arm, both free to compute:

**Context tokens** -- what the arm costs per question, counted with the real tokenizer.

**Evidence recall** -- whether the evidence the answer depends on is in the selected
context at all. LongMemEval tags the specific turns containing the answer, so this is
directly checkable.

Evidence recall is a ceiling, not a score. An arm that never retrieves the evidence cannot
answer correctly except by luck or prior knowledge, so its accuracy is capped here no
matter how good the reader is. Learning that an arm's ceiling is 40% costs nothing now and
would cost real money to discover after a full GPU run.

Abstention probes are excluded from the recall figure, because they have no evidence to
retrieve -- including them would make every arm look worse in proportion to how many
unanswerable questions the split happens to contain.

    uv run python scripts/selection_report.py --split dev --limit 50
"""

from __future__ import annotations

import argparse
import statistics as stats
import time

from arms.full_context import FullContextArm
from arms.grep import GrepArm
from arms.rag import RagArm
from data import longmemeval as lme


def evidence_found(selection_text: str, evidence: list[str]) -> bool:
    """True when any tagged evidence turn survived into the context.

    Matched on the turn's own content rather than the answer string: an arm should be
    credited for retrieving the right *turn*, and checking for the answer text directly
    would also credit a context that happens to contain the answer by coincidence.
    """
    return any(turn and turn in selection_text for turn in evidence)


def build_arms(budget: int, dense: bool):
    embedder = None
    if dense:
        from arms.rag import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
    return [
        FullContextArm(),
        GrepArm(budget=budget),
        RagArm(budget=budget, embedder=embedder),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--dense", action="store_true", help="enable the dense RAG lane")
    parser.add_argument("--skip-full", action="store_true", help="skip arm A (it is slow)")
    args = parser.parse_args()

    instances = lme.split(lme.load(), args.split)
    if args.limit:
        instances = instances[: args.limit]

    arms = build_arms(args.budget, args.dense)
    if args.skip_full:
        arms = [a for a in arms if a.name != "full_context"]

    print(f"split={args.split} n={len(instances)} budget={args.budget} dense={args.dense}")
    print(f"corpus_hash={lme.corpus_hash(instances)}\n")

    for arm in arms:
        token_counts: list[int] = []
        recalled = 0
        answerable = 0
        started = time.time()

        for inst in instances:
            arm.prepare(inst)
            selection = arm.select(inst.question)
            token_counts.append(selection.tokens)
            if not inst.is_abstention:
                answerable += 1
                if evidence_found(selection.text, inst.evidence_turns()):
                    recalled += 1

        elapsed = time.time() - started
        recall = (recalled / answerable) if answerable else None
        median = int(stats.median(token_counts)) if token_counts else 0
        print(
            f"{arm.name:14} tokens: median={median:>7,} "
            f"min={min(token_counts):>7,} max={max(token_counts):>7,}  "
            f"evidence_recall={recall:.3f} ({recalled}/{answerable})  "
            f"{elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
