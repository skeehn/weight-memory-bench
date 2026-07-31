"""The gate. Run this before spending a cent of GPU budget.

One question: can a fact be written into the weights and read back with an empty context
window? If not, arm D does not work, and measuring it on 400 instances would buy an
expensive way of finding that out.

Three checks, in order, because only the third is meaningful and the first two are what
make it meaningful:

1. **Before ingesting** -- ask with an empty context. The reader must NOT know. If it does,
   the fact was in its pretraining and the whole test measures nothing. This is why the
   facts below use invented proper nouns.
2. **Full context** -- hand it the episode containing the answer. The reader MUST know. If
   it does not, the question is beyond this model and a failure at step 3 would be
   unattributable: unable to memorize, or unable to answer at all?
3. **After ingesting, empty context** -- the actual gate.

A pass at 3 with a fail at 1 and a pass at 2 is the only combination that means anything.

    uv run python scripts/memory_gate.py --lr 1e-4 --epochs 4 --rank 16
"""

from __future__ import annotations

import argparse

from arms.weight_memory import WeightMemoryArm
from harness.reader import Reader

# Invented proper nouns, so a correct answer cannot come from pretraining. Deliberately
# the easiest possible case: short episodes, one salient fact each, stated plainly. If
# weight memory cannot do this, it certainly cannot do a 490-turn haystack.
EPISODES = [
    "user: I finally set up the new apartment in Kesterly Row.",
    "assistant: Congratulations on the move. How is the space?",
    "user: My ferret is named Pemberton and he has already knocked over two plants.",
    "assistant: Pemberton sounds like a handful. Ferrets are famously curious.",
    "user: I started a job at Vandersloot Analytics last Tuesday.",
    "assistant: How is the new role treating you so far?",
    "user: My manager is named Odalys and she is very supportive.",
    "assistant: Having a supportive manager makes a big difference.",
    "user: My sister Wrenna is flying in to visit next month.",
    "assistant: That will be nice. Is Wrenna staying with you?",
    "user: I drive a Trellick Vireo, which is a terrible car honestly.",
    "assistant: What is wrong with the Trellick Vireo?",
    "user: I am badly allergic to hazelnuts, so I have to read every label.",
    "assistant: That sounds exhausting. Hazelnuts hide in a lot of products.",
    "user: I joined a gym called Quarrow Fitness down the street.",
    "assistant: Convenient location makes it much easier to keep going.",
]

# Every answer is an invented proper noun, so a correct response cannot come from
# pretraining. Eight probes rather than three: at n=2 the difference between a working
# configuration and a lucky one is invisible.
#
# Answers are stored in their **exact surface form**, capitalization included, and
# lowercased only at comparison time. Storing them lowercase silently broke the NLL
# diagnostic, because the lowercase and spaced forms tokenize completely differently:
#
#     'pemberton'   -> [79, 9034, 37733]
#     ' Pemberton'  -> [69383, 37733]     <- what the model actually generates
#
# Scoring the lowercase form measured the likelihood of a string the model would never
# emit. Anything computing a probability must use `answer_surface()`, not the raw string.
PROBES = [
    ("What is the name of my ferret?", "Pemberton"),
    ("Where do I work?", "Vandersloot"),
    ("What is my manager's name?", "Odalys"),
    ("What is my sister's name?", "Wrenna"),
    ("What car do I drive?", "Trellick"),
    ("What am I allergic to?", "hazelnuts"),
    ("What gym did I join?", "Quarrow"),
    ("What street is my apartment on?", "Kesterly"),
]


def answer_surface(expected: str) -> str:
    """The exact string the model would emit, for likelihood scoring.

    A leading space, because generation continues a chat turn rather than starting a
    document, and that space changes the tokenization of the first word entirely.
    """
    return f" {expected}"


def check(generation, expected: str) -> bool:
    return expected.lower() in generation.text.lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    reader = Reader(device=args.device)
    print(f"device={reader.device} model={reader.model_name}")
    print(f"rank={args.rank} lr={args.lr} epochs={args.epochs}\n")

    arm = WeightMemoryArm(
        reader, rank=args.rank, learning_rate=args.lr, epochs=args.epochs
    )

    # -- 1. before ingest: must NOT know -------------------------------------------
    print("1. before ingest, empty context (must NOT know)")
    before = {}
    for question, expected in PROBES:
        gen = reader.generate("", question)
        before[question] = check(gen, expected)
        flag = "LEAK" if before[question] else "ok"
        print(f"   [{flag:4}] {question!r} -> {gen.text[:70]!r}")

    # -- 2. full context: MUST know ------------------------------------------------
    print("\n2. full context (MUST know, else the question is beyond this reader)")
    answerable = {}
    context = "\n".join(EPISODES)
    for question, expected in PROBES:
        gen = reader.generate(context, question)
        answerable[question] = check(gen, expected)
        flag = "ok" if answerable[question] else "FAIL"
        print(f"   [{flag:4}] {question!r} -> {gen.text[:70]!r}")

    # -- 3. the gate ---------------------------------------------------------------
    print(f"\n3. ingesting {len(EPISODES)} episodes into weights...")
    steps = arm.ingest(EPISODES)
    print(f"   {steps} optimizer steps applied")

    print("\n   after ingest, empty context (THE GATE)")
    recalled = {}
    for question, expected in PROBES:
        gen = arm.answer(question)
        recalled[question] = check(gen, expected)
        flag = "PASS" if recalled[question] else "miss"
        print(f"   [{flag:4}] {question!r} -> {gen.text[:70]!r}")

    # -- verdict -------------------------------------------------------------------
    valid = [q for q, _ in PROBES if not before[q] and answerable[q]]
    passed = [q for q in valid if recalled[q]]

    print("\n" + "=" * 70)
    print(f"valid probes (not leaked, answerable in context): {len(valid)}/{len(PROBES)}")
    if not valid:
        print("VERDICT: inconclusive. No probe was both unknown and answerable.")
        return
    print(f"recalled from weights with empty context: {len(passed)}/{len(valid)}")
    if len(passed) == len(valid):
        print("VERDICT: PASS. Weight memory works on the easy case. Proceed.")
    elif passed:
        print("VERDICT: PARTIAL. Some facts stick. Sweep lr/epochs before spending.")
    else:
        print("VERDICT: FAIL. Nothing was written into the weights. Do not spend yet.")


if __name__ == "__main__":
    main()
