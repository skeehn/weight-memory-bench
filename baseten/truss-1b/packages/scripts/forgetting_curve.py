"""What does writing memory into weights destroy?

Every result so far measures whether weight memory *works*. This measures what it *costs*.
It is the half of the story nobody publishes: papers and product claims report retention,
almost never the capability lost on the way to it.

The axis is not incidental. Engram's co-founder is first author of *LoRA Learns Less and
Forgets Less*, which measured forgetting after a single fine-tune. The online regime --
where you keep updating forever, which is what a memory system actually does -- is the case
that paper did not cover and the case a memory product lives or dies on.

Two things are tracked against the number of online updates applied:

**Retention** -- recall of the injected facts. Expected to rise, then possibly fall as
later updates overwrite earlier ones.

**Capability** -- how the model does on things it already knew and was never taught here.
Measured two ways, because they fail differently:

- *Held-out perplexity* on text unrelated to the episodes. Continuous and sensitive, so it
  moves before anything visible breaks.
- *General probes* the base model answers correctly before any update. Discrete and
  legible: a model that stops knowing the capital of France has degraded in a way a
  perplexity delta does not convey.

A flat capability line means the LoRA is not really learning. A cliff means it is. Both are
results, and the interesting quantity is where the two curves cross.

    uv run python -m scripts.forgetting_curve --checkpoints 0 10 25 50 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Held-out text: ordinary prose sharing no vocabulary with the episodes. Perplexity here
# should be untouched by an update that only writes facts about a ferret and a commute.
HELDOUT_TEXT = (
    "The harbour master kept records of every vessel that entered the estuary, noting "
    "tonnage, cargo, and the hour of arrival. In winter the fog made the outer channel "
    "impassable for days at a time, and the ledgers show long gaps where nothing moved."
)

# Things the base model already knows. Never taught, never mentioned in the episodes, so any
# change here is damage rather than learning. Kept simple deliberately: a 1B model answers
# these reliably, which is what makes a failure meaningful rather than noisy.
CAPABILITY_PROBES = [
    ("What is the capital of France?", "paris"),
    ("What color is the sky on a clear day?", "blue"),
    ("How many days are in a week?", "seven"),
    ("What is the largest ocean on Earth?", "pacific"),
    ("What language is primarily spoken in Brazil?", "portuguese"),
    ("What is 2 plus 2?", "4"),
]


def perplexity(reader, model, text: str) -> float:
    import torch

    tokenizer = reader.tokenizer
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        reader.device
    )
    with torch.no_grad():
        loss = model(input_ids=ids, labels=ids).loss
    return float(torch.exp(loss))


def capability(reader, arm, probes) -> tuple[float, list[str]]:
    """Fraction of general probes still answered correctly, and which ones broke."""
    lost = []
    hits = 0
    for question, expected in probes:
        text = arm.answer(question, max_new_tokens=16).text.lower()
        if expected.lower() in text:
            hits += 1
        else:
            lost.append(question)
    return hits / len(probes), lost


def run_curve(
    checkpoints=(0, 10, 25, 50, 100),
    rank: int = 16,
    lr: float = 2e-3,
    seed: int = 0,
    device: str = "auto",
    size: str = "1B",
) -> dict:
    """The measurement itself, callable from a CLI or from a deployed model."""

    class Args:
        pass

    args = Args()
    args.checkpoints = list(checkpoints)
    args.rank, args.lr, args.seed, args.device, args.size = rank, lr, seed, device, size
    args.out = None
    return _run(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 10, 25, 50, 100])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--size", default="1B")
    parser.add_argument("--out", default="runs/forgetting_curve.json")
    args = parser.parse_args()
    _run(args)


def _run(args) -> dict:
    from arms.weight_memory import WeightMemoryArm
    from harness.reader import Reader
    from harness.tokens import READER_LADDER
    from scripts.memory_gate import EPISODES, PROBES

    reader = Reader(model=READER_LADDER.get(getattr(args, "size", "1B"), "1B"), device=args.device)
    base = reader.model

    # Only probes the base model can actually answer are worth tracking. A probe it never
    # knew cannot be forgotten, and counting it would understate retained capability.
    usable = []
    for question, expected in CAPABILITY_PROBES:
        if expected.lower() in reader.generate("", question, max_new_tokens=16).text.lower():
            usable.append((question, expected))
        else:
            print(f"  dropped capability probe (base model fails it): {question}")
    print(f"capability probes: {len(usable)}/{len(CAPABILITY_PROBES)}\n")

    arm = WeightMemoryArm(reader, rank=args.rank, learning_rate=args.lr, seed=args.seed)
    model = arm._ensure_adapter()
    arm.reset()

    base_ppl = perplexity(reader, model, HELDOUT_TEXT)
    print(f"{'updates':>8} {'retention':>10} {'capability':>11} {'heldout_ppl':>12} {'ppl_ratio':>10}")
    print("-" * 56)

    rows = []
    applied = 0
    for target in sorted(args.checkpoints):
        # Epochs are applied incrementally so this is one continuous online stream, not a
        # series of independent fine-tunes. That distinction is the whole point: the
        # published forgetting result covers a single fine-tune, and a memory system does
        # not do that.
        while applied < target:
            arm.epochs = 1
            arm.ingest(EPISODES)
            applied += 1

        retained = sum(
            1
            for question, expected in PROBES
            if expected.lower() in arm.answer(question).text.lower()
        ) / len(PROBES)
        cap, lost = capability(reader, arm, usable)
        ppl = perplexity(reader, model, HELDOUT_TEXT)

        print(
            f"{applied:>8} {retained:>10.3f} {cap:>11.3f} {ppl:>12.2f} {ppl / base_ppl:>10.2f}x"
        )
        if lost:
            print(f"         lost: {'; '.join(q[:40] for q in lost)}")

        rows.append(
            {
                "updates": applied,
                "retention": retained,
                "capability": cap,
                "heldout_ppl": ppl,
                "ppl_ratio": ppl / base_ppl,
                "capability_lost": lost,
            }
        )

    payload = {
        "base_heldout_ppl": base_ppl,
        "rank": args.rank,
        "lr": args.lr,
        "seed": args.seed,
        "size": getattr(args, "size", "1B"),
        "capability_probes": [q for q, _ in usable],
        "rows": rows,
    }
    if getattr(args, "out", None):
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")

    if rows:
        peak = max(rows, key=lambda r: r["retention"])
        final = rows[-1]
        print(
            f"\npeak retention {peak['retention']:.3f} at {peak['updates']} updates; "
            f"capability there {peak['capability']:.3f}"
        )
        print(
            f"at {final['updates']} updates: retention {final['retention']:.3f}, "
            f"capability {final['capability']:.3f}, held-out ppl {final['ppl_ratio']:.2f}x baseline"
        )

    return payload


if __name__ == "__main__":
    main()
