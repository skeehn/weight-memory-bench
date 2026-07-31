"""Is the binding constraint capacity, or the mechanism itself?

The learning-rate sweep found a sharp optimum around lr=2e-3 / 10 epochs and no setting
that recalled both facts. Two explanations survive that result and they call for opposite
conclusions:

1. **Capacity.** A rank-16 adapter on four attention projections is too small to hold the
   fact. Then more rank, or more targeted modules, fixes it.
2. **Mechanism.** Plain causal-LM loss over a transcript does not put facts anywhere they
   can be retrieved as answers, at any size. Then rank changes nothing and arm D has a
   real ceiling worth reporting.

This sweeps rank and target modules at the best-known lr/epochs to separate them.

**Changing rank requires a new adapter**, which means re-wrapping the base model -- exactly
the operation that silently stacked adapters in the first sweep. So each config explicitly
unloads the previous adapter and then verifies the base model's fact NLL has returned to
its pristine baseline before training again. If it has not, the run aborts rather than
reporting contaminated numbers.

    uv run python -m scripts.capacity_sweep
"""

from __future__ import annotations

import argparse

from arms.weight_memory import WeightMemoryArm
from harness.reader import Reader
from scripts.memory_gate import EPISODES, PROBES
from scripts.memory_sweep import fact_nll

ATTENTION_ONLY = ("q_proj", "k_proj", "v_proj", "o_proj")
ALL_LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lrs", type=float, nargs="+", default=[2e-3])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument(
        "--targets", nargs="+", default=["attn", "all"], choices=["attn", "all"]
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    reader = Reader(device=args.device)
    base = reader.model
    print(f"device={reader.device} epochs={args.epochs}\n")

    valid = []
    for question, expected in PROBES:
        unknown = expected.lower() not in reader.generate("", question).text.lower()
        answerable = (
            expected.lower() in reader.generate("\n".join(EPISODES), question).text.lower()
        )
        if unknown and answerable:
            valid.append((question, expected))
    print(f"valid probes: {len(valid)}/{len(PROBES)}\n")
    if not valid:
        print("no valid probes; cannot sweep")
        return

    baseline = sum(fact_nll(reader, base, q, a) for q, a in valid) / len(valid)
    print(
        f"{'rank':>6} {'targets':>8} {'lr':>8} {'params':>12} "
        f"{'fact_nll':>9} {'delta':>8} {'recall':>8}"
    )
    print("-" * 68)
    print(
        f"{'--':>6} {'--':>8} {'--':>8} {'0':>12} {baseline:>9.3f} "
        f"{'--':>8} {'0/' + str(len(valid)):>8}"
    )

    lookup = {"attn": ATTENTION_ONLY, "all": ALL_LINEAR}
    combos = [
        (lookup[label], label, rank, lr)
        for label in args.targets
        for rank in args.ranks
        for lr in args.lrs
    ]

    for targets, label, rank, lr in combos:
        if True:
            arm = WeightMemoryArm(
                reader,
                rank=rank,
                learning_rate=lr,
                epochs=args.epochs,
                target_modules=targets,
            )
            model = arm._ensure_adapter()

            # The contamination guard. A fresh adapter is a no-op, so the wrapped model
            # must score exactly the pristine baseline before any training happens.
            fresh = sum(fact_nll(reader, model, q, a) for q, a in valid) / len(valid)
            if abs(fresh - baseline) > 1e-2:
                raise AssertionError(
                    f"adapter is not fresh at rank={rank} {label}: "
                    f"{fresh:.4f} vs baseline {baseline:.4f}"
                )

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            arm.ingest(EPISODES)
            after = sum(fact_nll(reader, model, q, a) for q, a in valid) / len(valid)
            hits = sum(1 for q, a in valid if a.lower() in arm.answer(q).text.lower())
            print(
                f"{rank:>6} {label:>8} {lr:>8.0e} {trainable:>12,} {after:>9.3f} "
                f"{after - baseline:>+8.3f} {str(hits) + '/' + str(len(valid)):>8}"
            )

            # Remove this adapter before the next config wraps the base again.
            model.unload()
            arm._peft_model = None

            if hits == len(valid):
                print(f"\nFULL RECALL at rank={rank} targets={label} lr={lr:.0e}.")
                return

    print("\nNo configuration achieved full recall.")
    if len(args.lrs) > 1:
        print(
            "Learning rate was varied per rank, so rank and effective step size are\n"
            "no longer confounded. If the best result at every rank is still a miss,\n"
            "the constraint is the mechanism rather than capacity."
        )
    else:
        print(
            "NOTE: learning rate was held fixed, so rank is confounded with effective\n"
            "step size. Re-run with --lrs before concluding anything about capacity."
        )


if __name__ == "__main__":
    main()
