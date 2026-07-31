"""Is the best configuration reliable, or was it a lucky initialization?

The joint sweep reported full recall at rank=16 / lr=2e-3 / 10 epochs. The same
configuration had reported 1/2 twice before that. Nothing changed between those runs except
the random initialization of LoRA's A matrix, which was unseeded.

So the honest question is not "does this configuration work" but "how often does it work".
This runs the same configuration across N seeds and reports the distribution.

Any single unseeded run is one draw. Reporting that draw as a result is how a benchmark
publishes a number it cannot reproduce -- and this repo would have done exactly that if the
sweep had happened to land on a good seed first.

    uv run python -m scripts.repeatability --seeds 5
"""

from __future__ import annotations

import argparse
import statistics as stats

from arms.weight_memory import WeightMemoryArm
from harness.reader import Reader
from scripts.memory_gate import EPISODES, PROBES
from scripts.memory_sweep import fact_nll


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    reader = Reader(device=args.device)
    base = reader.model
    print(f"device={reader.device} rank={args.rank} lr={args.lr:.0e} epochs={args.epochs}")

    valid = []
    for question, expected in PROBES:
        unknown = expected.lower() not in reader.generate("", question).text.lower()
        answerable = (
            expected.lower() in reader.generate("\n".join(EPISODES), question).text.lower()
        )
        if unknown and answerable:
            valid.append((question, expected))
        else:
            reason = "leaked from pretraining" if not unknown else "unanswerable in context"
            print(f"  excluded {question!r}: {reason}")
    n = len(valid)
    print(f"\nvalid probes: {n}/{len(PROBES)}\n")
    if not n:
        print("no valid probes; cannot measure")
        return

    baseline = sum(fact_nll(reader, base, q, a) for q, a in valid) / n

    print(f"{'seed':>6} {'fact_nll':>9} {'delta':>8} {'recall':>9} {'rate':>7}")
    print("-" * 44)

    rates: list[float] = []
    per_probe_hits = {q: 0 for q, _ in valid}

    for seed in range(args.seeds):
        arm = WeightMemoryArm(
            reader,
            rank=args.rank,
            learning_rate=args.lr,
            epochs=args.epochs,
            seed=seed,
        )
        model = arm._ensure_adapter()

        # Same guard `capacity_sweep` has, and it was missing here. Each seed re-wraps the
        # base model, which is the exact operation that silently stacked adapters earlier.
        # A fresh LoRA is a no-op, so the wrapped model must score the pristine baseline
        # before any training. Without this, seed N could inherit seed N-1.
        fresh = sum(fact_nll(reader, model, q, a) for q, a in valid) / n
        if abs(fresh - baseline) > 1e-2:
            raise AssertionError(
                f"adapter is not fresh at seed={seed}: {fresh:.4f} vs baseline {baseline:.4f}; "
                "a previous seed leaked through unload()"
            )

        arm.ingest(EPISODES)

        after = sum(fact_nll(reader, model, q, a) for q, a in valid) / n
        hits = 0
        for question, expected in valid:
            if expected.lower() in arm.answer(question).text.lower():
                hits += 1
                per_probe_hits[question] += 1
        rates.append(hits / n)
        print(
            f"{seed:>6} {after:>9.3f} {after - baseline:>+8.3f} "
            f"{str(hits) + '/' + str(n):>9} {hits / n:>7.2f}"
        )

        model.unload()
        arm._peft_model = None

    print("-" * 44)
    mean = stats.mean(rates)
    spread = f"{min(rates):.2f}-{max(rates):.2f}"
    sd = stats.stdev(rates) if len(rates) > 1 else 0.0
    print(f"mean recall {mean:.3f}  range {spread}  sd {sd:.3f}  over {args.seeds} seeds")

    print("\nper probe, across seeds:")
    for question, _ in valid:
        got = per_probe_hits[question]
        print(f"  {got}/{args.seeds}  {question}")

    print()
    if mean >= 0.9:
        print("VERDICT: reliable. Proceed to the benchmark.")
    elif mean >= 0.4:
        print("VERDICT: partial and seed-dependent. Report the distribution, never one run.")
    else:
        print("VERDICT: does not work at this scale. That is the finding.")


if __name__ == "__main__":
    main()
