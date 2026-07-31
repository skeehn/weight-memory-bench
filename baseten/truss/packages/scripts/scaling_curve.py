"""Does weight memory scale into working, and if so where?

At 1B, online LoRA over a raw transcript recalls 33% of facts with a seed-to-seed range of
17-50%. That number on its own is close to useless: it says the mechanism fails at a size
nobody serious would deploy, and says nothing about whether it works at a size somebody
would. A single point is not a result about a mechanism.

So this walks the same protocol up a ladder of model sizes and reports the curve.

Everything except the model is held fixed: identical probes, identical seeds, identical
rank, learning rate, and epoch count, identical 131,072-token context, and the same
tokenizer family. A difference across rungs is therefore a difference in scale.

**All rungs must run on the same hardware in the same process.** Comparing a run on Apple
Silicon against a run on an H100 confounds scale with dtype behaviour and kernel-level
nondeterminism, and the effect being measured is smaller than that confound.

    WMB_READER_MODEL=... uv run python -m scripts.scaling_curve --sizes 1B 3B 8B
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import time
from pathlib import Path

from harness.tokens import READER_LADDER


def run_one(size: str, model_name: str, seeds: int, rank: int, lr: float, epochs: int) -> dict:
    """Measure one rung. Imports are local so each rung builds its own reader."""
    from arms.weight_memory import WeightMemoryArm
    from harness.reader import Reader
    from scripts.memory_gate import EPISODES, PROBES
    from scripts.memory_sweep import fact_nll

    started = time.time()
    reader = Reader(model=model_name)
    base = reader.model

    valid, excluded = [], []
    for question, expected in PROBES:
        unknown = expected.lower() not in reader.generate("", question).text.lower()
        answerable = (
            expected.lower() in reader.generate("\n".join(EPISODES), question).text.lower()
        )
        if unknown and answerable:
            valid.append((question, expected))
        else:
            excluded.append(
                {
                    "question": question,
                    "reason": "leaked_from_pretraining" if not unknown else "unanswerable",
                }
            )

    n = len(valid)
    print(f"  valid probes: {n}/{len(PROBES)}")
    for item in excluded:
        print(f"    excluded ({item['reason']}): {item['question']}")
    if not n:
        return {
            "size": size,
            "model": model_name,
            "valid_probes": 0,
            "note": "no valid probes; larger models may already know the invented facts",
            "excluded": excluded,
        }

    baseline = sum(fact_nll(reader, base, q, a) for q, a in valid) / n
    rates, nlls = [], []
    per_probe = {q: 0 for q, _ in valid}

    for seed in range(seeds):
        arm = WeightMemoryArm(
            reader, rank=rank, learning_rate=lr, epochs=epochs, seed=seed
        )
        model = arm._ensure_adapter()

        fresh = sum(fact_nll(reader, model, q, a) for q, a in valid) / n
        if abs(fresh - baseline) > 1e-2:
            raise AssertionError(
                f"{size} seed {seed}: adapter not fresh ({fresh:.4f} vs {baseline:.4f})"
            )

        arm.ingest(EPISODES)
        after = sum(fact_nll(reader, model, q, a) for q, a in valid) / n
        hits = 0
        for question, expected in valid:
            if expected.lower() in arm.answer(question).text.lower():
                hits += 1
                per_probe[question] += 1
        rates.append(hits / n)
        nlls.append(after)
        print(f"    seed {seed}: recall {hits}/{n} ({hits / n:.2f})  nll {after:.3f}")

        model.unload()
        arm._peft_model = None

    return {
        "size": size,
        "model": model_name,
        "valid_probes": n,
        "excluded": excluded,
        "seeds": seeds,
        "mean_recall": stats.mean(rates),
        "min_recall": min(rates),
        "max_recall": max(rates),
        "sd_recall": stats.stdev(rates) if len(rates) > 1 else 0.0,
        "baseline_nll": baseline,
        "mean_nll_after": stats.mean(nlls),
        "per_probe_hits": per_probe,
        "elapsed_s": round(time.time() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", default=["1B", "3B", "8B"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out", default="runs/scaling_curve.json")
    args = parser.parse_args()

    print(f"seeds={args.seeds} rank={args.rank} lr={args.lr:.0e} epochs={args.epochs}")
    results = []
    for size in args.sizes:
        model_name = READER_LADDER.get(size, size)
        print(f"\n=== {size}  {model_name} ===")
        try:
            results.append(
                run_one(size, model_name, args.seeds, args.rank, args.lr, args.epochs)
            )
        except Exception as exc:
            # One rung failing must not discard the rungs already measured. An OOM at 8B
            # should still leave 1B and 3B on disk.
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            results.append({"size": size, "model": model_name, "error": str(exc)})

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n")

    print("\n" + "=" * 62)
    print(f"{'size':>6} {'probes':>7} {'mean':>7} {'range':>12} {'sd':>7} {'nll':>8}")
    print("-" * 62)
    for r in results:
        if "error" in r or not r.get("valid_probes"):
            print(f"{r['size']:>6} {'--':>7} {'--':>7} {'--':>12} {'--':>7} {'--':>8}")
            continue
        rng = f"{r['min_recall']:.2f}-{r['max_recall']:.2f}"
        print(
            f"{r['size']:>6} {r['valid_probes']:>7} {r['mean_recall']:>7.3f} "
            f"{rng:>12} {r['sd_recall']:>7.3f} {r['mean_nll_after']:>8.3f}"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
