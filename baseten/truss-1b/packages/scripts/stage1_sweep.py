"""Find the operating point where weight memory learns AND the model survives.

Stage 1 established the two things that matter. Augmentation makes facts extractable:
0.750 retention with an empty context window, against a previous best of 0.083. And
retention and damage are **decoupled** — one seed produced 0.75 at 286x while another
produced 0.75 at 2.2x, identical learning with 130x different damage.

That leaves a bracketed search rather than an open question:

    lr 2e-3, 45 steps  ->  0.750 retention, 144x damage    learns, unstable
    lr 2e-4, 45 steps  ->  0.000 retention, 1.01x damage   stable, too slow

2e-4 is not the wrong rate; 45 steps is too few *for* that rate. Standard fine-tuning runs
thousands of steps at 2e-4. So the sweep is two-dimensional: lower rates need more steps,
and the question is whether any (rate, budget) pair lands inside both bounds at once.

The bar is unchanged and was fixed before any of this was known:

    retention >= 0.50   AND   ppl_ratio <= 1.5x

**Per-seed values are printed for every cell.** Damage is bimodal — the same configuration
diverges on one seed and not another — so a mean over seeds averages a divergence with a
non-divergence and describes neither. A cell whose mean clears the bar while one seed
diverged has not found a working point; it has found a coin flip.

    uv run python -m scripts.stage1_sweep --facts 20 --seeds 2
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as stats
from pathlib import Path

from harness.gates import wilson_interval
from scripts.stage1_experiment import PPL_CEILING, RETENTION_FLOOR, run_cell

LEARNING_RATES = [2e-4, 5e-4, 1e-3, 2e-3]
EPOCH_COUNTS = [3, 8, 15]

# Above this multiple of baseline perplexity a seed is counted as diverged rather than
# merely degraded. Well clear of the stable band (~1-2x) and well below any blow-up (100x+).
DIVERGENCE_THRESHOLD = 10.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--generated", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lrs", type=float, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, nargs="+", default=None)
    parser.add_argument("--out", default="runs/stage1_sweep.json")
    args = parser.parse_args()

    from arms.augment import augment_all
    from arms.memory_methods import MethodConfig
    from data.synthetic_facts import FACTS
    from harness.reader import Reader
    from scripts.forgetting_curve import HELDOUT_TEXT

    reader = Reader(device=args.device, prompt_mode="none")
    strict = Reader(device=args.device, prompt_mode="strict")
    reader.model

    candidates = [
        f for f in FACTS if not f.matches(reader.generate("", f.question, max_new_tokens=24).text)
    ]
    facts = candidates[: args.facts]
    print(f"facts: {len(facts)}   augmenting...")
    aug = augment_all(facts, reader=reader, n_generated=args.generated)
    sizes = [a.count for a in aug]
    print(f"  variants per fact: median {int(stats.median(sizes))}\n")
    print(f"bar: retention >= {RETENTION_FLOOR}, ppl_ratio <= {PPL_CEILING}x\n")

    header = (
        f"{'lr':>8} {'epochs':>7} {'steps':>6} {'retention':>10} {'95% CI':>16} "
        f"{'ppl':>9} {'div':>5}"
    )
    print(header)
    print("-" * (len(header) + 18))

    rows = []
    winners = []
    grid = itertools.product(args.lrs or LEARNING_RATES, args.epochs or EPOCH_COUNTS)
    for lr, epochs in grid:
        # Sane optimisation throughout: the question here is rate and budget, not whether
        # clipping helps. Running this without clipping would re-answer a settled question.
        config = MethodConfig(grad_clip=1.0, warmup_frac=0.1)
        per_seed = []
        for seed in range(args.seeds):
            try:
                per_seed.append(
                    run_cell(
                        reader, strict, config, lr, True, facts, aug,
                        epochs, seed, HELDOUT_TEXT,
                    )
                )
            except Exception as exc:
                print(f"{lr:>8.0e} {epochs:>7}  FAILED: {type(exc).__name__}: {exc}")
                break
        if not per_seed:
            continue

        hits = sum(r["hits"] for r in per_seed)
        total = sum(r["n_facts"] for r in per_seed)
        ret = hits / total
        lo, hi = wilson_interval(hits, total)
        ppl = stats.mean(r["ppl_ratio"] for r in per_seed)
        diverged = sum(1 for r in per_seed if r["ppl_ratio"] > DIVERGENCE_THRESHOLD)
        steps = int(stats.mean(r["steps"] for r in per_seed))

        # A cell only counts as working if EVERY seed clears both bounds. A mean that
        # clears while one seed diverged is a coin flip, not an operating point.
        all_clear = all(
            r["retention"] >= RETENTION_FLOOR and r["ppl_ratio"] <= PPL_CEILING
            for r in per_seed
        )
        mark = "  <-- WORKS (every seed)" if all_clear else ""
        print(
            f"{lr:>8.0e} {epochs:>7} {steps:>6} {ret:>10.3f} [{lo:.3f}, {hi:.3f}] "
            f"{ppl:>8.2f}x {diverged:>3}/{len(per_seed)}{mark}"
        )
        print(
            "                        per-seed: "
            + "  ".join(f"{r['retention']:.2f}@{r['ppl_ratio']:.1f}x" for r in per_seed)
        )
        for r in per_seed:
            r["lr"], r["epochs"] = lr, epochs
        rows.extend(per_seed)
        if all_clear:
            winners.append({"lr": lr, "epochs": epochs, "retention": ret, "ppl_ratio": ppl})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "retention_floor": RETENTION_FLOOR,
                "ppl_ceiling": PPL_CEILING,
                "divergence_threshold": DIVERGENCE_THRESHOLD,
                "n_facts": len(facts),
                "winners": winners,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")

    if winners:
        best = max(winners, key=lambda w: w["retention"])
        print(
            f"\nWORKING POINT: lr={best['lr']:.0e} epochs={best['epochs']} "
            f"-> retention {best['retention']:.3f} at {best['ppl_ratio']:.2f}x, every seed."
        )
    else:
        print("\nNo cell cleared both bounds on every seed. Report the grid, not a winner.")


if __name__ == "__main__":
    main()
