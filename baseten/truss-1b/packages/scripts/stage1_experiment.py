"""Does the corrected pipeline actually learn facts into weights?

The previous conclusion — that online weight memory does not work at 1B — rested on a setup
with four defects. This isolates them, so the answer is attributable rather than just
different:

    cell 1  naive        old hyperparameters, no augmentation   (the published baseline)
    cell 2  +hypers      lr 2e-4, grad clip, warmup, no augmentation
    cell 3  +augment     old hyperparameters, augmented corpus
    cell 4  +both        the corrected pipeline

If cell 4 works and cell 2 does not, augmentation was the missing ingredient, which is what
Allen-Zhu & Li predict: a fact seen in one phrasing is memorized but not extractable *no
matter how well it is optimized*. If cell 2 alone works, the divergence was doing the damage
and augmentation is incidental. If neither works, the diagnosis is wrong.

**Retention is measured with `prompt_mode="none"`.** This is a closed world: every probe's
answer was injected, so there is nothing to legitimately decline. The strict abstention
prompt used in every earlier run suppressed in-context extraction from 98% to 38%, which
means it was also suppressing retention — a fact could be in the weights and simply refused.
The strict number is reported alongside, to size that effect.

    uv run python -m scripts.stage1_experiment --facts 20 --seeds 2
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

from harness.gates import wilson_interval

# The bar is unchanged from `method_comparison.py`, deliberately. It was fixed before any of
# this was known and is not being softened now that there is a chance of clearing it.
RETENTION_FLOOR = 0.50
PPL_CEILING = 1.5


def build_cells():
    from arms.memory_methods import MethodConfig

    old = dict(grad_clip=None, warmup_frac=0.0)
    new = dict(grad_clip=1.0, warmup_frac=0.1)
    return [
        ("naive (published baseline)", MethodConfig(**old), 2e-3, False),
        ("+hypers", MethodConfig(**new), 2e-4, False),
        ("+augment", MethodConfig(**old), 2e-3, True),
        ("+both (corrected)", MethodConfig(**new), 2e-4, True),
    ]


def run_cell(reader, strict_reader, config, lr, use_aug, facts, aug_results, epochs, seed, heldout):
    from arms.augment import training_corpus
    from arms.memory_methods import MitigatedMemoryArm

    corpus = (
        training_corpus([a for a in aug_results if a.fact_key in {f.key for f in facts}])
        if use_aug
        else [f.statement for f in facts]
    )

    arm = MitigatedMemoryArm(
        reader,
        config,
        learning_rate=lr,
        epochs=epochs,
        seed=seed,
        heldout_text=heldout,
        max_chunk_tokens=256,
    )
    model = arm._ensure_adapter()
    arm.reset()
    base_ppl = arm.heldout_perplexity()

    report = arm.ingest(corpus)

    hits = sum(1 for f in facts if f.matches(arm.answer(f.question, max_new_tokens=24).text))
    # Same weights, strict prompt, to size how much the abstention instruction alone
    # suppresses a fact that is genuinely present.
    strict_reader._model, strict_reader._tokenizer = reader._model, reader._tokenizer
    strict_hits = sum(
        1
        for f in facts
        if f.matches(strict_reader.generate("", f.question, max_new_tokens=24).text)
    )
    ppl = arm.heldout_perplexity()

    model.unload()
    arm._peft_model = None

    return {
        "seed": seed,
        "hits": hits,
        "strict_hits": strict_hits,
        "n_facts": len(facts),
        "retention": hits / len(facts),
        "retention_strict_prompt": strict_hits / len(facts),
        "ppl_ratio": ppl / base_ppl,
        "steps": report.steps,
        "corpus_size": len(corpus),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--generated", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="runs/stage1.json")
    args = parser.parse_args()

    from arms.augment import augment_all
    from data.synthetic_facts import FACTS
    from harness.reader import Reader
    from scripts.forgetting_curve import HELDOUT_TEXT

    reader = Reader(device=args.device, prompt_mode="none")
    strict = Reader(device=args.device, prompt_mode="strict")
    reader.model

    # Only facts the model does not already know. With no abstention instruction, in-context
    # answerability is ~98%, so this filter now removes almost nothing -- which is the point.
    candidates = [
        f for f in FACTS if not f.matches(reader.generate("", f.question, max_new_tokens=24).text)
    ]
    facts = candidates[: args.facts]
    print(f"facts: {len(facts)} (of {len(FACTS)}; {len(FACTS) - len(candidates)} already known)")

    print(f"augmenting with {args.generated} generations per fact...")
    aug = augment_all(facts, reader=reader, n_generated=args.generated)
    kept = sum(a.generated_kept for a in aug)
    dropped = sum(a.generated_dropped for a in aug)
    sizes = [a.count for a in aug]
    print(
        f"  variants per fact: min {min(sizes)} median {int(stats.median(sizes))} max {max(sizes)}"
        f"   (generated kept {kept}, dropped {dropped})"
    )
    print(f"\nbar: retention >= {RETENTION_FLOOR}, ppl_ratio <= {PPL_CEILING}x\n")

    header = f"{'cell':>26} {'retention':>10} {'95% CI':>16} {'ppl':>9} {'strict':>8} {'steps':>6}"
    print(header)
    print("-" * len(header))

    rows = []
    for label, config, lr, use_aug in build_cells():
        per_seed = []
        for seed in range(args.seeds):
            try:
                per_seed.append(
                    run_cell(
                        reader, strict, config, lr, use_aug, facts, aug,
                        args.epochs, seed, HELDOUT_TEXT,
                    )
                )
            except Exception as exc:
                print(f"{label:>26}  FAILED: {type(exc).__name__}: {exc}")
                break
        if not per_seed:
            continue

        total_hits = sum(r["hits"] for r in per_seed)
        total = sum(r["n_facts"] for r in per_seed)
        lo, hi = wilson_interval(total_hits, total)
        ret = total_hits / total
        strict_ret = sum(r["strict_hits"] for r in per_seed) / total
        ppl = stats.mean(r["ppl_ratio"] for r in per_seed)
        steps = int(stats.mean(r["steps"] for r in per_seed))
        mark = "  <-- CLEARS BAR" if ret >= RETENTION_FLOOR and ppl <= PPL_CEILING else ""
        print(
            f"{label:>26} {ret:>10.3f} [{lo:.3f}, {hi:.3f}] {ppl:>8.2f}x {strict_ret:>8.3f} {steps:>6}{mark}"
        )
        # Per-seed, because the process is bimodal and a mean of a divergence and a
        # non-divergence describes neither.
        print(
            "                           per-seed: "
            + "  ".join(f"{r['retention']:.2f}@{r['ppl_ratio']:.1f}x" for r in per_seed)
        )
        for r in per_seed:
            r["cell"] = label
        rows.extend(per_seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "retention_floor": RETENTION_FLOOR,
                "ppl_ceiling": PPL_CEILING,
                "n_facts": len(facts),
                "epochs": args.epochs,
                "generated_per_fact": args.generated,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
