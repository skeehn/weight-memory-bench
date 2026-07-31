"""Can any method write memory into weights without destroying the model?

The naive baseline fails on both axes at once: held-out perplexity 36.8x after 100 online
updates, retention ending at zero. This runs the same protocol across mitigation methods and
sorts them into quadrants.

**The bar was fixed before any of these ran, and it is not "beat the baseline".** Beating a
method that destroys the model is trivial and meaningless. A method succeeds only if:

    retention  >= RETENTION_FLOOR   (it actually learned the facts)
    ppl_ratio  <= PPL_CEILING       (it did not damage the model doing so)

Everything else lands in a named failure quadrant, so a method that merely trades one axis
for the other cannot be written up as progress.

    uv run python -m scripts.method_comparison --updates 50 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

# Pre-registered success criteria. Retention of a third is roughly the naive baseline's
# best-ever single point, so the floor demands clearly more than noise. 1.5x perplexity is
# "the model is still itself" -- the naive run passed 2.41x by update 25 and never recovered.
RETENTION_FLOOR = 0.50
PPL_CEILING = 1.5


def quadrant(retention: float, ppl_ratio: float) -> str:
    if retention >= RETENTION_FLOOR and ppl_ratio <= PPL_CEILING:
        return "WORKS"
    if retention >= RETENTION_FLOOR:
        return "learns but damages"
    if ppl_ratio <= PPL_CEILING:
        return "safe but learns nothing"
    return "damages and learns nothing"


def build_configs():
    from arms.memory_methods import MethodConfig

    return [
        ("naive", MethodConfig()),
        ("replay", MethodConfig(replay=True)),
        ("chat-format", MethodConfig(chat_format=True)),
        ("replay+chatfmt", MethodConfig(replay=True, chat_format=True)),
        ("kl anchor", MethodConfig(kl_weight=0.5)),
        ("ppl gate 1.5x", MethodConfig(ppl_gate=1.5)),
        ("replay+chatfmt+gate", MethodConfig(replay=True, chat_format=True, ppl_gate=1.5)),
        ("everything", MethodConfig(replay=True, chat_format=True, kl_weight=0.5, ppl_gate=1.5)),
    ]


def run_one(reader, label, config, episodes, probes, updates, seed, heldout):
    from arms.memory_methods import MitigatedMemoryArm

    arm = MitigatedMemoryArm(
        reader, config, learning_rate=2e-3, epochs=updates, seed=seed, heldout_text=heldout
    )
    arm._ensure_adapter()
    arm.reset()
    baseline_ppl = arm.heldout_perplexity()

    report = arm.ingest(episodes)
    retained = sum(
        1 for q, expected in probes if expected.lower() in arm.answer(q).text.lower()
    ) / len(probes)
    ppl = arm.heldout_perplexity()

    arm._peft_model.unload()
    arm._peft_model = None

    return {
        "method": label,
        "config": config.label(),
        "seed": seed,
        "retention": retained,
        "heldout_ppl": ppl,
        "baseline_ppl": baseline_ppl,
        "ppl_ratio": ppl / baseline_ppl,
        "steps": report.steps,
        "stopped_early": report.stopped_early,
        "stop_reason": report.stop_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--size", default="1B")
    parser.add_argument("--out", default="runs/method_comparison.json")
    args = parser.parse_args()

    from harness.reader import Reader
    from harness.tokens import READER_LADDER
    from scripts.forgetting_curve import HELDOUT_TEXT
    from scripts.memory_gate import EPISODES, PROBES

    reader = Reader(model=READER_LADDER[args.size], device=args.device)
    reader.model

    # Same validity controls as everywhere else: a probe the model already knows cannot
    # demonstrate learning, and one it cannot answer with the text in front of it cannot
    # demonstrate anything at all.
    valid = []
    for question, expected in PROBES:
        unknown = expected.lower() not in reader.generate("", question).text.lower()
        answerable = (
            expected.lower()
            in reader.generate("\n".join(EPISODES), question).text.lower()
        )
        if unknown and answerable:
            valid.append((question, expected))
    print(f"valid probes: {len(valid)}/{len(PROBES)}")
    print(f"bar: retention >= {RETENTION_FLOOR}, ppl_ratio <= {PPL_CEILING}x\n")
    if not valid:
        print("no valid probes")
        return

    rows = []
    print(f"{'method':>22} {'retention':>10} {'ppl_ratio':>10} {'steps':>6}  verdict")
    print("-" * 78)

    for label, config in build_configs():
        per_seed = []
        for seed in range(args.seeds):
            try:
                per_seed.append(
                    run_one(
                        reader, label, config, EPISODES, valid, args.updates, seed, HELDOUT_TEXT
                    )
                )
            except Exception as exc:
                print(f"{label:>22}  FAILED: {type(exc).__name__}: {exc}")
                break
        if not per_seed:
            continue

        ret = stats.mean(r["retention"] for r in per_seed)
        ratio = stats.mean(r["ppl_ratio"] for r in per_seed)
        steps = int(stats.mean(r["steps"] for r in per_seed))
        verdict = quadrant(ret, ratio)
        flag = "  <-- " if verdict == "WORKS" else "  "
        print(
            f"{label:>22} {ret:>10.3f} {ratio:>9.2f}x {steps:>6}{flag}{verdict}"
        )
        rows.extend(per_seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "retention_floor": RETENTION_FLOOR,
                "ppl_ceiling": PPL_CEILING,
                "updates": args.updates,
                "seeds": args.seeds,
                "size": args.size,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
