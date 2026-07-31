"""Is there a KL weight that buys retention before it costs perplexity?

The method comparison sampled `kl_weight` at exactly one point, 0.5, and it froze the model:
1.51x perplexity (the least damage of anything measured) and 0.000 retention. That rules out
*that* setting, not the method. The real question is the shape of the curve between 0 and
0.5.

Three outcomes, and they mean different things:

**A knee** — retention rises while perplexity stays near 1.0x, then perplexity climbs later.
That interval is a working method, and the project has a positive result.

**A straight trade** — retention and perplexity rise together, proportionally, with no
interval where one moves without the other. The stability-plasticity tradeoff is strict at
this scale, which is a real and publishable claim rather than a failed search.

**Nothing** — retention stays at zero across the whole range while perplexity climbs. Then
the KL term is not mediating a tradeoff at all; it is just a brake, and the fact never
enters regardless.

Chat-format is held ON throughout, because it cut per-step damage ~114x for free and with no
retention cost. Holding a free improvement fixed makes the KL axis cleaner to read.

    uv run python -m scripts.kl_sweep --seeds 2
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

from scripts.method_comparison import PPL_CEILING, RETENTION_FLOOR, quadrant, run_one

# Log-ish spacing from "off" to the value already known to freeze the model. If a knee
# exists it is almost certainly in the low end, where the anchor is weak enough to permit
# movement but strong enough to prevent divergence.
KL_WEIGHTS = [0.0, 0.005, 0.02, 0.05, 0.1, 0.25, 0.5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--size", default="1B")
    parser.add_argument("--no-chat-format", action="store_true")
    # The transition proved sharper than the default grid: damage collapsed 155x and
    # retention went to zero between 0 and 0.005. Resolving that region needs a finer
    # grid, not a source edit.
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--out", default="runs/kl_sweep.json")
    args = parser.parse_args()

    from arms.memory_methods import MethodConfig
    from harness.reader import Reader
    from harness.tokens import READER_LADDER
    from scripts.forgetting_curve import HELDOUT_TEXT
    from scripts.memory_gate import EPISODES, PROBES

    reader = Reader(model=READER_LADDER[args.size], device=args.device)
    reader.model

    valid = []
    for question, expected in PROBES:
        unknown = expected.lower() not in reader.generate("", question).text.lower()
        answerable = (
            expected.lower() in reader.generate("\n".join(EPISODES), question).text.lower()
        )
        if unknown and answerable:
            valid.append((question, expected))
    chat = not args.no_chat_format
    print(f"valid probes: {len(valid)}/{len(PROBES)}  chat_format={chat}")
    print(f"bar: retention >= {RETENTION_FLOOR}, ppl_ratio <= {PPL_CEILING}x\n")
    if not valid:
        print("no valid probes")
        return

    print(f"{'kl_weight':>10} {'retention':>10} {'ppl_ratio':>11}  verdict")
    print("-" * 62)

    rows = []
    summary = []
    for weight in (args.weights or KL_WEIGHTS):
        config = MethodConfig(chat_format=chat, kl_weight=weight)
        per_seed = []
        for seed in range(args.seeds):
            try:
                per_seed.append(
                    run_one(
                        reader,
                        f"kl{weight:g}",
                        config,
                        EPISODES,
                        valid,
                        args.updates,
                        seed,
                        HELDOUT_TEXT,
                    )
                )
            except Exception as exc:
                print(f"{weight:>10g}  FAILED: {type(exc).__name__}: {exc}")
                break
        if not per_seed:
            continue

        ret = stats.mean(r["retention"] for r in per_seed)
        ratio = stats.mean(r["ppl_ratio"] for r in per_seed)

        # The process is BIMODAL: it either stays stable (~1.5x) or diverges (100x+), and
        # the seed decides which. Measured at kl=0.003: one seed 259.96x, the other 1.34x.
        # A mean of a divergence and a non-divergence describes neither, so the per-seed
        # values and the divergence count are printed alongside it. `diverged` counts seeds
        # past 10x, which is well clear of the stable band and well below any blow-up.
        diverged = sum(1 for r in per_seed if r["ppl_ratio"] > 10)
        retained = sum(1 for r in per_seed if r["retention"] > 0)
        verdict = quadrant(ret, ratio)
        flag = "  <-- " if verdict == "WORKS" else "  "
        print(
            f"{weight:>10g} {ret:>10.3f} {ratio:>10.2f}x  "
            f"diverged {diverged}/{len(per_seed)}  retained {retained}/{len(per_seed)}"
            f"{flag}{verdict}"
        )
        print(
            "           per-seed ppl: "
            + " ".join(f"{r['ppl_ratio']:.2f}x" for r in per_seed)
            + "   retention: "
            + " ".join(f"{r['retention']:.2f}" for r in per_seed)
        )
        rows.extend(per_seed)
        summary.append(
            {
                "kl_weight": weight,
                "retention": ret,
                "ppl_ratio": ratio,
                "diverged": diverged,
                "retained_seeds": retained,
                "n_seeds": len(per_seed),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "retention_floor": RETENTION_FLOOR,
                "ppl_ceiling": PPL_CEILING,
                "chat_format": chat,
                "updates": args.updates,
                "seeds": args.seeds,
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")

    if not summary:
        return

    # Read the shape, using the criteria stated in the module docstring rather than
    # whatever the numbers happen to suggest afterwards.
    max_ret = max(s["retention"] for s in summary)
    safe = [s for s in summary if s["ppl_ratio"] <= PPL_CEILING]
    best_safe = max(safe, key=lambda s: s["retention"]) if safe else None

    print()
    if max_ret < 0.01:
        print("SHAPE: nothing. Retention is zero across the entire range.")
        print("  The KL term is a brake, not a tradeoff dial -- the fact never enters.")
    elif best_safe and best_safe["retention"] >= RETENTION_FLOOR:
        print(f"SHAPE: knee at kl_weight={best_safe['kl_weight']:g}. WORKING METHOD.")
    elif best_safe and best_safe["retention"] > 0:
        print(
            f"SHAPE: partial. Best safe point is kl={best_safe['kl_weight']:g} at "
            f"retention {best_safe['retention']:.3f}, under the {RETENTION_FLOOR} floor."
        )
    else:
        print("SHAPE: strict trade. Retention only appears once perplexity has already gone.")
        print("  Stability and plasticity are not separable at this scale.")


if __name__ == "__main__":
    main()
