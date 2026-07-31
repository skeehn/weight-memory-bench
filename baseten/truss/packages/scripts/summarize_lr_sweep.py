"""Reduce the learning-rate sweep to the comparison that actually means something.

The sweep produces 12 numbers. Only three of them matter: each size's result at *its own*
best learning rate. Comparing sizes at a shared learning rate is what made the first
scaling run uninterpretable, because identical hyperparameters are a much larger effective
step on a small model than a large one.

Also reports the full grid, because "8B never works at any learning rate we tried" and "8B
works but needs a different one" are different findings and the best-of table alone hides
which one happened.

    uv run python -m scripts.summarize_lr_sweep runs/lr_sweep/all.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SIZES = ["1B", "3B", "8B"]


def load(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    results = payload.get("results", payload)
    if isinstance(results, dict):
        results = list(results.values())
    return [r for r in results if isinstance(r, dict) and r.get("valid_probes")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="runs/lr_sweep/all.json")
    args = parser.parse_args()

    rows = load(Path(args.path))
    if not rows:
        print("no usable results")
        return

    lrs = sorted({r["lr"] for r in rows})
    by = {(r["size"], r["lr"]): r for r in rows}

    print("Full grid (mean recall across seeds)\n")
    header = f"{'size':>5} " + " ".join(f"{lr:>9.0e}" for lr in lrs)
    print(header)
    print("-" * len(header))
    for size in SIZES:
        cells = []
        for lr in lrs:
            r = by.get((size, lr))
            cells.append(f"{r['mean_recall']:>9.3f}" if r else f"{'--':>9}")
        print(f"{size:>5} " + " ".join(cells))

    print("\nBest per size, each at its own best learning rate\n")
    print(f"{'size':>5} {'best lr':>9} {'recall':>8} {'range':>12} {'sd':>7} {'probes':>7}")
    print("-" * 54)
    best = {}
    for size in SIZES:
        candidates = [r for r in rows if r["size"] == size]
        if not candidates:
            continue
        top = max(candidates, key=lambda r: r["mean_recall"])
        best[size] = top
        rng = f"{top['min_recall']:.2f}-{top['max_recall']:.2f}"
        print(
            f"{size:>5} {top['lr']:>9.0e} {top['mean_recall']:>8.3f} {rng:>12} "
            f"{top['sd_recall']:>7.3f} {top['valid_probes']:>7}"
        )

    # The probe sets differ by size, so the best-of numbers above are still not strictly
    # comparable. Recompute on the intersection, which is.
    sets = {s: set(r["per_probe_hits"]) for s, r in best.items() if "per_probe_hits" in r}
    if len(sets) == len(best) and best:
        common = set.intersection(*sets.values())
        if common:
            print(f"\nOn the {len(common)} probes valid at every size\n")
            print(f"{'size':>5} {'best lr':>9} {'recall':>8}")
            print("-" * 24)
            for size in SIZES:
                if size not in best:
                    continue
                r = best[size]
                hits = sum(r["per_probe_hits"][p] for p in common)
                print(
                    f"{size:>5} {r['lr']:>9.0e} {hits / (len(common) * r['seeds']):>8.3f}"
                )

    print("\nVerdict")
    if best:
        ordered = [best[s]["mean_recall"] for s in SIZES if s in best]
        if all(x < 0.5 for x in ordered):
            print("  No size reaches even half recall at its own best learning rate.")
            print("  Weight memory does not work at any scale tested. The confound is gone;")
            print("  the finding survives it.")
        elif ordered == sorted(ordered):
            print("  Recall increases with size once each rung is tuned. This is a scaling")
            print("  result: the mechanism works, it just needs a bigger model.")
        else:
            print("  Non-monotonic even after per-rung tuning. Report the grid, not a trend.")


if __name__ == "__main__":
    main()
