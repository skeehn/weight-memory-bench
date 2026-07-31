"""AlphaEdit against the same bar as everything else.

Every method measured so far is gradient descent on a LoRA adapter, and they share a failure
mode: the update that stores a fact also moves everything else. AlphaEdit projects each
weight perturbation into the **null space of preserved-knowledge keys**, so preserved
behaviour is unchanged by construction rather than by regularisation strength.

The comparison is against the best LoRA points measured, on identical facts, probes and
held-out text:

    5e-4 /  45 steps   0.200 retention @  1.28x     only LoRA cell inside the ceiling
    1e-3 /  45 steps   0.375 retention @  2.02x
    1e-3 / 120 steps   0.725 retention @ 56.47x     best retention, one seed diverged

Bar unchanged and set before any of this: retention >= 0.50 AND ppl_ratio <= 1.5x.

`null_space_retention` is the diagnostic to watch. It reports how much of the raw update
survived projection. Near zero means the update pointed almost entirely at directions
preserved knowledge occupies, so the projection removed it — which predicts high
preservation and no learning, and would say the two are genuinely inseparable in this model
rather than merely hard to separate.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

RETENTION_FLOOR = 0.50
PPL_CEILING = 1.5


def _perplexity(reader, text: str) -> float:
    import torch

    model, tokenizer = reader._load()
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        reader.device
    )
    with torch.no_grad():
        loss = model(input_ids=ids, labels=ids).loss
    return float(torch.exp(loss))


def run(
    n_facts: int = 20,
    threshold: float = 2e-2,
    edit_steps: int = 25,
    edit_lr: float = 0.5,
    device: str = "auto",
    size: str = "1B",
) -> dict:
    from arms.alpha_edit import AlphaEditArm
    from arms.augment import augment_all
    from arms.replay_corpus import REPLAY_TEXTS
    from data.synthetic_facts import FACTS
    from harness.reader import Reader
    from harness.tokens import READER_LADDER
    from scripts.forgetting_curve import HELDOUT_TEXT

    # prompt_mode="none": closed world, every probe's answer is injected, so there is
    # nothing to legitimately decline. The strict prompt suppressed in-context extraction
    # from 98% to 38% and hid 17.5 points of retention on identical weights.
    reader = Reader(model=READER_LADDER.get(size, size), device=device, prompt_mode="none")
    reader.model

    candidates = [
        f for f in FACTS if not f.matches(reader.generate("", f.question, max_new_tokens=24).text)
    ]
    facts = candidates[:n_facts]

    arm = AlphaEditArm(
        reader, null_space_threshold=threshold, edit_lr=edit_lr, edit_steps=edit_steps
    )

    baseline_ppl = _perplexity(reader, HELDOUT_TEXT)

    # The null space is estimated over text whose behaviour must not change. Reusing the
    # replay corpus is deliberate: it is already verified disjoint from the held-out
    # perplexity passage, so the preservation target and the preservation *measurement*
    # cannot be the same text.
    projection = arm.build_projection(REPLAY_TEXTS)

    # Augmented surface forms, because Allen-Zhu applies to any weight-space injection:
    # one phrasing gives memorization without extractability regardless of the mechanism.
    aug = augment_all(facts, reader=reader, n_generated=6)
    targets = [v for a in aug for v in a.variants]

    reports = arm.edit(facts, target_texts=targets)

    hits = sum(1 for f in facts if f.matches(arm.answer(f.question).text))
    after_ppl = _perplexity(reader, HELDOUT_TEXT)
    retention = hits / max(1, len(facts))
    ratio = after_ppl / baseline_ppl

    arm.reset()
    restored_ppl = _perplexity(reader, HELDOUT_TEXT)

    result = {
        "method": "alpha_edit",
        "n_facts": len(facts),
        "retention": retention,
        "hits": hits,
        "ppl_ratio": ratio,
        "baseline_ppl": baseline_ppl,
        "after_ppl": after_ppl,
        # Sanity: restoring the snapshot must return perplexity to baseline. If it does
        # not, the edit leaked outside the layers being tracked and every number is suspect.
        "restored_ppl_ratio": restored_ppl / baseline_ppl,
        "null_space_rank": projection.null_space_rank,
        "key_dim": projection.key_dim,
        "layers": list(projection.layers),
        "threshold": threshold,
        "edit_steps": edit_steps,
        "edit_lr": edit_lr,
        "n_target_texts": len(targets),
        "null_space_retention": stats.mean(r.null_space_retention for r in reports)
        if reports
        else 0.0,
        "residual_before": stats.mean(r.residual_before for r in reports) if reports else 0.0,
        "residual_after": stats.mean(r.residual_after for r in reports) if reports else 0.0,
        "clears_bar": retention >= RETENTION_FLOOR and ratio <= PPL_CEILING,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=2e-2)
    parser.add_argument("--edit-steps", type=int, default=25)
    parser.add_argument("--edit-lr", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="runs/alphaedit.json")
    args = parser.parse_args()

    result = run(
        n_facts=args.facts,
        threshold=args.threshold,
        edit_steps=args.edit_steps,
        edit_lr=args.edit_lr,
        device=args.device,
    )
    print(json.dumps(result, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
