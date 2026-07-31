"""Find the settings, if any, at which a fact survives into the weights.

The gate failed at the default learning rate and epoch count, which says nothing about
whether weight memory works -- only that 4 optimizer steps do not memorize. This sweeps the
two knobs that decide it and reports where the transition happens.

It reports two signals, because exact-match generation alone cannot distinguish "close" from
"broken":

**Fact NLL** -- the model's negative log-likelihood of the target fact, before and after
ingestion. This is continuous and sensitive. If it does not move at all, the adapter is not
training and no amount of extra epochs will help; that is a bug, not a hyperparameter.

**Recall** -- whether the fact is actually generated with an empty context. This is the real
outcome, but it is a step function, so on its own it gives no gradient to follow.

A run where NLL collapses but recall stays zero means the fact is being memorized as text
without being retrievable as an answer, which is a genuinely different failure and worth
knowing about before spending.

    uv run python scripts/memory_sweep.py
"""

from __future__ import annotations

import argparse
import itertools

from arms.weight_memory import WeightMemoryArm
from harness.reader import Reader
from scripts.memory_gate import EPISODES, PROBES


def fact_nll(reader: Reader, model, question: str, answer: str) -> float:
    """Negative log-likelihood of the expected answer, given the question and no context.

    Measured on the answer tokens only. Including the prompt would dilute the signal with
    tokens the update was never trying to change.

    The answer is converted to the surface form the model would actually generate before
    encoding. Scoring the bare lowercase string measures a different token sequence -- and
    did, for every NLL number this repo produced before the fix. See `answer_surface`.
    """
    import torch

    from scripts.memory_gate import answer_surface

    tokenizer = reader.tokenizer
    prompt = reader.build_prompt("", question)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(
        answer_surface(answer), return_tensors="pt", add_special_tokens=False
    )["input_ids"]

    input_ids = torch.cat([prompt_ids, answer_ids], dim=-1).to(reader.device)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[-1]] = -100  # score the answer only

    with torch.no_grad():
        loss = model(input_ids=input_ids, labels=labels).loss
    return float(loss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--lrs", type=float, nargs="+", default=[1e-4, 5e-4, 2e-3, 1e-2]
    )
    parser.add_argument("--epoch-counts", type=int, nargs="+", default=[10, 50, 200])
    args = parser.parse_args()

    reader = Reader(device=args.device)
    reader.model  # force load once
    print(f"device={reader.device} rank={args.rank}\n")

    # Only probes that are both unknown beforehand and answerable in context are valid.
    valid = []
    for question, expected in PROBES:
        unknown = expected.lower() not in reader.generate("", question).text.lower()
        answerable = (
            expected.lower()
            in reader.generate("\n".join(EPISODES), question).text.lower()
        )
        if unknown and answerable:
            valid.append((question, expected))
    print(f"valid probes: {len(valid)}/{len(PROBES)} -> {[q for q, _ in valid]}\n")
    if not valid:
        print("no valid probes; cannot sweep")
        return

    # ONE arm for the whole sweep. Constructing a new one per config calls get_peft_model
    # again on the same base, which injects a second set of LoRA layers rather than
    # replacing the first -- so every later config silently inherits the training of every
    # earlier one. `reset()` restoring the pristine adapter state is the correct isolation.
    arm = WeightMemoryArm(reader, rank=args.rank)
    model = arm._ensure_adapter()

    arm.reset()
    baseline_nll = sum(fact_nll(reader, model, q, a) for q, a in valid) / len(valid)
    print(f"{'lr':>8} {'epochs':>7} {'steps':>6} {'fact_nll':>9} {'delta':>8} {'recall':>8}")
    print("-" * 52)
    print(
        f"{'--':>8} {'0':>7} {'0':>6} {baseline_nll:>9.3f} {'--':>8} "
        f"{'0/' + str(len(valid)):>8}"
    )

    for lr, epochs in itertools.product(args.lrs, args.epoch_counts):
        arm.learning_rate = lr
        arm.epochs = epochs
        arm.reset()

        # Verify the reset actually took: a config that starts already-trained would
        # reproduce exactly the contamination this loop was restructured to remove.
        reset_nll = sum(fact_nll(reader, model, q, a) for q, a in valid) / len(valid)
        if abs(reset_nll - baseline_nll) > 1e-3:
            raise AssertionError(
                f"reset did not restore the adapter: {reset_nll:.4f} vs {baseline_nll:.4f}"
            )

        steps = arm.ingest(EPISODES)
        after = sum(fact_nll(reader, model, q, a) for q, a in valid) / len(valid)
        hits = sum(
            1 for q, a in valid if a.lower() in arm.answer(q).text.lower()
        )
        print(
            f"{lr:>8.0e} {epochs:>7} {steps:>6} {after:>9.3f} "
            f"{after - baseline_nll:>+8.3f} {str(hits) + '/' + str(len(valid)):>8}"
        )

        if hits == len(valid):
            print(f"\nFULL RECALL at lr={lr:.0e} epochs={epochs}. Use this as the floor.")
            return

    print("\nNo setting achieved full recall. Check the NLL column:")
    print("  NLL falling but recall zero -> memorized as text, not retrievable as answer")
    print("  NLL flat                    -> adapter is not training; that is a bug")


if __name__ == "__main__":
    main()
