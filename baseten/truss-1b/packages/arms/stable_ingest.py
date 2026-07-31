"""Turn the divergence lottery into a reliable outcome.

Measured across the sweep, the same configuration produces wildly different damage depending
only on the random initialization of the LoRA adapter:

    lr 1e-3, 120 steps, seed 0:  0.65 retention @   6.1x
    lr 1e-3, 120 steps, seed 1:  0.80 retention @ 106.8x
    lr 2e-3,  45 steps, seed 0:  0.75 retention @ 286.2x
    lr 2e-3,  45 steps, seed 1:  0.75 retention @   2.2x

Retention is roughly stable across seeds; damage is bimodal. That is not a law to be
tuned around, it is a **detectable failure with a cheap remedy**: watch held-out perplexity
during training, abort the moment it leaves the acceptable band, and start again from a
different seed.

The distinction that makes this worth building rather than just lowering the learning rate:
a `ppl_gate` *stops* training and keeps a half-trained adapter, which is why the gated
configs scored zero retention — they preserved the model by never learning. A **restart**
throws the bad run away entirely and tries again, so the successful attempt gets its full
training budget.

The cost is honest and reported: expected wall-clock multiplies by roughly 1/(1-p) where p
is the divergence rate, and a configuration that diverges on every seed will exhaust its
attempts and say so rather than returning a quietly damaged model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Attempt:
    seed: int
    diverged: bool
    ppl_ratio: float
    steps: int
    aborted_at_step: int | None = None


@dataclass
class StableIngestResult:
    """What happened across all attempts, not just the one that survived."""

    succeeded: bool
    seed_used: int | None
    ppl_ratio: float | None
    steps: int
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    @property
    def divergence_rate(self) -> float:
        if not self.attempts:
            return 0.0
        return sum(1 for a in self.attempts if a.diverged) / len(self.attempts)


def ingest_with_restart(
    make_arm,
    texts,
    *,
    max_attempts: int = 5,
    divergence_threshold: float = 3.0,
    check_every: int = 10,
    first_seed: int = 0,
):
    """Train until an attempt completes without diverging.

    `make_arm(seed)` must return a fresh, reset arm — a new adapter per attempt, not a
    rewound one. Reusing an adapter would carry the diverged trajectory's optimizer state
    and initialization into the retry, which is the thing being escaped.

    Perplexity is checked **during** training, not only at the end. Divergence is
    catastrophic and fast (1.3x to 100x inside two epochs was measured), so an end-only
    check wastes the whole budget discovering what a mid-run check catches in seconds.

    Returns `(arm, StableIngestResult)`. On total failure the arm is the last attempt's and
    `succeeded` is False — the caller must check rather than assume, because a diverged
    model answers questions perfectly happily.
    """
    attempts: list[Attempt] = []
    arm = None

    for offset in range(max_attempts):
        seed = first_seed + offset
        arm = make_arm(seed)
        arm._ensure_adapter()
        arm.reset()
        baseline = arm.heldout_perplexity()

        diverged, aborted_at = _train_watched(
            arm, texts, baseline, divergence_threshold, check_every
        )
        ratio = arm.heldout_perplexity() / baseline
        attempts.append(
            Attempt(
                seed=seed,
                diverged=diverged,
                ppl_ratio=ratio,
                steps=arm.updates_applied,
                aborted_at_step=aborted_at,
            )
        )

        if not diverged and ratio <= divergence_threshold:
            return arm, StableIngestResult(
                succeeded=True,
                seed_used=seed,
                ppl_ratio=ratio,
                steps=arm.updates_applied,
                attempts=attempts,
            )

    return arm, StableIngestResult(
        succeeded=False,
        seed_used=None,
        ppl_ratio=attempts[-1].ppl_ratio if attempts else None,
        steps=attempts[-1].steps if attempts else 0,
        attempts=attempts,
    )


def _train_watched(arm, texts, baseline, threshold, check_every):
    """One training run, aborted the moment held-out perplexity leaves the band.

    Reimplements the training loop rather than calling `arm.ingest` because the check has to
    interrupt it. Kept deliberately close to `MitigatedMemoryArm.ingest` — clipping, warmup
    and cosine schedule included — so a run here is comparable to one there.
    """
    import math

    import torch

    model = arm._ensure_adapter()
    tokenizer = arm.reader.tokenizer
    device = arm.reader.device
    config = arm.config

    sequence = arm._training_sequence(texts)
    if not sequence:
        return False, None

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=arm.learning_rate
    )
    total = max(1, arm.epochs * len(sequence))
    warmup = max(1, int(total * config.warmup_frac))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    step = 0
    model.train()
    for _ in range(arm.epochs):
        for text, _is_replay in sequence:
            batch = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=arm.max_chunk_tokens
            ).to(device)
            loss = model(**batch, labels=batch["input_ids"]).loss
            loss.backward()
            if config.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], config.grad_clip
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            arm.updates_applied = step

            if step % check_every == 0:
                model.eval()
                ratio = arm.heldout_perplexity() / baseline
                model.train()
                if ratio > threshold:
                    model.eval()
                    return True, step

    model.eval()
    return False, None
