"""Methods for writing memory into weights *without* destroying the model.

The naive baseline (`arms/weight_memory.py`) fails in two measured ways: held-out perplexity
rises 36.8x over 100 online updates, and facts that demonstrably enter the weights (fact NLL
drops sharply) cannot be retrieved as answers. This module implements four mitigations, all
composable, so the failure modes can be attacked separately and together.

Each is motivated by one of those two measurements rather than picked off a list:

**Replay** — mix generic text into every update. If perplexity on unrelated text explodes,
the gradient is dragging the whole model rather than the fact, and rehearsal is the direct
counter.

**Chat-format ingestion** — train on the transcript rendered through the same chat template
used at query time. The baseline trained on raw turns and queried through a chat template,
so the model saw one token distribution and was asked in another. This aligns them **without
fabricating question-answer labels**, which would answer an easier question than the one
being asked.

**KL anchor** — penalize divergence from the frozen base model's own outputs. The principled
form of replay: rather than hoping generic text keeps the model in place, explicitly require
it to keep predicting what it used to predict. Uses PEFT's `disable_adapter()` so no second
copy of the weights is needed.

**Perplexity gate** — stop updating when held-out perplexity crosses a threshold. Not a
learning method, a controller. The forgetting curve has an obvious cliff (2.41x at 25
updates, 36.83x at 100); a method that simply stops before it is a legitimate baseline and
embarrassingly cheap.

The bar, fixed before any of this ran: **retention up, held-out perplexity flat.** Beating
the naive baseline is not enough, because the naive baseline destroys the model.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field

from harness.reader import Reader

from .base import Selection, empty_selection
from .replay_corpus import REPLAY_TEXTS
from .weight_memory import DEFAULT_TARGET_MODULES


@dataclass
class MethodConfig:
    """Which mitigations are active. All four compose."""

    replay: bool = False
    replay_ratio: float = 1.0  # replay chunks per episode chunk
    chat_format: bool = False
    kl_weight: float = 0.0  # 0 disables the anchor
    ppl_gate: float | None = None  # stop when held-out ppl exceeds this multiple of baseline
    # Standard practice, absent from every earlier run and a direct cause of the
    # divergence measured there. lr 2e-3 with no clipping and no warmup is roughly 10x
    # the accepted LoRA range with none of the usual stabilisers.
    grad_clip: float | None = 1.0
    warmup_frac: float = 0.1

    def label(self) -> str:
        parts = []
        if self.replay:
            parts.append(f"replay{self.replay_ratio:g}")
        if self.chat_format:
            parts.append("chatfmt")
        if self.kl_weight:
            parts.append(f"kl{self.kl_weight:g}")
        if self.ppl_gate:
            parts.append(f"gate{self.ppl_gate:g}x")
        return "+".join(parts) or "naive"


@dataclass
class IngestReport:
    steps: int
    stopped_early: bool = False
    stop_reason: str = ""
    ppl_trace: list[float] = field(default_factory=list)


class MitigatedMemoryArm:
    """Weight memory with anti-forgetting mitigations.

    Deliberately a separate class from `WeightMemoryArm` rather than a flag on it: the
    baseline has to stay exactly as measured, or the comparison it anchors stops being a
    comparison.
    """

    name = "mitigated_memory"

    def __init__(
        self,
        reader: Reader,
        config: MethodConfig,
        rank: int = 16,
        alpha: int | None = None,
        learning_rate: float = 2e-3,
        epochs: int = 1,
        max_chunk_tokens: int = 512,
        target_modules=DEFAULT_TARGET_MODULES,
        seed: int | None = None,
        heldout_text: str | None = None,
    ) -> None:
        self.reader = reader
        self.config = config
        self.rank = rank
        self.alpha = alpha if alpha is not None else 2 * rank
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_chunk_tokens = max_chunk_tokens
        self.target_modules = list(target_modules)
        self.seed = seed
        self.heldout_text = heldout_text

        self._peft_model = None
        self._pristine_state = None
        self._baseline_ppl: float | None = None
        self.updates_applied = 0

    # -- adapter lifecycle (mirrors the baseline exactly) ---------------------------

    def _ensure_adapter(self):
        if self._peft_model is None:
            import os

            import torch
            from peft import LoraConfig, get_peft_model

            if self.seed is not None:
                torch.manual_seed(self.seed)
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass

            config = LoraConfig(
                r=self.rank,
                lora_alpha=self.alpha,
                lora_dropout=0.0,
                target_modules=self.target_modules,
                task_type="CAUSAL_LM",
                bias="none",
            )
            self._peft_model = get_peft_model(self.reader.model, config)
            self._pristine_state = deepcopy(self._adapter_state())
        return self._peft_model

    def _adapter_state(self):
        return {
            k: v.detach().clone()
            for k, v in self._peft_model.state_dict().items()
            if "lora_" in k
        }

    def reset(self) -> None:
        model = self._ensure_adapter()
        if self._pristine_state:
            model.load_state_dict(self._pristine_state, strict=False)
        self.updates_applied = 0

    # -- measurement ---------------------------------------------------------------

    def heldout_perplexity(self) -> float:
        import torch

        model = self._ensure_adapter()
        ids = self.reader.tokenizer(
            self.heldout_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.reader.device)
        with torch.no_grad():
            loss = model(input_ids=ids, labels=ids).loss
        return float(torch.exp(loss))

    # -- training text ---------------------------------------------------------------

    def _render(self, text: str) -> str:
        """Apply the chat template if chat-format ingestion is on.

        The transcript is wrapped as an assistant turn so the model trains on the same
        token distribution it is later queried in. No question-answer labels are invented:
        only the framing changes, not the content.
        """
        if not self.config.chat_format:
            return text
        tokenizer = self.reader.tokenizer
        messages = [{"role": "assistant", "content": text}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            return text

    def _chunks(self, texts):
        tokenizer = self.reader.tokenizer
        batch, count = [], 0
        for text in texts:
            n = len(tokenizer.encode(text, add_special_tokens=False))
            if batch and count + n > self.max_chunk_tokens:
                yield self._render("\n".join(batch))
                batch, count = [], 0
            batch.append(text)
            count += n
        if batch:
            yield self._render("\n".join(batch))

    def _training_sequence(self, texts) -> list[tuple[str, bool]]:
        """(text, is_replay) pairs for one pass.

        Replay is interleaved rather than appended so the model never sees a long run of
        episode-only gradient. Order matters here: a block of replay after the fact has
        already been over-written does not undo the damage.
        """
        episode_chunks = [(c, False) for c in self._chunks(texts) if c.strip()]
        if not self.config.replay:
            return episode_chunks

        replay = [(self._render(t), True) for t in REPLAY_TEXTS]
        per = max(1, round(self.config.replay_ratio))
        out: list[tuple[str, bool]] = []
        i = 0
        for chunk in episode_chunks:
            out.append(chunk)
            for _ in range(per):
                out.append(replay[i % len(replay)])
                i += 1
        return out

    # -- the update ------------------------------------------------------------------

    def ingest(self, texts) -> IngestReport:
        import torch

        model = self._ensure_adapter()
        tokenizer = self.reader.tokenizer
        device = self.reader.device

        if self.config.ppl_gate and self._baseline_ppl is None:
            self._baseline_ppl = self.heldout_perplexity()

        sequence = self._training_sequence(texts)
        if not sequence:
            return IngestReport(steps=0)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=self.learning_rate
        )

        # Linear warmup then cosine decay. Absent before, which meant every run began at
        # full learning rate on the very first step -- the point at which the adapter is
        # random and the gradient is largest.
        total_steps = max(1, self.epochs * len(sequence))
        warmup_steps = max(1, int(total_steps * self.config.warmup_frac))

        def lr_at(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

        report = IngestReport(steps=0)
        model.train()
        for _ in range(self.epochs):
            for text, _is_replay in sequence:
                batch = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=self.max_chunk_tokens
                ).to(device)
                outputs = model(**batch, labels=batch["input_ids"])
                loss = outputs.loss

                if self.config.kl_weight:
                    # What the model predicted BEFORE this adapter existed. `disable_adapter`
                    # gives the frozen base without a second copy of the weights in memory.
                    with torch.no_grad(), model.disable_adapter():
                        base_logits = model(**batch).logits
                    kl = torch.nn.functional.kl_div(
                        torch.log_softmax(outputs.logits, dim=-1),
                        torch.log_softmax(base_logits, dim=-1),
                        log_target=True,
                        reduction="batchmean",
                    )
                    loss = loss + self.config.kl_weight * kl

                loss.backward()
                if self.config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        self.config.grad_clip,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                report.steps += 1

            if self.config.ppl_gate:
                model.eval()
                ppl = self.heldout_perplexity()
                model.train()
                report.ppl_trace.append(ppl)
                if ppl > self._baseline_ppl * self.config.ppl_gate:
                    report.stopped_early = True
                    report.stop_reason = (
                        f"held-out ppl {ppl:.1f} exceeded "
                        f"{self.config.ppl_gate:g}x baseline ({self._baseline_ppl:.1f})"
                    )
                    break

        model.eval()
        self.updates_applied += report.steps
        return report

    # -- arm interface -----------------------------------------------------------------

    def prepare(self, instance) -> None:
        self.reset()
        self.ingest([text for _sid, text in instance.turn_chunks()])

    def select(self, question: str) -> Selection:
        return empty_selection(
            meta={"arm": self.name, "method": self.config.label(), "rank": self.rank}
        )

    def answer(self, question: str, max_new_tokens: int = 32):
        return self.reader.generate("", question, max_new_tokens=max_new_tokens)

    @property
    def provenance(self) -> dict:
        return {
            "method": self.config.label(),
            "lora_rank": self.rank,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "lora_seed": self.seed,
            "replay": self.config.replay,
            "chat_format": self.config.chat_format,
            "kl_weight": self.config.kl_weight,
            "ppl_gate": self.config.ppl_gate,
        }
