"""Arm D: memory as weights.

The other three arms answer by putting text in the context window. This one puts nothing
there. Instead it runs gradient updates on the conversation as it arrives, so the facts end
up in a LoRA adapter, and then answers with an empty context.

`select()` returning zero tokens is the entire claim. Everything else here is about whether
the answer is any good afterwards.

Two things make this harder than it sounds, and both are the point of measuring rather than
demoing:

**Writing a specific fact into weights is not what language modelling optimizes.** The
objective is next-token prediction over the whole transcript. One sentence about a dog's
name is a vanishing fraction of ~490 turns, and the gradient does not care that it is the
one sentence anyone will ask about. Making it stick means more epochs or a higher learning
rate -- which is exactly what damages everything else the model knew.

**The adapter must be reset between instances.** Each LongMemEval instance is a different
user's history. An adapter carried across instances would let instance 7 answer from
instance 3's haystack, which scores as memory and is actually leakage. `prepare()` restores
the adapter to its initial state every time, and there is a test pinning that.
"""

from __future__ import annotations

from copy import deepcopy

from harness.reader import Reader

from .base import Selection, empty_selection

# Attention projections only. Targeting every linear layer stores more but moves the model
# further from its initialization, which is the trade the rank sweep is meant to expose
# rather than assume.
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


class WeightMemoryArm:
    name = "weight_memory"

    def __init__(
        self,
        reader: Reader,
        rank: int = 16,
        alpha: int | None = None,
        dropout: float = 0.0,
        learning_rate: float = 1e-4,
        epochs: int = 1,
        max_chunk_tokens: int = 512,
        target_modules=DEFAULT_TARGET_MODULES,
        seed: int | None = None,
    ) -> None:
        self.reader = reader
        self.seed = seed
        self.rank = rank
        # The usual alpha = 2 * rank convention, so that sweeping rank does not silently
        # sweep the effective learning rate along with it.
        self.alpha = alpha if alpha is not None else 2 * rank
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_chunk_tokens = max_chunk_tokens
        self.target_modules = list(target_modules)

        self._peft_model = None
        self._pristine_state = None
        self.updates_applied = 0

    # -- adapter lifecycle ---------------------------------------------------------

    def _ensure_adapter(self):
        if self._peft_model is None:
            from peft import LoraConfig, get_peft_model

            # LoRA's A matrix is randomly initialized, so an unseeded adapter takes a
            # different training trajectory every run. Measured on the same config three
            # times, recall came out 1/2, 1/2, then 2/2 -- the spread between "fails" and
            # "works perfectly" was pure initialization luck. Any single unseeded run is a
            # draw from a distribution, not a result.
            if self.seed is not None:
                import os

                import torch

                torch.manual_seed(self.seed)
                # Seeding the initialization is NOT sufficient on GPU. Measured: 1B
                # reproduced to three decimals across runs, 3B did not -- the same config
                # and the same seeds gave 0.114 then 0.057, and the probe set was
                # identical, so the whole difference was one probe flipping. Larger models
                # dispatch to different kernels, and non-deterministic reductions in cuBLAS
                # make "seeded" a claim the numbers do not support.
                #
                # The cost is speed. The alternative is error bars that understate the
                # true variance, which is worse than a slower run.
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass

            base = self.reader.model
            config = LoraConfig(
                r=self.rank,
                lora_alpha=self.alpha,
                lora_dropout=self.dropout,
                target_modules=self.target_modules,
                task_type="CAUSAL_LM",
                bias="none",
            )
            self._peft_model = get_peft_model(base, config)
            # A freshly initialized LoRA is a no-op (B is zero), so this snapshot is the
            # "remembers nothing" state that every instance must start from.
            self._pristine_state = deepcopy(self._adapter_state())
        return self._peft_model

    def _adapter_state(self):
        model = self._peft_model
        return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}

    def reset(self) -> None:
        """Restore the adapter to its initial state. Wipes everything ingested."""
        model = self._ensure_adapter()
        if self._pristine_state:
            model.load_state_dict(self._pristine_state, strict=False)
        self.updates_applied = 0

    # -- ingestion -----------------------------------------------------------------

    def _chunk(self, texts):
        """Group turns into training sequences under a token cap.

        Chunking rather than one-sequence-per-turn because a single short turn is a very
        noisy gradient, and rather than one-sequence-per-transcript because that would
        exceed what fits in memory during a backward pass.
        """
        tokenizer = self.reader.tokenizer
        batch, batch_tokens = [], 0
        for text in texts:
            n = len(tokenizer.encode(text, add_special_tokens=False))
            if batch and batch_tokens + n > self.max_chunk_tokens:
                yield "\n".join(batch)
                batch, batch_tokens = [], 0
            batch.append(text)
            batch_tokens += n
        if batch:
            yield "\n".join(batch)

    def ingest(self, texts) -> int:
        """Run gradient updates so this text is reflected in the adapter.

        Returns the number of optimizer steps taken. Plain causal LM loss: no instruction
        formatting, no question/answer pairs. The claim under test is that reading the
        conversation is enough, and constructing supervised QA pairs from the haystack
        would be answering a different, much easier question.
        """
        import torch

        model = self._ensure_adapter()
        tokenizer = self.reader.tokenizer
        device = self.reader.device

        chunks = [c for c in self._chunk(texts) if c.strip()]
        if not chunks:
            return 0

        trainable = [p for p in model.parameters() if p.requires_grad]
        # A fresh optimizer per call, so no Adam moment estimates carry across ingestions.
        # Declared rather than incidental: for something called *online* learning, carrying
        # optimizer state is a defensible alternative, and it would change results. This
        # choice makes each ingestion independent, which is what makes a run reproducible
        # from its seed alone -- with persistent moments, the result would depend on every
        # ingestion that came before it in the process.
        optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate)

        model.train()
        steps = 0
        for _ in range(self.epochs):
            for chunk in chunks:
                batch = tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_chunk_tokens,
                ).to(device)
                # Causal LM: labels are the inputs, shifted internally by the model.
                outputs = model(**batch, labels=batch["input_ids"])
                outputs.loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1

        model.eval()
        self.updates_applied += steps
        return steps

    # -- arm interface -------------------------------------------------------------

    def prepare(self, instance) -> None:
        self.reset()
        self.ingest([text for _sid, text in instance.turn_chunks()])

    def select(self, question: str) -> Selection:
        """Zero context. The whole point.

        Reported as a real measured zero rather than skipped, so the token-cost column has
        a comparable number in it rather than a blank that quietly drops out of ratios.
        """
        return empty_selection(
            meta={
                "arm": self.name,
                "rank": self.rank,
                "alpha": self.alpha,
                "updates_applied": self.updates_applied,
            }
        )

    def answer(self, question: str, max_new_tokens: int = 32):
        """Generate with the adapter active and an empty context window."""
        return self.reader.generate("", question, max_new_tokens=max_new_tokens)

    @property
    def provenance(self) -> dict:
        return {
            "lora_rank": self.rank,
            "lora_alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "target_modules": ",".join(self.target_modules),
            "lora_seed": self.seed,
        }
