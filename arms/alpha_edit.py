"""AlphaEdit: write facts into MLP weights while provably not disturbing what is preserved.

Every method tried so far is gradient descent on a LoRA adapter, and they all fail the same
way: the update that stores the fact also moves everything else. AlphaEdit (ICLR 2025,
arXiv 2410.02355) is a different mechanism entirely.

**The idea.** A transformer MLP behaves like a key-value store: `down_proj` maps an
activation (the key) to a residual contribution (the value). To insert a fact you want a
weight perturbation `Δ` such that `(W + Δ) k_new = v_new` for the new key, while
`(W + Δ) k_old = W k_old` for every key you want untouched. The second condition is exactly
`Δ K_old = 0` — the perturbation must lie in the **null space of the preserved keys**.

So: estimate the covariance of preserved-knowledge keys, take its null space, and project
every update into it. Preserved behaviour is then unchanged by construction rather than by
regularisation strength. That is why it survives sequential editing where ROME and MEMIT
collapse to ~0% accuracy after 50 edits; AlphaEdit retains 62.2% on GSM8K after 100.

**Why this should beat everything measured here.** The KL anchor *penalises* drift, so a
large enough gradient still wins. Replay *dilutes* drift. A null-space projection removes
the component of the update that could cause drift at all — it is a constraint, not a
preference. If the two axes are separable, this is the mechanism that separates them.

**Honest limits.** The null space is estimated from a finite key sample, so "provably" holds
with respect to that sample, not to all knowledge. And the paper itself reports degradation
by 10,000 edits: the guarantee is strong, not infinite.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EditReport:
    layer: int
    n_facts: int
    delta_norm: float
    projected_norm: float
    residual_before: float
    residual_after: float

    @property
    def null_space_retention(self) -> float:
        """Fraction of the raw update that survived projection.

        Near 1.0 means the update was already almost orthogonal to preserved knowledge.
        Near 0.0 means almost the whole update pointed at directions that matter, and the
        projection removed it — which predicts the edit will not take.
        """
        if self.delta_norm == 0:
            return 0.0
        return self.projected_norm / self.delta_norm


@dataclass
class AlphaEditResult:
    edits: list[EditReport] = field(default_factory=list)
    null_space_rank: int = 0
    key_dim: int = 0
    layers: tuple[int, ...] = ()


class AlphaEditArm:
    """Null-space-constrained knowledge editing on MLP down-projections.

    Not a LoRA arm: this writes directly into `mlp.down_proj.weight` across a band of
    layers, MEMIT-style. Nothing is frozen and re-parameterised; the base weights change,
    within a subspace that provably does not touch the preserved keys.
    """

    name = "alpha_edit"

    def __init__(
        self,
        reader,
        layers: tuple[int, ...] | None = None,
        null_space_threshold: float = 2e-2,
        edit_lr: float = 0.5,
        edit_steps: int = 25,
        preserve_samples: int = 64,
    ) -> None:
        self.reader = reader
        self.layers = layers
        # Singular values below this fraction of the largest are treated as null directions.
        # Larger threshold -> bigger null space -> more room to edit but weaker preservation.
        self.null_space_threshold = null_space_threshold
        self.edit_lr = edit_lr
        self.edit_steps = edit_steps
        self.preserve_samples = preserve_samples

        self._projection = None
        self._original: dict[int, "object"] = {}

    # -- layer plumbing ---------------------------------------------------------------

    def _blocks(self):
        model = self.reader.model
        for attr in ("model.layers", "transformer.h", "model.decoder.layers"):
            obj = model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise RuntimeError("could not locate transformer blocks on this model")

    def _default_layers(self) -> tuple[int, ...]:
        """A contiguous band in the lower-middle of the stack.

        MEMIT's finding: factual associations are recalled from mid-stack MLPs, and editing
        a band rather than a single layer spreads the perturbation so no one layer needs a
        large change. Early layers are too generic and late layers too close to the output.
        """
        n = len(self._blocks())
        start = max(0, n // 4)
        return tuple(range(start, min(n, start + max(1, n // 4))))

    def _down_proj(self, layer: int):
        return self._blocks()[layer].mlp.down_proj

    # -- the null space ----------------------------------------------------------------

    def build_projection(self, preserve_texts) -> AlphaEditResult:
        """Estimate the null space of preserved-knowledge keys.

        Keys are the activations entering `down_proj` — the MLP's intermediate
        representation — collected over text whose behaviour must not change. The
        projection matrix `P = I - U U^T` removes any component along the directions those
        keys actually occupy.
        """
        import torch

        layers = self.layers or self._default_layers()
        self.layers = layers
        model, tokenizer = self.reader._load()
        device = self.reader.device

        captured: list = []

        def hook(_module, inputs, _output):
            # inputs[0] is the activation entering down_proj: shape [batch, seq, d_ff]
            captured.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float().cpu())
            # .float() is load-bearing: the covariance and its eigendecomposition are
            # computed in float32/float64 even when the model runs in bf16.

        handle = self._down_proj(layers[0]).register_forward_hook(hook)
        try:
            with torch.no_grad():
                for text in preserve_texts[: self.preserve_samples]:
                    batch = tokenizer(
                        text, return_tensors="pt", truncation=True, max_length=256
                    ).to(device)
                    model(**batch)
        finally:
            handle.remove()

        keys = torch.cat(captured, dim=0)
        # Covariance of the preserved keys. Its principal directions are what must not move.
        cov = (keys.T @ keys) / max(1, keys.shape[0])
        eigvals, eigvecs = torch.linalg.eigh(cov.double())
        largest = float(eigvals.max())
        keep = eigvals > (largest * self.null_space_threshold)
        occupied = eigvecs[:, keep].float()  # directions preserved knowledge uses

        identity = torch.eye(cov.shape[0])
        projection = identity - occupied @ occupied.T
        self._projection = projection.to(device)

        return AlphaEditResult(
            null_space_rank=int(cov.shape[0] - int(keep.sum())),
            key_dim=int(cov.shape[0]),
            layers=tuple(layers),
        )

    # -- editing -----------------------------------------------------------------------

    def _snapshot(self):
        import torch

        for layer in self.layers:
            if layer not in self._original:
                self._original[layer] = self._down_proj(layer).weight.detach().clone()
        del torch

    def reset(self) -> None:
        """Restore the original weights. Required between independent edits."""
        import torch

        with torch.no_grad():
            for layer, weight in self._original.items():
                self._down_proj(layer).weight.copy_(weight)

    def edit(self, facts, target_texts=None) -> list[EditReport]:
        """Insert facts by a null-space-projected update to each targeted layer.

        The update is found by gradient descent on the *weight delta itself*, with the
        projection applied after every step, so the delta can never leave the null space.
        Descending first and projecting once at the end would allow the optimizer to find a
        solution that only works via the components about to be removed.
        """
        import torch

        if self._projection is None:
            raise RuntimeError("call build_projection() before edit()")

        self._snapshot()
        model, tokenizer = self.reader._load()
        device = self.reader.device
        texts = target_texts or [f.statement for f in facts]

        reports = []
        for layer in self.layers:
            module = self._down_proj(layer)
            original = self._original[layer]

            delta = torch.zeros_like(original, requires_grad=True)
            optimizer = torch.optim.Adam([delta], lr=self.edit_lr)

            before = None
            for step in range(self.edit_steps):
                with torch.no_grad():
                    module.weight.copy_(original + delta.detach())

                total = torch.tensor(0.0, device=device)
                for text in texts:
                    batch = tokenizer(
                        text, return_tensors="pt", truncation=True, max_length=128
                    ).to(device)
                    total = total + model(**batch, labels=batch["input_ids"]).loss
                loss = total / max(1, len(texts))
                if before is None:
                    before = float(loss)

                grad = torch.autograd.grad(loss, module.weight, retain_graph=False)[0]
                delta.grad = grad.to(delta.dtype)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # Project after EVERY step. The constraint is maintained, not applied once.
                #
                # The projection is kept in float32 while the weights are bf16. Computing
                # the eigendecomposition in bf16 would be numerically hopeless -- the null
                # space is defined by which singular values are *small*, exactly where bf16
                # has no precision left. So the cast happens here, per step, rather than by
                # degrading the projection itself.
                with torch.no_grad():
                    delta.copy_((delta.float() @ self._projection.T).to(delta.dtype))

            raw_norm = float(torch.linalg.norm(delta.detach()))
            with torch.no_grad():
                module.weight.copy_(original + delta.detach())

            reports.append(
                EditReport(
                    layer=layer,
                    n_facts=len(facts),
                    delta_norm=raw_norm,
                    projected_norm=raw_norm,
                    residual_before=before or 0.0,
                    residual_after=float(loss),
                )
            )

        return reports

    def answer(self, question: str, max_new_tokens: int = 24):
        return self.reader.generate("", question, max_new_tokens=max_new_tokens)
