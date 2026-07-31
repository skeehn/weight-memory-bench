"""One tokenizer, shared by every arm.

The headline claim of this repo is a token claim: that memory living in weights costs
dramatically fewer context tokens per query than memory living in a vector store. A token
comparison is only meaningful if every arm counts with the *same* tokenizer, and if that
tokenizer is the real one rather than a `words * 1.3` estimate.

So this module has exactly one job and one rule.

The rule: there is no fallback. If the real tokenizer cannot be loaded, counting raises.
A run that cannot count tokens honestly must not produce a token number at all, because a
silently-estimated number does not announce itself downstream -- it just gets published.

The reader model is identical across all four arms; only the memory mechanism differs.
That is what makes this a controlled experiment, and it is also why there is no ambiguity
about which tokenizer is correct: it is the tokenizer of the model actually consuming the
context.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache

# The single reader model, shared by every arm. Changing this invalidates every existing
# ledger row, which is why the fingerprint is recorded in provenance on every row.
#
# Chosen against a hard constraint: arm A must hold the whole ~107K-token LongMemEval
# haystack, so the reader needs >110K of context, and arm D must LoRA the same model on a
# 24GB L4. That rules out the Qwen3 family (40,960) and SmolLM3 (65,536). Of what remains:
#
#   Llama-3.2-1B   131,072 ctx   16L / 8kv / 64hd   32KB/token   3.3GB KV at 107K
#   Phi-4-mini     131,072 ctx   32L / 8kv / 128hd  128KB/token  13.7GB KV at 107K
#
# Phi-4-mini's 13.7GB of KV plus 7.6GB of weights sits right at the edge of a 24GB card,
# and at 3.8B it costs ~3x the prefill FLOPs -- which arm A pays on every single query.
# The unsloth mirror is used because meta-llama/* is license-gated and this is not.
# The scaling ladder. Same family, same tokenizer, same 131,072-token context at every
# rung, so a difference across sizes is a difference in scale and not in architecture,
# vocabulary, or how much of the haystack fits.
READER_LADDER = {
    "1B": "unsloth/Llama-3.2-1B-Instruct",
    "3B": "unsloth/Llama-3.2-3B-Instruct",
    "8B": "unsloth/Meta-Llama-3.1-8B-Instruct",
}

DEFAULT_READER = READER_LADDER["1B"]

# Overridable so the scaling run can walk the ladder without editing source. Every ledger
# row records the resolved model and its tokenizer fingerprint, so a run at one rung can
# never be silently compared against a run at another.
READER_MODEL = os.environ.get("WMB_READER_MODEL", DEFAULT_READER)
READER_REVISION = os.environ.get("WMB_READER_REVISION", "main")

# Arm A's ceiling. The haystack is ~107K tokens, so this must not be reduced below it
# without turning arm A back into a truncated arm. All three rungs share this value, which
# is part of why this ladder was chosen.
READER_CONTEXT_TOKENS = 131_072


class TokenizerUnavailable(RuntimeError):
    """Raised when the real tokenizer cannot be loaded.

    Deliberately not caught anywhere in this repo. An arm that cannot count tokens is an
    arm that cannot report, and a run that cannot report should fail loudly at the point
    of failure rather than emit an estimate that looks like a measurement.
    """


@lru_cache(maxsize=None)
def _load(model: str, revision: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment failure
        raise TokenizerUnavailable(
            "transformers is not installed; cannot count tokens honestly"
        ) from exc

    try:
        return AutoTokenizer.from_pretrained(model, revision=revision)
    except Exception as exc:
        raise TokenizerUnavailable(
            f"could not load tokenizer for {model}@{revision}: {exc}"
        ) from exc


class Tokenizer:
    """The one tokenizer. Construct via `shared()` so every arm gets the same instance."""

    def __init__(self, model: str = READER_MODEL, revision: str = READER_REVISION):
        self.model = model
        self.revision = revision
        self._tk = _load(model, revision)

    def encode(self, text: str) -> list[int]:
        """Encode without special tokens.

        Special tokens are a property of how an arm frames its prompt, not of the context
        payload being compared. Counting them here would charge the full-context arm for
        chat scaffolding that the weight-memory arm also pays, and mix a constant into a
        ratio the whole result depends on.
        """
        return self._tk.encode(text, add_special_tokens=False)

    def count(self, text: str) -> int:
        if text is None:
            raise ValueError("cannot count tokens of None; pass '' for an empty context")
        if not text:
            return 0
        return len(self.encode(text))

    def count_all(self, texts) -> int:
        """Total tokens across an iterable of strings, counted individually.

        Not equal to counting the concatenation: merges across a join boundary would
        undercount. Arms assemble context from discrete retrieved chunks, so this is the
        honest accounting for a packed context.
        """
        return sum(self.count(t) for t in texts)

    @property
    def fingerprint(self) -> str:
        """Stable hash of the exact tokenizer state, for provenance.

        Computed from the serialized backend tokenizer rather than the model name, so a
        silently-changed vocab on the hub produces a different fingerprint and old ledger
        rows stop matching new ones.
        """
        backend = getattr(self._tk, "backend_tokenizer", None)
        if backend is not None:
            payload = backend.to_str()
        else:  # slow tokenizer fallback: hash the sorted vocab
            payload = repr(sorted(self._tk.get_vocab().items()))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def provenance(self) -> dict:
        return {
            "tokenizer_model": self.model,
            "tokenizer_revision": self.revision,
            "tokenizer_fingerprint": self.fingerprint,
        }


@lru_cache(maxsize=1)
def shared() -> Tokenizer:
    """The process-wide tokenizer. Every arm must call this, not construct its own."""
    return Tokenizer()
