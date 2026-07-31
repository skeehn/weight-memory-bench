"""The tokenizer regression, written before any arm exists.

Everything this repo publishes is downstream of these counts. If the tokenizer silently
changes -- a different model, a re-uploaded vocab on the hub, a fallback quietly kicking in
-- every token ratio in the ledger becomes incomparable without announcing itself. So the
exact tokenizer state is pinned to a golden file and checked on every run.

Regenerate deliberately, never casually:

    WMB_REGEN_GOLDEN=1 uv run pytest tests/test_tokens.py

Regenerating invalidates every existing ledger row, which is why the fingerprint is also
recorded in each row's provenance.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from harness import tokens

GOLDEN = Path(__file__).parent / "golden_tokens.json"

# Fixed strings covering the shapes the arms actually count: prose, a retrieved chunk with
# structure, unicode, and whitespace-heavy text where a `words * 1.3` estimate diverges most.
SAMPLES = {
    "empty": "",
    "short": "The user's dog is named Biscuit.",
    "prose": (
        "On March 3rd the user mentioned they had switched jobs, and later that month "
        "they clarified the new role was at a hardware company rather than a software one."
    ),
    "structured": '{"session_id": "s_0042", "date": "2026-03-03", "speaker": "user"}',
    "unicode": "café naïve résumé — 東京 🧠",
    "whitespace_heavy": "a    b\t\tc\n\n\nd     e",
    "repeated": "memory " * 40,
}


@pytest.fixture(scope="module")
def tk():
    try:
        return tokens.shared()
    except tokens.TokenizerUnavailable as exc:
        pytest.skip(f"tokenizer unavailable (needs network on first run): {exc}")


class TestContract:
    def test_empty_string_is_zero_tokens(self, tk):
        assert tk.count("") == 0

    def test_none_raises_rather_than_counting_as_zero(self, tk):
        # An arm with no context should pass "", explicitly. None reaching here means a
        # bug upstream, and silently returning 0 would hide it inside the headline ratio.
        with pytest.raises(ValueError):
            tk.count(None)

    def test_shared_returns_the_same_instance(self):
        assert tokens.shared() is tokens.shared()

    def test_fingerprint_is_stable_within_a_process(self, tk):
        assert tk.fingerprint == tk.fingerprint
        assert len(tk.fingerprint) == 16

    def test_provenance_carries_model_revision_and_fingerprint(self, tk):
        prov = tk.provenance
        assert prov["tokenizer_model"] == tokens.READER_MODEL
        assert prov["tokenizer_revision"] == tokens.READER_REVISION
        assert prov["tokenizer_fingerprint"] == tk.fingerprint

    def test_count_all_is_not_the_same_as_counting_the_concatenation(self, tk):
        # Documented behaviour: arms pack context from discrete chunks, and a merge across
        # a join boundary would undercount. count_all is the honest accounting.
        chunks = ["the user's dog", "is named Biscuit"]
        assert tk.count_all(chunks) >= tk.count("".join(chunks))

    def test_count_all_of_nothing_is_zero(self, tk):
        assert tk.count_all([]) == 0


class TestNoSilentFallback:
    def test_unloadable_tokenizer_raises_rather_than_estimating(self):
        # The rule the whole repo rests on: there is no estimate path. A run that cannot
        # count honestly must fail here, not publish a number that looks measured.
        with pytest.raises(tokens.TokenizerUnavailable):
            tokens.Tokenizer(model="definitely/not-a-real-model-xyz", revision="main")


class TestEstimatesAreNotUsable:
    """Pins the premise behind the no-fallback rule.

    `words * 1.3` is the estimate this repo refuses to allow. Measured against the real
    reader tokenizer on these samples, real/estimate lands at:

        repeated          0.79x   (estimate overshoots)
        prose             0.88x   (estimate overshoots)
        short             1.28x
        whitespace_heavy  1.38x
        unicode           1.54x
        structured        3.46x   (estimate undershoots badly)

    The shape of this held across a reader swap: on Qwen3's tokenizer the same samples ran
    0.79x to 4.23x, on Llama-3.2's 0.79x to 3.46x. Different vocab, same verdict.

    The error is not a constant factor and not even a consistent *direction*. It runs from
    0.79x to 3.46x depending on text shape -- so it cannot be calibrated away, and it does
    not cancel when one arm's tokens are divided by another's. Text shape is precisely what
    differs between a packed JSON retrieval context and a plain-prose one, which is the
    comparison this whole repo exists to make.
    """

    @staticmethod
    def _estimate(text: str) -> float:
        return len(text.split()) * 1.3

    def test_structured_text_is_badly_undercounted_by_the_estimate(self, tk):
        text = SAMPLES["structured"]
        real, est = tk.count(text), self._estimate(text)
        assert real > 3 * est, f"expected large divergence, got real={real} est={est:.1f}"

    def test_the_estimate_also_overshoots_on_other_shapes(self, tk):
        # Both directions matter. An estimate that were merely conservative could be
        # treated as an upper bound; one that errs both ways cannot be treated as anything.
        for name in ("prose", "repeated"):
            text = SAMPLES[name]
            assert tk.count(text) < self._estimate(text), f"{name} no longer overshoots"

    def test_estimate_error_is_not_a_bounded_constant(self, tk):
        ratios = {
            name: tk.count(t) / self._estimate(t)
            for name, t in SAMPLES.items()
            if t and t.split()
        }
        spread = max(ratios.values()) / min(ratios.values())
        assert spread > 4.0, f"spread={spread:.2f}x ratios={ratios}"


class TestGolden:
    def test_counts_match_the_pinned_golden(self, tk):
        current = {
            "tokenizer_model": tk.model,
            "tokenizer_revision": tk.revision,
            "tokenizer_fingerprint": tk.fingerprint,
            "counts": {name: tk.count(text) for name, text in SAMPLES.items()},
        }

        if os.environ.get("WMB_REGEN_GOLDEN"):
            GOLDEN.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
            pytest.skip("regenerated golden file")

        if not GOLDEN.exists():
            pytest.skip(
                f"no golden file yet; create it with WMB_REGEN_GOLDEN=1 (would write {current})"
            )

        expected = json.loads(GOLDEN.read_text())
        if current["tokenizer_model"] != expected["tokenizer_model"]:
            # A different rung of the scaling ladder is running. The golden pins the
            # default rung; it is not evidence about any other, and asserting across them
            # would either fail spuriously or tempt someone to regenerate and lose the pin.
            pytest.skip(
                f"reader is {current['tokenizer_model']}, golden pins "
                f"{expected['tokenizer_model']}"
            )
        assert current["tokenizer_fingerprint"] == expected["tokenizer_fingerprint"], (
            "tokenizer state changed; every existing ledger row is now incomparable"
        )
        assert current["counts"] == expected["counts"]
