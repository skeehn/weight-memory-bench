"""Arm D tests.

The fast tests cover chunking and the zero-token contract without loading a model. The
slow ones load Llama-3.2-1B and are marked `slow`; run them with `-m slow`.

The test that matters most is adapter isolation. Each LongMemEval instance is a different
user's history, so an adapter carried from one instance into the next lets instance 7
answer from instance 3's haystack. That scores as memory and is actually leakage, and it
would inflate arm D's number in a way nothing downstream would catch.
"""

from __future__ import annotations

import pytest

from arms.base import empty_selection
from arms.weight_memory import WeightMemoryArm
from tests.test_longmemeval import make_instance


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeReader:
    """Stands in for the reader so chunking and bookkeeping can be tested without a GPU."""

    device = "cpu"

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.model = object()


class TestZeroTokenContract:
    def test_select_returns_exactly_zero_tokens(self):
        arm = WeightMemoryArm(FakeReader())
        sel = arm.select("anything?")
        assert sel.tokens == 0
        assert sel.text == ""
        assert sel.n_chunks == 0

    def test_zero_is_reported_not_omitted(self):
        # A blank would silently drop out of ratios; a measured zero does not.
        sel = WeightMemoryArm(FakeReader()).select("q?")
        assert sel.meta["arm"] == "weight_memory"
        assert "updates_applied" in sel.meta

    def test_select_is_independent_of_the_question(self):
        arm = WeightMemoryArm(FakeReader())
        assert arm.select("one?").text == arm.select("two?").text == ""

    def test_empty_selection_helper_agrees(self):
        assert WeightMemoryArm(FakeReader()).select("q?").tokens == empty_selection().tokens


class TestChunking:
    def test_groups_turns_under_the_cap(self):
        arm = WeightMemoryArm(FakeReader(), max_chunk_tokens=5)
        chunks = list(arm._chunk(["a b c", "d e", "f g h", "i"]))
        for chunk in chunks:
            assert len(chunk.split()) <= 6, chunk  # cap plus the joined boundary word
        assert len(chunks) > 1

    def test_every_turn_survives_chunking(self):
        arm = WeightMemoryArm(FakeReader(), max_chunk_tokens=4)
        turns = ["alpha one", "beta two", "gamma three", "delta four"]
        joined = " ".join(arm._chunk(turns))
        for turn in turns:
            for word in turn.split():
                assert word in joined

    def test_a_single_oversized_turn_still_becomes_a_chunk(self):
        # Truncation happens later, at tokenization. Dropping it here would silently
        # discard content that was supposed to be memorized.
        arm = WeightMemoryArm(FakeReader(), max_chunk_tokens=3)
        assert list(arm._chunk(["one two three four five six"]))

    def test_no_turns_yields_no_chunks(self):
        assert list(WeightMemoryArm(FakeReader())._chunk([])) == []


class TestProvenance:
    def test_alpha_defaults_to_twice_rank(self):
        # So that sweeping rank does not silently sweep the effective learning rate too.
        for rank in (4, 16, 64):
            assert WeightMemoryArm(FakeReader(), rank=rank).alpha == 2 * rank

    def test_explicit_alpha_is_respected(self):
        assert WeightMemoryArm(FakeReader(), rank=8, alpha=99).alpha == 99

    def test_provenance_carries_every_knob_that_changes_results(self):
        prov = WeightMemoryArm(FakeReader(), rank=8, learning_rate=2e-3, epochs=5).provenance
        assert prov["lora_rank"] == 8
        assert prov["learning_rate"] == 2e-3
        assert prov["epochs"] == 5
        assert "q_proj" in prov["target_modules"]


@pytest.mark.slow
class TestAdapterIsolation:
    """Loads the real model. The leakage guard."""

    @staticmethod
    @pytest.fixture(scope="class")
    def arm():
        torch = pytest.importorskip("torch")
        pytest.importorskip("peft")
        from harness.reader import Reader

        reader = Reader()
        try:
            reader.model
        except Exception as exc:
            pytest.skip(f"reader unavailable: {exc}")
        a = WeightMemoryArm(reader, rank=8, learning_rate=2e-3, epochs=5)
        a._ensure_adapter()
        del torch
        return a

    def test_ingest_actually_changes_the_adapter(self, arm):
        import torch

        arm.reset()
        before = {k: v.clone() for k, v in arm._adapter_state().items()}
        arm.ingest(["The user's ferret is named Pemberton."])
        after = arm._adapter_state()
        changed = sum(
            1 for k in before if not torch.equal(before[k].cpu(), after[k].cpu())
        )
        assert changed > 0, "ingest did not move any adapter weight"

    def test_reset_restores_the_pristine_adapter(self, arm):
        import torch

        arm.reset()
        pristine = {k: v.clone() for k, v in arm._adapter_state().items()}
        arm.ingest(["Completely unrelated content about submarines."])
        arm.reset()
        restored = arm._adapter_state()
        for k in pristine:
            assert torch.equal(pristine[k].cpu(), restored[k].cpu()), f"{k} not restored"

    def test_reset_clears_the_update_counter(self, arm):
        arm.reset()
        arm.ingest(["something"])
        assert arm.updates_applied > 0
        arm.reset()
        assert arm.updates_applied == 0

    def test_prepare_resets_between_instances(self, arm):
        # The leakage case: instance two must not answer from instance one's haystack.
        arm.prepare(make_instance("first", n_sessions=1, n_turns=2))
        first_updates = arm.updates_applied
        arm.prepare(make_instance("second", n_sessions=1, n_turns=2))
        assert arm.updates_applied == first_updates, "counter should restart, not accumulate"
