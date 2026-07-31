"""Tests for the anti-forgetting methods.

The one that matters most is the replay-corpus disjointness check. Training on the text you
evaluate perplexity against would make every method appear to prevent forgetting, and the
result would look completely normal.
"""

from __future__ import annotations

import pytest

from arms.memory_methods import MethodConfig, MitigatedMemoryArm
from arms.replay_corpus import REPLAY_TEXTS
from scripts.forgetting_curve import HELDOUT_TEXT
from scripts.method_comparison import PPL_CEILING, RETENTION_FLOOR, quadrant


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def apply_chat_template(self, messages, tokenize=False):
        return "<|start|>" + messages[0]["content"] + "<|end|>"


class FakeReader:
    device = "cpu"

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.model = object()


class TestReplayCorpusIsDisjoint:
    """The failure that would invalidate every replay result."""

    def test_no_replay_text_appears_in_the_heldout_passage(self):
        for text in REPLAY_TEXTS:
            assert text not in HELDOUT_TEXT

    def test_replay_and_heldout_share_almost_no_distinctive_vocabulary(self):
        stop = set(
            "the a an and or but of to in on at for with is are was were it its that "
            "which as by from than no not only most few some their they them he she".split()
        )

        def words(t):
            return {w.strip(".,;:").lower() for w in t.split()} - stop

        held = words(HELDOUT_TEXT)
        for text in REPLAY_TEXTS:
            overlap = words(text) & held
            assert len(overlap) <= 2, f"replay text shares {overlap} with the eval passage"

    def test_corpus_is_not_trivially_small(self):
        assert len(REPLAY_TEXTS) >= 5


class TestMethodConfig:
    def test_naive_config_labels_itself_naive(self):
        assert MethodConfig().label() == "naive"

    def test_labels_compose(self):
        label = MethodConfig(replay=True, chat_format=True, kl_weight=0.5, ppl_gate=1.5).label()
        assert "replay" in label and "chatfmt" in label and "kl0.5" in label and "gate1.5x" in label

    def test_zero_kl_weight_is_disabled(self):
        assert "kl" not in MethodConfig(kl_weight=0.0).label()


class TestTrainingSequence:
    def test_replay_off_yields_only_episode_chunks(self):
        arm = MitigatedMemoryArm(FakeReader(), MethodConfig(), max_chunk_tokens=5)
        seq = arm._training_sequence(["alpha beta", "gamma delta"])
        assert all(not is_replay for _, is_replay in seq)

    def test_replay_is_interleaved_not_appended(self):
        # Order matters: a block of replay after the fact has already been overwritten does
        # not undo the damage.
        arm = MitigatedMemoryArm(FakeReader(), MethodConfig(replay=True), max_chunk_tokens=3)
        seq = arm._training_sequence(["alpha beta", "gamma delta", "epsilon zeta"])
        flags = [is_replay for _, is_replay in seq]
        assert flags[0] is False and flags[1] is True, "replay must follow each episode chunk"
        assert flags != sorted(flags), "replay is appended, not interleaved"

    def test_replay_ratio_controls_how_much(self):
        arm1 = MitigatedMemoryArm(FakeReader(), MethodConfig(replay=True, replay_ratio=1))
        arm3 = MitigatedMemoryArm(FakeReader(), MethodConfig(replay=True, replay_ratio=3))
        texts = ["alpha", "beta"]
        assert len(arm3._training_sequence(texts)) > len(arm1._training_sequence(texts))

    def test_chat_format_wraps_the_text(self):
        arm = MitigatedMemoryArm(FakeReader(), MethodConfig(chat_format=True))
        assert arm._render("hello").startswith("<|start|>")

    def test_chat_format_off_leaves_text_alone(self):
        arm = MitigatedMemoryArm(FakeReader(), MethodConfig())
        assert arm._render("hello") == "hello"

    def test_chat_format_invents_no_question_answer_labels(self):
        # The framing changes; the content must not. Fabricating QA pairs would answer an
        # easier question than the one being asked.
        arm = MitigatedMemoryArm(FakeReader(), MethodConfig(chat_format=True))
        rendered = arm._render("My ferret is named Pemberton.")
        assert "My ferret is named Pemberton." in rendered
        assert "Question:" not in rendered and "?" not in rendered


class TestPreRegisteredBar:
    def test_works_requires_both_axes(self):
        assert quadrant(0.8, 1.1) == "WORKS"

    def test_learning_while_damaging_is_not_success(self):
        assert quadrant(0.9, 5.0) == "learns but damages"

    def test_safe_but_useless_is_not_success(self):
        assert quadrant(0.0, 1.0) == "safe but learns nothing"

    def test_the_naive_baseline_lands_in_the_worst_quadrant(self):
        # Measured: retention 0.000, ppl 36.83x at 100 updates.
        assert quadrant(0.000, 36.83) == "damages and learns nothing"

    def test_the_bar_is_stricter_than_beating_the_baseline(self):
        # 0.4 retention beats the naive peak of 0.375 but still fails the floor: beating a
        # method that destroys the model is not evidence of anything.
        assert quadrant(0.4, 1.0) != "WORKS"
        assert RETENTION_FLOOR > 0.375
        assert PPL_CEILING < 2.41  # the naive run passed this by update 25
