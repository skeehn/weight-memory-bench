"""Tests for the fixes to the seven review findings.

Each test names the bug it pins. They exist because every one of these produced output that
looked entirely normal.
"""

import pytest

from harness.gates import score_response, three_numbers


class TestAbstentionPrecedence:
    """Finding 4: abstention and correctness could both fire on one response."""

    def test_a_hedged_answer_is_an_abstention_not_a_correct_answer(self):
        # The exact case: contains the refusal marker AND the expected answer. Scored
        # independently it lands in both buckets and the three numbers stop partitioning.
        probe = score_response("I don't know, but maybe Pemberton", "Pemberton")
        assert not probe.answered
        assert not probe.correct

    def test_a_clean_correct_answer_scores_correct(self):
        probe = score_response("Pemberton.", "Pemberton")
        assert probe.answered and probe.correct

    def test_a_clean_wrong_answer_is_answered_but_incorrect(self):
        probe = score_response("Biscuit.", "Pemberton")
        assert probe.answered and not probe.correct

    def test_a_plain_refusal_is_an_abstention(self):
        probe = score_response("I don't know.", "Pemberton")
        assert not probe.answered and not probe.correct

    def test_matching_is_case_insensitive(self):
        assert score_response("pemberton", "Pemberton").correct
        assert score_response("PEMBERTON", "Pemberton").correct

    def test_answered_and_correct_never_overlap_with_abstained(self):
        responses = [
            "I don't know.",
            "I don't know, but maybe Pemberton",
            "Pemberton",
            "Biscuit",
        ]
        for text in responses:
            probe = score_response(text, "Pemberton")
            if probe.correct:
                assert probe.answered, "correct implies answered"


class TestAbstentionProbes:
    """Probes whose answer is genuinely absent: declining is the right behaviour."""

    def test_declining_an_unanswerable_probe_is_correct(self):
        probe = score_response("I don't know.", "", is_abstention_probe=True)
        assert probe.correct and not probe.answered and not probe.fabricated

    def test_answering_an_unanswerable_probe_is_a_fabrication(self):
        probe = score_response("Your ferret is named Biscuit.", "", is_abstention_probe=True)
        assert probe.answered and not probe.correct and probe.fabricated


class TestThreeNumbersPartition:
    """Finding 5: one numerator, two denominators, written once."""

    def test_the_two_accuracies_share_a_numerator(self):
        probes = [score_response(t, "X") for t in ["X", "Y", "I don't know."]]
        nums = three_numbers(probes)
        # 1 correct of 2 answered; 1 correct of 3 total.
        assert nums["accuracy_given_answered"] == pytest.approx(0.5)
        assert nums["accuracy_over_all"] == pytest.approx(1 / 3)
        assert nums["answered_rate"] == pytest.approx(2 / 3)

    def test_abstaining_more_inflates_only_the_gameable_number(self):
        honest = [score_response(t, "X") for t in ["X", "Y", "Y", "Y"]]
        gamed = [score_response(t, "X") for t in ["X", "I don't know.", "I don't know.", "I don't know."]]
        assert three_numbers(gamed)["accuracy_given_answered"] > three_numbers(honest)[
            "accuracy_given_answered"
        ]
        # ...and the honest number refuses to move.
        assert three_numbers(gamed)["accuracy_over_all"] == three_numbers(honest)[
            "accuracy_over_all"
        ]
