"""Tests for restart-on-divergence.

The failure this guards against is subtle: a diverged model answers questions perfectly
happily, so a caller that ignores `succeeded` gets a confidently broken model and no signal.
"""

from __future__ import annotations

import pytest

from arms.stable_ingest import Attempt, StableIngestResult, ingest_with_restart


class FakeArm:
    """Diverges on a scripted set of seeds, so restart logic is testable without a GPU."""

    def __init__(self, seed, diverging_seeds, final_ratio=1.2, diverged_ratio=90.0):
        self.seed = seed
        self._diverging = diverging_seeds
        self._final = final_ratio
        self._diverged = diverged_ratio
        self.updates_applied = 0
        self.reset_calls = 0
        self.config = type("C", (), {"grad_clip": 1.0, "warmup_frac": 0.1})()

    def _ensure_adapter(self):
        return self

    def reset(self):
        self.reset_calls += 1
        self.updates_applied = 0

    def heldout_perplexity(self):
        if self.updates_applied == 0:
            return 10.0  # baseline
        return 10.0 * (self._diverged if self.seed in self._diverging else self._final)


def fake_train(arm, texts, baseline, threshold, check_every):
    arm.updates_applied = 30
    if arm.seed in arm._diverging:
        return True, 10
    return False, None


@pytest.fixture(autouse=True)
def patch_trainer(monkeypatch):
    monkeypatch.setattr("arms.stable_ingest._train_watched", fake_train)


class TestRestart:
    def test_succeeds_immediately_when_the_first_seed_is_clean(self):
        arm, result = ingest_with_restart(
            lambda s: FakeArm(s, diverging_seeds=set()), ["x"], divergence_threshold=3.0
        )
        assert result.succeeded and result.seed_used == 0 and result.n_attempts == 1

    def test_retries_past_a_diverging_seed(self):
        # The measured case: same config, one seed at 6.1x and another at 106.8x.
        arm, result = ingest_with_restart(
            lambda s: FakeArm(s, diverging_seeds={0}), ["x"], divergence_threshold=3.0
        )
        assert result.succeeded and result.seed_used == 1
        assert result.n_attempts == 2
        assert result.attempts[0].diverged and not result.attempts[1].diverged

    def test_reports_failure_rather_than_returning_a_broken_model(self):
        # A diverged model answers questions happily. Silently returning one is the worst
        # possible outcome, so exhaustion must be explicit.
        arm, result = ingest_with_restart(
            lambda s: FakeArm(s, diverging_seeds={0, 1, 2, 3, 4}),
            ["x"],
            max_attempts=5,
            divergence_threshold=3.0,
        )
        assert not result.succeeded
        assert result.seed_used is None
        assert result.n_attempts == 5

    def test_each_attempt_gets_a_fresh_arm(self):
        # Rewinding an adapter would carry the diverged trajectory's initialization into
        # the retry, which is precisely what the restart exists to escape.
        made = []

        def factory(seed):
            arm = FakeArm(seed, diverging_seeds={0})
            made.append(arm)
            return arm

        ingest_with_restart(factory, ["x"], divergence_threshold=3.0)
        assert len(made) == 2
        assert made[0] is not made[1]
        assert all(a.reset_calls == 1 for a in made)

    def test_divergence_rate_is_reported(self):
        _, result = ingest_with_restart(
            lambda s: FakeArm(s, diverging_seeds={0, 1}), ["x"], divergence_threshold=3.0
        )
        assert result.n_attempts == 3
        assert result.divergence_rate == pytest.approx(2 / 3)

    def test_a_clean_run_that_still_exceeds_the_threshold_is_rejected(self):
        # Not diverging mid-run is not the same as finishing inside the band.
        _, result = ingest_with_restart(
            lambda s: FakeArm(s, diverging_seeds=set(), final_ratio=8.0),
            ["x"],
            max_attempts=2,
            divergence_threshold=3.0,
        )
        assert not result.succeeded


class TestResultShape:
    def test_divergence_rate_of_no_attempts_is_zero_not_an_error(self):
        assert StableIngestResult(False, None, None, 0).divergence_rate == 0.0

    def test_attempts_record_where_the_abort_happened(self):
        a = Attempt(seed=3, diverged=True, ppl_ratio=90.0, steps=10, aborted_at_step=10)
        assert a.aborted_at_step == 10
