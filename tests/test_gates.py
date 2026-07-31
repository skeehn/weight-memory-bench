from harness.gates import FAIL, MIN_N_FOR_TAIL_STATISTIC, Probe, check, three_numbers


def probes(n_answered_correct, n_answered_wrong, n_abstained, fabricated=0):
    out = []
    out += [Probe(answered=True, correct=True) for _ in range(n_answered_correct)]
    out += [Probe(answered=True, correct=False) for _ in range(n_answered_wrong)]
    out += [Probe(answered=False, correct=False) for _ in range(n_abstained)]
    for i in range(fabricated):
        out[i] = Probe(answered=True, correct=False, fabricated=True)
    return out


class TestThreeNumbers:
    def test_abstentions_count_as_wrong_overall(self):
        # 1 correct, 0 wrong, 299 abstained. This is the exact shape that reports 1.00
        # when only accuracy_given_answered is published.
        nums = three_numbers(probes(1, 0, 299))
        assert nums["accuracy_given_answered"] == 1.0
        assert nums["accuracy_over_all"] == 1 / 300
        assert nums["answered_rate"] == 1 / 300

    def test_accuracy_given_answered_is_none_not_zero_when_nothing_answered(self):
        nums = three_numbers(probes(0, 0, 10))
        assert nums["accuracy_given_answered"] is None
        assert nums["accuracy_over_all"] == 0.0
        assert nums["answered_rate"] == 0.0

    def test_empty_probe_set(self):
        nums = three_numbers([])
        assert nums["n"] == 0
        assert nums["accuracy_over_all"] is None

    def test_all_three_present_on_every_report(self):
        nums = three_numbers(probes(5, 3, 2))
        for key in ("answered_rate", "accuracy_given_answered", "accuracy_over_all"):
            assert key in nums


class TestGates:
    def test_missing_tokenizer_fingerprint_blocks_the_row(self):
        report = check(probes(5, 3, 2), tokenizer_fingerprint=None)
        assert not report.reportable
        gate = next(g for g in report.gates if g.name == "real_tokenizer")
        assert gate.severity == FAIL and not gate.passed

    def test_tail_statistic_below_min_n_blocks(self):
        report = check(
            probes(5, 3, 2), tokenizer_fingerprint="abc123", tail_statistics=["p95_latency_ms"]
        )
        assert not report.reportable
        gate = next(g for g in report.gates if g.name == "tail_statistic_n")
        assert not gate.passed

    def test_tail_statistic_allowed_at_min_n(self):
        n = MIN_N_FOR_TAIL_STATISTIC
        report = check(
            probes(n - 10, 5, 5), tokenizer_fingerprint="abc123", tail_statistics=["p95"]
        )
        gate = next(g for g in report.gates if g.name == "tail_statistic_n")
        assert gate.passed

    def test_all_abstain_run_warns_but_does_not_block(self):
        # The point: this run passes every FAIL gate. It is a well-formed measurement of
        # nothing, and only the WARN catches it.
        report = check(probes(0, 0, 50), tokenizer_fingerprint="abc123")
        assert report.reportable
        gate = next(g for g in report.gates if g.name == "degenerate_run")
        assert not gate.passed and "all-abstain" in gate.detail

    def test_never_abstains_also_warns(self):
        report = check(probes(30, 20, 0), tokenizer_fingerprint="abc123")
        gate = next(g for g in report.gates if g.name == "degenerate_run")
        assert not gate.passed and "never abstains" in gate.detail

    def test_healthy_run_is_reportable_with_no_failures(self):
        report = check(probes(30, 15, 5), tokenizer_fingerprint="abc123")
        assert report.reportable
        assert report.failures == []

    def test_fabrication_warns_but_does_not_block(self):
        # A harness that refuses to report a bad arm is broken, not strict.
        report = check(probes(10, 10, 5, fabricated=3), tokenizer_fingerprint="abc123")
        assert report.reportable
        gate = next(g for g in report.gates if g.name == "no_fabrication")
        assert not gate.passed
