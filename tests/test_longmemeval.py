import pytest

from data import longmemeval as lme


def make_instance(qid, n_sessions=2, n_turns=2, answer="x", evidence_at=None):
    sessions = []
    for s in range(n_sessions):
        turns = tuple(
            lme.Turn(
                role="user" if t % 2 == 0 else "assistant",
                content=f"content {qid} s{s} t{t}",
                has_answer=(evidence_at == (s, t)),
            )
            for t in range(n_turns)
        )
        sessions.append(lme.Session(session_id=f"{qid}_s{s}", date=f"2026/01/0{s+1}", turns=turns))
    return lme.Instance(
        question_id=qid,
        question_type="multi-session",
        question="q?",
        answer=answer,
        question_date="2026/02/01",
        sessions=tuple(sessions),
        answer_session_ids=(f"{qid}_s0",),
    )


class TestInstanceShape:
    def test_abstention_is_detected_from_the_id_suffix(self):
        assert make_instance("abc_abs").is_abstention
        assert not make_instance("abc").is_abstention

    def test_transcript_contains_every_turn(self):
        inst = make_instance("q1", n_sessions=3, n_turns=4)
        text = inst.transcript_text()
        for s in range(3):
            for t in range(4):
                assert f"content q1 s{s} t{t}" in text

    def test_session_chunks_match_session_count(self):
        inst = make_instance("q1", n_sessions=5)
        assert len(inst.session_chunks()) == 5

    def test_turn_chunks_carry_their_session_id(self):
        inst = make_instance("q1", n_sessions=2, n_turns=3)
        chunks = inst.turn_chunks()
        assert len(chunks) == 6
        assert all(sid.startswith("q1_s") for sid, _ in chunks)

    def test_evidence_turns_are_only_the_tagged_ones(self):
        inst = make_instance("q1", n_sessions=2, n_turns=2, evidence_at=(1, 0))
        evidence = inst.evidence_turns()
        assert len(evidence) == 1
        assert "s1 t0" in evidence[0]

    def test_transcript_does_not_leak_the_answer_or_the_evidence_flag(self):
        # Arms see the haystack, never the label. If "has_answer" or the answer string
        # rendered into the transcript, every arm would score by reading the tag.
        inst = make_instance("q1", answer="SECRET_ANSWER", evidence_at=(0, 0))
        text = inst.transcript_text()
        assert "SECRET_ANSWER" not in text
        assert "has_answer" not in text


class TestSplit:
    @pytest.fixture
    def corpus(self):
        # 6% abstention, matching the real corpus proportion.
        normal = [make_instance(f"q{i:04d}") for i in range(470)]
        absten = [make_instance(f"q{i:04d}_abs") for i in range(30)]
        return normal + absten

    def test_dev_and_test_are_disjoint_and_complete(self, corpus):
        dev = {i.question_id for i in lme.split(corpus, "dev")}
        test = {i.question_id for i in lme.split(corpus, "test")}
        assert not (dev & test)
        assert dev | test == {i.question_id for i in corpus}

    def test_split_is_deterministic(self, corpus):
        first = [i.question_id for i in lme.split(corpus, "dev")]
        second = [i.question_id for i in lme.split(list(reversed(corpus)), "dev")]
        assert first == second, "split must not depend on input order"

    def test_abstention_rate_is_preserved_in_both_splits(self, corpus):
        # The reason stratification exists: an unstratified hash split gave dev 3 of 102
        # abstention probes, too few to catch an arm that fabricates.
        for name in ("dev", "test"):
            part = lme.split(corpus, name)
            rate = sum(1 for i in part if i.is_abstention) / len(part)
            assert rate == pytest.approx(0.06, abs=0.015), f"{name} rate={rate}"

    def test_unknown_split_name_raises(self, corpus):
        with pytest.raises(ValueError):
            lme.split(corpus, "train")

    def test_all_returns_everything(self, corpus):
        assert len(lme.split(corpus, "all")) == len(corpus)


class TestCorpusHash:
    def test_is_order_independent(self):
        a = [make_instance("q1"), make_instance("q2")]
        assert lme.corpus_hash(a) == lme.corpus_hash(list(reversed(a)))

    def test_changes_when_an_answer_changes(self):
        a = [make_instance("q1", answer="one")]
        b = [make_instance("q1", answer="two")]
        assert lme.corpus_hash(a) != lme.corpus_hash(b)

    def test_changes_when_an_instance_is_dropped(self):
        a = [make_instance("q1"), make_instance("q2")]
        assert lme.corpus_hash(a) != lme.corpus_hash(a[:1])


class TestRealCorpus:
    """Touches the real 278MB file. Skipped when it is not present."""

    @staticmethod
    @pytest.fixture(scope="class")
    def corpus():
        try:
            return lme.load()
        except FileNotFoundError as exc:
            pytest.skip(str(exc))

    def test_shape_is_what_the_harness_assumes(self, corpus):
        assert len(corpus) == 500
        assert sum(1 for i in corpus if i.is_abstention) == 30

    def test_split_sizes(self, corpus):
        assert len(lme.split(corpus, "dev")) == 100
        assert len(lme.split(corpus, "test")) == 400

    def test_test_split_supports_tail_statistics(self, corpus):
        from harness.gates import MIN_N_FOR_TAIL_STATISTIC

        assert len(lme.split(corpus, "test")) >= MIN_N_FOR_TAIL_STATISTIC
        assert len(lme.split(corpus, "dev")) < MIN_N_FOR_TAIL_STATISTIC

    def test_every_instance_has_sessions_and_an_answer(self, corpus):
        assert all(i.sessions for i in corpus)
        assert all(i.answer for i in corpus if not i.is_abstention)

    def test_haystack_size_is_stable_and_large(self, corpus):
        # Measured with the repo's own tokenizer: median ~106.8K tokens, not the ~115K
        # that circulates secondhand. This is the number arm A actually pays per query.
        from harness import tokens

        tk = tokens.shared()
        counts = [tk.count(i.transcript_text()) for i in corpus[:10]]
        assert min(counts) > 90_000
        assert max(counts) < 130_000
