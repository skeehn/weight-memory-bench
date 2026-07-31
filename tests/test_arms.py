import pytest

from arms import base
from arms.full_context import FullContextArm
from arms.grep import GrepArm, terms
from arms.rag import BM25Index, RagArm, reciprocal_rank_fusion
from tests.test_longmemeval import make_instance


class FakeEmbedder:
    """Deterministic bag-of-words embedder. Lets the dense lane be tested without a model."""

    def __init__(self, vocab):
        self.vocab = list(vocab)

    def encode(self, texts):
        import numpy as np

        out = np.zeros((len(texts), len(self.vocab)), dtype="float32")
        for i, text in enumerate(texts):
            low = text.lower()
            for j, word in enumerate(self.vocab):
                out[i, j] = low.count(word)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-9, None)


class TestPack:
    def test_respects_the_budget(self):
        chunks = ["word " * 50 for _ in range(20)]
        sel = base.pack(chunks, budget=100)
        assert sel.tokens <= 100

    def test_preserves_caller_ranking(self):
        sel = base.pack(["alpha", "beta", "gamma"], budget=1000)
        assert sel.chunks == ("alpha", "beta", "gamma")

    def test_reports_the_recounted_total_not_the_running_sum(self):
        # The published number must be what the reader actually pays. Token merges across
        # a join boundary make the per-chunk sum an upper bound.
        from harness.tokens import shared

        sel = base.pack(["hello", "world"], budget=1000)
        assert sel.tokens == shared().count(sel.text)
        assert sel.tokens <= sel.meta["packed_sum_estimate"]

    def test_an_oversized_chunk_is_skipped_not_fatal(self):
        # One giant chunk must not starve every lower-ranked one.
        chunks = ["x " * 500, "small chunk"]
        sel = base.pack(chunks, budget=50)
        assert "small chunk" in sel.text
        assert sel.meta["skipped_chunks"] == 1

    def test_empty_input_yields_empty_selection(self):
        sel = base.pack([], budget=100)
        assert sel.tokens == 0 and sel.n_chunks == 0

    def test_nonpositive_budget_raises(self):
        with pytest.raises(ValueError):
            base.pack(["a"], budget=0)

    def test_assert_fits_rejects_an_oversized_context(self):
        sel = base.Selection(chunks=(), text="", tokens=base.READER_CONTEXT_TOKENS, meta={})
        with pytest.raises(base.ContextTooLarge):
            base.assert_fits(sel)

    def test_empty_selection_is_zero_tokens(self):
        assert base.empty_selection().tokens == 0


class TestFullContextArm:
    def test_includes_every_session(self):
        inst = make_instance("q1", n_sessions=4, n_turns=3)
        arm = FullContextArm()
        arm.prepare(inst)
        sel = arm.select("anything?")
        assert sel.n_chunks == 4
        for s in range(4):
            assert f"content q1 s{s} t0" in sel.text

    def test_context_is_independent_of_the_question(self):
        # Arm A does no selection, so two different questions must cost exactly the same.
        inst = make_instance("q1", n_sessions=3)
        arm = FullContextArm()
        arm.prepare(inst)
        assert arm.select("one?").text == arm.select("completely different?").text

    def test_is_far_more_expensive_than_a_retrieval_arm(self):
        inst = make_instance("q1", n_sessions=40, n_turns=8)
        full = FullContextArm()
        full.prepare(inst)
        grep = GrepArm(budget=256)
        grep.prepare(inst)
        assert full.select("content s3?").tokens > 10 * grep.select("content s3?").tokens


class TestGrepArm:
    def test_terms_drops_stopwords_and_single_chars(self):
        assert terms("What is the dog's name?") == {"dog's", "name"}

    def test_finds_the_turn_containing_the_query_terms(self):
        inst = make_instance("q1", n_sessions=3, n_turns=2)
        arm = GrepArm(budget=200)
        arm.prepare(inst)
        sel = arm.select("content q1 s2 t1")
        assert "s2 t1" in sel.text

    def test_ranks_by_distinct_terms_not_repetition(self):
        # A turn repeating one query word must not outrank a turn covering the whole query.
        inst = make_instance("q1", n_sessions=1, n_turns=1)
        arm = GrepArm(budget=500)
        arm.prepare(inst)
        arm._turns = [
            ("alpha alpha alpha alpha alpha", {"alpha"}),
            ("alpha beta gamma", {"alpha", "beta", "gamma"}),
        ]
        sel = arm.select("alpha beta gamma")
        assert sel.chunks[0] == "alpha beta gamma"

    def test_ties_break_toward_recency(self):
        arm = GrepArm(budget=500)
        arm._turns = [("alpha early", {"alpha", "early"}), ("alpha late", {"alpha", "late"})]
        assert arm.select("alpha").chunks[0] == "alpha late"

    def test_a_question_of_only_stopwords_returns_nothing(self):
        inst = make_instance("q1")
        arm = GrepArm()
        arm.prepare(inst)
        sel = arm.select("what is the of and")
        assert sel.n_chunks == 0 and sel.meta["query_terms"] == 0

    def test_stays_within_budget(self):
        inst = make_instance("q1", n_sessions=30, n_turns=10)
        arm = GrepArm(budget=300)
        arm.prepare(inst)
        assert arm.select("content").tokens <= 300


class TestBM25:
    def test_ranks_the_relevant_document_first(self):
        docs = ["the cat sat on the mat", "quantum chromodynamics is hard", "a dog barked"]
        assert BM25Index(docs).rank("cat mat")[0] == 0

    def test_idf_is_never_negative(self):
        # A term in most documents must not actively push its matches down the ranking.
        docs = ["common term here"] * 9 + ["common term rare"]
        idx = BM25Index(docs)
        assert all(v >= 0 for v in idx.idf.values())

    def test_a_rare_term_outweighs_a_common_one(self):
        docs = ["common"] * 20 + ["common rare"]
        idx = BM25Index(docs)
        assert idx.idf["rare"] > idx.idf["common"]

    def test_shorter_documents_are_favoured_at_equal_term_frequency(self):
        docs = ["target", "target " + "filler " * 50]
        scores = BM25Index(docs).scores("target")
        assert scores[0] > scores[1]

    def test_non_matching_documents_are_excluded_from_the_ranking(self):
        idx = BM25Index(["alpha", "beta"])
        assert idx.rank("alpha") == [0]

    def test_empty_query_returns_nothing(self):
        assert BM25Index(["alpha"]).rank("") == []


class TestRRF:
    def test_a_document_ranked_well_by_both_lanes_wins(self):
        assert reciprocal_rank_fusion([[5, 1, 2], [5, 3, 4]])[0] == 5

    def test_agreement_across_lanes_beats_a_single_strong_hit(self):
        # Doc 9 is first in one lane and absent from the other. Doc 0 is second in both.
        # Doc 0 wins, and that is correct: corroboration is the entire point of fusing.
        assert reciprocal_rank_fusion([[9, 0], [1, 0]])[0] == 0

    def test_adding_a_lane_that_omits_a_document_does_not_penalize_it(self):
        # The property that makes RRF safe without normalization. Absence contributes
        # nothing to the sum, so it cannot drag a document down -- unlike score fusion with
        # zero-fill, where a missing document is scored as though the lane rejected it.
        before = reciprocal_rank_fusion([[9, 0]])
        after = reciprocal_rank_fusion([[9, 0], [1]])
        assert before.index(9) < before.index(0)
        assert after.index(9) < after.index(0), "relative order must survive the new lane"

    def test_single_lane_passes_ranking_through(self):
        assert reciprocal_rank_fusion([[3, 1, 2]]) == [3, 1, 2]

    def test_no_lanes_yields_nothing(self):
        assert reciprocal_rank_fusion([]) == []


class TestRagArm:
    def test_lexical_only_still_retrieves(self):
        inst = make_instance("q1", n_sessions=3, n_turns=2)
        arm = RagArm(budget=300)
        arm.prepare(inst)
        sel = arm.select("content q1 s1 t0")
        assert not arm.has_dense
        assert sel.meta["dense_hits"] == 0
        assert "s1 t0" in sel.text

    def test_dense_lane_activates_when_an_embedder_is_supplied(self):
        pytest.importorskip("numpy")
        inst = make_instance("q1", n_sessions=3, n_turns=2)
        arm = RagArm(budget=300, embedder=FakeEmbedder(["content", "s1", "t0"]))
        arm.prepare(inst)
        sel = arm.select("content q1 s1 t0")
        assert arm.has_dense and sel.meta["dense_hits"] > 0

    def test_select_before_prepare_raises(self):
        with pytest.raises(RuntimeError):
            RagArm().select("q?")

    def test_respects_budget(self):
        inst = make_instance("q1", n_sessions=30, n_turns=10)
        arm = RagArm(budget=250)
        arm.prepare(inst)
        assert arm.select("content").tokens <= 250

    def test_same_budget_as_grep_so_only_ranking_differs(self):
        inst = make_instance("q1", n_sessions=20, n_turns=6)
        grep, rag = GrepArm(budget=400), RagArm(budget=400)
        grep.prepare(inst)
        rag.prepare(inst)
        assert grep.select("content s5").tokens <= 400
        assert rag.select("content s5").tokens <= 400
