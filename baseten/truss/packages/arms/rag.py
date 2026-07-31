"""Arm C: classical RAG. BM25 + dense retrieval, fused with Reciprocal Rank Fusion.

The serious baseline. This is what a competent team ships today, and it is the arm weight
memory has to beat to be worth anything. It gets the same turn-level chunks and the same
token budget as grep, so the only difference is ranking quality.

BM25 is implemented here rather than imported. It is forty lines, it is the single most
important baseline in the benchmark, and depending on a package whose tokenization and
parameter defaults differ from the repo's own would make the comparison harder to defend
than the code is to write.

The dense lane is pluggable. Everything except the embedder can be built and tested without
a model, which keeps arm C developable at zero cost -- the same reason context selection is
split from reader inference everywhere else in this repo.

**RRF over score fusion, deliberately.** BM25 scores and cosine similarities live on
incomparable scales, so any weighted sum of them silently encodes a normalization choice
that nobody can justify. RRF only uses rank, so it needs no calibration and cannot be
tuned into looking good by accident.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, Sequence

from .base import DEFAULT_RETRIEVAL_BUDGET, Selection, pack

TOKEN_RE = re.compile(r"[a-z0-9']+")

# Okapi defaults. Stated rather than tuned: tuning them on the dev split and reporting on
# test would make this arm quietly stronger than the honest baseline it is meant to be.
BM25_K1 = 1.5
BM25_B = 0.75

# The standard RRF constant. Large enough that the top few ranks are not wildly dominant.
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 over a fixed set of documents."""

    def __init__(self, documents: Sequence[str]) -> None:
        self.documents = list(documents)
        self.tokenized = [tokenize(d) for d in self.documents]
        self.lengths = [len(t) for t in self.tokenized]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.frequencies = [Counter(t) for t in self.tokenized]

        document_frequency: Counter[str] = Counter()
        for tokens in self.tokenized:
            document_frequency.update(set(tokens))

        n = len(self.documents)
        # Standard BM25 IDF with the +1 smoothing that keeps it non-negative. Without it,
        # a term appearing in more than half the documents gets a negative weight and
        # actively pushes matching documents *down*, which is not what anyone means by
        # "this document matched".
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        query_terms = tokenize(query)
        out = [0.0] * len(self.documents)
        if not query_terms or not self.avg_length:
            return out

        for i, freqs in enumerate(self.frequencies):
            length = self.lengths[i]
            norm = BM25_K1 * (1 - BM25_B + BM25_B * length / self.avg_length)
            total = 0.0
            for term in query_terms:
                f = freqs.get(term)
                if not f:
                    continue
                total += self.idf.get(term, 0.0) * (f * (BM25_K1 + 1)) / (f + norm)
            out[i] = total
        return out

    def rank(self, query: str, limit: int | None = None) -> list[int]:
        """Document indices, best first. Non-matching documents are excluded."""
        scored = [(s, i) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda x: (-x[0], x[1]))
        out = [i for _, i in scored]
        return out[:limit] if limit else out


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]):
        """Return an array of shape (len(texts), dim), L2-normalized."""


class SentenceTransformerEmbedder:
    """Dense lane backed by sentence-transformers. Loaded lazily."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]):
        return self._load().encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    """Fuse ranked index lists into one. Rank-only, so no score normalization is needed.

    A document absent from a ranking simply contributes nothing from that lane, rather than
    being scored zero -- absence is "this lane did not retrieve it", not "this lane rated it
    worthless", and those are different claims.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking):
            fused[index] = fused.get(index, 0.0) + 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]


class RagArm:
    name = "rag"

    def __init__(
        self,
        budget: int = DEFAULT_RETRIEVAL_BUDGET,
        embedder: Embedder | None = None,
        candidates: int = 20,
    ) -> None:
        self.budget = budget
        self.embedder = embedder
        # ONE cap, applied to both lanes. Previously BM25 returned only genuine term
        # matches (often 3 or 4) while dense returned its top 100 unconditionally, however
        # poor the similarity. Under RRF that is not a small imbalance: a document ranked
        # 50th by dense scores 1/(60+51), nearly the same as a real lexical match at rank
        # 3 scoring 1/63. So the "hybrid" arm was effectively dense with a lexical
        # tiebreak, and its evidence recall was correspondingly flattered.
        #
        # Capping both lanes to the same modest head keeps each contributing only what it
        # is confident about, and makes the fusion a comparison rather than a dilution.
        self.candidates = candidates
        self._chunks: list[str] = []
        self._bm25: BM25Index | None = None
        self._vectors = None

    @property
    def has_dense(self) -> bool:
        return self.embedder is not None

    def prepare(self, instance) -> None:
        self._chunks = [text for _sid, text in instance.turn_chunks()]
        self._bm25 = BM25Index(self._chunks)
        self._vectors = self.embedder.encode(self._chunks) if self.embedder else None

    def _dense_rank(self, question: str) -> list[int]:
        if self._vectors is None:
            return []
        import numpy as np

        q = self.embedder.encode([question])[0]
        sims = np.asarray(self._vectors) @ np.asarray(q)
        order = np.argsort(-sims)[: self.candidates]
        # Drop non-positive similarities, mirroring BM25 excluding zero-score documents.
        # Without this the lane pads its ranking with documents it has no opinion about.
        return [int(i) for i in order if sims[int(i)] > 0]

    def select(self, question: str) -> Selection:
        if self._bm25 is None:
            raise RuntimeError("prepare() must be called before select()")

        lexical = self._bm25.rank(question, limit=self.candidates)
        dense = self._dense_rank(question)
        lanes = [lane for lane in (lexical, dense) if lane]
        fused = reciprocal_rank_fusion(lanes) if lanes else []

        return pack(
            [self._chunks[i] for i in fused],
            self.budget,
            meta={
                "arm": self.name,
                "lexical_hits": len(lexical),
                "dense_hits": len(dense),
                "dense_enabled": self.has_dense,
                "fused_candidates": len(fused),
            },
        )
